// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lazy I/O for snapshot-level cell flag state.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Cursor;
use std::sync::Arc;

use bytes::Bytes;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::{Expr, ScalarUDF, col};
use datafusion::scalar::ScalarValue;
use futures::{StreamExt, TryStreamExt};
use lance_datafusion::udf::{
    CELL_FLAG_NAME, FlagFragment, bound_cell_flag_udf_with_coverage, is_cell_flag_id_udf,
    is_unbound_cell_flag_udf,
};
#[cfg(feature = "substrait")]
use lance_datafusion::udf::{bound_cell_flag_flag_id, cell_flag_id};
use lance_table::format::{
    CellFlagDefinition, CellFlagFile, CellFlagFragment, CellFlagFragmentState, CellFlagRoot,
    CellFlagState, pb,
};
use object_store::path::Path;
use prost::Message;
use roaring::{RoaringBitmap, RoaringTreemap};
use uuid::Uuid;

use super::Dataset;
use super::transaction::{
    CellFlagDrop, CellFlagFragmentState as TransactionCellFlagFragmentState, CellFlagFragmentValue,
    CellFlagRegistration, CellFlagRename, CellFlagTransaction, Operation, Transaction, UpdateMode,
};
use crate::session::caches::{CellFlagBitmapKey, CellFlagRootKey};
use crate::{Error, Result};
use lance_core::utils::address::RowAddress;
use lance_io::object_store::ObjectStore;

pub const CELL_FLAGS_DIR: &str = "_cell_flags";
const CELL_FLAG_ROOTS_DIR: &str = "roots";
const CELL_FLAG_BITMAPS_DIR: &str = "bitmaps";
const MAX_CELL_FLAG_FILE_BYTES: u64 = 512 * 1024 * 1024;
/// Individual portable Roaring bitmaps at or below this size are embedded in
/// the root when they are also sparse, avoiding one metadata request per
/// fragment without turning dense mutation state into monolithic metadata.
const MAX_INLINE_CELL_FLAG_BITMAP_BYTES: usize = 64 * 1024;
/// Inline only states at or below 25% density; denser states benefit from
/// independent immutable objects during repeated sparse mutations.
const MAX_INLINE_CELL_FLAG_DENSITY_DENOMINATOR: u64 = 4;
/// Cap total embedded bitmap bytes so large/high-fragment-count roots retain
/// incremental immutable bitmap objects instead of becoming monolithic.
const MAX_INLINE_CELL_FLAG_ROOT_BYTES: usize = 4 * 1024 * 1024;
/// Small roots are copied into their manifest descriptor so cold planning does
/// not require a separate object-store request.
const MAX_INLINE_CELL_FLAG_ROOT_COPY_BYTES: usize = 64 * 1024;
/// Bound total root copies retained in one manifest across all tracked fields.
const MAX_INLINE_CELL_FLAG_MANIFEST_BYTES: usize = 4 * 1024 * 1024;

/// An explicit Boolean change to a registered field-scoped cell flag.
///
/// Cell flag changes never follow from ordinary value or NULL writes. They are
/// applied only when attached to the operation that writes or selects the rows.
///
/// ```
/// use lance::dataset::CellFlagChange;
///
/// let change = CellFlagChange::new("embedding", "computed", true);
/// assert_eq!(change.field(), "embedding");
/// assert_eq!(change.name(), "computed");
/// assert!(change.value());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellFlagChange {
    field: String,
    name: String,
    value: bool,
}

impl CellFlagChange {
    /// Create an explicit change for `name` on `field`.
    pub fn new(field: impl Into<String>, name: impl Into<String>, value: bool) -> Self {
        Self {
            field: field.into(),
            name: name.into(),
            value,
        }
    }

    /// Return the field path resolved against the operation's input snapshot.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Return the field-local flag name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the explicit Boolean value to write.
    pub fn value(&self) -> bool {
        self.value
    }
}

fn inline_cell_flag_bytes(file: &CellFlagFile) -> Result<Option<Bytes>> {
    let Some(bytes) = file.inline_bytes.as_ref() else {
        return Ok(None);
    };
    if bytes.len() as u64 != file.size_bytes {
        return Err(Error::invalid_input(format!(
            "Inline cell flag file '{}' has size {}, expected {}",
            file.path,
            bytes.len(),
            file.size_bytes
        )));
    }
    if file.size_bytes > MAX_CELL_FLAG_FILE_BYTES {
        return Err(Error::invalid_input(format!(
            "Cell flag file '{}' has invalid declared size {}; maximum is {}",
            file.path, file.size_bytes, MAX_CELL_FLAG_FILE_BYTES
        )));
    }
    Ok(Some(Bytes::copy_from_slice(bytes)))
}

pub async fn read_cell_flag_bytes(
    store: &ObjectStore,
    path: &Path,
    file: &CellFlagFile,
) -> Result<Bytes> {
    if let Some(bytes) = inline_cell_flag_bytes(file)? {
        return Ok(bytes);
    }
    if file.size_bytes > MAX_CELL_FLAG_FILE_BYTES {
        return Err(Error::invalid_input(format!(
            "Cell flag file '{}' has invalid declared size {}; maximum is {}",
            file.path, file.size_bytes, MAX_CELL_FLAG_FILE_BYTES
        )));
    }
    let known_size = usize::try_from(file.size_bytes).map_err(|_| {
        Error::invalid_input(format!(
            "Cell flag file '{}' size {} does not fit on this platform",
            file.path, file.size_bytes
        ))
    })?;
    let reader = store.open_with_size(path, known_size).await?;
    let bytes = reader.get_all().await?;
    if bytes.len() != known_size {
        return Err(Error::invalid_input(format!(
            "Cell flag file '{}' has size {}, expected {}",
            file.path,
            bytes.len(),
            file.size_bytes
        )));
    }
    Ok(bytes)
}

fn decode_cell_flag_bitmap(
    bytes: &[u8],
    source: &str,
    flag_id: u32,
    fragment: &CellFlagFragment,
) -> Result<RoaringBitmap> {
    let bitmap = RoaringBitmap::deserialize_from(&mut Cursor::new(bytes)).map_err(|error| {
        Error::invalid_input(format!(
            "Failed to decode cell flag bitmap '{}' for flag ID {}, fragment {}: {}",
            source, flag_id, fragment.fragment_id, error
        ))
    })?;
    if bitmap.is_empty() || bitmap.len() == fragment.physical_rows {
        return Err(Error::invalid_input(format!(
            "Partial cell flag bitmap '{}' for flag ID {}, fragment {} must be non-empty and non-full",
            source, flag_id, fragment.fragment_id
        )));
    }
    if bitmap
        .max()
        .is_some_and(|offset| offset as u64 >= fragment.physical_rows)
    {
        return Err(Error::invalid_input(format!(
            "Cell flag bitmap '{}' for flag ID {}, fragment {} contains an out-of-bounds row offset",
            source, flag_id, fragment.fragment_id
        )));
    }
    Ok(bitmap)
}

impl Dataset {
    /// Return the flag definitions registered in this snapshot.
    ///
    /// ```no_run
    /// # use lance::{Dataset, Result};
    /// # fn inspect(dataset: &Dataset) -> Result<()> {
    /// for definition in dataset.cell_flag_definitions() {
    ///     assert!(!definition.name.is_empty());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn cell_flag_definitions(&self) -> &[CellFlagDefinition] {
        &self.manifest.cell_flag_definitions
    }

    /// Register a field-scoped Boolean flag and initialize existing rows.
    ///
    /// The returned `flag_id` is stable for the lifetime of the registration.
    /// `initial_value` applies to rows visible in the current snapshot and is
    /// independent of the field's Arrow values and validity bits.
    ///
    /// ```no_run
    /// # use lance::{Dataset, Result};
    /// # async fn register(dataset: &mut Dataset) -> Result<()> {
    /// let definition = dataset
    ///     .register_cell_flag("embedding", "computed", false)
    ///     .await?;
    /// assert_eq!(definition.name, "computed");
    /// # Ok(())
    /// # }
    /// ```
    pub async fn register_cell_flag(
        &mut self,
        field: impl AsRef<str>,
        name: impl Into<String>,
        initial_value: bool,
    ) -> Result<CellFlagDefinition> {
        let field = field.as_ref();
        let field_id = self
            .schema()
            .field(field)
            .ok_or_else(|| Error::invalid_input(format!("Unknown field '{}'", field)))?
            .id;
        let name = name.into();
        if name.is_empty() {
            return Err(Error::invalid_input("Cell flag name must not be empty"));
        }
        if self.cell_flag_definition(field_id, &name).is_some() {
            return Err(Error::invalid_input(format!(
                "Cell flag '{}' is already registered for field '{}'",
                name, field
            )));
        }
        let definition = CellFlagDefinition {
            flag_id: self.manifest.next_cell_flag_id,
            field_id,
            name,
        };
        let transaction = Transaction::new(
            self.manifest.version,
            Operation::Project {
                schema: self.schema().clone(),
                preserves_nullability: true,
            },
            None,
        )
        .with_cell_flag_transaction(CellFlagTransaction {
            registrations: vec![CellFlagRegistration {
                flag_id: definition.flag_id,
                field_id: definition.field_id,
                name: definition.name.clone(),
                initial_value,
            }],
            ..Default::default()
        });
        self.apply_commit(transaction, &Default::default(), &Default::default())
            .await?;
        Ok(definition)
    }

    /// Rename a registered flag without changing its stable ID or row state.
    ///
    /// ```no_run
    /// # use lance::{Dataset, Result};
    /// # async fn rename(dataset: &mut Dataset) -> Result<()> {
    /// dataset
    ///     .rename_cell_flag("embedding", "computed", "ready")
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn rename_cell_flag(
        &mut self,
        field: impl AsRef<str>,
        name: impl AsRef<str>,
        new_name: impl Into<String>,
    ) -> Result<()> {
        let definition = self.resolve_cell_flag_definition(field.as_ref(), name.as_ref())?;
        let new_name = new_name.into();
        if new_name.is_empty() {
            return Err(Error::invalid_input("Cell flag name must not be empty"));
        }
        if self
            .cell_flag_definition(definition.field_id, &new_name)
            .is_some_and(|existing| existing.flag_id != definition.flag_id)
        {
            return Err(Error::invalid_input(format!(
                "Cell flag '{}' is already registered for field '{}'",
                new_name,
                field.as_ref()
            )));
        }
        let flag_id = definition.flag_id;
        let transaction = Transaction::new(
            self.manifest.version,
            Operation::Project {
                schema: self.schema().clone(),
                preserves_nullability: true,
            },
            None,
        )
        .with_cell_flag_transaction(CellFlagTransaction {
            renames: vec![CellFlagRename {
                flag_id,
                name: new_name,
            }],
            ..Default::default()
        });
        self.apply_commit(transaction, &Default::default(), &Default::default())
            .await
    }

    /// Drop a registered flag from the current schema snapshot.
    ///
    /// ```no_run
    /// # use lance::{Dataset, Result};
    /// # async fn drop_flag(dataset: &mut Dataset) -> Result<()> {
    /// dataset.drop_cell_flag("embedding", "computed").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn drop_cell_flag(
        &mut self,
        field: impl AsRef<str>,
        name: impl AsRef<str>,
    ) -> Result<()> {
        let flag_id = self
            .resolve_cell_flag_definition(field.as_ref(), name.as_ref())?
            .flag_id;
        let transaction = Transaction::new(
            self.manifest.version,
            Operation::Project {
                schema: self.schema().clone(),
                preserves_nullability: true,
            },
            None,
        )
        .with_cell_flag_transaction(CellFlagTransaction {
            drops: vec![CellFlagDrop { flag_id }],
            ..Default::default()
        });
        self.apply_commit(transaction, &Default::default(), &Default::default())
            .await
    }

    pub(crate) fn resolve_cell_flag_definition(
        &self,
        field: &str,
        name: &str,
    ) -> Result<&CellFlagDefinition> {
        let field_id = self
            .schema()
            .field(field)
            .ok_or_else(|| Error::invalid_input(format!("Unknown field '{}'", field)))?
            .id;
        self.cell_flag_definition(field_id, name).ok_or_else(|| {
            Error::invalid_input(format!(
                "Unknown cell flag '{}' for field '{}'",
                name, field
            ))
        })
    }

    pub(crate) fn resolve_cell_flag_changes(
        &self,
        changes: &[CellFlagChange],
    ) -> Result<HashMap<u32, bool>> {
        let mut resolved = HashMap::with_capacity(changes.len());
        for change in changes {
            let flag_id = self
                .resolve_cell_flag_definition(change.field(), change.name())?
                .flag_id;
            if resolved.insert(flag_id, change.value()).is_some() {
                return Err(Error::invalid_input(format!(
                    "Cell flag '{}' for field '{}' is changed more than once",
                    change.name(),
                    change.field()
                )));
            }
        }
        Ok(resolved)
    }

    pub(crate) fn cell_flag_definition_by_id(&self, flag_id: u32) -> Option<&CellFlagDefinition> {
        self.manifest
            .cell_flag_definitions
            .iter()
            .find(|definition| definition.flag_id == flag_id)
    }

    pub(crate) fn cell_flag_definition(
        &self,
        field_id: i32,
        name: &str,
    ) -> Option<&CellFlagDefinition> {
        self.manifest
            .cell_flag_definitions
            .iter()
            .find(|definition| definition.field_id == field_id && definition.name == name)
    }

    pub(crate) fn cell_flag_state(&self, flag_id: u32) -> Option<&CellFlagState> {
        self.manifest
            .cell_flag_states
            .binary_search_by_key(&flag_id, |state| state.flag_id)
            .ok()
            .map(|index| &self.manifest.cell_flag_states[index])
    }

    fn cell_flag_path(&self, file: &CellFlagFile) -> Result<Path> {
        let relative = Path::parse(file.path.as_str())?;
        match file.base_id {
            Some(base_id) => {
                let base_path = self.manifest.base_paths.get(&base_id).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Dataset base path with ID {} not found for cell flag file '{}'",
                        base_id, file.path
                    ))
                })?;
                if !base_path.is_dataset_root {
                    return Err(Error::invalid_input(format!(
                        "Dataset base path with ID {} is not a dataset root for cell flag file '{}'",
                        base_id, file.path
                    )));
                }
                let base = base_path.extract_path(self.session.store_registry())?;
                Ok(Path::from_iter(base.parts().chain(relative.parts())))
            }
            None => Ok(Path::from_iter(self.base.parts().chain(relative.parts()))),
        }
    }

    fn cell_flag_source_uri<'a>(&'a self, file: &'a CellFlagFile) -> Result<&'a str> {
        match file.base_id {
            Some(base_id) => self
                .manifest
                .base_paths
                .get(&base_id)
                .map(|base| base.path.as_str())
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Dataset base path with ID {} not found for cell flag file '{}'",
                        base_id, file.path
                    ))
                }),
            None => Ok(self.uri.as_str()),
        }
    }

    async fn load_cell_flag_root_shared(&self, flag_id: u32) -> Result<Option<Arc<CellFlagRoot>>> {
        let Some(descriptor) = self.cell_flag_state(flag_id) else {
            return Ok(None);
        };
        descriptor.root.validate_root_path_for_flag(flag_id)?;
        let path = self.cell_flag_path(&descriptor.root)?;
        let source_uri = self.cell_flag_source_uri(&descriptor.root)?;
        let key = CellFlagRootKey {
            version: self.manifest.version,
            source_uri: source_uri.to_string(),
            path: descriptor.root.path.clone(),
            size_bytes: descriptor.root.size_bytes,
            inline_hash: descriptor
                .root
                .inline_bytes
                .as_deref()
                .map(blake3::hash)
                .map(|hash| *hash.as_bytes()),
        };
        let root = self
            .metadata_cache
            .get_or_insert_with_key(key, || async {
                let store = self.object_store(descriptor.root.base_id).await?;
                let bytes = read_cell_flag_bytes(&store, &path, &descriptor.root).await?;
                let proto = pb::CellFlagRoot::decode(bytes.as_ref()).map_err(|error| {
                    Error::invalid_input(format!(
                        "Failed to decode cell flag root '{}' for flag ID {}: {}",
                        descriptor.root.path, flag_id, error
                    ))
                })?;
                let mut root = CellFlagRoot::try_from(proto)?;

                let fragments_by_id = self
                    .manifest
                    .fragments
                    .iter()
                    .map(|fragment| (fragment.id, fragment))
                    .collect::<HashMap<_, _>>();
                for entry in &mut root.fragments {
                    let fragment = fragments_by_id.get(&entry.fragment_id).ok_or_else(|| {
                        Error::invalid_input(format!(
                            "Cell flag root for flag ID {} references unknown fragment {}",
                            flag_id, entry.fragment_id
                        ))
                    })?;
                    let physical_rows = fragment.physical_rows.ok_or_else(|| {
                        Error::invalid_input(format!(
                            "Fragment {} has no physical row count for cell flag",
                            entry.fragment_id
                        ))
                    })?;
                    if entry.physical_rows != physical_rows as u64 {
                        return Err(Error::invalid_input(format!(
                            "Cell flag root for flag ID {} records {} physical rows for fragment {}, manifest records {}",
                            flag_id, entry.physical_rows, entry.fragment_id, physical_rows
                        )));
                    }
                    if let CellFlagFragmentState::Partial(file) = &mut entry.state
                        && file.base_id.is_none()
                    {
                        // Bitmap paths are relative to the dataset that owns the root.
                        file.base_id = descriptor.root.base_id;
                    }
                    if let CellFlagFragmentState::Partial(file) = &entry.state {
                        file.validate_bitmap_path_for_fragment(flag_id, entry.fragment_id)?;
                    }
                }
                Ok(root)
            })
            .await?;
        Ok(Some(root))
    }

    /// Load one flag's immutable root on demand.
    pub(crate) async fn load_cell_flag_root(&self, flag_id: u32) -> Result<Option<CellFlagRoot>> {
        Ok(self
            .load_cell_flag_root_shared(flag_id)
            .await?
            .map(|root| root.as_ref().clone()))
    }

    /// Verify that an inline root copy exactly matches its referenced immutable
    /// object. Normal queries trust the manifest copy; explicit validation
    /// checks both representations and object reachability.
    pub(crate) async fn validate_cell_flag_root_object(&self, flag_id: u32) -> Result<()> {
        let Some(descriptor) = self.cell_flag_state(flag_id) else {
            return Ok(());
        };
        let Some(inline_bytes) = descriptor.root.inline_bytes.as_ref() else {
            return Ok(());
        };
        let mut object_file = descriptor.root.clone();
        object_file.inline_bytes = None;
        let path = self.cell_flag_path(&object_file)?;
        let store = self.object_store(object_file.base_id).await?;
        let object_bytes = read_cell_flag_bytes(&store, &path, &object_file).await?;
        if object_bytes.as_ref() != inline_bytes.as_slice() {
            return Err(Error::invalid_input(format!(
                "Inline cell flag root '{}' for flag ID {} does not match its immutable object",
                descriptor.root.path, flag_id
            )));
        }
        Ok(())
    }

    async fn load_cell_flag_bitmap_shared(
        &self,
        flag_id: u32,
        fragment: &CellFlagFragment,
    ) -> Result<Arc<RoaringBitmap>> {
        match &fragment.state {
            CellFlagFragmentState::Partial(file) => {
                file.validate_bitmap_path_for_fragment(flag_id, fragment.fragment_id)?;
                let path = self.cell_flag_path(file)?;
                let source_uri = self.cell_flag_source_uri(file)?;
                let key = CellFlagBitmapKey {
                    version: self.manifest.version,
                    source_uri: source_uri.to_string(),
                    path: file.path.clone(),
                    size_bytes: file.size_bytes,
                    flag_id,
                    fragment_id: fragment.fragment_id,
                    physical_rows: fragment.physical_rows,
                    inline_hash: None,
                };
                self.metadata_cache
                    .get_or_insert_with_key(key, || async {
                        let store = self.object_store(file.base_id).await?;
                        let bytes = read_cell_flag_bytes(&store, &path, file).await?;
                        decode_cell_flag_bitmap(&bytes, &file.path, flag_id, fragment)
                    })
                    .await
            }
            CellFlagFragmentState::InlinePartial(bytes) => {
                let key = CellFlagBitmapKey {
                    version: self.manifest.version,
                    source_uri: self.uri.clone(),
                    path: "inline".to_string(),
                    size_bytes: bytes.len() as u64,
                    flag_id,
                    fragment_id: fragment.fragment_id,
                    physical_rows: fragment.physical_rows,
                    inline_hash: Some(*blake3::hash(bytes).as_bytes()),
                };
                self.metadata_cache
                    .get_or_insert_with_key(key, || async {
                        decode_cell_flag_bitmap(bytes, "inline root entry", flag_id, fragment)
                    })
                    .await
            }
            CellFlagFragmentState::All => Err(Error::internal(format!(
                "Cell flag bitmap requested for all-set fragment {}",
                fragment.fragment_id
            ))),
        }
    }

    /// Load and validate a partial cell flag bitmap.
    pub(crate) async fn load_cell_flag_bitmap(
        &self,
        flag_id: u32,
        fragment: &CellFlagFragment,
    ) -> Result<RoaringBitmap> {
        self.load_cell_flag_bitmap_shared(flag_id, fragment)
            .await
            .map(|bitmap| bitmap.as_ref().clone())
    }

    pub(crate) async fn write_cell_flag_bitmap(
        &self,
        write_store: &ObjectStore,
        flag_id: u32,
        fragment_id: u64,
        physical_rows: u64,
        bitmap: &RoaringBitmap,
    ) -> Result<CellFlagFragmentState> {
        if bitmap.is_empty() || bitmap.len() == physical_rows {
            return Err(Error::internal(format!(
                "Only partial cell flag bitmaps may be written for flag ID {}, fragment {}",
                flag_id, fragment_id
            )));
        }
        if bitmap
            .max()
            .is_some_and(|offset| offset as u64 >= physical_rows)
        {
            return Err(Error::internal(format!(
                "Cell flag bitmap for flag ID {}, fragment {} contains an out-of-bounds row offset",
                flag_id, fragment_id
            )));
        }
        let mut bytes = Vec::with_capacity(bitmap.serialized_size());
        bitmap.serialize_into(&mut bytes)?;
        let is_sparse = bitmap.len() <= physical_rows / MAX_INLINE_CELL_FLAG_DENSITY_DENOMINATOR;
        if is_sparse && bytes.len() <= MAX_INLINE_CELL_FLAG_BITMAP_BYTES {
            return Ok(CellFlagFragmentState::InlinePartial(bytes));
        }
        Ok(CellFlagFragmentState::Partial(
            self.write_cell_flag_bitmap_file(write_store, flag_id, fragment_id, &bytes)
                .await?,
        ))
    }

    async fn write_cell_flag_bitmap_file(
        &self,
        write_store: &ObjectStore,
        flag_id: u32,
        fragment_id: u64,
        bytes: &[u8],
    ) -> Result<CellFlagFile> {
        let relative = Path::from(CELL_FLAGS_DIR)
            .join(CELL_FLAG_BITMAPS_DIR)
            .join(flag_id.to_string())
            .join(fragment_id.to_string())
            .join(format!("{}.rbm", Uuid::new_v4()));
        let full_path = Path::from_iter(self.base.parts().chain(relative.parts()));
        write_store.put(&full_path, bytes).await?;
        Ok(CellFlagFile {
            path: relative.to_string(),
            size_bytes: bytes.len() as u64,
            base_id: None,
            inline_bytes: None,
        })
    }

    pub(crate) async fn write_cell_flag_root(
        &self,
        write_store: &ObjectStore,
        flag_id: u32,
        mut root: CellFlagRoot,
    ) -> Result<Option<CellFlagState>> {
        if root.fragments.is_empty() {
            return Ok(None);
        }
        root.fragments.sort_by_key(|entry| entry.fragment_id);
        if root
            .fragments
            .windows(2)
            .any(|entries| entries[0].fragment_id == entries[1].fragment_id)
        {
            return Err(Error::internal(format!(
                "Cell flag root for flag ID {} contains duplicate fragment IDs",
                flag_id
            )));
        }
        let mut inline_bytes = 0usize;
        for entry in &mut root.fragments {
            let spill = match &entry.state {
                CellFlagFragmentState::InlinePartial(bytes) => {
                    let next_inline_bytes =
                        inline_bytes.checked_add(bytes.len()).ok_or_else(|| {
                            Error::internal(format!(
                                "Inline cell flag bitmap size overflow for flag ID {}",
                                flag_id
                            ))
                        })?;
                    if next_inline_bytes <= MAX_INLINE_CELL_FLAG_ROOT_BYTES {
                        inline_bytes = next_inline_bytes;
                        None
                    } else {
                        Some(bytes.clone())
                    }
                }
                _ => None,
            };
            if let Some(bytes) = spill {
                let file = self
                    .write_cell_flag_bitmap_file(write_store, flag_id, entry.fragment_id, &bytes)
                    .await?;
                entry.state = CellFlagFragmentState::Partial(file);
            }
        }
        let bytes = pb::CellFlagRoot::from(&root).encode_to_vec();
        if bytes.len() as u64 > MAX_CELL_FLAG_FILE_BYTES {
            return Err(Error::internal(format!(
                "Cell flag root for flag ID {} has encoded size {}, maximum is {}",
                flag_id,
                bytes.len(),
                MAX_CELL_FLAG_FILE_BYTES
            )));
        }
        let relative = Path::from(CELL_FLAGS_DIR)
            .join(CELL_FLAG_ROOTS_DIR)
            .join(flag_id.to_string())
            .join(format!("{}.root", Uuid::new_v4()));
        let full_path = Path::from_iter(self.base.parts().chain(relative.parts()));
        write_store.put(&full_path, &bytes).await?;
        Ok(Some(CellFlagState {
            flag_id,
            root: CellFlagFile {
                path: relative.to_string(),
                size_bytes: bytes.len() as u64,
                base_id: None,
                inline_bytes: (bytes.len() <= MAX_INLINE_CELL_FLAG_ROOT_COPY_BYTES)
                    .then(|| bytes.to_vec()),
            },
        }))
    }

    /// Build exact states for explicit flag values on newly inserted rows.
    /// Unmentioned flags are false by absence.
    pub(crate) fn cell_flag_states_for_new_fragments(
        &self,
        new_fragments: &[lance_table::format::Fragment],
        changes: &HashMap<u32, bool>,
    ) -> Result<Vec<TransactionCellFlagFragmentState>> {
        let mut states = Vec::new();
        for (&flag_id, &value) in changes {
            if self.cell_flag_definition_by_id(flag_id).is_none() {
                return Err(Error::invalid_input(format!(
                    "Cell flag change references unknown flag ID {}",
                    flag_id
                )));
            }
            let state = if value {
                CellFlagFragmentValue::All
            } else {
                CellFlagFragmentValue::None
            };
            for fragment in new_fragments {
                states.push(TransactionCellFlagFragmentState {
                    fragment_path: fragment_path(fragment)?.to_string(),
                    flag_id,
                    state: state.clone(),
                });
            }
        }
        Ok(states)
    }

    /// Build exact states for newly inserted fragments. Every registered flag
    /// is represented so a surrounding rewrite transaction can distinguish
    /// inserted rows (false by default) from rows whose state must be remapped.
    pub(crate) fn exact_cell_flag_states_for_inserted_fragments(
        &self,
        new_fragments: &[lance_table::format::Fragment],
        changes: &HashMap<u32, bool>,
    ) -> Result<Vec<TransactionCellFlagFragmentState>> {
        for flag_id in changes.keys() {
            if self.cell_flag_definition_by_id(*flag_id).is_none() {
                return Err(Error::invalid_input(format!(
                    "Cell flag change references unknown flag ID {}",
                    flag_id
                )));
            }
        }
        let mut states =
            Vec::with_capacity(new_fragments.len() * self.cell_flag_definitions().len());
        for definition in self.cell_flag_definitions() {
            let state = if changes.get(&definition.flag_id).copied().unwrap_or(false) {
                CellFlagFragmentValue::All
            } else {
                CellFlagFragmentValue::None
            };
            for fragment in new_fragments {
                states.push(TransactionCellFlagFragmentState {
                    fragment_path: fragment_path(fragment)?.to_string(),
                    flag_id: definition.flag_id,
                    state: state.clone(),
                });
            }
        }
        Ok(states)
    }

    /// Remap flag membership from source rows to rewritten fragments. Explicit
    /// operation-local changes override the preserved source state.
    pub(crate) async fn cell_flag_states_for_rewritten_rows(
        &self,
        new_fragments: &[lance_table::format::Fragment],
        source_row_addresses: &[u64],
        changes: &HashMap<u32, bool>,
    ) -> Result<Vec<TransactionCellFlagFragmentState>> {
        let inserted_positions = RoaringTreemap::new();
        self.cell_flag_states_for_mapped_rows(
            new_fragments,
            source_row_addresses,
            &inserted_positions,
            changes,
            &HashMap::new(),
        )
        .await
    }

    /// Build exact states for output rows that mix rewritten target rows and
    /// newly inserted rows. Positions absent from `inserted_positions` preserve
    /// their source state unless overridden; inserted positions are false unless
    /// explicitly overridden.
    pub(crate) async fn cell_flag_states_for_mapped_rows(
        &self,
        new_fragments: &[lance_table::format::Fragment],
        source_row_addresses: &[u64],
        inserted_positions: &RoaringTreemap,
        matched_changes: &HashMap<u32, bool>,
        inserted_changes: &HashMap<u32, bool>,
    ) -> Result<Vec<TransactionCellFlagFragmentState>> {
        let output_rows = new_fragments.iter().try_fold(0usize, |total, fragment| {
            let rows = usize::try_from(fragment_physical_rows(fragment)?).map_err(|_| {
                Error::invalid_input(format!(
                    "Fragment {} has too many rows for this platform",
                    fragment.id
                ))
            })?;
            total
                .checked_add(rows)
                .ok_or_else(|| Error::invalid_input("Rewritten row count overflow"))
        })?;
        if output_rows != source_row_addresses.len() {
            return Err(Error::internal(format!(
                "Rewritten fragments contain {} rows but {} source row mappings were captured",
                output_rows,
                source_row_addresses.len()
            )));
        }
        if inserted_positions
            .max()
            .is_some_and(|position| position >= source_row_addresses.len() as u64)
        {
            return Err(Error::internal(
                "Inserted Cell Flag row position is outside the rewritten output",
            ));
        }

        let source_fragment_ids = source_row_addresses
            .iter()
            .enumerate()
            .filter(|(position, _)| !inserted_positions.contains(*position as u64))
            .map(|(_, address)| RowAddress::from(*address).fragment_id() as u64)
            .collect::<HashSet<_>>();
        let mut states = Vec::new();
        for definition in &self.manifest.cell_flag_definitions {
            let flag_id = definition.flag_id;
            let root = self
                .load_cell_flag_root(flag_id)
                .await?
                .unwrap_or(CellFlagRoot {
                    fragments: Vec::new(),
                });
            let mut source_states = HashMap::new();
            for fragment in root
                .fragments
                .into_iter()
                .filter(|fragment| source_fragment_ids.contains(&fragment.fragment_id))
            {
                let state = match &fragment.state {
                    CellFlagFragmentState::All => None,
                    CellFlagFragmentState::Partial(_) | CellFlagFragmentState::InlinePartial(_) => {
                        Some(self.load_cell_flag_bitmap(flag_id, &fragment).await?)
                    }
                };
                source_states.insert(fragment.fragment_id, state);
            }

            let mut source_cursor = 0usize;
            for fragment in new_fragments {
                let physical_rows =
                    usize::try_from(fragment_physical_rows(fragment)?).map_err(|_| {
                        Error::invalid_input(format!(
                            "Fragment {} has too many rows for this platform",
                            fragment.id
                        ))
                    })?;
                let mut bitmap = RoaringBitmap::new();
                for (new_offset, source_address) in source_row_addresses
                    [source_cursor..source_cursor + physical_rows]
                    .iter()
                    .enumerate()
                {
                    let output_position = (source_cursor + new_offset) as u64;
                    let is_set = if inserted_positions.contains(output_position) {
                        inserted_changes.get(&flag_id).copied().unwrap_or(false)
                    } else if let Some(value) = matched_changes.get(&flag_id) {
                        *value
                    } else {
                        let address = RowAddress::from(*source_address);
                        match source_states.get(&(address.fragment_id() as u64)) {
                            Some(None) => true,
                            Some(Some(source_bitmap)) => {
                                source_bitmap.contains(address.row_offset())
                            }
                            None => false,
                        }
                    };
                    if is_set {
                        bitmap.insert(new_offset as u32);
                    }
                }
                source_cursor += physical_rows;
                let state = if bitmap.is_empty() {
                    CellFlagFragmentValue::None
                } else if bitmap.len() == physical_rows as u64 {
                    CellFlagFragmentValue::All
                } else {
                    CellFlagFragmentValue::Partial(bitmap)
                };
                states.push(TransactionCellFlagFragmentState {
                    fragment_path: fragment_path(fragment)?.to_string(),
                    flag_id,
                    state,
                });
            }
        }
        Ok(states)
    }
}

fn field_reference_segments(expr: &Expr) -> Result<Vec<String>> {
    match expr {
        Expr::Column(column) => Ok(vec![column.name.clone()]),
        Expr::ScalarFunction(function) if function.func.name() == "get_field" => {
            if function.args.len() != 2 {
                return Err(Error::invalid_input(format!(
                    "cell_flag field reference contains get_field with {} arguments, expected 2",
                    function.args.len()
                )));
            }
            let mut segments = field_reference_segments(&function.args[0])?;
            let child = match &function.args[1] {
                Expr::Literal(ScalarValue::Utf8(Some(value)), _)
                | Expr::Literal(ScalarValue::LargeUtf8(Some(value)), _)
                | Expr::Literal(ScalarValue::Utf8View(Some(value)), _) => value.clone(),
                _ => {
                    return Err(Error::invalid_input(
                        "cell_flag(field, name) requires a direct field reference",
                    ));
                }
            };
            segments.push(child);
            Ok(segments)
        }
        _ => Err(Error::invalid_input(
            "cell_flag(field, name) requires a direct field reference as its first argument",
        )),
    }
}

fn is_cell_flag_call(expr: &Expr) -> Result<Option<(Vec<String>, String)>> {
    let Expr::ScalarFunction(function) = expr else {
        return Ok(None);
    };
    if !function.func.name().eq_ignore_ascii_case(CELL_FLAG_NAME)
        || !is_unbound_cell_flag_udf(function.func.as_ref())
    {
        return Ok(None);
    }
    if function.args.len() != 2 {
        return Err(Error::invalid_input(format!(
            "cell_flag expects a field reference and a flag name, received {} arguments",
            function.args.len()
        )));
    }
    let segments = field_reference_segments(&function.args[0])?;
    let name = match &function.args[1] {
        Expr::Literal(ScalarValue::Utf8(Some(name)), _)
        | Expr::Literal(ScalarValue::LargeUtf8(Some(name)), _)
        | Expr::Literal(ScalarValue::Utf8View(Some(name)), _) => name.clone(),
        _ => {
            return Err(Error::invalid_input(
                "cell_flag(field, name) requires a non-null string literal name",
            ));
        }
    };
    if name.is_empty() {
        return Err(Error::invalid_input(
            "cell_flag(field, name) requires a non-empty flag name",
        ));
    }
    Ok(Some((segments, name)))
}

fn is_cell_flag_id_call(expr: &Expr) -> Result<Option<u32>> {
    let Expr::ScalarFunction(function) = expr else {
        return Ok(None);
    };
    if !is_cell_flag_id_udf(function.func.as_ref()) {
        return Ok(None);
    }
    if function.args.len() != 1 {
        return Err(Error::invalid_input(format!(
            "Internal cell flag ID transport expects one argument, received {}",
            function.args.len()
        )));
    }
    match &function.args[0] {
        Expr::Literal(ScalarValue::UInt32(Some(flag_id)), _) => Ok(Some(*flag_id)),
        _ => Err(Error::invalid_input(
            "Internal cell flag ID transport requires a non-null UInt32 literal",
        )),
    }
}

pub fn expression_references_cell_flag(expression: &Expr) -> Result<bool> {
    let mut references_flag = false;
    let mut captured_error = None;
    expression
        .apply(|node| {
            match is_cell_flag_call(node) {
                Ok(Some(_)) => references_flag = true,
                Ok(None) => match is_cell_flag_id_call(node) {
                    Ok(Some(_)) => references_flag = true,
                    Ok(None) => {}
                    Err(error) => {
                        captured_error = Some(error);
                        return Ok(TreeNodeRecursion::Stop);
                    }
                },
                Err(error) => {
                    captured_error = Some(error);
                    return Ok(TreeNodeRecursion::Stop);
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .map_err(Error::from)?;
    if let Some(error) = captured_error {
        return Err(error);
    }
    Ok(references_flag)
}

fn resolve_cell_flag(dataset: &Dataset, segments: &[String], name: &str) -> Result<u32> {
    let Some(first) = segments.first() else {
        return Err(Error::invalid_input(
            "cell_flag(field, name) requires a non-empty field reference",
        ));
    };
    let mut field = dataset
        .schema()
        .fields
        .iter()
        .find(|field| field.name == *first)
        .or_else(|| {
            dataset
                .schema()
                .fields
                .iter()
                .find(|field| field.name.eq_ignore_ascii_case(first))
        })
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "cell_flag references unknown field '{}'",
                segments.join(".")
            ))
        })?;
    for segment in &segments[1..] {
        field = field
            .children
            .iter()
            .find(|child| child.name == *segment)
            .or_else(|| {
                field
                    .children
                    .iter()
                    .find(|child| child.name.eq_ignore_ascii_case(segment))
            })
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "cell_flag references unknown field '{}'",
                    segments.join(".")
                ))
            })?;
    }
    dataset
        .cell_flag_definition(field.id, name)
        .map(|definition| definition.flag_id)
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "cell_flag references unknown flag '{}' for field '{}' (stable field ID {})",
                name,
                segments.join("."),
                field.id
            ))
        })
}

pub fn cell_flag_call_flag_id(dataset: &Dataset, expression: &Expr) -> Result<Option<u32>> {
    is_cell_flag_call(expression)?
        .map(|(segments, name)| resolve_cell_flag(dataset, &segments, &name))
        .transpose()
}

pub async fn load_cell_flag_fragments(
    dataset: &Dataset,
    flag_id: u32,
    selected_fragment_ids: Option<Arc<HashSet<u32>>>,
) -> Result<HashMap<u32, FlagFragment>> {
    if dataset.cell_flag_definition_by_id(flag_id).is_none() {
        return Err(Error::invalid_input(format!(
            "cell_flag references unknown flag ID {}",
            flag_id
        )));
    }
    let Some(root) = dataset.load_cell_flag_root_shared(flag_id).await? else {
        return Ok(HashMap::new());
    };
    let io_parallelism = dataset.object_store.io_parallelism().max(1);
    futures::stream::iter(root.fragments.iter().cloned())
        .map(|fragment| {
            let selected_fragment_ids = selected_fragment_ids.clone();
            async move {
                let fragment_id = u32::try_from(fragment.fragment_id).map_err(|_| {
                    Error::invalid_input(format!(
                        "Cell flag fragment ID {} does not fit in a row address",
                        fragment.fragment_id
                    ))
                })?;
                if selected_fragment_ids
                    .as_ref()
                    .is_some_and(|selected| !selected.contains(&fragment_id))
                {
                    return Ok::<Option<(u32, FlagFragment)>, Error>(None);
                }
                let state = match &fragment.state {
                    CellFlagFragmentState::All => FlagFragment::All,
                    CellFlagFragmentState::Partial(_) | CellFlagFragmentState::InlinePartial(_) => {
                        let bitmap = dataset
                            .load_cell_flag_bitmap_shared(flag_id, &fragment)
                            .await?;
                        FlagFragment::Partial(bitmap)
                    }
                };
                Ok(Some((fragment_id, state)))
            }
        })
        .buffer_unordered(io_parallelism)
        .try_filter_map(|entry| async move { Ok(entry) })
        .try_collect::<HashMap<_, _>>()
        .await
}

#[cfg(feature = "substrait")]
pub fn unbind_cell_flag_expression(dataset: &Dataset, expression: Expr) -> Result<Expr> {
    Ok(expression
        .transform(|node| {
            let Expr::ScalarFunction(function) = &node else {
                return Ok(Transformed::no(node));
            };
            let Some(flag_id) = bound_cell_flag_flag_id(function.func.as_ref()) else {
                return Ok(Transformed::no(node));
            };
            let definition = dataset.cell_flag_definition_by_id(flag_id).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Bound cell_flag references unknown flag ID {}",
                    flag_id
                ))
            })?;
            Ok(Transformed::yes(cell_flag_id(definition.flag_id)))
        })
        .map(|transformed| transformed.data)?)
}

#[cfg(feature = "substrait")]
pub async fn bind_cell_flag_expression(
    dataset: &Dataset,
    expression: Expr,
    selected_fragments: Option<&[lance_table::format::Fragment]>,
) -> Result<Expr> {
    let Some(bindings) =
        CellFlagExprBindings::try_new(dataset, &[&expression], selected_fragments).await?
    else {
        return Ok(expression);
    };
    bindings.bind(expression, dataset)
}

/// Snapshot bindings shared by every cell flag expression in one scanner.
pub struct CellFlagExprBindings {
    functions: HashMap<u32, Arc<ScalarUDF>>,
}

impl CellFlagExprBindings {
    pub(crate) async fn try_new(
        dataset: &Dataset,
        expressions: &[&Expr],
        selected_fragments: Option<&[lance_table::format::Fragment]>,
    ) -> Result<Option<Self>> {
        let mut flag_ids = HashSet::new();
        let mut captured_error = None;
        for expression in expressions {
            expression
                .apply(|node| {
                    match is_cell_flag_call(node) {
                        Ok(Some((segments, name))) => {
                            match resolve_cell_flag(dataset, &segments, &name) {
                                Ok(flag_id) => {
                                    flag_ids.insert(flag_id);
                                }
                                Err(error) => {
                                    captured_error = Some(error);
                                    return Ok(TreeNodeRecursion::Stop);
                                }
                            }
                        }
                        Ok(None) => match is_cell_flag_id_call(node) {
                            Ok(Some(flag_id)) => {
                                if dataset.cell_flag_definition_by_id(flag_id).is_none() {
                                    captured_error = Some(Error::invalid_input(format!(
                                        "Internal cell flag transport references unknown flag ID {}",
                                        flag_id
                                    )));
                                    return Ok(TreeNodeRecursion::Stop);
                                }
                                flag_ids.insert(flag_id);
                            }
                            Ok(None) => {}
                            Err(error) => {
                                captured_error = Some(error);
                                return Ok(TreeNodeRecursion::Stop);
                            }
                        },
                        Err(error) => {
                            captured_error = Some(error);
                            return Ok(TreeNodeRecursion::Stop);
                        }
                    }
                    Ok(TreeNodeRecursion::Continue)
                })
                .map_err(Error::from)?;
            if captured_error.is_some() {
                break;
            }
        }
        if let Some(error) = captured_error {
            return Err(error);
        }
        if flag_ids.is_empty() {
            return Ok(None);
        }

        let selected_fragment_ids = selected_fragments.map(|fragments| {
            Arc::new(
                fragments
                    .iter()
                    .map(|fragment| fragment.id as u32)
                    .collect::<HashSet<_>>(),
            )
        });
        let covered_fragments = selected_fragments
            .unwrap_or(dataset.fragments())
            .iter()
            .map(|fragment| fragment.id as u32)
            .collect::<RoaringBitmap>();
        let mut functions = HashMap::with_capacity(flag_ids.len());
        for flag_id in flag_ids {
            let fragments =
                load_cell_flag_fragments(dataset, flag_id, selected_fragment_ids.clone()).await?;
            functions.insert(
                flag_id,
                Arc::new(bound_cell_flag_udf_with_coverage(
                    flag_id,
                    fragments,
                    covered_fragments.clone(),
                )),
            );
        }
        Ok(Some(Self { functions }))
    }

    pub(crate) fn bind(&self, expression: Expr, dataset: &Dataset) -> Result<Expr> {
        let mut captured_error = None;
        let transformed = expression
            .transform(|node| {
                let public_call = match is_cell_flag_call(&node) {
                    Ok(call) => call,
                    Err(error) => {
                        captured_error = Some(error);
                        return Ok(Transformed::no(node));
                    }
                };
                let flag_id = if let Some((segments, name)) = public_call {
                    match resolve_cell_flag(dataset, &segments, &name) {
                        Ok(flag_id) => flag_id,
                        Err(error) => {
                            captured_error = Some(error);
                            return Ok(Transformed::no(node));
                        }
                    }
                } else {
                    match is_cell_flag_id_call(&node) {
                        Ok(Some(flag_id)) => flag_id,
                        Ok(None) => return Ok(Transformed::no(node)),
                        Err(error) => {
                            captured_error = Some(error);
                            return Ok(Transformed::no(node));
                        }
                    }
                };
                let Some(function) = self.functions.get(&flag_id) else {
                    captured_error = Some(Error::internal(format!(
                        "Missing cell_flag binding for flag ID {}",
                        flag_id
                    )));
                    return Ok(Transformed::no(node));
                };
                Ok(Transformed::yes(Expr::ScalarFunction(
                    ScalarFunction::new_udf(function.clone(), vec![col(lance_core::ROW_ADDR)]),
                )))
            })
            .map_err(Error::from)?
            .data;
        if let Some(error) = captured_error {
            Err(error)
        } else {
            Ok(transformed)
        }
    }
}

fn fragment_path(fragment: &lance_table::format::Fragment) -> Result<&str> {
    fragment
        .files
        .first()
        .map(|file| file.path.as_str())
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "Fragment {} has no data file to identify cell flag state",
                fragment.id
            ))
        })
}

fn fragment_physical_rows(fragment: &lance_table::format::Fragment) -> Result<u64> {
    fragment
        .physical_rows
        .map(|rows| rows as u64)
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "Fragment {} has no physical row count for cell flag",
                fragment.id
            ))
        })
}

async fn materialize_fragment_bitmap(
    current: &Dataset,
    flag_id: u32,
    fragment_id: u64,
    entry: Option<&CellFlagFragment>,
) -> Result<RoaringBitmap> {
    let Some(entry) = entry else {
        return Ok(RoaringBitmap::new());
    };
    match &entry.state {
        CellFlagFragmentState::All => {
            let mut bitmap = RoaringBitmap::new();
            bitmap.insert_range(
                0..u32::try_from(entry.physical_rows).map_err(|_| {
                    Error::invalid_input(format!(
                        "Fragment {} has too many rows for cell flag",
                        fragment_id
                    ))
                })?,
            );
            Ok(bitmap)
        }
        CellFlagFragmentState::Partial(_) | CellFlagFragmentState::InlinePartial(_) => {
            current.load_cell_flag_bitmap(flag_id, entry).await
        }
    }
}

#[derive(Default)]
struct PendingRowChanges {
    set: RoaringBitmap,
    clear: RoaringBitmap,
}

fn add_row_change(
    pending: &mut HashMap<(u32, u64), PendingRowChanges>,
    flag_id: u32,
    fragment_id: u64,
    offsets: &RoaringBitmap,
    value: bool,
) {
    let change = pending.entry((flag_id, fragment_id)).or_default();
    if value {
        change.set |= offsets;
        change.clear -= offsets;
    } else {
        change.clear |= offsets;
        change.set -= offsets;
    }
}

fn validate_fragment_value(
    flag_id: u32,
    fragment_id: u64,
    physical_rows: u64,
    value: &CellFlagFragmentValue,
) -> Result<()> {
    if let CellFlagFragmentValue::Partial(bitmap) = value {
        if bitmap.is_empty() || bitmap.len() == physical_rows {
            return Err(Error::invalid_input(format!(
                "Partial cell flag state for flag ID {}, fragment {} must be non-empty and non-full",
                flag_id, fragment_id
            )));
        }
        if bitmap
            .max()
            .is_some_and(|offset| offset as u64 >= physical_rows)
        {
            return Err(Error::invalid_input(format!(
                "Cell flag state for flag ID {}, fragment {} contains an out-of-bounds row offset",
                flag_id, fragment_id
            )));
        }
    }
    Ok(())
}

/// Reconcile one transaction with the current head's cell flag roots.
///
/// This runs after conflict resolution and manifest construction on every
/// commit attempt, so compatible concurrent appends and unrelated field writes
/// are merged from the actual head rather than from the stale read snapshot.
pub async fn apply_cell_flag_transaction(
    current: &Dataset,
    write_store: &ObjectStore,
    manifest: &mut lance_table::format::Manifest,
    transaction: &Transaction,
) -> Result<()> {
    if matches!(
        transaction.operation,
        Operation::Restore { .. } | Operation::Clone { .. }
    ) {
        return Ok(());
    }

    let changes = transaction.cell_flag_transaction()?.unwrap_or_default();
    let final_field_ids: HashSet<i32> = manifest.schema.field_ids().into_iter().collect();

    let mut definitions = current.manifest.cell_flag_definitions.clone();
    let mut transfer_targets = HashSet::new();
    for transfer in &changes.transfers {
        if transfer.source_field_id == transfer.target_field_id {
            return Err(Error::invalid_input(format!(
                "Cell flag field transfer source and target are both {}",
                transfer.source_field_id
            )));
        }
        if !transfer_targets.insert(transfer.target_field_id) {
            return Err(Error::invalid_input(format!(
                "Field ID {} is the target of more than one cell flag transfer",
                transfer.target_field_id
            )));
        }
        if final_field_ids.contains(&transfer.source_field_id)
            || !final_field_ids.contains(&transfer.target_field_id)
        {
            return Err(Error::invalid_input(format!(
                "Cell flag transfer must replace removed field ID {} with current field ID {}",
                transfer.source_field_id, transfer.target_field_id
            )));
        }
        let mut transferred = false;
        for definition in &mut definitions {
            if definition.field_id == transfer.source_field_id {
                definition.field_id = transfer.target_field_id;
                transferred = true;
            }
        }
        if !transferred {
            return Err(Error::invalid_input(format!(
                "Cell flag transfer source field ID {} has no registered flags",
                transfer.source_field_id
            )));
        }
    }

    let mut dropped_flag_ids = HashSet::new();
    for drop in &changes.drops {
        if !dropped_flag_ids.insert(drop.flag_id) {
            return Err(Error::invalid_input(format!(
                "Cell flag ID {} is dropped more than once",
                drop.flag_id
            )));
        }
        if !definitions
            .iter()
            .any(|definition| definition.flag_id == drop.flag_id)
        {
            return Err(Error::invalid_input(format!(
                "Cannot drop unknown cell flag ID {}",
                drop.flag_id
            )));
        }
    }
    definitions.retain(|definition| {
        !dropped_flag_ids.contains(&definition.flag_id)
            && final_field_ids.contains(&definition.field_id)
    });

    let mut renamed_flag_ids = HashSet::new();
    for rename in &changes.renames {
        if rename.name.is_empty() {
            return Err(Error::invalid_input(format!(
                "Cell flag rename for flag ID {} has an empty name",
                rename.flag_id
            )));
        }
        if !renamed_flag_ids.insert(rename.flag_id) {
            return Err(Error::invalid_input(format!(
                "Cell flag ID {} is renamed more than once",
                rename.flag_id
            )));
        }
        let definition = definitions
            .iter_mut()
            .find(|definition| definition.flag_id == rename.flag_id)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Cannot rename unknown or dropped cell flag ID {}",
                    rename.flag_id
                ))
            })?;
        definition.name.clone_from(&rename.name);
    }

    let mut next_flag_id = current.manifest.next_cell_flag_id;
    let mut registration_initial_values = HashMap::new();
    for registration in &changes.registrations {
        if registration.name.is_empty() {
            return Err(Error::invalid_input(format!(
                "Cell flag registration {} for field ID {} has an empty name",
                registration.flag_id, registration.field_id
            )));
        }
        if registration.flag_id != next_flag_id {
            return Err(Error::invalid_input(format!(
                "Cell flag registration has ID {}, expected next stable ID {}",
                registration.flag_id, next_flag_id
            )));
        }
        if !final_field_ids.contains(&registration.field_id) {
            return Err(Error::invalid_input(format!(
                "Cannot register cell flag '{}' for unknown field ID {}",
                registration.name, registration.field_id
            )));
        }
        if registration_initial_values
            .insert(registration.flag_id, registration.initial_value)
            .is_some()
        {
            return Err(Error::invalid_input(format!(
                "Cell flag ID {} is registered more than once",
                registration.flag_id
            )));
        }
        definitions.push(CellFlagDefinition {
            flag_id: registration.flag_id,
            field_id: registration.field_id,
            name: registration.name.clone(),
        });
        next_flag_id = next_flag_id.checked_add(1).ok_or_else(|| {
            Error::invalid_input("The dataset has exhausted the stable cell flag ID space")
        })?;
    }

    let mut flag_ids = HashSet::new();
    let mut field_names = HashSet::new();
    for definition in &definitions {
        if !flag_ids.insert(definition.flag_id) {
            return Err(Error::invalid_input(format!(
                "Cell flag ID {} is registered more than once",
                definition.flag_id
            )));
        }
        if !field_names.insert((definition.field_id, definition.name.clone())) {
            return Err(Error::invalid_input(format!(
                "Cell flag name '{}' is already registered for field ID {}",
                definition.name, definition.field_id
            )));
        }
    }
    definitions.sort_by_key(|definition| definition.flag_id);
    manifest.cell_flag_definitions = definitions.clone();
    manifest.next_cell_flag_id = next_flag_id;

    if definitions.is_empty() {
        if let Some(change) = changes.row_changes.first() {
            return Err(Error::incompatible_transaction_source(
                format!(
                    "Cell flag row change references unknown flag ID {}",
                    change.flag_id
                )
                .into(),
            ));
        }
        if let Some(state) = changes.fragment_states.first() {
            return Err(Error::incompatible_transaction_source(
                format!(
                    "Cell flag fragment state references unknown flag ID {}",
                    state.flag_id
                )
                .into(),
            ));
        }
        manifest.cell_flag_states.clear();
        return Ok(());
    }

    let final_fragments: HashMap<u64, &lance_table::format::Fragment> = manifest
        .fragments
        .iter()
        .map(|fragment| (fragment.id, fragment))
        .collect();
    let current_fragment_ids: HashSet<u64> = current
        .manifest
        .fragments
        .iter()
        .map(|fragment| fragment.id)
        .collect();
    let final_fragment_ids: HashSet<u64> = final_fragments.keys().copied().collect();
    let fragment_ids_unchanged = final_fragment_ids == current_fragment_ids;
    let fragment_by_path: HashMap<&str, &lance_table::format::Fragment> = manifest
        .fragments
        .iter()
        .map(|fragment| Ok((fragment_path(fragment)?, fragment)))
        .collect::<Result<_>>()?;

    let mut overrides: HashMap<(u32, u64), &CellFlagFragmentValue> = HashMap::new();
    for override_state in &changes.fragment_states {
        if !flag_ids.contains(&override_state.flag_id) {
            return Err(Error::incompatible_transaction_source(
                format!(
                    "Cell flag fragment state references unknown flag ID {}",
                    override_state.flag_id
                )
                .into(),
            ));
        }
        let fragment = fragment_by_path
            .get(override_state.fragment_path.as_str())
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Cell flag state references unknown new fragment path '{}'",
                    override_state.fragment_path
                ))
            })?;
        if overrides
            .insert((override_state.flag_id, fragment.id), &override_state.state)
            .is_some()
        {
            return Err(Error::invalid_input(format!(
                "Duplicate cell flag state for flag ID {}, fragment path '{}'",
                override_state.flag_id, override_state.fragment_path
            )));
        }
    }

    let mut pending_changes: HashMap<(u32, u64), PendingRowChanges> = HashMap::new();
    for change in &changes.row_changes {
        if !flag_ids.contains(&change.flag_id) {
            return Err(Error::incompatible_transaction_source(
                format!(
                    "Cell flag row change references unknown flag ID {}",
                    change.flag_id
                )
                .into(),
            ));
        }
        let mut by_fragment: HashMap<u64, RoaringBitmap> = HashMap::new();
        for raw_address in &change.row_addresses {
            let address = RowAddress::from(raw_address);
            by_fragment
                .entry(address.fragment_id() as u64)
                .or_default()
                .insert(address.row_offset());
        }
        for (fragment_id, offsets) in by_fragment {
            add_row_change(
                &mut pending_changes,
                change.flag_id,
                fragment_id,
                &offsets,
                change.value,
            );
        }
    }

    let exact_rewrite_paths: Vec<&str> = match &transaction.operation {
        Operation::Rewrite { groups, .. } => groups
            .iter()
            .flat_map(|group| group.new_fragments.iter())
            .map(fragment_path)
            .collect::<Result<_>>()?,
        Operation::Update {
            new_fragments,
            update_mode: Some(UpdateMode::RewriteRows),
            ..
        } => new_fragments
            .iter()
            .map(fragment_path)
            .collect::<Result<_>>()?,
        _ => Vec::new(),
    };

    for path in &exact_rewrite_paths {
        let fragment = fragment_by_path.get(path).ok_or_else(|| {
            Error::internal(format!(
                "Rewritten fragment '{}' is missing from the final manifest",
                path
            ))
        })?;
        for definition in &definitions {
            if !overrides.contains_key(&(definition.flag_id, fragment.id)) {
                return Err(Error::invalid_input(format!(
                    "Physical row rewrite must supply exact state for cell flag ID {}, new fragment '{}'",
                    definition.flag_id, path
                )));
            }
        }
    }

    let allows_false_new_fragments = matches!(
        transaction.operation,
        Operation::Append { .. } | Operation::Overwrite { .. }
    );
    if !allows_false_new_fragments {
        for fragment_id in final_fragment_ids.difference(&current_fragment_ids) {
            for definition in &definitions {
                if !registration_initial_values.contains_key(&definition.flag_id)
                    && !overrides.contains_key(&(definition.flag_id, *fragment_id))
                {
                    return Err(Error::invalid_input(format!(
                        "Operation {} introduced fragment {} without exact state for cell flag ID {}",
                        transaction.operation, fragment_id, definition.flag_id
                    )));
                }
            }
        }
    }

    let mut descriptors = Vec::with_capacity(definitions.len());
    for definition in &definitions {
        let flag_id = definition.flag_id;
        let has_explicit_state_change = registration_initial_values.contains_key(&flag_id)
            || overrides
                .keys()
                .any(|(override_flag_id, _)| *override_flag_id == flag_id)
            || pending_changes
                .keys()
                .any(|(change_flag_id, _)| *change_flag_id == flag_id);

        let append_only_adds_false_state =
            matches!(transaction.operation, Operation::Append { .. })
                && current_fragment_ids.is_subset(&final_fragment_ids)
                && !registration_initial_values.contains_key(&flag_id)
                && !pending_changes
                    .keys()
                    .any(|(change_flag_id, _)| *change_flag_id == flag_id)
                && overrides
                    .iter()
                    .filter(|((override_flag_id, _), _)| *override_flag_id == flag_id)
                    .all(|((_, fragment_id), value)| {
                        !current_fragment_ids.contains(fragment_id)
                            && matches!(value, CellFlagFragmentValue::None)
                    });

        if (fragment_ids_unchanged
            && !has_explicit_state_change
            && !matches!(transaction.operation, Operation::Overwrite { .. }))
            || append_only_adds_false_state
        {
            if let Some(descriptor) = current.cell_flag_state(flag_id) {
                descriptors.push(descriptor.clone());
            }
            continue;
        }

        let mut states: BTreeMap<u64, CellFlagFragment> =
            if let Some(&initial_value) = registration_initial_values.get(&flag_id) {
                if initial_value {
                    final_fragments
                        .values()
                        .map(|fragment| {
                            Ok((
                                fragment.id,
                                CellFlagFragment {
                                    fragment_id: fragment.id,
                                    physical_rows: fragment_physical_rows(fragment)?,
                                    state: CellFlagFragmentState::All,
                                },
                            ))
                        })
                        .collect::<Result<_>>()?
                } else {
                    BTreeMap::new()
                }
            } else if matches!(transaction.operation, Operation::Overwrite { .. }) {
                BTreeMap::new()
            } else {
                current
                    .load_cell_flag_root(flag_id)
                    .await?
                    .map(|root| {
                        root.fragments
                            .into_iter()
                            .filter(|entry| final_fragments.contains_key(&entry.fragment_id))
                            .map(|entry| (entry.fragment_id, entry))
                            .collect()
                    })
                    .unwrap_or_default()
            };

        for ((override_flag_id, fragment_id), value) in &overrides {
            if *override_flag_id != flag_id {
                continue;
            }
            let fragment = final_fragments.get(fragment_id).ok_or_else(|| {
                Error::internal(format!("Final fragment {} is missing", fragment_id))
            })?;
            let physical_rows = fragment_physical_rows(fragment)?;
            validate_fragment_value(flag_id, *fragment_id, physical_rows, value)?;
            match value {
                CellFlagFragmentValue::None => {
                    states.remove(fragment_id);
                }
                CellFlagFragmentValue::All => {
                    states.insert(
                        *fragment_id,
                        CellFlagFragment {
                            fragment_id: *fragment_id,
                            physical_rows,
                            state: CellFlagFragmentState::All,
                        },
                    );
                }
                CellFlagFragmentValue::Partial(bitmap) => {
                    let state = current
                        .write_cell_flag_bitmap(
                            write_store,
                            flag_id,
                            *fragment_id,
                            physical_rows,
                            bitmap,
                        )
                        .await?;
                    states.insert(
                        *fragment_id,
                        CellFlagFragment {
                            fragment_id: *fragment_id,
                            physical_rows,
                            state,
                        },
                    );
                }
            }
        }

        for ((change_flag_id, fragment_id), change) in &pending_changes {
            if *change_flag_id != flag_id {
                continue;
            }
            let fragment = final_fragments.get(fragment_id).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Cell flag change for flag ID {} references fragment {} that is not in the new snapshot",
                    flag_id, fragment_id
                ))
            })?;
            let physical_rows = fragment_physical_rows(fragment)?;
            if change
                .set
                .max()
                .into_iter()
                .chain(change.clear.max())
                .any(|offset| offset as u64 >= physical_rows)
            {
                return Err(Error::invalid_input(format!(
                    "Cell flag change for flag ID {}, fragment {} contains an out-of-bounds row offset",
                    flag_id, fragment_id
                )));
            }
            let mut bitmap = materialize_fragment_bitmap(
                current,
                flag_id,
                *fragment_id,
                states.get(fragment_id),
            )
            .await?;
            bitmap |= &change.set;
            bitmap -= &change.clear;
            if bitmap.is_empty() {
                states.remove(fragment_id);
            } else if bitmap.len() == physical_rows {
                states.insert(
                    *fragment_id,
                    CellFlagFragment {
                        fragment_id: *fragment_id,
                        physical_rows,
                        state: CellFlagFragmentState::All,
                    },
                );
            } else {
                let state = current
                    .write_cell_flag_bitmap(
                        write_store,
                        flag_id,
                        *fragment_id,
                        physical_rows,
                        &bitmap,
                    )
                    .await?;
                states.insert(
                    *fragment_id,
                    CellFlagFragment {
                        fragment_id: *fragment_id,
                        physical_rows,
                        state,
                    },
                );
            }
        }

        if let Some(descriptor) = current
            .write_cell_flag_root(
                write_store,
                flag_id,
                CellFlagRoot {
                    fragments: states.into_values().collect(),
                },
            )
            .await?
        {
            descriptors.push(descriptor);
        }
    }

    descriptors.sort_by_key(|state| state.flag_id);
    let mut inline_manifest_bytes = 0usize;
    for descriptor in &mut descriptors {
        let Some(bytes) = descriptor.root.inline_bytes.as_ref() else {
            continue;
        };
        let next_inline_manifest_bytes = inline_manifest_bytes
            .checked_add(bytes.len())
            .ok_or_else(|| Error::internal("Inline cell flag manifest size overflow"))?;
        if next_inline_manifest_bytes <= MAX_INLINE_CELL_FLAG_MANIFEST_BYTES {
            inline_manifest_bytes = next_inline_manifest_bytes;
        } else {
            descriptor.root.inline_bytes = None;
        }
    }
    manifest.cell_flag_states = descriptors;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        ArrayRef, BooleanArray, Int32Array, RecordBatch, RecordBatchIterator, StructArray,
    };
    use arrow_schema::{DataType, Field, Fields, Schema};
    use lance_core::utils::tempfile::TempStrDir;

    use super::*;
    use crate::dataset::cleanup::CleanupPolicyBuilder;
    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::dataset::write::merge_insert::{MergeInsertBuilder, WhenMatched, WhenNotMatched};
    use crate::dataset::{
        ColumnAlteration, CommitBuilder, InsertBuilder, UpdateBuilder, WriteMode, WriteParams,
    };
    use crate::utils::test::copy_test_data_to_tmp;

    const FLAG_NAME: &str = "lancedb.computed";

    async fn dataset_with_rows(
        directory: &TempStrDir,
        max_rows_per_file: usize,
    ) -> Result<Dataset> {
        dataset_with_rows_and_stable_ids(directory, max_rows_per_file, false).await
    }

    async fn dataset_with_rows_and_stable_ids(
        directory: &TempStrDir,
        max_rows_per_file: usize,
        enable_stable_row_ids: bool,
    ) -> Result<Dataset> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..8)),
                Arc::new(Int32Array::from(vec![
                    Some(10),
                    None,
                    Some(12),
                    None,
                    Some(14),
                    Some(15),
                    None,
                    Some(17),
                ])),
            ],
        )?;
        Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            directory,
            Some(WriteParams {
                max_rows_per_file,
                enable_stable_row_ids,
                ..Default::default()
            }),
        )
        .await
    }

    async fn flag_rows(dataset: &Dataset, field: &str, name: &str) -> Result<Vec<(i32, bool)>> {
        let expression = format!("cell_flag({}, '{}')", field, name);
        let mut scanner = dataset.scan();
        scanner.project_with_transform(&[("id", "id"), ("flag", &expression)])?;
        let batch = scanner.try_into_batch().await?;
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let flags = batch
            .column_by_name("flag")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        let mut rows = (0..batch.num_rows())
            .map(|index| (ids.value(index), flags.value(index)))
            .collect::<Vec<_>>();
        rows.sort_unstable();
        Ok(rows)
    }

    async fn flagged_ids(dataset: &Dataset, field: &str, name: &str) -> Result<Vec<i32>> {
        let mut scanner = dataset.scan();
        scanner.project(&["id"])?;
        scanner.filter(&format!("cell_flag({}, '{}')", field, name))?;
        let batch = scanner.try_into_batch().await?;
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let mut values = ids.values().to_vec();
        values.sort_unstable();
        Ok(values)
    }

    #[tokio::test]
    async fn registry_is_field_scoped_and_initializes_independently_of_nulls() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 2).await?;

        let computed = dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;
        let reviewed = dataset
            .register_cell_flag("value", "reviewed", true)
            .await?;
        let same_name_other_field = dataset.register_cell_flag("id", FLAG_NAME, false).await?;

        assert_ne!(computed.flag_id, reviewed.flag_id);
        assert_ne!(computed.flag_id, same_name_other_field.flag_id);
        assert_eq!(dataset.cell_flag_definitions().len(), 3);
        assert_eq!(
            flagged_ids(&dataset, "value", FLAG_NAME).await?,
            Vec::<i32>::new()
        );
        assert_eq!(
            flagged_ids(&dataset, "value", "reviewed").await?,
            (0..8).collect::<Vec<_>>()
        );
        assert_eq!(
            flagged_ids(&dataset, "id", FLAG_NAME).await?,
            Vec::<i32>::new()
        );
        assert!(
            dataset
                .register_cell_flag("value", FLAG_NAME, false)
                .await
                .unwrap_err()
                .to_string()
                .contains("already registered")
        );
        Ok(())
    }

    #[tokio::test]
    async fn registration_initializes_legacy_fragment_row_counts() -> Result<()> {
        let directory = copy_test_data_to_tmp("v0.7.5/with_deletions")?;
        let mut dataset = Dataset::open(&directory.path_str()).await?;

        dataset.register_cell_flag("x", FLAG_NAME, true).await?;

        assert_eq!(
            dataset
                .count_rows(Some(format!("cell_flag(x, '{}')", FLAG_NAME)))
                .await?,
            90
        );
        assert!(
            dataset
                .manifest
                .fragments
                .iter()
                .all(|fragment| fragment.physical_rows.is_some())
        );
        Ok(())
    }

    #[tokio::test]
    async fn registry_uses_stable_ids_for_nested_fields() -> Result<()> {
        let directory = TempStrDir::default();
        let payload_fields = Fields::from(vec![
            Field::new("left", DataType::Int32, true),
            Field::new("right", DataType::Int32, true),
        ]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("payload", DataType::Struct(payload_fields.clone()), false),
        ]));
        let payload = StructArray::new(
            payload_fields,
            vec![
                Arc::new(Int32Array::from(vec![Some(10), None])) as ArrayRef,
                Arc::new(Int32Array::from(vec![None, Some(20)])) as ArrayRef,
            ],
            None,
        );
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values([0, 1])),
                Arc::new(payload),
            ],
        )?;
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &directory,
            None,
        )
        .await?;

        let left = dataset
            .register_cell_flag("payload.left", FLAG_NAME, true)
            .await?;
        let right = dataset
            .register_cell_flag("payload.right", FLAG_NAME, false)
            .await?;

        assert_ne!(left.field_id, right.field_id);
        assert_ne!(left.flag_id, right.flag_id);
        assert_eq!(
            flagged_ids(&dataset, "payload.left", FLAG_NAME).await?,
            vec![0, 1]
        );
        assert_eq!(
            flagged_ids(&dataset, "payload.right", FLAG_NAME).await?,
            Vec::<i32>::new()
        );
        Ok(())
    }

    #[tokio::test]
    async fn update_supports_flag_only_and_explicit_co_mutation() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 2).await?;
        dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;

        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id IN (1, 3)")?
            .set_cell_flag("value", FLAG_NAME, true)?
            .build()?
            .execute()
            .await?;
        let dataset = result.new_dataset.as_ref().clone();
        assert_eq!(flagged_ids(&dataset, "value", FLAG_NAME).await?, vec![1, 3]);

        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 1")?
            .set("value", "NULL")?
            .build()?
            .execute()
            .await?;
        let dataset = result.new_dataset.as_ref().clone();
        assert_eq!(flagged_ids(&dataset, "value", FLAG_NAME).await?, vec![1, 3]);

        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 3")?
            .set("value", "300")?
            .set_cell_flag("value", FLAG_NAME, false)?
            .build()?
            .execute()
            .await?;
        let dataset = result.new_dataset.as_ref().clone();
        assert_eq!(flagged_ids(&dataset, "value", FLAG_NAME).await?, vec![1]);
        Ok(())
    }

    #[tokio::test]
    async fn append_uses_only_explicit_flag_changes() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 2).await?;
        let definition = dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;

        let schema: Arc<Schema> = Arc::new(dataset.schema().into());
        let explicitly_set = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![8, 9])),
                Arc::new(Int32Array::from(vec![None, Some(19)])),
            ],
        )?;
        dataset = Dataset::write_with_cell_flags(
            RecordBatchIterator::new([Ok(explicitly_set)], schema.clone()),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
            [CellFlagChange::new("value", FLAG_NAME, true)],
        )
        .await?;
        let root_before_false_appends = dataset
            .cell_flag_state(definition.flag_id)
            .expect("explicitly set rows create a root")
            .clone();

        let default_false = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![10, 11])),
                Arc::new(Int32Array::from(vec![Some(20), None])),
            ],
        )?;
        dataset = Dataset::write(
            RecordBatchIterator::new([Ok(default_false)], schema.clone()),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await?;
        assert_eq!(
            dataset.cell_flag_state(definition.flag_id),
            Some(&root_before_false_appends)
        );

        let explicitly_cleared = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![12, 13])),
                Arc::new(Int32Array::from(vec![Some(22), None])),
            ],
        )?;
        dataset = Dataset::write_with_cell_flags(
            RecordBatchIterator::new([Ok(explicitly_cleared)], schema),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
            [CellFlagChange::new("value", FLAG_NAME, false)],
        )
        .await?;
        assert_eq!(
            dataset.cell_flag_state(definition.flag_id),
            Some(&root_before_false_appends)
        );

        assert_eq!(flagged_ids(&dataset, "value", FLAG_NAME).await?, vec![8, 9]);
        Ok(())
    }

    #[tokio::test]
    async fn batch_append_preserves_exact_fragment_flag_states() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 2).await?;
        dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;
        let dataset = Arc::new(dataset);
        let schema: Arc<Schema> = Arc::new(dataset.schema().into());
        let params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };

        let transaction = |id: i32, value: Option<i32>| {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(vec![id])) as ArrayRef,
                    Arc::new(Int32Array::from(vec![value])) as ArrayRef,
                ],
            )?;
            Ok::<_, Error>(batch)
        };
        let first = InsertBuilder::new(dataset.clone())
            .with_params(&params)
            .with_cell_flags([CellFlagChange::new("value", FLAG_NAME, true)])
            .execute_uncommitted(vec![transaction(8, None)?])
            .await?;
        let second = InsertBuilder::new(dataset.clone())
            .with_params(&params)
            .with_cell_flags([CellFlagChange::new("value", FLAG_NAME, true)])
            .execute_uncommitted(vec![transaction(9, Some(19))?])
            .await?;

        let committed = CommitBuilder::new(dataset)
            .execute_batch(vec![first, second])
            .await?
            .dataset;
        assert_eq!(
            flagged_ids(&committed, "value", FLAG_NAME).await?,
            vec![8, 9]
        );
        Ok(())
    }

    #[tokio::test]
    async fn overwrite_initializes_only_explicit_new_row_state() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 2).await?;
        dataset.register_cell_flag("value", FLAG_NAME, true).await?;

        let schema: Arc<Schema> = Arc::new(dataset.schema().into());
        let default_false = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![20, 21])),
                Arc::new(Int32Array::from(vec![None, Some(121)])),
            ],
        )?;
        dataset = Dataset::write(
            RecordBatchIterator::new([Ok(default_false)], schema.clone()),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await?;
        assert!(flagged_ids(&dataset, "value", FLAG_NAME).await?.is_empty());

        let explicitly_set = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![30, 31])),
                Arc::new(Int32Array::from(vec![Some(130), None])),
            ],
        )?;
        dataset = Dataset::write_with_cell_flags(
            RecordBatchIterator::new([Ok(explicitly_set)], schema),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
            [CellFlagChange::new("value", FLAG_NAME, true)],
        )
        .await?;
        assert_eq!(
            flagged_ids(&dataset, "value", FLAG_NAME).await?,
            vec![30, 31]
        );
        Ok(())
    }

    #[tokio::test]
    async fn overwrite_rejects_flag_change_for_removed_field() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 2).await?;
        dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![20]))])?;

        let error = Dataset::write_with_cell_flags(
            RecordBatchIterator::new([Ok(batch)], schema),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
            [CellFlagChange::new("value", FLAG_NAME, true)],
        )
        .await
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("fragment state references unknown flag ID")
        );
        Ok(())
    }

    #[tokio::test]
    async fn merge_changes_flags_only_on_matched_rows_when_explicit() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 2).await?;
        dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;

        let joined_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("joined", DataType::Int32, true),
        ]));
        let joined = RecordBatch::try_new(
            joined_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 3])),
                Arc::new(Int32Array::from(vec![Some(101), None])),
            ],
        )?;
        dataset
            .merge_with_cell_flags(
                RecordBatchIterator::new([Ok(joined)], joined_schema),
                "id",
                "id",
                &[CellFlagChange::new("value", FLAG_NAME, true)],
            )
            .await?;
        assert_eq!(flagged_ids(&dataset, "value", FLAG_NAME).await?, vec![1, 3]);

        let ordinary_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("ordinary", DataType::Int32, true),
        ]));
        let ordinary = RecordBatch::try_new(
            ordinary_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(Int32Array::from(vec![Some(201), None])),
            ],
        )?;
        dataset
            .merge(
                RecordBatchIterator::new([Ok(ordinary)], ordinary_schema),
                "id",
                "id",
            )
            .await?;
        assert_eq!(flagged_ids(&dataset, "value", FLAG_NAME).await?, vec![1, 3]);

        let flag_only_schema =
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let flag_only = RecordBatch::try_new(
            flag_only_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![3]))],
        )?;
        let fragments_before_flag_only = dataset.manifest.fragments.clone();
        dataset
            .merge_with_cell_flags(
                RecordBatchIterator::new([Ok(flag_only)], flag_only_schema),
                "id",
                "id",
                &[CellFlagChange::new("value", FLAG_NAME, false)],
            )
            .await?;
        assert_eq!(dataset.manifest.fragments, fragments_before_flag_only);
        assert_eq!(flagged_ids(&dataset, "value", FLAG_NAME).await?, vec![1]);
        Ok(())
    }

    #[tokio::test]
    async fn rename_drop_and_time_travel_preserve_snapshot_contracts() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 2).await?;
        dataset.register_cell_flag("value", FLAG_NAME, true).await?;
        let registered_version = dataset.version().version;

        dataset
            .rename_cell_flag("value", FLAG_NAME, "ready")
            .await?;
        assert_eq!(
            flagged_ids(&dataset, "value", "ready").await?,
            (0..8).collect::<Vec<_>>()
        );
        let mut removed_name_scan = dataset.scan();
        removed_name_scan.filter(&format!("cell_flag(value, '{}')", FLAG_NAME))?;
        assert!(
            removed_name_scan
                .try_into_batch()
                .await
                .unwrap_err()
                .to_string()
                .contains("unknown flag")
        );

        let historical = dataset.checkout_version(registered_version).await?;
        assert_eq!(
            flagged_ids(&historical, "value", FLAG_NAME).await?,
            (0..8).collect::<Vec<_>>()
        );

        dataset.drop_cell_flag("value", "ready").await?;
        assert!(dataset.cell_flag_definitions().is_empty());
        let renamed_snapshot = dataset.checkout_version(registered_version + 1).await?;
        assert_eq!(
            flagged_ids(&renamed_snapshot, "value", "ready").await?,
            (0..8).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[tokio::test]
    async fn restore_preserves_flag_id_high_water_mark() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 2).await?;
        let pre_flag_version = dataset.version().version;
        let historical_definition = dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;
        assert_eq!(historical_definition.flag_id, 0);

        let mut restored = dataset.checkout_version(pre_flag_version).await?;
        restored.restore().await?;
        assert!(restored.cell_flag_definitions().is_empty());
        assert_eq!(restored.manifest.next_cell_flag_id, 1);

        let new_definition = restored.register_cell_flag("id", "reviewed", false).await?;
        assert_eq!(new_definition.flag_id, 1);
        let historical = restored.checkout_version(pre_flag_version + 1).await?;
        assert_eq!(
            historical.cell_flag_definitions(),
            std::slice::from_ref(&historical_definition)
        );
        Ok(())
    }

    #[tokio::test]
    async fn field_rename_cast_and_drop_keep_flag_identity_or_remove_definition() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 2).await?;
        let definition = dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id IN (1, 3)")?
            .set_cell_flag("value", FLAG_NAME, true)?
            .build()?
            .execute()
            .await?;
        let mut dataset = result.new_dataset.as_ref().clone();

        dataset
            .alter_columns(&[ColumnAlteration::new("value".into()).rename("renamed".into())])
            .await?;
        let renamed = &dataset.cell_flag_definitions()[0];
        assert_eq!(renamed.flag_id, definition.flag_id);
        assert_eq!(renamed.field_id, definition.field_id);
        assert_eq!(
            flagged_ids(&dataset, "renamed", FLAG_NAME).await?,
            vec![1, 3]
        );

        dataset
            .alter_columns(&[ColumnAlteration::new("renamed".into()).cast_to(DataType::Int64)])
            .await?;
        let cast = &dataset.cell_flag_definitions()[0];
        assert_eq!(cast.flag_id, definition.flag_id);
        assert_ne!(cast.field_id, definition.field_id);
        assert_eq!(
            flagged_ids(&dataset, "renamed", FLAG_NAME).await?,
            vec![1, 3]
        );

        dataset.drop_columns(&["renamed"]).await?;
        assert!(dataset.cell_flag_definitions().is_empty());
        assert_eq!(dataset.manifest.next_cell_flag_id, definition.flag_id + 1);
        Ok(())
    }

    #[tokio::test]
    async fn delete_and_compaction_preserve_flags() -> Result<()> {
        for enable_stable_row_ids in [false, true] {
            let all_false_directory = TempStrDir::default();
            let mut all_false =
                dataset_with_rows_and_stable_ids(&all_false_directory, 1, enable_stable_row_ids)
                    .await?;
            all_false
                .register_cell_flag("value", FLAG_NAME, false)
                .await?;
            compact_files(&mut all_false, CompactionOptions::default(), None).await?;
            assert!(
                flagged_ids(&all_false, "value", FLAG_NAME)
                    .await?
                    .is_empty()
            );

            let directory = TempStrDir::default();
            let mut dataset =
                dataset_with_rows_and_stable_ids(&directory, 1, enable_stable_row_ids).await?;
            dataset
                .register_cell_flag("value", FLAG_NAME, false)
                .await?;
            let result = UpdateBuilder::new(Arc::new(dataset))
                .update_where("id IN (1, 2, 5, 7)")?
                .set_cell_flag("value", FLAG_NAME, true)?
                .build()?
                .execute()
                .await?;
            let mut dataset = result.new_dataset.as_ref().clone();

            dataset.delete("id = 2").await?;
            compact_files(&mut dataset, CompactionOptions::default(), None).await?;
            assert_eq!(
                flagged_ids(&dataset, "value", FLAG_NAME).await?,
                vec![1, 5, 7]
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn merge_insert_applies_matched_and_inserted_changes() -> Result<()> {
        for enable_stable_row_ids in [false, true] {
            let directory = TempStrDir::default();
            let mut dataset =
                dataset_with_rows_and_stable_ids(&directory, 2, enable_stable_row_ids).await?;
            dataset
                .register_cell_flag("value", FLAG_NAME, false)
                .await?;

            let source_schema: Arc<Schema> = Arc::new(dataset.schema().into());
            let source = RecordBatch::try_new(
                source_schema.clone(),
                vec![
                    Arc::new(Int32Array::from(vec![1, 8])),
                    Arc::new(Int32Array::from(vec![Some(101), None])),
                ],
            )?;
            let mut builder =
                MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])?;
            builder
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::InsertAll)
                .set_matched_cell_flag("value", FLAG_NAME, false)?
                .set_inserted_cell_flag("value", FLAG_NAME, true)?;
            let (dataset, stats) = builder
                .try_build()?
                .execute_reader(Box::new(RecordBatchIterator::new(
                    [Ok(source)],
                    source_schema.clone(),
                )))
                .await?;
            assert_eq!(stats.num_updated_rows, 1);
            assert_eq!(stats.num_inserted_rows, 1);
            assert_eq!(flagged_ids(&dataset, "value", FLAG_NAME).await?, vec![8]);

            let ordinary_source = RecordBatch::try_new(
                source_schema.clone(),
                vec![
                    Arc::new(Int32Array::from(vec![8, 9])),
                    Arc::new(Int32Array::from(vec![808, 909])),
                ],
            )?;
            let mut ordinary_builder =
                MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])?;
            ordinary_builder
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::InsertAll);
            let (dataset, ordinary_stats) = ordinary_builder
                .try_build()?
                .execute_reader(Box::new(RecordBatchIterator::new(
                    [Ok(ordinary_source)],
                    source_schema.clone(),
                )))
                .await?;
            assert_eq!(ordinary_stats.num_updated_rows, 1);
            assert_eq!(ordinary_stats.num_inserted_rows, 1);
            assert_eq!(flagged_ids(&dataset, "value", FLAG_NAME).await?, vec![8]);

            let fragments_before = dataset.manifest.fragments.clone();
            let key_only_schema =
                Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
            let key_only = RecordBatch::try_new(
                key_only_schema.clone(),
                vec![Arc::new(Int32Array::from(vec![1]))],
            )?;
            let mut builder = MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])?;
            builder
                .when_not_matched(WhenNotMatched::DoNothing)
                .set_matched_cell_flag("value", FLAG_NAME, true)?;
            let (dataset, stats) = builder
                .try_build()?
                .execute_reader(Box::new(RecordBatchIterator::new(
                    [Ok(key_only)],
                    key_only_schema,
                )))
                .await?;
            assert_eq!(stats.num_updated_rows, 1);
            assert_eq!(stats.num_inserted_rows, 0);
            assert_eq!(dataset.manifest.fragments, fragments_before);
            assert_eq!(flagged_ids(&dataset, "value", FLAG_NAME).await?, vec![1, 8]);
        }
        Ok(())
    }

    #[tokio::test]
    async fn expression_composes_with_values_and_nulls() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 2).await?;
        dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id IN (1, 2, 3)")?
            .set_cell_flag("value", FLAG_NAME, true)?
            .build()?
            .execute()
            .await?;
        let dataset = result.new_dataset.as_ref();

        let mut scanner = dataset.scan();
        scanner.project(&["id"])?;
        scanner.filter(&format!(
            "cell_flag(value, '{}') AND value IS NULL",
            FLAG_NAME
        ))?;
        let batch = scanner.try_into_batch().await?;
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(ids.values(), &[1, 3]);

        let rows = flag_rows(dataset, "value", FLAG_NAME).await?;
        assert_eq!(
            rows.into_iter()
                .filter_map(|(id, value)| value.then_some(id))
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );

        let mut computed_field = dataset.scan();
        computed_field.filter(&format!("cell_flag(value + 1, '{}')", FLAG_NAME))?;
        assert!(
            computed_field
                .try_into_batch()
                .await
                .unwrap_err()
                .to_string()
                .contains("requires a direct field reference")
        );

        let mut dynamic_name = dataset.scan();
        dynamic_name.filter("cell_flag(value, CAST(id AS VARCHAR))")?;
        let dynamic_name_error = dynamic_name.try_into_batch().await.unwrap_err();
        assert!(
            dynamic_name_error
                .to_string()
                .contains("string literal name"),
            "{dynamic_name_error}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn pure_filter_pushes_exact_mask_and_mixed_filter_falls_back() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 2).await?;
        let definition = dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id IN (1, 3)")?
            .set_cell_flag("value", FLAG_NAME, true)?
            .build()?
            .execute()
            .await?;
        let dataset = result.new_dataset.as_ref();

        let mut pushed = dataset.scan();
        pushed.project(&["id"])?;
        pushed.filter(&format!("cell_flag(value, '{}')", FLAG_NAME))?;
        let pushed_plan = pushed.explain_plan(false).await?;
        assert!(pushed_plan.contains("ScalarIndexQuery"), "{pushed_plan}");
        assert!(
            pushed_plan.contains(&format!("CellFlag(flag_id={})", definition.flag_id)),
            "{pushed_plan}"
        );
        assert_eq!(flagged_ids(dataset, "value", FLAG_NAME).await?, vec![1, 3]);

        let mut fallback = dataset.scan();
        fallback.project(&["id"])?;
        fallback.filter(&format!("cell_flag(value, '{}') OR id = 7", FLAG_NAME))?;
        let fallback_plan = fallback.explain_plan(false).await?;
        assert!(
            fallback_plan.contains("full_filter=cell_flag"),
            "{fallback_plan}"
        );
        assert!(
            !fallback_plan.contains("ScalarIndexQuery"),
            "{fallback_plan}"
        );
        let batch = fallback.try_into_batch().await?;
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let mut ids = ids.values().to_vec();
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 3, 7]);
        Ok(())
    }

    #[tokio::test]
    async fn concurrent_mutations_rebase_disjoint_fragments_and_serialize_registry() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 2).await?;
        dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;

        let first_reader = Arc::new(dataset.clone());
        let second_reader = Arc::new(dataset.clone());
        UpdateBuilder::new(first_reader)
            .update_where("id = 1")?
            .set_cell_flag("value", FLAG_NAME, true)?
            .build()?
            .execute()
            .await?;
        let second = UpdateBuilder::new(second_reader)
            .update_where("id = 5")?
            .set_cell_flag("value", FLAG_NAME, true)?
            .build()?
            .execute()
            .await?;
        assert_eq!(
            flagged_ids(second.new_dataset.as_ref(), "value", FLAG_NAME).await?,
            vec![1, 5]
        );

        let same_fragment_first_reader = second.new_dataset.clone();
        let same_fragment_second_reader = second.new_dataset.clone();
        UpdateBuilder::new(same_fragment_first_reader)
            .update_where("id = 0")?
            .set_cell_flag("value", FLAG_NAME, true)?
            .build()?
            .execute()
            .await?;
        let same_fragment_second = UpdateBuilder::new(same_fragment_second_reader)
            .update_where("id = 1")?
            .set_cell_flag("value", FLAG_NAME, false)?
            .build()?
            .execute()
            .await?;
        assert_eq!(
            flagged_ids(
                same_fragment_second.new_dataset.as_ref(),
                "value",
                FLAG_NAME
            )
            .await?,
            vec![0, 5]
        );

        let current = same_fragment_second.new_dataset.as_ref().clone();
        let mut first_registry_writer = current.clone();
        let mut stale_registry_writer = current;
        let first_definition = first_registry_writer
            .register_cell_flag("value", "reviewed", false)
            .await?;
        let error = stale_registry_writer
            .register_cell_flag("id", "reviewed", false)
            .await
            .unwrap_err();
        assert!(
            matches!(error, Error::RetryableCommitConflict { .. }),
            "unexpected error: {error:?}"
        );

        let mut refreshed = Dataset::open(directory.as_ref()).await?;
        let second_definition = refreshed
            .register_cell_flag("id", "reviewed", false)
            .await?;
        assert_ne!(first_definition.flag_id, second_definition.flag_id);
        assert_eq!(
            refreshed
                .cell_flag_definitions()
                .iter()
                .map(|definition| definition.flag_id)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );

        let mut stale_drop = refreshed.clone();
        UpdateBuilder::new(Arc::new(refreshed))
            .update_where("id = 7")?
            .set_cell_flag("value", FLAG_NAME, true)?
            .build()?
            .execute()
            .await?;
        let error = stale_drop
            .drop_cell_flag("value", FLAG_NAME)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::RetryableCommitConflict { .. }));
        Ok(())
    }

    #[tokio::test]
    async fn field_drop_conflicts_with_stale_flag_change() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 2).await?;
        dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;
        let mut field_writer = dataset.clone();
        let flag_writer = Arc::new(dataset);

        field_writer.drop_columns(&["value"]).await?;
        let drop_transaction = field_writer
            .read_transaction()
            .await?
            .expect("field drop transaction");
        assert_eq!(
            drop_transaction
                .cell_flag_transaction()?
                .expect("field drop carries Cell Flag registry changes")
                .drops,
            vec![CellFlagDrop { flag_id: 0 }]
        );
        let error = UpdateBuilder::new(flag_writer)
            .update_where("id = 1")?
            .set_cell_flag("value", FLAG_NAME, true)?
            .build()?
            .execute()
            .await
            .unwrap_err();

        assert!(
            matches!(error, Error::IncompatibleTransaction { .. }),
            "unexpected error: {error:?}"
        );
        let current = Dataset::open(directory.as_ref()).await?;
        assert!(current.cell_flag_definitions().is_empty());
        assert!(current.schema().field("value").is_none());
        Ok(())
    }

    #[tokio::test]
    async fn flag_only_rebase_over_delete_does_not_write_orphan_deletion_file() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 8).await?;
        dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;
        let mut delete_writer = dataset.clone();
        let flag_writer = Arc::new(dataset);

        delete_writer.delete("id = 7").await?;
        let deletion_directory = std::path::Path::new(directory.as_ref()).join("_deletions");
        let files_before = std::fs::read_dir(&deletion_directory)?.count();

        let result = UpdateBuilder::new(flag_writer)
            .update_where("id = 1")?
            .set_cell_flag("value", FLAG_NAME, true)?
            .build()?
            .execute()
            .await?;

        assert_eq!(
            std::fs::read_dir(&deletion_directory)?.count(),
            files_before
        );
        assert_eq!(result.new_dataset.count_rows(None).await?, 7);
        assert_eq!(
            flagged_ids(result.new_dataset.as_ref(), "value", FLAG_NAME).await?,
            vec![1]
        );
        Ok(())
    }

    #[tokio::test]
    async fn clone_and_cleanup_preserve_snapshot_owned_flag_state() -> Result<()> {
        let source_directory = TempStrDir::default();
        let shallow_directory = TempStrDir::default();
        let deep_directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&source_directory, 2).await?;
        let definition = dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;

        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 1")?
            .set_cell_flag("value", FLAG_NAME, true)?
            .build()?
            .execute()
            .await?;
        let result = UpdateBuilder::new(result.new_dataset)
            .update_where("id = 3")?
            .set_cell_flag("value", FLAG_NAME, true)?
            .build()?
            .execute()
            .await?;
        let mut dataset = result.new_dataset.as_ref().clone();
        dataset
            .tags()
            .create("flagged", dataset.version().version)
            .await?;

        let shallow = dataset
            .shallow_clone(shallow_directory.as_ref(), "flagged", None)
            .await?;
        let deep = dataset
            .deep_clone(deep_directory.as_ref(), "flagged", None)
            .await?;
        assert_eq!(flagged_ids(&shallow, "value", FLAG_NAME).await?, vec![1, 3]);
        assert_eq!(flagged_ids(&deep, "value", FLAG_NAME).await?, vec![1, 3]);
        assert!(
            shallow
                .cell_flag_state(definition.flag_id)
                .unwrap()
                .root
                .base_id
                .is_some()
        );
        assert!(
            deep.cell_flag_state(definition.flag_id)
                .unwrap()
                .root
                .base_id
                .is_none()
        );

        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id IN (1, 5)")?
            .set_cell_flag("value", FLAG_NAME, false)?
            .build()?
            .execute()
            .await?;
        let dataset = result.new_dataset.as_ref().clone();
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 5")?
            .set_cell_flag("value", FLAG_NAME, true)?
            .build()?
            .execute()
            .await?;
        let dataset = result.new_dataset.as_ref().clone();
        assert_eq!(flagged_ids(&dataset, "value", FLAG_NAME).await?, vec![3, 5]);

        let policy = CleanupPolicyBuilder::default()
            .retain_n_versions(&dataset, 1)
            .await?
            .delete_unverified(true)
            .error_if_tagged_old_versions(false)
            .build();
        let explanation = dataset.cleanup(policy.clone()).explain().await?;
        assert!(
            explanation
                .candidate_files
                .iter()
                .any(|file| file.path.contains(CELL_FLAGS_DIR))
        );
        dataset.cleanup(policy).execute().await?;

        dataset
            .validate_cell_flag_root_object(definition.flag_id)
            .await?;
        dataset.validate().await?;
        shallow.validate().await?;
        deep.validate().await?;
        assert_eq!(flagged_ids(&dataset, "value", FLAG_NAME).await?, vec![3, 5]);
        assert_eq!(flagged_ids(&shallow, "value", FLAG_NAME).await?, vec![1, 3]);
        assert_eq!(flagged_ids(&deep, "value", FLAG_NAME).await?, vec![1, 3]);
        Ok(())
    }
}
