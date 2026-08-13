// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lazy I/O for snapshot-level field assignment state.

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
    AssignmentFragment, IS_ASSIGNED_NAME, bound_is_assigned_udf_with_coverage,
    is_unbound_is_assigned_udf,
};
#[cfg(feature = "substrait")]
use lance_datafusion::udf::{bound_is_assigned_field_id, is_assigned};
use lance_table::format::{
    FieldAssignmentFile, FieldAssignmentFragment, FieldAssignmentFragmentState,
    FieldAssignmentRoot, FieldAssignmentState, pb,
};
use object_store::path::Path;
use prost::Message;
use roaring::RoaringBitmap;
use uuid::Uuid;

use super::Dataset;
use super::transaction::{
    FieldAssignmentFragmentState as TransactionFieldAssignmentFragmentState,
    FieldAssignmentFragmentValue, Operation, Transaction, UpdateMode,
};
use crate::session::caches::FieldAssignmentFileKey;
use crate::{Error, Result};
use lance_core::utils::address::RowAddress;
use lance_io::object_store::ObjectStore;

pub const FIELD_ASSIGNMENTS_DIR: &str = "_field_assignments";
const FIELD_ASSIGNMENT_ROOTS_DIR: &str = "roots";
const FIELD_ASSIGNMENT_BITMAPS_DIR: &str = "bitmaps";
const MAX_FIELD_ASSIGNMENT_FILE_BYTES: u64 = 512 * 1024 * 1024;
/// Individual portable Roaring bitmaps at or below this size are embedded in
/// the root when they are also sparse, avoiding one metadata request per
/// fragment without turning dense mutation state into monolithic metadata.
const MAX_INLINE_FIELD_ASSIGNMENT_BITMAP_BYTES: usize = 64 * 1024;
/// Inline only states at or below 25% density; denser states benefit from
/// independent immutable objects during repeated sparse mutations.
const MAX_INLINE_FIELD_ASSIGNMENT_DENSITY_DENOMINATOR: u64 = 4;
/// Cap total embedded bitmap bytes so large/high-fragment-count roots retain
/// incremental immutable bitmap objects instead of becoming monolithic.
const MAX_INLINE_FIELD_ASSIGNMENT_ROOT_BYTES: usize = 4 * 1024 * 1024;
/// Small roots are copied into their manifest descriptor so cold planning does
/// not require a separate object-store request.
const MAX_INLINE_FIELD_ASSIGNMENT_ROOT_COPY_BYTES: usize = 64 * 1024;
/// Bound total root copies retained in one manifest across all tracked fields.
const MAX_INLINE_FIELD_ASSIGNMENT_MANIFEST_BYTES: usize = 4 * 1024 * 1024;

fn inline_field_assignment_bytes(file: &FieldAssignmentFile) -> Result<Option<Bytes>> {
    let Some(bytes) = file.inline_bytes.as_ref() else {
        return Ok(None);
    };
    if bytes.len() as u64 != file.size_bytes {
        return Err(Error::invalid_input(format!(
            "Inline field assignment file '{}' has size {}, expected {}",
            file.path,
            bytes.len(),
            file.size_bytes
        )));
    }
    if file.size_bytes > MAX_FIELD_ASSIGNMENT_FILE_BYTES {
        return Err(Error::invalid_input(format!(
            "Field assignment file '{}' has invalid declared size {}; maximum is {}",
            file.path, file.size_bytes, MAX_FIELD_ASSIGNMENT_FILE_BYTES
        )));
    }
    Ok(Some(Bytes::copy_from_slice(bytes)))
}

pub async fn read_field_assignment_bytes(
    store: &ObjectStore,
    path: &Path,
    file: &FieldAssignmentFile,
) -> Result<Bytes> {
    if let Some(bytes) = inline_field_assignment_bytes(file)? {
        return Ok(bytes);
    }
    if file.size_bytes > MAX_FIELD_ASSIGNMENT_FILE_BYTES {
        return Err(Error::invalid_input(format!(
            "Field assignment file '{}' has invalid declared size {}; maximum is {}",
            file.path, file.size_bytes, MAX_FIELD_ASSIGNMENT_FILE_BYTES
        )));
    }
    let known_size = usize::try_from(file.size_bytes).map_err(|_| {
        Error::invalid_input(format!(
            "Field assignment file '{}' size {} does not fit on this platform",
            file.path, file.size_bytes
        ))
    })?;
    let reader = store.open_with_size(path, known_size).await?;
    let bytes = reader.get_all().await?;
    if bytes.len() != known_size {
        return Err(Error::invalid_input(format!(
            "Field assignment file '{}' has size {}, expected {}",
            file.path,
            bytes.len(),
            file.size_bytes
        )));
    }
    Ok(bytes)
}

fn decode_field_assignment_bitmap(
    bytes: &[u8],
    source: &str,
    field_id: i32,
    fragment: &FieldAssignmentFragment,
) -> Result<RoaringBitmap> {
    let bitmap = RoaringBitmap::deserialize_from(&mut Cursor::new(bytes)).map_err(|error| {
        Error::invalid_input(format!(
            "Failed to decode assignment bitmap '{}' for field ID {}, fragment {}: {}",
            source, field_id, fragment.fragment_id, error
        ))
    })?;
    if bitmap.is_empty() || bitmap.len() == fragment.physical_rows {
        return Err(Error::invalid_input(format!(
            "Partial assignment bitmap '{}' for field ID {}, fragment {} must be non-empty and non-full",
            source, field_id, fragment.fragment_id
        )));
    }
    if bitmap
        .max()
        .is_some_and(|offset| offset as u64 >= fragment.physical_rows)
    {
        return Err(Error::invalid_input(format!(
            "Assignment bitmap '{}' for field ID {}, fragment {} contains an out-of-bounds row offset",
            source, field_id, fragment.fragment_id
        )));
    }
    Ok(bitmap)
}

impl Dataset {
    pub(crate) fn tracked_field_ids(&self) -> impl Iterator<Item = i32> + '_ {
        self.manifest
            .field_assignment_states
            .iter()
            .map(|state| state.field_id)
    }

    pub(crate) fn field_assignment_state(&self, field_id: i32) -> Option<&FieldAssignmentState> {
        self.manifest
            .field_assignment_states
            .binary_search_by_key(&field_id, |state| state.field_id)
            .ok()
            .map(|index| &self.manifest.field_assignment_states[index])
    }

    fn field_assignment_path(&self, file: &FieldAssignmentFile) -> Result<Path> {
        let relative = Path::parse(file.path.as_str())?;
        match file.base_id {
            Some(base_id) => {
                let base_path = self.manifest.base_paths.get(&base_id).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Dataset base path with ID {} not found for field assignment file '{}'",
                        base_id, file.path
                    ))
                })?;
                if !base_path.is_dataset_root {
                    return Err(Error::invalid_input(format!(
                        "Dataset base path with ID {} is not a dataset root for field assignment file '{}'",
                        base_id, file.path
                    )));
                }
                let base = base_path.extract_path(self.session.store_registry())?;
                Ok(Path::from_iter(base.parts().chain(relative.parts())))
            }
            None => Ok(Path::from_iter(self.base.parts().chain(relative.parts()))),
        }
    }

    fn field_assignment_source_uri<'a>(&'a self, file: &'a FieldAssignmentFile) -> Result<&'a str> {
        match file.base_id {
            Some(base_id) => self
                .manifest
                .base_paths
                .get(&base_id)
                .map(|base| base.path.as_str())
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Dataset base path with ID {} not found for field assignment file '{}'",
                        base_id, file.path
                    ))
                }),
            None => Ok(self.uri.as_str()),
        }
    }

    async fn read_field_assignment_file(&self, file: &FieldAssignmentFile) -> Result<Arc<Bytes>> {
        if let Some(bytes) = inline_field_assignment_bytes(file)? {
            return Ok(Arc::new(bytes));
        }
        let path = self.field_assignment_path(file)?;
        let source_uri = self.field_assignment_source_uri(file)?;
        let key = FieldAssignmentFileKey {
            source_uri,
            path: &file.path,
            size_bytes: file.size_bytes,
        };
        self.metadata_cache
            .get_or_insert_with_key(key, || async {
                let store = self.object_store(file.base_id).await?;
                read_field_assignment_bytes(&store, &path, file).await
            })
            .await
    }

    /// Load one field's immutable assignment root on demand.
    pub(crate) async fn load_field_assignment_root(
        &self,
        field_id: i32,
    ) -> Result<Option<FieldAssignmentRoot>> {
        let Some(descriptor) = self.field_assignment_state(field_id) else {
            return Ok(None);
        };
        descriptor.root.validate_root_path_for_field(field_id)?;
        let bytes = self.read_field_assignment_file(&descriptor.root).await?;
        let proto = pb::FieldAssignmentRoot::decode(bytes.as_ref().as_ref()).map_err(|error| {
            Error::invalid_input(format!(
                "Failed to decode field assignment root '{}' for field ID {}: {}",
                descriptor.root.path, field_id, error
            ))
        })?;
        let mut root = FieldAssignmentRoot::try_from(proto)?;

        let fragments_by_id = self
            .manifest
            .fragments
            .iter()
            .map(|fragment| (fragment.id, fragment))
            .collect::<HashMap<_, _>>();
        for entry in &mut root.fragments {
            let fragment = fragments_by_id.get(&entry.fragment_id).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Field assignment root for field ID {} references unknown fragment {}",
                    field_id, entry.fragment_id
                ))
            })?;
            let physical_rows = fragment.physical_rows.ok_or_else(|| {
                Error::invalid_input(format!(
                    "Fragment {} has no physical row count for field assignment",
                    entry.fragment_id
                ))
            })?;
            if entry.physical_rows != physical_rows as u64 {
                return Err(Error::invalid_input(format!(
                    "Field assignment root for field ID {} records {} physical rows for fragment {}, manifest records {}",
                    field_id, entry.physical_rows, entry.fragment_id, physical_rows
                )));
            }
            if let FieldAssignmentFragmentState::Partial(file) = &mut entry.state
                && file.base_id.is_none()
            {
                // Bitmap paths are relative to the dataset that owns the root.
                // Materialize that inheritance before a future root reuses the page.
                file.base_id = descriptor.root.base_id;
            }
            if let FieldAssignmentFragmentState::Partial(file) = &entry.state {
                file.validate_bitmap_path_for_fragment(field_id, entry.fragment_id)?;
            }
        }
        Ok(Some(root))
    }

    /// Verify that an inline root copy exactly matches its referenced immutable
    /// object. Normal queries trust the manifest copy; explicit validation
    /// checks both representations and object reachability.
    pub(crate) async fn validate_field_assignment_root_object(&self, field_id: i32) -> Result<()> {
        let Some(descriptor) = self.field_assignment_state(field_id) else {
            return Ok(());
        };
        let Some(inline_bytes) = descriptor.root.inline_bytes.as_ref() else {
            return Ok(());
        };
        let mut object_file = descriptor.root.clone();
        object_file.inline_bytes = None;
        let path = self.field_assignment_path(&object_file)?;
        let store = self.object_store(object_file.base_id).await?;
        let object_bytes = read_field_assignment_bytes(&store, &path, &object_file).await?;
        if object_bytes.as_ref() != inline_bytes.as_slice() {
            return Err(Error::invalid_input(format!(
                "Inline field assignment root '{}' for field ID {} does not match its immutable object",
                descriptor.root.path, field_id
            )));
        }
        Ok(())
    }

    /// Load and validate a partial assignment bitmap.
    pub(crate) async fn load_field_assignment_bitmap(
        &self,
        field_id: i32,
        fragment: &FieldAssignmentFragment,
    ) -> Result<RoaringBitmap> {
        let bitmap = match &fragment.state {
            FieldAssignmentFragmentState::Partial(file) => {
                file.validate_bitmap_path_for_fragment(field_id, fragment.fragment_id)?;
                let bytes = self.read_field_assignment_file(file).await?;
                decode_field_assignment_bitmap(
                    bytes.as_ref().as_ref(),
                    &file.path,
                    field_id,
                    fragment,
                )?
            }
            FieldAssignmentFragmentState::InlinePartial(bytes) => {
                decode_field_assignment_bitmap(bytes, "inline root entry", field_id, fragment)?
            }
            FieldAssignmentFragmentState::All => {
                return Err(Error::internal(format!(
                    "Field assignment bitmap requested for all-assigned fragment {}",
                    fragment.fragment_id
                )));
            }
        };
        Ok(bitmap)
    }

    pub(crate) async fn write_field_assignment_bitmap(
        &self,
        write_store: &ObjectStore,
        field_id: i32,
        fragment_id: u64,
        physical_rows: u64,
        bitmap: &RoaringBitmap,
    ) -> Result<FieldAssignmentFragmentState> {
        if bitmap.is_empty() || bitmap.len() == physical_rows {
            return Err(Error::internal(format!(
                "Only partial assignment bitmaps may be written for field ID {}, fragment {}",
                field_id, fragment_id
            )));
        }
        if bitmap
            .max()
            .is_some_and(|offset| offset as u64 >= physical_rows)
        {
            return Err(Error::internal(format!(
                "Assignment bitmap for field ID {}, fragment {} contains an out-of-bounds row offset",
                field_id, fragment_id
            )));
        }
        let mut bytes = Vec::with_capacity(bitmap.serialized_size());
        bitmap.serialize_into(&mut bytes)?;
        let is_sparse =
            bitmap.len() <= physical_rows / MAX_INLINE_FIELD_ASSIGNMENT_DENSITY_DENOMINATOR;
        if is_sparse && bytes.len() <= MAX_INLINE_FIELD_ASSIGNMENT_BITMAP_BYTES {
            return Ok(FieldAssignmentFragmentState::InlinePartial(bytes));
        }
        Ok(FieldAssignmentFragmentState::Partial(
            self.write_field_assignment_bitmap_file(write_store, field_id, fragment_id, &bytes)
                .await?,
        ))
    }

    async fn write_field_assignment_bitmap_file(
        &self,
        write_store: &ObjectStore,
        field_id: i32,
        fragment_id: u64,
        bytes: &[u8],
    ) -> Result<FieldAssignmentFile> {
        let relative = Path::from(FIELD_ASSIGNMENTS_DIR)
            .join(FIELD_ASSIGNMENT_BITMAPS_DIR)
            .join(field_id.to_string())
            .join(fragment_id.to_string())
            .join(format!("{}.rbm", Uuid::new_v4()));
        let full_path = Path::from_iter(self.base.parts().chain(relative.parts()));
        write_store.put(&full_path, bytes).await?;
        Ok(FieldAssignmentFile {
            path: relative.to_string(),
            size_bytes: bytes.len() as u64,
            base_id: None,
            inline_bytes: None,
        })
    }

    pub(crate) async fn write_field_assignment_root(
        &self,
        write_store: &ObjectStore,
        field_id: i32,
        mut root: FieldAssignmentRoot,
    ) -> Result<FieldAssignmentState> {
        root.fragments.sort_by_key(|entry| entry.fragment_id);
        if root
            .fragments
            .windows(2)
            .any(|entries| entries[0].fragment_id == entries[1].fragment_id)
        {
            return Err(Error::internal(format!(
                "Field assignment root for field ID {} contains duplicate fragment IDs",
                field_id
            )));
        }
        let mut inline_bytes = 0usize;
        for entry in &mut root.fragments {
            let spill = match &entry.state {
                FieldAssignmentFragmentState::InlinePartial(bytes) => {
                    let next_inline_bytes =
                        inline_bytes.checked_add(bytes.len()).ok_or_else(|| {
                            Error::internal(format!(
                                "Inline assignment bitmap size overflow for field ID {}",
                                field_id
                            ))
                        })?;
                    if next_inline_bytes <= MAX_INLINE_FIELD_ASSIGNMENT_ROOT_BYTES {
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
                    .write_field_assignment_bitmap_file(
                        write_store,
                        field_id,
                        entry.fragment_id,
                        &bytes,
                    )
                    .await?;
                entry.state = FieldAssignmentFragmentState::Partial(file);
            }
        }
        let bytes = pb::FieldAssignmentRoot::from(&root).encode_to_vec();
        if bytes.len() as u64 > MAX_FIELD_ASSIGNMENT_FILE_BYTES {
            return Err(Error::internal(format!(
                "Field assignment root for field ID {} has encoded size {}, maximum is {}",
                field_id,
                bytes.len(),
                MAX_FIELD_ASSIGNMENT_FILE_BYTES
            )));
        }
        let relative = Path::from(FIELD_ASSIGNMENTS_DIR)
            .join(FIELD_ASSIGNMENT_ROOTS_DIR)
            .join(field_id.to_string())
            .join(format!("{}.root", Uuid::new_v4()));
        let full_path = Path::from_iter(self.base.parts().chain(relative.parts()));
        write_store.put(&full_path, &bytes).await?;
        Ok(FieldAssignmentState {
            field_id,
            root: FieldAssignmentFile {
                path: relative.to_string(),
                size_bytes: bytes.len() as u64,
                base_id: None,
                inline_bytes: (bytes.len() <= MAX_INLINE_FIELD_ASSIGNMENT_ROOT_COPY_BYTES)
                    .then(|| bytes.to_vec()),
            },
        })
    }

    /// Build exact states for newly inserted fragments from the stable field
    /// IDs supplied by the logical write.
    pub(crate) fn field_assignment_states_for_new_fragments(
        &self,
        new_fragments: &[lance_table::format::Fragment],
        supplied_field_ids: &HashSet<i32>,
    ) -> Result<Vec<TransactionFieldAssignmentFragmentState>> {
        let mut states = Vec::new();
        for field_id in self.tracked_field_ids() {
            let state = if supplied_field_ids.contains(&field_id) {
                FieldAssignmentFragmentValue::All
            } else {
                FieldAssignmentFragmentValue::None
            };
            for fragment in new_fragments {
                states.push(TransactionFieldAssignmentFragmentState {
                    fragment_path: fragment_path(fragment)?.to_string(),
                    field_id,
                    state: state.clone(),
                });
            }
        }
        Ok(states)
    }

    /// Remap assignment membership from rows in the current snapshot to newly
    /// written fragments. Logical writes and invalidations override the source
    /// state for their fields.
    pub(crate) async fn field_assignment_states_for_rewritten_rows(
        &self,
        new_fragments: &[lance_table::format::Fragment],
        source_row_addresses: &[u64],
        assigned_field_ids: &HashSet<i32>,
        invalidated_field_ids: &HashSet<i32>,
    ) -> Result<Vec<TransactionFieldAssignmentFragmentState>> {
        if !assigned_field_ids.is_disjoint(invalidated_field_ids) {
            return Err(Error::invalid_input(
                "A field cannot be assigned and invalidated by the same mutation",
            ));
        }
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
                "Rewritten fragments contain {} rows but {} source row addresses were captured",
                output_rows,
                source_row_addresses.len()
            )));
        }

        let source_fragment_ids = source_row_addresses
            .iter()
            .map(|address| RowAddress::from(*address).fragment_id() as u64)
            .collect::<HashSet<_>>();
        let mut states = Vec::new();
        for field_id in self.tracked_field_ids() {
            if assigned_field_ids.contains(&field_id) || invalidated_field_ids.contains(&field_id) {
                let state = if assigned_field_ids.contains(&field_id) {
                    FieldAssignmentFragmentValue::All
                } else {
                    FieldAssignmentFragmentValue::None
                };
                for fragment in new_fragments {
                    states.push(TransactionFieldAssignmentFragmentState {
                        fragment_path: fragment_path(fragment)?.to_string(),
                        field_id,
                        state: state.clone(),
                    });
                }
                continue;
            }

            let root = self
                .load_field_assignment_root(field_id)
                .await?
                .expect("tracked field must have an assignment root");
            let mut source_states = HashMap::new();
            for fragment in root
                .fragments
                .into_iter()
                .filter(|fragment| source_fragment_ids.contains(&fragment.fragment_id))
            {
                let state = match &fragment.state {
                    FieldAssignmentFragmentState::All => None,
                    FieldAssignmentFragmentState::Partial(_)
                    | FieldAssignmentFragmentState::InlinePartial(_) => Some(
                        self.load_field_assignment_bitmap(field_id, &fragment)
                            .await?,
                    ),
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
                for (new_offset, address) in source_row_addresses
                    [source_cursor..source_cursor + physical_rows]
                    .iter()
                    .enumerate()
                {
                    let address = RowAddress::from(*address);
                    let assigned = match source_states.get(&(address.fragment_id() as u64)) {
                        Some(None) => true,
                        Some(Some(source_bitmap)) => source_bitmap.contains(address.row_offset()),
                        None => false,
                    };
                    if assigned {
                        bitmap.insert(new_offset as u32);
                    }
                }
                source_cursor += physical_rows;
                let state = if bitmap.is_empty() {
                    FieldAssignmentFragmentValue::None
                } else if bitmap.len() == physical_rows as u64 {
                    FieldAssignmentFragmentValue::All
                } else {
                    FieldAssignmentFragmentValue::Partial(bitmap)
                };
                states.push(TransactionFieldAssignmentFragmentState {
                    fragment_path: fragment_path(fragment)?.to_string(),
                    field_id,
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
                    "is_assigned field reference contains get_field with {} arguments, expected 2",
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
                        "is_assigned(field) requires a direct field reference",
                    ));
                }
            };
            segments.push(child);
            Ok(segments)
        }
        _ => Err(Error::invalid_input(
            "is_assigned(field) requires exactly one direct field reference",
        )),
    }
}

fn is_assignment_call(expr: &Expr) -> Result<Option<Vec<String>>> {
    let Expr::ScalarFunction(function) = expr else {
        return Ok(None);
    };
    if !function.func.name().eq_ignore_ascii_case(IS_ASSIGNED_NAME)
        || !is_unbound_is_assigned_udf(function.func.as_ref())
    {
        return Ok(None);
    }
    if function.args.len() != 1 {
        return Err(Error::invalid_input(format!(
            "is_assigned expects exactly one field reference, received {} arguments",
            function.args.len()
        )));
    }
    field_reference_segments(&function.args[0]).map(Some)
}

pub fn expression_references_field_assignment(expression: &Expr) -> Result<bool> {
    let mut references_assignment = false;
    let mut captured_error = None;
    expression
        .apply(|node| {
            match is_assignment_call(node) {
                Ok(Some(_)) => references_assignment = true,
                Ok(None) => {}
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
    Ok(references_assignment)
}

fn resolve_assignment_field(dataset: &Dataset, segments: &[String]) -> Result<i32> {
    let Some(first) = segments.first() else {
        return Err(Error::invalid_input(
            "is_assigned(field) requires a non-empty field reference",
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
                "is_assigned references unknown field '{}'",
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
                    "is_assigned references unknown field '{}'",
                    segments.join(".")
                ))
            })?;
    }
    Ok(field.id)
}

pub fn field_assignment_call_field_id(dataset: &Dataset, expression: &Expr) -> Result<Option<i32>> {
    is_assignment_call(expression)?
        .map(|segments| resolve_assignment_field(dataset, &segments))
        .transpose()
}

pub async fn load_field_assignment_fragments(
    dataset: &Dataset,
    field_id: i32,
    selected_fragment_ids: Option<Arc<HashSet<u32>>>,
) -> Result<HashMap<u32, AssignmentFragment>> {
    if dataset.field_assignment_state(field_id).is_none() {
        let name = dataset
            .schema()
            .field_by_id(field_id)
            .map(|field| field.name.as_str())
            .unwrap_or("<unknown>");
        return Err(Error::invalid_input(format!(
            "is_assigned references untracked field '{}' (stable field ID {})",
            name, field_id
        )));
    }
    let root = dataset
        .load_field_assignment_root(field_id)
        .await?
        .ok_or_else(|| {
            Error::internal(format!(
                "Missing assignment root for tracked field ID {}",
                field_id
            ))
        })?;
    let io_parallelism = dataset.object_store.io_parallelism().max(1);
    futures::stream::iter(root.fragments)
        .map(|fragment| {
            let selected_fragment_ids = selected_fragment_ids.clone();
            async move {
                let fragment_id = u32::try_from(fragment.fragment_id).map_err(|_| {
                    Error::invalid_input(format!(
                        "Field assignment fragment ID {} does not fit in a row address",
                        fragment.fragment_id
                    ))
                })?;
                if selected_fragment_ids
                    .as_ref()
                    .is_some_and(|selected| !selected.contains(&fragment_id))
                {
                    return Ok::<Option<(u32, AssignmentFragment)>, Error>(None);
                }
                let state = match fragment.state {
                    FieldAssignmentFragmentState::All => AssignmentFragment::All,
                    FieldAssignmentFragmentState::Partial(_)
                    | FieldAssignmentFragmentState::InlinePartial(_) => {
                        let bitmap = dataset
                            .load_field_assignment_bitmap(field_id, &fragment)
                            .await?;
                        AssignmentFragment::Partial(Arc::new(bitmap))
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
pub fn unbind_field_assignment_expression(dataset: &Dataset, expression: Expr) -> Result<Expr> {
    Ok(expression
        .transform(|node| {
            let Expr::ScalarFunction(function) = &node else {
                return Ok(Transformed::no(node));
            };
            let Some(field_id) = bound_is_assigned_field_id(function.func.as_ref()) else {
                return Ok(Transformed::no(node));
            };
            if dataset.schema().field_by_id(field_id).is_none()
                || dataset.field_assignment_state(field_id).is_none()
            {
                return Err(Error::invalid_input(format!(
                    "Bound is_assigned references untracked or missing stable field ID {}",
                    field_id
                ))
                .into());
            }
            Ok(Transformed::yes(is_assigned(Expr::Literal(
                ScalarValue::Int32(Some(field_id)),
                None,
            ))))
        })
        .map(|transformed| transformed.data)?)
}

#[cfg(feature = "substrait")]
pub async fn bind_field_assignment_expression(
    dataset: &Dataset,
    expression: Expr,
    selected_fragments: Option<&[lance_table::format::Fragment]>,
) -> Result<Expr> {
    let planner = lance_datafusion::planner::Planner::new(Arc::new(arrow_schema::Schema::from(
        dataset.schema(),
    )));
    let expression = expression
        .transform(|node| {
            let Expr::ScalarFunction(function) = &node else {
                return Ok(Transformed::no(node));
            };
            if !is_unbound_is_assigned_udf(function.func.as_ref()) || function.args.len() != 1 {
                return Ok(Transformed::no(node));
            }
            let Expr::Literal(ScalarValue::Int32(Some(field_id)), _) = &function.args[0] else {
                return Ok(Transformed::no(node));
            };
            let field_path = dataset.schema().field_path(*field_id)?;
            let field = planner.parse_expr(&field_path)?;
            Ok(Transformed::yes(is_assigned(field)))
        })
        .map(|transformed| transformed.data)?;
    let Some(bindings) =
        FieldAssignmentExprBindings::try_new(dataset, &[&expression], selected_fragments).await?
    else {
        return Ok(expression);
    };
    bindings.bind(expression, dataset)
}

/// Snapshot bindings shared by every assignment expression in one scanner.
pub struct FieldAssignmentExprBindings {
    functions: HashMap<i32, Arc<ScalarUDF>>,
}

impl FieldAssignmentExprBindings {
    pub(crate) async fn try_new(
        dataset: &Dataset,
        expressions: &[&Expr],
        selected_fragments: Option<&[lance_table::format::Fragment]>,
    ) -> Result<Option<Self>> {
        let mut field_ids = HashSet::new();
        let mut captured_error = None;
        for expression in expressions {
            expression
                .apply(|node| {
                    match is_assignment_call(node) {
                        Ok(Some(segments)) => match resolve_assignment_field(dataset, &segments) {
                            Ok(field_id) => {
                                field_ids.insert(field_id);
                            }
                            Err(error) => {
                                captured_error = Some(error);
                                return Ok(TreeNodeRecursion::Stop);
                            }
                        },
                        Ok(None) => {}
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
        if field_ids.is_empty() {
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
        let io_parallelism = dataset.object_store.io_parallelism().max(1);
        let functions = futures::stream::iter(field_ids)
            .map(|field_id| {
                let selected_fragment_ids = selected_fragment_ids.clone();
                let covered_fragments = covered_fragments.clone();
                async move {
                    let fragments =
                        load_field_assignment_fragments(dataset, field_id, selected_fragment_ids)
                            .await?;
                    Ok::<(i32, Arc<ScalarUDF>), Error>((
                        field_id,
                        Arc::new(bound_is_assigned_udf_with_coverage(
                            field_id,
                            fragments,
                            covered_fragments,
                        )),
                    ))
                }
            })
            .buffer_unordered(io_parallelism)
            .try_collect::<HashMap<_, _>>()
            .await?;
        Ok(Some(Self { functions }))
    }

    pub(crate) fn bind(&self, expression: Expr, dataset: &Dataset) -> Result<Expr> {
        let mut captured_error = None;
        let transformed = expression
            .transform(|node| {
                let Some(segments) = (match is_assignment_call(&node) {
                    Ok(call) => call,
                    Err(error) => {
                        captured_error = Some(error);
                        return Ok(Transformed::no(node));
                    }
                }) else {
                    return Ok(Transformed::no(node));
                };
                let field_id = match resolve_assignment_field(dataset, &segments) {
                    Ok(field_id) => field_id,
                    Err(error) => {
                        captured_error = Some(error);
                        return Ok(Transformed::no(node));
                    }
                };
                let Some(function) = self.functions.get(&field_id) else {
                    captured_error = Some(Error::internal(format!(
                        "Missing is_assigned binding for field ID {}",
                        field_id
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
                "Fragment {} has no data file to identify assignment state",
                fragment.id
            ))
        })
}

fn fragment_supplies_field(fragment: &lance_table::format::Fragment, field_id: i32) -> bool {
    fragment
        .files
        .iter()
        .any(|file| file.fields.contains(&field_id))
}

fn fragment_physical_rows(fragment: &lance_table::format::Fragment) -> Result<u64> {
    fragment
        .physical_rows
        .map(|rows| rows as u64)
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "Fragment {} has no physical row count for field assignment",
                fragment.id
            ))
        })
}

async fn materialize_fragment_bitmap(
    current: &Dataset,
    field_id: i32,
    fragment_id: u64,
    entry: Option<&FieldAssignmentFragment>,
) -> Result<RoaringBitmap> {
    let Some(entry) = entry else {
        return Ok(RoaringBitmap::new());
    };
    match &entry.state {
        FieldAssignmentFragmentState::All => {
            let mut bitmap = RoaringBitmap::new();
            bitmap.insert_range(
                0..u32::try_from(entry.physical_rows).map_err(|_| {
                    Error::invalid_input(format!(
                        "Fragment {} has too many rows for field assignment",
                        fragment_id
                    ))
                })?,
            );
            Ok(bitmap)
        }
        FieldAssignmentFragmentState::Partial(_)
        | FieldAssignmentFragmentState::InlinePartial(_) => {
            current.load_field_assignment_bitmap(field_id, entry).await
        }
    }
}

#[derive(Default)]
struct PendingRowChanges {
    assigned: RoaringBitmap,
    unassigned: RoaringBitmap,
}

fn add_row_change(
    pending: &mut HashMap<(i32, u64), PendingRowChanges>,
    field_id: i32,
    fragment_id: u64,
    offsets: &RoaringBitmap,
    assigned: bool,
) {
    let change = pending.entry((field_id, fragment_id)).or_default();
    if assigned {
        change.assigned |= offsets;
        change.unassigned -= offsets;
    } else {
        change.unassigned |= offsets;
        change.assigned -= offsets;
    }
}

fn validate_fragment_value(
    field_id: i32,
    fragment_id: u64,
    physical_rows: u64,
    value: &FieldAssignmentFragmentValue,
) -> Result<()> {
    if let FieldAssignmentFragmentValue::Partial(bitmap) = value {
        if bitmap.is_empty() || bitmap.len() == physical_rows {
            return Err(Error::invalid_input(format!(
                "Partial assignment state for field ID {}, fragment {} must be non-empty and non-full",
                field_id, fragment_id
            )));
        }
        if bitmap
            .max()
            .is_some_and(|offset| offset as u64 >= physical_rows)
        {
            return Err(Error::invalid_input(format!(
                "Assignment state for field ID {}, fragment {} contains an out-of-bounds row offset",
                field_id, fragment_id
            )));
        }
    }
    Ok(())
}

/// Reconcile one transaction with the current head's assignment roots.
///
/// This runs after conflict resolution and manifest construction on every
/// commit attempt, so compatible concurrent appends and unrelated field writes
/// are merged from the actual head rather than from the stale read snapshot.
pub async fn apply_field_assignment_transaction(
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

    let changes = transaction
        .field_assignment_transaction()?
        .unwrap_or_default();
    let mut initializations = HashMap::new();
    for initialization in &changes.initializations {
        if initializations
            .insert(initialization.field_id, initialization.assigned)
            .is_some()
        {
            return Err(Error::invalid_input(format!(
                "Field ID {} is initialized for assignment tracking more than once",
                initialization.field_id
            )));
        }
        if current
            .manifest
            .field_assignment_states
            .iter()
            .any(|state| state.field_id == initialization.field_id)
        {
            return Err(Error::invalid_input(format!(
                "Field ID {} already has assignment tracking",
                initialization.field_id
            )));
        }
    }

    let final_field_ids: HashSet<i32> = manifest.schema.field_ids().into_iter().collect();
    let mut transfer_sources = HashMap::new();
    for transfer in &changes.transfers {
        if transfer.source_field_id == transfer.target_field_id {
            return Err(Error::invalid_input(format!(
                "Field assignment transfer source and target are both {}",
                transfer.source_field_id
            )));
        }
        if transfer_sources
            .insert(transfer.target_field_id, transfer.source_field_id)
            .is_some()
        {
            return Err(Error::invalid_input(format!(
                "Field ID {} is the target of more than one assignment transfer",
                transfer.target_field_id
            )));
        }
        if current
            .field_assignment_state(transfer.source_field_id)
            .is_none()
        {
            return Err(Error::invalid_input(format!(
                "Assignment transfer source field ID {} is not tracked",
                transfer.source_field_id
            )));
        }
        if final_field_ids.contains(&transfer.source_field_id)
            || !final_field_ids.contains(&transfer.target_field_id)
        {
            return Err(Error::invalid_input(format!(
                "Assignment transfer must replace removed field ID {} with current field ID {}",
                transfer.source_field_id, transfer.target_field_id
            )));
        }
        if initializations.contains_key(&transfer.target_field_id) {
            return Err(Error::invalid_input(format!(
                "Field ID {} cannot be initialized and receive an assignment transfer",
                transfer.target_field_id
            )));
        }
    }

    for field_id in initializations.keys() {
        if !final_field_ids.contains(field_id) {
            return Err(Error::invalid_input(format!(
                "Cannot initialize assignment tracking for unknown field ID {}",
                field_id
            )));
        }
    }

    let mut tracked_fields: Vec<i32> = current
        .manifest
        .field_assignment_states
        .iter()
        .map(|state| state.field_id)
        .chain(initializations.keys().copied())
        .chain(transfer_sources.keys().copied())
        .filter(|field_id| final_field_ids.contains(field_id))
        .collect();
    tracked_fields.sort_unstable();
    tracked_fields.dedup();
    if tracked_fields.is_empty() {
        manifest.field_assignment_states.clear();
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

    let mut overrides: HashMap<(i32, u64), &FieldAssignmentFragmentValue> = HashMap::new();
    for override_state in &changes.fragment_states {
        let fragment = fragment_by_path
            .get(override_state.fragment_path.as_str())
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Field assignment state references unknown new fragment path '{}'",
                    override_state.fragment_path
                ))
            })?;
        if overrides
            .insert(
                (override_state.field_id, fragment.id),
                &override_state.state,
            )
            .is_some()
        {
            return Err(Error::invalid_input(format!(
                "Duplicate field assignment state for field ID {}, fragment path '{}'",
                override_state.field_id, override_state.fragment_path
            )));
        }
    }

    let mut pending_changes: HashMap<(i32, u64), PendingRowChanges> = HashMap::new();
    for change in &changes.row_changes {
        if !tracked_fields.contains(&change.field_id) {
            return Err(Error::invalid_input(format!(
                "Assignment row change references untracked field ID {}",
                change.field_id
            )));
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
                change.field_id,
                fragment_id,
                &offsets,
                change.assigned,
            );
        }
    }

    if let Operation::Update {
        fields_modified,
        updated_fragments,
        updated_fragment_offsets,
        update_mode,
        ..
    } = &transaction.operation
    {
        let tracked_modified_fields = fields_modified
            .iter()
            .map(|field_id| *field_id as i32)
            .filter(|field_id| tracked_fields.contains(field_id))
            .collect::<Vec<_>>();
        if !tracked_modified_fields.is_empty()
            && !updated_fragments.is_empty()
            && !matches!(update_mode, Some(UpdateMode::RewriteColumns))
        {
            return Err(Error::invalid_input(
                "An in-place update of tracked fields must set update_mode='rewrite_columns' and supply exact updated_fragment_offsets",
            ));
        }
        for field_id in tracked_modified_fields {
            for fragment in updated_fragments {
                let offsets = updated_fragment_offsets
                    .as_ref()
                    .and_then(|all| all.0.get(&fragment.id))
                    .cloned()
                    .ok_or_else(|| {
                        Error::invalid_input(format!(
                            "In-place update of tracked field ID {} is missing exact updated_fragment_offsets for fragment {}",
                            field_id, fragment.id
                        ))
                    })?;
                add_row_change(&mut pending_changes, field_id, fragment.id, &offsets, true);
            }
        }
    }

    if let Operation::DataOverlay { groups } = &transaction.operation {
        for group in groups {
            for overlay in &group.overlays {
                for (field_position, field_id) in overlay.data_file.fields.iter().enumerate() {
                    if tracked_fields.contains(field_id) {
                        add_row_change(
                            &mut pending_changes,
                            *field_id,
                            group.fragment_id,
                            overlay.coverage_for_field(field_position)?.as_ref(),
                            true,
                        );
                    }
                }
            }
        }
    }

    if let Operation::DataReplacement { replacements } = &transaction.operation {
        for replacement in replacements {
            let fragment = final_fragments.get(&replacement.0).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Data replacement references fragment {} that is not in the new snapshot",
                    replacement.0
                ))
            })?;
            let physical_rows = u32::try_from(fragment_physical_rows(fragment)?).map_err(|_| {
                Error::invalid_input(format!(
                    "Fragment {} has too many rows for field assignment",
                    replacement.0
                ))
            })?;
            let mut all_rows = RoaringBitmap::new();
            all_rows.insert_range(0..physical_rows);
            for field_id in replacement
                .1
                .fields
                .iter()
                .copied()
                .filter(|field_id| tracked_fields.contains(field_id))
            {
                add_row_change(
                    &mut pending_changes,
                    field_id,
                    replacement.0,
                    &all_rows,
                    true,
                );
            }
        }
    }

    let mut descriptors = Vec::with_capacity(tracked_fields.len());
    for field_id in tracked_fields {
        let source_field_id = transfer_sources.get(&field_id).copied().unwrap_or(field_id);
        let has_explicit_change = initializations.contains_key(&field_id)
            || overrides
                .keys()
                .any(|(override_field_id, _)| *override_field_id == field_id)
            || pending_changes
                .keys()
                .any(|(change_field_id, _)| *change_field_id == field_id);
        if fragment_ids_unchanged && !has_explicit_change && source_field_id != field_id {
            let mut root = current
                .load_field_assignment_root(source_field_id)
                .await?
                .ok_or_else(|| {
                    Error::internal(format!(
                        "Missing assignment root for transferred field ID {}",
                        source_field_id
                    ))
                })?;
            for fragment in &mut root.fragments {
                if matches!(fragment.state, FieldAssignmentFragmentState::Partial(_)) {
                    let bitmap = current
                        .load_field_assignment_bitmap(source_field_id, fragment)
                        .await?;
                    let state = current
                        .write_field_assignment_bitmap(
                            write_store,
                            field_id,
                            fragment.fragment_id,
                            fragment.physical_rows,
                            &bitmap,
                        )
                        .await?;
                    fragment.state = state;
                }
            }
            descriptors.push(
                current
                    .write_field_assignment_root(write_store, field_id, root)
                    .await?,
            );
            continue;
        }
        if fragment_ids_unchanged
            && !has_explicit_change
            && let Some(descriptor) = current.field_assignment_state(field_id)
        {
            descriptors.push(descriptor.clone());
            continue;
        }

        let mut states: BTreeMap<u64, FieldAssignmentFragment> =
            if let Some(&assigned) = initializations.get(&field_id) {
                if assigned {
                    final_fragments
                        .values()
                        .map(|fragment| {
                            Ok((
                                fragment.id,
                                FieldAssignmentFragment {
                                    fragment_id: fragment.id,
                                    physical_rows: fragment_physical_rows(fragment)?,
                                    state: FieldAssignmentFragmentState::All,
                                },
                            ))
                        })
                        .collect::<Result<_>>()?
                } else {
                    BTreeMap::new()
                }
            } else {
                current
                    .load_field_assignment_root(source_field_id)
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

        match &transaction.operation {
            Operation::Append { .. } => {
                for fragment in manifest
                    .fragments
                    .iter()
                    .filter(|fragment| !current_fragment_ids.contains(&fragment.id))
                {
                    if fragment_supplies_field(fragment, field_id) {
                        states.insert(
                            fragment.id,
                            FieldAssignmentFragment {
                                fragment_id: fragment.id,
                                physical_rows: fragment_physical_rows(fragment)?,
                                state: FieldAssignmentFragmentState::All,
                            },
                        );
                    }
                }
            }
            Operation::Overwrite { .. } => {
                states.clear();
                for fragment in manifest.fragments.iter() {
                    if fragment_supplies_field(fragment, field_id) {
                        states.insert(
                            fragment.id,
                            FieldAssignmentFragment {
                                fragment_id: fragment.id,
                                physical_rows: fragment_physical_rows(fragment)?,
                                state: FieldAssignmentFragmentState::All,
                            },
                        );
                    }
                }
            }
            Operation::Rewrite { groups, .. } => {
                for group in groups {
                    for old in &group.old_fragments {
                        states.remove(&old.id);
                    }
                    for new_fragment in &group.new_fragments {
                        let final_fragment = fragment_by_path
                            .get(fragment_path(new_fragment)?)
                            .ok_or_else(|| {
                                Error::internal("Rewritten fragment is missing from final manifest")
                            })?;
                        if !overrides.contains_key(&(field_id, final_fragment.id)) {
                            return Err(Error::invalid_input(format!(
                                "Rewrite must supply assignment state for tracked field ID {}, new fragment '{}'",
                                field_id,
                                fragment_path(new_fragment)?
                            )));
                        }
                    }
                }
            }
            Operation::Update {
                new_fragments,
                update_mode: Some(UpdateMode::RewriteRows),
                ..
            } => {
                for new_fragment in new_fragments {
                    let final_fragment = fragment_by_path
                        .get(fragment_path(new_fragment)?)
                        .ok_or_else(|| {
                            Error::internal("Updated fragment is missing from final manifest")
                        })?;
                    if !overrides.contains_key(&(field_id, final_fragment.id)) {
                        return Err(Error::invalid_input(format!(
                            "Row rewrite must supply assignment state for tracked field ID {}, new fragment '{}'",
                            field_id,
                            fragment_path(new_fragment)?
                        )));
                    }
                }
            }
            Operation::Update { .. }
            | Operation::Delete { .. }
            | Operation::CreateIndex { .. }
            | Operation::DataReplacement { .. }
            | Operation::DataOverlay { .. }
            | Operation::Merge { .. }
            | Operation::Project { .. }
            | Operation::UpdateConfig { .. }
            | Operation::UpdateMemWalState { .. }
            | Operation::UpdateBases { .. }
            | Operation::ReserveFragments { .. }
            | Operation::Restore { .. }
            | Operation::Clone { .. } => {}
        }

        if !matches!(
            transaction.operation,
            Operation::Append { .. } | Operation::Overwrite { .. }
        ) && !initializations.contains_key(&field_id)
        {
            for fragment_id in final_fragment_ids.difference(&current_fragment_ids) {
                if !overrides.contains_key(&(field_id, *fragment_id)) {
                    return Err(Error::invalid_input(format!(
                        "Operation {} introduced fragment {} without assignment state for tracked field ID {}",
                        transaction.operation, fragment_id, field_id
                    )));
                }
            }
        }

        for ((override_field_id, fragment_id), value) in &overrides {
            if *override_field_id != field_id {
                continue;
            }
            let fragment = final_fragments.get(fragment_id).ok_or_else(|| {
                Error::internal(format!("Final fragment {} is missing", fragment_id))
            })?;
            let physical_rows = fragment_physical_rows(fragment)?;
            validate_fragment_value(field_id, *fragment_id, physical_rows, value)?;
            match value {
                FieldAssignmentFragmentValue::None => {
                    states.remove(fragment_id);
                }
                FieldAssignmentFragmentValue::All => {
                    states.insert(
                        *fragment_id,
                        FieldAssignmentFragment {
                            fragment_id: *fragment_id,
                            physical_rows,
                            state: FieldAssignmentFragmentState::All,
                        },
                    );
                }
                FieldAssignmentFragmentValue::Partial(bitmap) => {
                    let state = current
                        .write_field_assignment_bitmap(
                            write_store,
                            field_id,
                            *fragment_id,
                            physical_rows,
                            bitmap,
                        )
                        .await?;
                    states.insert(
                        *fragment_id,
                        FieldAssignmentFragment {
                            fragment_id: *fragment_id,
                            physical_rows,
                            state,
                        },
                    );
                }
            }
        }

        for ((change_field_id, fragment_id), change) in &pending_changes {
            if *change_field_id != field_id {
                continue;
            }
            let fragment = final_fragments.get(fragment_id).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Assignment change for field ID {} references fragment {} that is not in the new snapshot",
                    field_id, fragment_id
                ))
            })?;
            let physical_rows = fragment_physical_rows(fragment)?;
            if change
                .assigned
                .max()
                .into_iter()
                .chain(change.unassigned.max())
                .any(|offset| offset as u64 >= physical_rows)
            {
                return Err(Error::invalid_input(format!(
                    "Assignment change for field ID {}, fragment {} contains an out-of-bounds row offset",
                    field_id, fragment_id
                )));
            }
            let mut bitmap = materialize_fragment_bitmap(
                current,
                source_field_id,
                *fragment_id,
                states.get(fragment_id),
            )
            .await?;
            bitmap |= &change.assigned;
            bitmap -= &change.unassigned;
            if bitmap.is_empty() {
                states.remove(fragment_id);
            } else if bitmap.len() == physical_rows {
                states.insert(
                    *fragment_id,
                    FieldAssignmentFragment {
                        fragment_id: *fragment_id,
                        physical_rows,
                        state: FieldAssignmentFragmentState::All,
                    },
                );
            } else {
                let state = current
                    .write_field_assignment_bitmap(
                        write_store,
                        field_id,
                        *fragment_id,
                        physical_rows,
                        &bitmap,
                    )
                    .await?;
                states.insert(
                    *fragment_id,
                    FieldAssignmentFragment {
                        fragment_id: *fragment_id,
                        physical_rows,
                        state,
                    },
                );
            }
        }

        descriptors.push(
            current
                .write_field_assignment_root(
                    write_store,
                    field_id,
                    FieldAssignmentRoot {
                        fragments: states.into_values().collect(),
                    },
                )
                .await?,
        );
    }
    descriptors.sort_by_key(|state| state.field_id);
    let mut inline_manifest_bytes = 0usize;
    for descriptor in &mut descriptors {
        let Some(bytes) = descriptor.root.inline_bytes.as_ref() else {
            continue;
        };
        let next_inline_manifest_bytes = inline_manifest_bytes
            .checked_add(bytes.len())
            .ok_or_else(|| Error::internal("Inline assignment manifest size overflow"))?;
        if next_inline_manifest_bytes <= MAX_INLINE_FIELD_ASSIGNMENT_MANIFEST_BYTES {
            inline_manifest_bytes = next_inline_manifest_bytes;
        } else {
            descriptor.root.inline_bytes = None;
        }
    }
    manifest.field_assignment_states = descriptors;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        Array, ArrayRef, BooleanArray, Int32Array, Int64Array, RecordBatch, RecordBatchIterator,
        StructArray, UInt64Array,
    };
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use datafusion::functions_aggregate::count::count;
    use datafusion::logical_expr::{col, lit};
    use lance_core::{ROW_ID, utils::tempfile::TempStrDir};
    use lance_datafusion::udf::is_assigned;
    use lance_file::version::LanceFileVersion;
    use lance_file::writer::FileWriterOptions;
    use lance_index::{IndexType, scalar::ScalarIndexParams};
    use lance_io::utils::CachedFileSize;
    use lance_table::feature_flags::{FLAG_FIELD_ASSIGNMENT, FLAG_UNKNOWN};
    use lance_table::format::DataFile;
    use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};

    use super::*;
    use crate::dataset::cleanup::{CleanupPolicyBuilder, cleanup_old_versions};
    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::dataset::scanner::{AggregateExpr, ColumnOrdering};
    use crate::dataset::transaction::{DataOverlayGroup, Operation};
    use crate::dataset::write::merge_insert::{MergeInsertBuilder, WhenMatched, WhenNotMatched};
    use crate::dataset::{
        ColumnAlteration, FieldAssignment, FieldAssignmentAlteration, NewColumnTransform,
        UpdateBuilder, WriteDestination, WriteMode, WriteParams,
    };
    use crate::index::DatasetIndexExt;

    async fn tracked_dataset(directory: &TempStrDir, rows_per_file: usize) -> Result<Dataset> {
        tracked_dataset_with_stable_row_ids(directory, rows_per_file, false).await
    }

    async fn tracked_dataset_with_stable_row_ids(
        directory: &TempStrDir,
        rows_per_file: usize,
        enable_stable_row_ids: bool,
    ) -> Result<Dataset> {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..8))],
        )?;
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            directory,
            Some(WriteParams {
                max_rows_per_file: rows_per_file,
                enable_stable_row_ids,
                ..Default::default()
            }),
        )
        .await?;
        dataset
            .add_columns_with_assignment(
                NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![ArrowField::new(
                    "embedding",
                    DataType::Int32,
                    true,
                )]))),
                None,
                None,
                Some(FieldAssignment::Unassigned),
            )
            .await?;
        Ok(dataset)
    }

    async fn assigned_tracked_dataset(
        directory: &TempStrDir,
        rows_per_file: usize,
    ) -> Result<Dataset> {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("embedding", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..8)),
                Arc::new(Int32Array::from_iter_values(10..18)),
            ],
        )?;
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            directory,
            Some(WriteParams {
                max_rows_per_file: rows_per_file,
                ..Default::default()
            }),
        )
        .await?;
        dataset
            .alter_columns_with_assignment(
                &[],
                &[FieldAssignmentAlteration::new(
                    "embedding",
                    FieldAssignment::Assigned,
                )],
            )
            .await?;
        Ok(dataset)
    }

    async fn filtered_ids(dataset: &Dataset, filter: &str) -> Result<(Vec<i32>, String)> {
        let mut scanner = dataset.scan();
        scanner.project(&["id"])?;
        scanner.filter(filter)?;
        let plan = scanner.explain_plan(false).await?;
        let batch = scanner.try_into_batch().await?;
        let ids = batch
            .column_by_name("id")
            .expect("id projection must exist")
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("id must remain Int32");
        let mut ids = ids.values().to_vec();
        ids.sort_unstable();
        Ok((ids, plan))
    }

    async fn assignment_rows(dataset: &Dataset, field: &str) -> Result<Vec<(i32, bool)>> {
        let mut scanner = dataset.scan();
        scanner.project_with_transform(&[
            ("id", "id"),
            ("assigned", &format!("is_assigned({field})")),
        ])?;
        let batch = scanner.try_into_batch().await?;
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let assigned = batch
            .column_by_name("assigned")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        let mut rows = (0..batch.num_rows())
            .map(|index| (ids.value(index), assigned.value(index)))
            .collect::<Vec<_>>();
        rows.sort_by_key(|row| row.0);
        Ok(rows)
    }

    async fn commit_assignment_overlay(
        dataset: Dataset,
        fragment_id: u64,
        field_id: i32,
        offsets: RoaringBitmap,
        values: ArrayRef,
    ) -> Result<Dataset> {
        let read_version = dataset.version().version;
        let overlay_schema = dataset.schema().project_by_ids(&[field_id], true);
        let filename = format!("assignment-overlay-{}.lance", Uuid::new_v4());
        let path = dataset.data_dir().join(filename.as_str());
        let object_writer = dataset.object_store.create(&path).await?;
        let file_version = LanceFileVersion::Stable.resolve();
        let mut writer = lance_file::versions::create_writer(
            file_version,
            object_writer,
            overlay_schema,
            FileWriterOptions::default(),
        )?;
        writer.write_column(0, values).await?;
        let summary = writer.finish().await?;
        let mut data_file = DataFile::new_unstarted(filename, file_version);
        data_file.fields = writer
            .field_id_to_column_indices()
            .iter()
            .map(|(field_id, _)| *field_id as i32)
            .collect::<Vec<_>>()
            .into();
        data_file.column_indices = writer
            .field_id_to_column_indices()
            .iter()
            .map(|(_, column_index)| *column_index as i32)
            .collect::<Vec<_>>()
            .into();
        data_file.file_size_bytes = CachedFileSize::new(summary.size_bytes);

        Dataset::commit(
            WriteDestination::Dataset(Arc::new(dataset)),
            Operation::DataOverlay {
                groups: vec![DataOverlayGroup {
                    fragment_id,
                    overlays: vec![DataOverlayFile {
                        data_file,
                        coverage: OverlayCoverage::dense(offsets),
                        committed_version: 0,
                    }],
                }],
            },
            Some(read_version),
            None,
            None,
            Arc::new(Default::default()),
            false,
        )
        .await
    }

    #[tokio::test]
    async fn test_initial_assignment_expression_is_non_null_and_composable() -> Result<()> {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..6))],
        )?;
        let directory = TempStrDir::default();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &directory,
            Some(WriteParams {
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await?;

        let assigned_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "embedding",
            DataType::Int32,
            true,
        )]));
        dataset
            .add_columns_with_assignment(
                NewColumnTransform::AllNulls(assigned_schema),
                None,
                None,
                Some(FieldAssignment::Unassigned),
            )
            .await?;

        let mut scanner = dataset.scan();
        scanner.project_with_transform(&[
            ("assigned", "is_assigned(embedding)"),
            (
                "combined",
                "NOT (is_assigned(embedding) OR id < 0) AND (id >= 0 OR is_assigned(embedding))",
            ),
        ])?;
        let batch = scanner.try_into_batch().await?;
        let assigned = batch
            .column_by_name("assigned")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        let combined = batch
            .column_by_name("combined")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert_eq!(assigned.null_count(), 0);
        assert_eq!(assigned.iter().collect::<Vec<_>>(), vec![Some(false); 6]);
        assert_eq!(combined.iter().collect::<Vec<_>>(), vec![Some(true); 6]);

        let mut scanner = dataset.scan();
        scanner.filter("NOT is_assigned(embedding)")?;
        assert_eq!(scanner.count_rows().await?, 6);

        Ok(())
    }

    #[tokio::test]
    async fn test_assignment_filter_uses_exact_row_selection_pushdown() -> Result<()> {
        for enable_stable_row_ids in [false, true] {
            let directory = TempStrDir::default();
            let dataset =
                tracked_dataset_with_stable_row_ids(&directory, 2, enable_stable_row_ids).await?;
            let result = UpdateBuilder::new(Arc::new(dataset))
                .update_where("id IN (1, 3, 5, 7)")?
                .set("embedding", "id * 10")?
                .build()?
                .execute()
                .await?;
            let mut dataset = result.new_dataset.as_ref().clone();
            dataset.delete("id = 3").await?;

            let (ids, plan) = filtered_ids(&dataset, "is_assigned(embedding)").await?;
            assert_eq!(ids, vec![1, 5, 7]);
            assert!(
                plan.contains("ScalarIndexQuery")
                    && plan.contains("FieldAssignment")
                    && plan.contains("exact"),
                "assignment filter did not use exact row selection:\n{plan}"
            );

            let (ids, plan) = filtered_ids(&dataset, "NOT is_assigned(embedding)").await?;
            assert_eq!(ids, vec![0, 2, 4, 6]);
            assert!(
                plan.contains("ScalarIndexQuery"),
                "unexpected plan:\n{plan}"
            );

            let (ids, plan) = filtered_ids(&dataset, "is_assigned(embedding) AND id >= 5").await?;
            assert_eq!(ids, vec![5, 7]);
            assert!(
                plan.contains("ScalarIndexQuery"),
                "unexpected plan:\n{plan}"
            );

            // OR with an unindexed predicate must retain the correctness
            // fallback because the other side may select any physical row.
            let (ids, plan) = filtered_ids(&dataset, "is_assigned(embedding) OR id = 0").await?;
            assert_eq!(ids, vec![0, 1, 5, 7]);
            assert!(
                !plan.contains("ScalarIndexQuery"),
                "partial OR was incorrectly pushed down:\n{plan}"
            );

            dataset
                .create_index(
                    &["id"],
                    IndexType::Bitmap,
                    None,
                    &ScalarIndexParams::default(),
                    true,
                )
                .await?;
            let (ids, plan) = filtered_ids(&dataset, "is_assigned(embedding) OR id = 0").await?;
            assert_eq!(ids, vec![0, 1, 5, 7]);
            assert!(
                plan.contains("ScalarIndexQuery")
                    && plan.contains("FieldAssignment")
                    && plan.contains("Bitmap"),
                "assignment and scalar index were not combined:\n{plan}"
            );

            compact_files(&mut dataset, CompactionOptions::default(), None).await?;
            let (ids, plan) = filtered_ids(&dataset, "is_assigned(embedding)").await?;
            assert_eq!(ids, vec![1, 5, 7]);
            assert!(
                plan.contains("ScalarIndexQuery"),
                "unexpected plan:\n{plan}"
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_nested_field_assignment_and_parent_write_conflict() -> Result<()> {
        let value_field = Arc::new(ArrowField::new("value", DataType::Int32, true));
        let payload = StructArray::from(vec![(
            value_field.clone(),
            Arc::new(Int32Array::from(vec![Some(10), None])) as ArrayRef,
        )]);
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("payload", payload.data_type().clone(), false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..2)),
                Arc::new(payload),
            ],
        )?;
        let directory = TempStrDir::default();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &directory,
            None,
        )
        .await?;
        dataset
            .alter_columns_with_assignment(
                &[],
                &[FieldAssignmentAlteration::new(
                    "payload.value",
                    FieldAssignment::Unassigned,
                )],
            )
            .await?;

        let appended_payload = StructArray::from(vec![(
            value_field.clone(),
            Arc::new(Int32Array::from(vec![None])) as ArrayRef,
        )]);
        let appended_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("payload", appended_payload.data_type().clone(), false),
        ]));
        let appended = RecordBatch::try_new(
            appended_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![2])),
                Arc::new(appended_payload),
            ],
        )?;
        dataset = Dataset::write(
            RecordBatchIterator::new([Ok(appended)], appended_schema),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await?;
        assert_eq!(
            assignment_rows(&dataset, "payload.value").await?,
            vec![(0, false), (1, false), (2, true)]
        );

        let error = UpdateBuilder::new(Arc::new(dataset))
            .set("payload", "payload")?
            .invalidate_fields(["payload.value"])?
            .build()
            .unwrap_err();
        assert!(error.to_string().contains("both written and invalidated"));

        Ok(())
    }

    #[tokio::test]
    async fn test_append_distinguishes_explicit_null_from_omitted_field() -> Result<()> {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..2))],
        )?;
        let directory = TempStrDir::default();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &directory,
            None,
        )
        .await?;
        dataset
            .add_columns_with_assignment(
                NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![ArrowField::new(
                    "embedding",
                    DataType::Int32,
                    true,
                )]))),
                None,
                None,
                Some(FieldAssignment::Unassigned),
            )
            .await?;

        let explicit_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("embedding", DataType::Int32, true),
        ]));
        let explicit_nulls = RecordBatch::try_new(
            explicit_schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(2..4)),
                Arc::new(Int32Array::from(vec![None, None])),
            ],
        )?;
        dataset = Dataset::write(
            RecordBatchIterator::new([Ok(explicit_nulls)], explicit_schema),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await?;

        let omitted_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let omitted = RecordBatch::try_new(
            omitted_schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(4..6))],
        )?;
        dataset = Dataset::write(
            RecordBatchIterator::new([Ok(omitted)], omitted_schema),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await?;

        let mut scanner = dataset.scan();
        scanner.project_with_transform(&[
            ("id", "id"),
            ("embedding", "embedding"),
            ("assigned", "is_assigned(embedding)"),
        ])?;
        let batch = scanner.try_into_batch().await?;
        let embedding = batch
            .column_by_name("embedding")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let assigned = batch
            .column_by_name("assigned")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert_eq!(embedding.null_count(), 6);
        assert_eq!(
            assigned.iter().collect::<Vec<_>>(),
            vec![
                Some(false),
                Some(false),
                Some(true),
                Some(true),
                Some(false),
                Some(false),
            ]
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_update_invalidation_and_rename_preserve_stored_values() -> Result<()> {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("text", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..4)),
                Arc::new(Int32Array::from_iter_values(10..14)),
            ],
        )?;
        let directory = TempStrDir::default();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &directory,
            Some(WriteParams {
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await?;
        dataset
            .add_columns_with_assignment(
                NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![ArrowField::new(
                    "embedding",
                    DataType::Int32,
                    true,
                )]))),
                None,
                None,
                Some(FieldAssignment::Unassigned),
            )
            .await?;

        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id < 2")?
            .set("embedding", "id * 10")?
            .build()?
            .execute()
            .await?;
        dataset = result.new_dataset.as_ref().clone();
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 1")?
            .set("text", "99")?
            .invalidate_fields(["embedding"])?
            .build()?
            .execute()
            .await?;
        dataset = result.new_dataset.as_ref().clone();

        let tracked_field_id = dataset.schema().field("embedding").unwrap().id;
        dataset
            .alter_columns(&[
                ColumnAlteration::new("embedding".to_string()).rename("vector".to_string())
            ])
            .await?;
        assert_eq!(
            dataset.schema().field("vector").unwrap().id,
            tracked_field_id
        );

        let mut scanner = dataset.scan();
        scanner.project_with_transform(&[
            ("id", "id"),
            ("vector", "vector"),
            ("assigned", "is_assigned(vector)"),
        ])?;
        let batch = scanner.try_into_batch().await?;
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let vectors = batch
            .column_by_name("vector")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let assigned = batch
            .column_by_name("assigned")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        let mut rows = (0..batch.num_rows())
            .map(|index| {
                (
                    ids.value(index),
                    (!vectors.is_null(index)).then(|| vectors.value(index)),
                    assigned.value(index),
                )
            })
            .collect::<Vec<_>>();
        rows.sort_by_key(|row| row.0);
        assert_eq!(
            rows,
            vec![
                (0, Some(0), true),
                (1, Some(10), false),
                (2, None, false),
                (3, None, false),
            ]
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_invalidation_only_update_is_metadata_only() -> Result<()> {
        let directory = TempStrDir::default();
        let dataset = tracked_dataset(&directory, 2).await?;
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id < 2")?
            .set("embedding", "id * 10")?
            .build()?
            .execute()
            .await?;
        let dataset = result.new_dataset;
        let field_id = dataset.schema().field("embedding").unwrap().id;
        let fragments_before = dataset.manifest.fragments.clone();
        let root_before = dataset
            .field_assignment_state(field_id)
            .unwrap()
            .root
            .clone();

        let result = UpdateBuilder::new(dataset)
            .update_where("id = 0")?
            .invalidate_fields(["embedding"])?
            .build()?
            .execute()
            .await?;
        assert_eq!(result.rows_updated, 1);
        assert_eq!(result.new_dataset.manifest.fragments, fragments_before);
        assert_ne!(
            result
                .new_dataset
                .field_assignment_state(field_id)
                .unwrap()
                .root,
            root_before
        );
        assert_eq!(
            assignment_rows(&result.new_dataset, "embedding").await?,
            vec![
                (0, false),
                (1, true),
                (2, false),
                (3, false),
                (4, false),
                (5, false),
                (6, false),
                (7, false),
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_assignment_composes_in_ordering_and_aggregation() -> Result<()> {
        let directory = TempStrDir::default();
        let dataset = tracked_dataset(&directory, 2).await?;
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id IN (1, 3, 5, 7)")?
            .set("embedding", "id * 10")?
            .build()?
            .execute()
            .await?;
        let dataset = result.new_dataset;

        let mut invalid_ordering = dataset.scan();
        assert!(
            invalid_ordering
                .order_by(Some(vec![ColumnOrdering::asc_nulls_first(
                    "id + 1".to_string(),
                )]))
                .is_err()
        );

        let mut scanner = dataset.scan();
        scanner.project_with_transform(&[("id", "id"), ("assigned", "is_assigned(embedding)")])?;
        scanner.order_by(Some(vec![ColumnOrdering::asc_nulls_first(
            "is_assigned(embedding)".to_string(),
        )]))?;
        let batch = scanner.try_into_batch().await?;
        let assigned = batch
            .column_by_name("assigned")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert_eq!(
            assigned.iter().collect::<Vec<_>>(),
            vec![Some(false); 4]
                .into_iter()
                .chain(vec![Some(true); 4])
                .collect::<Vec<_>>()
        );

        let mut scanner = dataset.scan();
        scanner.aggregate(AggregateExpr::datafusion(
            vec![is_assigned(col("embedding"))],
            vec![count(lit(1)).alias("rows")],
        ))?;
        let batch = scanner.try_into_batch().await?;
        let assigned = batch
            .column(0)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        let counts = batch
            .column_by_name("rows")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let mut groups = (0..batch.num_rows())
            .map(|index| (assigned.value(index), counts.value(index)))
            .collect::<Vec<_>>();
        groups.sort_by_key(|group| group.0);
        assert_eq!(groups, vec![(false, 4), (true, 4)]);

        Ok(())
    }

    #[tokio::test]
    async fn test_delete_and_compaction_remap_assignment() -> Result<()> {
        let directory = TempStrDir::default();
        let dataset = tracked_dataset(&directory, 2).await?;
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id IN (1, 2, 5, 6)")?
            .set("embedding", "id * 10")?
            .build()?
            .execute()
            .await?;
        let mut dataset = result.new_dataset.as_ref().clone();
        dataset.delete("id IN (2, 4)").await?;
        let before = assignment_rows(&dataset, "embedding").await?;

        compact_files(&mut dataset, CompactionOptions::default(), None).await?;

        let after = assignment_rows(&dataset, "embedding").await?;
        assert_eq!(after, before);
        assert_eq!(
            after,
            vec![
                (0, false),
                (1, true),
                (3, false),
                (5, true),
                (6, true),
                (7, false),
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_time_travel_restore_and_cast_transfer_assignment() -> Result<()> {
        let directory = TempStrDir::default();
        let dataset = tracked_dataset(&directory, 2).await?;
        let unassigned_version = dataset.version().version;
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id < 3")?
            .set("embedding", "id * 10")?
            .build()?
            .execute()
            .await?;
        let mut dataset = result.new_dataset.as_ref().clone();
        assert_eq!(
            assignment_rows(&dataset, "embedding").await?,
            vec![
                (0, true),
                (1, true),
                (2, true),
                (3, false),
                (4, false),
                (5, false),
                (6, false),
                (7, false),
            ]
        );

        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 7")?
            .set("embedding", "70")?
            .build()?
            .execute()
            .await?;
        dataset = result.new_dataset.as_ref().clone();
        let old_field_id = dataset.schema().field("embedding").unwrap().id;
        let old_root = dataset
            .field_assignment_state(old_field_id)
            .unwrap()
            .root
            .clone();
        let expected = assignment_rows(&dataset, "embedding").await?;
        dataset
            .alter_columns(&[
                ColumnAlteration::new("embedding".to_string()).cast_to(DataType::Int64)
            ])
            .await?;
        let new_field_id = dataset.schema().field("embedding").unwrap().id;
        assert_ne!(new_field_id, old_field_id);
        let new_root = &dataset.field_assignment_state(new_field_id).unwrap().root;
        assert_ne!(new_root, &old_root);
        assert!(
            new_root
                .path
                .starts_with(&format!("_field_assignments/roots/{new_field_id}/"))
        );
        assert_eq!(assignment_rows(&dataset, "embedding").await?, expected);

        let mut historical = dataset.checkout_version(unassigned_version).await?;
        assert!(
            assignment_rows(&historical, "embedding")
                .await?
                .iter()
                .all(|(_, assigned)| !assigned)
        );
        historical.restore().await?;
        assert!(
            assignment_rows(&historical, "embedding")
                .await?
                .iter()
                .all(|(_, assigned)| !assigned)
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_assignment_state_is_loaded_only_when_referenced() -> Result<()> {
        let directory = TempStrDir::default();
        let dataset = assigned_tracked_dataset(&directory, 2).await?;
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id IN (1, 3, 5, 7)")?
            .invalidate_fields(["embedding"])?
            .build()?
            .execute()
            .await?;
        let dataset = result.new_dataset.as_ref();
        let field_id = dataset.schema().field("embedding").unwrap().id;
        let root = dataset
            .load_field_assignment_root(field_id)
            .await?
            .expect("tracked field must have a root");
        let bitmap = root
            .fragments
            .iter()
            .find_map(|fragment| match &fragment.state {
                FieldAssignmentFragmentState::Partial(bitmap) => Some(bitmap.clone()),
                _ => None,
            })
            .expect("dense partial assignment must use an external bitmap");
        let store = dataset.object_store(bitmap.base_id).await?;
        store
            .delete(&dataset.field_assignment_path(&bitmap)?)
            .await?;

        let mut ordinary = dataset.scan();
        ordinary.project(&["id"])?;
        assert_eq!(ordinary.try_into_batch().await?.num_rows(), 8);

        let mut assignment = dataset.scan();
        assignment.project_with_transform(&[("assigned", "is_assigned(embedding)")])?;
        let error = assignment.try_into_batch().await.unwrap_err();
        assert!(
            error.to_string().contains(&bitmap.path),
            "unexpected assignment read error: {error}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_assignment_planning_inlines_sparse_root_and_caches_external_fallback()
    -> Result<()> {
        use lance_io::{assert_io_eq, assert_io_gt};

        let directory = TempStrDir::default();
        let dataset = assigned_tracked_dataset(&directory, 4).await?;
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id % 4 != 0")?
            .invalidate_fields(["embedding"])?
            .build()?
            .execute()
            .await?;
        let dataset = result.new_dataset.as_ref();
        let field_id = dataset.schema().field("embedding").unwrap().id;
        assert!(
            dataset
                .field_assignment_state(field_id)
                .unwrap()
                .root
                .inline_bytes
                .is_some()
        );
        let root = dataset.load_field_assignment_root(field_id).await?.unwrap();
        assert!(root.fragments.iter().all(|fragment| matches!(
            fragment.state,
            FieldAssignmentFragmentState::InlinePartial(_)
        )));

        let _ = dataset.object_store.io_stats_incremental();
        let mut first = dataset.scan();
        first.project_with_transform(&[("assigned", "is_assigned(embedding)")])?;
        first.explain_plan(false).await?;
        let first_io = dataset.object_store.io_stats_incremental();
        assert_io_eq!(first_io, read_iops, 0);
        assert_io_eq!(first_io, read_bytes, 0);

        let mut external_root = dataset.clone();
        Arc::make_mut(&mut external_root.manifest)
            .field_assignment_states
            .iter_mut()
            .find(|state| state.field_id == field_id)
            .unwrap()
            .root
            .inline_bytes = None;
        let _ = external_root.object_store.io_stats_incremental();
        let mut second = external_root.scan();
        second.project_with_transform(&[("assigned", "is_assigned(embedding)")])?;
        second.explain_plan(false).await?;
        let cold_external_io = external_root.object_store.io_stats_incremental();
        assert_io_gt!(cold_external_io, read_iops, 0);

        let mut third = external_root.scan();
        third.project_with_transform(&[("assigned", "is_assigned(embedding)")])?;
        third.explain_plan(false).await?;
        let warm_io = external_root.object_store.io_stats_incremental();
        assert_io_eq!(warm_io, read_iops, 0);
        assert_io_eq!(warm_io, read_bytes, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_invalid_assignment_expressions_fail_during_planning() -> Result<()> {
        let directory = TempStrDir::default();
        let dataset = tracked_dataset(&directory, 2).await?;

        for expression in ["is_assigned(missing)", "is_assigned(id)", "is_assigned(42)"] {
            let mut scanner = dataset.scan();
            let error = match scanner.project_with_transform(&[("assigned", expression)]) {
                Ok(_) => scanner.try_into_batch().await.unwrap_err(),
                Err(error) => error,
            };
            assert!(
                matches!(error, Error::InvalidInput { .. }),
                "unexpected error for {expression}: {error}"
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent_appends_reconcile_assignment_from_current_head() -> Result<()> {
        let directory = TempStrDir::default();
        let dataset = Arc::new(tracked_dataset(&directory, 2).await?);

        let explicit_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("embedding", DataType::Int32, true),
        ]));
        let explicit = RecordBatch::try_new(
            explicit_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![8])),
                Arc::new(Int32Array::from(vec![None])),
            ],
        )?;
        let omitted_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let omitted = RecordBatch::try_new(
            omitted_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![9]))],
        )?;

        let explicit_write = Dataset::write(
            RecordBatchIterator::new([Ok(explicit)], explicit_schema),
            dataset.clone(),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        );
        let omitted_write = Dataset::write(
            RecordBatchIterator::new([Ok(omitted)], omitted_schema),
            dataset,
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        );
        let (explicit_result, omitted_result) = tokio::join!(explicit_write, omitted_write);
        explicit_result?;
        omitted_result?;

        let dataset = Dataset::open(&directory).await?;
        let rows = assignment_rows(&dataset, "embedding").await?;
        assert_eq!(rows.len(), 10);
        assert_eq!(rows[8], (8, true));
        assert_eq!(rows[9], (9, false));
        Ok(())
    }

    #[tokio::test]
    async fn test_update_only_merge_insert_on_row_id_assigns_only_matched_rows() -> Result<()> {
        let directory = TempStrDir::default();
        let dataset = tracked_dataset(&directory, 2).await?;
        let mut scanner = dataset.scan();
        scanner.with_row_id();
        let batch = scanner.try_into_batch().await?;
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let row_ids = batch
            .column_by_name(ROW_ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let selected = (0..batch.num_rows())
            .filter(|index| matches!(ids.value(*index), 1 | 6))
            .map(|index| row_ids.value(index))
            .collect::<Vec<_>>();

        let source_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new(ROW_ID, DataType::UInt64, false),
            ArrowField::new("embedding", DataType::Int32, true),
        ]));
        let source = RecordBatch::try_new(
            source_schema.clone(),
            vec![
                Arc::new(UInt64Array::from(selected.clone())),
                Arc::new(Int32Array::from(vec![None, Some(60)])),
            ],
        )?;
        let mut builder = MergeInsertBuilder::try_new(Arc::new(dataset), vec![ROW_ID.to_string()])?;
        builder
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing);
        let (dataset, stats) = builder
            .try_build()?
            .execute_reader(Box::new(RecordBatchIterator::new(
                [Ok(source)],
                source_schema,
            )))
            .await?;
        assert_eq!(stats.num_updated_rows, 2);
        assert_eq!(stats.num_inserted_rows, 0);
        assert_eq!(
            assignment_rows(&dataset, "embedding").await?,
            vec![
                (0, false),
                (1, true),
                (2, false),
                (3, false),
                (4, false),
                (5, false),
                (6, true),
                (7, false),
            ]
        );

        let invalidation_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new(ROW_ID, DataType::UInt64, false),
            ArrowField::new("id", DataType::Int32, false),
        ]));
        let invalidation = RecordBatch::try_new(
            invalidation_schema.clone(),
            vec![
                Arc::new(UInt64Array::from(selected)),
                Arc::new(Int32Array::from(vec![1, 6])),
            ],
        )?;
        let mut builder = MergeInsertBuilder::try_new(dataset, vec![ROW_ID.to_string()])?;
        builder
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .invalidate_fields(["embedding"])?;
        let (dataset, stats) = builder
            .try_build()?
            .execute_reader(Box::new(RecordBatchIterator::new(
                [Ok(invalidation)],
                invalidation_schema,
            )))
            .await?;
        assert_eq!(stats.num_updated_rows, 2);
        assert!(
            assignment_rows(&dataset, "embedding")
                .await?
                .iter()
                .all(|(_, assigned)| !assigned)
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_data_replacement_creates_assignment() -> Result<()> {
        let directory = TempStrDir::default();
        let dataset = tracked_dataset(&directory, 8).await?;
        let replacement_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "embedding",
            DataType::Int32,
            true,
        )]));
        let replacement = RecordBatch::try_new(
            replacement_schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(100..108))],
        )?;
        let path = "assignment-replacement.lance";
        let writer = dataset
            .object_store
            .create(&dataset.data_dir().join(path))
            .await?;
        let mut writer = lance_file::versions::v2_1::create_writer(
            writer,
            replacement_schema.as_ref().try_into()?,
            Default::default(),
        )?;
        writer.write_batch(&replacement).await?;
        writer.finish().await?;
        let data_file = dataset.create_data_file(path, None).await?;
        let read_version = dataset.version().version;

        let mut dataset = Dataset::commit(
            WriteDestination::Dataset(Arc::new(dataset)),
            Operation::DataReplacement {
                replacements: vec![crate::dataset::transaction::DataReplacementGroup(
                    0, data_file,
                )],
            },
            Some(read_version),
            None,
            None,
            Arc::new(Default::default()),
            false,
        )
        .await?;

        let expected = assignment_rows(&dataset, "embedding").await?;
        assert!(expected.iter().all(|(_, assigned)| *assigned));

        let merge_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("label", DataType::Int32, false),
        ]));
        let merge_batch = RecordBatch::try_new(
            merge_schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..8)),
                Arc::new(Int32Array::from_iter_values(200..208)),
            ],
        )?;
        dataset
            .merge(
                RecordBatchIterator::new([Ok(merge_batch)], merge_schema),
                "id",
                "id",
            )
            .await?;
        assert_eq!(assignment_rows(&dataset, "embedding").await?, expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_overwrite_and_drop_assignment_tracking() -> Result<()> {
        let directory = TempStrDir::default();
        let dataset = tracked_dataset(&directory, 2).await?;
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("embedding", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![20, 21, 22])),
                Arc::new(Int32Array::from(vec![None, Some(210), None])),
            ],
        )?;
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await?;
        assert_eq!(
            assignment_rows(&dataset, "embedding").await?,
            vec![(20, true), (21, true), (22, true)]
        );

        dataset.drop_columns(&["embedding"]).await?;
        assert!(dataset.manifest.field_assignment_states.is_empty());
        let mut scanner = dataset.scan();
        let error = match scanner.project_with_transform(&[("assigned", "is_assigned(embedding)")])
        {
            Ok(_) => scanner.try_into_batch().await.unwrap_err(),
            Err(error) => error,
        };
        assert!(matches!(error, Error::InvalidInput { .. }));

        Ok(())
    }

    #[tokio::test]
    async fn test_data_overlay_coverage_creates_assignment() -> Result<()> {
        let directory = TempStrDir::default();
        let dataset = tracked_dataset(&directory, 8).await?;
        let field_id = dataset.schema().field("embedding").unwrap().id;
        let dataset = commit_assignment_overlay(
            dataset,
            0,
            field_id,
            RoaringBitmap::from_iter([1, 4]),
            Arc::new(Int32Array::from(vec![None, Some(40)])),
        )
        .await?;
        let root = dataset
            .load_field_assignment_root(field_id)
            .await?
            .expect("tracked field must have a root");
        assert!(matches!(
            root.fragments.as_slice(),
            [FieldAssignmentFragment {
                state: FieldAssignmentFragmentState::InlinePartial(_),
                ..
            }]
        ));

        assert_eq!(
            assignment_rows(&dataset, "embedding").await?,
            vec![
                (0, false),
                (1, true),
                (2, false),
                (3, false),
                (4, true),
                (5, false),
                (6, false),
                (7, false),
            ]
        );
        let batch = dataset.scan().try_into_batch().await?;
        let embedding = batch
            .column_by_name("embedding")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert!(embedding.is_null(1));
        assert_eq!(embedding.value(4), 40);

        Ok(())
    }

    #[tokio::test]
    async fn test_assignment_clone_storage_and_writer_compatibility() -> Result<()> {
        let source_directory = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("embedding", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..8)),
                Arc::new(Int32Array::from_iter_values(10..18)),
            ],
        )?;
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &source_directory,
            None,
        )
        .await?;
        dataset
            .alter_columns_with_assignment(
                &[],
                &[FieldAssignmentAlteration::new(
                    "embedding",
                    FieldAssignment::Assigned,
                )],
            )
            .await?;
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id IN (1, 6)")?
            .invalidate_fields(["embedding"])?
            .build()?
            .execute()
            .await?;
        let mut dataset = result.new_dataset.as_ref().clone();
        let expected = assignment_rows(&dataset, "embedding").await?;
        assert_eq!(expected.iter().filter(|(_, assigned)| !assigned).count(), 2);
        assert_ne!(
            dataset.manifest.reader_feature_flags & FLAG_FIELD_ASSIGNMENT,
            0
        );
        assert_ne!(
            dataset.manifest.writer_feature_flags & FLAG_FIELD_ASSIGNMENT,
            0
        );

        let mut unsupported_writer = dataset.clone();
        Arc::make_mut(&mut unsupported_writer.manifest).writer_feature_flags |= FLAG_UNKNOWN;
        let error = unsupported_writer
            .update_config([("unsupported", "writer")])
            .await
            .unwrap_err();
        assert!(error.to_string().contains("cannot be written"));

        let shallow_directory = TempStrDir::default();
        let shallow = dataset
            .shallow_clone(shallow_directory.as_ref(), dataset.version().version, None)
            .await?;
        let field_id = shallow.schema().field("embedding").unwrap().id;
        assert!(
            shallow
                .field_assignment_state(field_id)
                .unwrap()
                .root
                .base_id
                .is_some()
        );
        assert_eq!(assignment_rows(&shallow, "embedding").await?, expected);

        let deep_directory = TempStrDir::default();
        let deep = dataset
            .deep_clone(deep_directory.as_ref(), dataset.version().version, None)
            .await?;
        let deep_descriptor = deep.field_assignment_state(field_id).unwrap();
        assert_eq!(deep_descriptor.root.base_id, None);
        let deep_root = deep.load_field_assignment_root(field_id).await?.unwrap();
        assert!(
            deep_root
                .fragments
                .iter()
                .any(|fragment| matches!(fragment.state, FieldAssignmentFragmentState::Partial(_)))
        );
        assert!(
            deep_root
                .fragments
                .iter()
                .all(|fragment| match &fragment.state {
                    FieldAssignmentFragmentState::All => true,
                    FieldAssignmentFragmentState::Partial(bitmap) => bitmap.base_id.is_none(),
                    FieldAssignmentFragmentState::InlinePartial(_) => true,
                })
        );
        deep.validate().await?;
        let mut corrupted_inline_root = deep.clone();
        let inline_bytes = Arc::make_mut(&mut corrupted_inline_root.manifest)
            .field_assignment_states
            .iter_mut()
            .find(|state| state.field_id == field_id)
            .and_then(|state| state.root.inline_bytes.as_mut())
            .expect("small deep-cloned root must retain an inline copy");
        inline_bytes[0] ^= 0xff;
        let error = corrupted_inline_root.validate().await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("does not match its immutable object"),
            "unexpected inline root validation error: {error}"
        );

        // Prove the deep clone owns its assignment objects instead of falling
        // back to the source dataset after reopening with a fresh session.
        let source_descriptor = dataset.field_assignment_state(field_id).unwrap().clone();
        let source_root = dataset.load_field_assignment_root(field_id).await?.unwrap();
        for fragment in &source_root.fragments {
            if let FieldAssignmentFragmentState::Partial(bitmap) = &fragment.state {
                let store = dataset.object_store(bitmap.base_id).await?;
                store
                    .delete(&dataset.field_assignment_path(bitmap)?)
                    .await?;
            }
        }
        let store = dataset.object_store(source_descriptor.root.base_id).await?;
        store
            .delete(&dataset.field_assignment_path(&source_descriptor.root)?)
            .await?;
        let reopened = Dataset::open(deep_directory.as_ref()).await?;
        assert_eq!(assignment_rows(&reopened, "embedding").await?, expected);

        Ok(())
    }

    #[tokio::test]
    async fn test_cleanup_retains_branch_assignment_objects() -> Result<()> {
        let directory = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("embedding", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..8)),
                Arc::new(Int32Array::from_iter_values(10..18)),
            ],
        )?;
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &directory,
            None,
        )
        .await?;
        dataset
            .alter_columns_with_assignment(
                &[],
                &[FieldAssignmentAlteration::new(
                    "embedding",
                    FieldAssignment::Assigned,
                )],
            )
            .await?;
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 1")?
            .invalidate_fields(["embedding"])?
            .build()?
            .execute()
            .await?;
        dataset = result.new_dataset.as_ref().clone();
        let field_id = dataset.schema().field("embedding").unwrap().id;
        let branch_descriptor = dataset.field_assignment_state(field_id).unwrap().clone();
        let branch_root = dataset.load_field_assignment_root(field_id).await?.unwrap();
        let branch_paths = std::iter::once(branch_descriptor.root.clone())
            .chain(branch_root.fragments.iter().filter_map(|fragment| {
                if let FieldAssignmentFragmentState::Partial(bitmap) = &fragment.state {
                    Some(bitmap.clone())
                } else {
                    None
                }
            }))
            .collect::<Vec<_>>();
        let expected_branch = assignment_rows(&dataset, "embedding").await?;
        dataset
            .create_branch("assignment-cleanup", dataset.version().version, None)
            .await?;

        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 2")?
            .invalidate_fields(["embedding"])?
            .build()?
            .execute()
            .await?;
        dataset = result.new_dataset.as_ref().clone();
        let policy = CleanupPolicyBuilder::default()
            .retain_n_versions(&dataset, 1)
            .await?
            .delete_unverified(true)
            .build();
        cleanup_old_versions(&dataset, policy).await?;
        for file in &branch_paths {
            let store = dataset.object_store(file.base_id).await?;
            assert!(store.exists(&dataset.field_assignment_path(file)?).await?);
        }
        let branch = dataset.checkout_branch("assignment-cleanup").await?;
        assert_eq!(
            assignment_rows(&branch, "embedding").await?,
            expected_branch
        );

        dataset.force_delete_branch("assignment-cleanup").await?;
        dataset = Dataset::open(directory.as_ref()).await?;
        assert!(dataset.branches().list().await?.is_empty());
        assert_ne!(
            dataset.field_assignment_state(field_id).unwrap().root.path,
            branch_descriptor.root.path
        );
        let policy = CleanupPolicyBuilder::default()
            .retain_n_versions(&dataset, 1)
            .await?
            .delete_unverified(true)
            .build();
        let explanation = dataset.cleanup(policy.clone()).explain().await?;
        assert!(
            explanation.candidate_files.iter().any(|candidate| candidate
                .path
                .ends_with(branch_descriptor.root.path.as_str())),
            "cleanup did not classify old assignment root '{}' as a candidate: {:?}",
            branch_descriptor.root.path,
            explanation.candidate_files
        );
        cleanup_old_versions(&dataset, policy).await?;
        for file in &branch_paths {
            let store = dataset.object_store(file.base_id).await?;
            assert!(
                !store.exists(&dataset.field_assignment_path(file)?).await?,
                "unreferenced branch assignment object '{}' remains after cleanup",
                file.path
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_cleanup_retains_parent_bitmap_referenced_by_branch_local_root() -> Result<()> {
        let directory = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("embedding", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..8)),
                Arc::new(Int32Array::from_iter_values(0..8)),
            ],
        )?;
        let mut parent = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &directory,
            Some(WriteParams {
                max_rows_per_file: 4,
                ..Default::default()
            }),
        )
        .await?;
        parent
            .alter_columns_with_assignment(
                &[],
                &[FieldAssignmentAlteration::new(
                    "embedding",
                    FieldAssignment::Assigned,
                )],
            )
            .await?;
        let result = UpdateBuilder::new(Arc::new(parent))
            .update_where("id = 1 OR id = 5")?
            .invalidate_fields(["embedding"])?
            .build()?
            .execute()
            .await?;
        let mut parent = result.new_dataset.as_ref().clone();
        let field_id = parent.schema().field("embedding").unwrap().id;
        let mut branch = parent
            .create_branch("assignment-local-root", parent.version().version, None)
            .await?;

        let result = UpdateBuilder::new(Arc::new(branch))
            .update_where("id = 0")?
            .invalidate_fields(["embedding"])?
            .build()?
            .execute()
            .await?;
        branch = result.new_dataset.as_ref().clone();
        let branch_descriptor = branch.field_assignment_state(field_id).unwrap();
        assert_eq!(branch_descriptor.root.base_id, None);
        let branch_root = branch.load_field_assignment_root(field_id).await?.unwrap();
        let inherited_bitmap = branch_root
            .fragments
            .iter()
            .find_map(|fragment| match &fragment.state {
                FieldAssignmentFragmentState::Partial(bitmap) if bitmap.base_id.is_some() => {
                    Some(bitmap.clone())
                }
                _ => None,
            })
            .unwrap_or_else(|| {
                panic!(
                    "branch-local root must retain an inherited parent bitmap: {:?}",
                    branch_root
                )
            });

        let branch_policy = CleanupPolicyBuilder::default()
            .retain_n_versions(&branch, 1)
            .await?
            .delete_unverified(true)
            .build();
        cleanup_old_versions(&branch, branch_policy).await?;

        let result = UpdateBuilder::new(Arc::new(parent))
            .update_where("id >= 0")?
            .set("embedding", "id * 100")?
            .build()?
            .execute()
            .await?;
        parent = result.new_dataset.as_ref().clone();
        let parent_policy = CleanupPolicyBuilder::default()
            .retain_n_versions(&parent, 1)
            .await?
            .delete_unverified(true)
            .build();
        cleanup_old_versions(&parent, parent_policy).await?;

        let inherited_path = Path::from_iter(
            parent
                .base
                .parts()
                .chain(Path::parse(&inherited_bitmap.path)?.parts()),
        );
        assert!(
            parent.object_store.exists(&inherited_path).await?,
            "parent cleanup must retain the bitmap referenced by the branch-local root"
        );
        let reopened_branch = parent.checkout_branch("assignment-local-root").await?;
        assert_eq!(
            assignment_rows(&reopened_branch, "embedding").await?,
            vec![
                (0, false),
                (1, false),
                (2, true),
                (3, true),
                (4, true),
                (5, false),
                (6, true),
                (7, true),
            ]
        );

        Ok(())
    }
}
