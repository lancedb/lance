// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lazy I/O for snapshot-level cell flag state.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

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
    CellFlagState, Manifest, cell_flag_bitmap_memory_size,
    decode_cell_flag_bitmap as decode_adaptive_cell_flag_bitmap, decode_cell_flag_root,
    encode_cell_flag_bitmap, encode_cell_flag_query_bitmap, encode_cell_flag_root, pb,
};
use object_store::{GetOptions, path::Path};
use roaring::{RoaringBitmap, RoaringTreemap};
use tokio::sync::Semaphore;
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
pub const ASSUME_CELL_FLAG_WRITER_GATE_DEPLOYED_ENV: &str =
    "LANCE_ASSUME_CELL_FLAG_WRITER_GATE_DEPLOYED";
const CELL_FLAG_ROOTS_DIR: &str = "roots";
const CELL_FLAG_BITMAPS_DIR: &str = "bitmaps";
const MAX_CELL_FLAG_FILE_BYTES: u64 = 32 * 1024 * 1024;
/// Individual portable Roaring bitmaps at or below this size are embedded in
/// the root when they are also sparse, avoiding one metadata request per
/// fragment without turning dense mutation state into monolithic metadata.
const MAX_INLINE_CELL_FLAG_BITMAP_BYTES: usize = 64 * 1024;
/// Inline only states at or below 25% density; denser states benefit from
/// independent immutable objects during repeated sparse mutations.
/// Cap total embedded bitmap bytes so large/high-fragment-count roots retain
/// incremental immutable bitmap objects instead of becoming monolithic.
const MAX_INLINE_CELL_FLAG_ROOT_BYTES: usize = 4 * 1024 * 1024;
/// Small roots are copied into their manifest descriptor so cold planning does
/// not require a separate object-store request.
const MAX_INLINE_CELL_FLAG_ROOT_COPY_BYTES: usize = 4 * 1024 * 1024;
/// Bound total root copies retained in one manifest across all tracked fields.
const MAX_INLINE_CELL_FLAG_MANIFEST_BYTES: usize = 4 * 1024 * 1024;
const CELL_FLAG_QUERY_MEMORY_BUDGET_BYTES: usize = 64 * 1024 * 1024;
const CELL_FLAG_WRITE_MEMORY_BUDGET_BYTES: usize = 64 * 1024 * 1024;
const CELL_FLAG_WRITE_MEMORY_PERMIT_BYTES: usize = 1024 * 1024;
const CELL_FLAG_WRITE_MEMORY_PERMITS: usize =
    CELL_FLAG_WRITE_MEMORY_BUDGET_BYTES / CELL_FLAG_WRITE_MEMORY_PERMIT_BYTES;
const CELL_FLAG_BITMAP_ENCODING_HEADER_BYTES: usize = 13;
const CELL_FLAG_BITMAP_COMPRESSED_LENGTH_BYTES: usize = 8;
// Zstd level 1 retains a compression context and match tables in addition to
// its output Vec. Reserve headroom for that workspace and candidate metadata.
const CELL_FLAG_BITMAP_ENCODER_FIXED_MEMORY_BYTES: usize = 4 * 1024 * 1024;

fn cell_flag_write_memory_weight(bytes: usize) -> u32 {
    bytes
        .max(1)
        .div_ceil(CELL_FLAG_WRITE_MEMORY_PERMIT_BYTES)
        .min(CELL_FLAG_WRITE_MEMORY_PERMITS) as u32
}

fn checked_cell_flag_memory_sum(label: &str, values: &[usize]) -> Result<usize> {
    values.iter().try_fold(0usize, |total, value| {
        total.checked_add(*value).ok_or_else(|| {
            Error::invalid_input(format!("Cell Flag {label} memory estimate overflow"))
        })
    })
}

fn cell_flag_bitmap_encoder_memory_bytes_for_sizes(
    roaring_bytes: usize,
    bitset_bytes: usize,
) -> Result<usize> {
    let compressed_roaring_bytes = zstd::zstd_safe::compress_bound(roaring_bytes);
    let compressed_bitset_bytes =
        (bitset_bytes != 0).then(|| zstd::zstd_safe::compress_bound(bitset_bytes));
    let compressed_bitset_bytes = compressed_bitset_bytes.unwrap_or_default();

    // encode_cell_flag_bitmap retains the raw and compressed buffers while it
    // compares raw, Zstd, bitset, and Zstd-bitset candidates. Account for both
    // compressed payload copies as well as the encoder's final output phase.
    let adaptive_candidate_peak = checked_cell_flag_memory_sum(
        "adaptive bitmap candidate",
        &[
            roaring_bytes,
            compressed_roaring_bytes,
            compressed_roaring_bytes
                .checked_add(CELL_FLAG_BITMAP_COMPRESSED_LENGTH_BYTES)
                .ok_or_else(|| {
                    Error::invalid_input(
                        "Cell Flag compressed Roaring memory estimate overflow".to_string(),
                    )
                })?,
            bitset_bytes,
            compressed_bitset_bytes,
            if bitset_bytes == 0 {
                0
            } else {
                compressed_bitset_bytes
                    .checked_add(CELL_FLAG_BITMAP_COMPRESSED_LENGTH_BYTES)
                    .ok_or_else(|| {
                        Error::invalid_input(
                            "Cell Flag compressed bitset memory estimate overflow".to_string(),
                        )
                    })?
            },
            CELL_FLAG_BITMAP_ENCODER_FIXED_MEMORY_BYTES,
        ],
    )?;
    let adaptive_output_peak = checked_cell_flag_memory_sum(
        "adaptive bitmap output",
        &[
            compressed_roaring_bytes,
            compressed_bitset_bytes,
            roaring_bytes,
            roaring_bytes
                .checked_add(CELL_FLAG_BITMAP_ENCODING_HEADER_BYTES)
                .ok_or_else(|| {
                    Error::invalid_input(
                        "Cell Flag adaptive bitmap output size overflow".to_string(),
                    )
                })?,
            CELL_FLAG_BITMAP_ENCODER_FIXED_MEMORY_BYTES,
        ],
    )?;

    // The query encoder is only invoked when the adaptive output is small
    // enough to inline. Its raw/Zstd candidates and final output coexist with
    // that retained adaptive output until the inline decision is complete.
    let inline_query_peak = checked_cell_flag_memory_sum(
        "query bitmap",
        &[
            MAX_INLINE_CELL_FLAG_BITMAP_BYTES,
            roaring_bytes,
            compressed_roaring_bytes,
            compressed_roaring_bytes
                .checked_add(CELL_FLAG_BITMAP_COMPRESSED_LENGTH_BYTES)
                .ok_or_else(|| {
                    Error::invalid_input(
                        "Cell Flag query compressed memory estimate overflow".to_string(),
                    )
                })?,
            roaring_bytes
                .checked_add(CELL_FLAG_BITMAP_ENCODING_HEADER_BYTES)
                .ok_or_else(|| {
                    Error::invalid_input("Cell Flag query bitmap output size overflow".to_string())
                })?,
            CELL_FLAG_BITMAP_ENCODER_FIXED_MEMORY_BYTES,
        ],
    )?;

    Ok(adaptive_candidate_peak
        .max(adaptive_output_peak)
        .max(inline_query_peak))
}

fn cell_flag_bitmap_encoder_memory_bytes(bitmap: &RoaringBitmap) -> Result<usize> {
    let roaring_bytes = bitmap.serialized_size();
    let bitset_bytes = bitmap
        .max()
        .map(|value| value as usize / 8 + 1)
        .filter(|bitset_bytes| *bitset_bytes < roaring_bytes)
        .unwrap_or_default();
    cell_flag_bitmap_encoder_memory_bytes_for_sizes(roaring_bytes, bitset_bytes)
}

fn cell_flag_row_change_memory_bytes(
    previous_state: Option<&CellFlagFragment>,
    change: &PendingRowChanges,
) -> Result<usize> {
    const ROARING_CONTAINER_OVERHEAD_BYTES: usize = 1024 * 1024;
    let previous_bytes = previous_state
        .map(cell_flag_query_memory_bytes)
        .transpose()?
        .unwrap_or_default();
    let input_bytes = previous_bytes
        .checked_add(change.set.serialized_size())
        .and_then(|bytes| bytes.checked_add(change.clear.serialized_size()))
        .and_then(|bytes| bytes.checked_add(ROARING_CONTAINER_OVERHEAD_BYTES))
        .ok_or_else(|| Error::invalid_input("Cell Flag row change size overflow".to_string()))?;
    if input_bytes as u64 > MAX_CELL_FLAG_FILE_BYTES {
        return Err(Error::invalid_input(format!(
            "Cell Flag row change requires {} bytes before serialization, maximum is {}",
            input_bytes, MAX_CELL_FLAG_FILE_BYTES
        )));
    }
    let encoder_bytes = cell_flag_bitmap_encoder_memory_bytes_for_sizes(
        input_bytes,
        input_bytes.saturating_sub(1),
    )?;
    input_bytes.checked_add(encoder_bytes).ok_or_else(|| {
        Error::invalid_input("Cell Flag row change memory estimate overflow".to_string())
    })
}
fn cell_flag_query_memory_bytes(fragment: &CellFlagFragment) -> Result<usize> {
    match &fragment.state {
        CellFlagFragmentState::All => Ok(0),
        CellFlagFragmentState::Partial(file) => {
            usize::try_from(file.memory_size_bytes).map_err(|_| {
                Error::invalid_input("Cell Flag bitmap memory size exceeds this platform")
            })
        }
        CellFlagFragmentState::InlinePartial(bytes) => cell_flag_bitmap_memory_size(bytes),
    }
}

fn reserve_cell_flag_query_memory(
    query_memory_bytes: &AtomicUsize,
    required_bytes: usize,
) -> Result<()> {
    query_memory_bytes
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current
                .checked_add(required_bytes)
                .filter(|total| *total <= CELL_FLAG_QUERY_MEMORY_BUDGET_BYTES)
        })
        .map(|_| ())
        .map_err(|current| {
            Error::invalid_input(format!(
                "Cell Flag query requires more than the {} byte binding budget (already reserved {}, requested {})",
                CELL_FLAG_QUERY_MEMORY_BUDGET_BYTES, current, required_bytes
            ))
        })
}

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
    if !file.path.is_empty() && bytes.len() as u64 != file.size_bytes {
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
    // A bounded GET caps the response body while its metadata still reports the
    // full object size, so a forged descriptor cannot trigger an unbounded read.
    let result = store
        .inner
        .get_opts(
            path,
            GetOptions {
                range: Some((0..file.size_bytes).into()),
                ..Default::default()
            },
        )
        .await?;
    let actual_size = result.meta.size;
    if actual_size != file.size_bytes {
        return Err(Error::invalid_input(format!(
            "Cell flag file '{}' has size {}, expected {}",
            file.path, actual_size, file.size_bytes
        )));
    }
    let bytes = result.bytes().await?;
    if bytes.len() != known_size {
        return Err(Error::invalid_input(format!(
            "Cell flag file '{}' range read returned size {}, expected {}",
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
    let memory_size = cell_flag_bitmap_memory_size(bytes)?;
    if let CellFlagFragmentState::Partial(file) = &fragment.state
        && memory_size as u64 != file.memory_size_bytes
    {
        return Err(Error::invalid_input(format!(
            "Cell flag bitmap '{}' for flag ID {}, fragment {} declares memory size {}, expected {}",
            source, flag_id, fragment.fragment_id, memory_size, file.memory_size_bytes
        )));
    }
    let bitmap = decode_adaptive_cell_flag_bitmap(bytes).map_err(|error| {
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

pub fn cell_flag_manifest_identity(manifest: &Manifest) -> String {
    fn update(hasher: &mut blake3::Hasher, value: &[u8]) {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value);
    }

    let mut hasher = blake3::Hasher::new();
    update(&mut hasher, b"lance-cell-flag-dataset-v1");
    update(&mut hasher, &manifest.version.to_le_bytes());
    update(
        &mut hasher,
        manifest
            .transaction_file
            .as_deref()
            .unwrap_or_default()
            .as_bytes(),
    );
    for fragment in manifest.fragments.iter() {
        update(&mut hasher, &fragment.id.to_le_bytes());
        for file in &fragment.files {
            update(&mut hasher, file.path.as_bytes());
        }
    }
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest.as_bytes()[..16]);
    bytes[6] = (bytes[6] & 0x0f) | 0x80;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Uuid::from_bytes(bytes).hyphenated().to_string()
}

impl Dataset {
    fn cell_flag_bootstrap_identity(&self) -> String {
        cell_flag_manifest_identity(&self.manifest)
    }

    pub(crate) fn cell_flag_transaction_identity(&self) -> String {
        self.manifest
            .cell_flag_dataset_id
            .clone()
            .unwrap_or_else(|| self.cell_flag_bootstrap_identity())
    }

    pub(crate) fn validate_cell_flag_transaction_identity(
        &self,
        changes: &CellFlagTransaction,
    ) -> Result<()> {
        match self.manifest.cell_flag_dataset_id.as_deref() {
            Some(dataset_id) if changes.dataset_identity == dataset_id => Ok(()),
            Some(_) => Err(Error::incompatible_transaction_source(
                "Cell flag transaction belongs to a different dataset incarnation".into(),
            )),
            None if self.manifest.next_cell_flag_id == 0
                && self.manifest.cell_flag_definitions.is_empty()
                && self.manifest.cell_flag_states.is_empty()
                && !changes.registrations.is_empty()
                && changes.dataset_identity == self.cell_flag_bootstrap_identity() =>
            {
                Ok(())
            }
            None if !changes.registrations.is_empty() => {
                Err(Error::incompatible_transaction_source(
                    "Cell flag transaction belongs to a different dataset incarnation".into(),
                ))
            }
            None => Err(Error::incompatible_transaction_source(
                "Cell flag transaction cannot target a dataset without a Cell Flag incarnation"
                    .into(),
            )),
        }
    }

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
    /// The first registration requires
    /// [`ASSUME_CELL_FLAG_WRITER_GATE_DEPLOYED_ENV`] to be set after every
    /// possible writer has deployed the gate-only compatibility release.
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
        self.register_cell_flags([(field.as_ref().to_owned(), name.into(), initial_value)])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| {
                Error::internal("Single Cell Flag registration returned no result".to_string())
            })
    }

    /// Register multiple field-scoped Boolean flags in one atomic commit.
    ///
    /// Registrations are applied in input order and receive consecutive stable
    /// IDs. If any registration is invalid, none of them are committed.
    ///
    /// ```no_run
    /// # use lance::{Dataset, Result};
    /// # async fn register(dataset: &mut Dataset) -> Result<()> {
    /// let definitions = dataset
    ///     .register_cell_flags([
    ///         ("embedding", "computed", false),
    ///         ("embedding", "reviewed", true),
    ///     ])
    ///     .await?;
    /// assert_eq!(definitions.len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn register_cell_flags<I, F, N>(
        &mut self,
        registrations: I,
    ) -> Result<Vec<CellFlagDefinition>>
    where
        I: IntoIterator<Item = (F, N, bool)>,
        F: AsRef<str>,
        N: Into<String>,
    {
        let registrations = registrations
            .into_iter()
            .map(|(field, name, initial_value)| {
                (field.as_ref().to_owned(), name.into(), initial_value)
            })
            .collect::<Vec<_>>();
        if registrations.is_empty() {
            return Err(Error::invalid_input(
                "At least one Cell Flag registration is required",
            ));
        }
        if self.manifest.next_cell_flag_id == 0
            && !cfg!(test)
            && std::env::var_os(ASSUME_CELL_FLAG_WRITER_GATE_DEPLOYED_ENV).is_none()
        {
            return Err(Error::not_supported(format!(
                "Registering the first Cell Flag is disabled until every writer has the gate-only release; set {} only after that deployment is complete",
                ASSUME_CELL_FLAG_WRITER_GATE_DEPLOYED_ENV
            )));
        }
        let mut next_flag_id = self.manifest.next_cell_flag_id;
        let mut field_names = HashSet::with_capacity(registrations.len());
        let mut definitions = Vec::with_capacity(registrations.len());
        let mut transaction_registrations = Vec::with_capacity(registrations.len());
        for (field, name, initial_value) in registrations {
            let field_id = self
                .schema()
                .field(&field)
                .ok_or_else(|| Error::invalid_input(format!("Unknown field '{}'", field)))?
                .id;
            if name.is_empty() {
                return Err(Error::invalid_input("Cell flag name must not be empty"));
            }
            if self.cell_flag_definition(field_id, &name).is_some()
                || !field_names.insert((field_id, name.clone()))
            {
                return Err(Error::invalid_input(format!(
                    "Cell flag '{}' is already registered for field '{}'",
                    name, field
                )));
            }
            let following_flag_id = next_flag_id.checked_add(1).ok_or_else(|| {
                Error::invalid_input("The dataset has exhausted the stable cell flag ID space")
            })?;
            let definition = CellFlagDefinition {
                flag_id: next_flag_id,
                field_id,
                name,
            };
            transaction_registrations.push(CellFlagRegistration {
                flag_id: definition.flag_id,
                field_id: definition.field_id,
                name: definition.name.clone(),
                initial_value,
            });
            definitions.push(definition);
            next_flag_id = following_flag_id;
        }
        let transaction = Transaction::new(
            self.manifest.version,
            Operation::Project {
                schema: self.schema().clone(),
                preserves_nullability: true,
            },
            None,
        )
        .with_cell_flag_transaction_for_dataset(
            CellFlagTransaction {
                registrations: transaction_registrations,
                ..Default::default()
            },
            self,
        )?;
        self.apply_commit(transaction, &Default::default(), &Default::default())
            .await?;
        Ok(definitions)
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
        .with_cell_flag_transaction_for_dataset(
            CellFlagTransaction {
                renames: vec![CellFlagRename {
                    flag_id,
                    name: new_name,
                }],
                ..Default::default()
            },
            self,
        )?;
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
        .with_cell_flag_transaction_for_dataset(
            CellFlagTransaction {
                drops: vec![CellFlagDrop { flag_id }],
                ..Default::default()
            },
            self,
        )?;
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
        let path = if descriptor.root.path.is_empty() {
            self.base.clone()
        } else {
            self.cell_flag_path(&descriptor.root)?
        };
        let source_uri = self.cell_flag_source_uri(&descriptor.root)?;
        let key = CellFlagRootKey {
            source_uri: source_uri.to_string(),
            path: descriptor.root.path.clone(),
            size_bytes: descriptor.root.size_bytes,
            memory_size_bytes: descriptor.root.memory_size_bytes,
            flag_id,
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
                let (proto, memory_size) =
                    decode_cell_flag_root(bytes.as_ref()).map_err(|error| {
                        Error::invalid_input(format!(
                            "Failed to decode cell flag root '{}' for flag ID {}: {}",
                            descriptor.root.path, flag_id, error
                        ))
                    })?;
                if memory_size as u64 != descriptor.root.memory_size_bytes {
                    return Err(Error::invalid_input(format!(
                        "Cell flag root '{}' for flag ID {} declares memory size {}, expected {}",
                        descriptor.root.path,
                        flag_id,
                        memory_size,
                        descriptor.root.memory_size_bytes
                    )));
                }
                let mut root = CellFlagRoot::try_from(proto)?;

                for entry in &mut root.fragments {
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
        for entry in &root.fragments {
            let fragment_id = u32::try_from(entry.fragment_id).map_err(|_| {
                Error::invalid_input(format!(
                    "Cell flag root for flag ID {} references invalid fragment {}",
                    flag_id, entry.fragment_id
                ))
            })?;
            let fragment = self
                .get_fragment_metadata_by_id(fragment_id)
                .ok_or_else(|| {
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
        }
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
        if descriptor.root.path.is_empty() {
            return Ok(());
        }
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
                    source_uri: source_uri.to_string(),
                    path: file.path.clone(),
                    size_bytes: file.size_bytes,
                    memory_size_bytes: file.memory_size_bytes,
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
                    source_uri: self.uri.clone(),
                    path: "inline".to_string(),
                    size_bytes: bytes.len() as u64,
                    memory_size_bytes: cell_flag_bitmap_memory_size(bytes)? as u64,
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
        // The adaptive candidates include both query codecs, so a result above
        // the inline limit proves the query encoding cannot fit either. Only
        // pay for the query-friendly encoding when inlining remains possible.
        let bytes = encode_cell_flag_bitmap(bitmap)?;
        if bytes.len() <= MAX_INLINE_CELL_FLAG_BITMAP_BYTES {
            let query_bytes = encode_cell_flag_query_bitmap(bitmap)?;
            if query_bytes.len() <= MAX_INLINE_CELL_FLAG_BITMAP_BYTES {
                cell_flag_bitmap_memory_size(&query_bytes)?;
                return Ok(CellFlagFragmentState::InlinePartial(query_bytes));
            }
        }
        if bytes.len() as u64 > MAX_CELL_FLAG_FILE_BYTES {
            return Err(Error::invalid_input(format!(
                "Cell flag bitmap for flag ID {}, fragment {} has encoded size {}, maximum is {}",
                flag_id,
                fragment_id,
                bytes.len(),
                MAX_CELL_FLAG_FILE_BYTES
            )));
        }
        let memory_size = cell_flag_bitmap_memory_size(&bytes)?;
        Ok(CellFlagFragmentState::Partial(
            self.write_cell_flag_bitmap_file(
                write_store,
                flag_id,
                fragment_id,
                memory_size,
                &bytes,
            )
            .await?,
        ))
    }

    async fn write_cell_flag_bitmap_file(
        &self,
        write_store: &ObjectStore,
        flag_id: u32,
        fragment_id: u64,
        memory_size: usize,
        bytes: &[u8],
    ) -> Result<CellFlagFile> {
        if bytes.len() as u64 > MAX_CELL_FLAG_FILE_BYTES {
            return Err(Error::invalid_input(format!(
                "Cell flag bitmap for flag ID {}, fragment {} has size {}, maximum is {}",
                flag_id,
                fragment_id,
                bytes.len(),
                MAX_CELL_FLAG_FILE_BYTES
            )));
        }
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
            memory_size_bytes: memory_size as u64,
            base_id: None,
            inline_bytes: None,
        })
    }

    pub(crate) async fn write_cell_flag_root(
        &self,
        write_store: &ObjectStore,
        flag_id: u32,
        mut root: CellFlagRoot,
        max_inline_bytes: usize,
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
                    .write_cell_flag_bitmap_file(
                        write_store,
                        flag_id,
                        entry.fragment_id,
                        cell_flag_bitmap_memory_size(&bytes)?,
                        &bytes,
                    )
                    .await?;
                entry.state = CellFlagFragmentState::Partial(file);
            }
        }
        let proto_root = pb::CellFlagRoot::from(&root);
        let (bytes, memory_size) = encode_cell_flag_root(&proto_root)?;
        if bytes.len() as u64 > MAX_CELL_FLAG_FILE_BYTES {
            return Err(Error::internal(format!(
                "Cell flag root for flag ID {} has encoded size {}, maximum is {}",
                flag_id,
                bytes.len(),
                MAX_CELL_FLAG_FILE_BYTES
            )));
        }
        let inline_only =
            bytes.len() <= MAX_INLINE_CELL_FLAG_ROOT_COPY_BYTES && bytes.len() <= max_inline_bytes;
        let (path, size_bytes, inline_bytes) = if inline_only {
            (String::new(), 0, Some(bytes.to_vec()))
        } else {
            let relative = Path::from(CELL_FLAGS_DIR)
                .join(CELL_FLAG_ROOTS_DIR)
                .join(flag_id.to_string())
                .join(format!("{}.root", Uuid::new_v4()));
            let full_path = Path::from_iter(self.base.parts().chain(relative.parts()));
            write_store.put(&full_path, &bytes).await?;
            (relative.to_string(), bytes.len() as u64, None)
        };
        Ok(Some(CellFlagState {
            flag_id,
            root: CellFlagFile {
                path,
                size_bytes,
                memory_size_bytes: memory_size as u64,
                base_id: None,
                inline_bytes,
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
            if !value {
                continue;
            }
            for fragment in new_fragments {
                states.push(TransactionCellFlagFragmentState {
                    fragment_path: fragment_path(fragment)?.to_string(),
                    flag_id,
                    state: CellFlagFragmentValue::All,
                });
            }
        }
        Ok(states)
    }

    /// Build state for explicitly true flags on newly inserted fragments.
    /// False is represented by absence.
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
        let mut states = Vec::with_capacity(new_fragments.len() * changes.len());
        for (&flag_id, &value) in changes {
            if !value {
                continue;
            }
            for fragment in new_fragments {
                states.push(TransactionCellFlagFragmentState {
                    fragment_path: fragment_path(fragment)?.to_string(),
                    flag_id,
                    state: CellFlagFragmentValue::All,
                });
            }
        }
        Ok(states)
    }

    pub(crate) async fn cell_flag_rewrite_required(
        &self,
        source_fragment_ids: &HashSet<u64>,
        matched_changes: &HashMap<u32, bool>,
        inserted_changes: &HashMap<u32, bool>,
    ) -> Result<bool> {
        if matched_changes
            .values()
            .chain(inserted_changes.values())
            .any(|value| *value)
        {
            return Ok(true);
        }
        for state in &self.manifest.cell_flag_states {
            let Some(root) = self.load_cell_flag_root_shared(state.flag_id).await? else {
                continue;
            };
            if root
                .fragments
                .iter()
                .any(|fragment| source_fragment_ids.contains(&fragment.fragment_id))
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(crate) async fn materialized_cell_flag_fragment_ids(&self) -> Result<HashSet<u64>> {
        let mut fragment_ids = HashSet::new();
        for state in &self.manifest.cell_flag_states {
            if let Some(root) = self.load_cell_flag_root_shared(state.flag_id).await? {
                fragment_ids.extend(root.fragments.iter().map(|fragment| fragment.fragment_id));
            }
        }
        Ok(fragment_ids)
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

    /// Remap flag membership for an order-preserving compaction without
    /// materializing one source address per output row for every flag.
    pub(crate) async fn cell_flag_states_for_compacted_rows(
        &self,
        new_fragments: &[lance_table::format::Fragment],
        source_row_addresses: &RoaringTreemap,
    ) -> Result<Vec<TransactionCellFlagFragmentState>> {
        let output_rows = new_fragments.iter().try_fold(0_u64, |total, fragment| {
            total
                .checked_add(fragment_physical_rows(fragment)?)
                .ok_or_else(|| Error::invalid_input("Compacted row count overflow"))
        })?;
        if output_rows != source_row_addresses.len() {
            return Err(Error::internal(format!(
                "Compacted fragments contain {} rows but {} source row addresses were captured",
                output_rows,
                source_row_addresses.len()
            )));
        }

        let source_fragment_layout = compacted_source_fragment_layout(source_row_addresses)?;
        let source_fragments = source_fragment_layout
            .keys()
            .copied()
            .collect::<HashSet<_>>();
        let mut output_ends = Vec::with_capacity(new_fragments.len());
        let mut output_end = 0_u64;
        for fragment in new_fragments {
            output_end = output_end
                .checked_add(fragment_physical_rows(fragment)?)
                .ok_or_else(|| Error::invalid_input("Compacted row count overflow"))?;
            output_ends.push(output_end);
        }
        let mut states = Vec::new();

        for definition in &self.manifest.cell_flag_definitions {
            let flag_id = definition.flag_id;
            let Some(root) = self.load_cell_flag_root_shared(flag_id).await? else {
                continue;
            };
            let source_states = select_cell_flag_fragments(&root, Some(&source_fragments));
            if source_states.is_empty() {
                continue;
            }

            let mut output_bitmaps = (0..new_fragments.len())
                .map(|_| RoaringBitmap::new())
                .collect::<Vec<_>>();
            for source_state in source_states {
                let fragment_id = u32::try_from(source_state.fragment_id).map_err(|_| {
                    Error::invalid_input(format!(
                        "Cell flag fragment ID {} does not fit in a row address",
                        source_state.fragment_id
                    ))
                })?;
                let Some(&(source_live_prefix, source_live_offsets)) =
                    source_fragment_layout.get(&fragment_id)
                else {
                    continue;
                };
                match &source_state.state {
                    CellFlagFragmentState::All => {
                        remap_compacted_cell_flag_offsets(
                            source_live_offsets,
                            source_live_prefix,
                            source_live_offsets,
                            &output_ends,
                            &mut output_bitmaps,
                        )?;
                    }
                    CellFlagFragmentState::Partial(_) | CellFlagFragmentState::InlinePartial(_) => {
                        let source_flag_offsets = self
                            .load_cell_flag_bitmap_shared(flag_id, &source_state)
                            .await?;
                        remap_compacted_cell_flag_offsets(
                            source_live_offsets,
                            source_live_prefix,
                            source_flag_offsets.as_ref(),
                            &output_ends,
                            &mut output_bitmaps,
                        )?;
                    }
                }
            }

            for (fragment, bitmap) in new_fragments.iter().zip(output_bitmaps) {
                let physical_rows = fragment_physical_rows(fragment)?;
                let state = if bitmap.is_empty() {
                    CellFlagFragmentValue::None
                } else if bitmap.len() == physical_rows {
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
            .map(|(_, address)| RowAddress::from(*address).fragment_id())
            .collect::<HashSet<_>>();
        let mut states = Vec::new();
        for definition in &self.manifest.cell_flag_definitions {
            let flag_id = definition.flag_id;
            let produces_true = matched_changes.get(&flag_id) == Some(&true)
                || inserted_changes.get(&flag_id) == Some(&true);
            if self.cell_flag_state(flag_id).is_none() && !produces_true {
                continue;
            }
            let source_fragments = match self.load_cell_flag_root_shared(flag_id).await? {
                Some(root) => select_cell_flag_fragments(&root, Some(&source_fragment_ids)),
                None => Vec::new(),
            };
            if source_fragments.is_empty() && !produces_true {
                continue;
            }
            let mut source_states = HashMap::new();
            for fragment in source_fragments {
                let state = match &fragment.state {
                    CellFlagFragmentState::All => None,
                    CellFlagFragmentState::Partial(_) | CellFlagFragmentState::InlinePartial(_) => {
                        Some(
                            self.load_cell_flag_bitmap_shared(flag_id, &fragment)
                                .await?,
                        )
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

pub fn field_reference_segments(expr: &Expr) -> Result<Vec<String>> {
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

pub fn field_reference_column(expr: &Expr) -> Option<&datafusion::common::Column> {
    match expr {
        Expr::Column(column) => Some(column),
        Expr::ScalarFunction(function) if function.func.name() == "get_field" => {
            function.args.first().and_then(field_reference_column)
        }
        _ => None,
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

fn resolve_cell_flag(dataset: &Dataset, segments: &[String], name: &str) -> Result<(u32, i32)> {
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
    let definition = dataset
        .cell_flag_definition(field.id, name)
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "cell_flag references unknown flag '{}' for field '{}' (stable field ID {})",
                name,
                segments.join("."),
                field.id
            ))
        })?;
    Ok((definition.flag_id, field.id))
}

pub struct CellFlagCallBinding {
    pub flag_id: u32,
    pub field_id: i32,
    pub relation: Option<datafusion::common::TableReference>,
    pub field_expression: Expr,
}

pub fn cell_flag_call_flag_id_and_relation(
    dataset: &Dataset,
    expression: &Expr,
) -> Result<Option<CellFlagCallBinding>> {
    let Some((segments, name)) = is_cell_flag_call(expression)? else {
        return Ok(None);
    };
    let Expr::ScalarFunction(function) = expression else {
        unreachable!("is_cell_flag_call only matches scalar functions");
    };
    let relation = function
        .args
        .first()
        .and_then(field_reference_column)
        .and_then(|column| column.relation.clone());
    let (flag_id, field_id) = resolve_cell_flag(dataset, &segments, &name)?;
    Ok(Some(CellFlagCallBinding {
        flag_id,
        field_id,
        relation,
        field_expression: function.args[0].clone(),
    }))
}

pub async fn load_cell_flag_fragments(
    dataset: &Dataset,
    flag_id: u32,
    selected_fragment_ids: Option<Arc<HashSet<u32>>>,
    query_memory_bytes: &AtomicUsize,
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
    let fragments = select_cell_flag_fragments(&root, selected_fragment_ids.as_deref());
    let required_bytes = fragments.iter().try_fold(0usize, |total, fragment| {
        total
            .checked_add(cell_flag_query_memory_bytes(fragment)?)
            .ok_or_else(|| {
                Error::invalid_input("Cell Flag query memory estimate overflow".to_string())
            })
    })?;
    reserve_cell_flag_query_memory(query_memory_bytes, required_bytes)?;
    let io_parallelism = dataset.object_store.io_parallelism().max(1);
    futures::stream::iter(fragments)
        .map(|fragment| async move {
            let fragment_id = u32::try_from(fragment.fragment_id).map_err(|_| {
                Error::invalid_input(format!(
                    "Cell flag fragment ID {} does not fit in a row address",
                    fragment.fragment_id
                ))
            })?;
            let state = match &fragment.state {
                CellFlagFragmentState::All => FlagFragment::All,
                CellFlagFragmentState::Partial(_) | CellFlagFragmentState::InlinePartial(_) => {
                    let bitmap = dataset
                        .load_cell_flag_bitmap_shared(flag_id, &fragment)
                        .await?;
                    FlagFragment::Partial(bitmap)
                }
            };
            Ok::<(u32, FlagFragment), Error>((fragment_id, state))
        })
        .buffer_unordered(io_parallelism)
        .try_collect::<HashMap<_, _>>()
        .await
}

fn select_cell_flag_fragments(
    root: &CellFlagRoot,
    selected_fragment_ids: Option<&HashSet<u32>>,
) -> Vec<CellFlagFragment> {
    let Some(selected_fragment_ids) = selected_fragment_ids else {
        return root.fragments.clone();
    };
    if selected_fragment_ids.is_empty() {
        return Vec::new();
    }

    if selected_fragment_ids.len().saturating_mul(8) <= root.fragments.len() {
        selected_fragment_ids
            .iter()
            .filter_map(|fragment_id| {
                root.fragments
                    .binary_search_by_key(&u64::from(*fragment_id), |entry| entry.fragment_id)
                    .ok()
                    .map(|position| root.fragments[position].clone())
            })
            .collect()
    } else {
        root.fragments
            .iter()
            .filter(|entry| {
                u32::try_from(entry.fragment_id)
                    .ok()
                    .is_some_and(|fragment_id| selected_fragment_ids.contains(&fragment_id))
            })
            .cloned()
            .collect()
    }
}

#[cfg(feature = "substrait")]
pub fn cell_flag_expression_for_transport(expression: Expr) -> Result<Expr> {
    Ok(expression
        .transform(|node| {
            let Expr::ScalarFunction(function) = &node else {
                return Ok(Transformed::no(node));
            };
            let Some(flag_id) = bound_cell_flag_flag_id(function.func.as_ref()) else {
                return Ok(Transformed::no(node));
            };
            Ok(Transformed::yes(cell_flag_id(flag_id)))
        })
        .map(|transformed| transformed.data)?)
}

#[cfg(feature = "substrait")]
pub fn unbind_cell_flag_expression(dataset: &Dataset, expression: Expr) -> Result<Expr> {
    let mut unknown_flag_id = None;
    expression.apply(|node| {
        let Expr::ScalarFunction(function) = node else {
            return Ok(TreeNodeRecursion::Continue);
        };
        if let Some(flag_id) = bound_cell_flag_flag_id(function.func.as_ref())
            && dataset.cell_flag_definition_by_id(flag_id).is_none()
        {
            unknown_flag_id = Some(flag_id);
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    if let Some(flag_id) = unknown_flag_id {
        return Err(Error::invalid_input(format!(
            "Bound cell_flag references unknown flag ID {}",
            flag_id
        )));
    }
    cell_flag_expression_for_transport(expression)
}

#[cfg(feature = "substrait")]
pub async fn bind_cell_flag_expressions(
    dataset: &Dataset,
    expressions: Vec<Expr>,
    selected_fragments: Option<&[lance_table::format::Fragment]>,
) -> Result<Vec<Expr>> {
    let expression_refs = expressions.iter().collect::<Vec<_>>();
    let Some(bindings) =
        CellFlagExprBindings::try_new(dataset, &expression_refs, selected_fragments).await?
    else {
        return Ok(expressions);
    };
    expressions
        .into_iter()
        .map(|expression| bindings.bind(expression, dataset))
        .collect()
}

/// Snapshot bindings shared by every cell flag expression in one scanner.
pub struct CellFlagExprBindings {
    functions: HashMap<u32, Arc<ScalarUDF>>,
}

fn resolve_cell_flag_expression_ids_impl(
    dataset: &Dataset,
    expressions: &[&Expr],
    field_prefix: Option<&str>,
) -> Result<Vec<u32>> {
    let mut flag_ids = HashSet::new();
    let mut captured_error = None;
    for expression in expressions {
        expression
            .apply(|node| {
                match is_cell_flag_call(node) {
                    Ok(Some((segments, name))) => {
                        let segments = field_prefix
                            .filter(|prefix| segments.first().is_some_and(|part| part == prefix))
                            .map_or(segments.as_slice(), |_| &segments[1..]);
                        match resolve_cell_flag(dataset, segments, &name) {
                            Ok((flag_id, _)) => {
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
    let mut flag_ids = flag_ids.into_iter().collect::<Vec<_>>();
    flag_ids.sort_unstable();
    Ok(flag_ids)
}

pub(super) fn resolve_cell_flag_expression_ids(
    dataset: &Dataset,
    expressions: &[&Expr],
) -> Result<Vec<u32>> {
    resolve_cell_flag_expression_ids_impl(dataset, expressions, None)
}

pub(super) fn resolve_target_cell_flag_expression_ids(
    dataset: &Dataset,
    expressions: &[&Expr],
) -> Result<Vec<u32>> {
    resolve_cell_flag_expression_ids_impl(dataset, expressions, Some("target"))
}

impl CellFlagExprBindings {
    async fn try_new_impl(
        dataset: &Dataset,
        expressions: &[&Expr],
        selected_fragments: Option<&[lance_table::format::Fragment]>,
        field_prefix: Option<&str>,
    ) -> Result<Option<Self>> {
        let flag_ids = resolve_cell_flag_expression_ids_impl(dataset, expressions, field_prefix)?;
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
        let query_memory_bytes = AtomicUsize::new(0);
        for flag_id in flag_ids {
            let fragments = load_cell_flag_fragments(
                dataset,
                flag_id,
                selected_fragment_ids.clone(),
                &query_memory_bytes,
            )
            .await?;
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

    pub(crate) async fn try_new(
        dataset: &Dataset,
        expressions: &[&Expr],
        selected_fragments: Option<&[lance_table::format::Fragment]>,
    ) -> Result<Option<Self>> {
        Self::try_new_impl(dataset, expressions, selected_fragments, None).await
    }

    pub(crate) async fn try_new_for_target(
        dataset: &Dataset,
        expressions: &[&Expr],
        selected_fragments: Option<&[lance_table::format::Fragment]>,
    ) -> Result<Option<Self>> {
        Self::try_new_impl(dataset, expressions, selected_fragments, Some("target")).await
    }

    pub(crate) fn bind(&self, expression: Expr, dataset: &Dataset) -> Result<Expr> {
        self.bind_impl(expression, dataset, None)
    }

    pub(crate) fn bind_target(&self, expression: Expr, dataset: &Dataset) -> Result<Expr> {
        self.bind_impl(expression, dataset, Some("target"))
    }

    fn bind_impl(
        &self,
        expression: Expr,
        dataset: &Dataset,
        field_prefix: Option<&str>,
    ) -> Result<Expr> {
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
                    let segments = field_prefix
                        .filter(|prefix| segments.first().is_some_and(|part| part == prefix))
                        .map_or(segments.as_slice(), |_| &segments[1..]);
                    match resolve_cell_flag(dataset, segments, &name) {
                        Ok((flag_id, _)) => flag_id,
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

fn compacted_source_fragment_layout(
    source_row_addresses: &RoaringTreemap,
) -> Result<HashMap<u32, (u64, &RoaringBitmap)>> {
    let mut live_prefix = 0_u64;
    let mut layout = HashMap::new();
    for (fragment_id, live_offsets) in source_row_addresses.bitmaps() {
        layout.insert(fragment_id, (live_prefix, live_offsets));
        live_prefix = live_prefix
            .checked_add(live_offsets.len())
            .ok_or_else(|| Error::invalid_input("Compacted source row count overflow"))?;
    }
    Ok(layout)
}

fn remap_compacted_cell_flag_offsets(
    source_live_offsets: &RoaringBitmap,
    source_live_prefix: u64,
    source_flag_offsets: &RoaringBitmap,
    output_ends: &[u64],
    output_bitmaps: &mut [RoaringBitmap],
) -> Result<()> {
    for source_offset in source_flag_offsets.iter() {
        if !source_live_offsets.contains(source_offset) {
            continue;
        }
        let output_position = source_live_prefix
            .checked_add(source_live_offsets.rank(source_offset) - 1)
            .ok_or_else(|| Error::internal("Compacted Cell Flag position overflow"))?;
        let output_fragment = output_ends.partition_point(|end| *end <= output_position);
        let output_start = output_fragment
            .checked_sub(1)
            .map_or(0, |previous| output_ends[previous]);
        let output_bitmap = output_bitmaps.get_mut(output_fragment).ok_or_else(|| {
            Error::internal("Compacted Cell Flag row maps outside the output fragments")
        })?;
        output_bitmap.insert(
            u32::try_from(output_position - output_start)
                .map_err(|_| Error::internal("Compacted Cell Flag offset exceeds u32"))?,
        );
    }
    Ok(())
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
        CellFlagFragmentState::Partial(file) => {
            file.validate_bitmap_path_for_fragment(flag_id, entry.fragment_id)?;
            let path = current.cell_flag_path(file)?;
            let store = current.object_store(file.base_id).await?;
            let bytes = read_cell_flag_bytes(&store, &path, file).await?;
            decode_cell_flag_bitmap(&bytes, &file.path, flag_id, entry)
        }
        CellFlagFragmentState::InlinePartial(bytes) => {
            decode_cell_flag_bitmap(bytes, "inline root entry", flag_id, entry)
        }
    }
}

#[derive(Clone, Default)]
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

struct InlineCellFlagManifestBudget {
    used_bytes: usize,
    max_bytes: usize,
}

impl InlineCellFlagManifestBudget {
    fn from_states<'a>(
        states: impl IntoIterator<Item = &'a CellFlagState>,
        max_bytes: usize,
    ) -> Result<Self> {
        let mut budget = Self {
            used_bytes: 0,
            max_bytes,
        };
        for state in states {
            budget.reserve(Some(state))?;
        }
        Ok(budget)
    }

    fn inline_bytes(state: Option<&CellFlagState>) -> usize {
        state
            .and_then(|state| state.root.inline_bytes.as_ref())
            .map_or(0, Vec::len)
    }

    /// Release the current descriptor while reserving every other flag's
    /// inline root. The returned capacity is safe to use for its replacement.
    fn release(&mut self, state: Option<&CellFlagState>) -> Result<usize> {
        let bytes = Self::inline_bytes(state);
        self.used_bytes = self.used_bytes.checked_sub(bytes).ok_or_else(|| {
            Error::internal("Inline cell flag manifest budget accounting underflow")
        })?;
        Ok(self.max_bytes - self.used_bytes)
    }

    fn reserve(&mut self, state: Option<&CellFlagState>) -> Result<()> {
        self.used_bytes = self
            .used_bytes
            .checked_add(Self::inline_bytes(state))
            .ok_or_else(|| Error::internal("Inline cell flag manifest size overflow"))?;
        if self.used_bytes > self.max_bytes {
            return Err(Error::invalid_input(format!(
                "Inline Cell Flag roots require {} bytes, maximum is {}",
                self.used_bytes, self.max_bytes
            )));
        }
        Ok(())
    }
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
    let changes = transaction.cell_flag_transaction()?;
    if transaction.cell_flag_rewrite_requires_affected_rows()
        && changes
            .as_ref()
            .is_some_and(|changes| changes.has_writes() && changes.affected_rows.is_none())
    {
        return Err(Error::invalid_input(
            "Cell Flag update with existing-row rewrites requires the complete affected-row set",
        ));
    }
    if let Some(changes) = changes.as_ref() {
        if current.manifest.cell_flag_dataset_id.is_none()
            && current.manifest.version != transaction.read_version
        {
            current
                .checkout_version(transaction.read_version)
                .await?
                .validate_cell_flag_transaction_identity(changes)?;
        } else {
            current.validate_cell_flag_transaction_identity(changes)?;
        }
        if let Some(flag_id) = changes
            .read_flag_ids
            .iter()
            .find(|flag_id| current.cell_flag_definition_by_id(**flag_id).is_none())
        {
            return Err(Error::invalid_input(format!(
                "Cell Flag transaction reads unknown flag ID {}",
                flag_id
            )));
        }
    }
    let changes = changes.unwrap_or_default();
    if matches!(
        transaction.operation,
        Operation::Restore { .. } | Operation::Clone { .. }
    ) {
        if transaction.cell_flag_transaction_payload().is_some() {
            return Err(Error::invalid_input(
                "Restore and clone transactions cannot carry cell flag changes",
            ));
        }
        manifest.cell_flag_dataset_id = match &transaction.operation {
            Operation::Clone { .. } if manifest.next_cell_flag_id > 0 => {
                Some(cell_flag_manifest_identity(manifest))
            }
            _ => current.manifest.cell_flag_dataset_id.clone(),
        };
        return Ok(());
    }
    if current.manifest.cell_flag_dataset_id.is_none() && changes.has_writes() {
        manifest.cell_flag_dataset_id = Some(changes.dataset_identity.clone());
    }
    if changes.is_empty() && current.manifest.cell_flag_definitions.is_empty() {
        manifest.cell_flag_definitions.clear();
        manifest.cell_flag_states.clear();
        manifest.next_cell_flag_id = current.manifest.next_cell_flag_id;
        manifest.cell_flag_dataset_id = current.manifest.cell_flag_dataset_id.clone();
        return Ok(());
    }
    if changes.is_empty() && matches!(transaction.operation, Operation::Append { .. }) {
        // Append manifest construction already inherits the immutable registry
        // and roots. New fragment state is false by absence.
        return Ok(());
    }
    let final_field_ids: HashSet<i32> = manifest.schema.field_ids().into_iter().collect();

    if matches!(transaction.operation, Operation::Merge { .. }) {
        let transfer_sources = changes
            .transfers
            .iter()
            .map(|transfer| transfer.source_field_id)
            .collect::<HashSet<_>>();
        let dropped_flag_ids = changes
            .drops
            .iter()
            .map(|drop| drop.flag_id)
            .collect::<HashSet<_>>();
        if let Some(definition) = current
            .manifest
            .cell_flag_definitions
            .iter()
            .find(|definition| {
                !final_field_ids.contains(&definition.field_id)
                    && !transfer_sources.contains(&definition.field_id)
                    && !dropped_flag_ids.contains(&definition.flag_id)
            })
        {
            return Err(Error::invalid_input(format!(
                "Merge removes field ID {} tracked by cell flag ID {} without an explicit field transfer or flag drop",
                definition.field_id, definition.flag_id
            )));
        }
        if let Some(drop) = changes.drops.iter().find(|drop| {
            current
                .manifest
                .cell_flag_definitions
                .iter()
                .find(|definition| definition.flag_id == drop.flag_id)
                .is_some_and(|definition| final_field_ids.contains(&definition.field_id))
        }) {
            return Err(Error::invalid_input(format!(
                "Merge cannot drop cell flag ID {} while its field remains in the final schema",
                drop.flag_id
            )));
        }
    }

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

    let exact_rewrite_requirements: Vec<(HashSet<u64>, Vec<&str>)> = match &transaction.operation {
        Operation::Rewrite { groups, .. } => groups
            .iter()
            .map(|group| {
                Ok((
                    group
                        .old_fragments
                        .iter()
                        .map(|fragment| fragment.id)
                        .collect(),
                    group
                        .new_fragments
                        .iter()
                        .map(fragment_path)
                        .collect::<Result<_>>()?,
                ))
            })
            .collect::<Result<_>>()?,
        Operation::Update {
            updated_fragments,
            removed_fragment_ids,
            new_fragments,
            update_mode,
            ..
        } if !matches!(update_mode, Some(UpdateMode::RewriteColumns))
            && (!removed_fragment_ids.is_empty() || !new_fragments.is_empty()) =>
        {
            let mut source_fragment_ids = changes
                .affected_rows
                .iter()
                .flat_map(|rows| rows.iter().map(|(fragment_id, _)| *fragment_id as u64))
                .collect::<HashSet<_>>();
            if source_fragment_ids.is_empty() {
                source_fragment_ids.extend(updated_fragments.iter().map(|fragment| fragment.id));
                source_fragment_ids.extend(removed_fragment_ids.iter().copied());
            }
            vec![(
                source_fragment_ids,
                new_fragments
                    .iter()
                    .map(fragment_path)
                    .collect::<Result<_>>()?,
            )]
        }
        _ => Vec::new(),
    };
    let exact_rewrite_group_by_source_fragment = exact_rewrite_requirements
        .iter()
        .enumerate()
        .flat_map(|(group_index, (source_fragment_ids, _))| {
            source_fragment_ids
                .iter()
                .map(move |fragment_id| (*fragment_id, group_index))
        })
        .collect::<HashMap<_, _>>();
    let has_exact_rewrite = !exact_rewrite_requirements.is_empty();
    let has_exact_rewrite_sources = !exact_rewrite_group_by_source_fragment.is_empty();

    let mut descriptors = Vec::with_capacity(definitions.len());
    let mut inline_manifest_budget = InlineCellFlagManifestBudget::from_states(
        definitions
            .iter()
            .filter_map(|definition| current.cell_flag_state(definition.flag_id)),
        MAX_INLINE_CELL_FLAG_MANIFEST_BYTES,
    )?;
    for definition in &definitions {
        let flag_id = definition.flag_id;
        let current_descriptor = current.cell_flag_state(flag_id);
        let max_inline_bytes = inline_manifest_budget.release(current_descriptor)?;
        let has_explicit_state_change = registration_initial_values.contains_key(&flag_id)
            || overrides
                .keys()
                .any(|(override_flag_id, _)| *override_flag_id == flag_id)
            || pending_changes
                .keys()
                .any(|(change_flag_id, _)| *change_flag_id == flag_id);
        let current_rewrite_root =
            if !has_exact_rewrite_sources || current.cell_flag_state(flag_id).is_none() {
                None
            } else {
                current.load_cell_flag_root_shared(flag_id).await?
            };
        let required_exact_groups = current_rewrite_root
            .as_ref()
            .map(|root| {
                root.fragments
                    .iter()
                    .filter_map(|fragment| {
                        exact_rewrite_group_by_source_fragment
                            .get(&fragment.fragment_id)
                            .copied()
                    })
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_default();
        let exact_rewrite_needs_state = !required_exact_groups.is_empty();
        if exact_rewrite_needs_state {
            for group_index in required_exact_groups {
                for path in &exact_rewrite_requirements[group_index].1 {
                    let fragment = fragment_by_path.get(path).ok_or_else(|| {
                        Error::internal(format!(
                            "Rewritten fragment '{}' is missing from the final manifest",
                            path
                        ))
                    })?;
                    if !overrides.contains_key(&(flag_id, fragment.id)) {
                        return Err(Error::invalid_input(format!(
                            "Physical row rewrite must supply exact state for cell flag ID {}, new fragment '{}'",
                            flag_id, path
                        )));
                    }
                }
            }
        }

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
            || (has_exact_rewrite && !exact_rewrite_needs_state && !has_explicit_state_change)
        {
            if let Some(descriptor) = current_descriptor {
                inline_manifest_budget.reserve(Some(descriptor))?;
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
                let root = if let Some(root) = current_rewrite_root {
                    Some(root.as_ref().clone())
                } else {
                    current.load_cell_flag_root(flag_id).await?
                };
                root.map(|root| {
                    root.fragments
                        .into_iter()
                        .filter(|entry| final_fragments.contains_key(&entry.fragment_id))
                        .map(|entry| (entry.fragment_id, entry))
                        .collect()
                })
                .unwrap_or_default()
            };

        let mut flag_overrides = Vec::new();
        for ((_, fragment_id), value) in overrides
            .iter()
            .filter(|((override_flag_id, _), _)| *override_flag_id == flag_id)
        {
            let fragment = final_fragments.get(fragment_id).ok_or_else(|| {
                Error::internal(format!("Final fragment {} is missing", fragment_id))
            })?;
            let physical_rows = fragment_physical_rows(fragment)?;
            validate_fragment_value(flag_id, *fragment_id, physical_rows, value)?;
            let serialized_bytes = match value {
                CellFlagFragmentValue::Partial(bitmap) => bitmap.serialized_size(),
                CellFlagFragmentValue::None | CellFlagFragmentValue::All => 0,
            };
            if serialized_bytes as u64 > MAX_CELL_FLAG_FILE_BYTES {
                return Err(Error::invalid_input(format!(
                    "Cell Flag override for flag ID {}, fragment {} has encoded size {}, maximum is {}",
                    flag_id, fragment_id, serialized_bytes, MAX_CELL_FLAG_FILE_BYTES
                )));
            }
            let write_memory_bytes = match value {
                CellFlagFragmentValue::Partial(bitmap) => {
                    cell_flag_bitmap_encoder_memory_bytes(bitmap)?
                }
                CellFlagFragmentValue::None | CellFlagFragmentValue::All => 0,
            };
            flag_overrides.push((*fragment_id, physical_rows, write_memory_bytes));
        }
        let io_parallelism = write_store.io_parallelism().max(1);
        let write_memory = Arc::new(Semaphore::new(CELL_FLAG_WRITE_MEMORY_PERMITS));
        let override_states = futures::stream::iter(flag_overrides)
            .map(|(fragment_id, physical_rows, write_memory_bytes)| {
                let write_memory = write_memory.clone();
                let value = *overrides
                    .get(&(flag_id, fragment_id))
                    .expect("validated Cell Flag override must exist");
                async move {
                    let fragment = match value {
                        CellFlagFragmentValue::None => None,
                        CellFlagFragmentValue::All => Some(CellFlagFragment {
                            fragment_id,
                            physical_rows,
                            state: CellFlagFragmentState::All,
                        }),
                        CellFlagFragmentValue::Partial(bitmap) => {
                            let _memory = write_memory
                                .acquire_many_owned(cell_flag_write_memory_weight(
                                    write_memory_bytes,
                                ))
                                .await
                                .map_err(|_| {
                                    Error::internal(
                                        "Cell Flag write byte semaphore closed".to_string(),
                                    )
                                })?;
                            let state = current
                                .write_cell_flag_bitmap(
                                    write_store,
                                    flag_id,
                                    fragment_id,
                                    physical_rows,
                                    bitmap,
                                )
                                .await?;
                            Some(CellFlagFragment {
                                fragment_id,
                                physical_rows,
                                state,
                            })
                        }
                    };
                    Ok::<_, Error>((fragment_id, fragment))
                }
            })
            .buffer_unordered(io_parallelism)
            .try_collect::<Vec<_>>()
            .await?;
        for (fragment_id, fragment) in override_states {
            if let Some(fragment) = fragment {
                states.insert(fragment_id, fragment);
            } else {
                states.remove(&fragment_id);
            }
        }

        let mut flag_changes = Vec::new();
        for ((_, fragment_id), change) in pending_changes
            .iter()
            .filter(|((change_flag_id, _), _)| *change_flag_id == flag_id)
        {
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
            let write_memory_bytes =
                cell_flag_row_change_memory_bytes(states.get(fragment_id), change)?;
            flag_changes.push((*fragment_id, physical_rows, write_memory_bytes));
        }
        let changed_states = futures::stream::iter(flag_changes)
            .map(|(fragment_id, physical_rows, write_memory_bytes)| {
                let write_memory = write_memory.clone();
                let change = pending_changes
                    .get(&(flag_id, fragment_id))
                    .expect("validated Cell Flag row change must exist");
                let previous_state = states.get(&fragment_id);
                async move {
                    let _memory = write_memory
                        .acquire_many_owned(cell_flag_write_memory_weight(write_memory_bytes))
                        .await
                        .map_err(|_| {
                            Error::internal("Cell Flag write byte semaphore closed".to_string())
                        })?;
                    let mut bitmap =
                        materialize_fragment_bitmap(current, flag_id, fragment_id, previous_state)
                            .await?;
                    bitmap |= &change.set;
                    bitmap -= &change.clear;
                    let fragment = if bitmap.is_empty() {
                        None
                    } else if bitmap.len() == physical_rows {
                        Some(CellFlagFragment {
                            fragment_id,
                            physical_rows,
                            state: CellFlagFragmentState::All,
                        })
                    } else {
                        let state = current
                            .write_cell_flag_bitmap(
                                write_store,
                                flag_id,
                                fragment_id,
                                physical_rows,
                                &bitmap,
                            )
                            .await?;
                        Some(CellFlagFragment {
                            fragment_id,
                            physical_rows,
                            state,
                        })
                    };
                    Ok::<_, Error>((fragment_id, fragment))
                }
            })
            .buffer_unordered(io_parallelism)
            .try_collect::<Vec<_>>()
            .await?;
        for (fragment_id, fragment) in changed_states {
            if let Some(fragment) = fragment {
                states.insert(fragment_id, fragment);
            } else {
                states.remove(&fragment_id);
            }
        }

        if let Some(descriptor) = current
            .write_cell_flag_root(
                write_store,
                flag_id,
                CellFlagRoot {
                    fragments: states.into_values().collect(),
                },
                max_inline_bytes,
            )
            .await?
        {
            inline_manifest_budget.reserve(Some(&descriptor))?;
            descriptors.push(descriptor);
        }
    }

    descriptors.sort_by_key(|state| state.flag_id);
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
    use lance_index::IndexType;
    use lance_index::scalar::ScalarIndexParams;
    use lance_select::RowAddrTreeMap;

    use super::*;
    use crate::dataset::cleanup::CleanupPolicyBuilder;
    use crate::dataset::optimize::{
        CompactionOptions, CompactionTask, IgnoreRemap, TaskData, commit_compaction, compact_files,
    };
    use crate::dataset::transaction::{CellFlagRowChange, RewriteGroup, TransactionBuilder};
    use crate::dataset::write::merge_insert::{
        MergeInsertBuilder, WhenMatched, WhenNotMatched, WhenNotMatchedBySource,
    };
    use crate::dataset::{
        ColumnAlteration, CommitBuilder, DeleteBuilder, InsertBuilder, NewColumnTransform,
        UpdateBuilder, WriteMode, WriteParams,
    };
    use crate::index::DatasetIndexExt;
    use crate::utils::test::copy_test_data_to_tmp;

    const FLAG_NAME: &str = "lancedb.computed";

    #[test]
    fn sparse_fragment_selection_uses_sorted_root_entries() {
        let root = CellFlagRoot {
            fragments: (0..100)
                .map(|fragment_id| CellFlagFragment {
                    fragment_id,
                    physical_rows: 1,
                    state: CellFlagFragmentState::All,
                })
                .collect(),
        };
        let selected = HashSet::from([1, 99, 101]);

        let fragments = select_cell_flag_fragments(&root, Some(&selected));
        assert_eq!(
            fragments
                .iter()
                .map(|fragment| fragment.fragment_id)
                .collect::<HashSet<_>>(),
            HashSet::from([1, 99])
        );
    }

    #[test]
    fn inline_manifest_budget_reserves_unchanged_roots_before_rewrite() {
        let state = |flag_id, bytes| CellFlagState {
            flag_id,
            root: CellFlagFile {
                path: String::new(),
                size_bytes: 0,
                memory_size_bytes: bytes as u64,
                base_id: None,
                inline_bytes: Some(vec![0; bytes]),
            },
        };
        let first = state(0, 49);
        let second = state(1, 50);
        let mut budget = InlineCellFlagManifestBudget::from_states([&first, &second], 100).unwrap();

        // Replacing the first root has only 50 bytes available because the
        // unchanged second root must remain reserved. A 51-byte replacement
        // therefore has to be external rather than making the commit fail
        // when the second descriptor is visited later.
        assert_eq!(budget.release(Some(&first)).unwrap(), 50);
        budget.reserve(None).unwrap();
        assert_eq!(budget.release(Some(&second)).unwrap(), 100);
        budget.reserve(Some(&second)).unwrap();
        assert_eq!(budget.used_bytes, 50);
    }

    #[test]
    fn query_memory_budget_is_cumulative_across_flags() {
        let reserved = AtomicUsize::new(0);
        reserve_cell_flag_query_memory(&reserved, 40 * 1024 * 1024).unwrap();

        let error = reserve_cell_flag_query_memory(&reserved, 25 * 1024 * 1024).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("binding budget"));
        assert_eq!(reserved.load(Ordering::Relaxed), 40 * 1024 * 1024);
    }

    #[test]
    fn write_memory_budget_accounts_for_all_bitmap_encoder_buffers() {
        let roaring_bytes = 8 * 1024 * 1024;
        let bitset_bytes = 4 * 1024 * 1024;
        let compressed_roaring_bytes = zstd::zstd_safe::compress_bound(roaring_bytes);
        let compressed_bitset_bytes = zstd::zstd_safe::compress_bound(bitset_bytes);
        let estimate =
            cell_flag_bitmap_encoder_memory_bytes_for_sizes(roaring_bytes, bitset_bytes).unwrap();

        let adaptive_candidates = roaring_bytes
            + 2 * compressed_roaring_bytes
            + bitset_bytes
            + 2 * compressed_bitset_bytes
            + 2 * CELL_FLAG_BITMAP_COMPRESSED_LENGTH_BYTES
            + CELL_FLAG_BITMAP_ENCODER_FIXED_MEMORY_BYTES;
        let inline_query = MAX_INLINE_CELL_FLAG_BITMAP_BYTES
            + 2 * roaring_bytes
            + 2 * compressed_roaring_bytes
            + CELL_FLAG_BITMAP_COMPRESSED_LENGTH_BYTES
            + CELL_FLAG_BITMAP_ENCODING_HEADER_BYTES
            + CELL_FLAG_BITMAP_ENCODER_FIXED_MEMORY_BYTES;
        assert!(estimate >= adaptive_candidates);
        assert!(estimate >= inline_query);
        assert_eq!(
            cell_flag_write_memory_weight(estimate),
            estimate
                .div_ceil(CELL_FLAG_WRITE_MEMORY_PERMIT_BYTES)
                .min(CELL_FLAG_WRITE_MEMORY_PERMITS) as u32
        );
    }

    #[test]
    fn large_bitmap_encoder_reserves_the_complete_write_budget() {
        let estimate =
            cell_flag_bitmap_encoder_memory_bytes_for_sizes(16 * 1024 * 1024, 8 * 1024 * 1024)
                .unwrap();

        assert!(estimate > CELL_FLAG_WRITE_MEMORY_BUDGET_BYTES);
        assert_eq!(
            cell_flag_write_memory_weight(estimate),
            CELL_FLAG_WRITE_MEMORY_PERMITS as u32
        );
    }

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

    #[tokio::test]
    async fn external_cell_flag_read_rejects_actual_size_mismatch() -> Result<()> {
        let directory = TempStrDir::default();
        let dataset = dataset_with_rows(&directory, 2).await?;
        let relative = Path::from(CELL_FLAGS_DIR).join("size-mismatch.root");
        let full_path = Path::from_iter(dataset.base.parts().chain(relative.parts()));
        dataset.object_store.put(&full_path, b"12").await?;
        let file = CellFlagFile {
            path: relative.to_string(),
            size_bytes: 1,
            memory_size_bytes: 1,
            base_id: None,
            inline_bytes: None,
        };

        let error = read_cell_flag_bytes(&dataset.object_store, &full_path, &file)
            .await
            .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("has size 2, expected 1"));
        Ok(())
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
    async fn fragment_staging_rejects_cell_flag_read_dependencies() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 2).await?;
        dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;
        let fragment = dataset.get_fragment(0).expect("first fragment");

        let delete_error = fragment
            .clone()
            .delete(&format!("cell_flag(value, '{}')", FLAG_NAME))
            .await
            .unwrap_err();
        assert!(delete_error.to_string().contains("Fragment-level delete"));

        let add_columns_error = fragment
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![(
                    "copied_flag".to_string(),
                    format!("cell_flag(value, '{}')", FLAG_NAME),
                )]),
                None,
                None,
            )
            .await
            .unwrap_err();
        assert!(
            add_columns_error
                .to_string()
                .contains("Fragment-level add_columns")
        );
        Ok(())
    }

    #[tokio::test]
    async fn batch_registration_is_atomic_and_uses_one_version() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 2).await?;
        let initial_version = dataset.version().version;

        let definitions = dataset
            .register_cell_flags([
                ("value", "state_a", false),
                ("value", "state_b", true),
                ("value", "state_c", false),
                ("value", "state_d", true),
                ("value", "merge_state", false),
            ])
            .await?;

        assert_eq!(dataset.version().version, initial_version + 1);
        assert_eq!(dataset.manifest.cell_flag_states.len(), 2);
        assert!(dataset.manifest.cell_flag_states.iter().all(|state| {
            state.root.path.is_empty()
                && state.root.size_bytes == 0
                && state.root.inline_bytes.is_some()
        }));
        assert_eq!(
            definitions
                .iter()
                .map(|definition| definition.flag_id)
                .collect::<Vec<_>>(),
            (0..5).collect::<Vec<_>>()
        );
        assert_eq!(dataset.cell_flag_definitions(), definitions);
        assert_eq!(
            flagged_ids(&dataset, "value", "state_b").await?,
            (0..8).collect::<Vec<_>>()
        );
        assert_eq!(
            flagged_ids(&dataset, "value", "merge_state").await?,
            Vec::<i32>::new()
        );

        for invalid_registrations in [
            vec![("id", "duplicate", false), ("id", "duplicate", true)],
            vec![("id", "valid", false), ("unknown", "invalid", false)],
        ] {
            let version = dataset.version().version;
            let definitions_before = dataset.cell_flag_definitions().to_vec();
            assert!(
                dataset
                    .register_cell_flags(invalid_registrations)
                    .await
                    .is_err()
            );
            assert_eq!(dataset.version().version, version);
            assert_eq!(dataset.cell_flag_definitions(), definitions_before);
        }

        assert!(
            dataset
                .register_cell_flags(Vec::<(&str, &str, bool)>::new())
                .await
                .unwrap_err()
                .to_string()
                .contains("At least one")
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
        assert_eq!(
            dataset
                .count_rows(Some(format!("cell_flag(x, '{}') OR x < 0", FLAG_NAME)))
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
    async fn registration_rejects_unreadable_manifest_metadata_size() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 1).await?;
        let initial_version = dataset.version().version;
        let oversized_name = "x".repeat(6_300_000);

        let error = dataset
            .register_cell_flag("value", oversized_name, false)
            .await
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("Encoded Cell Flag manifest metadata")
        );
        assert_eq!(dataset.version().version, initial_version);
        let reopened = Dataset::open(directory.as_ref()).await?;
        assert_eq!(reopened.version().version, initial_version);
        assert!(reopened.cell_flag_definitions().is_empty());
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
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![(
                    "mirror".to_string(),
                    "false".to_string(),
                )]),
                None,
                None,
            )
            .await?;

        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id IN (1, 3)")?
            .set_cell_flag("value", FLAG_NAME, true)?
            .build()?
            .execute()
            .await?;
        let dataset = result.new_dataset.as_ref().clone();
        assert_eq!(flagged_ids(&dataset, "value", FLAG_NAME).await?, vec![1, 3]);

        let value_expression = format!("cell_flag(value, '{}')", FLAG_NAME);
        let result = UpdateBuilder::new(Arc::new(dataset))
            .set("mirror", &value_expression)?
            .build()?
            .execute()
            .await?;
        let mut dataset = result.new_dataset.as_ref().clone();
        assert_eq!(dataset.count_rows(Some("mirror".to_string())).await?, 2);
        assert_eq!(flagged_ids(&dataset, "value", FLAG_NAME).await?, vec![1, 3]);
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![(
                    "copied_flag".to_string(),
                    format!("cell_flag(value, '{}')", FLAG_NAME),
                )]),
                None,
                None,
            )
            .await?;
        assert_eq!(
            dataset.count_rows(Some("copied_flag".to_string())).await?,
            2
        );
        assert_eq!(
            dataset
                .read_transaction()
                .await?
                .expect("add_columns transaction")
                .cell_flag_transaction()?
                .expect("flag-reading add_columns carries a sidecar")
                .read_flag_ids,
            vec![0]
        );

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
        let cached_root_before = dataset
            .load_cell_flag_root_shared(definition.flag_id)
            .await?
            .expect("explicitly set rows create a root");

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
        assert!(
            dataset
                .read_transaction()
                .await?
                .expect("append transaction")
                .cell_flag_transaction()?
                .is_none()
        );
        assert_eq!(
            dataset.cell_flag_state(definition.flag_id),
            Some(&root_before_false_appends)
        );
        let cached_root_after = dataset
            .load_cell_flag_root_shared(definition.flag_id)
            .await?
            .expect("ordinary append preserves the root");
        assert!(Arc::ptr_eq(&cached_root_before, &cached_root_after));

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
    async fn cached_root_is_revalidated_against_each_snapshot() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 2).await?;
        let definition = dataset.register_cell_flag("value", FLAG_NAME, true).await?;
        let root = dataset
            .load_cell_flag_root_shared(definition.flag_id)
            .await?
            .expect("initial true state has a root");
        let root_fragment = root.fragments.first().expect("root has a fragment");
        let fragment_id = u32::try_from(root_fragment.fragment_id).unwrap();

        let mut mismatched_rows = dataset.clone();
        let manifest = Arc::make_mut(&mut mismatched_rows.manifest);
        let fragments = Arc::make_mut(&mut manifest.fragments);
        let fragment = fragments
            .iter_mut()
            .find(|fragment| fragment.id == root_fragment.fragment_id)
            .expect("root fragment exists in the source snapshot");
        fragment.physical_rows = Some(fragment.physical_rows.unwrap() + 1);
        let error = mismatched_rows
            .load_cell_flag_root_shared(definition.flag_id)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("records"));
        assert!(error.to_string().contains("manifest records"));

        let mut missing_fragment = dataset.clone();
        let mut fragment_bitmap = missing_fragment.fragment_bitmap.as_ref().clone();
        fragment_bitmap.remove(fragment_id);
        missing_fragment.fragment_bitmap = Arc::new(fragment_bitmap);
        let error = missing_fragment
            .load_cell_flag_root_shared(definition.flag_id)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("references unknown fragment"));
        Ok(())
    }

    #[tokio::test]
    async fn cached_inline_root_is_namespaced_by_flag_id() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 2).await?;
        let definitions = dataset
            .register_cell_flags([
                ("value", "first_namespace", false),
                ("value", "second_namespace", false),
            ])
            .await?;
        let fragment = dataset
            .manifest
            .fragments
            .first()
            .expect("dataset has one fragment");
        let fragment_id = fragment.id;
        let physical_rows = fragment_physical_rows(fragment)?;
        let root = CellFlagRoot {
            fragments: vec![CellFlagFragment {
                fragment_id,
                physical_rows,
                state: CellFlagFragmentState::Partial(CellFlagFile {
                    path: format!(
                        "_cell_flags/bitmaps/{}/{}/{}.rbm",
                        definitions[0].flag_id,
                        fragment_id,
                        Uuid::new_v4()
                    ),
                    size_bytes: 1,
                    memory_size_bytes: 1,
                    base_id: None,
                    inline_bytes: None,
                }),
            }],
        };
        let (inline_bytes, memory_size) = encode_cell_flag_root(&pb::CellFlagRoot::from(&root))?;
        let descriptor = CellFlagFile {
            path: String::new(),
            size_bytes: 0,
            memory_size_bytes: memory_size as u64,
            base_id: None,
            inline_bytes: Some(inline_bytes),
        };
        Arc::make_mut(&mut dataset.manifest).cell_flag_states = definitions
            .iter()
            .map(|definition| CellFlagState {
                flag_id: definition.flag_id,
                root: descriptor.clone(),
            })
            .collect();

        dataset
            .load_cell_flag_root_shared(definitions[0].flag_id)
            .await?
            .expect("first flag root is valid");
        let error = dataset
            .load_cell_flag_root_shared(definitions[1].flag_id)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("expected flag ID 1"));
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
    async fn overwrite_preserves_only_matching_field_identities() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 2).await?;
        let definition = dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;

        let reordered_schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, true),
            Field::new("id", DataType::Int32, false),
        ]));
        let reordered = RecordBatch::try_new(
            reordered_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![Some(20), None])),
                Arc::new(Int32Array::from(vec![2, 3])),
            ],
        )?;
        dataset = Dataset::write(
            RecordBatchIterator::new([Ok(reordered)], reordered_schema),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await?;

        assert_eq!(
            dataset.cell_flag_definitions(),
            std::slice::from_ref(&definition)
        );
        assert_eq!(
            dataset.schema().field("value").unwrap().id,
            definition.field_id
        );
        assert_ne!(
            dataset.schema().field("id").unwrap().id,
            definition.field_id
        );
        assert!(flagged_ids(&dataset, "value", FLAG_NAME).await?.is_empty());

        let replacement_schema = Arc::new(Schema::new(vec![
            Field::new("other_id", DataType::Int32, false),
            Field::new("unrelated", DataType::Int32, true),
        ]));
        let replacement = RecordBatch::try_new(
            replacement_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![4, 5])),
                Arc::new(Int32Array::from(vec![Some(40), None])),
            ],
        )?;
        dataset = Dataset::write(
            RecordBatchIterator::new([Ok(replacement)], replacement_schema),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await?;

        assert!(dataset.cell_flag_definitions().is_empty());
        let replacement_definition = dataset
            .register_cell_flag("unrelated", FLAG_NAME, false)
            .await?;
        assert!(replacement_definition.flag_id > definition.flag_id);
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
    async fn raw_merge_cannot_remove_a_tracked_field_without_a_transfer() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 2).await?;
        let definition = dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;

        let mut manifest = dataset.manifest.as_ref().clone();
        let replacement_field_id = manifest.max_field_id() + 1;
        manifest
            .schema
            .fields
            .iter_mut()
            .find(|field| field.id == definition.field_id)
            .unwrap()
            .id = replacement_field_id;
        let transaction = Transaction::new(
            dataset.manifest.version,
            Operation::Merge {
                fragments: manifest.fragments.to_vec(),
                schema: manifest.schema.clone(),
                preserves_nullability: true,
            },
            None,
        );

        let error = apply_cell_flag_transaction(
            &dataset,
            dataset.object_store.as_ref(),
            &mut manifest,
            &transaction,
        )
        .await
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("without an explicit field transfer or flag drop"),
            "got {error}"
        );
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

            let staged = DeleteBuilder::new(
                Arc::new(dataset.clone()),
                format!("id = 2 AND cell_flag(value, '{}')", FLAG_NAME),
            )
            .execute_uncommitted()
            .await?;
            assert_eq!(
                staged
                    .transaction
                    .cell_flag_transaction()?
                    .expect("flag-reading delete carries a sidecar")
                    .read_flag_ids,
                vec![0]
            );
            let mut commit = CommitBuilder::new(Arc::new(dataset));
            if let Some(affected_rows) = staged.affected_rows {
                commit = commit.with_affected_rows(affected_rows);
            }
            dataset = commit.execute(staged.transaction).await?;
            compact_files(&mut dataset, CompactionOptions::default(), None).await?;
            assert_eq!(
                flagged_ids(&dataset, "value", FLAG_NAME).await?,
                vec![1, 5, 7]
            );
        }
        Ok(())
    }

    #[test]
    fn compaction_precomputes_cross_fragment_live_prefixes() {
        let source_row_addresses = (0..1024_u32)
            .flat_map(|fragment_id| {
                [0, 2].map(move |offset| u64::from(RowAddress::new_from_parts(fragment_id, offset)))
            })
            .collect::<RoaringTreemap>();
        let layout = compacted_source_fragment_layout(&source_row_addresses).unwrap();

        assert_eq!(layout.len(), 1024);
        assert_eq!(layout.get(&0).map(|(prefix, _)| *prefix), Some(0));
        assert_eq!(layout.get(&1023).map(|(prefix, _)| *prefix), Some(2046));

        let &(live_prefix, live_offsets) = layout.get(&1023).unwrap();
        let mut output_bitmaps = vec![RoaringBitmap::new()];
        remap_compacted_cell_flag_offsets(
            live_offsets,
            live_prefix,
            &RoaringBitmap::from_iter([2]),
            &[2048],
            &mut output_bitmaps,
        )
        .unwrap();
        assert_eq!(output_bitmaps, vec![RoaringBitmap::from_iter([2047])]);
    }

    #[tokio::test]
    async fn sparse_compaction_remaps_multiple_flags_across_output_fragments() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 4).await?;
        let computed = dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;
        let reviewed = dataset
            .register_cell_flag("value", "lancedb.reviewed", false)
            .await?;

        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id IN (0, 3, 5, 7)")?
            .set_cell_flag("value", FLAG_NAME, true)?
            .build()?
            .execute()
            .await?;
        let result = UpdateBuilder::new(result.new_dataset)
            .update_where("id IN (1, 2, 6)")?
            .set_cell_flag("value", "lancedb.reviewed", true)?
            .build()?
            .execute()
            .await?;
        let dataset = result.new_dataset.as_ref();
        let source_fragments = dataset.manifest.fragments.as_ref();
        assert_eq!(source_fragments.len(), 2);

        // Rows 2 and 4 are absent from this compaction result. The remaining
        // source addresses retain their original order across both fragments.
        let source_row_addresses = [
            (source_fragments[0].id, 0),
            (source_fragments[0].id, 1),
            (source_fragments[0].id, 3),
            (source_fragments[1].id, 1),
            (source_fragments[1].id, 2),
            (source_fragments[1].id, 3),
        ]
        .into_iter()
        .map(|(fragment_id, offset)| {
            u64::from(RowAddress::new_from_parts(fragment_id as u32, offset))
        })
        .collect::<RoaringTreemap>();

        let mut output_fragments = source_fragments.to_vec();
        output_fragments[0].id = 100;
        output_fragments[0].physical_rows = Some(2);
        output_fragments[0].files[0].path = "compacted-0.lance".to_string();
        output_fragments[1].id = 101;
        output_fragments[1].physical_rows = Some(4);
        output_fragments[1].files[0].path = "compacted-1.lance".to_string();

        let states = dataset
            .cell_flag_states_for_compacted_rows(&output_fragments, &source_row_addresses)
            .await?;
        let state = |flag_id, fragment_path: &str| {
            &states
                .iter()
                .find(|state| state.flag_id == flag_id && state.fragment_path == fragment_path)
                .unwrap()
                .state
        };

        assert_eq!(
            state(computed.flag_id, "compacted-0.lance"),
            &CellFlagFragmentValue::Partial(RoaringBitmap::from_iter([0]))
        );
        assert_eq!(
            state(computed.flag_id, "compacted-1.lance"),
            &CellFlagFragmentValue::Partial(RoaringBitmap::from_iter([0, 1, 3]))
        );
        assert_eq!(
            state(reviewed.flag_id, "compacted-0.lance"),
            &CellFlagFragmentValue::Partial(RoaringBitmap::from_iter([1]))
        );
        assert_eq!(
            state(reviewed.flag_id, "compacted-1.lance"),
            &CellFlagFragmentValue::Partial(RoaringBitmap::from_iter([2]))
        );
        assert_eq!(states.len(), 4);
        Ok(())
    }

    #[tokio::test]
    async fn disjoint_compaction_reuses_cell_flag_root() -> Result<()> {
        for enable_stable_row_ids in [false, true] {
            let directory = TempStrDir::default();
            let mut dataset =
                dataset_with_rows_and_stable_ids(&directory, 2, enable_stable_row_ids).await?;
            dataset
                .register_cell_flag("value", FLAG_NAME, false)
                .await?;
            let result = UpdateBuilder::new(Arc::new(dataset))
                .update_where("id = 0")?
                .set_cell_flag("value", FLAG_NAME, true)?
                .build()?
                .execute()
                .await?;
            let mut dataset = result.new_dataset.as_ref().clone();
            let root_before = dataset.manifest.cell_flag_states[0].root.clone();
            let options = CompactionOptions::default();
            let task = CompactionTask {
                task: TaskData {
                    fragments: dataset.fragments()[1..].to_vec(),
                },
                read_version: dataset.version().version,
                options: options.clone(),
            };
            let completed = task.execute(&dataset).await?;

            commit_compaction(
                &mut dataset,
                vec![completed],
                Arc::new(IgnoreRemap::default()),
                &options,
            )
            .await?;

            assert_eq!(dataset.manifest.cell_flag_states[0].root, root_before);
            assert_eq!(flagged_ids(&dataset, "value", FLAG_NAME).await?, vec![0]);
        }
        Ok(())
    }

    #[tokio::test]
    async fn rewrite_drops_flag_state_for_a_group_without_output() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 2).await?;
        dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 0")?
            .set_cell_flag("value", FLAG_NAME, true)?
            .build()?
            .execute()
            .await?;
        let dataset = result.new_dataset.as_ref().clone();
        let removed = dataset.fragments()[0].clone();
        let rewritten = dataset.fragments()[1].clone();
        let mut replacement = rewritten.clone();
        replacement.id = 0;
        let transaction = Transaction::new(
            dataset.manifest.version,
            Operation::Rewrite {
                groups: vec![
                    RewriteGroup {
                        old_fragments: vec![removed],
                        new_fragments: Vec::new(),
                    },
                    RewriteGroup {
                        old_fragments: vec![rewritten],
                        new_fragments: vec![replacement],
                    },
                ],
                rewritten_indices: Vec::new(),
                frag_reuse_index: None,
            },
            None,
        )
        .with_cell_flag_transaction_for_dataset(CellFlagTransaction::default(), &dataset)?;

        CommitBuilder::new(Arc::new(dataset))
            .execute(transaction)
            .await?;
        let reopened = Dataset::open(directory.as_ref()).await?;
        assert!(flagged_ids(&reopened, "value", FLAG_NAME).await?.is_empty());
        let definition = &reopened.cell_flag_definitions()[0];
        if let Some(root) = reopened.load_cell_flag_root(definition.flag_id).await? {
            assert!(
                root.fragments
                    .iter()
                    .all(|fragment| fragment.fragment_id != 0)
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
                .set_matched_cell_flag("value", FLAG_NAME, false)?;
            assert!(
                builder
                    .set_matched_cell_flag("value", FLAG_NAME, true)
                    .is_err()
            );
            builder.set_inserted_cell_flag("value", FLAG_NAME, true)?;
            assert!(
                builder
                    .set_inserted_cell_flag("value", FLAG_NAME, false)
                    .is_err()
            );
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

            let mut indexed_dataset = dataset.as_ref().clone();
            indexed_dataset
                .create_index(
                    &["id"],
                    IndexType::Scalar,
                    None,
                    &ScalarIndexParams::default(),
                    false,
                )
                .await?;
            let dataset = Arc::new(indexed_dataset);

            let conditional_source = RecordBatch::try_new(
                source_schema.clone(),
                vec![
                    Arc::new(Int32Array::from(vec![1, 8])),
                    Arc::new(Int32Array::from(vec![111, 888])),
                ],
            )?;
            let mut conditional_builder =
                MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])?;
            conditional_builder
                .when_matched(WhenMatched::UpdateIf(format!(
                    "cell_flag(target.value, '{}')",
                    FLAG_NAME
                )))
                .when_not_matched(WhenNotMatched::DoNothing);
            let (dataset, conditional_stats) = conditional_builder
                .try_build()?
                .execute_reader(Box::new(RecordBatchIterator::new(
                    [Ok(conditional_source)],
                    source_schema.clone(),
                )))
                .await?;
            assert_eq!(conditional_stats.num_updated_rows, 1);
            assert_eq!(flagged_ids(&dataset, "value", FLAG_NAME).await?, vec![8]);

            let mut scanner = dataset.scan();
            scanner.with_row_id();
            let row_id_source = scanner.try_into_batch().await?;
            let row_id_source_schema = row_id_source.schema();
            let mut row_id_builder =
                MergeInsertBuilder::try_new(dataset, vec![lance_core::ROW_ID.to_string()])?;
            row_id_builder
                .when_matched(WhenMatched::UpdateIf(format!(
                    "cell_flag(target.value, '{}')",
                    FLAG_NAME
                )))
                .when_not_matched(WhenNotMatched::DoNothing);
            let (dataset, row_id_stats) = row_id_builder
                .try_build()?
                .execute_reader(Box::new(RecordBatchIterator::new(
                    [Ok(row_id_source)],
                    row_id_source_schema,
                )))
                .await?;
            assert_eq!(row_id_stats.num_updated_rows, 1);
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

            let delete_if_source = RecordBatch::try_new(
                source_schema.clone(),
                vec![
                    Arc::new(Int32Array::from(vec![8])),
                    Arc::new(Int32Array::from(vec![808])),
                ],
            )?;
            let delete_if = WhenNotMatchedBySource::delete_if(
                dataset.as_ref(),
                &format!("cell_flag(value, '{}')", FLAG_NAME),
            )?;
            let mut delete_if_builder =
                MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])?;
            delete_if_builder
                .when_not_matched(WhenNotMatched::DoNothing)
                .when_not_matched_by_source(delete_if);
            let (dataset, delete_if_stats) = delete_if_builder
                .try_build()?
                .execute_reader(Box::new(RecordBatchIterator::new(
                    [Ok(delete_if_source)],
                    source_schema.clone(),
                )))
                .await?;
            assert_eq!(delete_if_stats.num_deleted_rows, 1);
            assert_eq!(flagged_ids(&dataset, "value", FLAG_NAME).await?, vec![8]);
        }
        Ok(())
    }

    #[tokio::test]
    async fn indexed_merge_records_and_conflicts_on_cell_flag_reads() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 2).await?;
        let definition = dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 1")?
            .set_cell_flag("value", FLAG_NAME, true)?
            .build()?
            .execute()
            .await?;
        let mut dataset = result.new_dataset.as_ref().clone();
        dataset
            .create_index(
                &["id"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await?;
        let dataset = Arc::new(dataset);

        let source_schema: Arc<Schema> = Arc::new(dataset.schema().into());
        let source = RecordBatch::try_new(
            source_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Int32Array::from(vec![111])),
            ],
        )?;
        let mut builder = MergeInsertBuilder::try_new(dataset.clone(), vec!["id".to_string()])?;
        builder
            .when_matched(WhenMatched::UpdateIf(format!(
                "cell_flag(target.value, '{}')",
                FLAG_NAME
            )))
            .when_not_matched(WhenNotMatched::DoNothing);
        let staged = builder
            .try_build()?
            .execute_uncommitted(RecordBatchIterator::new([Ok(source)], source_schema))
            .await?;
        assert_eq!(
            staged
                .transaction
                .cell_flag_transaction()?
                .expect("flag-reading merge carries a sidecar")
                .read_flag_ids,
            vec![definition.flag_id]
        );

        let concurrent = UpdateBuilder::new(dataset)
            .update_where("id = 7")?
            .set_cell_flag("value", FLAG_NAME, true)?
            .build()?
            .execute()
            .await?;
        let mut commit = CommitBuilder::new(concurrent.new_dataset);
        if let Some(affected_rows) = staged.affected_rows {
            commit = commit.with_affected_rows(affected_rows);
        }
        let error = commit.execute(staged.transaction).await.unwrap_err();
        assert!(matches!(error, Error::RetryableCommitConflict { .. }));
        Ok(())
    }

    #[tokio::test]
    async fn pure_insert_merge_does_not_load_existing_cell_flag_roots() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 2).await?;
        let definition = dataset.register_cell_flag("value", FLAG_NAME, true).await?;
        let dataset = Dataset::open(directory.as_ref()).await?;
        let descriptor = dataset
            .cell_flag_state(definition.flag_id)
            .unwrap()
            .root
            .clone();
        let root_key = CellFlagRootKey {
            source_uri: dataset.cell_flag_source_uri(&descriptor)?.to_string(),
            path: descriptor.path.clone(),
            size_bytes: descriptor.size_bytes,
            memory_size_bytes: descriptor.memory_size_bytes,
            flag_id: definition.flag_id,
            inline_hash: descriptor
                .inline_bytes
                .as_deref()
                .map(blake3::hash)
                .map(|hash| *hash.as_bytes()),
        };
        dataset.metadata_cache.clear().await;
        assert!(
            dataset
                .metadata_cache
                .get_with_key(&root_key)
                .await
                .is_none()
        );

        let source_schema: Arc<Schema> = Arc::new(dataset.schema().into());
        let source = RecordBatch::try_new(
            source_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![100, 101])),
                Arc::new(Int32Array::from(vec![1000, 1010])),
            ],
        )?;
        let mut builder = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])?;
        builder
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll);
        let (dataset, stats) = builder
            .try_build()?
            .execute_reader(Box::new(RecordBatchIterator::new(
                [Ok(source)],
                source_schema,
            )))
            .await?;
        assert_eq!(stats.num_updated_rows, 0);
        assert_eq!(stats.num_inserted_rows, 2);
        assert_eq!(
            dataset.cell_flag_state(definition.flag_id).unwrap().root,
            descriptor
        );
        assert!(
            dataset
                .metadata_cache
                .get_with_key(&root_key)
                .await
                .is_none()
        );
        assert_eq!(
            flagged_ids(&dataset, "value", FLAG_NAME).await?,
            (0..8).collect::<Vec<_>>()
        );
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
    async fn flag_only_update_conflicts_with_same_row_delete() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 8).await?;
        dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;
        let mut delete_writer = dataset.clone();
        let flag_writer = Arc::new(dataset);

        delete_writer.delete("id = 1").await?;
        let error = UpdateBuilder::new(flag_writer)
            .update_where("id = 1")?
            .set_cell_flag("value", FLAG_NAME, true)?
            .conflict_retries(0)
            .build()?
            .execute()
            .await
            .unwrap_err();

        assert!(matches!(error, Error::TooMuchWriteContention { .. }));
        let current = Dataset::open(directory.as_ref()).await?;
        assert_eq!(current.count_rows(None).await?, 7);
        assert!(flagged_ids(&current, "value", FLAG_NAME).await?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn commit_rejects_mismatched_cell_flag_affected_rows() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 8).await?;
        let definition = dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;
        let operation = Operation::Update {
            removed_fragment_ids: Vec::new(),
            updated_fragments: vec![dataset.fragments()[0].clone()],
            new_fragments: Vec::new(),
            fields_modified: Vec::new(),
            compacted_sstables: Vec::new(),
            fields_for_preserving_frag_bitmap: Vec::new(),
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_fragment_offsets: None,
        };
        let transaction = Transaction::new(dataset.manifest.version, operation, None)
            .with_cell_flag_transaction_for_dataset(
                CellFlagTransaction {
                    row_changes: vec![CellFlagRowChange {
                        flag_id: definition.flag_id,
                        value: true,
                        row_addresses: roaring::RoaringTreemap::from_iter([0_u64]),
                    }],
                    affected_rows: Some(RowAddrTreeMap::from_iter([0_u64])),
                    ..Default::default()
                },
                &dataset,
            )?;

        let error = CommitBuilder::new(Arc::new(dataset))
            .with_affected_rows(RowAddrTreeMap::from_iter([1_u64]))
            .execute(transaction)
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("do not match the Cell Flag transaction")
        );
        Ok(())
    }

    #[tokio::test]
    async fn carrier_preserving_transaction_edits_keep_cell_flag_changes() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 8).await?;
        let definition = dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;
        let operation = Operation::Update {
            removed_fragment_ids: Vec::new(),
            updated_fragments: Vec::new(),
            new_fragments: Vec::new(),
            fields_modified: Vec::new(),
            compacted_sstables: Vec::new(),
            fields_for_preserving_frag_bitmap: Vec::new(),
            update_mode: None,
            inserted_rows_filter: None,
            updated_fragment_offsets: None,
        };
        let transaction = Transaction::new(dataset.manifest.version, operation, None)
            .with_cell_flag_transaction_for_dataset(
                CellFlagTransaction {
                    row_changes: vec![CellFlagRowChange {
                        flag_id: definition.flag_id,
                        value: true,
                        row_addresses: roaring::RoaringTreemap::from_iter([0_u64]),
                    }],
                    ..Default::default()
                },
                &dataset,
            )?
            .with_application_transaction_properties(Some(Arc::new(HashMap::from_iter([(
                "application".to_string(),
                "value".to_string(),
            )]))))?;
        let transaction = transaction.with_uuid("ea5b9838-d30b-4b80-9938-403f96af3b24")?;
        assert_eq!(transaction.uuid, "ea5b9838-d30b-4b80-9938-403f96af3b24");
        let transaction = transaction.regenerate_uuid()?;
        assert_ne!(transaction.uuid, "ea5b9838-d30b-4b80-9938-403f96af3b24");
        assert_eq!(
            transaction.application_transaction_properties(),
            Some(HashMap::from_iter([(
                "application".to_string(),
                "value".to_string(),
            )]))
        );
        assert!(transaction.cell_flag_transaction()?.is_some());

        let committed = CommitBuilder::new(Arc::new(dataset))
            .execute(transaction)
            .await?;
        assert_eq!(flagged_ids(&committed, "value", FLAG_NAME).await?, vec![0]);
        Ok(())
    }

    #[tokio::test]
    async fn ordinary_transaction_can_use_the_internal_sidecar_uuid_prefix() -> Result<()> {
        let directory = TempStrDir::default();
        let mut dataset = dataset_with_rows(&directory, 8).await?;
        dataset
            .register_cell_flag("value", FLAG_NAME, false)
            .await?;
        let operation = Operation::Project {
            schema: dataset.schema().clone(),
            preserves_nullability: true,
        };
        let transaction = TransactionBuilder::new(dataset.manifest.version, operation)
            .uuid("4c434601-0000-8000-8000-000000000000".to_string())
            .build();

        let committed = CommitBuilder::new(Arc::new(dataset))
            .execute(transaction)
            .await?;
        assert_eq!(committed.cell_flag_definitions().len(), 1);
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
        let source_dataset_id = dataset.manifest.cell_flag_dataset_id.as_ref().unwrap();
        let shallow_dataset_id = shallow.manifest.cell_flag_dataset_id.as_ref().unwrap();
        let deep_dataset_id = deep.manifest.cell_flag_dataset_id.as_ref().unwrap();
        let mut timestamp_changed_manifest = shallow.manifest.as_ref().clone();
        timestamp_changed_manifest.timestamp_nanos += 1;
        assert_ne!(source_dataset_id, shallow_dataset_id);
        assert_ne!(source_dataset_id, deep_dataset_id);
        assert_ne!(shallow_dataset_id, deep_dataset_id);
        assert_eq!(
            cell_flag_manifest_identity(&timestamp_changed_manifest),
            cell_flag_manifest_identity(&shallow.manifest)
        );
        assert_eq!(
            shallow_dataset_id,
            &cell_flag_manifest_identity(&shallow.manifest)
        );
        assert_eq!(
            deep_dataset_id,
            &cell_flag_manifest_identity(&deep.manifest)
        );
        assert_eq!(
            Uuid::parse_str(shallow_dataset_id)
                .unwrap()
                .get_version_num(),
            8
        );
        assert_eq!(
            Uuid::parse_str(deep_dataset_id).unwrap().get_version_num(),
            8
        );
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
            shallow
                .cell_flag_state(definition.flag_id)
                .unwrap()
                .root
                .path
                .is_empty()
        );
        assert!(
            deep.cell_flag_state(definition.flag_id)
                .unwrap()
                .root
                .base_id
                .is_none()
        );
        assert!(
            deep.cell_flag_state(definition.flag_id)
                .unwrap()
                .root
                .path
                .is_empty()
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
                .all(|file| !file.path.contains(CELL_FLAGS_DIR))
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
