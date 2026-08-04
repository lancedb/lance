// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};
use lance_core::Result;
use lance_core::datatypes::{Field, NullabilityComparison, Schema, SchemaCompareOptions};
use lance_encoding::decoder::DecoderPlugins;
use lance_file::reader::FileReader;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_table::format::overlay::{
    DataOverlayFile, OverlayCoverage, TOMBSTONE_FIELD_ID, tombstone_overlay_fields,
};
use lance_table::format::{DataFile, Fragment};
use roaring::RoaringBitmap;

use super::CommitBuilder;
use super::column_write_error::ColumnWriteError;
use super::retry::{RetryConfig, RetryExecutor, execute_with_retry};
use crate::Dataset;
use crate::dataset::COLUMN_WRITE_READ_VERSION_METADATA_KEY;
use crate::dataset::transaction::{DataReplacementGroup, Operation, Transaction};

/// See [`Dataset::commit_column_writes`].
///
/// The staged files are the source of truth: each file's footer carries the
/// column schema it was written with and the dataset version it was prepared
/// against, and the commit is validated against those records.
pub async fn commit_column_writes(
    ds: &mut Dataset,
    replacements: Vec<DataReplacementGroup>,
    transaction_properties: Option<Arc<HashMap<String, String>>>,
) -> Result<()> {
    if replacements.is_empty() {
        return Err(ColumnWriteError::NoReplacements.into());
    }

    let staged_schemas = read_staged_schemas(ds, &replacements).await?;
    let plan = plan_column_writes(ds, &replacements, &staged_schemas).await?;

    let dataset = Arc::new(ds.clone());
    let job = CommitColumnWritesJob {
        dataset: dataset.clone(),
        replacements: Arc::new(replacements),
        plan: Arc::new(plan),
        transaction_properties,
    };
    let new_dataset = execute_with_retry(job, dataset, RetryConfig::default()).await?;
    *ds = Arc::try_unwrap(new_dataset).unwrap_or_else(|arc| (*arc).clone());
    Ok(())
}

/// Read every staged file's footer schema, in `replacements` order.
async fn read_staged_schemas(
    ds: &Dataset,
    replacements: &[DataReplacementGroup],
) -> Result<Vec<Schema>> {
    let scheduler = ScanScheduler::new(
        ds.object_store.clone(),
        SchedulerConfig::max_bandwidth(&ds.object_store),
    );
    // A column write stages one file per fragment, so this fans out over the
    // whole dataset; `buffered` keeps the schemas aligned with `replacements`.
    futures::stream::iter(replacements)
        .map(|DataReplacementGroup(_, file)| read_staged_schema(ds, &scheduler, file))
        .buffered(ds.object_store.io_parallelism())
        .try_collect()
        .await
}

/// Read one staged file's footer schema.
async fn read_staged_schema(
    ds: &Dataset,
    scheduler: &Arc<ScanScheduler>,
    file: &DataFile,
) -> Result<Schema> {
    let path = ds.data_dir().join(file.path.as_str());
    let file_scheduler = scheduler.open_file(&path, &file.file_size_bytes).await?;
    let reader = FileReader::try_open(
        file_scheduler,
        None,
        Arc::<DecoderPlugins>::default(),
        &ds.metadata_cache.file_metadata_cache(&path),
        ds.file_reader_options.clone().unwrap_or_default(),
    )
    .await?;
    Ok(reader.schema().as_ref().clone())
}

/// The staged columns' union schema, which of them already existed at
/// prepare time, and which top-level columns each fragment covers.
struct ColumnWritePlan {
    new_columns: Vec<Field>,
    preexisting_field_ids: HashSet<i32>,
    covered_by_fragment: HashMap<u64, HashSet<i32>>,
    /// Each staged field's complete prepare-time coverage identity on its
    /// target fragment. A replacement is only current while that exact
    /// state holds, absence included.
    ///
    /// Row count is deliberately not part of this identity: a fragment id's
    /// physical row count is immutable, so staged files sized at prepare time
    /// still fit at commit time. Operations that change a fragment's row count
    /// give it a new id (compaction, merge insert), which the orphan check
    /// catches instead, and `merge_fragments_valid` rejects any Merge that
    /// tries to resize a fragment in place.
    prepare_backing: HashMap<u64, Vec<(i32, FieldBacking)>>,
}

async fn plan_column_writes(
    ds: &Dataset,
    replacements: &[DataReplacementGroup],
    staged_schemas: &[Schema],
) -> Result<ColumnWritePlan> {
    // All staged files must be prepared against one dataset version.
    let mut read_version: Option<u64> = None;
    for (DataReplacementGroup(_, file), schema) in replacements.iter().zip(staged_schemas) {
        let stamped = schema
            .metadata
            .get(COLUMN_WRITE_READ_VERSION_METADATA_KEY)
            .and_then(|stamp| stamp.parse::<u64>().ok())
            .ok_or_else(|| ColumnWriteError::NotStaged {
                path: file.path.clone(),
            })?;
        match read_version {
            None => read_version = Some(stamped),
            Some(prev) if prev == stamped => {}
            Some(prev) => {
                return Err(ColumnWriteError::MixedReadVersions {
                    first: prev,
                    second: stamped,
                }
                .into());
            }
        }
    }
    let read_version = read_version.expect("replacements are non-empty");

    // Union the staged top-level columns; files sharing a field id must agree
    // on its full recursive schema.
    let mut new_columns: Vec<Field> = Vec::new();
    for schema in staged_schemas {
        for field in &schema.fields {
            match new_columns.iter().find(|f| f.id == field.id) {
                None => new_columns.push(field.clone()),
                Some(existing) if fields_match(existing, field) => {}
                Some(existing) => {
                    return Err(ColumnWriteError::StagedSchemasDisagree {
                        field_id: field.id,
                        name: existing.name.clone(),
                        data_type: existing.data_type().to_string(),
                        other_name: field.name.clone(),
                        other_data_type: field.data_type().to_string(),
                    }
                    .into());
                }
            }
        }
    }

    // Recorded field ids must come from the file's own staged schema, and a
    // fragment's files must cover disjoint ids.
    let mut covered_by_fragment: HashMap<u64, HashSet<i32>> = HashMap::new();
    let mut recorded_by_fragment: HashMap<u64, HashSet<i32>> = HashMap::new();
    for (DataReplacementGroup(frag_id, file), schema) in replacements.iter().zip(staged_schemas) {
        let mut subtree_ids = HashSet::new();
        for field in &schema.fields {
            collect_field_ids(field, &mut subtree_ids);
        }
        if file.fields.is_empty() {
            return Err(ColumnWriteError::NoStagedFieldIds {
                path: file.path.clone(),
            }
            .into());
        }
        let recorded = recorded_by_fragment.entry(*frag_id).or_default();
        for field_id in file.fields.iter() {
            if !subtree_ids.contains(field_id) {
                return Err(ColumnWriteError::StagedFieldNotInSchema {
                    path: file.path.clone(),
                    field_id: *field_id,
                }
                .into());
            }
            if !recorded.insert(*field_id) {
                return Err(ColumnWriteError::DuplicateFieldCoverage {
                    fragment_id: *frag_id,
                    field_id: *field_id,
                }
                .into());
            }
        }
        covered_by_fragment
            .entry(*frag_id)
            .or_default()
            .extend(schema.fields.iter().map(|field| field.id));
    }

    // Classify each staged column against the schema it was prepared under.
    // Ids that were already this exact field are incremental rewrites; ids
    // that were absent must still be unclaimed at commit time.
    let prepare_dataset = ds
        .checkout_version(read_version)
        .await
        .map_err(|_| ColumnWriteError::ReadVersionUnavailable { read_version })?;
    let prepare_schema = prepare_dataset.schema();
    let mut preexisting_field_ids = HashSet::new();
    for field in &new_columns {
        match prepare_schema.field_by_id(field.id) {
            None => {}
            Some(existing) if fields_match(existing, field) => {
                preexisting_field_ids.insert(field.id);
            }
            Some(existing) => {
                return Err(ColumnWriteError::FieldContractConflict {
                    field_id: field.id,
                    name: field.name.clone(),
                    data_type: field.data_type().to_string(),
                    other_name: existing.name.clone(),
                    other_data_type: existing.data_type().to_string(),
                    read_version,
                }
                .into());
            }
        }
    }

    // Record which file backs each rewritten field at prepare time; a
    // concurrent rewrite of the same field changes that backing, and the
    // staged bytes must then be recomputed rather than replayed.
    let prepare_fragments: HashMap<u64, Fragment> = prepare_dataset
        .get_fragments()
        .into_iter()
        .map(|frag| (frag.id() as u64, frag.metadata().clone()))
        .collect();
    let mut prepare_backing: HashMap<u64, Vec<(i32, FieldBacking)>> = HashMap::new();
    for DataReplacementGroup(frag_id, file) in replacements {
        let Some(prepare_fragment) = prepare_fragments.get(frag_id) else {
            return Err(ColumnWriteError::FragmentNotInReadVersion {
                fragment_id: *frag_id,
                read_version,
            }
            .into());
        };
        let bindings = prepare_backing.entry(*frag_id).or_default();
        for field_id in file.fields.iter() {
            bindings.push((*field_id, field_backing(prepare_fragment, *field_id)));
        }
    }

    Ok(ColumnWritePlan {
        new_columns,
        preexisting_field_ids,
        covered_by_fragment,
        prepare_backing,
    })
}

/// Full recursive field-contract comparison: names, types, nullability, and
/// field ids, at every level.
fn fields_match(a: &Field, b: &Field) -> bool {
    let a_schema = Schema {
        fields: vec![a.clone()],
        metadata: HashMap::new(),
    };
    let b_schema = Schema {
        fields: vec![b.clone()],
        metadata: HashMap::new(),
    };
    let options = SchemaCompareOptions {
        compare_field_ids: true,
        compare_nullability: NullabilityComparison::Strict,
        ..Default::default()
    };
    a_schema.check_compatible(&b_schema, &options).is_ok()
}

fn collect_field_ids(field: &Field, ids: &mut HashSet<i32>) {
    ids.insert(field.id);
    for child in &field.children {
        collect_field_ids(child, ids);
    }
}

#[derive(Clone)]
struct CommitColumnWritesJob {
    dataset: Arc<Dataset>,
    replacements: Arc<Vec<DataReplacementGroup>>,
    plan: Arc<ColumnWritePlan>,
    transaction_properties: Option<Arc<HashMap<String, String>>>,
}

impl RetryExecutor for CommitColumnWritesJob {
    type Data = Transaction;
    type Result = Arc<Dataset>;

    /// Build the Merge transaction against the job's current dataset version.
    /// Re-run on each retry so a concurrent commit's data files are preserved.
    async fn execute_impl(&self) -> Result<Self::Data> {
        let mut replacement_map: HashMap<u64, Vec<&DataFile>> = HashMap::new();
        for DataReplacementGroup(frag_id, file) in self.replacements.iter() {
            replacement_map.entry(*frag_id).or_default().push(file);
        }

        let mut schema = self.dataset.schema().clone();
        for field in &self.plan.new_columns {
            let was_preexisting = self.plan.preexisting_field_ids.contains(&field.id);
            match schema.field_by_id(field.id) {
                None if !was_preexisting => schema.fields.push(field.clone()),
                // Rewrite of a column that existed when the write was
                // prepared: the occupant must still match the full contract.
                Some(existing) if was_preexisting && fields_match(existing, field) => {}
                // The schema moved underneath the prepared write (a
                // formerly-new id claimed concurrently, a rewrite target
                // changed, or a field dropped). Staged files cannot be
                // renumbered; replaying them would clobber the winner.
                _ => {
                    return Err(ColumnWriteError::StaleSchema {
                        field_id: field.id,
                        name: field.name.clone(),
                        data_type: field.data_type().to_string(),
                    }
                    .into());
                }
            }
        }

        // A new non-nullable column must cover every live fragment; an
        // uncovered one (e.g. appended concurrently) would synthesize
        // invalid nulls. Uncovered nullable columns read as null.
        let live_fragments = self.dataset.get_fragments();
        for field in &self.plan.new_columns {
            if self.plan.preexisting_field_ids.contains(&field.id) || field.nullable {
                continue;
            }
            for frag in &live_fragments {
                let frag_id = frag.id() as u64;
                let covered = self
                    .plan
                    .covered_by_fragment
                    .get(&frag_id)
                    .is_some_and(|ids| ids.contains(&field.id));
                if !covered {
                    return Err(ColumnWriteError::UncoveredNonNullableColumn {
                        name: field.name.clone(),
                        fragment_id: frag_id,
                    }
                    .into());
                }
            }
        }

        let mut orphaned: HashSet<u64> = replacement_map.keys().copied().collect();
        let mut fragments = Vec::with_capacity(live_fragments.len());
        for frag in live_fragments {
            let frag_id = frag.id() as u64;
            let mut metadata = frag.metadata().clone();
            if let Some(data_files) = replacement_map.get(&frag_id) {
                orphaned.remove(&frag_id);
                // A rewritten field must still be backed by the file it had
                // at prepare time; a different backing means a concurrent
                // rewrite won and the staged bytes are stale.
                if let Some(bindings) = self.plan.prepare_backing.get(&frag_id) {
                    for (field_id, prepare_state) in bindings {
                        if field_backing(&metadata, *field_id) != *prepare_state {
                            return Err(ColumnWriteError::StaleFragmentData {
                                fragment_id: frag_id,
                                field_id: *field_id,
                            }
                            .into());
                        }
                    }
                }
                for data_file in data_files {
                    replace_column_coverage(&mut metadata, data_file);
                }
            }
            fragments.push(metadata);
        }

        // An orphaned replacement (its fragment was rewritten, e.g. by
        // compaction) would otherwise be dropped silently, committing a
        // column with missing (null) data.
        if !orphaned.is_empty() {
            let mut ids: Vec<u64> = orphaned.into_iter().collect();
            ids.sort_unstable();
            return Err(ColumnWriteError::OrphanedReplacements { fragment_ids: ids }.into());
        }

        let operation = Operation::Merge { fragments, schema };
        let mut transaction = Transaction::new(self.dataset.manifest.version, operation, None);
        transaction.transaction_properties = self.transaction_properties.clone();
        Ok(transaction)
    }

    async fn commit(&self, dataset: Arc<Dataset>, transaction: Self::Data) -> Result<Self::Result> {
        CommitBuilder::new(dataset)
            .execute(transaction)
            .await
            .map(Arc::new)
    }

    fn update_dataset(&mut self, dataset: Arc<Dataset>) {
        self.dataset = dataset;
    }
}

/// A field's complete coverage identity on a fragment: the base data file
/// backing it (None: uncovered) and the overlays covering it, in order.
#[derive(Clone, PartialEq)]
struct FieldBacking {
    base: Option<DataFileWitness>,
    overlays: Vec<OverlayBacking>,
}

/// A data file's identity projected to one field: everything a reader uses
/// to locate and decode the field's values. `file_size_bytes` is a cache
/// hint and excluded. Paths alone are not identity; a commit can change the
/// routing under the same path.
#[derive(Clone, PartialEq)]
struct DataFileWitness {
    path: String,
    base_id: Option<u32>,
    file_major_version: u32,
    file_minor_version: u32,
    column_index: Option<i32>,
}

fn field_data_file_witness(file: &DataFile, field_id: i32) -> Option<DataFileWitness> {
    let position = file.fields.iter().position(|id| *id == field_id)?;
    Some(DataFileWitness {
        path: file.path.clone(),
        base_id: file.base_id,
        file_major_version: file.file_major_version,
        file_minor_version: file.file_minor_version,
        column_index: file.column_indices.get(position).copied(),
    })
}

/// One overlay's identity projected to a single field: its data-file witness,
/// committed version, and the coverage bitmap that applies to the field.
#[derive(Clone, PartialEq)]
struct OverlayBacking {
    file: DataFileWitness,
    committed_version: u64,
    coverage: Arc<RoaringBitmap>,
}

fn field_overlay_backing(overlay: &DataOverlayFile, field_id: i32) -> Option<OverlayBacking> {
    let position = overlay
        .data_file
        .fields
        .iter()
        .position(|id| *id == field_id)?;
    let coverage = match &overlay.coverage {
        OverlayCoverage::Shared(bitmap) => bitmap.clone(),
        OverlayCoverage::PerField(bitmaps) => bitmaps.get(position)?.clone(),
    };
    Some(OverlayBacking {
        file: field_data_file_witness(&overlay.data_file, field_id)?,
        committed_version: overlay.committed_version,
        coverage,
    })
}

fn field_backing(fragment: &Fragment, field_id: i32) -> FieldBacking {
    let base = fragment
        .files
        .iter()
        .find_map(|file| field_data_file_witness(file, field_id));
    let overlays = fragment
        .overlays
        .iter()
        .filter_map(|overlay| field_overlay_backing(overlay, field_id))
        .collect();
    FieldBacking { base, overlays }
}

/// Make `data_file` the fragment's authoritative coverage of its fields:
/// tombstone them in existing files and overlays (the update-columns idiom),
/// drop fully tombstoned files, and append `data_file`.
fn replace_column_coverage(fragment: &mut Fragment, data_file: &DataFile) {
    let replaced: HashSet<i32> = data_file.fields.iter().copied().collect();
    for file in &mut fragment.files {
        file.fields = file
            .fields
            .iter()
            .map(|field| {
                if replaced.contains(field) {
                    TOMBSTONE_FIELD_ID
                } else {
                    *field
                }
            })
            .collect::<Vec<_>>()
            .into();
    }
    fragment
        .files
        .retain(|file| file.fields.iter().any(|&field| field != TOMBSTONE_FIELD_ID));
    let replaced_u32: Vec<u32> = replaced
        .iter()
        .filter(|&&field| field >= 0)
        .map(|&field| field as u32)
        .collect();
    tombstone_overlay_fields(&mut fragment.overlays, &replaced_u32);
    fragment.files.push(data_file.clone());
}
