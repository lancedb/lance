// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dataset-side engines for the fragment column statistics index.
//!
//! Creation is register-only in this version: the index starts empty, no
//! historical fragments are examined, and coverage arrives with future
//! writes. The seeds-based builder below is the maintenance/backfill engine
//! used by those later paths; it never scans column data, only data file
//! footers, and tolerates missing or incomplete payloads per fragment.

use lance_core::{Error, Result};
use lance_file::reader::FileReaderOptions;
use lance_index::scalar::CreatedIndex;
use lance_index::scalar::fragstats::{
    FragmentStatsRecord, build_fragment_stats_from_seeds, fold_seed_payload,
};
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::seed::{FragmentSeed, SEED_META_KEY_PREFIX};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use lance_table::transaction::DerivedStatsUpdate;
use uuid::Uuid;

use crate::dataset::Dataset;
use crate::dataset::index::LanceIndexStoreExt;

/// Build fragment statistics index files for `column` from the fragments'
/// embedded zone statistics (data file footers). No column data is scanned.
///
/// Per-fragment tolerant: a fragment whose payload is missing, unreadable, or
/// provably incomplete simply gets no record. Returns the created index and
/// the fragment ids that received records; the committed coverage bitmap must
/// claim only those.
pub(crate) async fn build_fragment_stats_index_from_seeds(
    dataset: &Dataset,
    column: &str,
    index_uuid: Uuid,
    fragment_ids: Option<Vec<u32>>,
) -> Result<(CreatedIndex, Vec<u32>)> {
    let field = dataset.schema().field(column).ok_or_else(|| {
        Error::invalid_input(format!("column {column} does not exist in the schema"))
    })?;
    let data_type = field.data_type();
    let meta_key = format!("{}{}", SEED_META_KEY_PREFIX, column);

    let scheduler = ScanScheduler::new(
        dataset.object_store.clone(),
        SchedulerConfig::max_bandwidth(&dataset.object_store),
    );

    let mut seeds: Vec<(FragmentSeed, u64)> = Vec::new();
    for fragment in dataset.fragments().iter() {
        if let Some(requested) = &fragment_ids
            && !requested.contains(&(fragment.id as u32))
        {
            continue;
        }
        let Some(expected_rows) = fragment.physical_rows else {
            continue;
        };
        // The payload lives in the data file that carries this column.
        let Some(data_file) = fragment
            .files
            .iter()
            .find(|file| file.fields.contains(&field.id))
        else {
            continue;
        };

        let path = dataset
            .base
            .clone()
            .join(crate::dataset::DATA_DIR)
            .join(data_file.path.as_str());
        let Ok(file_scheduler) = scheduler.open_file(&path, &CachedFileSize::unknown()).await
        else {
            continue;
        };
        let Ok(reader) = lance_file::reader::FileReader::try_open(
            file_scheduler,
            None,
            Default::default(),
            &dataset.metadata_cache.file_metadata_cache(&path),
            FileReaderOptions::default(),
        )
        .await
        else {
            continue;
        };

        let Some(meta_value) = reader
            .metadata()
            .file_schema
            .metadata
            .get(&meta_key)
            .cloned()
        else {
            continue;
        };
        let Some(buf_index) = meta_value
            .split(':')
            .next()
            .and_then(|s| s.parse::<u32>().ok())
        else {
            continue;
        };
        let Ok(bytes) = reader.read_global_buffer(buf_index).await else {
            continue;
        };

        seeds.push((
            FragmentSeed {
                fragment_id: fragment.id,
                bytes,
                metadata_value: meta_value,
            },
            expected_rows as u64,
        ));
    }

    let store = LanceIndexStore::from_dataset_for_new(dataset, &index_uuid)?;
    build_fragment_stats_from_seeds(seeds, &data_type, &store).await
}

/// Register-only creation: write an empty statistics file so the segment is
/// well-formed, claim no coverage. Future writes populate the index.
pub(crate) async fn register_empty_fragment_stats_index(
    dataset: &Dataset,
    column: &str,
    index_uuid: Uuid,
) -> Result<CreatedIndex> {
    let field = dataset.schema().field(column).ok_or_else(|| {
        Error::invalid_input(format!("column {column} does not exist in the schema"))
    })?;
    let store = LanceIndexStore::from_dataset_for_new(dataset, &index_uuid)?;
    let (created, covered) =
        build_fragment_stats_from_seeds(Vec::new(), &field.data_type(), &store).await?;
    debug_assert!(covered.is_empty());
    Ok(created)
}

/// Bucket span for statistics segments: fixed fragment-id ranges,
/// `bucket = fragment_id / SPAN`. An implementation policy, not format:
/// readers locate segments through their coverage bitmaps alone.
pub(crate) const FRAGMENT_STATS_BUCKET_SPAN: u64 = 65_536;

/// Derive the statistics-index changes for one commit attempt.
///
/// For every fragment column stats index, folds the transaction's NEW
/// fragments' footer payloads into records (keyed by this attempt's
/// prospective final ids), merges them into the affected fixed-range bucket
/// segments (copy on write), and returns the segment replacements for the
/// manifest build to splice and validate. Re-derived per attempt; any error
/// or mismatch degrades to "no statistics registered", never a failed commit.
pub(crate) async fn derive_fragment_stats_update(
    dataset: &Dataset,
    transaction: &crate::dataset::transaction::Transaction,
) -> Result<DerivedStatsUpdate> {
    use crate::dataset::transaction::Operation;

    let new_fragments: &[lance_table::format::Fragment] = match &transaction.operation {
        Operation::Append { fragments, .. } => fragments,
        Operation::Update { new_fragments, .. } => new_fragments,
        _ => return Ok(DerivedStatsUpdate::default()),
    };
    if new_fragments.is_empty() {
        return Ok(DerivedStatsUpdate::default());
    }

    let indices = crate::index::load_all_indices(dataset).await?;
    let stats_indices: Vec<&lance_table::format::IndexMetadata> = indices
        .iter()
        .filter(|index| {
            index.index_details.as_ref().is_some_and(|details| {
                details
                    .type_url
                    .ends_with("FragmentColumnStatsIndexDetails")
            })
        })
        .collect();
    if stats_indices.is_empty() {
        return Ok(DerivedStatsUpdate::default());
    }

    // Prospective final ids, replicating the manifest build's assignment over
    // the same fresh manifest (id 0 is the unassigned sentinel). The build's
    // validation drops the whole update if this ever disagrees.
    let mut next_id = dataset
        .manifest
        .max_fragment_id()
        .map(|id| id + 1)
        .unwrap_or(0);
    let assigned: Vec<(u64, &lance_table::format::Fragment)> = new_fragments
        .iter()
        .map(|fragment| {
            if fragment.id != 0 {
                (fragment.id, fragment)
            } else {
                let id = next_id;
                next_id += 1;
                (id, fragment)
            }
        })
        .collect();

    let scheduler = ScanScheduler::new(
        dataset.object_store.clone(),
        SchedulerConfig::max_bandwidth(&dataset.object_store),
    );

    let mut update = DerivedStatsUpdate::default();
    // One logical index = one name; the manifest lists each SEGMENT as its own
    // IndexMetadata row, so group first or every segment would redo the work
    // and push duplicate replacements.
    let mut by_name: std::collections::BTreeMap<&str, Vec<&lance_table::format::IndexMetadata>> =
        std::collections::BTreeMap::new();
    for segment in &stats_indices {
        by_name
            .entry(segment.name.as_str())
            .or_default()
            .push(segment);
    }
    for segments in by_name.values() {
        let index = segments[0];
        let Some(field_id) = index.keyed_field() else {
            continue;
        };
        let Ok(column) = dataset.schema().field_path(field_id) else {
            continue;
        };
        let Some(data_type) = dataset.schema().field(&column).map(|f| f.data_type()) else {
            continue;
        };
        let meta_key = format!("{}{}", SEED_META_KEY_PREFIX, column);

        let mut new_records: Vec<FragmentStatsRecord> = Vec::new();
        for (final_id, fragment) in &assigned {
            let Some(expected_rows) = fragment.physical_rows else {
                continue;
            };
            let Some(data_file) = fragment
                .files
                .iter()
                .find(|file| file.fields.contains(&field_id))
            else {
                continue;
            };
            let path = dataset
                .base
                .clone()
                .join(crate::dataset::DATA_DIR)
                .join(data_file.path.as_str());
            let Ok(file_scheduler) = scheduler.open_file(&path, &CachedFileSize::unknown()).await
            else {
                continue;
            };
            let Ok(reader) = lance_file::reader::FileReader::try_open(
                file_scheduler,
                None,
                Default::default(),
                &dataset.metadata_cache.file_metadata_cache(&path),
                FileReaderOptions::default(),
            )
            .await
            else {
                continue;
            };
            let Some(meta_value) = reader
                .metadata()
                .file_schema
                .metadata
                .get(&meta_key)
                .cloned()
            else {
                continue;
            };
            let Some(buf_index) = meta_value
                .split(':')
                .next()
                .and_then(|s| s.parse::<u32>().ok())
            else {
                continue;
            };
            let Ok(bytes) = reader.read_global_buffer(buf_index).await else {
                continue;
            };
            let seed = FragmentSeed {
                fragment_id: *final_id,
                bytes,
                metadata_value: meta_value,
            };
            if let Some(record) = fold_seed_payload(&seed, expected_rows as u64, &data_type) {
                new_records.push(record);
            }
        }
        if new_records.is_empty() {
            continue;
        }

        // Group into fixed-range buckets and copy-on-write each touched one.
        let mut by_bucket: std::collections::BTreeMap<u64, Vec<FragmentStatsRecord>> =
            std::collections::BTreeMap::new();
        for record in new_records {
            by_bucket
                .entry(record.fragment_id / FRAGMENT_STATS_BUCKET_SPAN)
                .or_default()
                .push(record);
        }
        // A segment whose bitmap is empty contributes nothing (the creation
        // placeholder, or coverage centrally withdrawn to zero) and may be
        // dropped by any later commit.
        for candidate in segments {
            if candidate
                .fragment_bitmap
                .as_ref()
                .is_some_and(|bitmap| bitmap.is_empty())
                && !update.replaced_uuids.contains(&candidate.uuid)
            {
                update.replaced_uuids.push(candidate.uuid);
            }
        }
        for (bucket, records) in by_bucket {
            // Segments never span buckets, so any covered id identifies the
            // bucket; an empty bitmap identifies nothing and is never matched.
            // Replace EVERY matching segment: duplicates must not be left
            // standing to fork coverage.
            let existing: Vec<_> = segments
                .iter()
                .filter(|candidate| {
                    candidate
                        .fragment_bitmap
                        .as_ref()
                        .and_then(|bitmap| bitmap.min())
                        .is_some_and(|id| u64::from(id) / FRAGMENT_STATS_BUCKET_SPAN == bucket)
                })
                .collect();
            // New records win over existing ones with the same fragment id.
            let mut merged: std::collections::BTreeMap<u64, FragmentStatsRecord> =
                std::collections::BTreeMap::new();
            // Coverage stays valid as of the OLDEST replaced segment: newer
            // overlays must keep marking carried records stale, so the copy
            // must not advance their version stamp.
            let mut carried_version: Option<u64> = None;
            for meta in &existing {
                let Some(bitmap) = meta.fragment_bitmap.as_ref() else {
                    continue;
                };
                use crate::index::DatasetIndexInternalExt;
                let opened = dataset
                    .open_scalar_index(
                        &column,
                        &meta.uuid,
                        &lance_index::metrics::NoOpMetricsCollector,
                    )
                    .await?;
                if let Some(stats) = opened
                    .as_any()
                    .downcast_ref::<lance_index::scalar::fragstats::FragmentColumnStatsIndex>(
                ) {
                    for record in stats.records() {
                        // Records outside the bitmap are coverage someone
                        // withdrew (in-place column rewrite); carrying them
                        // forward would re-legitimize stale bounds.
                        if !bitmap.contains(record.fragment_id as u32) {
                            continue;
                        }
                        merged.insert(record.fragment_id, record.clone());
                    }
                }
                carried_version = Some(
                    carried_version.map_or(meta.dataset_version, |v| v.min(meta.dataset_version)),
                );
                update.replaced_uuids.push(meta.uuid);
            }
            for record in records {
                merged.insert(record.fragment_id, record);
            }

            let new_uuid = Uuid::new_v4();
            let store = LanceIndexStore::from_dataset_for_new(dataset, &new_uuid)?;
            let bitmap: roaring::RoaringBitmap = merged.keys().map(|id| *id as u32).collect();
            let created = lance_index::scalar::fragstats::write_fragment_stats_segment(
                merged.into_values().collect(),
                &data_type,
                &store,
            )
            .await?;
            update
                .new_segments
                .push(lance_table::format::IndexMetadata {
                    uuid: new_uuid,
                    name: index.name.clone(),
                    fields: index.fields.clone(),
                    covering_fields: vec![],
                    dataset_version: carried_version.unwrap_or(dataset.manifest.version),
                    fragment_bitmap: Some(bitmap),
                    index_details: Some(std::sync::Arc::new(created.index_details)),
                    index_version: created.index_version as i32,
                    created_at: Some(chrono::Utc::now()),
                    base_id: None,
                    files: Some(lance_index::scalar::index_files_to_table(created.files)),
                });
        }
    }
    Ok(update)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{WriteMode, WriteParams};
    use crate::index::DatasetIndexExt;
    use crate::index::DatasetIndexInternalExt;
    use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_index::IndexType;
    use lance_index::metrics::NoOpMetricsCollector;
    use lance_index::scalar::ScalarIndexParams;
    use lance_index::scalar::fragstats::FragmentColumnStatsIndex;
    use std::sync::Arc;

    fn int_batch(start: i64, end: i64) -> (Arc<ArrowSchema>, RecordBatch) {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "i",
            DataType::Int64,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(start..end))],
        )
        .unwrap();
        (schema, batch)
    }

    async fn fragment_has_seed(dataset: &crate::Dataset, fragment_id: u64, column: &str) -> bool {
        let fragment = dataset
            .fragments()
            .iter()
            .find(|f| f.id == fragment_id)
            .unwrap()
            .clone();
        let meta_key = format!("{}{}", SEED_META_KEY_PREFIX, column);
        let scheduler = ScanScheduler::new(
            dataset.object_store.clone(),
            SchedulerConfig::max_bandwidth(&dataset.object_store),
        );
        let path = dataset
            .base
            .clone()
            .join(crate::dataset::DATA_DIR)
            .join(fragment.files[0].path.as_str());
        let file_scheduler = scheduler
            .open_file(&path, &CachedFileSize::unknown())
            .await
            .unwrap();
        let reader = lance_file::reader::FileReader::try_open(
            file_scheduler,
            None,
            Default::default(),
            &dataset.metadata_cache.file_metadata_cache(&path),
            FileReaderOptions::default(),
        )
        .await
        .unwrap();
        reader
            .metadata()
            .file_schema
            .metadata
            .contains_key(&meta_key)
    }

    #[tokio::test]
    async fn test_create_registers_empty_and_enables_seed_collection() {
        let test_uri = TempStrDir::default();
        let (schema, batch) = int_batch(0, 200);
        let params = WriteParams {
            max_rows_per_file: 100,
            ..Default::default()
        };
        let mut dataset = crate::Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema.clone()),
            &test_uri,
            Some(params),
        )
        .await
        .unwrap();
        assert_eq!(dataset.fragments().len(), 2);
        // No index yet: writes carry no statistics payloads.
        assert!(!fragment_has_seed(&dataset, 0, "i").await);

        dataset
            .create_index(
                &["i"],
                IndexType::FragmentColumnStats,
                None,
                &ScalarIndexParams::new("FragmentColumnStats".to_string()),
                true,
            )
            .await
            .unwrap();

        // Register-only: the index exists, claims nothing, examined nothing.
        let indices = dataset.load_indices().await.unwrap();
        let index_meta = indices.first().unwrap();
        assert_eq!(index_meta.fragment_bitmap.as_ref().unwrap().len(), 0);
        let index = dataset
            .open_scalar_index(&"i", &index_meta.uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let stats = index
            .as_any()
            .downcast_ref::<FragmentColumnStatsIndex>()
            .unwrap();
        assert_eq!(stats.num_fragments(), 0);
        // Pre-existing fragments were not touched: still no payloads.
        assert!(!fragment_has_seed(&dataset, 0, "i").await);

        // Creating the index turned collection on: the next append carries
        // the zone statistics payload for the indexed column.
        let (_, batch) = int_batch(200, 300);
        let append = WriteParams {
            mode: WriteMode::Append,
            max_rows_per_file: 100,
            ..Default::default()
        };
        let dataset = crate::Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            Arc::new(dataset),
            Some(append),
        )
        .await
        .unwrap();
        assert!(fragment_has_seed(&dataset, 2, "i").await);
        assert!(!fragment_has_seed(&dataset, 0, "i").await);
    }

    #[tokio::test]
    async fn test_seeds_backfill_engine_covers_only_complete_payloads() {
        let test_uri = TempStrDir::default();
        let (schema, batch) = int_batch(0, 200);
        let params = WriteParams {
            max_rows_per_file: 100,
            ..Default::default()
        };
        let mut dataset = crate::Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema.clone()),
            &test_uri,
            Some(params),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["i"],
                IndexType::FragmentColumnStats,
                None,
                &ScalarIndexParams::new("FragmentColumnStats".to_string()),
                true,
            )
            .await
            .unwrap();
        // Fragments 2 and 3 are written with collection active.
        let (_, batch) = int_batch(200, 400);
        let append = WriteParams {
            mode: WriteMode::Append,
            max_rows_per_file: 100,
            ..Default::default()
        };
        let dataset = crate::Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            Arc::new(dataset),
            Some(append),
        )
        .await
        .unwrap();

        // The engine folds only fragments with payloads: 0 and 1 have none.
        let uuid = uuid::Uuid::new_v4();
        let (_created, covered) = build_fragment_stats_index_from_seeds(&dataset, "i", uuid, None)
            .await
            .unwrap();
        assert_eq!(covered, vec![2, 3]);
    }

    #[tokio::test]
    async fn test_same_commit_registration_on_append() {
        let test_uri = TempStrDir::default();
        let (schema, batch) = int_batch(0, 200);
        let params = WriteParams {
            max_rows_per_file: 100,
            ..Default::default()
        };
        let mut dataset = crate::Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema.clone()),
            &test_uri,
            Some(params),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["i"],
                IndexType::FragmentColumnStats,
                None,
                &ScalarIndexParams::new("FragmentColumnStats".to_string()),
                true,
            )
            .await
            .unwrap();

        // The append itself must register coverage: data and statistics land
        // in one commit.
        let (_, batch) = int_batch(200, 400);
        let append = WriteParams {
            mode: WriteMode::Append,
            max_rows_per_file: 100,
            ..Default::default()
        };
        let dataset = crate::Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema.clone()),
            Arc::new(dataset),
            Some(append),
        )
        .await
        .unwrap();
        let version_after_first_append = dataset.version().version;

        let indices = dataset.load_indices().await.unwrap();
        let metas: Vec<_> = indices.iter().filter(|i| i.name.contains("i_")).collect();
        let covered: roaring::RoaringBitmap = metas
            .iter()
            .filter_map(|m| m.fragment_bitmap.clone())
            .reduce(|a, b| a | b)
            .unwrap();
        assert!(covered.contains(2) && covered.contains(3));
        assert!(!covered.contains(0) && !covered.contains(1));
        // Exactly one populated segment: per-segment iteration must not fork
        // the bucket into duplicates.
        assert_eq!(
            metas
                .iter()
                .filter(|m| m.fragment_bitmap.as_ref().is_some_and(|b| !b.is_empty()))
                .count(),
            1
        );
        let first_segment_uuid = metas
            .iter()
            .find(|m| m.fragment_bitmap.as_ref().unwrap().len() > 0)
            .unwrap()
            .uuid;

        // Second append: the tail bucket is rewritten copy-on-write in that
        // commit; the superseded segment uuid leaves the manifest.
        let (_, batch) = int_batch(400, 500);
        let append = WriteParams {
            mode: WriteMode::Append,
            max_rows_per_file: 100,
            ..Default::default()
        };
        let dataset = crate::Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            Arc::new(dataset),
            Some(append),
        )
        .await
        .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let metas: Vec<_> = indices.iter().filter(|i| i.name.contains("i_")).collect();
        let covered: roaring::RoaringBitmap = metas
            .iter()
            .filter_map(|m| m.fragment_bitmap.clone())
            .reduce(|a, b| a | b)
            .unwrap();
        assert_eq!(covered.len(), 3);
        assert!(covered.contains(4));
        assert!(metas.iter().all(|m| m.uuid != first_segment_uuid));
        assert_eq!(
            metas
                .iter()
                .filter(|m| m.fragment_bitmap.as_ref().is_some_and(|b| !b.is_empty()))
                .count(),
            1
        );

        // The records themselves resolve and prune correctly.
        let populated = metas
            .iter()
            .find(|m| m.fragment_bitmap.as_ref().unwrap().len() == 3)
            .unwrap();
        let index = dataset
            .open_scalar_index(&"i", &populated.uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let stats = index
            .as_any()
            .downcast_ref::<FragmentColumnStatsIndex>()
            .unwrap();
        assert_eq!(stats.num_fragments(), 3);
        use lance_index::scalar::SargableQuery;
        let excluded = stats
            .excluded_fragments(&SargableQuery::Range(
                std::ops::Bound::Included(datafusion::common::ScalarValue::Int64(Some(450))),
                std::ops::Bound::Unbounded,
            ))
            .unwrap();
        assert!(excluded.contains(2) && excluded.contains(3));
        assert!(!excluded.contains(4));

        // Time travel: the old snapshot still resolves its original segment.
        let old = dataset
            .checkout_version(version_after_first_append)
            .await
            .unwrap();
        let old_indices = old.load_indices().await.unwrap();
        assert!(old_indices.iter().any(|m| m.uuid == first_segment_uuid));
    }
}
