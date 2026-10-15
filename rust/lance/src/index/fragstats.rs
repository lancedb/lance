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
use lance_index::scalar::fragstats::build_fragment_stats_from_seeds;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::seed::{FragmentSeed, SEED_META_KEY_PREFIX};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
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
}
