// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Hamming distance clustering for IVF_FLAT indices.
//!
//! This module provides functionality to perform pairwise hamming distance
//! computation and clustering on specific partitions of IVF_FLAT indices.

use std::time::Instant;

use arrow_array::RecordBatchReader;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_schema::DataType;
use lance_core::{Error, Result};
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::vector::VectorIndex;
use lance_index::vector::flat::index::{FlatBinQuantizer, FlatIndex};
use lance_index::vector::flat::storage::FLAT_COLUMN;
use lance_index::vector::storage::VectorStore;
use lance_linalg::distance::{
    ClusteringResult, cluster_pairwise_result, extract_hashes_from_fixed_list,
    pairwise_hamming_distance_parallel,
};
use rand::rng;
use rand::seq::index::sample;

use crate::dataset::Dataset;
use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};

use super::ivf::v2::IVFIndex;

/// Perform pairwise hamming distance clustering on a partition of an IVF_FLAT index.
///
/// This function loads a specific partition from an IVF_FLAT index on a hash column,
/// computes pairwise hamming distances between all hashes in the partition,
/// filters by threshold, and clusters the results using union-find.
///
/// # Arguments
///
/// * `dataset` - The Lance dataset
/// * `index_name` - Name of the IVF_FLAT index on the hash column
/// * `partition_id` - The partition ID within the IVF_FLAT index
/// * `hamming_threshold` - Maximum hamming distance to consider as similar
///
/// # Returns
///
/// A `RecordBatchReader` yielding batches with columns:
/// - `representative`: UInt64 - The representative row ID for each cluster
/// - `duplicates`: `List<UInt64>` - List of duplicate row IDs in each cluster
///
/// # Errors
///
/// Returns an error if:
/// - The index doesn't exist or is not an IVF_FLAT index
/// - The indexed column has wrong type (must be `FixedSizeList<UInt8, 8>`)
/// - The partition ID is out of range
pub async fn hamming_clustering_for_ivf_partition(
    dataset: &Dataset,
    index_name: &str,
    partition_id: usize,
    hamming_threshold: u32,
) -> Result<Box<dyn RecordBatchReader + Send>> {
    // Load indices and find the IVF_FLAT index
    let indices = dataset.load_indices().await?;
    let index_meta = indices
        .iter()
        .find(|idx| idx.name == index_name)
        .ok_or_else(|| {
            Error::invalid_input(format!("Index '{}' not found on dataset", index_name))
        })?;

    // Get the column name from the index metadata
    let schema = dataset.schema();
    let field_id = index_meta
        .fields
        .first()
        .ok_or_else(|| Error::invalid_input(format!("Index '{}' has no fields", index_name)))?;
    let field = schema.field_by_id(*field_id).ok_or_else(|| {
        Error::invalid_input(format!(
            "Field with id {} not found in schema for index '{}'",
            field_id, index_name
        ))
    })?;
    let column = &field.name;

    // Check column is FixedSizeList<UInt8, 8>
    let data_type = field.data_type();
    match data_type {
        DataType::FixedSizeList(inner, 8) => {
            if *inner.data_type() != DataType::UInt8 {
                return Err(Error::invalid_input(format!(
                    "Column '{}' must be FixedSizeList<UInt8, 8>, got FixedSizeList<{:?}, 8>",
                    column,
                    inner.data_type()
                )));
            }
        }
        _ => {
            return Err(Error::invalid_input(format!(
                "Column '{}' must be FixedSizeList<UInt8, 8>, got {:?}",
                column, data_type
            )));
        }
    }

    // Open the vector index
    let index = dataset
        .open_vector_index(column, &index_meta.uuid, &NoOpMetricsCollector)
        .await?;

    // Try to downcast to IVFIndex<FlatIndex, FlatBinQuantizer> (IVF_FLAT for binary data)
    let ivf_index = index
        .as_any()
        .downcast_ref::<IVFIndex<FlatIndex, FlatBinQuantizer>>()
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "Index '{}' is not an IVF_FLAT index for binary data",
                index_name
            ))
        })?;

    // Check partition ID is valid
    let num_partitions = ivf_index.ivf_model().num_partitions();
    if partition_id >= num_partitions {
        return Err(Error::invalid_input(format!(
            "Partition ID {} is out of range (0..{})",
            partition_id, num_partitions
        )));
    }

    // Load the partition storage
    let storage = ivf_index.load_partition_storage(partition_id, None).await?;

    // Get row IDs
    let row_id_slice: Vec<u64> = storage.row_ids().copied().collect();

    if row_id_slice.is_empty() {
        let empty = ClusteringResult {
            clusters: Vec::new(),
        };
        return Ok(empty.into_reader(None));
    }

    // Get vectors from the storage batches
    let batches: Vec<_> = storage.to_batches()?.collect();
    if batches.is_empty() {
        let empty = ClusteringResult {
            clusters: Vec::new(),
        };
        return Ok(empty.into_reader(None));
    }

    // Extract the hash vectors from the FLAT_COLUMN
    let mut all_hashes = Vec::new();
    for batch in &batches {
        let vectors = batch
            .column_by_name(FLAT_COLUMN)
            .ok_or_else(|| {
                Error::invalid_input(format!("Column '{}' not found in storage", FLAT_COLUMN))
            })?
            .as_fixed_size_list();
        let hashes = extract_hashes_from_fixed_list(vectors)?;
        all_hashes.extend(hashes);
    }

    // Compute pairwise hamming distances with threshold filtering
    let pairwise_result = pairwise_hamming_distance_parallel(
        &all_hashes,
        Some(&row_id_slice),
        Some(hamming_threshold),
    );

    // Cluster the results
    let clustering = cluster_pairwise_result(&pairwise_result);

    Ok(clustering.into_reader(None))
}

/// Get partition statistics for an IVF_FLAT index.
pub async fn get_ivf_partition_info(
    dataset: &Dataset,
    index_name: &str,
) -> Result<Vec<PartitionInfo>> {
    let indices = dataset.load_indices().await?;
    let index_meta = indices
        .iter()
        .find(|idx| idx.name == index_name)
        .ok_or_else(|| {
            Error::invalid_input(format!("Index '{}' not found on dataset", index_name))
        })?;

    // Get the column name from the index metadata
    let schema = dataset.schema();
    let field_id = index_meta
        .fields
        .first()
        .ok_or_else(|| Error::invalid_input(format!("Index '{}' has no fields", index_name)))?;
    let field = schema.field_by_id(*field_id).ok_or_else(|| {
        Error::invalid_input(format!(
            "Field with id {} not found in schema for index '{}'",
            field_id, index_name
        ))
    })?;
    let column = &field.name;

    let index = dataset
        .open_vector_index(column, &index_meta.uuid, &NoOpMetricsCollector)
        .await?;

    let ivf_index = index
        .as_any()
        .downcast_ref::<IVFIndex<FlatIndex, FlatBinQuantizer>>()
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "Index '{}' is not an IVF_FLAT index for binary data",
                index_name
            ))
        })?;

    let num_partitions = ivf_index.ivf_model().num_partitions();
    let mut partition_infos = Vec::with_capacity(num_partitions);

    for i in 0..num_partitions {
        partition_infos.push(PartitionInfo {
            partition_id: i,
            size: ivf_index.ivf_model().partition_size(i),
        });
    }

    Ok(partition_infos)
}

/// Information about an IVF partition.
#[derive(Debug, Clone)]
pub struct PartitionInfo {
    pub partition_id: usize,
    pub size: usize,
}

/// Perform pairwise hamming distance clustering on sampled rows from a dataset.
///
/// This function samples N rows randomly from the dataset, extracts hashes,
/// computes pairwise hamming distances, and clusters the results.
/// It's useful for benchmarking and testing without requiring an IVF index.
///
/// # Arguments
///
/// * `dataset` - The Lance dataset
/// * `column` - Name of the hash column (must be `FixedSizeList<UInt8, 8>`)
/// * `sample_size` - Number of rows to sample (if None or >= total rows, uses all rows)
/// * `hamming_threshold` - Maximum hamming distance to consider as similar
///
/// # Returns
///
/// A `RecordBatchReader` yielding batches with columns:
/// - `representative`: UInt64 - The representative row ID for each cluster
/// - `duplicates`: `List<UInt64>` - List of duplicate row IDs in each cluster
pub async fn hamming_clustering_for_sample(
    dataset: &Dataset,
    column: &str,
    sample_size: Option<usize>,
    hamming_threshold: u32,
) -> Result<Box<dyn RecordBatchReader + Send>> {
    // Validate column exists and has correct type
    let schema = dataset.schema();
    let field = schema.field(column).ok_or_else(|| {
        Error::invalid_input(format!("Column '{}' not found in dataset schema", column))
    })?;

    // Check column is FixedSizeList<UInt8, 8>
    let data_type = field.data_type();
    match data_type {
        DataType::FixedSizeList(inner, 8) => {
            if *inner.data_type() != DataType::UInt8 {
                return Err(Error::invalid_input(format!(
                    "Column '{}' must be FixedSizeList<UInt8, 8>, got FixedSizeList<{:?}, 8>",
                    column,
                    inner.data_type()
                )));
            }
        }
        _ => {
            return Err(Error::invalid_input(format!(
                "Column '{}' must be FixedSizeList<UInt8, 8>, got {:?}",
                column, data_type
            )));
        }
    }

    // Get total row count
    let total_rows: usize = dataset
        .get_fragments()
        .iter()
        .filter_map(|f| f.metadata().physical_rows)
        .sum();

    let use_sampling = sample_size.is_some_and(|s| s < total_rows);
    let effective_sample = sample_size.unwrap_or(total_rows).min(total_rows);

    // Read data
    let (hashes, row_ids) = if use_sampling {
        // Random sample using take() with _rowid (take uses positional indices)
        let indices: Vec<u64> = sample(&mut rng(), total_rows, effective_sample)
            .iter()
            .map(|i| i as u64)
            .collect();

        let batch = dataset
            .take(
                &indices,
                crate::dataset::ProjectionRequest::from_columns(
                    [column, "_rowid"],
                    dataset.schema(),
                ),
            )
            .await?;

        let rowid_col = batch.column_by_name("_rowid").ok_or_else(|| {
            Error::invalid_input("_rowid column not found in take result".to_string())
        })?;
        let row_ids = rowid_col.as_primitive::<UInt64Type>();
        let row_id_vec: Vec<u64> = row_ids.values().to_vec();

        let hash_col = batch.column_by_name(column).ok_or_else(|| {
            Error::invalid_input(format!("Column '{}' not found in result", column))
        })?;
        let hashes_arr = hash_col.as_fixed_size_list();
        let hashes = extract_hashes_from_fixed_list(hashes_arr)?;

        (hashes, row_id_vec)
    } else {
        // Full scan
        let batch = dataset
            .scan()
            .project(&[column])?
            .with_row_id()
            .try_into_batch()
            .await?;

        let rowid_col = batch.column_by_name("_rowid").ok_or_else(|| {
            Error::invalid_input("_rowid column not found in scan result".to_string())
        })?;
        let row_ids = rowid_col.as_primitive::<UInt64Type>();
        let row_id_vec: Vec<u64> = row_ids.values().to_vec();

        let hash_col = batch.column_by_name(column).ok_or_else(|| {
            Error::invalid_input(format!("Column '{}' not found in result", column))
        })?;
        let hashes_arr = hash_col.as_fixed_size_list();
        let hashes = extract_hashes_from_fixed_list(hashes_arr)?;

        (hashes, row_id_vec)
    };

    if hashes.len() < 2 {
        let empty = ClusteringResult {
            clusters: Vec::new(),
        };
        return Ok(empty.into_reader(None));
    }

    // Compute pairwise hamming distances
    let pairwise =
        pairwise_hamming_distance_parallel(&hashes, Some(&row_ids), Some(hamming_threshold));

    // Cluster edges
    let clustering = cluster_pairwise_result(&pairwise);

    Ok(clustering.into_reader(None))
}

/// Perform pairwise hamming distance clustering on a contiguous range of rows from a fragment.
///
/// This function reads a contiguous range of rows from a specific fragment,
/// extracts hashes, computes pairwise hamming distances, and clusters the results.
/// Unlike sampling, this reads sequential rows which is useful for distributed
/// processing where each worker handles a specific range of a fragment.
///
/// # Arguments
///
/// * `dataset` - The Lance dataset
/// * `column` - Name of the hash column (must be `FixedSizeList<UInt8, 8>`)
/// * `fragment_id` - The fragment ID to read from
/// * `start_row` - The starting row offset within the fragment
/// * `num_rows` - Number of rows to read from the start position
/// * `hamming_threshold` - Maximum hamming distance to consider as similar
///
/// # Returns
///
/// A `RecordBatchReader` yielding batches with columns:
/// - `representative`: UInt64 - The representative row ID for each cluster
/// - `duplicates`: `List<UInt64>` - List of duplicate row IDs in each cluster
///
/// # Errors
///
/// Returns an error if:
/// - The fragment doesn't exist
/// - The column has wrong type (must be `FixedSizeList<UInt8, 8>`)
/// - The row range is out of bounds
pub async fn hamming_clustering_for_range(
    dataset: &Dataset,
    column: &str,
    fragment_id: usize,
    start_row: usize,
    num_rows: usize,
    hamming_threshold: u32,
) -> Result<Box<dyn RecordBatchReader + Send>> {
    // Validate column exists and has correct type
    let schema = dataset.schema();
    let field = schema.field(column).ok_or_else(|| {
        Error::invalid_input(format!("Column '{}' not found in dataset schema", column))
    })?;

    // Check column is FixedSizeList<UInt8, 8>
    let data_type = field.data_type();
    match data_type {
        DataType::FixedSizeList(inner, 8) => {
            if *inner.data_type() != DataType::UInt8 {
                return Err(Error::invalid_input(format!(
                    "Column '{}' must be FixedSizeList<UInt8, 8>, got FixedSizeList<{:?}, 8>",
                    column,
                    inner.data_type()
                )));
            }
        }
        _ => {
            return Err(Error::invalid_input(format!(
                "Column '{}' must be FixedSizeList<UInt8, 8>, got {:?}",
                column, data_type
            )));
        }
    }

    // Get the fragment
    let fragment = dataset.get_fragment(fragment_id).ok_or_else(|| {
        Error::invalid_input(format!("Fragment with ID {} not found", fragment_id))
    })?;

    // Get fragment metadata for physical row count
    let fragment_meta = fragment.metadata().clone();
    let physical_rows = fragment_meta
        .physical_rows
        .ok_or_else(|| Error::invalid_input("Fragment has no physical_rows metadata"))?;

    // Validate the range
    if start_row >= physical_rows {
        return Err(Error::invalid_input(format!(
            "start_row {} is out of range for fragment with {} physical rows",
            start_row, physical_rows
        )));
    }

    // Adjust num_rows if it exceeds available rows
    let effective_num_rows = num_rows.min(physical_rows - start_row);

    if effective_num_rows == 0 {
        let empty = ClusteringResult {
            clusters: Vec::new(),
        };
        return Ok(empty.into_reader(None));
    }

    // Use scanner with the specific fragment and limit/offset
    let batch = dataset
        .scan()
        .with_fragments(vec![fragment_meta])
        .project(&[column])?
        .with_row_id()
        .limit(Some(effective_num_rows as i64), Some(start_row as i64))?
        .try_into_batch()
        .await?;

    // Extract row IDs
    let rowid_col = batch.column_by_name("_rowid").ok_or_else(|| {
        Error::invalid_input("_rowid column not found in scan result".to_string())
    })?;
    let row_ids = rowid_col.as_primitive::<UInt64Type>();
    let row_id_vec: Vec<u64> = row_ids.values().to_vec();

    // Extract hashes
    let hash_col = batch
        .column_by_name(column)
        .ok_or_else(|| Error::invalid_input(format!("Column '{}' not found in result", column)))?;
    let hashes_arr = hash_col.as_fixed_size_list();
    let hashes = extract_hashes_from_fixed_list(hashes_arr)?;

    if hashes.len() < 2 {
        let empty = ClusteringResult {
            clusters: Vec::new(),
        };
        return Ok(empty.into_reader(None));
    }

    // Compute pairwise hamming distances
    let pairwise =
        pairwise_hamming_distance_parallel(&hashes, Some(&row_id_vec), Some(hamming_threshold));

    // Cluster edges
    let clustering = cluster_pairwise_result(&pairwise);

    Ok(clustering.into_reader(None))
}

/// Perform pairwise hamming distance clustering on provided hashes (no I/O).
///
/// This is useful for benchmarking the pure compute performance without I/O.
/// Logs timing information via tracing.
///
/// # Arguments
///
/// * `hashes` - Vector of 64-bit hash values
/// * `row_ids` - Optional row IDs (defaults to indices if None)
/// * `hamming_threshold` - Maximum hamming distance to consider as similar
///
/// # Returns
///
/// A `RecordBatchReader` yielding batches with columns:
/// - `representative`: UInt64 - The representative row ID for each cluster
/// - `duplicates`: `List<UInt64>` - List of duplicate row IDs in each cluster
pub fn hamming_clustering_from_hashes(
    hashes: &[u64],
    row_ids: Option<&[u64]>,
    hamming_threshold: u32,
) -> Box<dyn RecordBatchReader + Send> {
    let num_rows = hashes.len();
    if num_rows < 2 {
        let empty = ClusteringResult {
            clusters: Vec::new(),
        };
        return empty.into_reader(None);
    }

    let total_pairs = (num_rows as u64) * (num_rows as u64 - 1) / 2;

    // Compute pairwise hamming distances
    let t_compute_start = Instant::now();
    let pairwise = pairwise_hamming_distance_parallel(hashes, row_ids, Some(hamming_threshold));
    let compute_time = t_compute_start.elapsed();

    // Cluster edges
    let t_cluster_start = Instant::now();
    let clustering = cluster_pairwise_result(&pairwise);
    let cluster_time = t_cluster_start.elapsed();

    // Log timing info
    let pairs_per_sec = if compute_time.as_secs_f64() > 0.0 {
        total_pairs as f64 / compute_time.as_secs_f64()
    } else {
        0.0
    };
    tracing::info!(
        num_rows,
        total_pairs,
        edges = pairwise.len(),
        compute_time_ms = compute_time.as_millis(),
        cluster_time_ms = cluster_time.as_millis(),
        pairs_per_sec_millions = pairs_per_sec / 1_000_000.0,
        num_clusters = clustering.num_clusters(),
        num_duplicates = clustering.num_duplicates(),
        "Hamming clustering completed"
    );

    clustering.into_reader(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;

    /// Helper to collect all clusters from a reader.
    fn collect_clusters(reader: Box<dyn RecordBatchReader + Send>) -> Vec<(u64, Vec<u64>)> {
        let mut clusters = Vec::new();
        for batch in reader {
            let batch = batch.unwrap();
            let reps = batch
                .column(0)
                .as_any()
                .downcast_ref::<arrow_array::UInt64Array>()
                .unwrap();
            let dups = batch
                .column(1)
                .as_any()
                .downcast_ref::<arrow_array::ListArray>()
                .unwrap();

            for i in 0..batch.num_rows() {
                let rep = reps.value(i);
                let dup_arr = dups.value(i);
                let dup_values = dup_arr
                    .as_any()
                    .downcast_ref::<arrow_array::UInt64Array>()
                    .unwrap();
                let duplicates: Vec<u64> = dup_values.values().to_vec();
                clusters.push((rep, duplicates));
            }
        }
        clusters
    }

    #[test]
    fn test_hamming_clustering_from_hashes_basic() {
        // Create some test hashes with known distances
        let hashes = vec![
            0b0000u64, // hash 0
            0b0001u64, // hash 1 - distance 1 from hash 0
            0b0011u64, // hash 2 - distance 1 from hash 1, distance 2 from hash 0
            0b1111u64, // hash 3 - distance 2 from hash 2, distance 4 from hash 0
        ];

        let reader = hamming_clustering_from_hashes(&hashes, None, 1);
        let clusters = collect_clusters(reader);

        // With threshold 1, pairs (0,1) and (1,2) should be connected
        // This forms one cluster: {0, 1, 2}
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].1.len(), 2); // 2 duplicates in the cluster
    }

    #[test]
    fn test_hamming_clustering_from_hashes_no_clusters() {
        // All hashes are far apart
        let hashes = vec![
            0x0000000000000000u64,
            0xFFFFFFFFFFFFFFFFu64,
            0xAAAAAAAAAAAAAAAAu64,
        ];

        let reader = hamming_clustering_from_hashes(&hashes, None, 5);
        let clusters = collect_clusters(reader);

        // With threshold 5, no pairs should be connected (min distance is 32)
        assert_eq!(clusters.len(), 0);
    }

    #[test]
    fn test_hamming_clustering_from_hashes_with_row_ids() {
        let hashes = vec![0b0000u64, 0b0001u64];
        let row_ids = vec![100u64, 200u64];

        let reader = hamming_clustering_from_hashes(&hashes, Some(&row_ids), 1);
        let clusters = collect_clusters(reader);

        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].0, 100); // representative
        assert_eq!(clusters[0].1, vec![200]); // duplicates
    }

    #[tokio::test]
    async fn test_hamming_clustering_for_ivf_partition() {
        use arrow_array::{FixedSizeListArray, RecordBatchIterator, UInt8Array};
        use arrow_schema::{Field, Schema};
        use lance_arrow::FixedSizeListArrayExt;
        use lance_index::vector::ivf::IvfBuildParams;
        use std::sync::Arc;
        use tempfile::tempdir;

        // Create test data with hash column (FixedSizeList<UInt8, 8>)
        let schema = Arc::new(Schema::new(vec![Field::new(
            "hash",
            arrow_schema::DataType::FixedSizeList(
                Arc::new(Field::new("item", arrow_schema::DataType::UInt8, true)),
                8,
            ),
            false,
        )]));

        // Generate hashes with some duplicates (similar hashes)
        let num_rows = 100;
        let mut hash_bytes = Vec::with_capacity(num_rows * 8);
        for i in 0..num_rows {
            // Create groups of similar hashes
            let base = (i / 10) as u64; // 10 groups
            let variation = (i % 10) as u64;
            let hash = base.wrapping_mul(0x123456789) ^ variation;
            hash_bytes.extend_from_slice(&hash.to_le_bytes());
        }
        let values = UInt8Array::from(hash_bytes);
        let hash_array =
            FixedSizeListArray::try_new_from_values(values, 8).expect("create hash array");

        let batch =
            arrow_array::RecordBatch::try_new(schema.clone(), vec![Arc::new(hash_array)]).unwrap();

        // Write dataset
        let temp_dir = tempdir().unwrap();
        let uri = temp_dir.path().to_str().unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = crate::Dataset::write(reader, uri, None).await.unwrap();

        // Create IVF_FLAT index with 4 partitions
        let ivf_params = IvfBuildParams::new(4);
        let params = crate::index::vector::VectorIndexParams::with_ivf_flat_params(
            lance_linalg::distance::MetricType::Hamming,
            ivf_params,
        );

        dataset
            .create_index(
                &["hash"],
                crate::index::IndexType::Vector,
                None,
                &params,
                false,
            )
            .await
            .unwrap();

        // Load and test
        let dataset = crate::Dataset::open(uri).await.unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let index_name = &indices[0].name;

        // Test clustering on partition 0
        let reader = hamming_clustering_for_ivf_partition(&dataset, index_name, 0, 10)
            .await
            .unwrap();
        let clusters = collect_clusters(reader);

        // Verify we get valid results (may or may not have clusters depending on data distribution)
        // At minimum, verify no panics and valid schema
        for (rep, dups) in &clusters {
            assert!(*rep < num_rows as u64 * 10); // row IDs should be reasonable
            for dup in dups {
                assert!(*dup < num_rows as u64 * 10);
            }
        }
    }

    #[tokio::test]
    async fn test_hamming_clustering_for_ivf_partition_invalid_index() {
        use arrow_array::{FixedSizeListArray, RecordBatchIterator, UInt8Array};
        use arrow_schema::{Field, Schema};
        use lance_arrow::FixedSizeListArrayExt;
        use std::sync::Arc;
        use tempfile::tempdir;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "hash",
            arrow_schema::DataType::FixedSizeList(
                Arc::new(Field::new("item", arrow_schema::DataType::UInt8, true)),
                8,
            ),
            false,
        )]));

        let values = UInt8Array::from(vec![0u8; 80]); // 10 rows * 8 bytes
        let hash_array = FixedSizeListArray::try_new_from_values(values, 8).unwrap();
        let batch =
            arrow_array::RecordBatch::try_new(schema.clone(), vec![Arc::new(hash_array)]).unwrap();

        let temp_dir = tempdir().unwrap();
        let uri = temp_dir.path().to_str().unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let dataset = crate::Dataset::write(reader, uri, None).await.unwrap();

        // Test with non-existent index
        let result = hamming_clustering_for_ivf_partition(&dataset, "nonexistent", 0, 10).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("not found"), "Error: {}", err);
    }

    #[tokio::test]
    async fn test_hamming_clustering_for_sample_integration() {
        use arrow_array::{FixedSizeListArray, RecordBatchIterator, UInt8Array};
        use arrow_schema::{Field, Schema};
        use lance_arrow::FixedSizeListArrayExt;
        use std::sync::Arc;
        use tempfile::tempdir;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "hash",
            arrow_schema::DataType::FixedSizeList(
                Arc::new(Field::new("item", arrow_schema::DataType::UInt8, true)),
                8,
            ),
            false,
        )]));

        // Create 50 rows with some duplicate hashes
        let num_rows = 50;
        let mut hash_bytes = Vec::with_capacity(num_rows * 8);
        for i in 0..num_rows {
            // Create some identical hashes (groups of 5)
            let hash = (i / 5) as u64;
            hash_bytes.extend_from_slice(&hash.to_le_bytes());
        }
        let values = UInt8Array::from(hash_bytes);
        let hash_array = FixedSizeListArray::try_new_from_values(values, 8).unwrap();
        let batch =
            arrow_array::RecordBatch::try_new(schema.clone(), vec![Arc::new(hash_array)]).unwrap();

        let temp_dir = tempdir().unwrap();
        let uri = temp_dir.path().to_str().unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        crate::Dataset::write(reader, uri, None).await.unwrap();

        let dataset = crate::Dataset::open(uri).await.unwrap();

        // Test full scan (no sampling)
        let reader = hamming_clustering_for_sample(&dataset, "hash", None, 0)
            .await
            .unwrap();
        let clusters = collect_clusters(reader);

        // With threshold 0 (exact match) and groups of 5 identical hashes,
        // we should have 10 clusters with 4 duplicates each
        assert_eq!(clusters.len(), 10);
        for (_, dups) in &clusters {
            assert_eq!(dups.len(), 4);
        }

        // Test with sampling
        let reader = hamming_clustering_for_sample(&dataset, "hash", Some(20), 0)
            .await
            .unwrap();
        let clusters = collect_clusters(reader);
        // With sampling, we may get fewer clusters
        assert!(clusters.len() <= 10);
    }

    #[tokio::test]
    async fn test_hamming_clustering_for_range_integration() {
        use arrow_array::{FixedSizeListArray, RecordBatchIterator, UInt8Array};
        use arrow_schema::{Field, Schema};
        use lance_arrow::FixedSizeListArrayExt;
        use std::sync::Arc;
        use tempfile::tempdir;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "hash",
            arrow_schema::DataType::FixedSizeList(
                Arc::new(Field::new("item", arrow_schema::DataType::UInt8, true)),
                8,
            ),
            false,
        )]));

        // Create 50 rows with some duplicate hashes (groups of 5 identical hashes)
        let num_rows = 50;
        let mut hash_bytes = Vec::with_capacity(num_rows * 8);
        for i in 0..num_rows {
            let hash = (i / 5) as u64;
            hash_bytes.extend_from_slice(&hash.to_le_bytes());
        }
        let values = UInt8Array::from(hash_bytes);
        let hash_array = FixedSizeListArray::try_new_from_values(values, 8).unwrap();
        let batch =
            arrow_array::RecordBatch::try_new(schema.clone(), vec![Arc::new(hash_array)]).unwrap();

        let temp_dir = tempdir().unwrap();
        let uri = temp_dir.path().to_str().unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        crate::Dataset::write(reader, uri, None).await.unwrap();

        let dataset = crate::Dataset::open(uri).await.unwrap();

        // Get fragment info
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        let fragment_id = fragments[0].id() as usize;

        // Test reading range from the fragment
        // Reading rows 0-25 should cover groups 0-4 (5 groups, each with 5 rows)
        let reader = hamming_clustering_for_range(&dataset, "hash", fragment_id, 0, 25, 0)
            .await
            .unwrap();
        let clusters = collect_clusters(reader);

        // With threshold 0 and 25 rows (groups 0-4), we should have 5 clusters
        // Each cluster has 4 duplicates (5 identical hashes - 1 representative = 4 duplicates)
        assert_eq!(clusters.len(), 5);
        for (_, dups) in &clusters {
            assert_eq!(dups.len(), 4);
        }

        // Test reading a different range (rows 25-50)
        let reader = hamming_clustering_for_range(&dataset, "hash", fragment_id, 25, 25, 0)
            .await
            .unwrap();
        let clusters = collect_clusters(reader);

        // Should have 5 clusters (groups 5-9)
        assert_eq!(clusters.len(), 5);
        for (_, dups) in &clusters {
            assert_eq!(dups.len(), 4);
        }
    }

    #[tokio::test]
    async fn test_hamming_clustering_for_range_invalid_fragment() {
        use arrow_array::{FixedSizeListArray, RecordBatchIterator, UInt8Array};
        use arrow_schema::{Field, Schema};
        use lance_arrow::FixedSizeListArrayExt;
        use std::sync::Arc;
        use tempfile::tempdir;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "hash",
            arrow_schema::DataType::FixedSizeList(
                Arc::new(Field::new("item", arrow_schema::DataType::UInt8, true)),
                8,
            ),
            false,
        )]));

        let values = UInt8Array::from(vec![0u8; 80]); // 10 rows * 8 bytes
        let hash_array = FixedSizeListArray::try_new_from_values(values, 8).unwrap();
        let batch =
            arrow_array::RecordBatch::try_new(schema.clone(), vec![Arc::new(hash_array)]).unwrap();

        let temp_dir = tempdir().unwrap();
        let uri = temp_dir.path().to_str().unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        crate::Dataset::write(reader, uri, None).await.unwrap();

        let dataset = crate::Dataset::open(uri).await.unwrap();

        // Test with non-existent fragment
        let result = hamming_clustering_for_range(&dataset, "hash", 999, 0, 10, 0).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("not found"), "Error: {}", err);

        // Test with out-of-range start_row
        let result = hamming_clustering_for_range(&dataset, "hash", 0, 1000, 10, 0).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("out of range"), "Error: {}", err);
    }
}
