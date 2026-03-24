// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Hamming distance clustering for IVF_FLAT indices.
//!
//! This module provides functionality to perform pairwise hamming distance
//! computation and clustering on specific partitions of IVF_FLAT indices.

use std::time::{Duration, Instant};

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_schema::DataType;
use lance_core::{Error, Result};
use lance_index::DatasetIndexExt;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::vector::VectorIndex;
use lance_index::vector::flat::index::{FlatBinQuantizer, FlatIndex};
use lance_index::vector::flat::storage::FLAT_COLUMN;
use lance_index::vector::storage::VectorStore;
use lance_linalg::distance::{
    ClusteringResult, PairwiseResult, cluster_pairwise_result, extract_hashes_from_fixed_list,
    pairwise_hamming_distance_parallel,
};
use rand::rng;
use rand::seq::index::sample;

use crate::dataset::Dataset;
use crate::index::DatasetIndexInternalExt;

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
/// A `ClusteringResult` containing clusters of similar row IDs.
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
) -> Result<ClusteringResult> {
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
        .open_vector_index(column, &index_meta.uuid.to_string(), &NoOpMetricsCollector)
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
    let storage = ivf_index.load_partition_storage(partition_id).await?;

    // Get row IDs
    let row_id_slice: Vec<u64> = storage.row_ids().copied().collect();

    if row_id_slice.is_empty() {
        return Ok(ClusteringResult {
            clusters: Vec::new(),
        });
    }

    // Get vectors from the storage batches
    let batches: Vec<_> = storage.to_batches()?.collect();
    if batches.is_empty() {
        return Ok(ClusteringResult {
            clusters: Vec::new(),
        });
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

    Ok(clustering)
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
        .open_vector_index(column, &index_meta.uuid.to_string(), &NoOpMetricsCollector)
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

/// Result of hamming clustering with timing information.
#[derive(Debug, Clone)]
pub struct HammingClusterResult {
    /// The clustering result.
    pub clustering: ClusteringResult,
    /// The pairwise result (edges found).
    pub pairwise: PairwiseResult,
    /// Number of rows processed.
    pub num_rows: usize,
    /// Total number of pairs compared.
    pub total_pairs: u64,
    /// Time spent reading data.
    pub read_time: Duration,
    /// Time spent extracting hashes.
    pub extract_time: Duration,
    /// Time spent computing pairwise distances.
    pub compute_time: Duration,
    /// Time spent clustering.
    pub cluster_time: Duration,
}

impl HammingClusterResult {
    /// Pairs compared per second during the compute phase.
    pub fn pairs_per_sec(&self) -> f64 {
        self.total_pairs as f64 / self.compute_time.as_secs_f64()
    }

    /// Total processing time (excluding read).
    pub fn processing_time(&self) -> Duration {
        self.extract_time + self.compute_time + self.cluster_time
    }

    /// Total time including read.
    pub fn total_time(&self) -> Duration {
        self.read_time + self.processing_time()
    }
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
/// A `HammingClusterResult` containing clusters and timing information.
pub async fn hamming_clustering_sampled(
    dataset: &Dataset,
    column: &str,
    sample_size: Option<usize>,
    hamming_threshold: u32,
) -> Result<HammingClusterResult> {
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

    // Stage 1: Read data
    let t_read_start = Instant::now();
    let (hashes, row_ids) = if use_sampling {
        // Random sample using take()
        let indices: Vec<u64> = sample(&mut rng(), total_rows, effective_sample)
            .iter()
            .map(|i| i as u64)
            .collect();

        let projection =
            crate::dataset::ProjectionRequest::from_columns([column], dataset.schema());
        let batch = dataset.take(&indices, projection).await?;

        let hash_col = batch.column_by_name(column).ok_or_else(|| {
            Error::invalid_input(format!("Column '{}' not found in result", column))
        })?;
        let hashes_arr = hash_col.as_fixed_size_list();
        let hashes = extract_hashes_from_fixed_list(hashes_arr)?;

        (hashes, indices)
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
    let read_time = t_read_start.elapsed();

    // Stage 2: Already extracted hashes during read
    let extract_time = Duration::ZERO; // Hashes extracted during read

    let num_rows = hashes.len();
    if num_rows < 2 {
        return Ok(HammingClusterResult {
            clustering: ClusteringResult {
                clusters: Vec::new(),
            },
            pairwise: PairwiseResult::default(),
            num_rows,
            total_pairs: 0,
            read_time,
            extract_time,
            compute_time: Duration::ZERO,
            cluster_time: Duration::ZERO,
        });
    }

    let total_pairs = (num_rows as u64) * (num_rows as u64 - 1) / 2;

    // Stage 3: Compute pairwise hamming distances
    let t_compute_start = Instant::now();
    let pairwise =
        pairwise_hamming_distance_parallel(&hashes, Some(&row_ids), Some(hamming_threshold));
    let compute_time = t_compute_start.elapsed();

    // Stage 4: Cluster edges
    let t_cluster_start = Instant::now();
    let clustering = cluster_pairwise_result(&pairwise);
    let cluster_time = t_cluster_start.elapsed();

    Ok(HammingClusterResult {
        clustering,
        pairwise,
        num_rows,
        total_pairs,
        read_time,
        extract_time,
        compute_time,
        cluster_time,
    })
}

/// Perform pairwise hamming distance clustering on provided hashes (no I/O).
///
/// This is useful for benchmarking the pure compute performance without I/O.
///
/// # Arguments
///
/// * `hashes` - Vector of 64-bit hash values
/// * `row_ids` - Optional row IDs (defaults to indices if None)
/// * `hamming_threshold` - Maximum hamming distance to consider as similar
///
/// # Returns
///
/// A `HammingClusterResult` containing clusters and timing information.
pub fn hamming_cluster_hashes(
    hashes: &[u64],
    row_ids: Option<&[u64]>,
    hamming_threshold: u32,
) -> HammingClusterResult {
    let num_rows = hashes.len();
    if num_rows < 2 {
        return HammingClusterResult {
            clustering: ClusteringResult {
                clusters: Vec::new(),
            },
            pairwise: PairwiseResult::default(),
            num_rows,
            total_pairs: 0,
            read_time: Duration::ZERO,
            extract_time: Duration::ZERO,
            compute_time: Duration::ZERO,
            cluster_time: Duration::ZERO,
        };
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

    HammingClusterResult {
        clustering,
        pairwise,
        num_rows,
        total_pairs,
        read_time: Duration::ZERO,
        extract_time: Duration::ZERO,
        compute_time,
        cluster_time,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_cluster_hashes_basic() {
        // Create some test hashes with known distances
        let hashes = vec![
            0b0000u64, // hash 0
            0b0001u64, // hash 1 - distance 1 from hash 0
            0b0011u64, // hash 2 - distance 1 from hash 1, distance 2 from hash 0
            0b1111u64, // hash 3 - distance 2 from hash 2, distance 4 from hash 0
        ];

        let result = hamming_cluster_hashes(&hashes, None, 1);

        // With threshold 1, pairs (0,1) and (1,2) should be connected
        // This forms one cluster: {0, 1, 2}
        assert_eq!(result.num_rows, 4);
        assert_eq!(result.total_pairs, 6); // C(4,2) = 6
        assert_eq!(result.clustering.num_clusters(), 1);
        assert_eq!(result.clustering.num_duplicates(), 2); // 2 duplicates in the cluster
    }

    #[test]
    fn test_hamming_cluster_hashes_no_clusters() {
        // All hashes are far apart
        let hashes = vec![
            0x0000000000000000u64,
            0xFFFFFFFFFFFFFFFFu64,
            0xAAAAAAAAAAAAAAAAu64,
        ];

        let result = hamming_cluster_hashes(&hashes, None, 5);

        // With threshold 5, no pairs should be connected (min distance is 32)
        assert_eq!(result.clustering.num_clusters(), 0);
        assert_eq!(result.pairwise.len(), 0);
    }

    #[test]
    fn test_hamming_cluster_hashes_with_row_ids() {
        let hashes = vec![0b0000u64, 0b0001u64];
        let row_ids = vec![100u64, 200u64];

        let result = hamming_cluster_hashes(&hashes, Some(&row_ids), 1);

        assert_eq!(result.clustering.num_clusters(), 1);
        assert_eq!(result.clustering.clusters[0].representative, 100);
        assert_eq!(result.clustering.clusters[0].duplicates, vec![200]);
    }

    #[tokio::test]
    async fn test_hamming_cluster_partition_invalid_column() {
        // Integration tests would require a real dataset with an IVF_FLAT index
        // Unit tests verify error handling for edge cases
    }
}
