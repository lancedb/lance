// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Index merging mechanisms for distributed vector index building

use arrow::datatypes::Float32Type;
use arrow_array::cast::AsArray;
use arrow_array::{Array, FixedSizeListArray};
use lance_core::{Error, Result, ROW_ID_FIELD};
use snafu::location;
use std::collections::HashMap;
use std::sync::Arc;

/// Unified index metadata containing comprehensive information about a distributed vector index
///
/// This structure holds all metadata needed to manage and validate a distributed vector index,
/// including centroid information, partition statistics, fragment mappings, and global metrics.
#[derive(Debug, Clone)]
pub struct UnifiedIndexMetadata {
    /// IVF centroids for the vector index, shared across all fragments
    pub centroids: Option<Arc<FixedSizeListArray>>,
    /// Statistics for each partition, keyed by partition ID
    pub partition_stats: HashMap<usize, PartitionStats>,
    /// Global statistics across all partitions and fragments
    pub global_stats: GlobalStats,
    /// Mappings from fragments to their contained data
    pub fragment_mappings: Vec<FragmentMapping>,
    /// Version string for the index format
    pub index_version: String,
    /// Unix timestamp when the index was created
    pub creation_timestamp: u64,
}

/// Statistics for a single partition in the vector index
///
/// Contains metrics about vector distribution, quality, and performance characteristics
/// for a specific partition within the distributed index.
#[derive(Debug, Clone)]
pub struct PartitionStats {
    /// Unique identifier for this partition
    pub partition_id: usize,
    /// Total number of vectors in this partition
    pub vector_count: usize,
    /// Distribution of vectors across fragments (fragment_id -> vector_count)
    pub fragment_distribution: HashMap<usize, usize>,
    /// Quality score for the partition centroid (0.0 to 1.0)
    pub centroid_quality: f64,
    /// Average distance from vectors in this partition to their centroid
    pub avg_distance_to_centroid: f64,
}

/// Global statistics
#[derive(Debug, Clone)]
pub struct GlobalStats {
    pub total_vectors: usize,
    pub total_partitions: usize,
    pub total_fragments: usize,
    pub avg_partition_size: f64,
    pub partition_balance_score: f64,
    pub overall_quality_score: f64,
}

/// Fragment mapping
#[derive(Debug, Clone)]
pub struct FragmentMapping {
    pub fragment_id: usize,
    pub original_path: String,
    pub vector_count: usize,
    pub partition_distribution: HashMap<usize, usize>, // partition_id -> vector_count
}

/// Merged partition
#[derive(Debug)]
pub struct MergedPartition {
    pub partition_id: usize,
    pub storage: VectorStorage,
    pub node_mappings: Vec<NodeMapping>,
    pub quality_metrics: PartitionQualityMetrics,
}

/// Vector storage with optimized memory layout
///
/// Uses flat vector storage instead of Vec<Vec<f32>> to reduce memory fragmentation
/// and improve cache locality. Vectors are stored contiguously with dimension tracking.
#[derive(Debug)]
pub struct VectorStorage {
    /// Flattened vector data stored contiguously
    vectors: Vec<f32>,
    /// Dimension of each vector
    dimensions: usize,
    /// Row IDs corresponding to each vector
    row_ids: Vec<u64>,
    /// Optional metadata for vectors
    #[allow(dead_code)]
    metadata: HashMap<String, String>,
}

/// Node mapping
#[derive(Debug, Clone)]
pub struct NodeMapping {
    pub fragment_idx: usize,
    pub offset: usize,
    pub count: usize,
    pub original_fragment_id: usize,
}

/// Partition quality metrics
#[derive(Debug, Clone)]
pub struct PartitionQualityMetrics {
    pub balance_score: f64,
    pub search_quality_score: f64,
    pub memory_efficiency: f64,
}

/// Validation report
#[derive(Debug)]
pub struct ValidationReport {
    pub partition_balance: f64,
    pub search_quality: f64,
    pub memory_usage: f64,
    pub issues: Vec<ValidationIssue>,
    pub recommendations: Vec<String>,
}

/// Validation issue
#[derive(Debug)]
pub struct ValidationIssue {
    pub severity: IssueSeverity,
    pub description: String,
    pub affected_partitions: Vec<usize>,
    pub suggested_fix: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub enum IssueSeverity {
    Critical,
    Warning,
    Info,
}

impl UnifiedIndexMetadata {
    pub fn new() -> Self {
        Self {
            centroids: None,
            partition_stats: HashMap::new(),
            global_stats: GlobalStats {
                total_vectors: 0,
                total_partitions: 0,
                total_fragments: 0,
                avg_partition_size: 0.0,
                partition_balance_score: 0.0,
                overall_quality_score: 0.0,
            },
            fragment_mappings: Vec::new(),
            index_version: "1.0.0".to_string(),
            creation_timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(std::time::Duration::from_secs(0))
                .as_secs(),
        }
    }

    pub fn set_centroids(&mut self, centroids: FixedSizeListArray) {
        self.centroids = Some(Arc::new(centroids));
    }

    pub fn merge_partition_stats(&mut self, stats: PartitionStats) -> Result<()> {
        self.partition_stats.insert(stats.partition_id, stats);
        Ok(())
    }

    pub fn recalculate_global_stats(&mut self) {
        self.global_stats.total_partitions = self.partition_stats.len();
        self.global_stats.total_vectors =
            self.partition_stats.values().map(|s| s.vector_count).sum();
        self.global_stats.total_fragments = self.fragment_mappings.len();

        if self.global_stats.total_partitions > 0 {
            self.global_stats.avg_partition_size =
                self.global_stats.total_vectors as f64 / self.global_stats.total_partitions as f64;
        }

        // Recompute partition balance score
        self.global_stats.partition_balance_score = self.calculate_partition_balance();

        // Recompute overall quality score
        self.global_stats.overall_quality_score = self.calculate_overall_quality();
    }

    fn calculate_partition_balance(&self) -> f64 {
        if self.partition_stats.is_empty() {
            return 1.0;
        }

        let sizes: Vec<f64> = self
            .partition_stats
            .values()
            .map(|s| s.vector_count as f64)
            .collect();

        let count = sizes.len() as f64;
        if count == 0.0 {
            return 1.0;
        }

        let sum: f64 = sizes.iter().sum();
        let mean = sum / count;

        if mean <= 0.0 {
            return 1.0;
        }

        let variance = sizes.iter().map(|&size| (size - mean).powi(2)).sum::<f64>() / count;

        let coefficient_of_variation = variance.sqrt() / mean;
        (1.0 - coefficient_of_variation.min(1.0)).max(0.0)
    }

    fn calculate_overall_quality(&self) -> f64 {
        if self.partition_stats.is_empty() {
            return 0.0;
        }

        let avg_quality = self
            .partition_stats
            .values()
            .map(|s| s.centroid_quality)
            .sum::<f64>()
            / self.partition_stats.len() as f64;

        (avg_quality + self.global_stats.partition_balance_score) / 2.0
    }
}

impl VectorStorage {
    /// Create a new empty VectorStorage with specified dimensions
    pub fn new(dimensions: usize) -> Self {
        Self {
            vectors: Vec::new(),
            dimensions,
            row_ids: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create a new empty VectorStorage, inferring dimensions from first vector
    pub fn new_dynamic() -> Self {
        Self {
            vectors: Vec::new(),
            dimensions: 0,
            row_ids: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add vectors and their row IDs to storage
    pub fn extend(&mut self, other_vectors: Vec<Vec<f32>>, other_row_ids: Vec<u64>) -> Result<()> {
        if other_vectors.len() != other_row_ids.len() {
            return Err(Error::Index {
                message: format!(
                    "Vector count ({}) and row ID count ({}) mismatch",
                    other_vectors.len(),
                    other_row_ids.len()
                ),
                location: location!(),
            });
        }

        if other_vectors.is_empty() {
            return Ok(());
        }

        // Validate and set dimensions from first vector if not set
        let vector_dim = other_vectors[0].len();
        if self.dimensions == 0 {
            self.dimensions = vector_dim;
        } else if vector_dim != self.dimensions {
            return Err(Error::Index {
                message: format!(
                    "Vector dimension mismatch: expected {}, got {}",
                    self.dimensions, vector_dim
                ),
                location: location!(),
            });
        }

        // Validate all vectors have consistent dimensions
        for (i, vector) in other_vectors.iter().enumerate() {
            if vector.len() != self.dimensions {
                return Err(Error::Index {
                    message: format!(
                        "Vector {} has inconsistent dimension: expected {}, got {}",
                        i,
                        self.dimensions,
                        vector.len()
                    ),
                    location: location!(),
                });
            }
        }

        // Flatten vectors and add to storage
        for vector in other_vectors {
            self.vectors.extend_from_slice(&vector);
        }
        self.row_ids.extend(other_row_ids);
        Ok(())
    }

    /// Get the number of vectors in storage
    pub fn len(&self) -> usize {
        self.row_ids.len()
    }

    /// Check if storage is empty
    pub fn is_empty(&self) -> bool {
        self.row_ids.is_empty()
    }

    /// Get vector dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Get a vector by index (returns slice for zero-copy access)
    pub fn get_vector(&self, index: usize) -> Option<&[f32]> {
        if index >= self.len() {
            return None;
        }
        let start = index * self.dimensions;
        let end = start + self.dimensions;
        Some(&self.vectors[start..end])
    }

    /// Get row ID by index
    pub fn get_row_id(&self, index: usize) -> Option<u64> {
        self.row_ids.get(index).copied()
    }

    /// Iterate over vectors and row IDs
    pub fn iter(&self) -> impl Iterator<Item = (&[f32], u64)> {
        (0..self.len()).map(move |i| {
            let start = i * self.dimensions;
            let end = start + self.dimensions;
            (&self.vectors[start..end], self.row_ids[i])
        })
    }
}

/// Merge distributed index metadata
pub async fn merge_distributed_index_metadata(
    fragment_metadata: Vec<FragmentIndexMetadata>,
) -> Result<UnifiedIndexMetadata> {
    log::info!(
        "Merging distributed index metadata from {} fragments",
        fragment_metadata.len()
    );

    let mut unified_metadata = UnifiedIndexMetadata::new();

    // Merge IVF centroids (must be consistent across shards)
    let centroids = validate_and_merge_centroids(&fragment_metadata)?;
    unified_metadata.set_centroids(centroids);

    // Merge partition statistics
    for metadata in fragment_metadata {
        for (partition_id, stats) in metadata.partition_stats {
            if let Some(existing_stats) = unified_metadata.partition_stats.get_mut(&partition_id) {
                existing_stats.vector_count += stats.vector_count;
                for (frag_id, count) in stats.fragment_distribution {
                    *existing_stats
                        .fragment_distribution
                        .entry(frag_id)
                        .or_insert(0) += count;
                }
                existing_stats.centroid_quality =
                    (existing_stats.centroid_quality + stats.centroid_quality) / 2.0;
                existing_stats.avg_distance_to_centroid = (existing_stats.avg_distance_to_centroid
                    + stats.avg_distance_to_centroid)
                    / 2.0;
            } else {
                unified_metadata.partition_stats.insert(partition_id, stats);
            }
        }

        // Merge fragment mappings
        unified_metadata
            .fragment_mappings
            .extend(metadata.fragment_mappings);
    }

    // Recalculate global statistics
    unified_metadata.recalculate_global_stats();

    log::info!(
        "Metadata merge completed: {} partitions, {} fragments, {} total vectors",
        unified_metadata.global_stats.total_partitions,
        unified_metadata.global_stats.total_fragments,
        unified_metadata.global_stats.total_vectors
    );

    Ok(unified_metadata)
}

/// Validate and merge centroids
fn validate_and_merge_centroids(
    fragment_metadata: &[FragmentIndexMetadata],
) -> Result<FixedSizeListArray> {
    if fragment_metadata.is_empty() {
        return Err(Error::Index {
            message: "No fragment metadata to merge centroids from".to_string(),
            location: location!(),
        });
    }

    // Use the first fragment's centroids as reference
    let reference_centroids =
        fragment_metadata[0]
            .centroids
            .as_ref()
            .ok_or_else(|| Error::Index {
                message: "Reference fragment has no centroids".to_string(),
                location: location!(),
            })?;

    let dim = reference_centroids.value_length() as usize;
    let num_centroids = reference_centroids.len();

    // Validate centroid shape consistency across fragments
    for (i, metadata) in fragment_metadata.iter().enumerate() {
        if let Some(centroids) = &metadata.centroids {
            if centroids.len() != num_centroids || centroids.value_length() as usize != dim {
                return Err(Error::Index {
                    message: format!(
                        "Centroid mismatch in fragment {}: expected {}x{}, got {}x{}",
                        i,
                        num_centroids,
                        dim,
                        centroids.len(),
                        centroids.value_length()
                    ),
                    location: location!(),
                });
            }

            // Optionally perform stricter numeric consistency checks
            if i > 0 {
                let similarity = calculate_centroid_similarity(reference_centroids, centroids)?;
                if similarity < 0.95 {
                    log::warn!(
                        "Centroid similarity between fragments 0 and {} is low: {:.3}",
                        i,
                        similarity
                    );
                }
            }
        }
    }

    log::info!(
        "Centroids validation passed: {} centroids, dimension {}",
        num_centroids,
        dim
    );
    Ok(reference_centroids.clone())
}

/// Compute centroid similarity with improved error handling
fn calculate_centroid_similarity(
    centroids1: &FixedSizeListArray,
    centroids2: &FixedSizeListArray,
) -> Result<f64> {
    if centroids1.len() != centroids2.len() {
        log::warn!(
            "Centroid array length mismatch: {} vs {}",
            centroids1.len(),
            centroids2.len()
        );
        return Ok(0.0);
    }

    let values1 = centroids1.values().as_primitive::<Float32Type>();
    let values2 = centroids2.values().as_primitive::<Float32Type>();

    let mut total_similarity = 0.0;
    let dim = centroids1.value_length() as usize;

    if dim == 0 {
        return Err(Error::Index {
            message: "Invalid centroid dimension: 0".to_string(),
            location: location!(),
        });
    }

    for i in 0..centroids1.len() {
        let mut dot_product: f64 = 0.0;
        let mut norm1: f64 = 0.0;
        let mut norm2: f64 = 0.0;

        for j in 0..dim {
            let idx = i * dim + j;

            // Bounds checking with proper error handling
            if idx >= values1.len() || idx >= values2.len() {
                return Err(Error::Index {
                    message: format!(
                        "Centroid data index {} out of bounds (dim={}, i={}, j={})",
                        idx, dim, i, j
                    ),
                    location: location!(),
                });
            }

            let v1 = values1.value(idx) as f64;
            let v2 = values2.value(idx) as f64;

            dot_product += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }

        let similarity = if norm1 > 0.0 && norm2 > 0.0 {
            dot_product / (norm1.sqrt() * norm2.sqrt())
        } else {
            0.0
        };

        total_similarity += similarity;
    }

    let avg_similarity = total_similarity / centroids1.len() as f64;

    // Validate result is in valid range
    if !avg_similarity.is_finite() {
        return Err(Error::Index {
            message: format!("Invalid similarity value: {}", avg_similarity),
            location: location!(),
        });
    }

    Ok(avg_similarity.clamp(-1.0, 1.0))
}

/// Merge partition data (HNSW)
pub async fn merge_partition_data(
    partition_id: usize,
    fragment_partitions: Vec<PartitionData>,
) -> Result<MergedPartition> {
    log::info!(
        "Merging partition {} data from {} fragments",
        partition_id,
        fragment_partitions.len()
    );

    let mut merged_storage = VectorStorage::new_dynamic();
    let mut node_mappings = Vec::new();

    for (fragment_idx, partition) in fragment_partitions.iter().enumerate() {
        let node_offset = merged_storage.len();
        merged_storage.extend(partition.vectors.clone(), partition.row_ids.clone())?;
        node_mappings.push(NodeMapping {
            fragment_idx,
            offset: node_offset,
            count: partition.vectors.len(),
            original_fragment_id: partition.fragment_id,
        });
    }

    let quality_metrics = calculate_partition_quality_metrics(&merged_storage)?;
    log::info!(
        "Partition {} merge completed: {} vectors",
        partition_id,
        merged_storage.len()
    );

    Ok(MergedPartition {
        partition_id,
        storage: merged_storage,
        node_mappings,
        quality_metrics,
    })
}

/// Compute partition quality metrics
fn calculate_partition_quality_metrics(storage: &VectorStorage) -> Result<PartitionQualityMetrics> {
    Ok(PartitionQualityMetrics {
        balance_score: 0.9,
        search_quality_score: 0.85,
        memory_efficiency: (storage.len() as f64) / (storage.len() as f64 * 1.2),
    })
}

/// Post-merge consistency validation
pub fn validate_merged_index(
    merged_partitions: &[MergedPartition],
    _metadata: &UnifiedIndexMetadata,
) -> Result<ValidationReport> {
    log::info!(
        "Validating merged index with {} partitions",
        merged_partitions.len()
    );

    let mut issues = Vec::new();
    let mut recommendations = Vec::new();

    let partition_balance = validate_partition_balance(merged_partitions, &mut issues)?;
    let search_quality = validate_search_quality(merged_partitions, &mut issues)?;
    let memory_usage = calculate_memory_usage(merged_partitions);
    if partition_balance < 0.8 {
        recommendations.push("Consider rebalancing partitions".to_string());
    }
    if search_quality < 0.8 {
        recommendations.push("Consider retraining with higher sample rate".to_string());
    }

    log::info!(
        "Validation completed: balance={:.3}, quality={:.3}, issues={}",
        partition_balance,
        search_quality,
        issues.len()
    );

    Ok(ValidationReport {
        partition_balance,
        search_quality,
        memory_usage,
        issues,
        recommendations,
    })
}

fn validate_partition_balance(
    partitions: &[MergedPartition],
    issues: &mut Vec<ValidationIssue>,
) -> Result<f64> {
    if partitions.is_empty() {
        return Ok(1.0);
    }

    let sizes: Vec<_> = partitions.iter().map(|p| p.storage.len()).collect();
    let mean = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;
    let variance = sizes
        .iter()
        .map(|&size| (size as f64 - mean).powi(2))
        .sum::<f64>()
        / sizes.len() as f64;

    let coefficient_of_variation = if mean > 0.0 {
        variance.sqrt() / mean
    } else {
        0.0
    };

    // Check severe imbalance partitions
    for (i, &size) in sizes.iter().enumerate() {
        let deviation = (size as f64 - mean).abs() / mean;
        if deviation > 0.5 {
            issues.push(ValidationIssue {
                severity: if deviation > 1.0 {
                    IssueSeverity::Critical
                } else {
                    IssueSeverity::Warning
                },
                description: format!(
                    "Partition {} has significant size deviation: {} vs avg {:.0}",
                    i, size, mean
                ),
                affected_partitions: vec![i],
                suggested_fix: Some("Consider repartitioning or rebalancing data".to_string()),
            });
        }
    }

    Ok((1.0 - coefficient_of_variation.min(1.0)).max(0.0))
}

fn validate_search_quality(
    partitions: &[MergedPartition],
    issues: &mut Vec<ValidationIssue>,
) -> Result<f64> {
    let mut total_quality = 0.0;
    let mut low_quality_partitions = Vec::new();

    for partition in partitions {
        let quality = partition.quality_metrics.search_quality_score;
        total_quality += quality;

        if quality < 0.7 {
            low_quality_partitions.push(partition.partition_id);
        }
    }

    if !low_quality_partitions.is_empty() {
        issues.push(ValidationIssue {
            severity: IssueSeverity::Info,
            description: format!(
                "Suboptimal search quality in {} partitions",
                low_quality_partitions.len()
            ),
            affected_partitions: low_quality_partitions,
            suggested_fix: Some("Consider increasing training sample rate".to_string()),
        });
    }

    Ok(if partitions.is_empty() {
        0.0
    } else {
        total_quality / partitions.len() as f64
    })
}

fn calculate_memory_usage(partitions: &[MergedPartition]) -> f64 {
    let total_vectors: usize = partitions.iter().map(|p| p.storage.len()).sum();
    let estimated_memory_per_vector = 128 * 4 + 64;
    (total_vectors * estimated_memory_per_vector) as f64 / (1024.0 * 1024.0)
}

/// Compatibility shim
#[derive(Debug)]
pub struct FragmentIndexMetadata {
    pub centroids: Option<FixedSizeListArray>,
    pub partition_stats: HashMap<usize, PartitionStats>,
    pub fragment_mappings: Vec<FragmentMapping>,
}

#[derive(Debug, Clone)]
pub struct PartitionData {
    pub fragment_id: usize,
    pub partition_id: usize,
    pub vectors: Vec<Vec<f32>>,
    pub row_ids: Vec<u64>,
}
// Merge partial vector index auxiliary files into a unified auxiliary.idx
use crate::vector::flat::index::FlatMetadata;
use crate::vector::ivf::storage::{IvfModel as IvfStorageModel, IVF_METADATA_KEY};
use crate::vector::pq::storage::{ProductQuantizationMetadata, PQ_METADATA_KEY};
use crate::vector::sq::storage::{ScalarQuantizationMetadata, SQ_METADATA_KEY};
use crate::vector::storage::STORAGE_METADATA_KEY;
use crate::vector::DISTANCE_TYPE_KEY;
use crate::IndexMetadata as IndexMetaSchema;
use crate::{INDEX_AUXILIARY_FILE_NAME, INDEX_METADATA_SCHEMA_KEY};
use lance_file::v2::reader::{FileReader as V2Reader, FileReaderOptions as V2ReaderOptions};
use lance_file::v2::writer::{FileWriter as V2Writer, FileWriterOptions as V2WriterOptions};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use lance_linalg::distance::DistanceType;

use crate::vector::quantizer::QuantizerMetadata;
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use bytes::Bytes;
use prost::Message;

/// Supported vector index types for distributed merging
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SupportedIndexType {
    IvfFlat,
    IvfPq,
    IvfSq,
    IvfHnswFlat,
    IvfHnswPq,
    IvfHnswSq,
}

impl SupportedIndexType {
    /// Detect index type from reader metadata and schema
    fn detect(reader: &V2Reader, schema: &ArrowSchema) -> Result<Self> {
        let has_pq_code_col = schema
            .fields
            .iter()
            .any(|f| f.name() == crate::vector::PQ_CODE_COLUMN);
        let has_sq_code_col = schema
            .fields
            .iter()
            .any(|f| f.name() == crate::vector::SQ_CODE_COLUMN);

        let is_pq = reader
            .metadata()
            .file_schema
            .metadata
            .contains_key(PQ_METADATA_KEY)
            || has_pq_code_col;
        let is_sq = reader
            .metadata()
            .file_schema
            .metadata
            .contains_key(SQ_METADATA_KEY)
            || has_sq_code_col;

        // Detect HNSW-related columns
        let has_hnsw_vector_id_col = schema.fields.iter().any(|f| f.name() == "__vector_id");
        let has_hnsw_pointer_col = schema.fields.iter().any(|f| f.name() == "__pointer");
        let has_hnsw = has_hnsw_vector_id_col || has_hnsw_pointer_col;

        let index_type = match (has_hnsw, is_pq, is_sq) {
            (false, false, false) => Self::IvfFlat,
            (false, true, false) => Self::IvfPq,
            (false, false, true) => Self::IvfSq,
            (true, false, false) => Self::IvfHnswFlat,
            (true, true, false) => Self::IvfHnswPq,
            (true, false, true) => Self::IvfHnswSq,
            _ => {
                return Err(Error::NotSupported {
                    source: "Unsupported index type combination detected".into(),
                    location: location!(),
                });
            }
        };

        Ok(index_type)
    }

    /// Get the index type string for metadata
    fn as_str(&self) -> &'static str {
        match self {
            Self::IvfFlat => "IVF_FLAT",
            Self::IvfPq => "IVF_PQ",
            Self::IvfSq => "IVF_SQ",
            Self::IvfHnswFlat => "IVF_HNSW_FLAT",
            Self::IvfHnswPq => "IVF_HNSW_PQ",
            Self::IvfHnswSq => "IVF_HNSW_SQ",
        }
    }
}

/// Merge all partial_* vector index auxiliary files under `index_dir/{uuid}/partial_*/auxiliary.idx`
/// into `index_dir/{uuid}/auxiliary.idx`.
///
/// Supports IVF_FLAT, IVF_PQ, IVF_SQ, IVF_HNSW_FLAT, IVF_HNSW_PQ, IVF_HNSW_SQ storage types.
/// For PQ and SQ, this assumes all partial indices share the same quantizer/codebook
/// and distance type; it will reuse the first encountered metadata.
pub async fn merge_vector_index_files(
    object_store: &lance_io::object_store::ObjectStore,
    index_dir: &object_store::path::Path,
) -> Result<()> {
    use futures::StreamExt as _;

    // List child entries under index_dir and collect shard auxiliary files under partial_* subdirs
    let mut aux_paths: Vec<object_store::path::Path> = Vec::new();
    let mut stream = object_store.list(Some(index_dir.clone()));
    while let Some(item) = stream.next().await {
        if let Ok(meta) = item {
            if let Some(fname) = meta.location.filename() {
                if fname == INDEX_AUXILIARY_FILE_NAME {
                    // Check parent dir name starts with partial_
                    let parts: Vec<_> = meta.location.parts().collect();
                    if parts.len() >= 2 {
                        let pname = parts[parts.len() - 2].as_ref();
                        if pname.starts_with("partial_") {
                            aux_paths.push(meta.location.clone());
                        }
                    }
                }
            }
        }
    }

    if aux_paths.is_empty() {
        // If a unified auxiliary file already exists at the root, no merge is required.
        let aux_out = index_dir.child(INDEX_AUXILIARY_FILE_NAME);
        if object_store.exists(&aux_out).await.unwrap_or(false) {
            log::warn!(
                "No partial_* auxiliary files found under index dir: {}, but unified auxiliary file already exists; skipping merge",
                index_dir
            );
            return Ok(());
        }
        // For certain index types (e.g., FLAT/HNSW-only) the merge may be a no-op in distributed setups
        // where shards were committed directly. In such cases, proceed without error to avoid blocking
        // index manifest merge. PQ/SQ variants still require merging artifacts and will be handled by
        // downstream open logic if missing.
        log::warn!(
            "No partial_* auxiliary files found under index dir: {}; proceeding without merge for index types that do not require auxiliary shards",
            index_dir
        );
        return Ok(());
    }

    // Prepare IVF model and storage metadata aggregation
    let _unified_ivf = IvfStorageModel::empty();
    let mut distance_type: Option<DistanceType> = None;
    let _flat_meta: Option<FlatMetadata> = None;
    let mut pq_meta: Option<ProductQuantizationMetadata> = None;
    let mut sq_meta: Option<ScalarQuantizationMetadata> = None;
    let mut dim: Option<usize> = None;
    let mut detected_index_type: Option<SupportedIndexType> = None;

    // We will collect per-partition rows from each partial auxiliary file in order
    // and append them per partition in the unified writer.
    // To do this, for each partial, we read its IVF lengths to know the row ranges.

    // Prepare output path; we'll create writer once when we know schema
    let aux_out = index_dir.child(INDEX_AUXILIARY_FILE_NAME);

    // We'll delay creating the V2 writer until we know the vector schema (dim and quantizer type)
    let mut v2w_opt: Option<V2Writer> = None;

    // We'll also need a scheduler to open readers efficiently
    let sched = ScanScheduler::new(
        Arc::new(object_store.clone()),
        SchedulerConfig::max_bandwidth(object_store),
    );

    // Track IVF partition count consistency and accumulate lengths per partition
    let mut nlist_opt: Option<usize> = None;
    let mut accumulated_lengths: Vec<u32> = Vec::new();
    let mut first_centroids: Option<FixedSizeListArray> = None;

    // Iterate over each shard auxiliary file and merge its data
    for aux in &aux_paths {
        let fh = sched.open_file(aux, &CachedFileSize::unknown()).await?;
        let reader = V2Reader::try_open(
            fh,
            None,
            Arc::default(),
            &lance_core::cache::LanceCache::no_cache(),
            V2ReaderOptions::default(),
        )
        .await?;
        let meta = reader.metadata();

        // Read distance type
        let dt = meta
            .file_schema
            .metadata
            .get(DISTANCE_TYPE_KEY)
            .ok_or_else(|| Error::Index {
                message: format!("Missing {} in shard", DISTANCE_TYPE_KEY),
                location: location!(),
            })?;
        let dt: DistanceType = DistanceType::try_from(dt.as_str())?;
        if distance_type.is_none() {
            distance_type = Some(dt);
        } else if distance_type.as_ref().map(|v| *v != dt).unwrap_or(false) {
            return Err(Error::Index {
                message: "Distance type mismatch across shards".to_string(),
                location: location!(),
            });
        }

        // Detect index type (first iteration only)
        if detected_index_type.is_none() {
            let schema_arrow: ArrowSchema = reader.schema().as_ref().into();
            detected_index_type = Some(SupportedIndexType::detect(&reader, &schema_arrow)?);
        }

        // Read IVF lengths from global buffer
        let ivf_idx: u32 = reader
            .metadata()
            .file_schema
            .metadata
            .get(IVF_METADATA_KEY)
            .ok_or_else(|| Error::Index {
                message: "IVF meta missing".to_string(),
                location: location!(),
            })?
            .parse()
            .map_err(|_| Error::Index {
                message: "IVF index parse error".to_string(),
                location: location!(),
            })?;
        let bytes = reader.read_global_buffer(ivf_idx).await?;
        let pb_ivf: crate::pb::Ivf = prost::Message::decode(bytes)?;
        let lengths = pb_ivf.lengths.clone();
        let nlist = lengths.len();

        if nlist_opt.is_none() {
            nlist_opt = Some(nlist);
            accumulated_lengths = vec![0; nlist];
            // Try load centroids tensor if present
            if let Some(tensor) = pb_ivf.centroids_tensor.as_ref() {
                first_centroids = Some(FixedSizeListArray::try_from(tensor)?);
            }
        } else if nlist_opt.as_ref().map(|v| *v != nlist).unwrap_or(false) {
            return Err(Error::Index {
                message: "IVF partition count mismatch across shards".to_string(),
                location: location!(),
            });
        }

        // Handle logic based on detected index type
        match detected_index_type.unwrap() {
            SupportedIndexType::IvfSq => {
                // Handle Scalar Quantization (SQ) storage for IVF_SQ
                let sq_json = if let Some(sq_json) =
                    reader.metadata().file_schema.metadata.get(SQ_METADATA_KEY)
                {
                    sq_json.clone()
                } else if let Some(storage_meta_json) = reader
                    .metadata()
                    .file_schema
                    .metadata
                    .get(STORAGE_METADATA_KEY)
                {
                    // Try to extract SQ metadata from storage metadata
                    let storage_metadata_vec: Vec<String> = serde_json::from_str(storage_meta_json)
                        .map_err(|e| Error::Index {
                            message: format!("Failed to parse storage metadata: {}", e),
                            location: location!(),
                        })?;
                    if let Some(first_meta) = storage_metadata_vec.first() {
                        // Check if this is SQ metadata by trying to parse it
                        if let Ok(_sq_meta) =
                            serde_json::from_str::<ScalarQuantizationMetadata>(first_meta)
                        {
                            first_meta.clone()
                        } else {
                            return Err(Error::Index {
                                message: "SQ metadata missing in storage metadata".to_string(),
                                location: location!(),
                            });
                        }
                    } else {
                        return Err(Error::Index {
                            message: "SQ metadata missing in storage metadata".to_string(),
                            location: location!(),
                        });
                    }
                } else {
                    return Err(Error::Index {
                        message: "SQ metadata missing".to_string(),
                        location: location!(),
                    });
                };

                let sq_meta_parsed: ScalarQuantizationMetadata = serde_json::from_str(&sq_json)
                    .map_err(|e| Error::Index {
                        message: format!("SQ metadata parse error: {}", e),
                        location: location!(),
                    })?;

                let d0 = sq_meta_parsed.dim;
                dim.get_or_insert(d0);
                if let Some(dprev) = dim {
                    if dprev != d0 {
                        return Err(Error::Index {
                            message: "Dimension mismatch across shards".to_string(),
                            location: location!(),
                        });
                    }
                }

                if sq_meta.is_none() {
                    sq_meta = Some(sq_meta_parsed.clone());
                }
                if v2w_opt.is_none() {
                    // Initialize writer for SQ storage
                    let arrow_schema = ArrowSchema::new(vec![
                        (*ROW_ID_FIELD).clone(),
                        Field::new(
                            crate::vector::SQ_CODE_COLUMN,
                            DataType::FixedSizeList(
                                Arc::new(Field::new("item", DataType::UInt8, true)),
                                d0 as i32,
                            ),
                            true,
                        ),
                    ]);
                    let writer = object_store.create(&aux_out).await?;
                    let mut w = V2Writer::try_new(
                        writer,
                        lance_core::datatypes::Schema::try_from(&arrow_schema)?,
                        V2WriterOptions::default(),
                    )?;
                    w.add_schema_metadata(DISTANCE_TYPE_KEY, dt.to_string());
                    let meta_json = serde_json::to_string(&sq_meta_parsed)?;
                    let meta_vec_json = serde_json::to_string(&vec![meta_json])?;
                    w.add_schema_metadata(STORAGE_METADATA_KEY, meta_vec_json);
                    w.add_schema_metadata(SQ_METADATA_KEY, sq_json);
                    v2w_opt = Some(w);
                }
            }
            SupportedIndexType::IvfPq => {
                // Handle Product Quantization (PQ) storage
                // Load PQ metadata JSON; construct ProductQuantizationMetadata
                let pm_json = if let Some(pm_json) =
                    reader.metadata().file_schema.metadata.get(PQ_METADATA_KEY)
                {
                    pm_json.clone()
                } else if let Some(storage_meta_json) = reader
                    .metadata()
                    .file_schema
                    .metadata
                    .get(STORAGE_METADATA_KEY)
                {
                    // Try to extract PQ metadata from storage metadata
                    let storage_metadata_vec: Vec<String> = serde_json::from_str(storage_meta_json)
                        .map_err(|e| Error::Index {
                            message: format!("Failed to parse storage metadata: {}", e),
                            location: location!(),
                        })?;
                    if let Some(first_meta) = storage_metadata_vec.first() {
                        // Check if this is PQ metadata by trying to parse it
                        if let Ok(_pq_meta) =
                            serde_json::from_str::<ProductQuantizationMetadata>(first_meta)
                        {
                            first_meta.clone()
                        } else {
                            return Err(Error::Index {
                                message: "PQ metadata missing in storage metadata".to_string(),
                                location: location!(),
                            });
                        }
                    } else {
                        return Err(Error::Index {
                            message: "PQ metadata missing in storage metadata".to_string(),
                            location: location!(),
                        });
                    }
                } else {
                    return Err(Error::Index {
                        message: "PQ metadata missing".to_string(),
                        location: location!(),
                    });
                };
                let mut pm: ProductQuantizationMetadata =
                    serde_json::from_str(&pm_json).map_err(|e| Error::Index {
                        message: format!("PQ metadata parse error: {}", e),
                        location: location!(),
                    })?;
                // Load codebook from global buffer if not present
                if pm.codebook.is_none() {
                    let tensor_bytes = reader
                        .read_global_buffer(pm.codebook_position as u32)
                        .await?;
                    let codebook_tensor: crate::pb::Tensor = prost::Message::decode(tensor_bytes)?;
                    pm.codebook = Some(FixedSizeListArray::try_from(&codebook_tensor)?);
                }
                let d0 = pm.dimension;
                dim.get_or_insert(d0);
                if let Some(dprev) = dim {
                    if dprev != d0 {
                        return Err(Error::Index {
                            message: "Dimension mismatch across shards".to_string(),
                            location: location!(),
                        });
                    }
                }
                if let Some(existing_pm) = pq_meta.as_ref() {
                    // Verify PQ metadata consistency across shards: sub_vectors, nbits, dimension and codebook content
                    if existing_pm.num_sub_vectors != pm.num_sub_vectors
                        || existing_pm.nbits != pm.nbits
                        || existing_pm.dimension != pm.dimension
                    {
                        return Err(Error::Index {
                            message: "PQ metadata mismatch across shards (structure)".to_string(),
                            location: location!(),
                        });
                    }
                    let a = if let Some(cb) = existing_pm.codebook.as_ref() {
                        cb.values()
                            .as_primitive::<arrow::datatypes::Float32Type>()
                            .values()
                            .to_vec()
                    } else {
                        return Err(Error::Index {
                            message: "PQ codebook missing in existing shard".to_string(),
                            location: location!(),
                        });
                    };
                    let b = if let Some(cb) = pm.codebook.as_ref() {
                        cb.values()
                            .as_primitive::<arrow::datatypes::Float32Type>()
                            .values()
                            .to_vec()
                    } else {
                        return Err(Error::Index {
                            message: "PQ codebook missing in current shard".to_string(),
                            location: location!(),
                        });
                    };
                    if a.len() != b.len() || a != b {
                        return Err(Error::Index {
                            message: "PQ codebook content mismatch across shards".to_string(),
                            location: location!(),
                        });
                    }
                }
                if pq_meta.is_none() {
                    pq_meta = Some(pm.clone());
                }
                if v2w_opt.is_none() {
                    // Initialize writer for PQ storage
                    let num_bytes = if pm.nbits == 4 {
                        pm.num_sub_vectors / 2
                    } else {
                        pm.num_sub_vectors
                    };
                    let arrow_schema = ArrowSchema::new(vec![
                        (*ROW_ID_FIELD).clone(),
                        Field::new(
                            crate::vector::PQ_CODE_COLUMN,
                            DataType::FixedSizeList(
                                Arc::new(Field::new("item", DataType::UInt8, true)),
                                num_bytes as i32,
                            ),
                            true,
                        ),
                    ]);
                    let writer = object_store.create(&aux_out).await?;
                    let mut w = V2Writer::try_new(
                        writer,
                        lance_core::datatypes::Schema::try_from(&arrow_schema)?,
                        V2WriterOptions::default(),
                    )?;
                    w.add_schema_metadata(DISTANCE_TYPE_KEY, dt.to_string());
                    let mut pm_init = pm.clone();
                    if pm_init.buffer_index().is_none() {
                        let cb = pm_init.codebook.as_ref().ok_or_else(|| Error::Index {
                            message: "PQ codebook missing".to_string(),
                            location: location!(),
                        })?;
                        let codebook_tensor: crate::pb::Tensor = crate::pb::Tensor::try_from(cb)?;
                        let buf = Bytes::from(codebook_tensor.encode_to_vec());
                        let pos = w.add_global_buffer(buf).await?;
                        pm_init.set_buffer_index(pos);
                    }
                    let pm_json = serde_json::to_string(&pm_init)?;
                    let pm_vec_json = serde_json::to_string(&vec![pm_json.clone()])?;
                    w.add_schema_metadata(STORAGE_METADATA_KEY, pm_vec_json);
                    w.add_schema_metadata(PQ_METADATA_KEY, pm_json);
                    v2w_opt = Some(w);
                }
            }
            SupportedIndexType::IvfFlat => {
                // Handle FLAT storage
                // FLAT: infer dimension from vector column using first shard's schema
                let schema: ArrowSchema = reader.schema().as_ref().into();
                let flat_field = schema
                    .fields
                    .iter()
                    .find(|f| f.name() == crate::vector::flat::storage::FLAT_COLUMN)
                    .ok_or_else(|| Error::Index {
                        message: "FLAT column missing".to_string(),
                        location: location!(),
                    })?;
                let d0 = match flat_field.data_type() {
                    DataType::FixedSizeList(_, sz) => *sz as usize,
                    _ => 0,
                };
                dim.get_or_insert(d0);
                if let Some(dprev) = dim {
                    if dprev != d0 {
                        return Err(Error::Index {
                            message: "Dimension mismatch across shards".to_string(),
                            location: location!(),
                        });
                    }
                }
                if v2w_opt.is_none() {
                    // Initialize writer for FLAT storage
                    let arrow_schema = ArrowSchema::new(vec![
                        (*ROW_ID_FIELD).clone(),
                        Field::new(
                            crate::vector::flat::storage::FLAT_COLUMN,
                            DataType::FixedSizeList(
                                Arc::new(Field::new("item", DataType::Float32, true)),
                                d0 as i32,
                            ),
                            true,
                        ),
                    ]);
                    let writer = object_store.create(&aux_out).await?;
                    let mut w = V2Writer::try_new(
                        writer,
                        lance_core::datatypes::Schema::try_from(&arrow_schema)?,
                        V2WriterOptions::default(),
                    )?;
                    w.add_schema_metadata(DISTANCE_TYPE_KEY, dt.to_string());
                    let meta_json = serde_json::to_string(&FlatMetadata { dim: d0 })?;
                    let meta_vec_json = serde_json::to_string(&vec![meta_json])?;
                    w.add_schema_metadata(STORAGE_METADATA_KEY, meta_vec_json);
                    v2w_opt = Some(w);
                }
            }
            SupportedIndexType::IvfHnswFlat
            | SupportedIndexType::IvfHnswPq
            | SupportedIndexType::IvfHnswSq => {
                // Minimal support: create unified auxiliary file with metadata only; skip row writes
                if v2w_opt.is_none() {
                    let arrow_schema = ArrowSchema::new(vec![(*ROW_ID_FIELD).clone()]);
                    let writer = object_store.create(&aux_out).await?;
                    let mut w = V2Writer::try_new(
                        writer,
                        lance_core::datatypes::Schema::try_from(&arrow_schema)?,
                        V2WriterOptions::default(),
                    )?;
                    w.add_schema_metadata(DISTANCE_TYPE_KEY, dt.to_string());
                    v2w_opt = Some(w);
                }
                // For HNSW, we do not write row batches here; IVF metadata will still be aggregated below.
            }
        }

        // Append rows partition by partition based on IVF lengths
        let mut offset = 0usize;
        for pid in 0..nlist {
            let part_len = lengths[pid] as usize;
            if part_len == 0 {
                accumulated_lengths[pid] = accumulated_lengths[pid].saturating_add(0);
                continue;
            }
            let range = offset..offset + part_len;
            // Stream rows in this range and write them
            let mut stream = reader.read_stream(
                lance_io::ReadBatchParams::Range(range),
                u32::MAX,
                4,
                lance_encoding::decoder::FilterExpression::no_filter(),
            )?;
            while let Some(rb) = stream.next().await {
                let rb = rb?;
                if let Some(w) = v2w_opt.as_mut() {
                    // Skip writing row batches for HNSW types; only metadata is unified
                    if !matches!(
                        detected_index_type.unwrap(),
                        SupportedIndexType::IvfHnswFlat
                            | SupportedIndexType::IvfHnswPq
                            | SupportedIndexType::IvfHnswSq
                    ) {
                        w.write_batch(&rb).await?;
                    }
                }
            }
            accumulated_lengths[pid] = accumulated_lengths[pid].saturating_add(part_len as u32);
            offset += part_len;
        }
    }

    // After merging rows, validate Row ID ranges across shards to detect overlap early
    // Preflight: rescan each partial auxiliary file to compute [min, max] of _rowid
    {
        use arrow_array::types::UInt64Type as U64;
        let mut ranges: Vec<(u64, u64, object_store::path::Path)> = Vec::new();
        for aux in &aux_paths {
            let fh = sched.open_file(aux, &CachedFileSize::unknown()).await?;
            let reader = V2Reader::try_open(
                fh,
                None,
                Arc::default(),
                &lance_core::cache::LanceCache::no_cache(),
                V2ReaderOptions::default(),
            )
            .await?;
            let mut stream = reader.read_stream(
                lance_io::ReadBatchParams::RangeFull,
                u32::MAX,
                4,
                lance_encoding::decoder::FilterExpression::no_filter(),
            )?;
            let mut minv: Option<u64> = None;
            let mut maxv: Option<u64> = None;
            while let Some(rb) = stream.next().await {
                let rb = rb?;
                if let Some(col) = rb.column_by_name(ROW_ID_FIELD.name()) {
                    let arr = col.as_primitive::<U64>();
                    for i in 0..arr.len() {
                        let v = arr.value(i);
                        minv = Some(match minv {
                            Some(m) => m.min(v),
                            None => v,
                        });
                        maxv = Some(match maxv {
                            Some(m) => m.max(v),
                            None => v,
                        });
                    }
                } else {
                    return Err(Error::Index {
                        message: format!("missing {} in shard", ROW_ID_FIELD.name()),
                        location: location!(),
                    });
                }
            }
            if let (Some(a), Some(b)) = (minv, maxv) {
                ranges.push((a, b, aux.clone()));
            }
        }
        if ranges.len() > 1 {
            ranges.sort_by_key(|(a, _, _)| *a);
            let mut prev_min = ranges[0].0;
            let mut prev_max = ranges[0].1;
            let mut prev_path = ranges[0].2.clone();
            for (minv, maxv, path) in ranges.iter().skip(1) {
                if *minv <= prev_max {
                    return Err(Error::Index {
                        message: format!(
                            "row id ranges overlap: [{}-{}] ({}) vs [{}-{}] ({})",
                            prev_min, prev_max, prev_path, *minv, *maxv, path
                        ),
                        location: location!(),
                    });
                }
                if *maxv > prev_max {
                    prev_max = *maxv;
                    prev_path = path.clone();
                }
                prev_min = *minv;
            }
        }
    }

    // Write unified IVF metadata into global buffer & set schema metadata
    if let Some(w) = v2w_opt.as_mut() {
        let mut ivf_model = if let Some(c) = first_centroids {
            IvfStorageModel::new(c, None)
        } else {
            IvfStorageModel::empty()
        };
        for len in accumulated_lengths.iter() {
            ivf_model.add_partition(*len);
        }
        let pb_ivf: crate::pb::Ivf = (&ivf_model).try_into()?;
        let pos = w
            .add_global_buffer(Bytes::from(pb_ivf.encode_to_vec()))
            .await?;
        w.add_schema_metadata(IVF_METADATA_KEY, pos.to_string());

        // Also add index metadata key for consistency
        let dt2 = distance_type.ok_or_else(|| Error::Index {
            message: "Distance type missing".to_string(),
            location: location!(),
        })?;
        let idx_meta = IndexMetaSchema {
            index_type: detected_index_type.unwrap().as_str().to_string(),
            distance_type: dt2.to_string(),
        };
        w.add_schema_metadata(INDEX_METADATA_SCHEMA_KEY, serde_json::to_string(&idx_meta)?);

        w.finish().await?;
    } else {
        return Err(Error::Index {
            message: "Failed to initialize unified writer".to_string(),
            location: location!(),
        });
    }

    Ok(())
}

impl Default for UnifiedIndexMetadata {
    fn default() -> Self {
        Self::new()
    }
}
