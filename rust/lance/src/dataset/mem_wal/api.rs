// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dataset API extensions for MemWAL.
//!
//! This module provides the user-facing API for initializing and using MemWAL
//! on a Dataset.

use std::sync::Arc;

use crate::index::DatasetIndexExt;
use async_trait::async_trait;
use lance_core::{Error, Result};
use lance_index::mem_wal::{MEM_WAL_INDEX_NAME, MemWalIndexDetails, ShardSpec};
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::pq::ProductQuantizer;
use lance_io::object_store::ObjectStore;
use lance_linalg::distance::DistanceType;
use uuid::Uuid;

use crate::Dataset;
use crate::dataset::CommitBuilder;
use crate::dataset::transaction::{Operation, Transaction};
use crate::index::DatasetIndexInternalExt;
use crate::index::mem_wal::{load_mem_wal_index_details, new_mem_wal_index_meta};

use super::ShardWriterConfig;
use super::write::MemIndexConfig;
use super::write::ShardWriter;

/// Configuration for initializing MemWAL on a Dataset.
#[derive(Debug, Clone, Default)]
pub struct MemWalConfig {
    /// Optional shard specification for partitioning writes.
    ///
    /// If None, MemWAL is initialized without any shard spec (manual shard management).
    ///
    /// TODO: Add `add_shard_spec()` API to add shard specs after initialization.
    pub shard_spec: Option<ShardSpec>,
    /// Index names to maintain in MemTables.
    /// These must reference indexes already defined on the base table.
    pub maintained_indexes: Vec<String>,
}

/// Shard initialization options for MemWAL.
#[derive(Debug, Clone, Default)]
pub struct MemWalShardConfig {
    /// Number of shards managed by the MemWAL index.
    pub num_shards: u32,
}

/// Extension trait for Dataset to support MemWAL operations.
#[async_trait]
pub trait DatasetMemWalExt {
    /// Initialize MemWAL on this dataset.
    ///
    /// Creates the MemWalIndex system index with the given configuration.
    /// All indexes in `maintained_indexes` must already exist on the dataset.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut dataset = Dataset::open("s3://bucket/dataset").await?;
    /// dataset.initialize_mem_wal(MemWalConfig {
    ///     shard_spec: None,
    ///     maintained_indexes: vec!["id_btree".to_string()],
    /// }).await?;
    /// ```
    async fn initialize_mem_wal(&mut self, config: MemWalConfig) -> Result<()>;

    /// Initialize MemWAL with explicit shard state.
    ///
    /// This preserves [`MemWalConfig`] struct-literal compatibility while
    /// allowing callers that need precomputed shard mappings to initialize the
    /// MemWAL index with inline shard snapshots.
    async fn initialize_mem_wal_with_shards(
        &mut self,
        config: MemWalConfig,
        shard_config: MemWalShardConfig,
    ) -> Result<()> {
        if shard_config.num_shards == 0 {
            self.initialize_mem_wal(config).await
        } else {
            Err(Error::invalid_input(
                "initialize_mem_wal_with_shards is not implemented for this DatasetMemWalExt implementer",
            ))
        }
    }

    /// Return the MemWAL index details for this dataset, if MemWAL is initialized.
    async fn mem_wal_index_details(&self) -> Result<Option<MemWalIndexDetails>> {
        Ok(None)
    }

    /// List current MemWAL shard IDs from object storage directory listing.
    async fn list_mem_wal_latest_shard_ids(&self) -> Result<Vec<Uuid>> {
        Ok(Vec::new())
    }

    /// Get a ShardWriter for the specified shard.
    ///
    /// Automatically loads index configurations from the MemWalIndex
    /// and creates the appropriate in-memory indexes.
    ///
    /// # Arguments
    ///
    /// * `shard_id` - UUID identifying this shard
    /// * `config` - Writer configuration (durability, buffer sizes, etc.)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let writer = dataset.mem_wal_writer(
    ///     Uuid::new_v4(),
    ///     ShardWriterConfig::default(),
    /// ).await?;
    /// writer.put(vec![batch1, batch2]).await?;
    /// ```
    async fn mem_wal_writer(
        &self,
        shard_id: Uuid,
        config: ShardWriterConfig,
    ) -> Result<ShardWriter>;
}

#[async_trait]
impl DatasetMemWalExt for Dataset {
    async fn initialize_mem_wal(&mut self, config: MemWalConfig) -> Result<()> {
        initialize_mem_wal_impl(self, config, MemWalShardConfig::default()).await
    }

    async fn initialize_mem_wal_with_shards(
        &mut self,
        config: MemWalConfig,
        shard_config: MemWalShardConfig,
    ) -> Result<()> {
        initialize_mem_wal_impl(self, config, shard_config).await
    }

    async fn mem_wal_index_details(&self) -> Result<Option<MemWalIndexDetails>> {
        let Some(index_meta) = self.load_index_by_name(MEM_WAL_INDEX_NAME).await? else {
            return Ok(None);
        };

        load_mem_wal_index_details(index_meta).map(Some)
    }

    async fn list_mem_wal_latest_shard_ids(&self) -> Result<Vec<Uuid>> {
        let prefix = super::util::mem_wal_path(&self.branch_location().path);
        let object_store = self.object_store(None).await?;
        let list_result = object_store
            .inner
            .list_with_delimiter(Some(&prefix))
            .await
            .map_err(|e| {
                Error::io(format!(
                    "failed to list MemWAL shard directories at {}: {}",
                    prefix, e
                ))
            })?;
        let mut ids = Vec::new();
        for shard_prefix in list_result.common_prefixes {
            if let Some(name) = shard_prefix.filename()
                && let Ok(shard_id) = Uuid::parse_str(name)
            {
                ids.push(shard_id);
            }
        }
        ids.sort();
        Ok(ids)
    }

    async fn mem_wal_writer(
        &self,
        shard_id: Uuid,
        mut config: ShardWriterConfig,
    ) -> Result<ShardWriter> {
        use lance_index::metrics::NoOpMetricsCollector;

        // Load MemWalIndex to get maintained_indexes
        let mem_wal_index = self
            .open_mem_wal_index(&NoOpMetricsCollector)
            .await?
            .ok_or_else(|| {
                Error::invalid_input(
                    "MemWAL is not initialized on this dataset. Call initialize_mem_wal() first.",
                )
            })?;

        // Get maintained_indexes from the MemWalIndex details
        let maintained_indexes = &mem_wal_index.details.maintained_indexes;

        // Load index configs for each maintained index
        let mut index_configs = Vec::new();
        for index_name in maintained_indexes {
            let index_meta = self.load_index_by_name(index_name).await?.ok_or_else(|| {
                Error::invalid_input(format!(
                    "Index '{}' from maintained_indexes not found on dataset",
                    index_name
                ))
            })?;

            // Detect index type and create appropriate config
            let type_url = index_meta
                .index_details
                .as_ref()
                .map(|d| d.type_url.as_str())
                .unwrap_or("");

            let index_type = MemIndexConfig::detect_index_type(type_url)?;

            match index_type {
                "btree" => {
                    index_configs.push(MemIndexConfig::btree_from_metadata(
                        &index_meta,
                        self.schema(),
                    )?);
                }
                "fts" => {
                    index_configs.push(MemIndexConfig::fts_from_metadata(
                        &index_meta,
                        self.schema(),
                    )?);
                }
                "vector" => {
                    // Vector index - load IVF-PQ config from base table
                    let vector_config =
                        load_vector_index_config(self, index_name, &index_meta).await?;
                    index_configs.push(vector_config);
                }
                _ => {
                    return Err(Error::invalid_input(format!(
                        "Unknown index type: {}",
                        index_type
                    )));
                }
            };
        }

        // Set shard_id in config
        config.shard_id = shard_id;

        // Get object store and base path
        let base_uri = self.uri();
        let (store, base_path) = ObjectStore::from_uri(base_uri).await?;

        // Create ShardWriter
        ShardWriter::open(
            store,
            base_path,
            base_uri,
            config,
            Arc::new(self.schema().into()),
            index_configs,
        )
        .await
    }
}

async fn initialize_mem_wal_impl(
    dataset: &mut Dataset,
    config: MemWalConfig,
    shard_config: MemWalShardConfig,
) -> Result<()> {
    let pk_fields = dataset.schema().unenforced_primary_key();
    if pk_fields.is_empty() {
        return Err(Error::invalid_input(
            "MemWAL requires a primary key on the dataset. \
             Define a primary key using the 'lance-schema:unenforced-primary-key' Arrow field metadata.",
        ));
    }

    let indices = dataset.load_indices().await?;
    for index_name in &config.maintained_indexes {
        if !indices.iter().any(|idx| &idx.name == index_name) {
            return Err(Error::invalid_input(format!(
                "Index '{}' not found on dataset. maintained_indexes must reference existing indexes.",
                index_name
            )));
        }
    }

    if indices.iter().any(|idx| idx.name == MEM_WAL_INDEX_NAME) {
        return Err(Error::invalid_input(
            "MemWAL is already initialized on this dataset. Use update methods instead.",
        ));
    }

    let details = MemWalIndexDetails {
        num_shards: shard_config.num_shards,
        inline_snapshots: None,
        shard_specs: config.shard_spec.into_iter().collect(),
        maintained_indexes: config.maintained_indexes,
        ..Default::default()
    };

    let index_meta = new_mem_wal_index_meta(dataset.manifest.version, details)?;
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: vec![index_meta],
            removed_indices: vec![],
        },
        None,
    );

    let new_dataset = CommitBuilder::new(Arc::new(dataset.clone()))
        .execute(transaction)
        .await?;

    *dataset = new_dataset;

    Ok(())
}

/// Load vector index configuration from the base table's IVF-PQ index.
///
/// Opens the vector index and extracts the IVF model and PQ codebook
/// to create an in-memory IVF-PQ index config.
async fn load_vector_index_config(
    dataset: &Dataset,
    index_name: &str,
    index_meta: &lance_table::format::IndexMetadata,
) -> Result<MemIndexConfig> {
    use lance_index::metrics::NoOpMetricsCollector;

    // Get the column name for this index
    let field_id = index_meta.fields.first().ok_or_else(|| {
        Error::invalid_input(format!("Vector index '{}' has no fields", index_name))
    })?;

    let field = dataset.schema().field_by_id(*field_id).ok_or_else(|| {
        Error::invalid_input(format!("Field not found for vector index '{}'", index_name))
    })?;

    let column = field.name.clone();

    // Load IVF-PQ components
    let index_uuid = index_meta.uuid.to_string();
    let (ivf_model, pq, distance_type) = load_ivf_pq_components(
        dataset,
        index_name,
        &index_uuid,
        &column,
        &NoOpMetricsCollector,
    )
    .await?;

    Ok(MemIndexConfig::ivf_pq(
        index_name.to_string(),
        *field_id,
        column,
        ivf_model,
        pq,
        distance_type,
    ))
}

/// Load IVF model and ProductQuantizer from an IVF-PQ index.
async fn load_ivf_pq_components(
    dataset: &Dataset,
    index_name: &str,
    index_uuid: &str,
    column_name: &str,
    metrics: &dyn lance_index::metrics::MetricsCollector,
) -> Result<(IvfModel, ProductQuantizer, DistanceType)> {
    use crate::index::vector::ivf::v2::IvfPq;
    use lance_index::vector::VectorIndex;

    // Open the vector index using UUID
    let index = dataset
        .open_vector_index(column_name, index_uuid, metrics)
        .await?;

    // Try to downcast to IvfPq (IVFIndex<FlatIndex, ProductQuantizer>)
    // This covers IVF-PQ indexes which are the most common
    let ivf_index = index.as_any().downcast_ref::<IvfPq>().ok_or_else(|| {
        Error::invalid_input(format!(
            "Vector index '{}' is not an IVF-PQ index. Only IVF-PQ indexes are supported for MemWAL.",
            index_name
        ))
    })?;

    // Extract IVF model and distance type from the index
    let ivf_model = ivf_index.ivf_model().clone();
    let distance_type = ivf_index.metric_type();

    // Get the quantizer and convert to ProductQuantizer
    let quantizer = ivf_index.quantizer();
    let pq = ProductQuantizer::try_from(quantizer)?;

    Ok((ivf_model, pq, distance_type))
}
