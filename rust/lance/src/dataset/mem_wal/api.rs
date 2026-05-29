// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dataset API extensions for MemWAL.
//!
//! This module provides the user-facing API for initializing and using MemWAL
//! on a Dataset.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::DataType;
use async_trait::async_trait;
use lance_core::{Error, Result};
use lance_index::mem_wal::{MEM_WAL_INDEX_NAME, MemWalIndexDetails, ShardingField, ShardingSpec};
use lance_io::object_store::ObjectStore;
use uuid::Uuid;

use crate::Dataset;
use crate::dataset::CommitBuilder;
use crate::dataset::transaction::{Operation, Transaction};
use crate::index::DatasetIndexExt;
use crate::index::DatasetIndexInternalExt;
use crate::index::mem_wal::{load_mem_wal_index_details, new_mem_wal_index_meta};

use super::ShardWriterConfig;
use super::write::MemIndexConfig;
use super::write::ShardWriter;

/// Spec id of the sole sharding spec installed by [`InitializeMemWalBuilder`].
const SHARDING_SPEC_ID: u32 = 1;

/// Field id, within the sharding spec, of the derived shard-routing value.
const SHARDING_FIELD_ID: &str = "bucket";

/// Result type of the derived shard-routing value.
const SHARDING_RESULT_TYPE: &str = "int32";

/// Transform name for [`InitializeMemWalBuilder::bucket_sharding`]. Matches
/// Iceberg's `bucket(col, N)` partition transform name.
const BUCKET_TRANSFORM: &str = "bucket";

/// Transform name for [`InitializeMemWalBuilder::unsharded`]: every row maps to
/// a single shard.
const UNSHARDED_TRANSFORM: &str = "unsharded";

/// Transform name for [`InitializeMemWalBuilder::identity_sharding`]: the shard
/// value is the raw value of the source column.
const IDENTITY_TRANSFORM: &str = "identity";

/// Parameter key holding the bucket count `N` on the bucket transform.
const NUM_BUCKETS_PARAM: &str = "num_buckets";

/// Inclusive upper bound for `num_buckets`. Bounds the number of distinct
/// MemWAL shards a single bucket spec can address, which caps how many shard
/// manifests the dataset has to manage.
const MAX_NUM_BUCKETS: u32 = 1024;

/// How writes are partitioned into MemWAL shards.
#[derive(Debug)]
enum Sharding {
    /// No sharding spec is recorded; shards are managed manually.
    Manual,
    /// A single shard; every row is routed to it.
    Unsharded,
    /// Hash-bucket a shard key into `num_buckets` shards.
    Bucket { column: String, num_buckets: u32 },
    /// Shard by the raw value of `column` (identity transform).
    Identity { column: String },
}

/// Builder for initializing MemWAL on a [`Dataset`].
///
/// Created by [`DatasetMemWalExt::initialize_mem_wal`]. Choose a sharding
/// strategy and the indexes to maintain, then call [`execute`](Self::execute).
///
/// # Example
///
/// ```ignore
/// use lance::dataset::mem_wal::DatasetMemWalExt;
///
/// dataset
///     .initialize_mem_wal()
///     .bucket_sharding("id", 16)
///     .maintained_indexes(["id_btree"])
///     .execute()
///     .await?;
/// ```
#[must_use = "InitializeMemWalBuilder does nothing unless `.execute()` is awaited"]
pub struct InitializeMemWalBuilder<'a> {
    dataset: &'a mut Dataset,
    sharding: Sharding,
    maintained_indexes: Vec<String>,
    writer_config_defaults: HashMap<String, String>,
}

impl<'a> InitializeMemWalBuilder<'a> {
    fn new(dataset: &'a mut Dataset) -> Self {
        Self {
            dataset,
            sharding: Sharding::Manual,
            maintained_indexes: Vec::new(),
            writer_config_defaults: HashMap::new(),
        }
    }

    /// Route every row to a single MemWAL shard.
    pub fn unsharded(mut self) -> Self {
        self.sharding = Sharding::Unsharded;
        self
    }

    /// Hash-bucket `column` into `num_buckets` shards.
    ///
    /// `column` must name a scalar dataset column that can be hash-bucketed.
    /// `num_buckets` must be in `[1, 1024]`. These constraints are validated
    /// by [`execute`](Self::execute).
    pub fn bucket_sharding(mut self, column: impl Into<String>, num_buckets: u32) -> Self {
        self.sharding = Sharding::Bucket {
            column: column.into(),
            num_buckets,
        };
        self
    }

    /// Shard by the raw value of `column` (the identity transform).
    ///
    /// Each distinct value of `column` becomes its own shard; use this when the
    /// data is already partitioned by that column. For primary-key tables, the
    /// caller is responsible for ensuring every primary key maps consistently
    /// to a single value of `column`. `column` must be a scalar column that
    /// exists on the dataset; it is validated by [`execute`](Self::execute).
    pub fn identity_sharding(mut self, column: impl Into<String>) -> Self {
        self.sharding = Sharding::Identity {
            column: column.into(),
        };
        self
    }

    /// Set the base-table indexes to maintain in MemTables, replacing any
    /// previously set list.
    ///
    /// Each name must reference an index that already exists on the dataset.
    /// The primary key btree, when present, is maintained implicitly and must
    /// not be listed.
    pub fn maintained_indexes<I, S>(mut self, indexes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.maintained_indexes = indexes.into_iter().map(Into::into).collect();
        self
    }

    /// Record `config` as the default `ShardWriter` configuration.
    ///
    /// Every tunable field is persisted into the MemWAL index so that all
    /// writers — across processes and restarts — start from the same
    /// defaults. Shard identity (`shard_id`, `shard_spec_id`) is not a
    /// configuration default and is not recorded. These remain defaults only:
    /// an individual writer may still override any value at runtime in its own
    /// (non-persisted) `ShardWriterConfig`.
    ///
    /// Merges into any defaults already set; a key set via
    /// [`add_writer_config_default`](Self::add_writer_config_default) afterwards wins.
    pub fn writer_config_defaults(mut self, config: ShardWriterConfig) -> Self {
        self.writer_config_defaults
            .extend(writer_config_to_defaults(&config));
        self
    }

    /// Record a single arbitrary writer-configuration default.
    ///
    /// Use this for keys not covered by
    /// [`writer_config_defaults`](Self::writer_config_defaults).
    pub fn add_writer_config_default(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.writer_config_defaults.insert(key.into(), value.into());
        self
    }

    /// Initialize MemWAL on the dataset, committing the MemWAL system index.
    ///
    /// Fails if any maintained index does not exist, if the selected sharding
    /// configuration is invalid, or if MemWAL is already initialized.
    pub async fn execute(self) -> Result<()> {
        let Self {
            dataset,
            sharding,
            maintained_indexes,
            writer_config_defaults,
        } = self;

        // Resolve (and validate) the sharding choice before any I/O.
        let (sharding_specs, num_shards) = resolve_sharding(dataset, sharding)?;

        let indices = dataset.load_indices().await?;
        for index_name in &maintained_indexes {
            if !indices.iter().any(|idx| &idx.name == index_name) {
                return Err(Error::invalid_input(format!(
                    "Index '{}' not found on dataset. maintained_indexes must reference existing indexes.",
                    index_name
                )));
            }
        }
        if indices.iter().any(|idx| idx.name == MEM_WAL_INDEX_NAME) {
            return Err(Error::invalid_input(
                "MemWAL is already initialized on this dataset.",
            ));
        }

        let details = MemWalIndexDetails {
            num_shards,
            sharding_specs,
            maintained_indexes,
            writer_config_defaults,
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
}

/// Resolve a [`Sharding`] choice into the sharding specs and shard count to
/// persist in [`MemWalIndexDetails`].
fn resolve_sharding(dataset: &Dataset, sharding: Sharding) -> Result<(Vec<ShardingSpec>, u32)> {
    match sharding {
        Sharding::Manual => Ok((Vec::new(), 0)),
        Sharding::Unsharded => Ok((vec![unsharded_sharding_spec()], 1)),
        Sharding::Bucket {
            column,
            num_buckets,
        } => Ok((
            vec![bucket_sharding_spec(dataset, &column, num_buckets)?],
            num_buckets,
        )),
        Sharding::Identity { column } => Ok((vec![identity_sharding_spec(dataset, &column)?], 0)),
    }
}

/// Build the sharding spec for [`InitializeMemWalBuilder::unsharded`].
fn unsharded_sharding_spec() -> ShardingSpec {
    ShardingSpec {
        spec_id: SHARDING_SPEC_ID,
        fields: vec![ShardingField {
            field_id: SHARDING_FIELD_ID.to_string(),
            source_ids: Vec::new(),
            transform: Some(UNSHARDED_TRANSFORM.to_string()),
            expression: None,
            result_type: SHARDING_RESULT_TYPE.to_string(),
            parameters: HashMap::new(),
        }],
    }
}

/// Build the sharding spec for [`InitializeMemWalBuilder::bucket_sharding`].
fn bucket_sharding_spec(dataset: &Dataset, column: &str, num_buckets: u32) -> Result<ShardingSpec> {
    if num_buckets == 0 || num_buckets > MAX_NUM_BUCKETS {
        return Err(Error::invalid_input(format!(
            "bucket_sharding: num_buckets must be in [1, {}], got {}",
            MAX_NUM_BUCKETS, num_buckets
        )));
    }

    let source_field = dataset.schema().field(column).ok_or_else(|| {
        Error::invalid_input(format!(
            "bucket_sharding: column '{}' not found on the dataset",
            column
        ))
    })?;

    let data_type = source_field.data_type();
    if !is_bucket_sharding_supported_type(&data_type) {
        return Err(Error::invalid_input(format!(
            "bucket_sharding: column '{}' has type {:?}, which cannot be used as a shard key",
            column, data_type
        )));
    }

    Ok(ShardingSpec {
        spec_id: SHARDING_SPEC_ID,
        fields: vec![ShardingField {
            field_id: SHARDING_FIELD_ID.to_string(),
            source_ids: vec![source_field.id],
            transform: Some(BUCKET_TRANSFORM.to_string()),
            expression: None,
            result_type: SHARDING_RESULT_TYPE.to_string(),
            parameters: HashMap::from([(NUM_BUCKETS_PARAM.to_string(), num_buckets.to_string())]),
        }],
    })
}

/// Build the sharding spec for [`InitializeMemWalBuilder::identity_sharding`].
fn identity_sharding_spec(dataset: &Dataset, column: &str) -> Result<ShardingSpec> {
    let field = dataset.schema().field(column).ok_or_else(|| {
        Error::invalid_input(format!(
            "identity_sharding: column '{}' not found on the dataset",
            column
        ))
    })?;

    let data_type = field.data_type();
    let result_type = scalar_result_type(&data_type).ok_or_else(|| {
        Error::invalid_input(format!(
            "identity_sharding: column '{}' has type {:?}, which cannot be used as a shard key",
            column, data_type
        ))
    })?;

    Ok(ShardingSpec {
        spec_id: SHARDING_SPEC_ID,
        fields: vec![ShardingField {
            field_id: SHARDING_FIELD_ID.to_string(),
            source_ids: vec![field.id],
            transform: Some(IDENTITY_TRANSFORM.to_string()),
            expression: None,
            result_type: result_type.to_string(),
            parameters: HashMap::new(),
        }],
    })
}

fn is_bucket_sharding_supported_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Date32
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Utf8
            | DataType::LargeUtf8
    )
}

/// The Arrow type name for a scalar column usable as a shard key, or `None`
/// for types that cannot be a shard key.
fn scalar_result_type(data_type: &DataType) -> Option<&'static str> {
    Some(match data_type {
        DataType::Int8 => "int8",
        DataType::Int16 => "int16",
        DataType::Int32 => "int32",
        DataType::Int64 => "int64",
        DataType::UInt8 => "uint8",
        DataType::UInt16 => "uint16",
        DataType::UInt32 => "uint32",
        DataType::UInt64 => "uint64",
        DataType::Utf8 => "utf8",
        DataType::LargeUtf8 => "large_utf8",
        DataType::Boolean => "boolean",
        _ => return None,
    })
}

/// Extract the tunable defaults from a [`ShardWriterConfig`] into the persisted
/// string map. Shard identity (`shard_id`, `shard_spec_id`) is not a default.
/// `Duration` knobs are recorded in milliseconds with a `_ms` key suffix.
fn writer_config_to_defaults(config: &ShardWriterConfig) -> HashMap<String, String> {
    let mut defaults = HashMap::from([
        (
            "durable_write".to_string(),
            config.durable_write.to_string(),
        ),
        (
            "sync_indexed_write".to_string(),
            config.sync_indexed_write.to_string(),
        ),
        (
            "max_wal_buffer_size".to_string(),
            config.max_wal_buffer_size.to_string(),
        ),
        (
            "max_memtable_size".to_string(),
            config.max_memtable_size.to_string(),
        ),
        (
            "max_memtable_rows".to_string(),
            config.max_memtable_rows.to_string(),
        ),
        (
            "max_memtable_batches".to_string(),
            config.max_memtable_batches.to_string(),
        ),
        (
            "manifest_scan_batch_size".to_string(),
            config.manifest_scan_batch_size.to_string(),
        ),
        (
            "max_unflushed_memtable_bytes".to_string(),
            config.max_unflushed_memtable_bytes.to_string(),
        ),
        (
            "backpressure_log_interval_ms".to_string(),
            config.backpressure_log_interval.as_millis().to_string(),
        ),
        (
            "async_index_buffer_rows".to_string(),
            config.async_index_buffer_rows.to_string(),
        ),
        (
            "async_index_interval_ms".to_string(),
            config.async_index_interval.as_millis().to_string(),
        ),
        (
            "enable_memtable".to_string(),
            config.enable_memtable.to_string(),
        ),
    ]);
    if let Some(interval) = config.max_wal_flush_interval {
        defaults.insert(
            "max_wal_flush_interval_ms".to_string(),
            interval.as_millis().to_string(),
        );
    }
    if let Some(interval) = config.stats_log_interval {
        defaults.insert(
            "stats_log_interval_ms".to_string(),
            interval.as_millis().to_string(),
        );
    }
    defaults
}

/// Extension trait for Dataset to support MemWAL operations.
#[async_trait]
pub trait DatasetMemWalExt {
    /// Begin initializing MemWAL on this dataset.
    ///
    /// Returns an [`InitializeMemWalBuilder`]; configure the sharding strategy
    /// and maintained indexes, then call [`InitializeMemWalBuilder::execute`].
    fn initialize_mem_wal(&mut self) -> InitializeMemWalBuilder<'_>;

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
    fn initialize_mem_wal(&mut self) -> InitializeMemWalBuilder<'_> {
        InitializeMemWalBuilder::new(self)
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
            // A maintained index can split into multiple physical segments
            // (e.g. `optimize_indices(append)` deltas), which the singular
            // `load_index_by_name` rejects. Every segment carries the same
            // type and params, so take the first match.
            let index_meta = self
                .load_indices_by_name(index_name)
                .await?
                .into_iter()
                .next()
                .ok_or_else(|| {
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

/// Build an in-memory HNSW vector index configuration from a base-table
/// vector index entry.
///
/// HNSW does not require any centroids/codebook from the base table — it is
/// self-contained. The only thing we read from the base index is the distance
/// type (so the in-memory index uses the same metric as the base). If the
/// base index is unreadable for some reason, we default to L2.
async fn load_vector_index_config(
    dataset: &Dataset,
    index_name: &str,
    index_meta: &lance_table::format::IndexMetadata,
) -> Result<MemIndexConfig> {
    use lance_index::metrics::NoOpMetricsCollector;

    let field_id = index_meta.fields.first().ok_or_else(|| {
        Error::invalid_input(format!("Vector index '{}' has no fields", index_name))
    })?;

    let field = dataset.schema().field_by_id(*field_id).ok_or_else(|| {
        Error::invalid_input(format!("Field not found for vector index '{}'", index_name))
    })?;
    let column = field.name.clone();

    // Inherit the base table's distance type so the in-memory index and the
    // base index produce comparable distances. Surface the open error
    // instead of silently defaulting to L2 — flushed `IVF_HNSW_SQ` files
    // bake this metric into their on-disk metadata, so a wrong default would
    // be durable corruption.
    let distance_type = dataset
        .open_vector_index(&column, &index_meta.uuid.to_string(), &NoOpMetricsCollector)
        .await
        .map_err(|e| {
            Error::invalid_input(format!(
                "Failed to open base vector index '{}' to inherit distance type: {}",
                index_name, e
            ))
        })?
        .metric_type();

    Ok(MemIndexConfig::hnsw(
        index_name.to_string(),
        *field_id,
        column,
        distance_type,
    ))
}
