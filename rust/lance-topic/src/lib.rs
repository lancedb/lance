// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Kafka-like topic primitives backed by Lance MemWAL-compatible WAL files.
//!
//! The initial topic API provides at-least-once delivery. Consumers commit
//! entry-level offsets to Lance-backed consumer group tables.

mod metadata;
mod partition;
mod wal;

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock as StdRwLock};

use arrow_array::{
    Array, ArrayRef, LargeBinaryArray, RecordBatch, RecordBatchIterator, StringArray, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use arrow_select::take::take_record_batch;
use futures::future::{join_all, try_join_all};
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::mem_wal::{
    DatasetMemWalExt, MemWalConfig, MemWalShardSnapshot, ShardManifestStore,
};
use lance_arrow::json::{JsonArray, decode_json, json_field};
use lance_core::datatypes::Schema as LanceSchema;
use lance_core::{Error, Result};
use lance_index::mem_wal::{MemWalIndexDetails, ShardField, ShardSpec};
use lance_io::object_store::ObjectStore;
use lance_namespace::{ErrorCode, LanceNamespace, NamespaceError};
use lance_namespace_impls::DirectoryNamespaceBuilder;
use object_store::path::Path;
use serde_json::Value;
use uuid::Uuid;

use metadata::ConsumerGroupOffset;
pub use metadata::StartPosition;
use partition::{Partitioner, assigned_position_for_partition, murmur3_x86_32};
pub use wal::{TopicEntry, WalAppendResult, WalAppender, WalTailer};

const TOPIC_ID_COLUMN: &str = "id";
const TOPIC_PRODUCER_ID_COLUMN: &str = "producer_id";
const TOPIC_PAYLOAD_COLUMN: &str = "payload";
const LANCE_UNENFORCED_PRIMARY_KEY: &str = "lance-schema:unenforced-primary-key";
const TOPIC_SHARD_SPEC_ID: u32 = 1;
const TOPIC_PARTITION_FIELD_ID: &str = "topic_partition_id";
const TOPIC_PRODUCER_FIELD_ID: &str = "producer_id";
const CONSUMER_GROUP_NAMESPACE_SEGMENT: &str = "consumer_group";
const CONSUMER_GROUP_POSITION_COLUMN: &str = "consumer_position";
const CONSUMER_GROUP_PARTITION_ID_COLUMN: &str = "topic_partition_id";
const CONSUMER_GROUP_PRODUCER_ID_COLUMN: &str = "producer_id";
const CONSUMER_GROUP_NEXT_ENTRY_POSITION_COLUMN: &str = "next_entry_position";
const CONSUMER_GROUP_SHARD_SPEC_ID: u32 = 1;
const CONSUMER_GROUP_POSITION_FIELD_ID: &str = "consumer_position";
const CONSUMER_GROUP_PARTITION_FIELD_ID: &str = "topic_partition_id";

/// Configuration for creating a topic.
#[derive(Debug, Clone)]
pub struct TopicConfig {
    partition_count: u32,
}

impl TopicConfig {
    /// Create topic configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of topic partitions.
    pub fn with_partition_count(mut self, partition_count: u32) -> Self {
        self.partition_count = partition_count;
        self
    }
}

impl Default for TopicConfig {
    fn default() -> Self {
        Self { partition_count: 1 }
    }
}

#[derive(Debug, Clone)]
enum TopicTarget {
    Directory {
        root: String,
        table_id: Vec<String>,
    },
    Namespace {
        namespace_client: Arc<dyn LanceNamespace>,
        table_id: Vec<String>,
    },
}

#[derive(Debug, Clone)]
struct NamespaceTarget {
    namespace_client: Arc<dyn LanceNamespace>,
    table_id: Vec<String>,
}

impl TopicTarget {
    async fn into_namespace(self) -> Result<NamespaceTarget> {
        match self {
            Self::Directory { root, table_id } => {
                let namespace_client = Arc::new(
                    DirectoryNamespaceBuilder::new(root)
                        .build()
                        .await
                        .map_err(|e| Error::namespace_source(Box::new(e)))?,
                );
                Ok(NamespaceTarget {
                    namespace_client,
                    table_id,
                })
            }
            Self::Namespace {
                namespace_client,
                table_id,
            } => Ok(NamespaceTarget {
                namespace_client,
                table_id,
            }),
        }
    }
}

/// Builder for creating or opening a topic.
#[derive(Debug, Clone, Default)]
pub struct TopicBuilder {
    target: Option<TopicTarget>,
    config: TopicConfig,
}

impl TopicBuilder {
    /// Create an empty topic builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a topic builder using a default directory namespace rooted at `root`.
    pub fn from_directory<I, S>(root: impl Into<String>, table_id: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::new().directory(root, table_id)
    }

    /// Create a topic builder for a namespace-managed table.
    pub fn from_namespace<I, S>(namespace_client: Arc<dyn LanceNamespace>, table_id: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::new().namespace(namespace_client, table_id)
    }

    /// Use a default directory namespace rooted at `root`.
    pub fn directory<I, S>(mut self, root: impl Into<String>, table_id: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.target = Some(TopicTarget::Directory {
            root: root.into(),
            table_id: collect_table_id(table_id),
        });
        self
    }

    /// Set the namespace client and table identifier.
    pub fn namespace<I, S>(mut self, namespace_client: Arc<dyn LanceNamespace>, table_id: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.target = Some(TopicTarget::Namespace {
            namespace_client,
            table_id: collect_table_id(table_id),
        });
        self
    }

    /// Set the number of topic partitions.
    pub fn partition_count(mut self, partition_count: u32) -> Self {
        self.config = self.config.with_partition_count(partition_count);
        self
    }

    /// Create the topic table and initialize its MemWAL index.
    pub async fn create(self) -> Result<Topic> {
        let Self { target, config } = self;
        create_topic(required_target(target)?, config).await
    }

    /// Open an existing topic.
    pub async fn open(self) -> Result<Topic> {
        let Self { target, .. } = self;
        let target = required_target(target)?.into_namespace().await?;
        validate_table_id(&target.table_id)?;
        let dataset =
            open_namespace_dataset(target.namespace_client.clone(), target.table_id.clone())
                .await?;
        Topic::from_dataset(dataset, target.namespace_client, target.table_id).await
    }
}

fn required_target(target: Option<TopicTarget>) -> Result<TopicTarget> {
    target.ok_or_else(|| {
        Error::invalid_input(
            "topic builder requires either a directory namespace root or namespace client and table_id",
        )
    })
}

fn collect_table_id<I, S>(table_id: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    table_id.into_iter().map(Into::into).collect()
}

fn validate_table_id(table_id: &[String]) -> Result<()> {
    if table_id.is_empty() {
        return Err(Error::invalid_input("topic table_id cannot be empty"));
    }
    for segment in table_id {
        if segment.is_empty() {
            return Err(Error::invalid_input(format!(
                "topic table_id {:?} cannot contain an empty segment",
                table_id
            )));
        }
    }
    Ok(())
}

/// Topic partition metadata discovered from latest MemWAL shard manifests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopicPartition {
    /// Logical topic partition id derived from `id`.
    pub partition_id: u32,
    /// Producer shard id.
    pub producer_id: String,
    /// MemWAL shard id used to store this physical producer shard's WAL files.
    pub shard_id: Uuid,
    /// MemWAL shard spec used to route rows to this partition.
    pub shard_spec_id: u32,
}

/// A Lance topic rooted at a namespace-managed table.
#[derive(Debug, Clone)]
pub struct Topic {
    namespace_client: Arc<dyn LanceNamespace>,
    table_id: Arc<Vec<String>>,
    dataset: Arc<Dataset>,
    object_store: Arc<ObjectStore>,
    base_path: Path,
    schema: Arc<ArrowSchema>,
    primary_key_columns: Arc<Vec<String>>,
    mem_wal_index_details: Arc<MemWalIndexDetails>,
    partition_count: u32,
    partitions: Arc<StdRwLock<Vec<TopicPartition>>>,
}

impl Topic {
    /// Start building a topic.
    pub fn builder() -> TopicBuilder {
        TopicBuilder::new()
    }

    /// Create a new namespace-managed topic.
    pub async fn create<I, S>(
        namespace_client: Arc<dyn LanceNamespace>,
        table_id: I,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        TopicBuilder::from_namespace(namespace_client, table_id)
            .create()
            .await
    }

    /// Create a new namespace-managed topic with configuration.
    pub async fn create_with_config<I, S>(
        namespace_client: Arc<dyn LanceNamespace>,
        table_id: I,
        config: TopicConfig,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        TopicBuilder::from_namespace(namespace_client, table_id)
            .partition_count(config.partition_count)
            .create()
            .await
    }

    /// Open an existing namespace-managed topic.
    pub async fn open<I, S>(namespace_client: Arc<dyn LanceNamespace>, table_id: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        TopicBuilder::from_namespace(namespace_client, table_id)
            .open()
            .await
    }

    async fn from_dataset(
        dataset: Dataset,
        namespace_client: Arc<dyn LanceNamespace>,
        table_id: Vec<String>,
    ) -> Result<Self> {
        let object_store = Arc::new(dataset.object_store().clone());
        let base_path = dataset.branch_location().path;
        let lance_schema = dataset.schema();
        validate_topic_schema(lance_schema)?;
        let primary_key_columns = primary_key_columns(lance_schema)?;
        let mem_wal_index_details = dataset.mem_wal_index_details().await?.ok_or_else(|| {
            Error::invalid_input(
                "topic table is missing MemWAL index; create it with TopicBuilder::create",
            )
        })?;
        let partition_count = validate_topic_shard_spec(&mem_wal_index_details)?;
        let partitions = topic_partitions_from_mem_wal_listing(&dataset, partition_count).await?;
        let schema = Arc::new(ArrowSchema::from(lance_schema));
        Ok(Self {
            namespace_client,
            table_id: Arc::new(table_id),
            dataset: Arc::new(dataset),
            object_store,
            base_path,
            schema,
            primary_key_columns: Arc::new(primary_key_columns),
            mem_wal_index_details: Arc::new(mem_wal_index_details),
            partition_count,
            partitions: Arc::new(StdRwLock::new(partitions)),
        })
    }

    /// Namespace table id for this topic.
    pub fn table_id(&self) -> &[String] {
        self.table_id.as_slice()
    }

    /// Namespace table id used to store a consumer group's committed offsets.
    pub fn consumer_group_table_id(&self, group_id: &str) -> Result<Vec<String>> {
        metadata::validate_group_id(group_id)?;
        let mut table_id = self.table_id.as_ref().clone();
        table_id.push(CONSUMER_GROUP_NAMESPACE_SEGMENT.to_string());
        table_id.push(group_id.to_string());
        Ok(table_id)
    }

    /// Start building a topic consumer group.
    pub fn consumer_group(&self, group_id: impl Into<String>) -> ConsumerGroupBuilder {
        ConsumerGroupBuilder::new(self.clone(), group_id)
    }

    /// Number of topic partitions.
    pub fn partition_count(&self) -> u32 {
        self.partition_count
    }

    /// Primary key columns used for hash partitioning.
    pub fn primary_key_columns(&self) -> &[String] {
        self.primary_key_columns.as_slice()
    }

    /// MemWAL index details describing topic shard routing.
    pub fn mem_wal_index_details(&self) -> &MemWalIndexDetails {
        self.mem_wal_index_details.as_ref()
    }

    /// Backing Lance dataset.
    pub fn dataset(&self) -> &Dataset {
        self.dataset.as_ref()
    }

    /// Topic record schema.
    pub fn schema(&self) -> &Arc<ArrowSchema> {
        &self.schema
    }

    /// Physical topic shard metadata discovered so far.
    pub fn partitions(&self) -> Result<Vec<TopicPartition>> {
        self.current_partitions()
    }

    /// Refresh physical shard metadata from latest MemWAL shard manifests.
    pub async fn refresh_partitions(&self) -> Result<Vec<TopicPartition>> {
        let partitions =
            topic_partitions_from_mem_wal_listing(self.dataset.as_ref(), self.partition_count)
                .await?;
        self.replace_partitions(partitions.clone())?;
        Ok(partitions)
    }

    /// Create a producer for this topic and claim its partition writer epochs.
    pub async fn producer(&self, producer_id: impl Into<String>) -> Result<Producer> {
        Producer::open(self.clone(), producer_id.into()).await
    }

    fn current_partitions(&self) -> Result<Vec<TopicPartition>> {
        self.partitions
            .read()
            .map_err(|_| Error::internal("topic partition metadata lock is poisoned"))
            .map(|partitions| partitions.clone())
    }

    fn replace_partitions(&self, partitions: Vec<TopicPartition>) -> Result<()> {
        *self
            .partitions
            .write()
            .map_err(|_| Error::internal("topic partition metadata lock is poisoned"))? =
            partitions;
        Ok(())
    }

    fn partition(&self, partition_id: u32, producer_id: &str) -> Result<TopicPartition> {
        self.current_partitions()?
            .into_iter()
            .find(|partition| {
                partition.partition_id == partition_id && partition.producer_id == producer_id
            })
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "partition_id {} producer_id '{}' does not have a discovered MemWAL shard",
                    partition_id, producer_id
                ))
            })
    }

    async fn ensure_partition_shard(
        &self,
        partition_id: u32,
        producer_id: &str,
    ) -> Result<TopicPartition> {
        validate_logical_partition(self, partition_id)?;
        if let Ok(partition) = self.partition(partition_id, producer_id) {
            return Ok(partition);
        }

        self.refresh_partitions().await?;
        if let Ok(partition) = self.partition(partition_id, producer_id) {
            return Ok(partition);
        }

        let shard_id = topic_shard_id(partition_id, producer_id);
        let shard_field_values =
            HashMap::from([(TOPIC_PARTITION_FIELD_ID.to_string(), partition_id as i32)]);
        let shard_field_string_values =
            HashMap::from([(TOPIC_PRODUCER_FIELD_ID.to_string(), producer_id.to_string())]);
        let manifest_store =
            ShardManifestStore::new(self.object_store.clone(), &self.base_path, shard_id, 2);
        manifest_store
            .initialize_shard_with_string_values(
                TOPIC_SHARD_SPEC_ID,
                shard_field_values,
                shard_field_string_values,
            )
            .await?;

        let partition = TopicPartition {
            partition_id,
            producer_id: producer_id.to_string(),
            shard_id,
            shard_spec_id: TOPIC_SHARD_SPEC_ID,
        };
        let mut partitions = self.current_partitions()?;
        if let Some(existing) = partitions.iter().find(|existing| {
            existing.partition_id == partition_id && existing.producer_id == producer_id
        }) {
            return Ok(existing.clone());
        }
        partitions.push(partition.clone());
        partitions.sort_by(|a, b| {
            a.partition_id
                .cmp(&b.partition_id)
                .then_with(|| a.producer_id.cmp(&b.producer_id))
        });
        self.replace_partitions(partitions)?;
        Ok(partition)
    }

    fn wal_tailer(&self, partition_id: u32, producer_id: &str) -> Result<WalTailer> {
        let partition = self.partition(partition_id, producer_id)?;
        Ok(WalTailer::new(
            self.object_store.clone(),
            self.base_path.clone(),
            partition.partition_id,
            partition.producer_id,
            partition.shard_id,
        ))
    }

    fn validate_batch_schema(&self, batch: &RecordBatch) -> Result<()> {
        if batch.schema_ref().fields() != self.schema.fields() {
            return Err(Error::invalid_input(format!(
                "record batch schema does not match topic schema: expected fields {:?}, got fields {:?}",
                self.schema.fields(),
                batch.schema_ref().fields()
            )));
        }
        Ok(())
    }
}

async fn create_topic(target: TopicTarget, config: TopicConfig) -> Result<Topic> {
    if config.partition_count == 0 {
        return Err(Error::invalid_input(
            "partition_count must be greater than 0",
        ));
    }
    if config.partition_count > i32::MAX as u32 {
        return Err(Error::invalid_input(format!(
            "partition_count {} exceeds supported maximum {}",
            config.partition_count,
            i32::MAX
        )));
    }
    let target = target.into_namespace().await?;
    validate_table_id(&target.table_id)?;

    let schema = topic_schema();
    let reader = RecordBatchIterator::new(
        vec![Ok(RecordBatch::new_empty(schema.clone()))].into_iter(),
        schema.clone(),
    );
    let mut dataset = Dataset::write_into_namespace(
        reader,
        target.namespace_client.clone(),
        target.table_id.clone(),
        Some(topic_write_params()),
    )
    .await?;

    let lance_schema = dataset.schema();
    let shard_spec = topic_shard_spec(config.partition_count, lance_schema)?;
    dataset
        .initialize_mem_wal(MemWalConfig {
            shard_spec: Some(shard_spec),
            maintained_indexes: Vec::new(),
        })
        .await?;

    Topic::from_dataset(dataset, target.namespace_client, target.table_id).await
}

async fn open_namespace_dataset(
    namespace_client: Arc<dyn LanceNamespace>,
    table_id: Vec<String>,
) -> Result<Dataset> {
    DatasetBuilder::from_namespace(namespace_client, table_id)
        .await?
        .load()
        .await
}

fn topic_write_params() -> WriteParams {
    WriteParams {
        auto_cleanup: None,
        skip_auto_cleanup: true,
        ..Default::default()
    }
}

fn topic_schema() -> Arc<ArrowSchema> {
    let id_metadata =
        HashMap::from([(LANCE_UNENFORCED_PRIMARY_KEY.to_string(), "true".to_string())]);
    Arc::new(ArrowSchema::new(vec![
        Field::new(TOPIC_ID_COLUMN, DataType::Utf8, false).with_metadata(id_metadata),
        Field::new(TOPIC_PRODUCER_ID_COLUMN, DataType::Utf8, false),
        json_field(TOPIC_PAYLOAD_COLUMN, false),
    ]))
}

fn validate_topic_schema(schema: &LanceSchema) -> Result<()> {
    let arrow_schema = ArrowSchema::from(schema);
    if arrow_schema.fields() != topic_schema().fields() {
        return Err(Error::invalid_input(format!(
            "topic table schema must be fixed id/producer_id/payload schema: expected fields {:?}, got fields {:?}",
            topic_schema().fields(),
            arrow_schema.fields()
        )));
    }
    let primary_key_columns = primary_key_columns(schema)?;
    if primary_key_columns != [TOPIC_ID_COLUMN.to_string()] {
        return Err(Error::invalid_input(format!(
            "topic table must use '{}' as its only unenforced primary key column, got {:?}",
            TOPIC_ID_COLUMN, primary_key_columns
        )));
    }
    Ok(())
}

fn primary_key_columns(schema: &LanceSchema) -> Result<Vec<String>> {
    Ok(schema
        .unenforced_primary_key()
        .into_iter()
        .map(|field| field.name.clone())
        .collect())
}

fn topic_shard_spec(partition_count: u32, schema: &LanceSchema) -> Result<ShardSpec> {
    let primary_key_fields = schema.unenforced_primary_key();
    let source_ids = primary_key_fields
        .iter()
        .map(|field| field.id)
        .collect::<Vec<_>>();
    if source_ids.is_empty() {
        return Err(Error::invalid_input(
            "topics require an unenforced primary key in the schema",
        ));
    }

    let mut parameters = HashMap::new();
    parameters.insert("num_buckets".to_string(), partition_count.to_string());
    let mut producer_parameters = HashMap::new();
    producer_parameters.insert(
        "source_column".to_string(),
        TOPIC_PRODUCER_ID_COLUMN.to_string(),
    );

    Ok(ShardSpec {
        spec_id: TOPIC_SHARD_SPEC_ID,
        fields: vec![
            ShardField {
                field_id: TOPIC_PARTITION_FIELD_ID.to_string(),
                source_ids,
                transform: Some(if primary_key_fields.len() == 1 {
                    "bucket".to_string()
                } else {
                    "multi_bucket".to_string()
                }),
                expression: None,
                result_type: "int32".to_string(),
                parameters,
            },
            ShardField {
                field_id: TOPIC_PRODUCER_FIELD_ID.to_string(),
                source_ids: vec![schema.field_id(TOPIC_PRODUCER_ID_COLUMN)?],
                transform: Some("identity".to_string()),
                expression: None,
                result_type: "utf8".to_string(),
                parameters: producer_parameters,
            },
        ],
    })
}

fn validate_topic_shard_spec(details: &MemWalIndexDetails) -> Result<u32> {
    let matching = details
        .shard_specs
        .iter()
        .filter(|spec| {
            spec.fields.len() == 2
                && spec.fields[0].field_id == TOPIC_PARTITION_FIELD_ID
                && spec.fields[1].field_id == TOPIC_PRODUCER_FIELD_ID
        })
        .collect::<Vec<_>>();
    if matching.len() != 1 {
        return Err(Error::invalid_input(format!(
            "topic MemWAL index must contain exactly one partition+producer shard spec, found {}",
            matching.len()
        )));
    }

    let spec = matching[0];
    if spec.spec_id != TOPIC_SHARD_SPEC_ID {
        return Err(Error::invalid_input(format!(
            "topic MemWAL shard spec id must be {}, got {}",
            TOPIC_SHARD_SPEC_ID, spec.spec_id
        )));
    }
    let partition_field = &spec.fields[0];
    match partition_field.transform.as_deref() {
        Some("bucket") | Some("multi_bucket") => {}
        other => {
            return Err(Error::invalid_input(format!(
                "topic MemWAL partition shard field must use bucket or multi_bucket transform, got {:?}",
                other
            )));
        }
    }
    let partition_count = partition_field
        .parameters
        .get("num_buckets")
        .ok_or_else(|| {
            Error::invalid_input("topic MemWAL partition shard spec is missing num_buckets")
        })?
        .parse::<u32>()
        .map_err(|e| {
            Error::invalid_input(format!(
                "topic MemWAL partition shard spec num_buckets must be a u32: {}",
                e
            ))
        })?;
    if partition_count == 0 {
        return Err(Error::invalid_input(
            "topic MemWAL partition shard spec num_buckets must be greater than 0",
        ));
    }
    if partition_count > i32::MAX as u32 {
        return Err(Error::invalid_input(format!(
            "topic MemWAL partition shard spec num_buckets {} exceeds supported maximum {}",
            partition_count,
            i32::MAX
        )));
    }

    let producer_field = &spec.fields[1];
    if producer_field.transform.as_deref() != Some("identity") {
        return Err(Error::invalid_input(format!(
            "topic MemWAL producer shard field must use identity transform, got {:?}",
            producer_field.transform
        )));
    }
    if producer_field.result_type != "utf8" {
        return Err(Error::invalid_input(format!(
            "topic MemWAL producer shard field must produce utf8, got {}",
            producer_field.result_type
        )));
    }
    Ok(partition_count)
}

fn topic_shard_id(partition_id: u32, producer_id: &str) -> Uuid {
    let mut input = Vec::with_capacity(44 + producer_id.len());
    input.extend_from_slice(b"lance_topic_physical_shard_v2");
    input.extend_from_slice(&TOPIC_SHARD_SPEC_ID.to_le_bytes());
    input.extend_from_slice(&partition_id.to_le_bytes());
    input.extend_from_slice(&(producer_id.len() as u64).to_le_bytes());
    input.extend_from_slice(producer_id.as_bytes());

    let mut bytes = [0; 16];
    for seed in 0..4_u32 {
        let hash = murmur3_x86_32(&input, seed);
        bytes[(seed as usize) * 4..(seed as usize + 1) * 4].copy_from_slice(&hash.to_be_bytes());
    }
    bytes[6] = (bytes[6] & 0x0f) | 0x50;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Uuid::from_bytes(bytes)
}

async fn topic_partitions_from_mem_wal_listing(
    dataset: &Dataset,
    partition_count: u32,
) -> Result<Vec<TopicPartition>> {
    let mem_wal_shards = dataset.list_mem_wal_shards_latest().await?;
    let mut partitions = Vec::with_capacity(mem_wal_shards.len());
    let mut seen = HashSet::with_capacity(mem_wal_shards.len());
    for shard in mem_wal_shards {
        if shard.shard_spec_id != TOPIC_SHARD_SPEC_ID {
            return Err(Error::invalid_input(format!(
                "MemWAL shard manifest for shard {} has shard_spec_id {}, expected {}",
                shard.shard_id, shard.shard_spec_id, TOPIC_SHARD_SPEC_ID
            )));
        }
        let partition_id = shard_field_value(&shard, TOPIC_PARTITION_FIELD_ID)?;
        let producer_id = shard_field_string_value(&shard, TOPIC_PRODUCER_FIELD_ID)?;
        if partition_id >= partition_count {
            return Err(Error::invalid_input(format!(
                "MemWAL shard manifest partition_id {} is outside [0, {})",
                partition_id, partition_count
            )));
        }
        if !seen.insert((partition_id, producer_id.clone())) {
            return Err(Error::invalid_input(format!(
                "MemWAL shard manifests contain duplicate partition_id {} producer_id '{}'",
                partition_id, producer_id
            )));
        }
        partitions.push(TopicPartition {
            partition_id,
            producer_id,
            shard_id: shard.shard_id,
            shard_spec_id: shard.shard_spec_id,
        });
    }

    partitions.sort_by(|a, b| {
        a.partition_id
            .cmp(&b.partition_id)
            .then_with(|| a.producer_id.cmp(&b.producer_id))
    });
    Ok(partitions)
}

fn shard_field_value(snapshot: &MemWalShardSnapshot, field_id: &str) -> Result<u32> {
    let value = snapshot.shard_field_values.get(field_id).ok_or_else(|| {
        Error::invalid_input(format!(
            "MemWAL shard manifest for shard {} is missing '{}' field",
            snapshot.shard_id, field_id
        ))
    })?;
    if *value < 0 {
        return Err(Error::invalid_input(format!(
            "MemWAL shard manifest for shard {} has negative '{}' value {}",
            snapshot.shard_id, field_id, value
        )));
    }
    Ok(*value as u32)
}

#[derive(Debug, Clone)]
struct ConsumerGroupShard {
    consumer_position: u32,
    partition_id: u32,
    shard_id: Uuid,
    shard_spec_id: u32,
}

/// Builder for creating or opening a topic consumer group table.
#[derive(Debug, Clone)]
pub struct ConsumerGroupBuilder {
    topic: Topic,
    group_id: String,
}

impl ConsumerGroupBuilder {
    /// Create a consumer group builder.
    pub fn new(topic: Topic, group_id: impl Into<String>) -> Self {
        Self {
            topic,
            group_id: group_id.into(),
        }
    }

    /// Create the consumer group table and then open it.
    pub async fn create(self) -> Result<ConsumerGroup> {
        ConsumerGroup::create(self.topic, self.group_id).await
    }

    /// Open an existing consumer group table.
    pub async fn open(self) -> Result<ConsumerGroup> {
        ConsumerGroup::open(self.topic, self.group_id).await
    }

    /// Open the consumer group table, creating it first if it does not exist.
    pub async fn open_or_create(self) -> Result<ConsumerGroup> {
        ConsumerGroup::open_or_create(self.topic, self.group_id).await
    }
}

/// A topic consumer group backed by its own Lance table and MemWAL.
#[derive(Debug, Clone)]
pub struct ConsumerGroup {
    topic: Topic,
    group_id: String,
    table_id: Arc<Vec<String>>,
    store: ConsumerGroupStore,
}

impl ConsumerGroup {
    async fn create(topic: Topic, group_id: String) -> Result<Self> {
        metadata::validate_group_id(&group_id)?;
        let table_id = topic.consumer_group_table_id(&group_id)?;
        create_consumer_group_dataset(topic.namespace_client.clone(), table_id).await?;
        Self::open(topic, group_id).await
    }

    async fn open(topic: Topic, group_id: String) -> Result<Self> {
        metadata::validate_group_id(&group_id)?;
        let table_id = topic.consumer_group_table_id(&group_id)?;
        let dataset =
            open_namespace_dataset(topic.namespace_client.clone(), table_id.clone()).await?;
        Self::from_dataset(topic, group_id, table_id, dataset).await
    }

    async fn open_or_create(topic: Topic, group_id: String) -> Result<Self> {
        metadata::validate_group_id(&group_id)?;
        let table_id = topic.consumer_group_table_id(&group_id)?;
        let dataset =
            open_or_create_consumer_group_dataset(topic.namespace_client.clone(), table_id.clone())
                .await?;
        Self::from_dataset(topic, group_id, table_id, dataset).await
    }

    async fn from_dataset(
        topic: Topic,
        group_id: String,
        table_id: Vec<String>,
        dataset: Dataset,
    ) -> Result<Self> {
        let store = ConsumerGroupStore::from_dataset(group_id.clone(), dataset).await?;
        Ok(Self {
            topic,
            group_id,
            table_id: Arc::new(table_id),
            store,
        })
    }

    /// Consumer group id.
    pub fn group_id(&self) -> &str {
        &self.group_id
    }

    /// Namespace table id used to store this consumer group's commits.
    pub fn table_id(&self) -> &[String] {
        self.table_id.as_slice()
    }

    /// Backing consumer group Lance dataset.
    pub fn dataset(&self) -> &Dataset {
        self.store.dataset.as_ref()
    }

    /// Create a consumer at the given position within this group.
    ///
    /// `position` is 0-indexed and must be less than `total`. If `total` exceeds
    /// the topic's partition count, some consumers will be idle.
    pub async fn consumer(&self, position: u32, total: u32) -> Result<Consumer> {
        Consumer::open(self.clone(), position, total).await
    }
}

#[derive(Debug, Clone)]
struct ConsumerGroupStore {
    dataset: Arc<Dataset>,
    object_store: Arc<ObjectStore>,
    base_path: Path,
    group_id: String,
}

impl ConsumerGroupStore {
    async fn from_dataset(group_id: String, dataset: Dataset) -> Result<Self> {
        validate_consumer_group_table(dataset.schema(), &dataset.mem_wal_index_details().await?)?;

        Ok(Self {
            object_store: Arc::new(dataset.object_store().clone()),
            base_path: dataset.branch_location().path,
            dataset: Arc::new(dataset),
            group_id,
        })
    }

    async fn writers(
        &self,
        position: u32,
        assigned_partitions: &[u32],
    ) -> Result<HashMap<u32, ConsumerGroupWriter>> {
        let mut writers = HashMap::with_capacity(assigned_partitions.len());
        for &partition_id in assigned_partitions {
            let shard = self
                .ensure_consumer_partition_shard(position, partition_id)
                .await?;
            let writer = WalAppender::open(
                self.object_store.clone(),
                self.base_path.clone(),
                shard.partition_id,
                format!("consumer-{}", position),
                shard.shard_id,
                shard.shard_spec_id,
            )
            .await?;
            writers.insert(
                partition_id,
                ConsumerGroupWriter {
                    position,
                    partition_id,
                    writer,
                },
            );
        }
        Ok(writers)
    }

    async fn read_all_offsets(&self) -> Result<Vec<ConsumerGroupOffset>> {
        let mut offsets = HashMap::<(u32, String), u64>::new();
        for shard in consumer_group_shards_from_mem_wal_listing(self.dataset.as_ref()).await? {
            let tailer = WalTailer::new(
                self.object_store.clone(),
                self.base_path.clone(),
                shard.partition_id,
                format!("consumer-{}", shard.consumer_position),
                shard.shard_id,
            );
            let mut position = tailer.first_position().await?;
            let next_position = tailer.next_position().await?;
            while position < next_position {
                if let Some(entry) = tailer.read_entry(position).await? {
                    for batch in &entry.batches {
                        for offset in consumer_group_offsets_from_batch(batch)? {
                            offsets
                                .entry((offset.partition_id, offset.producer_id.clone()))
                                .and_modify(|position| {
                                    *position = (*position).max(offset.next_entry_position)
                                })
                                .or_insert(offset.next_entry_position);
                        }
                    }
                }
                position = position.checked_add(1).ok_or_else(|| {
                    Error::io(format!(
                        "consumer group '{}' WAL entry position overflow for position {} partition_id {}",
                        self.group_id, shard.consumer_position, shard.partition_id
                    ))
                })?;
            }
        }

        let mut offsets = offsets
            .into_iter()
            .map(
                |((partition_id, producer_id), next_entry_position)| ConsumerGroupOffset {
                    partition_id,
                    producer_id,
                    next_entry_position,
                },
            )
            .collect::<Vec<_>>();
        offsets.sort_by(|a, b| {
            a.partition_id
                .cmp(&b.partition_id)
                .then_with(|| a.producer_id.cmp(&b.producer_id))
        });
        Ok(offsets)
    }

    async fn ensure_consumer_partition_shard(
        &self,
        position: u32,
        partition_id: u32,
    ) -> Result<ConsumerGroupShard> {
        if partition_id > i32::MAX as u32 {
            return Err(Error::invalid_input(format!(
                "consumer group partition_id {} exceeds supported maximum {}",
                partition_id,
                i32::MAX
            )));
        }
        let shard_id = consumer_group_shard_id(position, partition_id);
        let shard_field_values = HashMap::from([
            (
                CONSUMER_GROUP_POSITION_FIELD_ID.to_string(),
                position as i32,
            ),
            (
                CONSUMER_GROUP_PARTITION_FIELD_ID.to_string(),
                partition_id as i32,
            ),
        ]);
        ShardManifestStore::new(self.object_store.clone(), &self.base_path, shard_id, 2)
            .initialize_shard(CONSUMER_GROUP_SHARD_SPEC_ID, shard_field_values)
            .await?;

        Ok(ConsumerGroupShard {
            consumer_position: position,
            partition_id,
            shard_id,
            shard_spec_id: CONSUMER_GROUP_SHARD_SPEC_ID,
        })
    }
}

#[derive(Debug)]
struct ConsumerGroupWriter {
    position: u32,
    partition_id: u32,
    writer: WalAppender,
}

impl ConsumerGroupWriter {
    async fn commit_offsets(&self, offsets: &[ConsumerGroupOffset]) -> Result<()> {
        if offsets.is_empty() {
            return Ok(());
        }
        if offsets
            .iter()
            .any(|offset| offset.partition_id != self.partition_id)
        {
            return Err(Error::invalid_input(format!(
                "consumer group writer for partition_id {} cannot commit offsets for other partitions",
                self.partition_id
            )));
        }
        self.writer.check_fenced().await?;
        self.writer
            .append(vec![consumer_group_offset_batch(
                self.position,
                self.partition_id,
                offsets,
            )?])
            .await?;
        Ok(())
    }
}

async fn open_or_create_consumer_group_dataset(
    namespace_client: Arc<dyn LanceNamespace>,
    table_id: Vec<String>,
) -> Result<Dataset> {
    match open_namespace_dataset(namespace_client.clone(), table_id.clone()).await {
        Ok(dataset) => Ok(dataset),
        Err(error) if is_namespace_error_code(&error, ErrorCode::TableNotFound) => {
            match create_consumer_group_dataset(namespace_client.clone(), table_id.clone()).await {
                Ok(dataset) => Ok(dataset),
                Err(error) if is_namespace_error_code(&error, ErrorCode::TableAlreadyExists) => {
                    open_namespace_dataset(namespace_client, table_id).await
                }
                Err(error) => Err(error),
            }
        }
        Err(error) => Err(error),
    }
}

async fn create_consumer_group_dataset(
    namespace_client: Arc<dyn LanceNamespace>,
    table_id: Vec<String>,
) -> Result<Dataset> {
    validate_table_id(&table_id)?;
    let schema = consumer_group_schema();
    let reader = RecordBatchIterator::new(
        vec![Ok(RecordBatch::new_empty(schema.clone()))].into_iter(),
        schema,
    );
    let mut dataset = Dataset::write_into_namespace(
        reader,
        namespace_client,
        table_id,
        Some(topic_write_params()),
    )
    .await?;
    let shard_spec = consumer_group_shard_spec(dataset.schema())?;
    dataset
        .initialize_mem_wal(MemWalConfig {
            shard_spec: Some(shard_spec),
            maintained_indexes: Vec::new(),
        })
        .await?;
    Ok(dataset)
}

fn is_namespace_error_code(error: &Error, code: ErrorCode) -> bool {
    match error {
        Error::Namespace { source, .. } => {
            source
                .downcast_ref::<NamespaceError>()
                .is_some_and(|namespace_error| namespace_error.code() == code)
                || source
                    .downcast_ref::<Error>()
                    .is_some_and(|error| is_namespace_error_code(error, code))
        }
        _ => false,
    }
}

fn consumer_group_schema() -> Arc<ArrowSchema> {
    let pk_metadata =
        HashMap::from([(LANCE_UNENFORCED_PRIMARY_KEY.to_string(), "true".to_string())]);
    Arc::new(ArrowSchema::new(vec![
        Field::new(CONSUMER_GROUP_POSITION_COLUMN, DataType::UInt32, false)
            .with_metadata(pk_metadata),
        Field::new(CONSUMER_GROUP_PARTITION_ID_COLUMN, DataType::UInt32, false),
        Field::new(CONSUMER_GROUP_PRODUCER_ID_COLUMN, DataType::Utf8, false),
        Field::new(
            CONSUMER_GROUP_NEXT_ENTRY_POSITION_COLUMN,
            DataType::UInt64,
            false,
        ),
    ]))
}

fn validate_consumer_group_table(
    schema: &LanceSchema,
    details: &Option<MemWalIndexDetails>,
) -> Result<()> {
    let arrow_schema = ArrowSchema::from(schema);
    if arrow_schema.fields() != consumer_group_schema().fields() {
        return Err(Error::invalid_input(format!(
            "consumer group table schema must be fixed consumer offset schema: expected fields {:?}, got fields {:?}",
            consumer_group_schema().fields(),
            arrow_schema.fields()
        )));
    }

    let details = details
        .as_ref()
        .ok_or_else(|| Error::invalid_input("consumer group table is missing MemWAL index"))?;
    validate_consumer_group_shard_spec(details)?;
    Ok(())
}

fn consumer_group_shard_spec(schema: &LanceSchema) -> Result<ShardSpec> {
    Ok(ShardSpec {
        spec_id: CONSUMER_GROUP_SHARD_SPEC_ID,
        fields: vec![
            ShardField {
                field_id: CONSUMER_GROUP_POSITION_FIELD_ID.to_string(),
                source_ids: vec![schema.field_id(CONSUMER_GROUP_POSITION_COLUMN)?],
                transform: Some("identity".to_string()),
                expression: None,
                result_type: "int32".to_string(),
                parameters: HashMap::from([(
                    "source_column".to_string(),
                    CONSUMER_GROUP_POSITION_COLUMN.to_string(),
                )]),
            },
            ShardField {
                field_id: CONSUMER_GROUP_PARTITION_FIELD_ID.to_string(),
                source_ids: vec![schema.field_id(CONSUMER_GROUP_PARTITION_ID_COLUMN)?],
                transform: Some("identity".to_string()),
                expression: None,
                result_type: "int32".to_string(),
                parameters: HashMap::from([(
                    "source_column".to_string(),
                    CONSUMER_GROUP_PARTITION_ID_COLUMN.to_string(),
                )]),
            },
        ],
    })
}

fn validate_consumer_group_shard_spec(details: &MemWalIndexDetails) -> Result<()> {
    let matching = details
        .shard_specs
        .iter()
        .filter(|spec| {
            spec.fields.len() == 2
                && spec.spec_id == CONSUMER_GROUP_SHARD_SPEC_ID
                && spec.fields[0].field_id == CONSUMER_GROUP_POSITION_FIELD_ID
                && spec.fields[1].field_id == CONSUMER_GROUP_PARTITION_FIELD_ID
        })
        .collect::<Vec<_>>();
    if matching.len() != 1 {
        return Err(Error::invalid_input(format!(
            "consumer group MemWAL index must contain exactly one position+partition shard spec, found {}",
            matching.len()
        )));
    }

    for field in &matching[0].fields {
        if field.transform.as_deref() != Some("identity") {
            return Err(Error::invalid_input(format!(
                "consumer group MemWAL shard field must use identity transform, got {:?}",
                field.transform
            )));
        }
        if field.result_type != "int32" {
            return Err(Error::invalid_input(format!(
                "consumer group MemWAL shard field must produce int32, got {}",
                field.result_type
            )));
        }
    }
    Ok(())
}

async fn consumer_group_shards_from_mem_wal_listing(
    dataset: &Dataset,
) -> Result<Vec<ConsumerGroupShard>> {
    let mem_wal_shards = dataset.list_mem_wal_shards_latest().await?;
    let mut shards = Vec::with_capacity(mem_wal_shards.len());
    let mut seen = HashSet::with_capacity(mem_wal_shards.len());
    for shard in mem_wal_shards {
        if shard.shard_spec_id != CONSUMER_GROUP_SHARD_SPEC_ID {
            return Err(Error::invalid_input(format!(
                "consumer group MemWAL shard manifest for shard {} has shard_spec_id {}, expected {}",
                shard.shard_id, shard.shard_spec_id, CONSUMER_GROUP_SHARD_SPEC_ID
            )));
        }
        let consumer_position = shard_field_value(&shard, CONSUMER_GROUP_POSITION_FIELD_ID)?;
        let partition_id = shard_field_value(&shard, CONSUMER_GROUP_PARTITION_FIELD_ID)?;
        if !seen.insert((consumer_position, partition_id)) {
            return Err(Error::invalid_input(format!(
                "consumer group MemWAL shard manifests contain duplicate position {} partition_id {}",
                consumer_position, partition_id
            )));
        }
        shards.push(ConsumerGroupShard {
            consumer_position,
            partition_id,
            shard_id: shard.shard_id,
            shard_spec_id: shard.shard_spec_id,
        });
    }
    shards.sort_by_key(|shard| (shard.consumer_position, shard.partition_id));
    Ok(shards)
}

fn shard_field_string_value(snapshot: &MemWalShardSnapshot, field_id: &str) -> Result<String> {
    snapshot
        .shard_field_string_values
        .get(field_id)
        .cloned()
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "MemWAL shard manifest for shard {} is missing string '{}' field",
                snapshot.shard_id, field_id
            ))
        })
}

fn consumer_group_shard_id(position: u32, partition_id: u32) -> Uuid {
    let mut input = Vec::with_capacity(48);
    input.extend_from_slice(b"lance_topic_consumer_group_shard_v2");
    input.extend_from_slice(&CONSUMER_GROUP_SHARD_SPEC_ID.to_le_bytes());
    input.extend_from_slice(&position.to_le_bytes());
    input.extend_from_slice(&partition_id.to_le_bytes());

    let mut bytes = [0; 16];
    for seed in 0..4_u32 {
        let hash = murmur3_x86_32(&input, seed);
        bytes[(seed as usize) * 4..(seed as usize + 1) * 4].copy_from_slice(&hash.to_be_bytes());
    }
    bytes[6] = (bytes[6] & 0x0f) | 0x50;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Uuid::from_bytes(bytes)
}

fn consumer_group_offset_batch(
    position: u32,
    partition_id: u32,
    offsets: &[ConsumerGroupOffset],
) -> Result<RecordBatch> {
    use arrow_array::UInt32Array;
    RecordBatch::try_new(
        consumer_group_schema(),
        vec![
            Arc::new(UInt32Array::from_iter_values(std::iter::repeat_n(
                position,
                offsets.len(),
            ))) as ArrayRef,
            Arc::new(UInt32Array::from_iter_values(std::iter::repeat_n(
                partition_id,
                offsets.len(),
            ))) as ArrayRef,
            Arc::new(StringArray::from_iter_values(
                offsets.iter().map(|offset| offset.producer_id.as_str()),
            )) as ArrayRef,
            Arc::new(UInt64Array::from_iter_values(
                offsets.iter().map(|offset| offset.next_entry_position),
            )) as ArrayRef,
        ],
    )
    .map_err(|e| {
        Error::arrow(format!(
            "failed to create consumer group offset batch: {}",
            e
        ))
    })
}

fn consumer_group_offsets_from_batch(batch: &RecordBatch) -> Result<Vec<ConsumerGroupOffset>> {
    let partition_ids = batch
        .column_by_name(CONSUMER_GROUP_PARTITION_ID_COLUMN)
        .and_then(|c| c.as_any().downcast_ref::<arrow_array::UInt32Array>())
        .ok_or_else(|| {
            Error::invalid_input("consumer group offset batch is missing topic_partition_id column")
        })?;
    let producer_ids = batch
        .column_by_name(CONSUMER_GROUP_PRODUCER_ID_COLUMN)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| {
            Error::invalid_input("consumer group offset batch is missing producer_id column")
        })?;
    let next_entry_positions = batch
        .column_by_name(CONSUMER_GROUP_NEXT_ENTRY_POSITION_COLUMN)
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| {
            Error::invalid_input(
                "consumer group offset batch is missing next_entry_position column",
            )
        })?;
    let mut offsets = Vec::with_capacity(batch.num_rows());
    for row_idx in 0..batch.num_rows() {
        offsets.push(ConsumerGroupOffset {
            partition_id: partition_ids.value(row_idx),
            producer_id: producer_ids.value(row_idx).to_string(),
            next_entry_position: next_entry_positions.value(row_idx),
        });
    }
    Ok(offsets)
}

/// Topic producer.
#[derive(Debug, Clone)]
pub struct Producer {
    topic: Topic,
    producer_id: String,
    partition_writers: Arc<Vec<WalAppender>>,
    fenced: Arc<AtomicBool>,
}

impl Producer {
    async fn open(topic: Topic, producer_id: String) -> Result<Self> {
        if producer_id.is_empty() {
            return Err(Error::invalid_input("producer_id cannot be empty"));
        }

        let mut partition_writers = Vec::with_capacity(topic.partition_count() as usize);
        for partition_id in 0..topic.partition_count() {
            let partition = topic
                .ensure_partition_shard(partition_id, &producer_id)
                .await?;
            partition_writers.push(
                WalAppender::open(
                    topic.object_store.clone(),
                    topic.base_path.clone(),
                    partition.partition_id,
                    partition.producer_id.clone(),
                    partition.shard_id,
                    partition.shard_spec_id,
                )
                .await?,
            );
        }

        Ok(Self {
            topic,
            producer_id,
            partition_writers: Arc::new(partition_writers),
            fenced: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Producer id used by this producer.
    pub fn producer_id(&self) -> &str {
        &self.producer_id
    }

    /// Send one JSON payload.
    pub async fn send(&self, id: impl Into<String>, payload: Value) -> Result<ProduceResult> {
        self.send_batch([id], [payload]).await
    }

    /// Send JSON payloads keyed by id.
    pub async fn send_batch<I, Id, P>(&self, ids: I, payloads: P) -> Result<ProduceResult>
    where
        I: IntoIterator<Item = Id>,
        Id: Into<String>,
        P: IntoIterator<Item = Value>,
    {
        let ids = ids.into_iter().map(Into::into).collect::<Vec<_>>();
        let payloads = payloads.into_iter().collect::<Vec<_>>();
        self.send_record_batch(message_batch(ids, payloads, &self.producer_id)?)
            .await
    }

    async fn send_record_batch(&self, batch: RecordBatch) -> Result<ProduceResult> {
        self.ensure_not_fenced()?;
        if batch.num_rows() == 0 {
            return Err(Error::invalid_input("cannot send an empty record batch"));
        }
        self.topic.validate_batch_schema(&batch)?;

        let partitioner = Partitioner::new(
            self.topic.partition_count(),
            self.topic.primary_key_columns().to_vec(),
        )?;
        let partitioned_batches = partitioner.partition_batch(&batch)?;

        let produce_futures = partitioned_batches
            .into_iter()
            .map(|(partition_id, partition_batch)| {
                let partition_writer = self.partition_writer(partition_id)?;
                Ok(async move { partition_writer.append(vec![partition_batch]).await })
            })
            .collect::<Result<Vec<_>>>()?;
        let results = join_all(produce_futures).await;
        let mut entries = Vec::with_capacity(results.len());
        let mut first_error = None;
        for result in results {
            match result {
                Ok(entry) => entries.push(entry),
                Err(error) => {
                    if is_fencing_error(&error) {
                        self.mark_fenced();
                    }
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }

        Ok(ProduceResult {
            num_rows: batch.num_rows(),
            entries,
        })
    }

    #[cfg(test)]
    async fn send_to_partition(
        &self,
        partition_id: u32,
        batches: Vec<RecordBatch>,
    ) -> Result<ProduceResult> {
        self.ensure_not_fenced()?;
        for batch in &batches {
            self.topic.validate_batch_schema(batch)?;
        }
        let num_rows = batches.iter().map(RecordBatch::num_rows).sum();
        let partition_writer = self.partition_writer(partition_id)?;
        let entry = match partition_writer.append(batches).await {
            Ok(entry) => entry,
            Err(error) => {
                if is_fencing_error(&error) {
                    self.mark_fenced();
                }
                return Err(error);
            }
        };

        Ok(ProduceResult {
            num_rows,
            entries: vec![entry],
        })
    }

    /// Check every partition writer owned by this producer for fencing.
    pub async fn check_fenced(&self) -> Result<()> {
        match try_join_all(self.partition_writers.iter().map(WalAppender::check_fenced)).await {
            Ok(_) => Ok(()),
            Err(error) => {
                if is_fencing_error(&error) {
                    self.mark_fenced();
                }
                Err(error)
            }
        }
    }

    fn ensure_not_fenced(&self) -> Result<()> {
        if self.fenced.load(Ordering::Acquire) {
            return Err(Error::io(format!(
                "producer_id {} has been fenced",
                self.producer_id
            )));
        }
        Ok(())
    }

    fn mark_fenced(&self) {
        self.fenced.store(true, Ordering::Release);
    }

    fn partition_writer(&self, partition_id: u32) -> Result<&WalAppender> {
        self.partition_writers
            .get(partition_id as usize)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "partition_id {} is out of range for topic with {} partitions",
                    partition_id,
                    self.topic.partition_count()
                ))
            })
    }
}

/// Result of a producer send.
#[derive(Debug, Clone)]
pub struct ProduceResult {
    /// Total input rows accepted by the producer.
    pub num_rows: usize,
    /// WAL entries created by this send.
    pub entries: Vec<WalAppendResult>,
}

/// A decoded topic message.
#[derive(Debug, Clone, PartialEq)]
pub struct TopicMessage {
    /// Message id used as the topic partition key.
    pub id: String,
    /// JSON payload.
    pub payload: Value,
}

/// Poll options.
#[derive(Debug, Clone)]
pub struct PollOptions {
    /// Maximum WAL entries to read from each assigned physical producer shard.
    pub max_entries_per_partition: usize,
}

impl Default for PollOptions {
    fn default() -> Self {
        Self {
            max_entries_per_partition: 1,
        }
    }
}

/// A batch of records read from a topic partition.
#[derive(Debug, Clone)]
pub struct TopicBatch {
    /// Partition this batch came from.
    pub partition_id: u32,
    /// Producer shard this batch came from.
    pub producer_id: String,
    /// WAL entry position.
    pub entry_position: u64,
    /// Next offset to commit after processing this batch.
    pub next_entry_position: u64,
    /// Arrow batches stored in the WAL entry.
    pub batches: Vec<RecordBatch>,
}

impl TopicBatch {
    /// Create a topic batch from a WAL entry.
    pub fn from_entry(entry: TopicEntry) -> Result<Self> {
        let next_entry_position = entry.entry_position.checked_add(1).ok_or_else(|| {
            Error::io(format!(
                "entry_position overflow for partition_id {} at {}",
                entry.partition_id, entry.entry_position
            ))
        })?;

        Ok(Self {
            partition_id: entry.partition_id,
            producer_id: entry.producer_id,
            entry_position: entry.entry_position,
            next_entry_position,
            batches: entry.batches,
        })
    }

    /// Number of rows in this topic batch.
    pub fn num_rows(&self) -> usize {
        self.batches.iter().map(RecordBatch::num_rows).sum()
    }

    /// Decode this topic batch into id/payload messages.
    pub fn messages(&self) -> Result<Vec<TopicMessage>> {
        let mut messages = Vec::with_capacity(self.num_rows());
        for batch in &self.batches {
            messages.extend(record_batch_to_messages(batch)?);
        }
        Ok(messages)
    }
}

/// Topic consumer.
#[derive(Debug)]
pub struct Consumer {
    topic: Topic,
    consumer_group: ConsumerGroupStore,
    offset_writers: HashMap<u32, ConsumerGroupWriter>,
    fenced: Arc<AtomicBool>,
    group_id: String,
    position: u32,
    total: u32,
    assigned_partitions: Vec<u32>,
    assigned_shards: Vec<(u32, String)>,
    next_entry_positions: HashMap<(u32, String), u64>,
}

impl Consumer {
    async fn open(consumer_group: ConsumerGroup, position: u32, total: u32) -> Result<Self> {
        if total == 0 {
            return Err(Error::invalid_input(
                "total consumers must be greater than 0",
            ));
        }
        if position >= total {
            return Err(Error::invalid_input(format!(
                "consumer position {} must be less than total {}",
                position, total
            )));
        }

        let topic = consumer_group.topic.clone();
        topic.refresh_partitions().await?;
        let store = consumer_group.store.clone();
        let assigned_partitions = compute_assigned_partitions(&topic, position, total);
        let offset_writers = store.writers(position, &assigned_partitions).await?;
        let assigned_shards = assigned_shards_for_partitions(&topic, &assigned_partitions)?;
        let committed_positions = committed_position_map(&store).await?;
        let next_entry_positions = initial_next_entry_positions(
            &topic,
            &assigned_shards,
            &committed_positions,
            StartPosition::Earliest,
        )
        .await?;

        Ok(Self {
            topic,
            consumer_group: store,
            offset_writers,
            fenced: Arc::new(AtomicBool::new(false)),
            group_id: consumer_group.group_id,
            position,
            total,
            assigned_partitions,
            assigned_shards,
            next_entry_positions,
        })
    }

    /// Consumer group id.
    pub fn group_id(&self) -> &str {
        &self.group_id
    }

    /// Consumer position within the group.
    pub fn position(&self) -> u32 {
        self.position
    }

    /// Total consumers in the group.
    pub fn total(&self) -> u32 {
        self.total
    }

    /// Topic partition ids assigned to this consumer.
    pub fn assigned_partitions(&self) -> &[u32] {
        &self.assigned_partitions
    }

    /// Poll at most one WAL entry from each assigned physical producer shard.
    ///
    /// If reading any assigned partition fails, the entire poll fails and the
    /// consumer keeps its previous in-memory offsets.
    pub async fn poll(&mut self) -> Result<Vec<TopicBatch>> {
        self.poll_with_options(PollOptions::default()).await
    }

    /// Poll topic data with explicit options.
    ///
    /// If reading any assigned partition fails, the entire poll fails and the
    /// consumer keeps its previous in-memory offsets.
    pub async fn poll_with_options(&mut self, options: PollOptions) -> Result<Vec<TopicBatch>> {
        self.ensure_not_fenced()?;
        if let Err(error) = self.refresh_assignment().await {
            if is_fencing_error(&error) {
                self.mark_fenced();
            }
            return Err(error);
        }
        let result = poll_shards(
            self.topic.clone(),
            &self.assigned_shards,
            &mut self.next_entry_positions,
            options,
        )
        .await;
        if let Err(error) = &result
            && is_fencing_error(error)
        {
            self.mark_fenced();
        }
        result
    }

    /// Commit offsets for processed topic batches.
    pub async fn commit(&self, batches: &[TopicBatch]) -> Result<()> {
        self.ensure_not_fenced()?;
        let mut latest = HashMap::<(u32, String), u64>::new();
        for batch in batches {
            if !self.assigned_partitions.contains(&batch.partition_id) {
                return Err(Error::invalid_input(format!(
                    "cannot commit offset for unassigned partition_id {}",
                    batch.partition_id
                )));
            }
            self.topic
                .partition(batch.partition_id, &batch.producer_id)?;
            latest
                .entry((batch.partition_id, batch.producer_id.clone()))
                .and_modify(|position| *position = (*position).max(batch.next_entry_position))
                .or_insert(batch.next_entry_position);
        }

        if let Err(error) = self.commit_latest_offsets(latest).await {
            if is_fencing_error(&error) {
                self.mark_fenced();
            }
            return Err(error);
        }

        Ok(())
    }

    async fn refresh_assignment(&mut self) -> Result<()> {
        self.topic.refresh_partitions().await?;
        let assigned_shards =
            assigned_shards_for_partitions(&self.topic, &self.assigned_partitions)?;
        let assigned_shard_set = assigned_shards.iter().cloned().collect::<HashSet<_>>();
        self.next_entry_positions
            .retain(|key, _| assigned_shard_set.contains(key));
        let committed_positions = committed_position_map(&self.consumer_group).await?;
        add_missing_next_entry_positions(
            &self.topic,
            &assigned_shards,
            &committed_positions,
            StartPosition::Earliest,
            &mut self.next_entry_positions,
        )
        .await?;
        self.assigned_shards = assigned_shards;
        Ok(())
    }

    /// Commit the consumer's current in-memory offsets.
    pub async fn commit_current(&self) -> Result<()> {
        self.ensure_not_fenced()?;
        let mut latest = HashMap::<(u32, String), u64>::new();
        for (partition_id, producer_id) in &self.assigned_shards {
            let key = (*partition_id, producer_id.clone());
            let next_entry_position = *self.next_entry_positions.get(&key).ok_or_else(|| {
                Error::internal(format!(
                    "missing next entry position for assigned partition_id {} producer_id '{}'",
                    partition_id, producer_id
                ))
            })?;
            latest.insert(key, next_entry_position);
        }
        if let Err(error) = self.commit_latest_offsets(latest).await {
            if is_fencing_error(&error) {
                self.mark_fenced();
            }
            return Err(error);
        }
        Ok(())
    }

    async fn commit_latest_offsets(&self, latest: HashMap<(u32, String), u64>) -> Result<()> {
        let mut by_partition = HashMap::<u32, Vec<ConsumerGroupOffset>>::new();
        for ((partition_id, producer_id), next_entry_position) in latest {
            by_partition
                .entry(partition_id)
                .or_default()
                .push(ConsumerGroupOffset {
                    partition_id,
                    producer_id,
                    next_entry_position,
                });
        }

        let commits = by_partition.iter().map(|(partition_id, offsets)| {
            let writer = self.offset_writers.get(partition_id).ok_or_else(|| {
                Error::internal(format!(
                    "missing consumer group writer for partition_id {}",
                    partition_id
                ))
            });
            async move { writer?.commit_offsets(offsets).await }
        });
        try_join_all(commits).await?;
        Ok(())
    }

    fn ensure_not_fenced(&self) -> Result<()> {
        if self.fenced.load(Ordering::Acquire) {
            return Err(Error::io(format!(
                "consumer position {} in group '{}' has been fenced",
                self.position, self.group_id
            )));
        }
        Ok(())
    }

    fn mark_fenced(&self) {
        self.fenced.store(true, Ordering::Release);
    }
}

async fn initial_next_entry_positions(
    topic: &Topic,
    assigned_shards: &[(u32, String)],
    committed_positions: &HashMap<(u32, String), u64>,
    start_position: StartPosition,
) -> Result<HashMap<(u32, String), u64>> {
    let mut next_entry_positions = HashMap::with_capacity(assigned_shards.len());
    add_missing_next_entry_positions(
        topic,
        assigned_shards,
        committed_positions,
        start_position,
        &mut next_entry_positions,
    )
    .await?;
    Ok(next_entry_positions)
}

async fn add_missing_next_entry_positions(
    topic: &Topic,
    assigned_shards: &[(u32, String)],
    committed_positions: &HashMap<(u32, String), u64>,
    start_position: StartPosition,
    next_entry_positions: &mut HashMap<(u32, String), u64>,
) -> Result<()> {
    for (partition_id, producer_id) in assigned_shards {
        let key = (*partition_id, producer_id.clone());
        if next_entry_positions.contains_key(&key) {
            continue;
        }
        let position = if let Some(position) = committed_positions.get(&key) {
            *position
        } else {
            match start_position {
                StartPosition::Earliest => {
                    topic
                        .wal_tailer(*partition_id, producer_id)?
                        .first_position()
                        .await?
                }
                StartPosition::Latest => {
                    topic
                        .wal_tailer(*partition_id, producer_id)?
                        .next_position()
                        .await?
                }
            }
        };

        next_entry_positions.insert(key, position);
    }
    Ok(())
}

async fn poll_shards(
    topic: Topic,
    assigned_shards: &[(u32, String)],
    next_entry_positions: &mut HashMap<(u32, String), u64>,
    options: PollOptions,
) -> Result<Vec<TopicBatch>> {
    if options.max_entries_per_partition == 0 {
        return Err(Error::invalid_input(
            "max_entries_per_partition must be greater than 0",
        ));
    }

    let read_futures = assigned_shards.iter().map(|(partition_id, producer_id)| {
        let topic = topic.clone();
        let key = (*partition_id, producer_id.clone());
        let start_position = next_entry_positions.get(&key).copied().ok_or_else(|| {
            Error::internal(format!(
                "missing next entry position for assigned partition_id {} producer_id {}",
                partition_id, producer_id
            ))
        });
        async move {
            let mut position = start_position?;
            let tailer = topic.wal_tailer(key.0, &key.1)?;
            let mut batches = Vec::new();
            for _ in 0..options.max_entries_per_partition {
                let Some(entry) = tailer.read_entry(position).await? else {
                    break;
                };
                let batch = TopicBatch::from_entry(entry)?;
                position = batch.next_entry_position;
                batches.push(batch);
            }
            Ok::<_, Error>((key, position, batches))
        }
    });
    let results = try_join_all(read_futures).await?;

    let mut out = Vec::new();
    let mut updated_positions = next_entry_positions.clone();
    for (key, position, batches) in results {
        updated_positions.insert(key, position);
        out.extend(batches);
    }
    *next_entry_positions = updated_positions;
    Ok(out)
}

async fn committed_position_map(
    consumer_group: &ConsumerGroupStore,
) -> Result<HashMap<(u32, String), u64>> {
    Ok(consumer_group
        .read_all_offsets()
        .await?
        .into_iter()
        .map(|offset| {
            (
                (offset.partition_id, offset.producer_id),
                offset.next_entry_position,
            )
        })
        .collect())
}

fn compute_assigned_partitions(topic: &Topic, position: u32, total: u32) -> Vec<u32> {
    (0..topic.partition_count())
        .filter(|&partition_id| assigned_position_for_partition(partition_id, total) == position)
        .collect()
}

fn assigned_shards_for_partitions(
    topic: &Topic,
    assigned_partitions: &[u32],
) -> Result<Vec<(u32, String)>> {
    let assigned_partition_set = assigned_partitions.iter().copied().collect::<HashSet<_>>();
    Ok(topic
        .current_partitions()?
        .into_iter()
        .filter(|partition| assigned_partition_set.contains(&partition.partition_id))
        .map(|partition| (partition.partition_id, partition.producer_id))
        .collect())
}

fn validate_logical_partition(topic: &Topic, partition_id: u32) -> Result<()> {
    if partition_id >= topic.partition_count() {
        return Err(Error::invalid_input(format!(
            "partition_id {} is out of range for topic with {} logical partitions",
            partition_id,
            topic.partition_count()
        )));
    }
    Ok(())
}

fn is_fencing_error(error: &Error) -> bool {
    error.to_string().contains("fenced")
}

fn message_batch(ids: Vec<String>, payloads: Vec<Value>, producer_id: &str) -> Result<RecordBatch> {
    if ids.len() != payloads.len() {
        return Err(Error::invalid_input(format!(
            "ids length ({}) must match payloads length ({})",
            ids.len(),
            payloads.len()
        )));
    }
    if ids.is_empty() {
        return Err(Error::invalid_input("cannot send an empty message batch"));
    }

    let payload_strings = payloads
        .into_iter()
        .map(|payload| {
            serde_json::to_string(&payload).map_err(|e| {
                Error::invalid_input(format!("failed to encode topic payload as JSON: {}", e))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let payload = JsonArray::try_from_iter(payload_strings.iter().map(Some))
        .map_err(|e| {
            Error::invalid_input(format!("failed to encode topic payload as JSONB: {}", e))
        })?
        .into_inner();

    RecordBatch::try_new(
        topic_schema(),
        vec![
            Arc::new(StringArray::from(ids)) as ArrayRef,
            Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                producer_id,
                payload_strings.len(),
            ))) as ArrayRef,
            Arc::new(payload) as ArrayRef,
        ],
    )
    .map_err(|e| Error::arrow(format!("failed to create topic message batch: {}", e)))
}

fn record_batch_to_messages(batch: &RecordBatch) -> Result<Vec<TopicMessage>> {
    let id_idx = batch
        .schema()
        .index_of(TOPIC_ID_COLUMN)
        .map_err(|e| Error::invalid_input(format!("topic batch is missing id column: {}", e)))?;
    let payload_idx = batch.schema().index_of(TOPIC_PAYLOAD_COLUMN).map_err(|e| {
        Error::invalid_input(format!("topic batch is missing payload column: {}", e))
    })?;

    let ids = batch
        .column(id_idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "topic id column must be Utf8, got {}",
                batch.column(id_idx).data_type()
            ))
        })?;
    let payloads = batch
        .column(payload_idx)
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "topic payload column must be LargeBinary JSONB, got {}",
                batch.column(payload_idx).data_type()
            ))
        })?;

    let mut messages = Vec::with_capacity(batch.num_rows());
    for row_idx in 0..batch.num_rows() {
        if ids.is_null(row_idx) || payloads.is_null(row_idx) {
            return Err(Error::invalid_input(format!(
                "topic message at row {} contains null id or payload",
                row_idx
            )));
        }
        let payload_json = decode_json(payloads.value(row_idx));
        let payload = serde_json::from_str(&payload_json).map_err(|e| {
            Error::invalid_input(format!(
                "failed to decode topic payload JSON at row {}: {}",
                row_idx, e
            ))
        })?;
        messages.push(TopicMessage {
            id: ids.value(row_idx).to_string(),
            payload,
        });
    }
    Ok(messages)
}

fn take_rows(batch: &RecordBatch, row_indices: &[u32]) -> Result<RecordBatch> {
    if row_indices.len() == batch.num_rows()
        && row_indices
            .iter()
            .enumerate()
            .all(|(idx, row_idx)| *row_idx as usize == idx)
    {
        return Ok(batch.clone());
    }

    let indices = arrow_array::UInt32Array::from(row_indices.to_vec());
    take_record_batch(batch, &indices).map_err(|e| {
        Error::io(format!(
            "failed to take partitioned record batch rows: {}",
            e
        ))
    })
}

fn shard_base_path(base_path: &Path, shard_id: &Uuid) -> Path {
    base_path
        .child("_mem_wal")
        .child(shard_id.as_hyphenated().to_string())
}

fn shard_wal_path(base_path: &Path, shard_id: &Uuid) -> Path {
    shard_base_path(base_path, shard_id).child("wal")
}

#[cfg(test)]
mod tests {
    use arrow_array::Int32Array;
    use arrow_schema::{DataType, Field, Schema};
    use lance::index::DatasetIndexExt;
    use lance_arrow::json::is_json_field;
    use lance_index::mem_wal::MEM_WAL_INDEX_NAME;
    use serde_json::{Value, json};
    use tempfile::TempDir;

    use super::*;

    fn topic_uri(temp_dir: &TempDir) -> String {
        format!("file://{}", temp_dir.path().display())
    }

    fn topic_root(temp_dir: &TempDir) -> String {
        temp_dir.path().to_string_lossy().to_string()
    }

    fn topic_table_path(temp_dir: &TempDir) -> std::path::PathBuf {
        temp_dir.path().join("topic.lance")
    }

    fn topic_builder(temp_dir: &TempDir) -> TopicBuilder {
        Topic::builder().directory(topic_root(temp_dir), ["topic"])
    }

    async fn create_topic(temp_dir: &TempDir, partition_count: u32) -> Topic {
        topic_builder(temp_dir)
            .partition_count(partition_count)
            .create()
            .await
            .unwrap()
    }

    async fn topic_consumer(topic: &Topic, group_id: &str) -> Consumer {
        topic
            .consumer_group(group_id)
            .open_or_create()
            .await
            .unwrap()
            .consumer(0, 1)
            .await
            .unwrap()
    }

    fn mismatched_batch(ids: Vec<i32>) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)])),
            vec![Arc::new(Int32Array::from(ids))],
        )
        .unwrap()
    }

    fn message_values(start: i32, end: i32) -> (Vec<String>, Vec<Value>) {
        let ids = (start..end).map(|id| id.to_string()).collect::<Vec<_>>();
        let payloads = (start..end)
            .map(|id| json!({ "value": id }))
            .collect::<Vec<_>>();
        (ids, payloads)
    }

    async fn produce_to_partition(
        topic: &Topic,
        producer_id: &str,
        partition_id: u32,
        values: Vec<i32>,
    ) -> ProduceResult {
        let batch = batch_for_partition(topic, producer_id, partition_id, values);
        topic
            .producer(producer_id)
            .await
            .unwrap()
            .send_to_partition(partition_id, vec![batch])
            .await
            .unwrap()
    }

    fn batch_for_partition(
        topic: &Topic,
        producer_id: &str,
        partition_id: u32,
        values: Vec<i32>,
    ) -> RecordBatch {
        let mut ids = Vec::with_capacity(values.len());
        let mut payloads = Vec::with_capacity(values.len());
        for value in values {
            ids.push(id_for_partition(topic, producer_id, partition_id, value));
            payloads.push(json!({ "value": value }));
        }
        message_batch(ids, payloads, producer_id).unwrap()
    }

    fn id_for_partition(topic: &Topic, producer_id: &str, partition_id: u32, value: i32) -> String {
        let partitioner = Partitioner::new(
            topic.partition_count(),
            topic.primary_key_columns().to_vec(),
        )
        .unwrap();
        for nonce in 0..10_000 {
            let id = format!("partition-{partition_id}-value-{value}-{nonce}");
            let candidate = message_batch(
                vec![id.clone()],
                vec![json!({ "value": value })],
                producer_id,
            )
            .unwrap();
            let partitions = partitioner.partition_batch(&candidate).unwrap();
            if partitions.len() == 1 && partitions[0].0 == partition_id {
                return id;
            }
        }
        panic!("failed to find id for partition {partition_id}");
    }

    fn count_rows(batches: &[TopicBatch]) -> usize {
        batches.iter().map(TopicBatch::num_rows).sum()
    }

    async fn committed_offsets(topic: &Topic, group_id: &str) -> Vec<ConsumerGroupOffset> {
        topic
            .consumer_group(group_id)
            .open()
            .await
            .unwrap()
            .store
            .read_all_offsets()
            .await
            .unwrap()
    }

    fn test_wal_entry_filename(entry_position: u64) -> String {
        format!("{:064b}.arrow", entry_position.reverse_bits())
    }

    #[tokio::test]
    async fn test_partition_writer_and_tailer_round_trip() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = topic_uri(&temp_dir);
        let (store, base_path) = ObjectStore::from_uri(&uri).await.unwrap();
        let shard_id = Uuid::new_v4();

        let writer = WalAppender::open(store.clone(), base_path.clone(), 0, "p7", shard_id, 1)
            .await
            .unwrap();
        let first = writer
            .append(vec![
                message_batch(
                    vec!["1".to_string(), "2".to_string()],
                    vec![json!({ "value": 1 }), json!({ "value": 2 })],
                    "p7",
                )
                .unwrap(),
            ])
            .await
            .unwrap();
        let second = writer
            .append(vec![
                message_batch(vec!["3".to_string()], vec![json!({ "value": 3 })], "p7").unwrap(),
            ])
            .await
            .unwrap();

        assert_eq!(first.entry_position, 1);
        assert_eq!(first.producer_id, "p7");
        assert_eq!(first.num_rows, 2);
        assert_eq!(second.entry_position, 2);

        let tailer = WalTailer::new(store, base_path, 0, "p7", shard_id);
        let first_read = tailer.read_entry(1).await.unwrap().unwrap();
        let second_read = tailer.read_entry(2).await.unwrap().unwrap();
        let missing = tailer.read_entry(3).await.unwrap();

        assert_eq!(first_read.producer_id, "p7");
        assert_eq!(first_read.batches.len(), 1);
        assert_eq!(first_read.batches[0].num_rows(), 2);
        assert_eq!(second_read.batches[0].num_rows(), 1);
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn test_producer_hash_partitions_by_primary_key() {
        let temp_dir = tempfile::tempdir().unwrap();
        let topic = create_topic(&temp_dir, 2).await;

        assert_eq!(topic.primary_key_columns(), &["id".to_string()]);

        let (ids, payloads) = message_values(0, 20);
        let result = topic
            .producer("p0")
            .await
            .unwrap()
            .send_batch(ids, payloads)
            .await
            .unwrap();

        assert_eq!(result.num_rows, 20);
        assert_eq!(result.entries.len(), 2);
        assert!(result.entries.iter().all(|entry| entry.num_rows > 0));
        assert!(result.entries.iter().all(|entry| entry.producer_id == "p0"));
    }

    #[tokio::test]
    async fn test_producer_send_and_consumer_messages() {
        let temp_dir = tempfile::tempdir().unwrap();
        let topic = create_topic(&temp_dir, 1).await;

        topic
            .producer("p0")
            .await
            .unwrap()
            .send("message-1", json!({ "kind": "created", "version": 1 }))
            .await
            .unwrap();

        let mut consumer = topic_consumer(&topic, "message-group").await;
        let polled = consumer.poll().await.unwrap();
        let messages = polled[0].messages().unwrap();
        assert_eq!(
            messages,
            vec![TopicMessage {
                id: "message-1".to_string(),
                payload: json!({ "kind": "created", "version": 1 }),
            }]
        );
    }

    #[tokio::test]
    async fn test_producer_rejects_empty_or_mismatched_batches() {
        let temp_dir = tempfile::tempdir().unwrap();
        let topic = create_topic(&temp_dir, 1).await;
        let producer = topic.producer("p0").await.unwrap();

        let err = producer
            .send_batch(["message-1"], Vec::<Value>::new())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("ids length"), "{}", err);

        let err = producer
            .send_batch(Vec::<String>::new(), Vec::<Value>::new())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("empty message batch"), "{}", err);
    }

    #[tokio::test]
    async fn test_producer_shards_are_discovered_dynamically() {
        let temp_dir = tempfile::tempdir().unwrap();
        let topic = create_topic(&temp_dir, 1).await;

        assert_eq!(topic.partition_count(), 1);
        topic
            .producer("p0")
            .await
            .unwrap()
            .send("same-key", json!({ "producer": 0 }))
            .await
            .unwrap();

        let mut consumer = topic_consumer(&topic, "multi-producer-group").await;
        let polled = consumer.poll().await.unwrap();
        assert_eq!(count_rows(&polled), 1);
        assert_eq!(polled[0].producer_id, "p0");

        topic
            .producer("p17")
            .await
            .unwrap()
            .send("same-key", json!({ "producer": 17 }))
            .await
            .unwrap();
        let polled = consumer.poll().await.unwrap();
        let producer_ids = polled
            .iter()
            .map(|batch| batch.producer_id.clone())
            .collect::<std::collections::HashSet<_>>();

        assert_eq!(count_rows(&polled), 1);
        assert_eq!(
            producer_ids,
            std::collections::HashSet::from(["p17".to_string()])
        );
        consumer.commit_current().await.unwrap();

        let offsets = committed_offsets(&topic, "multi-producer-group").await;
        assert_eq!(
            offsets
                .iter()
                .find(|offset| offset.partition_id == 0 && offset.producer_id == "p0")
                .map(|offset| offset.next_entry_position),
            Some(2)
        );
        assert_eq!(
            offsets
                .iter()
                .find(|offset| offset.partition_id == 0 && offset.producer_id == "p17")
                .map(|offset| offset.next_entry_position),
            Some(2)
        );
    }

    #[tokio::test]
    async fn test_producer_detects_fencing_after_wal_create_conflict() {
        let temp_dir = tempfile::tempdir().unwrap();
        let topic = create_topic(&temp_dir, 2).await;

        let first = topic.producer("p9").await.unwrap();
        first
            .send_to_partition(0, vec![batch_for_partition(&topic, "p9", 0, vec![1])])
            .await
            .unwrap();

        let second = topic.producer("p9").await.unwrap();
        second.check_fenced().await.unwrap();

        let partitions = topic.refresh_partitions().await.unwrap();
        assert_eq!(partitions.len(), 2);
        assert_eq!(
            partitions
                .iter()
                .map(|partition| partition.shard_id)
                .collect::<std::collections::HashSet<_>>(),
            std::collections::HashSet::from([topic_shard_id(0, "p9"), topic_shard_id(1, "p9")])
        );

        let err = first.check_fenced().await.unwrap_err();
        assert!(err.to_string().contains("fenced"), "{}", err);

        let err = first
            .send_to_partition(1, vec![batch_for_partition(&topic, "p9", 1, vec![2])])
            .await
            .unwrap_err();
        assert!(err.to_string().contains("fenced"), "{}", err);
        second
            .send_to_partition(0, vec![batch_for_partition(&topic, "p9", 0, vec![3])])
            .await
            .unwrap();

        let err = first
            .send_to_partition(0, vec![batch_for_partition(&topic, "p9", 0, vec![4])])
            .await
            .unwrap_err();
        assert!(err.to_string().contains("fenced"), "{}", err);
    }

    #[tokio::test]
    async fn test_topic_schema_is_persisted_and_enforced() {
        let temp_dir = tempfile::tempdir().unwrap();
        let topic = create_topic(&temp_dir, 2).await;

        let reopened = topic_builder(&temp_dir).open().await.unwrap();
        assert_eq!(topic.schema().fields(), reopened.schema().fields());

        let err = reopened
            .producer("p0")
            .await
            .unwrap()
            .send_to_partition(0, vec![mismatched_batch(vec![1])])
            .await
            .unwrap_err();
        assert!(err.to_string().contains("schema does not match"), "{}", err);
    }

    #[tokio::test]
    async fn test_open_rejects_table_without_mem_wal_index() {
        let temp_dir = tempfile::tempdir().unwrap();
        let schema = topic_schema();
        let reader = RecordBatchIterator::new(
            vec![Ok(RecordBatch::new_empty(schema.clone()))].into_iter(),
            schema,
        );
        let namespace = Arc::new(
            DirectoryNamespaceBuilder::new(topic_root(&temp_dir))
                .build()
                .await
                .unwrap(),
        );
        let table_id = vec!["topic".to_string()];
        Dataset::write_into_namespace(
            reader,
            namespace.clone(),
            table_id.clone(),
            Some(topic_write_params()),
        )
        .await
        .unwrap();

        let err = Topic::builder()
            .namespace(namespace, table_id)
            .open()
            .await
            .unwrap_err();
        assert!(err.to_string().contains("MemWAL index"), "{}", err);
    }

    #[tokio::test]
    async fn test_builder_can_create_and_open_namespace_topic() {
        let temp_dir = tempfile::tempdir().unwrap();
        let namespace = Arc::new(
            DirectoryNamespaceBuilder::new(topic_root(&temp_dir))
                .build()
                .await
                .unwrap(),
        );
        let table_id = vec!["workspace".to_string(), "topic".to_string()];

        let topic = Topic::builder()
            .namespace(namespace.clone(), table_id.clone())
            .partition_count(2)
            .create()
            .await
            .unwrap();
        assert_eq!(topic.partition_count(), 2);

        let reopened = Topic::builder()
            .namespace(namespace, table_id)
            .open()
            .await
            .unwrap();
        assert_eq!(reopened.partition_count(), 2);
    }

    #[tokio::test]
    async fn test_topic_creates_real_mem_wal_index_for_primary_key() {
        let temp_dir = tempfile::tempdir().unwrap();
        let topic = create_topic(&temp_dir, 4).await;

        let details = topic.mem_wal_index_details();
        assert_eq!(details.num_shards, 0);
        assert_eq!(details.shard_specs.len(), 1);
        assert_eq!(details.shard_specs[0].spec_id, 1);
        assert_eq!(details.shard_specs[0].fields.len(), 2);
        assert_eq!(
            details.shard_specs[0].fields[0].transform.as_deref(),
            Some("bucket")
        );
        assert_eq!(
            details.shard_specs[0].fields[0]
                .parameters
                .get("num_buckets"),
            Some(&"4".to_string())
        );
        assert_eq!(
            details.shard_specs[0].fields[1].transform.as_deref(),
            Some("identity")
        );
        assert_eq!(
            details.shard_specs[0].fields[1]
                .parameters
                .get("source_column"),
            Some(&TOPIC_PRODUCER_ID_COLUMN.to_string())
        );
        assert!(topic.partitions().unwrap().is_empty());

        produce_to_partition(&topic, "p7", 3, vec![1]).await;
        let partitions = topic.refresh_partitions().await.unwrap();
        assert_eq!(partitions.len(), 4);
        let partition = partitions
            .iter()
            .find(|partition| partition.partition_id == 3 && partition.producer_id == "p7")
            .unwrap();
        assert_eq!(partition.shard_spec_id, 1);
        assert_eq!(partition.shard_id, topic_shard_id(3, "p7"));

        let mem_wal_index = topic
            .dataset()
            .load_index_by_name(MEM_WAL_INDEX_NAME)
            .await
            .unwrap();
        assert!(mem_wal_index.is_some());
        assert!(
            topic
                .dataset()
                .metadata()
                .get("lance_topic.partition_count")
                .is_none()
        );
        assert!(
            topic
                .dataset()
                .metadata()
                .get("lance_topic.producer_count")
                .is_none()
        );
        assert!(
            topic_table_path(&temp_dir)
                .join("_mem_wal")
                .join(partition.shard_id.to_string())
                .join("manifest")
                .exists()
        );
        assert!(
            !topic_table_path(&temp_dir)
                .join("_lance_topic")
                .join("config.json")
                .exists()
        );
    }

    #[tokio::test]
    async fn test_topic_uses_fixed_json_payload_schema() {
        let temp_dir = tempfile::tempdir().unwrap();
        let topic = create_topic(&temp_dir, 1).await;
        let schema = topic.schema();

        let id_field = schema.field_with_name(TOPIC_ID_COLUMN).unwrap();
        assert_eq!(id_field.data_type(), &DataType::Utf8);
        assert_eq!(
            id_field.metadata().get(LANCE_UNENFORCED_PRIMARY_KEY),
            Some(&"true".to_string())
        );
        let producer_id_field = schema.field_with_name(TOPIC_PRODUCER_ID_COLUMN).unwrap();
        assert_eq!(producer_id_field.data_type(), &DataType::Utf8);
        assert!(is_json_field(
            schema.field_with_name(TOPIC_PAYLOAD_COLUMN).unwrap()
        ));
        assert!(
            !topic
                .dataset()
                .config()
                .contains_key("lance.auto_cleanup.interval")
        );
        assert!(
            !topic
                .dataset()
                .config()
                .contains_key("lance.auto_cleanup.older_than")
        );
    }

    #[tokio::test]
    async fn test_wal_tailer_reads_entries() {
        let temp_dir = tempfile::tempdir().unwrap();
        let topic = create_topic(&temp_dir, 2).await;

        let result = produce_to_partition(&topic, "p0", 1, vec![1, 2]).await;
        let entry = result.entries.first().unwrap();
        assert_eq!(entry.partition_id, 1);
        assert_eq!(entry.producer_id, "p0");

        let partition = topic.partition(1, "p0").unwrap();
        let tailer = WalTailer::new(
            topic.object_store.clone(),
            topic.base_path.clone(),
            1,
            "p0",
            partition.shard_id,
        );
        let next = tailer.next_position().await.unwrap();
        assert_eq!(next, 2);

        let read = tailer.read_entry(1).await.unwrap().unwrap();
        assert_eq!(read.batches[0].num_rows(), 2);
        assert!(tailer.read_entry(2).await.unwrap().is_none());

        produce_to_partition(&topic, "p0", 1, vec![3]).await;
        let read = tailer.read_entry(2).await.unwrap().unwrap();
        assert_eq!(read.batches[0].num_rows(), 1);
    }

    #[tokio::test]
    async fn test_consumed_batches_can_be_reprocessed() {
        let temp_dir = tempfile::tempdir().unwrap();
        let topic = create_topic(&temp_dir, 1).await;
        produce_to_partition(&topic, "p0", 0, vec![1]).await;

        let mut consumer = topic_consumer(&topic, "reprocess-group").await;
        let polled = consumer.poll().await.unwrap();
        assert_eq!(polled.len(), 1);

        topic
            .producer("p0")
            .await
            .unwrap()
            .send_to_partition(0, polled[0].batches.clone())
            .await
            .unwrap();
        let polled_again = consumer.poll().await.unwrap();
        assert_eq!(count_rows(&polled_again), 1);
    }

    #[tokio::test]
    async fn test_consumer_commit_and_resume() {
        let temp_dir = tempfile::tempdir().unwrap();
        let topic = create_topic(&temp_dir, 2).await;

        let (ids, payloads) = message_values(0, 20);
        topic
            .producer("p0")
            .await
            .unwrap()
            .send_batch(ids, payloads)
            .await
            .unwrap();

        let mut consumer = topic_consumer(&topic, "group-a").await;
        let first_poll = consumer.poll().await.unwrap();
        assert_eq!(count_rows(&first_poll), 20);
        consumer.commit(&first_poll).await.unwrap();

        let offsets = committed_offsets(&topic, "group-a").await;
        assert_eq!(
            offsets
                .iter()
                .find(|offset| offset.partition_id == 0 && offset.producer_id == "p0")
                .map(|offset| offset.next_entry_position),
            Some(2)
        );
        assert_eq!(
            offsets
                .iter()
                .find(|offset| offset.partition_id == 1 && offset.producer_id == "p0")
                .map(|offset| offset.next_entry_position),
            Some(2)
        );

        let mut resumed = topic_consumer(&topic, "group-a").await;
        assert!(resumed.poll().await.unwrap().is_empty());

        let (ids, payloads) = message_values(20, 30);
        topic
            .producer("p0")
            .await
            .unwrap()
            .send_batch(ids, payloads)
            .await
            .unwrap();
        let second_poll = resumed.poll().await.unwrap();
        assert_eq!(count_rows(&second_poll), 10);
    }

    #[tokio::test]
    async fn test_consumer_group_builder_create_then_open() {
        let temp_dir = tempfile::tempdir().unwrap();
        let topic = create_topic(&temp_dir, 2).await;

        let group = topic
            .consumer_group("created-group")
            .create()
            .await
            .unwrap();
        assert_eq!(group.group_id(), "created-group");
        assert!(
            group
                .dataset()
                .mem_wal_index_details()
                .await
                .unwrap()
                .is_some()
        );

        let reopened = topic.consumer_group("created-group").open().await.unwrap();
        let producer = topic.producer("p0").await.unwrap();
        producer
            .send("created-group-message", json!({ "value": 1 }))
            .await
            .unwrap();
        let mut consumer = reopened.consumer(0, 1).await.unwrap();
        let polled = consumer.poll().await.unwrap();
        assert_eq!(count_rows(&polled), 1);
        consumer.commit(&polled).await.unwrap();
    }

    #[tokio::test]
    async fn test_consumer_claims_epoch_per_partition() {
        let temp_dir = tempfile::tempdir().unwrap();
        let topic = create_topic(&temp_dir, 2).await;
        produce_to_partition(&topic, "p0", 0, vec![1]).await;
        produce_to_partition(&topic, "p0", 1, vec![2]).await;
        let group = topic
            .consumer_group("fenced-group")
            .open_or_create()
            .await
            .unwrap();

        let first = group.consumer(0, 1).await.unwrap();
        let shards = consumer_group_shards_from_mem_wal_listing(group.dataset())
            .await
            .unwrap();
        assert_eq!(shards.len(), 2);
        assert_eq!(
            shards
                .iter()
                .map(|shard| (shard.consumer_position, shard.partition_id))
                .collect::<std::collections::HashSet<_>>(),
            std::collections::HashSet::from([(0, 0), (0, 1)])
        );

        let second = group.consumer(0, 1).await.unwrap();
        assert_eq!(second.position(), 0);

        let err = first.commit_current().await.unwrap_err();
        assert!(err.to_string().contains("fenced"), "{}", err);
    }

    #[tokio::test]
    async fn test_consumer_group_assignment_with_positions() {
        let temp_dir = tempfile::tempdir().unwrap();
        let topic = create_topic(&temp_dir, 8).await;
        for partition_id in 0..topic.partition_count() {
            produce_to_partition(&topic, "p0", partition_id, vec![partition_id as i32]).await;
        }

        let group = topic
            .consumer_group("hashed-group")
            .open_or_create()
            .await
            .unwrap();
        let total = 3u32;
        let mut consumers = Vec::new();
        for position in 0..total {
            consumers.push(group.consumer(position, total).await.unwrap());
        }

        let mut assigned = std::collections::HashSet::new();
        let mut total_rows = 0;
        for consumer in &mut consumers {
            let polled = consumer.poll().await.unwrap();
            for partition_id in consumer.assigned_partitions() {
                assert!(
                    assigned.insert(*partition_id),
                    "topic partition {} assigned more than once",
                    partition_id
                );
            }
            total_rows += count_rows(&polled);
        }

        let expected = (0..topic.partition_count()).collect::<std::collections::HashSet<_>>();
        assert_eq!(assigned, expected);
        assert_eq!(total_rows, topic.partition_count() as usize);
    }

    #[tokio::test]
    async fn test_consumer_validates_position_and_total() {
        let temp_dir = tempfile::tempdir().unwrap();
        let topic = create_topic(&temp_dir, 1).await;
        let group = topic
            .consumer_group("bad-group")
            .open_or_create()
            .await
            .unwrap();

        let err = group.consumer(0, 0).await.unwrap_err();
        assert!(err.to_string().contains("total"), "{}", err);

        let err = group.consumer(3, 2).await.unwrap_err();
        assert!(err.to_string().contains("position"), "{}", err);
    }

    #[tokio::test]
    async fn test_concurrent_partition_commits_are_merged() {
        let temp_dir = tempfile::tempdir().unwrap();
        let topic = create_topic(&temp_dir, 8).await;
        for partition_id in 0..topic.partition_count() {
            produce_to_partition(&topic, "p0", partition_id, vec![partition_id as i32]).await;
        }

        let group = topic
            .consumer_group("merged-group")
            .open_or_create()
            .await
            .unwrap();
        let total = 3u32;
        let mut consumers = Vec::new();
        for position in 0..total {
            consumers.push(group.consumer(position, total).await.unwrap());
        }

        let mut total_rows = 0;
        for consumer in &mut consumers {
            total_rows += count_rows(&consumer.poll().await.unwrap());
        }
        assert_eq!(total_rows, topic.partition_count() as usize);

        futures::future::try_join_all(consumers.iter().map(Consumer::commit_current))
            .await
            .unwrap();

        let mut resumed_rows = 0;
        for position in 0..total {
            let mut consumer = group.consumer(position, total).await.unwrap();
            resumed_rows += count_rows(&consumer.poll().await.unwrap());
        }
        assert_eq!(resumed_rows, 0);
    }

    #[tokio::test]
    async fn test_wal_tailer_rejects_corrupt_entry() {
        let temp_dir = tempfile::tempdir().unwrap();
        let topic = create_topic(&temp_dir, 1).await;
        produce_to_partition(&topic, "p0", 0, vec![1]).await;

        let shard_id = topic.partition(0, "p0").unwrap().shard_id;
        let wal_dir = topic_table_path(&temp_dir)
            .join("_mem_wal")
            .join(shard_id.to_string())
            .join("wal");
        std::fs::write(wal_dir.join(test_wal_entry_filename(0)), b"not arrow").unwrap();

        let tailer = WalTailer::new(
            topic.object_store.clone(),
            topic.base_path.clone(),
            0,
            "p0",
            shard_id,
        );
        // Corrupt file at position 0 (before FIRST_WAL_ENTRY_POSITION=1) causes read error
        assert!(tailer.read_entry(0).await.is_err());
        // Valid entry at position 1
        let read = tailer.read_entry(1).await.unwrap().unwrap();
        assert_eq!(read.batches[0].num_rows(), 1);
    }

    #[tokio::test]
    async fn test_committed_offsets_do_not_regress() {
        let temp_dir = tempfile::tempdir().unwrap();
        let topic = create_topic(&temp_dir, 1).await;
        produce_to_partition(&topic, "p0", 0, vec![1]).await;
        produce_to_partition(&topic, "p0", 0, vec![2]).await;
        produce_to_partition(&topic, "p0", 0, vec![3]).await;

        let mut consumer = topic_consumer(&topic, "monotonic-group").await;
        let batches = consumer
            .poll_with_options(PollOptions {
                max_entries_per_partition: 3,
            })
            .await
            .unwrap();
        assert_eq!(batches.len(), 3);

        consumer.commit(&[batches[0].clone()]).await.unwrap();
        consumer.commit(&[batches[2].clone()]).await.unwrap();
        consumer.commit(&[batches[1].clone()]).await.unwrap();

        let offsets = committed_offsets(&topic, "monotonic-group").await;
        assert_eq!(
            offsets
                .iter()
                .find(|offset| offset.partition_id == 0 && offset.producer_id == "p0")
                .map(|offset| offset.next_entry_position),
            Some(4)
        );

        let mut resumed = topic_consumer(&topic, "monotonic-group").await;
        assert!(resumed.poll().await.unwrap().is_empty());
    }

    #[test]
    fn test_group_id_rejects_path_segments() {
        assert!(metadata::validate_group_id("service.v1").is_ok());
        assert!(metadata::validate_group_id("a/commits").is_err());
        assert!(metadata::validate_group_id("a\\commits").is_err());
        assert!(metadata::validate_group_id("a$commits").is_err());
        assert!(metadata::validate_group_id("..").is_err());
    }
}
