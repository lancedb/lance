// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Kafka-like queue primitives backed by Lance MemWAL-compatible WAL files.
//!
//! The initial queue API provides at-least-once delivery. Consumers commit
//! entry-level offsets to the backing Lance table metadata.

mod metadata;
mod partition;
mod wal;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, LargeBinaryArray, RecordBatch, RecordBatchIterator, StringArray, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use arrow_select::take::take_record_batch;
use futures::future::try_join_all;
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::mem_wal::{
    DatasetMemWalExt, MemWalConfig, MemWalShardConfig, MemWalShardSnapshot,
};
use lance_arrow::json::{JsonArray, decode_json, json_field};
use lance_core::datatypes::Schema as LanceSchema;
use lance_core::{Error, Result};
use lance_index::mem_wal::{MemWalIndexDetails, ShardField, ShardSpec};
use lance_io::object_store::ObjectStore;
use lance_namespace::LanceNamespace;
use object_store::path::Path;
use serde_json::Value;
use uuid::Uuid;

use metadata::ConsumerGroupOffset;
pub use metadata::StartPosition;
use partition::{Partitioner, consumer_slot_for_partition};
pub use wal::{QueueEntry, WalAppendResult, WalAppender, WalTailer};

const QUEUE_ID_COLUMN: &str = "id";
const QUEUE_PRODUCER_ID_COLUMN: &str = "producer_id";
const QUEUE_PAYLOAD_COLUMN: &str = "payload";
const LANCE_UNENFORCED_PRIMARY_KEY: &str = "lance-schema:unenforced-primary-key";
const QUEUE_SHARD_SPEC_ID: u32 = 1;
const QUEUE_PARTITION_FIELD_ID: &str = "queue_partition_id";
const QUEUE_PRODUCER_FIELD_ID: &str = "producer_id";

/// Configuration for creating a queue.
#[derive(Debug, Clone)]
pub struct QueueConfig {
    partition_count: u32,
    producer_count: u32,
}

impl QueueConfig {
    /// Create queue configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of queue partitions.
    pub fn with_partition_count(mut self, partition_count: u32) -> Self {
        self.partition_count = partition_count;
        self
    }

    /// Set the number of producer shard slots per logical queue partition.
    pub fn with_producer_count(mut self, producer_count: u32) -> Self {
        self.producer_count = producer_count;
        self
    }
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            partition_count: 1,
            producer_count: 1,
        }
    }
}

#[derive(Debug, Clone)]
enum QueueTarget {
    Uri(String),
    Namespace {
        namespace_client: Arc<dyn LanceNamespace>,
        table_id: Vec<String>,
    },
}

/// Builder for creating or opening a queue.
#[derive(Debug, Clone, Default)]
pub struct QueueBuilder {
    target: Option<QueueTarget>,
    config: QueueConfig,
}

impl QueueBuilder {
    /// Create an empty queue builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a queue builder for a physical table URI.
    pub fn from_uri(uri: impl Into<String>) -> Self {
        Self::new().uri(uri)
    }

    /// Create a queue builder for a namespace-managed table.
    pub fn from_namespace<I, S>(namespace_client: Arc<dyn LanceNamespace>, table_id: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::new().namespace(namespace_client, table_id)
    }

    /// Set the physical table URI.
    pub fn uri(mut self, uri: impl Into<String>) -> Self {
        self.target = Some(QueueTarget::Uri(uri.into()));
        self
    }

    /// Set the namespace client and table identifier.
    pub fn namespace<I, S>(mut self, namespace_client: Arc<dyn LanceNamespace>, table_id: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.target = Some(QueueTarget::Namespace {
            namespace_client,
            table_id: table_id.into_iter().map(Into::into).collect(),
        });
        self
    }

    /// Set the number of queue partitions.
    pub fn partition_count(mut self, partition_count: u32) -> Self {
        self.config = self.config.with_partition_count(partition_count);
        self
    }

    /// Set the number of producer shard slots per logical queue partition.
    pub fn producer_count(mut self, producer_count: u32) -> Self {
        self.config = self.config.with_producer_count(producer_count);
        self
    }

    /// Create the queue table and initialize its MemWAL index.
    pub async fn create(self) -> Result<Queue> {
        let Self { target, config } = self;
        create_queue(required_target(target)?, config).await
    }

    /// Open an existing queue.
    pub async fn open(self) -> Result<Queue> {
        let Self { target, .. } = self;
        let dataset = match required_target(target)? {
            QueueTarget::Uri(uri) => Dataset::open(&uri).await?,
            QueueTarget::Namespace {
                namespace_client,
                table_id,
            } => {
                DatasetBuilder::from_namespace(namespace_client, table_id)
                    .await?
                    .load()
                    .await?
            }
        };
        Queue::from_dataset(dataset).await
    }
}

fn required_target(target: Option<QueueTarget>) -> Result<QueueTarget> {
    target.ok_or_else(|| {
        Error::invalid_input(
            "queue builder requires either a table URI or namespace client and table_id",
        )
    })
}

/// Queue partition metadata derived from the table's MemWAL shard snapshots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueuePartition {
    /// Logical queue partition id derived from `id`.
    pub partition_id: u32,
    /// Producer shard slot id.
    pub producer_id: u32,
    /// MemWAL shard id used to store this physical producer shard's WAL files.
    pub shard_id: Uuid,
    /// MemWAL shard spec used to route rows to this partition.
    pub shard_spec_id: u32,
}

/// A Lance queue rooted at a URI.
#[derive(Debug, Clone)]
pub struct Queue {
    uri: String,
    dataset: Arc<Dataset>,
    object_store: Arc<ObjectStore>,
    base_path: Path,
    schema: Arc<ArrowSchema>,
    primary_key_columns: Arc<Vec<String>>,
    mem_wal_index_details: Arc<MemWalIndexDetails>,
    partition_count: u32,
    producer_count: u32,
    partitions: Arc<Vec<QueuePartition>>,
}

impl Queue {
    /// Start building a queue.
    pub fn builder() -> QueueBuilder {
        QueueBuilder::new()
    }

    /// Create a new queue.
    pub async fn create(uri: impl AsRef<str>) -> Result<Self> {
        QueueBuilder::from_uri(uri.as_ref()).create().await
    }

    /// Create a new queue with configuration.
    pub async fn create_with_config(uri: impl AsRef<str>, config: QueueConfig) -> Result<Self> {
        QueueBuilder::from_uri(uri.as_ref())
            .partition_count(config.partition_count)
            .producer_count(config.producer_count)
            .create()
            .await
    }

    /// Open an existing queue.
    pub async fn open(uri: impl AsRef<str>) -> Result<Self> {
        QueueBuilder::from_uri(uri.as_ref()).open().await
    }

    async fn from_dataset(dataset: Dataset) -> Result<Self> {
        let uri = dataset.uri().to_string();
        let object_store = Arc::new(dataset.object_store().clone());
        let base_path = dataset.branch_location().path;
        let lance_schema = dataset.schema();
        validate_queue_schema(lance_schema)?;
        let primary_key_columns = primary_key_columns(lance_schema)?;
        let mem_wal_index_details = dataset.mem_wal_index_details().await?.ok_or_else(|| {
            Error::invalid_input(
                "queue table is missing MemWAL index; create it with QueueBuilder::create",
            )
        })?;
        let partitions = queue_partitions(&dataset, &mem_wal_index_details).await?;
        let partition_count = partitions
            .iter()
            .map(|partition| partition.partition_id)
            .max()
            .map(|partition_id| partition_id + 1)
            .unwrap_or(0);
        let producer_count = partitions
            .iter()
            .map(|partition| partition.producer_id)
            .max()
            .map(|producer_id| producer_id + 1)
            .unwrap_or(0);
        let schema = Arc::new(ArrowSchema::from(lance_schema));
        Ok(Self {
            uri,
            dataset: Arc::new(dataset),
            object_store,
            base_path,
            schema,
            primary_key_columns: Arc::new(primary_key_columns),
            mem_wal_index_details: Arc::new(mem_wal_index_details),
            partition_count,
            producer_count,
            partitions: Arc::new(partitions),
        })
    }

    /// Queue URI.
    pub fn uri(&self) -> &str {
        &self.uri
    }

    /// Number of queue partitions.
    pub fn partition_count(&self) -> u32 {
        self.partition_count
    }

    /// Number of configured producer shard slots.
    pub fn producer_count(&self) -> u32 {
        self.producer_count
    }

    /// Primary key columns used for hash partitioning.
    pub fn primary_key_columns(&self) -> &[String] {
        self.primary_key_columns.as_slice()
    }

    /// MemWAL index details describing queue shard routing.
    pub fn mem_wal_index_details(&self) -> &MemWalIndexDetails {
        self.mem_wal_index_details.as_ref()
    }

    /// Backing Lance dataset.
    pub fn dataset(&self) -> &Dataset {
        self.dataset.as_ref()
    }

    /// Queue record schema.
    pub fn schema(&self) -> &Arc<ArrowSchema> {
        &self.schema
    }

    /// Physical queue shard metadata.
    pub fn partitions(&self) -> &[QueuePartition] {
        self.partitions.as_slice()
    }

    /// Create a producer for this queue.
    pub fn producer(&self, producer_id: u32) -> Result<Producer> {
        Producer::new(self.clone(), producer_id)
    }

    /// Create a consumer for this queue.
    pub async fn consumer(&self, config: ConsumerConfig) -> Result<Consumer> {
        Consumer::open(self.clone(), config).await
    }

    fn partition(&self, partition_id: u32, producer_id: u32) -> Result<&QueuePartition> {
        self.partitions
            .iter()
            .find(|partition| {
                partition.partition_id == partition_id && partition.producer_id == producer_id
            })
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "partition_id {} producer_id {} is out of range for queue with {} partitions and {} producer slots",
                    partition_id,
                    producer_id,
                    self.partition_count(),
                    self.producer_count()
                ))
            })
    }

    fn wal_tailer(&self, partition_id: u32, producer_id: u32) -> Result<WalTailer> {
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
                "record batch schema does not match queue schema: expected fields {:?}, got fields {:?}",
                self.schema.fields(),
                batch.schema_ref().fields()
            )));
        }
        Ok(())
    }
}

async fn create_queue(target: QueueTarget, config: QueueConfig) -> Result<Queue> {
    if config.partition_count == 0 {
        return Err(Error::invalid_input(
            "partition_count must be greater than 0",
        ));
    }
    if config.producer_count == 0 {
        return Err(Error::invalid_input(
            "producer_count must be greater than 0",
        ));
    }
    if config.partition_count > i32::MAX as u32 {
        return Err(Error::invalid_input(format!(
            "partition_count {} exceeds supported maximum {}",
            config.partition_count,
            i32::MAX
        )));
    }
    if config.producer_count > i32::MAX as u32 {
        return Err(Error::invalid_input(format!(
            "producer_count {} exceeds supported maximum {}",
            config.producer_count,
            i32::MAX
        )));
    }
    let num_shards = config
        .partition_count
        .checked_mul(config.producer_count)
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "partition_count {} * producer_count {} overflows u32",
                config.partition_count, config.producer_count
            ))
        })?;

    let schema = queue_schema();
    let reader = RecordBatchIterator::new(
        vec![Ok(RecordBatch::new_empty(schema.clone()))].into_iter(),
        schema.clone(),
    );
    let mut dataset = match target {
        QueueTarget::Uri(uri) => {
            Dataset::write(reader, uri.as_str(), Some(queue_write_params())).await?
        }
        QueueTarget::Namespace {
            namespace_client,
            table_id,
        } => {
            Dataset::write_into_namespace(
                reader,
                namespace_client,
                table_id,
                Some(queue_write_params()),
            )
            .await?
        }
    };

    let lance_schema = dataset.schema();
    let shard_spec = queue_shard_spec(config.partition_count, config.producer_count, lance_schema)?;
    let mut initial_shards = Vec::with_capacity(num_shards as usize);
    for partition_id in 0..config.partition_count {
        for producer_id in 0..config.producer_count {
            initial_shards.push(
                MemWalShardSnapshot::new(Uuid::new_v4(), QUEUE_SHARD_SPEC_ID)
                    .with_shard_field_value(QUEUE_PARTITION_FIELD_ID, partition_id as i32)
                    .with_shard_field_value(QUEUE_PRODUCER_FIELD_ID, producer_id as i32),
            );
        }
    }
    dataset
        .initialize_mem_wal_with_shards(
            MemWalConfig {
                shard_spec: Some(shard_spec),
                maintained_indexes: Vec::new(),
            },
            MemWalShardConfig {
                num_shards,
                initial_shards,
            },
        )
        .await?;

    Queue::from_dataset(dataset).await
}

fn queue_write_params() -> WriteParams {
    WriteParams {
        auto_cleanup: None,
        skip_auto_cleanup: true,
        ..Default::default()
    }
}

fn queue_schema() -> Arc<ArrowSchema> {
    let id_metadata =
        HashMap::from([(LANCE_UNENFORCED_PRIMARY_KEY.to_string(), "true".to_string())]);
    Arc::new(ArrowSchema::new(vec![
        Field::new(QUEUE_ID_COLUMN, DataType::Utf8, false).with_metadata(id_metadata),
        Field::new(QUEUE_PRODUCER_ID_COLUMN, DataType::UInt32, false),
        json_field(QUEUE_PAYLOAD_COLUMN, false),
    ]))
}

fn validate_queue_schema(schema: &LanceSchema) -> Result<()> {
    let arrow_schema = ArrowSchema::from(schema);
    if arrow_schema.fields() != queue_schema().fields() {
        return Err(Error::invalid_input(format!(
            "queue table schema must be fixed id/producer_id/payload schema: expected fields {:?}, got fields {:?}",
            queue_schema().fields(),
            arrow_schema.fields()
        )));
    }
    let primary_key_columns = primary_key_columns(schema)?;
    if primary_key_columns != [QUEUE_ID_COLUMN.to_string()] {
        return Err(Error::invalid_input(format!(
            "queue table must use '{}' as its only unenforced primary key column, got {:?}",
            QUEUE_ID_COLUMN, primary_key_columns
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

fn queue_shard_spec(
    partition_count: u32,
    producer_count: u32,
    schema: &LanceSchema,
) -> Result<ShardSpec> {
    let primary_key_fields = schema.unenforced_primary_key();
    let source_ids = primary_key_fields
        .iter()
        .map(|field| field.id)
        .collect::<Vec<_>>();
    if source_ids.is_empty() {
        return Err(Error::invalid_input(
            "queues require an unenforced primary key in the schema",
        ));
    }

    let mut parameters = HashMap::new();
    parameters.insert("num_buckets".to_string(), partition_count.to_string());
    let mut producer_parameters = HashMap::new();
    producer_parameters.insert(
        "source_column".to_string(),
        QUEUE_PRODUCER_ID_COLUMN.to_string(),
    );
    producer_parameters.insert("producer_count".to_string(), producer_count.to_string());

    Ok(ShardSpec {
        spec_id: QUEUE_SHARD_SPEC_ID,
        fields: vec![
            ShardField {
                field_id: QUEUE_PARTITION_FIELD_ID.to_string(),
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
                field_id: QUEUE_PRODUCER_FIELD_ID.to_string(),
                source_ids: vec![schema.field_id(QUEUE_PRODUCER_ID_COLUMN)?],
                transform: Some("identity".to_string()),
                expression: None,
                result_type: "int32".to_string(),
                parameters: producer_parameters,
            },
        ],
    })
}

async fn queue_partitions(
    dataset: &Dataset,
    details: &MemWalIndexDetails,
) -> Result<Vec<QueuePartition>> {
    if details.num_shards == 0 {
        return Err(Error::invalid_input(
            "queue MemWAL index has no shards configured",
        ));
    }

    let shard_spec = queue_shard_spec_from_details(details)?;
    let partition_count = shard_spec.fields[0]
        .parameters
        .get("num_buckets")
        .ok_or_else(|| Error::invalid_input("queue MemWAL shard spec is missing num_buckets"))?
        .parse::<u32>()
        .map_err(|e| {
            Error::invalid_input(format!(
                "queue MemWAL shard spec has invalid num_buckets: {}",
                e
            ))
        })?;
    let producer_count = shard_spec.fields[1]
        .parameters
        .get("producer_count")
        .ok_or_else(|| Error::invalid_input("queue MemWAL shard spec is missing producer_count"))?
        .parse::<u32>()
        .map_err(|e| {
            Error::invalid_input(format!(
                "queue MemWAL shard spec has invalid producer_count: {}",
                e
            ))
        })?;
    let snapshots = dataset.mem_wal_shard_snapshots().await?;
    if snapshots.len() != details.num_shards as usize {
        return Err(Error::invalid_input(format!(
            "queue MemWAL shard snapshot count ({}) does not match num_shards ({})",
            snapshots.len(),
            details.num_shards
        )));
    }

    let mut partitions = Vec::with_capacity(snapshots.len());
    let mut seen = HashSet::with_capacity(snapshots.len());
    for snapshot in snapshots {
        if snapshot.shard_spec_id != shard_spec.spec_id {
            continue;
        }
        let partition_bucket = snapshot
            .shard_field_values
            .get(QUEUE_PARTITION_FIELD_ID)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "queue MemWAL shard snapshot for shard {} is missing '{}' field",
                    snapshot.shard_id, QUEUE_PARTITION_FIELD_ID
                ))
            })?;
        let producer_bucket = snapshot
            .shard_field_values
            .get(QUEUE_PRODUCER_FIELD_ID)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "queue MemWAL shard snapshot for shard {} is missing '{}' field",
                    snapshot.shard_id, QUEUE_PRODUCER_FIELD_ID
                ))
            })?;
        if *partition_bucket < 0 || *partition_bucket as u32 >= partition_count {
            return Err(Error::invalid_input(format!(
                "queue MemWAL shard snapshot partition bucket {} is outside [0, {})",
                partition_bucket, partition_count
            )));
        }
        if *producer_bucket < 0 {
            return Err(Error::invalid_input(format!(
                "queue MemWAL shard snapshot producer bucket {} is negative",
                producer_bucket
            )));
        }
        if *producer_bucket as u32 >= producer_count {
            return Err(Error::invalid_input(format!(
                "queue MemWAL shard snapshot producer bucket {} is outside [0, {})",
                producer_bucket, producer_count
            )));
        }
        let partition_id = *partition_bucket as u32;
        let producer_id = *producer_bucket as u32;
        if !seen.insert((partition_id, producer_id)) {
            return Err(Error::invalid_input(format!(
                "queue MemWAL shard snapshots contain duplicate partition_id {} producer_id {}",
                partition_id, producer_id
            )));
        }
        partitions.push(QueuePartition {
            partition_id,
            producer_id,
            shard_id: snapshot.shard_id,
            shard_spec_id: snapshot.shard_spec_id,
        });
    }

    partitions.sort_by_key(|partition| (partition.partition_id, partition.producer_id));
    if partitions.len() != details.num_shards as usize {
        return Err(Error::invalid_input(format!(
            "queue MemWAL shard snapshots cover {} physical producer shards but num_shards is {}",
            partitions.len(),
            details.num_shards
        )));
    }

    if partition_count
        .checked_mul(producer_count)
        .filter(|expected| *expected == details.num_shards)
        .is_none()
    {
        return Err(Error::invalid_input(format!(
            "queue MemWAL num_shards {} is not divisible by logical partition_count {}",
            details.num_shards, partition_count
        )));
    }
    for partition_id in 0..partition_count {
        for producer_id in 0..producer_count {
            if !seen.contains(&(partition_id, producer_id)) {
                return Err(Error::invalid_input(format!(
                    "queue MemWAL shard snapshots are missing partition_id {} producer_id {}",
                    partition_id, producer_id
                )));
            }
        }
    }
    Ok(partitions)
}

fn queue_shard_spec_from_details(details: &MemWalIndexDetails) -> Result<&ShardSpec> {
    let matching = details
        .shard_specs
        .iter()
        .filter(|spec| {
            spec.fields.len() == 2
                && spec.fields[0].field_id == QUEUE_PARTITION_FIELD_ID
                && spec.fields[0].result_type == "int32"
                && spec.fields[1].field_id == QUEUE_PRODUCER_FIELD_ID
                && spec.fields[1].result_type == "int32"
        })
        .collect::<Vec<_>>();
    if matching.len() != 1 {
        return Err(Error::invalid_input(format!(
            "queue MemWAL index must contain exactly one partition+producer shard spec, found {}",
            matching.len()
        )));
    }

    let spec = matching[0];
    let partition_field = &spec.fields[0];
    match partition_field.transform.as_deref() {
        Some("bucket") | Some("multi_bucket") => {}
        other => {
            return Err(Error::invalid_input(format!(
                "queue MemWAL partition shard field must use bucket or multi_bucket transform, got {:?}",
                other
            )));
        }
    }
    let producer_field = &spec.fields[1];
    if producer_field.transform.as_deref() != Some("identity") {
        return Err(Error::invalid_input(format!(
            "queue MemWAL producer shard field must use identity transform, got {:?}",
            producer_field.transform
        )));
    }
    Ok(spec)
}

/// Queue producer.
#[derive(Debug, Clone)]
pub struct Producer {
    queue: Queue,
    producer_id: u32,
    appenders: Arc<Vec<WalAppender>>,
}

impl Producer {
    fn new(queue: Queue, producer_id: u32) -> Result<Self> {
        if producer_id >= queue.producer_count() {
            return Err(Error::invalid_input(format!(
                "producer_id {} is out of range for queue with {} producer slots",
                producer_id,
                queue.producer_count()
            )));
        }

        let mut appenders = Vec::with_capacity(queue.partition_count() as usize);
        for partition_id in 0..queue.partition_count() {
            let partition = queue.partition(partition_id, producer_id)?;
            appenders.push(WalAppender::new(
                queue.object_store.clone(),
                queue.base_path.clone(),
                partition.partition_id,
                partition.producer_id,
                partition.shard_id,
                partition.shard_spec_id,
            ));
        }

        Ok(Self {
            queue,
            producer_id,
            appenders: Arc::new(appenders),
        })
    }

    /// Producer shard slot id used by this producer.
    pub fn producer_id(&self) -> u32 {
        self.producer_id
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
        self.send_record_batch(message_batch(ids, payloads, self.producer_id)?)
            .await
    }

    async fn send_record_batch(&self, batch: RecordBatch) -> Result<ProduceResult> {
        if batch.num_rows() == 0 {
            return Err(Error::invalid_input("cannot send an empty record batch"));
        }
        self.queue.validate_batch_schema(&batch)?;

        let partitioner = Partitioner::new(
            self.queue.partition_count(),
            self.queue.primary_key_columns().to_vec(),
        )?;
        let partitioned_batches = partitioner.partition_batch(&batch)?;

        let mut produce_futures = Vec::with_capacity(partitioned_batches.len());
        for (partition_id, partition_batch) in partitioned_batches {
            let appender = self.appender(partition_id)?;
            produce_futures.push(async move { appender.append(vec![partition_batch]).await });
        }
        let entries = try_join_all(produce_futures).await?;

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
        for batch in &batches {
            self.queue.validate_batch_schema(batch)?;
        }
        let num_rows = batches.iter().map(RecordBatch::num_rows).sum();
        let appender = self.appender(partition_id)?;
        let entry = appender.append(batches).await?;

        Ok(ProduceResult {
            num_rows,
            entries: vec![entry],
        })
    }

    fn appender(&self, partition_id: u32) -> Result<&WalAppender> {
        self.queue.partition(partition_id, self.producer_id)?;
        self.appenders.get(partition_id as usize).ok_or_else(|| {
            Error::invalid_input(format!(
                "partition_id {} is out of range for queue with {} partitions",
                partition_id,
                self.queue.partition_count()
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

/// A decoded queue message.
#[derive(Debug, Clone, PartialEq)]
pub struct QueueMessage {
    /// Message id used as the queue partition key.
    pub id: String,
    /// JSON payload.
    pub payload: Value,
}

/// Consumer configuration.
#[derive(Debug, Clone)]
pub struct ConsumerConfig {
    group_id: String,
    assignment: ConsumerAssignment,
    start_position: StartPosition,
}

#[derive(Debug, Clone)]
enum ConsumerAssignment {
    All,
    ConsumerPartition {
        partition_count: u32,
        partition_id: u32,
    },
    ManualPartitions(Vec<u32>),
}

impl ConsumerConfig {
    /// Create a consumer configuration for a consumer group.
    pub fn new(group_id: impl Into<String>) -> Self {
        Self {
            group_id: group_id.into(),
            assignment: ConsumerAssignment::All,
            start_position: StartPosition::Earliest,
        }
    }

    /// Assign the consumer to one deterministic consumer partition.
    ///
    /// Queue partitions are assigned to consumer partitions using stable
    /// rendezvous hashing over `(queue_partition_id, partition_id)`.
    pub fn with_consumer_partition(mut self, partition_count: u32, partition_id: u32) -> Self {
        self.assignment = ConsumerAssignment::ConsumerPartition {
            partition_count,
            partition_id,
        };
        self
    }

    /// Manually assign the consumer to a subset of queue partitions.
    pub fn with_partitions<I>(mut self, partitions: I) -> Self
    where
        I: IntoIterator<Item = u32>,
    {
        self.assignment = ConsumerAssignment::ManualPartitions(partitions.into_iter().collect());
        self
    }

    /// Set the starting position for partitions with no committed offset.
    pub fn with_start_position(mut self, start_position: StartPosition) -> Self {
        self.start_position = start_position;
        self
    }
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

/// A batch of records read from a queue partition.
#[derive(Debug, Clone)]
pub struct QueueBatch {
    /// Partition this batch came from.
    pub partition_id: u32,
    /// Producer shard this batch came from.
    pub producer_id: u32,
    /// WAL entry position.
    pub entry_position: u64,
    /// Next offset to commit after processing this batch.
    pub next_entry_position: u64,
    /// Arrow batches stored in the WAL entry.
    pub batches: Vec<RecordBatch>,
}

impl QueueBatch {
    fn from_entry(entry: QueueEntry) -> Result<Self> {
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

    /// Number of rows in this queue batch.
    pub fn num_rows(&self) -> usize {
        self.batches.iter().map(RecordBatch::num_rows).sum()
    }

    /// Decode this queue batch into id/payload messages.
    pub fn messages(&self) -> Result<Vec<QueueMessage>> {
        let mut messages = Vec::with_capacity(self.num_rows());
        for batch in &self.batches {
            messages.extend(record_batch_to_messages(batch)?);
        }
        Ok(messages)
    }
}

/// Queue consumer.
#[derive(Debug)]
pub struct Consumer {
    queue: Queue,
    group_id: String,
    assigned_partitions: Vec<u32>,
    assigned_shards: Vec<(u32, u32)>,
    next_entry_positions: HashMap<(u32, u32), u64>,
}

impl Consumer {
    async fn open(queue: Queue, config: ConsumerConfig) -> Result<Self> {
        metadata::validate_group_id(&config.group_id)?;

        let assigned_partitions = assigned_consumer_partitions(&queue, config.assignment)?;

        let mut metadata_dataset = queue.dataset.as_ref().clone();
        metadata_dataset.checkout_latest().await?;
        let committed_positions: HashMap<(u32, u32), u64> =
            metadata::read_all_group_offsets(metadata_dataset.metadata(), &config.group_id)?
                .into_iter()
                .map(|offset| {
                    (
                        (offset.partition_id, offset.producer_id),
                        offset.next_entry_position,
                    )
                })
                .collect();

        let assigned_shards = assigned_partitions
            .iter()
            .flat_map(|partition_id| {
                (0..queue.producer_count()).map(|producer_id| (*partition_id, producer_id))
            })
            .collect::<Vec<_>>();
        let mut next_entry_positions = HashMap::with_capacity(assigned_shards.len());
        for (partition_id, producer_id) in &assigned_shards {
            let key = (*partition_id, *producer_id);
            let position = if let Some(position) = committed_positions.get(&key) {
                *position
            } else {
                match config.start_position {
                    StartPosition::Earliest => {
                        queue
                            .wal_tailer(*partition_id, *producer_id)?
                            .first_position()
                            .await?
                    }
                    StartPosition::Latest => {
                        queue
                            .wal_tailer(*partition_id, *producer_id)?
                            .next_position()
                            .await?
                    }
                }
            };

            next_entry_positions.insert(key, position);
        }

        Ok(Self {
            queue,
            group_id: config.group_id,
            assigned_partitions,
            assigned_shards,
            next_entry_positions,
        })
    }

    /// Queue partition ids assigned to this consumer.
    pub fn assigned_partitions(&self) -> &[u32] {
        &self.assigned_partitions
    }

    /// Poll at most one WAL entry from each assigned physical producer shard.
    ///
    /// If reading any assigned partition fails, the entire poll fails and the
    /// consumer keeps its previous in-memory offsets.
    pub async fn poll(&mut self) -> Result<Vec<QueueBatch>> {
        self.poll_with_options(PollOptions::default()).await
    }

    /// Poll queue data with explicit options.
    ///
    /// If reading any assigned partition fails, the entire poll fails and the
    /// consumer keeps its previous in-memory offsets.
    pub async fn poll_with_options(&mut self, options: PollOptions) -> Result<Vec<QueueBatch>> {
        if options.max_entries_per_partition == 0 {
            return Err(Error::invalid_input(
                "max_entries_per_partition must be greater than 0",
            ));
        }

        let mut out = Vec::new();
        let mut next_entry_positions = self.next_entry_positions.clone();
        for (partition_id, producer_id) in &self.assigned_shards {
            let tailer = self.queue.wal_tailer(*partition_id, *producer_id)?;
            for _ in 0..options.max_entries_per_partition {
                let key = (*partition_id, *producer_id);
                let position = *next_entry_positions.get(&key).ok_or_else(|| {
                    Error::internal(format!(
                        "missing next entry position for assigned partition_id {} producer_id {}",
                        partition_id, producer_id
                    ))
                })?;

                let Some(entry) = tailer.read_entry(position).await? else {
                    break;
                };

                let batch = QueueBatch::from_entry(entry)?;
                next_entry_positions.insert(key, batch.next_entry_position);
                out.push(batch);
            }
        }

        self.next_entry_positions = next_entry_positions;
        Ok(out)
    }

    /// Commit offsets for processed queue batches.
    pub async fn commit(&self, batches: &[QueueBatch]) -> Result<()> {
        let mut latest = HashMap::<(u32, u32), u64>::new();
        for batch in batches {
            if !self.assigned_partitions.contains(&batch.partition_id) {
                return Err(Error::invalid_input(format!(
                    "cannot commit offset for unassigned partition_id {}",
                    batch.partition_id
                )));
            }
            self.queue
                .partition(batch.partition_id, batch.producer_id)?;
            latest
                .entry((batch.partition_id, batch.producer_id))
                .and_modify(|position| *position = (*position).max(batch.next_entry_position))
                .or_insert(batch.next_entry_position);
        }

        let offsets = latest
            .into_iter()
            .map(
                |((partition_id, producer_id), next_entry_position)| ConsumerGroupOffset {
                    partition_id,
                    producer_id,
                    next_entry_position,
                },
            )
            .collect::<Vec<_>>();
        metadata::write_group_offsets(self.queue.dataset.as_ref(), &self.group_id, &offsets)
            .await?;

        Ok(())
    }

    /// Commit the consumer's current in-memory offsets.
    pub async fn commit_current(&self) -> Result<()> {
        let mut offsets = Vec::with_capacity(self.assigned_shards.len());
        for (partition_id, producer_id) in &self.assigned_shards {
            let key = (*partition_id, *producer_id);
            let next_entry_position = *self.next_entry_positions.get(&key).ok_or_else(|| {
                Error::internal(format!(
                    "missing next entry position for assigned partition_id {} producer_id {}",
                    partition_id, producer_id
                ))
            })?;
            offsets.push(ConsumerGroupOffset {
                partition_id: *partition_id,
                producer_id: *producer_id,
                next_entry_position,
            });
        }
        metadata::write_group_offsets(self.queue.dataset.as_ref(), &self.group_id, &offsets)
            .await?;
        Ok(())
    }
}

fn assigned_consumer_partitions(queue: &Queue, assignment: ConsumerAssignment) -> Result<Vec<u32>> {
    match assignment {
        ConsumerAssignment::All => Ok((0..queue.partition_count()).collect()),
        ConsumerAssignment::ConsumerPartition {
            partition_count,
            partition_id,
        } => assigned_hashed_partitions(queue, partition_count, partition_id),
        ConsumerAssignment::ManualPartitions(partitions) => {
            validate_manual_partitions(queue, partitions)
        }
    }
}

fn assigned_hashed_partitions(
    queue: &Queue,
    partition_count: u32,
    partition_id: u32,
) -> Result<Vec<u32>> {
    if partition_count == 0 {
        return Err(Error::invalid_input(
            "consumer partition_count must be greater than 0",
        ));
    }
    if partition_id >= partition_count {
        return Err(Error::invalid_input(format!(
            "consumer partition_id {} must be less than partition_count {}",
            partition_id, partition_count
        )));
    }

    (0..queue.partition_count())
        .filter_map(|queue_partition_id| {
            match consumer_slot_for_partition(queue_partition_id, partition_count) {
                Ok(slot_id) if slot_id == partition_id => Some(Ok(queue_partition_id)),
                Ok(_) => None,
                Err(error) => Some(Err(error)),
            }
        })
        .collect()
}

fn validate_manual_partitions(queue: &Queue, partitions: Vec<u32>) -> Result<Vec<u32>> {
    let mut seen = HashSet::with_capacity(partitions.len());
    for partition_id in &partitions {
        validate_logical_partition(queue, *partition_id)?;
        if !seen.insert(*partition_id) {
            return Err(Error::invalid_input(format!(
                "partition_id {} is assigned more than once",
                partition_id
            )));
        }
    }
    Ok(partitions)
}

fn validate_logical_partition(queue: &Queue, partition_id: u32) -> Result<()> {
    if partition_id >= queue.partition_count() {
        return Err(Error::invalid_input(format!(
            "partition_id {} is out of range for queue with {} logical partitions",
            partition_id,
            queue.partition_count()
        )));
    }
    Ok(())
}

fn message_batch(ids: Vec<String>, payloads: Vec<Value>, producer_id: u32) -> Result<RecordBatch> {
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
                Error::invalid_input(format!("failed to encode queue payload as JSON: {}", e))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let payload = JsonArray::try_from_iter(payload_strings.iter().map(Some))
        .map_err(|e| {
            Error::invalid_input(format!("failed to encode queue payload as JSONB: {}", e))
        })?
        .into_inner();

    RecordBatch::try_new(
        queue_schema(),
        vec![
            Arc::new(StringArray::from(ids)) as ArrayRef,
            Arc::new(UInt32Array::from_value(producer_id, payload_strings.len())) as ArrayRef,
            Arc::new(payload) as ArrayRef,
        ],
    )
    .map_err(|e| Error::arrow(format!("failed to create queue message batch: {}", e)))
}

fn record_batch_to_messages(batch: &RecordBatch) -> Result<Vec<QueueMessage>> {
    let id_idx = batch
        .schema()
        .index_of(QUEUE_ID_COLUMN)
        .map_err(|e| Error::invalid_input(format!("queue batch is missing id column: {}", e)))?;
    let payload_idx = batch.schema().index_of(QUEUE_PAYLOAD_COLUMN).map_err(|e| {
        Error::invalid_input(format!("queue batch is missing payload column: {}", e))
    })?;

    let ids = batch
        .column(id_idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "queue id column must be Utf8, got {}",
                batch.column(id_idx).data_type()
            ))
        })?;
    let payloads = batch
        .column(payload_idx)
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "queue payload column must be LargeBinary JSONB, got {}",
                batch.column(payload_idx).data_type()
            ))
        })?;

    let mut messages = Vec::with_capacity(batch.num_rows());
    for row_idx in 0..batch.num_rows() {
        if ids.is_null(row_idx) || payloads.is_null(row_idx) {
            return Err(Error::invalid_input(format!(
                "queue message at row {} contains null id or payload",
                row_idx
            )));
        }
        let payload_json = decode_json(payloads.value(row_idx));
        let payload = serde_json::from_str(&payload_json).map_err(|e| {
            Error::invalid_input(format!(
                "failed to decode queue payload JSON at row {}: {}",
                row_idx, e
            ))
        })?;
        messages.push(QueueMessage {
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
    use lance_namespace::models::{
        DeclareTableRequest, DeclareTableResponse, DescribeTableRequest, DescribeTableResponse,
    };
    use serde_json::{Value, json};
    use tempfile::TempDir;

    use super::*;

    #[derive(Debug)]
    struct StaticNamespace {
        uri: String,
    }

    #[async_trait::async_trait]
    impl LanceNamespace for StaticNamespace {
        async fn declare_table(
            &self,
            _request: DeclareTableRequest,
        ) -> Result<DeclareTableResponse> {
            Ok(DeclareTableResponse {
                location: Some(self.uri.clone()),
                ..Default::default()
            })
        }

        async fn describe_table(
            &self,
            _request: DescribeTableRequest,
        ) -> Result<DescribeTableResponse> {
            Ok(DescribeTableResponse {
                location: Some(self.uri.clone()),
                table_uri: Some(self.uri.clone()),
                ..Default::default()
            })
        }

        fn namespace_id(&self) -> String {
            format!("static:{}", self.uri)
        }
    }

    fn queue_uri(temp_dir: &TempDir) -> String {
        format!("file://{}", temp_dir.path().display())
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
        queue: &Queue,
        producer_id: u32,
        partition_id: u32,
        values: Vec<i32>,
    ) -> ProduceResult {
        let batch = batch_for_partition(queue, producer_id, partition_id, values);
        queue
            .producer(producer_id)
            .unwrap()
            .send_to_partition(partition_id, vec![batch])
            .await
            .unwrap()
    }

    fn batch_for_partition(
        queue: &Queue,
        producer_id: u32,
        partition_id: u32,
        values: Vec<i32>,
    ) -> RecordBatch {
        let mut ids = Vec::with_capacity(values.len());
        let mut payloads = Vec::with_capacity(values.len());
        for value in values {
            ids.push(id_for_partition(queue, producer_id, partition_id, value));
            payloads.push(json!({ "value": value }));
        }
        message_batch(ids, payloads, producer_id).unwrap()
    }

    fn id_for_partition(queue: &Queue, producer_id: u32, partition_id: u32, value: i32) -> String {
        let partitioner = Partitioner::new(
            queue.partition_count(),
            queue.primary_key_columns().to_vec(),
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

    fn count_rows(batches: &[QueueBatch]) -> usize {
        batches.iter().map(QueueBatch::num_rows).sum()
    }

    fn test_wal_entry_filename(entry_position: u64) -> String {
        format!("{:064b}.arrow", entry_position.reverse_bits())
    }

    #[tokio::test]
    async fn test_wal_appender_and_tailer_round_trip() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = queue_uri(&temp_dir);
        let (store, base_path) = ObjectStore::from_uri(&uri).await.unwrap();
        let shard_id = Uuid::new_v4();

        let appender = WalAppender::new(store.clone(), base_path.clone(), 0, 7, shard_id, 1);
        let first = appender
            .append(vec![
                message_batch(
                    vec!["1".to_string(), "2".to_string()],
                    vec![json!({ "value": 1 }), json!({ "value": 2 })],
                    7,
                )
                .unwrap(),
            ])
            .await
            .unwrap();
        let second = appender
            .append(vec![
                message_batch(vec!["3".to_string()], vec![json!({ "value": 3 })], 7).unwrap(),
            ])
            .await
            .unwrap();

        assert_eq!(first.entry_position, 1);
        assert_eq!(first.producer_id, 7);
        assert_eq!(first.num_rows, 2);
        assert_eq!(second.entry_position, 2);

        let tailer = WalTailer::new(store, base_path, 0, 7, shard_id);
        let first_read = tailer.read_entry(1).await.unwrap().unwrap();
        let second_read = tailer.read_entry(2).await.unwrap().unwrap();
        let missing = tailer.read_entry(3).await.unwrap();

        assert_eq!(first_read.producer_id, 7);
        assert_eq!(first_read.batches.len(), 1);
        assert_eq!(first_read.batches[0].num_rows(), 2);
        assert_eq!(second_read.batches[0].num_rows(), 1);
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn test_producer_hash_partitions_by_primary_key() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = queue_uri(&temp_dir);
        let queue = Queue::builder()
            .uri(&uri)
            .partition_count(2)
            .create()
            .await
            .unwrap();

        assert_eq!(queue.primary_key_columns(), &["id".to_string()]);

        let (ids, payloads) = message_values(0, 20);
        let result = queue
            .producer(0)
            .unwrap()
            .send_batch(ids, payloads)
            .await
            .unwrap();

        assert_eq!(result.num_rows, 20);
        assert_eq!(result.entries.len(), 2);
        assert!(result.entries.iter().all(|entry| entry.num_rows > 0));
        assert!(result.entries.iter().all(|entry| entry.producer_id == 0));
    }

    #[tokio::test]
    async fn test_producer_send_and_consumer_messages() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = queue_uri(&temp_dir);
        let queue = Queue::builder().uri(&uri).create().await.unwrap();

        queue
            .producer(0)
            .unwrap()
            .send("message-1", json!({ "kind": "created", "version": 1 }))
            .await
            .unwrap();

        let mut consumer = queue
            .consumer(ConsumerConfig::new("message-group"))
            .await
            .unwrap();
        let polled = consumer.poll().await.unwrap();
        let messages = polled[0].messages().unwrap();
        assert_eq!(
            messages,
            vec![QueueMessage {
                id: "message-1".to_string(),
                payload: json!({ "kind": "created", "version": 1 }),
            }]
        );
    }

    #[tokio::test]
    async fn test_producer_rejects_empty_or_mismatched_batches() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = queue_uri(&temp_dir);
        let queue = Queue::builder().uri(&uri).create().await.unwrap();
        let producer = queue.producer(0).unwrap();

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
    async fn test_producer_count_defines_physical_shards() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = queue_uri(&temp_dir);
        let queue = Queue::builder()
            .uri(&uri)
            .partition_count(1)
            .producer_count(2)
            .create()
            .await
            .unwrap();

        assert_eq!(queue.partition_count(), 1);
        assert_eq!(queue.producer_count(), 2);
        assert!(queue.producer(2).is_err());
        queue
            .producer(0)
            .unwrap()
            .send("same-key", json!({ "producer": 0 }))
            .await
            .unwrap();
        queue
            .producer(1)
            .unwrap()
            .send("same-key", json!({ "producer": 1 }))
            .await
            .unwrap();

        let mut consumer = queue
            .consumer(ConsumerConfig::new("multi-producer-group"))
            .await
            .unwrap();
        let polled = consumer.poll().await.unwrap();
        let producer_ids = polled
            .iter()
            .map(|batch| batch.producer_id)
            .collect::<std::collections::HashSet<_>>();

        assert_eq!(count_rows(&polled), 2);
        assert_eq!(producer_ids, std::collections::HashSet::from([0, 1]));
        consumer.commit_current().await.unwrap();

        let mut metadata_dataset = queue.dataset().clone();
        metadata_dataset.checkout_latest().await.unwrap();
        assert_eq!(
            metadata_dataset
                .metadata()
                .get("lance_queue.group.multi-producer-group.commits.0.0.next_entry_position"),
            Some(&"2".to_string())
        );
        assert_eq!(
            metadata_dataset
                .metadata()
                .get("lance_queue.group.multi-producer-group.commits.0.1.next_entry_position"),
            Some(&"2".to_string())
        );
    }

    #[tokio::test]
    async fn test_queue_schema_is_persisted_and_enforced() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = queue_uri(&temp_dir);
        let queue = Queue::builder()
            .uri(&uri)
            .partition_count(2)
            .create()
            .await
            .unwrap();

        let reopened = Queue::builder().uri(&uri).open().await.unwrap();
        assert_eq!(queue.schema().fields(), reopened.schema().fields());

        let err = reopened
            .producer(0)
            .unwrap()
            .send_to_partition(0, vec![mismatched_batch(vec![1])])
            .await
            .unwrap_err();
        assert!(err.to_string().contains("schema does not match"), "{}", err);
    }

    #[tokio::test]
    async fn test_open_rejects_table_without_mem_wal_index() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = queue_uri(&temp_dir);
        let schema = queue_schema();
        let reader = RecordBatchIterator::new(
            vec![Ok(RecordBatch::new_empty(schema.clone()))].into_iter(),
            schema,
        );
        Dataset::write(reader, &uri, Some(queue_write_params()))
            .await
            .unwrap();

        let err = Queue::builder().uri(&uri).open().await.unwrap_err();
        assert!(err.to_string().contains("missing MemWAL index"), "{}", err);
    }

    #[tokio::test]
    async fn test_builder_can_create_and_open_namespace_queue() {
        let temp_dir = tempfile::tempdir().unwrap();
        let table_uri = format!("file://{}/table.lance", temp_dir.path().display());
        let namespace = Arc::new(StaticNamespace { uri: table_uri });
        let table_id = vec!["workspace".to_string(), "queue".to_string()];

        let queue = Queue::builder()
            .namespace(namespace.clone(), table_id.clone())
            .partition_count(2)
            .create()
            .await
            .unwrap();
        assert_eq!(queue.partition_count(), 2);

        let reopened = Queue::builder()
            .namespace(namespace, table_id)
            .open()
            .await
            .unwrap();
        assert_eq!(reopened.partition_count(), 2);
    }

    #[tokio::test]
    async fn test_queue_creates_real_mem_wal_index_for_primary_key() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = queue_uri(&temp_dir);
        let queue = Queue::builder()
            .uri(&uri)
            .partition_count(4)
            .producer_count(3)
            .create()
            .await
            .unwrap();

        let details = queue.mem_wal_index_details();
        assert_eq!(details.num_shards, 12);
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
                .get("producer_count"),
            Some(&"3".to_string())
        );
        assert_eq!(queue.partitions()[0].shard_spec_id, 1);
        assert_eq!(queue.partitions()[0].partition_id, 0);
        assert_eq!(queue.partitions()[0].producer_id, 0);
        assert_eq!(queue.partitions().last().unwrap().partition_id, 3);
        assert_eq!(queue.partitions().last().unwrap().producer_id, 2);

        let mem_wal_index = queue
            .dataset()
            .load_index_by_name(MEM_WAL_INDEX_NAME)
            .await
            .unwrap();
        assert!(mem_wal_index.is_some());
        assert!(!temp_dir.path().join("_lance_queue").exists());
    }

    #[tokio::test]
    async fn test_queue_uses_fixed_json_payload_schema() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = queue_uri(&temp_dir);
        let queue = Queue::builder().uri(&uri).create().await.unwrap();
        let schema = queue.schema();

        let id_field = schema.field_with_name(QUEUE_ID_COLUMN).unwrap();
        assert_eq!(id_field.data_type(), &DataType::Utf8);
        assert_eq!(
            id_field.metadata().get(LANCE_UNENFORCED_PRIMARY_KEY),
            Some(&"true".to_string())
        );
        let producer_id_field = schema.field_with_name(QUEUE_PRODUCER_ID_COLUMN).unwrap();
        assert_eq!(producer_id_field.data_type(), &DataType::UInt32);
        assert!(is_json_field(
            schema.field_with_name(QUEUE_PAYLOAD_COLUMN).unwrap()
        ));
        assert!(
            !queue
                .dataset()
                .config()
                .contains_key("lance.auto_cleanup.interval")
        );
        assert!(
            !queue
                .dataset()
                .config()
                .contains_key("lance.auto_cleanup.older_than")
        );
    }

    #[tokio::test]
    async fn test_send_to_partition_latest_and_commit_current() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = queue_uri(&temp_dir);
        let queue = Queue::builder()
            .uri(&uri)
            .partition_count(2)
            .create()
            .await
            .unwrap();

        let first = produce_to_partition(&queue, 0, 1, vec![1, 2]).await;
        let first = first.entries.first().unwrap();
        assert_eq!(first.partition_id, 1);
        assert_eq!(first.producer_id, 0);

        let mut latest = queue
            .consumer(
                ConsumerConfig::new("latest-group")
                    .with_partitions([1])
                    .with_start_position(StartPosition::Latest),
            )
            .await
            .unwrap();
        assert!(latest.poll().await.unwrap().is_empty());

        produce_to_partition(&queue, 0, 1, vec![3]).await;
        let polled = latest.poll().await.unwrap();
        assert_eq!(count_rows(&polled), 1);
        latest.commit_current().await.unwrap();

        let mut resumed = queue
            .consumer(ConsumerConfig::new("latest-group").with_partitions([1]))
            .await
            .unwrap();
        assert!(resumed.poll().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_consumed_batches_can_be_reenqueued() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = queue_uri(&temp_dir);
        let queue = Queue::builder().uri(&uri).create().await.unwrap();
        produce_to_partition(&queue, 0, 0, vec![1]).await;

        let mut consumer = queue
            .consumer(ConsumerConfig::new("requeue-group"))
            .await
            .unwrap();
        let polled = consumer.poll().await.unwrap();
        assert_eq!(polled.len(), 1);

        queue
            .producer(0)
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
        let uri = queue_uri(&temp_dir);
        let queue = Queue::builder()
            .uri(&uri)
            .partition_count(2)
            .create()
            .await
            .unwrap();

        let (ids, payloads) = message_values(0, 20);
        queue
            .producer(0)
            .unwrap()
            .send_batch(ids, payloads)
            .await
            .unwrap();

        let mut consumer = queue
            .consumer(ConsumerConfig::new("group-a"))
            .await
            .unwrap();
        let first_poll = consumer.poll().await.unwrap();
        assert_eq!(count_rows(&first_poll), 20);
        consumer.commit(&first_poll).await.unwrap();

        let mut metadata_dataset = queue.dataset().clone();
        metadata_dataset.checkout_latest().await.unwrap();
        assert_eq!(
            metadata_dataset
                .metadata()
                .get("lance_queue.group.group-a.commits.0.0.next_entry_position"),
            Some(&"2".to_string())
        );
        assert_eq!(
            metadata_dataset
                .metadata()
                .get("lance_queue.group.group-a.commits.1.0.next_entry_position"),
            Some(&"2".to_string())
        );
        assert!(!temp_dir.path().join("_lance_queue").exists());

        let mut resumed = queue
            .consumer(ConsumerConfig::new("group-a"))
            .await
            .unwrap();
        assert!(resumed.poll().await.unwrap().is_empty());

        let (ids, payloads) = message_values(20, 30);
        queue
            .producer(0)
            .unwrap()
            .send_batch(ids, payloads)
            .await
            .unwrap();
        let second_poll = resumed.poll().await.unwrap();
        assert_eq!(count_rows(&second_poll), 10);
    }

    #[tokio::test]
    async fn test_consumer_partition_assignment_uses_stable_hashing() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = queue_uri(&temp_dir);
        let queue = Queue::builder()
            .uri(&uri)
            .partition_count(8)
            .create()
            .await
            .unwrap();
        for partition_id in 0..queue.partition_count() {
            produce_to_partition(&queue, 0, partition_id, vec![partition_id as i32]).await;
        }

        let mut assigned = std::collections::HashSet::new();
        let mut total_rows = 0;
        for consumer_partition_id in 0..3 {
            let mut consumer = queue
                .consumer(
                    ConsumerConfig::new("hashed-group")
                        .with_consumer_partition(3, consumer_partition_id),
                )
                .await
                .unwrap();
            for partition_id in consumer.assigned_partitions() {
                assert!(
                    assigned.insert(*partition_id),
                    "queue partition {} assigned more than once",
                    partition_id
                );
            }
            total_rows += count_rows(&consumer.poll().await.unwrap());
        }

        let expected = (0..queue.partition_count()).collect::<std::collections::HashSet<_>>();
        assert_eq!(assigned, expected);
        assert_eq!(total_rows, queue.partition_count() as usize);
    }

    #[tokio::test]
    async fn test_consumer_partition_assignment_validates_slot() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = queue_uri(&temp_dir);
        let queue = Queue::builder().uri(&uri).create().await.unwrap();

        let err = queue
            .consumer(ConsumerConfig::new("bad-group").with_consumer_partition(0, 0))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("partition_count"), "{}", err);

        let err = queue
            .consumer(ConsumerConfig::new("bad-group").with_consumer_partition(2, 2))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("partition_id"), "{}", err);
    }

    #[tokio::test]
    async fn test_concurrent_partition_commits_are_merged() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = queue_uri(&temp_dir);
        let queue = Queue::builder()
            .uri(&uri)
            .partition_count(2)
            .create()
            .await
            .unwrap();
        produce_to_partition(&queue, 0, 0, vec![1]).await;
        produce_to_partition(&queue, 0, 1, vec![2]).await;

        let mut left = queue
            .consumer(ConsumerConfig::new("merged-group").with_partitions([0]))
            .await
            .unwrap();
        let mut right = queue
            .consumer(ConsumerConfig::new("merged-group").with_partitions([1]))
            .await
            .unwrap();
        assert_eq!(count_rows(&left.poll().await.unwrap()), 1);
        assert_eq!(count_rows(&right.poll().await.unwrap()), 1);

        let (left_commit, right_commit) =
            tokio::join!(left.commit_current(), right.commit_current());
        left_commit.unwrap();
        right_commit.unwrap();

        let mut resumed = queue
            .consumer(ConsumerConfig::new("merged-group"))
            .await
            .unwrap();
        assert!(resumed.poll().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_failed_poll_does_not_advance_in_memory_offsets() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = queue_uri(&temp_dir);
        let queue = Queue::builder()
            .uri(&uri)
            .partition_count(2)
            .create()
            .await
            .unwrap();
        produce_to_partition(&queue, 0, 0, vec![1]).await;

        let shard_id = queue.partitions()[1].shard_id;
        let corrupt_wal_dir = temp_dir
            .path()
            .join("_mem_wal")
            .join(shard_id.to_string())
            .join("wal");
        std::fs::create_dir_all(&corrupt_wal_dir).unwrap();
        std::fs::write(
            corrupt_wal_dir.join(test_wal_entry_filename(0)),
            b"not arrow",
        )
        .unwrap();

        let mut consumer = queue
            .consumer(ConsumerConfig::new("failed-poll-group"))
            .await
            .unwrap();
        assert!(consumer.poll().await.is_err());
        consumer.commit_current().await.unwrap();

        let mut resumed = queue
            .consumer(ConsumerConfig::new("failed-poll-group").with_partitions([0]))
            .await
            .unwrap();
        let polled = resumed.poll().await.unwrap();
        assert_eq!(count_rows(&polled), 1);
    }

    #[tokio::test]
    async fn test_committed_offsets_do_not_regress() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = queue_uri(&temp_dir);
        let queue = Queue::builder().uri(&uri).create().await.unwrap();
        produce_to_partition(&queue, 0, 0, vec![1]).await;
        produce_to_partition(&queue, 0, 0, vec![2]).await;

        let mut consumer = queue
            .consumer(ConsumerConfig::new("monotonic-group"))
            .await
            .unwrap();
        let batches = consumer
            .poll_with_options(PollOptions {
                max_entries_per_partition: 2,
            })
            .await
            .unwrap();
        assert_eq!(batches.len(), 2);

        consumer.commit(&[batches[1].clone()]).await.unwrap();
        consumer.commit(&[batches[0].clone()]).await.unwrap();

        let mut metadata_dataset = queue.dataset().clone();
        metadata_dataset.checkout_latest().await.unwrap();
        assert_eq!(
            metadata_dataset
                .metadata()
                .get("lance_queue.group.monotonic-group.commits.0.0.next_entry_position"),
            Some(&"3".to_string())
        );

        let mut resumed = queue
            .consumer(ConsumerConfig::new("monotonic-group"))
            .await
            .unwrap();
        assert!(resumed.poll().await.unwrap().is_empty());
    }

    #[test]
    fn test_group_id_rejects_reserved_commit_delimiter() {
        assert!(metadata::validate_group_id("service.v1").is_ok());
        assert!(metadata::validate_group_id("a.commits").is_err());
        assert!(metadata::validate_group_id("a.commits.b").is_err());
    }
}
