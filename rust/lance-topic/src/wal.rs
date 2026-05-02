// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::Schema as ArrowSchema;
use bytes::Bytes;
use futures::StreamExt;
use lance::dataset::mem_wal::ShardManifestStore;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::metadata::{self, AppendError};
use crate::shard_wal_path;

const WRITER_EPOCH_KEY: &str = "writer_epoch";
const TOPIC_PARTITION_KEY: &str = "lance_topic_partition";
const TOPIC_PRODUCER_ID_KEY: &str = "lance_topic_producer_id";
const TOPIC_SHARD_ID_KEY: &str = "lance_topic_shard_id";
const FIRST_WAL_ENTRY_POSITION: u64 = 1;
const MAX_APPEND_CREATE_CONFLICTS: usize = 1024;
const APPEND_CONFLICT_REFRESH_INTERVAL: usize = 16;

/// Result of appending a WAL entry.
#[derive(Debug, Clone)]
pub struct WalAppendResult {
    /// Topic partition id.
    pub partition_id: u32,
    /// Producer shard id.
    pub producer_id: String,
    /// MemWAL shard id used by this partition.
    pub shard_id: Uuid,
    /// WAL entry position written.
    pub entry_position: u64,
    /// Number of Arrow batches in the entry.
    pub num_batches: usize,
    /// Number of rows in the entry.
    pub num_rows: usize,
    /// Serialized Arrow IPC stream size.
    pub wal_bytes: usize,
}

/// A topic WAL entry read from storage.
#[derive(Debug, Clone)]
pub struct TopicEntry {
    /// Topic partition id.
    pub partition_id: u32,
    /// Producer shard id.
    pub producer_id: String,
    /// MemWAL shard id used by this partition.
    pub shard_id: Uuid,
    /// WAL entry position.
    pub entry_position: u64,
    /// Arrow batches stored in this WAL entry.
    pub batches: Vec<RecordBatch>,
}

/// WAL appender for a single physical shard with epoch fencing.
#[derive(Debug)]
pub struct WalAppender {
    object_store: Arc<ObjectStore>,
    wal_dir: Path,
    manifest_store: Arc<ShardManifestStore>,
    partition_id: u32,
    producer_id: String,
    shard_id: Uuid,
    writer_epoch: u64,
    next_entry_position: Mutex<Option<u64>>,
}

impl WalAppender {
    /// Open a partition writer and claim a new writer epoch.
    pub async fn open(
        object_store: Arc<ObjectStore>,
        base_path: Path,
        partition_id: u32,
        producer_id: impl Into<String>,
        shard_id: Uuid,
        shard_spec_id: u32,
    ) -> Result<Self> {
        let producer_id = producer_id.into();
        let manifest_store = Arc::new(ShardManifestStore::new(
            object_store.clone(),
            &base_path,
            shard_id,
            2,
        ));
        let (writer_epoch, _) = manifest_store.claim_epoch(shard_spec_id).await?;
        Ok(Self {
            object_store,
            wal_dir: shard_wal_path(&base_path, &shard_id),
            manifest_store,
            partition_id,
            producer_id,
            shard_id,
            writer_epoch,
            next_entry_position: Mutex::new(None),
        })
    }

    /// Create a partition writer with an already-claimed writer epoch.
    pub fn new_with_writer_epoch(
        object_store: Arc<ObjectStore>,
        base_path: Path,
        partition_id: u32,
        producer_id: impl Into<String>,
        shard_id: Uuid,
        _shard_spec_id: u32,
        writer_epoch: u64,
    ) -> Self {
        let producer_id = producer_id.into();
        let manifest_store = Arc::new(ShardManifestStore::new(
            object_store.clone(),
            &base_path,
            shard_id,
            2,
        ));
        Self {
            object_store,
            wal_dir: shard_wal_path(&base_path, &shard_id),
            manifest_store,
            partition_id,
            producer_id,
            shard_id,
            writer_epoch,
            next_entry_position: Mutex::new(None),
        }
    }

    /// Append batches as one durable WAL entry.
    pub async fn append(&self, batches: Vec<RecordBatch>) -> Result<WalAppendResult> {
        validate_batches(&batches)?;
        let wal_data = Bytes::from(serialize_batches(
            &batches,
            self.partition_id,
            &self.producer_id,
            self.shard_id,
            self.writer_epoch,
        )?);
        let wal_bytes = wal_data.len();
        let num_batches = batches.len();
        let num_rows = batches.iter().map(RecordBatch::num_rows).sum();

        let mut next_entry_position = self.next_entry_position.lock().await;
        if next_entry_position.is_none() {
            *next_entry_position = Some(self.next_position().await?);
        }

        let mut create_conflicts = 0;
        loop {
            let entry_position = next_entry_position.ok_or_else(|| {
                Error::internal(format!(
                    "missing cached WAL entry position for partition_id {} shard_id {}",
                    self.partition_id, self.shard_id
                ))
            })?;
            let filename = wal_entry_filename(entry_position);
            match metadata::append(
                self.object_store.as_ref(),
                &self.wal_dir,
                &filename,
                wal_data.clone(),
            )
            .await
            {
                Ok(()) => {
                    *next_entry_position = Some(increment_entry_position(
                        entry_position,
                        self.partition_id,
                        self.shard_id,
                    )?);
                    return Ok(WalAppendResult {
                        partition_id: self.partition_id,
                        producer_id: self.producer_id.clone(),
                        shard_id: self.shard_id,
                        entry_position,
                        num_batches,
                        num_rows,
                        wal_bytes,
                    });
                }
                Err(AppendError::AlreadyExists) => {
                    self.check_fenced().await?;
                    create_conflicts += 1;
                    if create_conflicts >= MAX_APPEND_CREATE_CONFLICTS {
                        return Err(Error::io(format!(
                            "failed to append WAL entry for partition_id {} shard_id {} after {} create conflicts",
                            self.partition_id, self.shard_id, create_conflicts
                        )));
                    }
                    if create_conflicts % APPEND_CONFLICT_REFRESH_INTERVAL == 0 {
                        *next_entry_position = Some(self.next_position().await?);
                    } else {
                        *next_entry_position = Some(increment_entry_position(
                            entry_position,
                            self.partition_id,
                            self.shard_id,
                        )?);
                    }
                }
                Err(AppendError::Other(error)) => {
                    self.check_fenced().await?;
                    return Err(error);
                }
            }
        }
    }

    /// Find the next append position from the WAL directory listing.
    pub async fn next_position(&self) -> Result<u64> {
        next_position_from_listing(
            self.object_store.as_ref(),
            &self.wal_dir,
            self.partition_id,
            self.shard_id,
        )
        .await
    }

    /// Find the earliest listed WAL position, or the first valid topic WAL position.
    pub async fn first_position(&self) -> Result<u64> {
        first_position_from_listing(
            self.object_store.as_ref(),
            &self.wal_dir,
            self.partition_id,
            self.shard_id,
        )
        .await
    }

    /// Writer epoch recorded in the shard manifest.
    pub fn writer_epoch(&self) -> u64 {
        self.writer_epoch
    }

    /// Check that this writer's epoch has not been fenced.
    pub async fn check_fenced(&self) -> Result<()> {
        self.manifest_store.check_fenced(self.writer_epoch).await
    }
}

const MAX_CURSOR_PROBE: u64 = 4096;

/// Ordered reader for MemWAL-compatible topic WAL entries.
#[derive(Debug, Clone)]
pub struct WalTailer {
    object_store: Arc<ObjectStore>,
    wal_dir: Path,
    manifest_store: Arc<ShardManifestStore>,
    partition_id: u32,
    producer_id: String,
    shard_id: Uuid,
    update_cursor: bool,
}

impl WalTailer {
    /// Create a WAL tailer for a topic partition shard.
    pub fn new(
        object_store: Arc<ObjectStore>,
        base_path: Path,
        partition_id: u32,
        producer_id: impl Into<String>,
        shard_id: Uuid,
    ) -> Self {
        let manifest_store = Arc::new(ShardManifestStore::new(
            object_store.clone(),
            &base_path,
            shard_id,
            2,
        ));
        Self {
            object_store,
            wal_dir: shard_wal_path(&base_path, &shard_id),
            manifest_store,
            partition_id,
            producer_id: producer_id.into(),
            shard_id,
            update_cursor: false,
        }
    }

    /// Enable async best-effort cursor updates on read.
    ///
    /// When enabled, successful `read_entry` calls asynchronously update
    /// `wal_entry_position_last_seen` in the shard manifest. Only enable this
    /// for shards that do not have active writers claiming epochs, such as
    /// consumer group offset shards.
    pub fn with_cursor_updates(mut self, enabled: bool) -> Self {
        self.update_cursor = enabled;
        self
    }

    /// Read a WAL entry. Returns `None` if the entry does not exist yet.
    ///
    /// When cursor updates are enabled via [`with_cursor_updates`], successful
    /// reads asynchronously update `wal_entry_position_last_seen` in the shard
    /// manifest as a best-effort hint for future tailers.
    pub async fn read_entry(&self, entry_position: u64) -> Result<Option<TopicEntry>> {
        let path = self.entry_path(entry_position);
        let data = match self.object_store.inner.get(&path).await {
            Ok(data) => data,
            Err(object_store::Error::NotFound { .. }) => return Ok(None),
            Err(e) => {
                return Err(Error::io(format!(
                    "failed to read WAL entry {} for partition_id {} shard_id {}: {}",
                    entry_position, self.partition_id, self.shard_id, e
                )));
            }
        };

        let bytes = data.bytes().await.map_err(|e| {
            Error::io(format!(
                "failed to read WAL entry bytes at {} for partition_id {} shard_id {}: {}",
                path, self.partition_id, self.shard_id, e
            ))
        })?;
        let batches = read_batches(bytes, self.partition_id, &self.producer_id, self.shard_id)?;

        if self.update_cursor {
            self.fire_cursor_update(entry_position);
        }

        Ok(Some(TopicEntry {
            partition_id: self.partition_id,
            producer_id: self.producer_id.clone(),
            shard_id: self.shard_id,
            entry_position,
            batches,
        }))
    }

    /// Find the next append position.
    ///
    /// Uses `wal_entry_position_last_seen` from the shard manifest as a probe
    /// hint and scans forward to find the true tip. Falls back to a full
    /// directory listing if the hint is unavailable or stale.
    pub async fn next_position(&self) -> Result<u64> {
        if let Some(hint) = self.manifest_cursor_hint().await
            && hint >= FIRST_WAL_ENTRY_POSITION
            && let Some(tip) = self.probe_forward(hint).await?
        {
            return Ok(tip);
        }
        next_position_from_listing(
            self.object_store.as_ref(),
            &self.wal_dir,
            self.partition_id,
            self.shard_id,
        )
        .await
    }

    /// Find the earliest listed WAL position, or the first valid topic WAL position.
    pub async fn first_position(&self) -> Result<u64> {
        first_position_from_listing(
            self.object_store.as_ref(),
            &self.wal_dir,
            self.partition_id,
            self.shard_id,
        )
        .await
    }

    /// Read `wal_entry_position_last_seen` from the shard manifest as a hint.
    async fn manifest_cursor_hint(&self) -> Option<u64> {
        let manifest = self.manifest_store.read_latest().await.ok()??;
        let hint = manifest.wal_entry_position_last_seen;
        if hint > 0 { Some(hint) } else { None }
    }

    /// Probe forward from `hint` to find the actual next position.
    ///
    /// Returns `None` if the hint entry itself does not exist (stale cursor),
    /// causing the caller to fall back to listing.
    async fn probe_forward(&self, hint: u64) -> Result<Option<u64>> {
        if !self.entry_exists(hint).await? {
            return Ok(None);
        }
        let mut pos = hint + 1;
        while pos - hint <= MAX_CURSOR_PROBE {
            if !self.entry_exists(pos).await? {
                return Ok(Some(pos));
            }
            pos += 1;
        }
        // Exceeded probe limit — fall back to listing
        Ok(None)
    }

    async fn entry_exists(&self, entry_position: u64) -> Result<bool> {
        let path = self.entry_path(entry_position);
        match self.object_store.inner.head(&path).await {
            Ok(_) => Ok(true),
            Err(object_store::Error::NotFound { .. }) => Ok(false),
            Err(e) => Err(Error::io(format!(
                "failed to check WAL entry {} for shard {}: {}",
                entry_position, self.shard_id, e
            ))),
        }
    }

    /// Fire-and-forget update of `wal_entry_position_last_seen` in the shard manifest.
    fn fire_cursor_update(&self, entry_position: u64) {
        let manifest_store = self.manifest_store.clone();
        tokio::spawn(async move {
            let _ = update_manifest_cursor(&manifest_store, entry_position).await;
        });
    }

    fn entry_path(&self, entry_position: u64) -> Path {
        self.wal_dir.child(wal_entry_filename(entry_position))
    }
}

async fn update_manifest_cursor(manifest_store: &ShardManifestStore, entry_position: u64) {
    let Ok(Some(manifest)) = manifest_store.read_latest().await else {
        return;
    };
    if entry_position <= manifest.wal_entry_position_last_seen {
        return;
    }
    let mut updated = manifest;
    updated.version += 1;
    updated.wal_entry_position_last_seen = entry_position;
    // Best-effort: silently ignore version conflicts or any other write errors.
    let _ = manifest_store.write(&updated).await;
}

fn validate_batches(batches: &[RecordBatch]) -> Result<()> {
    if batches.is_empty() {
        return Err(Error::invalid_input(
            "cannot append an empty batch list to WAL",
        ));
    }

    let schema = batches[0].schema();
    for (idx, batch) in batches.iter().enumerate() {
        if batch.num_rows() == 0 {
            return Err(Error::invalid_input(format!(
                "cannot append empty batch at index {} to WAL",
                idx
            )));
        }
        if batch.schema_ref().fields() != schema.fields() {
            return Err(Error::invalid_input(format!(
                "batch at index {} has a different schema from the first batch",
                idx
            )));
        }
    }

    Ok(())
}

fn serialize_batches(
    batches: &[RecordBatch],
    partition_id: u32,
    producer_id: &str,
    shard_id: Uuid,
    writer_epoch: u64,
) -> Result<Vec<u8>> {
    let schema = batches[0].schema();
    let mut metadata = schema.metadata().clone();
    metadata.insert(WRITER_EPOCH_KEY.to_string(), writer_epoch.to_string());
    metadata.insert(TOPIC_PARTITION_KEY.to_string(), partition_id.to_string());
    metadata.insert(TOPIC_PRODUCER_ID_KEY.to_string(), producer_id.to_owned());
    metadata.insert(TOPIC_SHARD_ID_KEY.to_string(), shard_id.to_string());
    let schema_with_metadata = Arc::new(ArrowSchema::new_with_metadata(
        schema.fields().to_vec(),
        metadata,
    ));

    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &schema_with_metadata)
            .map_err(|e| Error::io(format!("failed to create Arrow IPC stream writer: {}", e)))?;
        for batch in batches {
            writer.write(batch).map_err(|e| {
                Error::io(format!("failed to write batch to WAL IPC stream: {}", e))
            })?;
        }
        writer
            .finish()
            .map_err(|e| Error::io(format!("failed to finish WAL IPC stream: {}", e)))?;
    }

    Ok(buffer)
}

fn read_batches(
    bytes: Bytes,
    partition_id: u32,
    producer_id: &str,
    shard_id: Uuid,
) -> Result<Vec<RecordBatch>> {
    let cursor = Cursor::new(bytes);
    let reader = StreamReader::try_new(cursor, None)
        .map_err(|e| Error::io(format!("failed to open WAL IPC stream reader: {}", e)))?;
    validate_wal_schema_metadata(
        reader.schema().metadata(),
        partition_id,
        producer_id,
        shard_id,
    )?;

    let mut batches = Vec::new();
    for batch in reader {
        let batch =
            batch.map_err(|e| Error::io(format!("failed to read WAL IPC stream batch: {}", e)))?;
        batches.push(strip_internal_wal_metadata(batch)?);
    }
    Ok(batches)
}

fn validate_wal_schema_metadata(
    metadata: &HashMap<String, String>,
    partition_id: u32,
    producer_id: &str,
    shard_id: Uuid,
) -> Result<()> {
    let actual_partition_id = metadata
        .get(TOPIC_PARTITION_KEY)
        .ok_or_else(|| Error::io("WAL entry is missing topic partition metadata"))?
        .parse::<u32>()
        .map_err(|e| {
            Error::io(format!(
                "failed to parse WAL topic partition metadata: {}",
                e
            ))
        })?;
    if actual_partition_id != partition_id {
        return Err(Error::io(format!(
            "WAL entry partition metadata mismatch: expected {}, got {}",
            partition_id, actual_partition_id
        )));
    }

    let actual_producer_id = metadata
        .get(TOPIC_PRODUCER_ID_KEY)
        .ok_or_else(|| Error::io("WAL entry is missing topic producer metadata"))?;
    if actual_producer_id != producer_id {
        return Err(Error::io(format!(
            "WAL entry producer metadata mismatch: expected {}, got {}",
            producer_id, actual_producer_id
        )));
    }

    let actual_shard_id = metadata
        .get(TOPIC_SHARD_ID_KEY)
        .ok_or_else(|| Error::io("WAL entry is missing topic shard metadata"))
        .and_then(|value| {
            Uuid::parse_str(value)
                .map_err(|e| Error::io(format!("failed to parse WAL topic shard metadata: {}", e)))
        })?;
    if actual_shard_id != shard_id {
        return Err(Error::io(format!(
            "WAL entry shard metadata mismatch: expected {}, got {}",
            shard_id, actual_shard_id
        )));
    }

    Ok(())
}

fn strip_internal_wal_metadata(batch: RecordBatch) -> Result<RecordBatch> {
    let schema = batch.schema();
    let mut metadata = schema.metadata().clone();
    let removed_writer_epoch = metadata.remove(WRITER_EPOCH_KEY).is_some();
    let removed_partition_id = metadata.remove(TOPIC_PARTITION_KEY).is_some();
    let removed_producer_id = metadata.remove(TOPIC_PRODUCER_ID_KEY).is_some();
    let removed_shard_id = metadata.remove(TOPIC_SHARD_ID_KEY).is_some();
    let had_internal_metadata =
        removed_writer_epoch || removed_partition_id || removed_producer_id || removed_shard_id;
    if !had_internal_metadata {
        return Ok(batch);
    }

    let logical_schema = Arc::new(ArrowSchema::new_with_metadata(
        schema.fields().to_vec(),
        metadata,
    ));
    RecordBatch::try_new(logical_schema, batch.columns().to_vec())
        .map_err(|e| Error::io(format!("failed to strip internal WAL metadata: {}", e)))
}

async fn next_position_from_listing(
    object_store: &ObjectStore,
    wal_dir: &Path,
    partition_id: u32,
    shard_id: Uuid,
) -> Result<u64> {
    let mut max_position = None::<u64>;
    let mut stream = object_store.inner.list(Some(wal_dir));

    while let Some(item) = stream.next().await {
        let meta = item.map_err(|e| {
            Error::io(format!(
                "failed to list WAL directory for partition_id {} shard_id {}: {}",
                partition_id, shard_id, e
            ))
        })?;
        if let Some(filename) = meta.location.filename()
            && let Some(position) = parse_bit_reversed_filename(filename, "arrow")
        {
            max_position = Some(max_position.map_or(position, |max| max.max(position)));
        }
    }

    match max_position {
        Some(position) => increment_entry_position(position, partition_id, shard_id),
        None => Ok(FIRST_WAL_ENTRY_POSITION),
    }
}

async fn first_position_from_listing(
    object_store: &ObjectStore,
    wal_dir: &Path,
    partition_id: u32,
    shard_id: Uuid,
) -> Result<u64> {
    let mut min_position = None::<u64>;
    let mut stream = object_store.inner.list(Some(wal_dir));

    while let Some(item) = stream.next().await {
        let meta = item.map_err(|e| {
            Error::io(format!(
                "failed to list WAL directory for partition_id {} shard_id {}: {}",
                partition_id, shard_id, e
            ))
        })?;
        if let Some(filename) = meta.location.filename()
            && let Some(position) = parse_bit_reversed_filename(filename, "arrow")
        {
            min_position = Some(min_position.map_or(position, |min| min.min(position)));
        }
    }

    match min_position {
        Some(position) => Ok(position),
        None => Ok(FIRST_WAL_ENTRY_POSITION),
    }
}

fn increment_entry_position(entry_position: u64, partition_id: u32, shard_id: Uuid) -> Result<u64> {
    entry_position.checked_add(1).ok_or_else(|| {
        Error::io(format!(
            "WAL entry position overflow for partition_id {} shard_id {}",
            partition_id, shard_id
        ))
    })
}

fn bit_reverse_u64(value: u64) -> u64 {
    value.reverse_bits()
}

fn bit_reversed_filename(value: u64, extension: &str) -> String {
    format!("{:064b}.{}", bit_reverse_u64(value), extension)
}

fn wal_entry_filename(entry_position: u64) -> String {
    bit_reversed_filename(entry_position, "arrow")
}

fn parse_bit_reversed_filename(filename: &str, expected_extension: &str) -> Option<u64> {
    let (stem, extension) = filename.rsplit_once('.')?;
    if extension != expected_extension
        || stem.len() != 64
        || !stem.chars().all(|char| char == '0' || char == '1')
    {
        return None;
    }
    let reversed = u64::from_str_radix(stem, 2).ok()?;
    Some(bit_reverse_u64(reversed))
}
