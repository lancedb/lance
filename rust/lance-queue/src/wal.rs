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

use crate::metadata::{PutCreateError, put_create};
use crate::shard_wal_path;

const WRITER_EPOCH_KEY: &str = "writer_epoch";
const QUEUE_PARTITION_KEY: &str = "lance_queue_partition";
const QUEUE_SHARD_ID_KEY: &str = "lance_queue_shard_id";
const FIRST_WAL_ENTRY_POSITION: u64 = 1;
const MAX_APPEND_CREATE_CONFLICTS: usize = 1024;
const APPEND_CONFLICT_REFRESH_INTERVAL: usize = 16;

/// Result of appending a WAL entry.
#[derive(Debug, Clone)]
pub struct WalAppendResult {
    /// Queue partition id.
    pub partition_id: u32,
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

/// A queue WAL entry read from storage.
#[derive(Debug, Clone)]
pub struct QueueEntry {
    /// Queue partition id.
    pub partition_id: u32,
    /// MemWAL shard id used by this partition.
    pub shard_id: Uuid,
    /// WAL entry position.
    pub entry_position: u64,
    /// Arrow batches stored in this WAL entry.
    pub batches: Vec<RecordBatch>,
}

/// Standalone WAL appender using MemWAL-compatible storage layout.
#[derive(Debug)]
pub struct WalAppender {
    object_store: Arc<ObjectStore>,
    wal_dir: Path,
    manifest_store: Arc<ShardManifestStore>,
    partition_id: u32,
    shard_id: Uuid,
    shard_spec_id: u32,
    writer_epoch: Mutex<Option<u64>>,
    next_entry_position: Mutex<Option<u64>>,
}

impl WalAppender {
    /// Create a WAL appender for a queue partition.
    pub fn new(
        object_store: Arc<ObjectStore>,
        base_path: Path,
        partition_id: u32,
        shard_id: Uuid,
        shard_spec_id: u32,
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
            shard_id,
            shard_spec_id,
            writer_epoch: Mutex::new(None),
            next_entry_position: Mutex::new(None),
        }
    }

    /// Append batches as one durable WAL entry.
    pub async fn append(&self, batches: Vec<RecordBatch>) -> Result<WalAppendResult> {
        validate_batches(&batches)?;
        let writer_epoch = self.writer_epoch().await?;
        self.manifest_store.check_fenced(writer_epoch).await?;
        let wal_data = Bytes::from(serialize_batches(
            &batches,
            self.partition_id,
            self.shard_id,
            writer_epoch,
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
            match put_create(
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
                        shard_id: self.shard_id,
                        entry_position,
                        num_batches,
                        num_rows,
                        wal_bytes,
                    });
                }
                Err(PutCreateError::AlreadyExists) => {
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
                Err(PutCreateError::Other(error)) => return Err(error),
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

    /// Find the earliest listed WAL position, or the first valid queue WAL position.
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
    pub async fn writer_epoch(&self) -> Result<u64> {
        let mut writer_epoch = self.writer_epoch.lock().await;
        if let Some(epoch) = *writer_epoch {
            return Ok(epoch);
        }

        let epoch = if let Some(manifest) = self.manifest_store.read_latest().await? {
            manifest.writer_epoch
        } else {
            match self.manifest_store.claim_epoch(self.shard_spec_id).await {
                Ok((epoch, _)) => epoch,
                Err(error) => match self.manifest_store.read_latest().await? {
                    Some(manifest) => manifest.writer_epoch,
                    None => return Err(error),
                },
            }
        };
        *writer_epoch = Some(epoch);
        Ok(epoch)
    }
}

/// Ordered reader for MemWAL-compatible queue WAL entries.
#[derive(Debug, Clone)]
pub struct WalTailer {
    object_store: Arc<ObjectStore>,
    wal_dir: Path,
    partition_id: u32,
    shard_id: Uuid,
}

impl WalTailer {
    /// Create a WAL tailer for a queue partition.
    pub fn new(
        object_store: Arc<ObjectStore>,
        base_path: Path,
        partition_id: u32,
        shard_id: Uuid,
    ) -> Self {
        Self {
            object_store,
            wal_dir: shard_wal_path(&base_path, &shard_id),
            partition_id,
            shard_id,
        }
    }

    /// Read a WAL entry. Returns `None` if the entry does not exist yet.
    pub async fn read_entry(&self, entry_position: u64) -> Result<Option<QueueEntry>> {
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
        let batches = read_batches(bytes, self.partition_id, self.shard_id)?;

        Ok(Some(QueueEntry {
            partition_id: self.partition_id,
            shard_id: self.shard_id,
            entry_position,
            batches,
        }))
    }

    /// Find the next position after the currently listed entries.
    pub async fn next_position(&self) -> Result<u64> {
        next_position_from_listing(
            self.object_store.as_ref(),
            &self.wal_dir,
            self.partition_id,
            self.shard_id,
        )
        .await
    }

    /// Find the earliest listed WAL position, or the first valid queue WAL position.
    pub async fn first_position(&self) -> Result<u64> {
        first_position_from_listing(
            self.object_store.as_ref(),
            &self.wal_dir,
            self.partition_id,
            self.shard_id,
        )
        .await
    }

    fn entry_path(&self, entry_position: u64) -> Path {
        self.wal_dir.child(wal_entry_filename(entry_position))
    }
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
    shard_id: Uuid,
    writer_epoch: u64,
) -> Result<Vec<u8>> {
    let schema = batches[0].schema();
    let mut metadata = schema.metadata().clone();
    metadata.insert(WRITER_EPOCH_KEY.to_string(), writer_epoch.to_string());
    metadata.insert(QUEUE_PARTITION_KEY.to_string(), partition_id.to_string());
    metadata.insert(QUEUE_SHARD_ID_KEY.to_string(), shard_id.to_string());
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

fn read_batches(bytes: Bytes, partition_id: u32, shard_id: Uuid) -> Result<Vec<RecordBatch>> {
    let cursor = Cursor::new(bytes);
    let reader = StreamReader::try_new(cursor, None)
        .map_err(|e| Error::io(format!("failed to open WAL IPC stream reader: {}", e)))?;
    validate_wal_schema_metadata(reader.schema().metadata(), partition_id, shard_id)?;

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
    shard_id: Uuid,
) -> Result<()> {
    let actual_partition_id = metadata
        .get(QUEUE_PARTITION_KEY)
        .ok_or_else(|| Error::io("WAL entry is missing queue partition metadata"))?
        .parse::<u32>()
        .map_err(|e| {
            Error::io(format!(
                "failed to parse WAL queue partition metadata: {}",
                e
            ))
        })?;
    if actual_partition_id != partition_id {
        return Err(Error::io(format!(
            "WAL entry partition metadata mismatch: expected {}, got {}",
            partition_id, actual_partition_id
        )));
    }

    let actual_shard_id = metadata
        .get(QUEUE_SHARD_ID_KEY)
        .ok_or_else(|| Error::io("WAL entry is missing queue shard metadata"))
        .and_then(|value| {
            Uuid::parse_str(value)
                .map_err(|e| Error::io(format!("failed to parse WAL queue shard metadata: {}", e)))
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
    let removed_partition_id = metadata.remove(QUEUE_PARTITION_KEY).is_some();
    let removed_shard_id = metadata.remove(QUEUE_SHARD_ID_KEY).is_some();
    let had_internal_metadata = removed_writer_epoch || removed_partition_id || removed_shard_id;
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
