// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    future::Future,
    ops::{DerefMut, Range},
    panic::AssertUnwindSafe,
    sync::Arc,
    task::Poll,
};

use arrow::array::AsArray;
use arrow::datatypes::{UInt8Type, UInt32Type, UInt64Type};
use arrow_array::Array;
use arrow_array::RecordBatch;
use arrow_array::builder::{LargeBinaryBuilder, PrimitiveBuilder, StringBuilder};
use arrow_schema::DataType as ArrowDataType;
use bytes::Bytes;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt, TryStreamExt, stream};
use lance_arrow::{BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY, FieldExt};
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use lance_io::scheduler::{FileScheduler, ScanScheduler, SchedulerConfig};
use object_store::path::Path;
use tokio::io::AsyncWriteExt;
use tokio::sync::{Mutex, OnceCell, oneshot};
use url::Url;

use super::take::TakeBuilder;
use super::write::ExternalBlobMode;
use super::{Dataset, ProjectionRequest};
use arrow_array::StructArray;
use lance_core::datatypes::{BlobKind, BlobVersion};
use lance_core::utils::blob::blob_path;
use lance_core::{Error, Result, utils::address::RowAddress};
use lance_io::traits::{Reader, WriteExt, Writer};
use lance_io::utils::CachedFileSize;

const INLINE_MAX: usize = 64 * 1024; // 64KB inline cutoff
const DEDICATED_THRESHOLD: usize = 4 * 1024 * 1024; // 4MB dedicated cutoff
const PACK_FILE_MAX_SIZE: usize = 1024 * 1024 * 1024; // 1GiB per .pack sidecar
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ResolvedExternalBase {
    pub base_id: u32,
    pub relative_path: String,
}

#[derive(Clone, Debug)]
pub(super) struct ExternalBaseCandidate {
    pub base_id: u32,
    pub store_prefix: String,
    pub base_path: Path,
    pub store_params: ObjectStoreParams,
}

#[derive(Debug)]
pub(super) struct ExternalBaseResolver {
    candidates: Vec<ExternalBaseCandidate>,
    store_registry: Arc<ObjectStoreRegistry>,
}

impl ExternalBaseResolver {
    pub(super) fn new(
        candidates: Vec<ExternalBaseCandidate>,
        store_registry: Arc<ObjectStoreRegistry>,
    ) -> Self {
        Self {
            candidates,
            store_registry,
        }
    }

    pub(crate) async fn resolve_external_uri(
        &self,
        uri: &str,
    ) -> Result<Option<ResolvedExternalBase>> {
        let uri_path = ObjectStore::extract_path_from_uri(self.store_registry.clone(), uri)?;

        let mut best_match: Option<(usize, ResolvedExternalBase)> = None;
        for candidate in &self.candidates {
            let uri_store_prefix = self
                .store_registry
                .calculate_object_store_prefix(uri, candidate.store_params.storage_options())?;
            if candidate.store_prefix != uri_store_prefix {
                continue;
            }
            let Some(relative_parts) = uri_path.prefix_match(&candidate.base_path) else {
                continue;
            };
            let relative_path = Path::from_iter(relative_parts);
            if relative_path.as_ref().is_empty() {
                continue;
            }
            let prefix_len = candidate.base_path.parts().count();
            if best_match
                .as_ref()
                .map(|(current_len, _)| prefix_len > *current_len)
                .unwrap_or(true)
            {
                best_match = Some((
                    prefix_len,
                    ResolvedExternalBase {
                        base_id: candidate.base_id,
                        relative_path: relative_path.to_string(),
                    },
                ));
            }
        }

        Ok(best_match.map(|(_, matched)| matched))
    }
}

// Maintains rolling `.blob` sidecar files for packed blobs.
// Layout: data/{data_file_key}/{obfuscated_blob_id:032b}.blob where each file is an
// unframed concatenation of blob payloads; descriptors store (blob_id,
// position, size) to locate each slice. A dedicated struct keeps path state
// and rolling size separate from the per-batch preprocessor logic, so we can
// reuse the same writer across rows and close/roll files cleanly on finish.
struct PackWriter {
    object_store: ObjectStore,
    data_dir: Path,
    data_file_key: String,
    max_pack_size: usize,
    current_blob_id: Option<u32>,
    writer: Option<Box<dyn lance_io::traits::Writer>>,
    current_size: usize,
}

impl PackWriter {
    fn new(object_store: ObjectStore, data_dir: Path, data_file_key: String) -> Self {
        Self {
            object_store,
            data_dir,
            data_file_key,
            max_pack_size: PACK_FILE_MAX_SIZE,
            current_blob_id: None,
            writer: None,
            current_size: 0,
        }
    }

    async fn start_new_pack(&mut self, blob_id: u32) -> Result<()> {
        let path = blob_path(&self.data_dir, &self.data_file_key, blob_id);
        let writer = self.object_store.create(&path).await?;
        self.writer = Some(writer);
        self.current_blob_id = Some(blob_id);
        self.current_size = 0;
        Ok(())
    }

    /// Append `data` to the current `.blob` file, rolling to a new file when
    /// `max_pack_size` would be exceeded.
    ///
    /// alloc_blob_id: called only when a new pack file is opened; returns the
    /// blob_id used as the file name.
    ///
    /// Returns `(blob_id, position)` where
    /// position is the start offset of this payload in that pack file.
    async fn write_with_allocator<F>(
        &mut self,
        alloc_blob_id: &mut F,
        source: BlobWriteSource<'_>,
    ) -> Result<(u32, u64)>
    where
        F: FnMut() -> u32,
    {
        let len = source.size();
        if self
            .current_blob_id
            .map(|_| self.current_size + len > self.max_pack_size)
            .unwrap_or(true)
        {
            let blob_id = alloc_blob_id();
            self.finish().await?;
            self.start_new_pack(blob_id).await?;
        }

        let writer = self.writer.as_mut().expect("pack writer is initialized");
        let position = self.current_size as u64;
        source.write_to(writer.as_mut()).await?;
        self.current_size += len;
        Ok((self.current_blob_id.expect("pack blob id"), position))
    }

    async fn finish(&mut self) -> Result<()> {
        if let Some(mut writer) = self.writer.take() {
            Writer::shutdown(writer.as_mut()).await?;
        }
        self.current_blob_id = None;
        self.current_size = 0;
        Ok(())
    }
}

/// Preprocesses blob v2 columns on the write path so the encoder only sees lightweight descriptors:
///
/// - Spills large blobs to sidecar files before encoding, reducing memory/CPU and avoiding copying huge payloads through page builders.
/// - Emits `blob_id/blob_size` tied to the data file stem, giving readers a stable path independent of temporary fragment IDs assigned during write.
/// - Leaves small inline blobs and URI rows unchanged for compatibility.
pub struct BlobPreprocessor {
    object_store: ObjectStore,
    data_dir: Path,
    data_file_key: String,
    local_counter: u32,
    pack_writer: PackWriter,
    blob_v2_cols: Vec<bool>,
    dedicated_thresholds: Vec<usize>,
    writer_metadata: Vec<HashMap<String, String>>,
    external_base_resolver: Option<Arc<ExternalBaseResolver>>,
    allow_external_blob_outside_bases: bool,
    external_blob_mode: ExternalBlobMode,
    source_store_registry: Arc<ObjectStoreRegistry>,
    source_store_params: ObjectStoreParams,
}

/// A logical slice of an external blob that can be materialized or streamed into Lance-managed
/// storage.
struct ExternalBlobSource {
    reader: Box<dyn Reader>,
    start: u64,
    size: u64,
}

/// A blob payload source used by packed and dedicated writers.
///
/// Inline blobs still need an in-memory byte slice because they are embedded into the descriptor
/// array, while external ingest can stream bytes from the source reader.
enum BlobWriteSource<'a> {
    Bytes(&'a [u8]),
    External(&'a ExternalBlobSource),
}

impl ExternalBlobSource {
    /// Return the logical payload size after applying any external slice.
    fn size(&self) -> u64 {
        self.size
    }

    /// Convert the logical slice into the current reader API's usize-based range.
    fn reader_range(&self) -> Result<Range<usize>> {
        let start = usize::try_from(self.start).map_err(|_| {
            Error::invalid_input(format!(
                "External blob position {} does not fit into usize",
                self.start
            ))
        })?;
        let size = usize::try_from(self.size).map_err(|_| {
            Error::invalid_input(format!(
                "External blob size {} does not fit into usize",
                self.size
            ))
        })?;
        let end = start.checked_add(size).ok_or_else(|| {
            Error::invalid_input(format!(
                "External blob range overflows usize: position={}, size={}",
                self.start, self.size
            ))
        })?;
        Ok(start..end)
    }

    /// Materialize the slice into memory for the inline blob path.
    async fn read_all(&self) -> Result<bytes::Bytes> {
        let range = self.reader_range()?;
        self.reader.get_range(range).await.map_err(Into::into)
    }

    /// Stream the slice into a writer for packed or dedicated blob storage.
    async fn copy_to_writer(&self, writer: &mut dyn Writer) -> Result<()> {
        let range = self.reader_range()?;
        writer
            .copy_range_from_reader(self.reader.as_ref(), range)
            .await?;
        Ok(())
    }
}

impl BlobWriteSource<'_> {
    /// Return the payload size regardless of whether bytes come from memory or an external reader.
    fn size(&self) -> usize {
        match self {
            Self::Bytes(data) => data.len(),
            Self::External(source) => usize::try_from(source.size())
                .expect("packed and inline external blobs must fit into usize"),
        }
    }

    /// Write the payload into Lance-managed storage without forcing callers to branch on source
    /// type.
    async fn write_to(&self, writer: &mut dyn Writer) -> Result<()> {
        match self {
            Self::Bytes(data) => {
                writer.write_all(data).await?;
                Ok(())
            }
            Self::External(source) => source.copy_to_writer(writer).await,
        }
    }
}

impl BlobPreprocessor {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        object_store: ObjectStore,
        data_dir: Path,
        data_file_key: String,
        schema: &lance_core::datatypes::Schema,
        external_base_resolver: Option<Arc<ExternalBaseResolver>>,
        allow_external_blob_outside_bases: bool,
        external_blob_mode: ExternalBlobMode,
        source_store_registry: Arc<ObjectStoreRegistry>,
        source_store_params: ObjectStoreParams,
        pack_file_size_threshold: Option<usize>,
    ) -> Self {
        let mut pack_writer = PackWriter::new(
            object_store.clone(),
            data_dir.clone(),
            data_file_key.clone(),
        );
        if let Some(max_bytes) = pack_file_size_threshold {
            pack_writer.max_pack_size = max_bytes;
        }
        let arrow_schema = arrow_schema::Schema::from(schema);
        let fields = arrow_schema.fields();
        let blob_v2_cols = fields.iter().map(|field| field.is_blob_v2()).collect();
        let dedicated_thresholds = fields
            .iter()
            .map(|field| dedicated_threshold_from_metadata(field.as_ref()))
            .collect();
        let writer_metadata = fields
            .iter()
            .map(|field| field.metadata().clone())
            .collect();
        Self {
            object_store,
            data_dir,
            data_file_key,
            // Start at 1 to avoid a potential all-zero blob_id value.
            local_counter: 1,
            pack_writer,
            blob_v2_cols,
            dedicated_thresholds,
            writer_metadata,
            external_base_resolver,
            allow_external_blob_outside_bases,
            external_blob_mode,
            source_store_registry,
            source_store_params,
        }
    }

    fn next_blob_id(&mut self) -> u32 {
        let id = self.local_counter;
        self.local_counter += 1;
        id
    }

    async fn write_dedicated(&mut self, blob_id: u32, source: BlobWriteSource<'_>) -> Result<Path> {
        let path = blob_path(&self.data_dir, &self.data_file_key, blob_id);
        let mut writer = self.object_store.create(&path).await?;
        source.write_to(writer.as_mut()).await?;
        Writer::shutdown(&mut writer).await?;
        Ok(path)
    }

    async fn write_packed(&mut self, source: BlobWriteSource<'_>) -> Result<(u32, u64)> {
        let (counter, pack_writer) = (&mut self.local_counter, &mut self.pack_writer);
        pack_writer
            .write_with_allocator(
                &mut || {
                    let id = *counter;
                    *counter += 1;
                    id
                },
                source,
            )
            .await
    }

    async fn resolve_external_reference(&mut self, uri: &str) -> Result<(u32, String)> {
        let mapped = if let Some(resolver) = &self.external_base_resolver {
            resolver.resolve_external_uri(uri).await?
        } else {
            None
        };
        if let Some(mapped) = mapped {
            return Ok((mapped.base_id, mapped.relative_path));
        }

        if self.allow_external_blob_outside_bases {
            let normalized = normalize_external_absolute_uri(uri)?;
            return Ok((0, normalized));
        }

        Err(Error::invalid_input(format!(
            "External blob URI '{}' is outside registered external bases (dataset root is not allowed). Set allow_external_blob_outside_bases=true to store it as absolute external URI.",
            uri
        )))
    }

    async fn open_external_source(
        &mut self,
        uri: &str,
        position: Option<u64>,
        size: Option<u64>,
    ) -> Result<ExternalBlobSource> {
        let (object_store, path) = ObjectStore::from_uri_and_params(
            self.source_store_registry.clone(),
            uri,
            &self.source_store_params,
        )
        .await?;
        let reader = object_store.open(&path).await?;
        match (position, size) {
            (Some(position), Some(size)) => {
                position.checked_add(size).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "External blob range overflows u64: position={}, size={}",
                        position, size
                    ))
                })?;
                Ok(ExternalBlobSource {
                    reader,
                    start: position,
                    size,
                })
            }
            (None, None) => {
                let size = reader.size().await? as u64;
                Ok(ExternalBlobSource {
                    reader,
                    start: 0,
                    size,
                })
            }
            _ => Err(Error::invalid_input(format!(
                "External blob URI '{}' must set both position and size when slicing for ingest",
                uri
            ))),
        }
    }

    pub(crate) async fn preprocess_batch(&mut self, batch: &RecordBatch) -> Result<RecordBatch> {
        let expected_columns = self.blob_v2_cols.len();
        if batch.num_columns() != expected_columns {
            return Err(Error::invalid_input(format!(
                "Unexpected number of columns: expected {}, got {}",
                expected_columns,
                batch.num_columns()
            )));
        }

        let batch_schema = batch.schema();
        let batch_fields = batch_schema.fields();

        let mut new_columns = Vec::with_capacity(batch.num_columns());
        let mut new_fields = Vec::with_capacity(batch.num_columns());

        for idx in 0..batch.num_columns() {
            let array = batch.column(idx);
            let field = &batch_fields[idx];
            if !self.blob_v2_cols[idx] {
                new_columns.push(array.clone());
                new_fields.push(field.clone());
                continue;
            }

            let struct_arr = array
                .as_any()
                .downcast_ref::<arrow_array::StructArray>()
                .ok_or_else(|| Error::invalid_input("Blob column was not a struct array"))?;

            let data_col = struct_arr
                .column_by_name("data")
                .ok_or_else(|| Error::invalid_input("Blob struct missing `data` field"))?
                .as_binary::<i64>();
            let uri_col = struct_arr
                .column_by_name("uri")
                .ok_or_else(|| Error::invalid_input("Blob struct missing `uri` field"))?
                .as_string::<i32>();
            let position_col = struct_arr
                .column_by_name("position")
                .map(|col| col.as_primitive::<UInt64Type>());
            let size_col = struct_arr
                .column_by_name("size")
                .map(|col| col.as_primitive::<UInt64Type>());

            let mut data_builder = LargeBinaryBuilder::with_capacity(struct_arr.len(), 0);
            let mut uri_builder = StringBuilder::with_capacity(struct_arr.len(), 0);
            let mut blob_id_builder =
                PrimitiveBuilder::<arrow_array::types::UInt32Type>::with_capacity(struct_arr.len());
            let mut blob_size_builder =
                PrimitiveBuilder::<arrow_array::types::UInt64Type>::with_capacity(struct_arr.len());
            let mut kind_builder = PrimitiveBuilder::<UInt8Type>::with_capacity(struct_arr.len());
            let mut position_builder =
                PrimitiveBuilder::<arrow_array::types::UInt64Type>::with_capacity(struct_arr.len());

            let struct_nulls = struct_arr.nulls();

            for i in 0..struct_arr.len() {
                if struct_arr.is_null(i) {
                    data_builder.append_null();
                    uri_builder.append_null();
                    blob_id_builder.append_null();
                    blob_size_builder.append_null();
                    kind_builder.append_null();
                    position_builder.append_null();
                    continue;
                }

                let has_data = !data_col.is_null(i);
                let has_uri = !uri_col.is_null(i);
                let has_position = position_col
                    .as_ref()
                    .map(|col| !col.is_null(i))
                    .unwrap_or(false);
                let has_size = size_col
                    .as_ref()
                    .map(|col| !col.is_null(i))
                    .unwrap_or(false);
                let data_len = if has_data { data_col.value(i).len() } else { 0 };

                let dedicated_threshold = self.dedicated_thresholds[idx];
                if has_data && data_len > dedicated_threshold {
                    let blob_id = self.next_blob_id();
                    self.write_dedicated(blob_id, BlobWriteSource::Bytes(data_col.value(i)))
                        .await?;

                    kind_builder.append_value(BlobKind::Dedicated as u8);
                    data_builder.append_null();
                    uri_builder.append_null();
                    blob_id_builder.append_value(blob_id);
                    blob_size_builder.append_value(data_len as u64);
                    position_builder.append_null();
                    continue;
                }

                if has_data && data_len > INLINE_MAX {
                    let (pack_blob_id, position) = self
                        .write_packed(BlobWriteSource::Bytes(data_col.value(i)))
                        .await?;

                    kind_builder.append_value(BlobKind::Packed as u8);
                    data_builder.append_null();
                    uri_builder.append_null();
                    blob_id_builder.append_value(pack_blob_id);
                    blob_size_builder.append_value(data_len as u64);
                    position_builder.append_value(position);
                    continue;
                }

                if has_uri {
                    let uri_val = uri_col.value(i);
                    if self.external_blob_mode == ExternalBlobMode::Ingest {
                        let position = if has_position {
                            Some(
                                position_col
                                    .as_ref()
                                    .expect("position column must exist")
                                    .value(i),
                            )
                        } else {
                            None
                        };
                        let size = if has_size {
                            Some(size_col.as_ref().expect("size column must exist").value(i))
                        } else {
                            None
                        };
                        let source = self.open_external_source(uri_val, position, size).await?;
                        let data_len = source.size();

                        if data_len > dedicated_threshold as u64 {
                            let blob_id = self.next_blob_id();
                            self.write_dedicated(blob_id, BlobWriteSource::External(&source))
                                .await?;

                            kind_builder.append_value(BlobKind::Dedicated as u8);
                            data_builder.append_null();
                            uri_builder.append_null();
                            blob_id_builder.append_value(blob_id);
                            blob_size_builder.append_value(data_len);
                            position_builder.append_null();
                            continue;
                        }

                        if data_len > INLINE_MAX as u64 {
                            let (pack_blob_id, position) = self
                                .write_packed(BlobWriteSource::External(&source))
                                .await?;

                            kind_builder.append_value(BlobKind::Packed as u8);
                            data_builder.append_null();
                            uri_builder.append_null();
                            blob_id_builder.append_value(pack_blob_id);
                            blob_size_builder.append_value(data_len);
                            position_builder.append_value(position);
                            continue;
                        }

                        let data = source.read_all().await?;

                        kind_builder.append_value(BlobKind::Inline as u8);
                        data_builder.append_value(data.as_ref());
                        uri_builder.append_null();
                        blob_id_builder.append_null();
                        blob_size_builder.append_null();
                        position_builder.append_null();
                        continue;
                    }

                    let (external_base_id, external_uri_or_path) =
                        self.resolve_external_reference(uri_val).await?;
                    kind_builder.append_value(BlobKind::External as u8);
                    data_builder.append_null();
                    uri_builder.append_value(external_uri_or_path);
                    blob_id_builder.append_value(external_base_id);
                    if has_position && has_size {
                        let position = position_col
                            .as_ref()
                            .expect("position column must exist")
                            .value(i);
                        let size = size_col.as_ref().expect("size column must exist").value(i);
                        blob_size_builder.append_value(size);
                        position_builder.append_value(position);
                    } else {
                        blob_size_builder.append_null();
                        position_builder.append_null();
                    }
                    continue;
                }

                if has_data {
                    kind_builder.append_value(BlobKind::Inline as u8);
                    let value = data_col.value(i);
                    data_builder.append_value(value);
                    uri_builder.append_null();
                    blob_id_builder.append_null();
                    blob_size_builder.append_null();
                    position_builder.append_null();
                } else {
                    data_builder.append_null();
                    uri_builder.append_null();
                    blob_id_builder.append_null();
                    blob_size_builder.append_null();
                    kind_builder.append_null();
                    position_builder.append_null();
                }
            }

            let child_fields = vec![
                arrow_schema::Field::new("kind", ArrowDataType::UInt8, true),
                arrow_schema::Field::new("data", ArrowDataType::LargeBinary, true),
                arrow_schema::Field::new("uri", ArrowDataType::Utf8, true),
                arrow_schema::Field::new("blob_id", ArrowDataType::UInt32, true),
                arrow_schema::Field::new("blob_size", ArrowDataType::UInt64, true),
                arrow_schema::Field::new("position", ArrowDataType::UInt64, true),
            ];

            let struct_array = arrow_array::StructArray::try_new(
                child_fields.clone().into(),
                vec![
                    Arc::new(kind_builder.finish()),
                    Arc::new(data_builder.finish()),
                    Arc::new(uri_builder.finish()),
                    Arc::new(blob_id_builder.finish()),
                    Arc::new(blob_size_builder.finish()),
                    Arc::new(position_builder.finish()),
                ],
                struct_nulls.cloned(),
            )?;

            new_columns.push(Arc::new(struct_array));
            new_fields.push(Arc::new(
                arrow_schema::Field::new(
                    field.name(),
                    ArrowDataType::Struct(child_fields.into()),
                    field.is_nullable(),
                )
                .with_metadata(self.writer_metadata[idx].clone()),
            ));
        }

        let new_schema = Arc::new(arrow_schema::Schema::new_with_metadata(
            new_fields
                .iter()
                .map(|f| f.as_ref().clone())
                .collect::<Vec<_>>(),
            batch_schema.metadata().clone(),
        ));

        RecordBatch::try_new(new_schema, new_columns)
            .map_err(|e| Error::invalid_input(e.to_string()))
    }

    pub(crate) async fn finish(&mut self) -> Result<()> {
        self.pack_writer.finish().await
    }
}

fn dedicated_threshold_from_metadata(field: &arrow_schema::Field) -> usize {
    field
        .metadata()
        .get(BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY)
        .and_then(|value| value.parse::<i64>().ok())
        .filter(|value| *value > 0)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(DEDICATED_THRESHOLD)
}

pub async fn preprocess_blob_batches(
    batches: &[RecordBatch],
    pre: &mut BlobPreprocessor,
) -> Result<Vec<RecordBatch>> {
    let mut out = Vec::with_capacity(batches.len());
    for batch in batches {
        out.push(pre.preprocess_batch(batch).await?);
    }
    Ok(out)
}

/// Mutable state for a [`BlobFile`] cursor.
///
/// The cursor is logical to the blob slice, not the backing object. Once closed,
/// subsequent cursor-based and range-based reads are rejected, but reads that
/// were already in flight may still complete.
#[derive(Debug)]
enum BlobFileState {
    Open(u64),
    Closed,
}

/// Shared physical read context for blob handles that resolve to the same object.
///
/// Blob descriptors are logical slices over a backing object (data file, packed
/// sidecar, dedicated sidecar, or external object). Multiple [`BlobFile`] values
/// can point at different regions of that same object. This struct gives those
/// handles a single lazy-open scheduler plus a lightweight pending queue so
/// concurrent reads can be opportunistically grouped before reaching Lance's
/// existing I/O scheduler.
#[derive(Debug)]
struct BlobSource {
    object_store: Arc<ObjectStore>,
    path: Path,
    file_size: CachedFileSize,
    scheduler: OnceCell<FileScheduler>,
    pending_reads: Mutex<PendingBlobReads>,
}

impl BlobSource {
    /// Create a shared read context for one physical backing object.
    fn new(object_store: Arc<ObjectStore>, path: Path) -> Self {
        Self {
            object_store,
            path,
            file_size: CachedFileSize::unknown(),
            scheduler: OnceCell::new(),
            pending_reads: Mutex::new(PendingBlobReads::default()),
        }
    }

    /// Read one or more physical ranges from this source.
    ///
    /// Concurrent callers enqueue their requests into `pending_reads`. The first
    /// caller in a drain cycle becomes the leader and spawns the batch drain task.
    /// The mutex critical section only updates in-memory queue bookkeeping; the
    /// actual scheduler submission happens after the lock is released.
    async fn read_ranges(self: &Arc<Self>, ranges: Vec<Range<u64>>) -> Result<Vec<Bytes>> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }

        let scheduler = self
            .scheduler
            .get_or_try_init(|| async {
                ScanScheduler::new(
                    self.object_store.clone(),
                    SchedulerConfig::max_bandwidth(self.object_store.as_ref()),
                )
                .open_file(&self.path, &self.file_size)
                .await
            })
            .await?;

        let (response_tx, response_rx) = oneshot::channel();
        let should_spawn = {
            let mut pending_reads = self.pending_reads.lock().await;
            pending_reads.requests.push(PendingBlobRead {
                ranges,
                response: response_tx,
            });
            if pending_reads.is_draining {
                false
            } else {
                pending_reads.is_draining = true;
                true
            }
        };

        if should_spawn {
            let source = self.clone();
            let scheduler = scheduler.clone();
            tokio::spawn(async move {
                let result = AssertUnwindSafe(source.clone().drain_pending_reads(scheduler))
                    .catch_unwind()
                    .await;
                if let Err(panic) = result {
                    let mut pending_reads = source.pending_reads.lock().await;
                    pending_reads.is_draining = false;
                    std::panic::resume_unwind(panic);
                }
            });
        }

        response_rx.await.map_err(|_| {
            Error::internal("Blob source read task dropped the response".to_string())
        })?
    }

    /// Drain currently queued requests and submit them as scheduler batches.
    ///
    /// Each loop iteration grabs the queued requests with a short mutex hold and
    /// immediately releases the lock before any I/O is awaited.
    async fn drain_pending_reads(self: Arc<Self>, scheduler: FileScheduler) {
        loop {
            let batch = {
                let mut pending_reads = self.pending_reads.lock().await;
                if pending_reads.requests.is_empty() {
                    pending_reads.is_draining = false;
                    return;
                }
                std::mem::take(&mut pending_reads.requests)
            };
            fulfill_pending_blob_reads(&scheduler, batch).await;
        }
    }
}

/// Queue of pending logical blob reads for one [`BlobSource`].
///
/// `is_draining` marks whether a leader task is already draining the queue.
#[derive(Default, Debug)]
struct PendingBlobReads {
    requests: Vec<PendingBlobRead>,
    is_draining: bool,
}

/// Pending logical blob reads waiting to be grouped into one scheduler batch.
///
/// This queue exists only to combine overlapping concurrent calls. The actual
/// coalescing and physical I/O scheduling still happens in [`FileScheduler`].
#[derive(Debug)]
struct PendingBlobRead {
    ranges: Vec<Range<u64>>,
    response: oneshot::Sender<Result<Vec<Bytes>>>,
}

/// Submit one grouped batch of pending blob reads to Lance's [`FileScheduler`].
///
/// The function flattens all logical requests into one range list, preserves the
/// caller-visible order for each request, and fans the bytes back out after the
/// scheduler completes its own merge / split logic.
async fn fulfill_pending_blob_reads(scheduler: &FileScheduler, batch: Vec<PendingBlobRead>) {
    let total_ranges = batch
        .iter()
        .map(|request| request.ranges.len())
        .sum::<usize>();
    let mut request_ranges = Vec::with_capacity(total_ranges);
    let mut response = batch
        .iter()
        .map(|request| vec![Bytes::new(); request.ranges.len()])
        .collect::<Vec<_>>();

    for (request_idx, request) in batch.iter().enumerate() {
        for (range_idx, range) in request.ranges.iter().enumerate() {
            if range.is_empty() {
                continue;
            }
            request_ranges.push((range.clone(), request_idx, range_idx));
        }
    }

    let result = if request_ranges.is_empty() {
        Ok(())
    } else {
        request_ranges.sort_by_key(|(range, _, _)| (range.start, range.end));
        let priority = request_ranges[0].0.start;
        match scheduler
            .submit_request(
                request_ranges
                    .iter()
                    .map(|(range, _, _)| range.clone())
                    .collect::<Vec<_>>(),
                priority,
            )
            .await
        {
            Ok(bytes_vec) => {
                for ((_, request_idx, range_idx), bytes) in
                    request_ranges.into_iter().zip(bytes_vec)
                {
                    response[request_idx][range_idx] = bytes;
                }
                Ok(())
            }
            Err(err) => Err(err),
        }
    };

    match result {
        Ok(()) => {
            for (request, bytes) in batch.into_iter().zip(response) {
                let _ = request.response.send(Ok(bytes));
            }
        }
        Err(err) => {
            let message = format!(
                "Failed to read blob source {}: {}",
                scheduler.reader().path(),
                err
            );
            for request in batch {
                let _ = request.response.send(Err(Error::io(message.clone())));
            }
        }
    }
}

/// Cache key for sharing one [`BlobSource`] across multiple blob descriptors.
///
/// We include the store prefix as well as the path so the same path string in
/// different object stores is never conflated.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct BlobSourceKey {
    store_prefix: String,
    path: String,
}

impl BlobSourceKey {
    /// Build the cache key for one shared [`BlobSource`].
    fn new(source: &BlobSource) -> Self {
        Self {
            store_prefix: source.object_store.store_prefix.clone(),
            path: source.path.to_string(),
        }
    }
}

/// Return a shared [`BlobSource`] for the given physical object.
///
/// This keeps all blob handles that resolve to the same `(store, path)` on a
/// single lazy-open scheduler and pending-read queue.
fn shared_blob_source(
    source_cache: &mut HashMap<BlobSourceKey, Arc<BlobSource>>,
    object_store: Arc<ObjectStore>,
    path: &Path,
) -> Arc<BlobSource> {
    let key = BlobSourceKey {
        store_prefix: object_store.store_prefix.clone(),
        path: path.to_string(),
    };
    source_cache
        .entry(key)
        .or_insert_with(|| Arc::new(BlobSource::new(object_store, path.clone())))
        .clone()
}

/// A file-like object that represents a blob in a dataset
#[derive(Debug)]
pub struct BlobFile {
    source: Arc<BlobSource>,
    state: Arc<Mutex<BlobFileState>>,
    position: u64,
    size: u64,
    kind: BlobKind,
    uri: Option<String>,
}

/// Base-aware physical location metadata used while resolving blob reads.
///
/// This is cached per fragment so repeated rows from the same fragment do not
/// recompute the object store, data directory, and data file key.
#[derive(Clone)]
struct BlobReadLocation {
    object_store: Arc<ObjectStore>,
    data_file_dir: Path,
    data_file_key: String,
    data_file_path: Path,
}

impl BlobFile {
    fn with_source(
        source: Arc<BlobSource>,
        position: u64,
        size: u64,
        kind: BlobKind,
        uri: Option<String>,
    ) -> Self {
        Self {
            source,
            position,
            size,
            kind,
            uri,
            state: Arc::new(Mutex::new(BlobFileState::Open(0))),
        }
    }

    /// Create an inline blob reader backed by a data file.
    ///
    /// This constructor assumes the caller has already resolved multi-base routing
    /// (base-aware object store and file path). It does not inspect dataset metadata.
    ///
    /// # Parameters
    ///
    /// * `object_store` - The store that owns `path`; reads are issued against this store.
    /// * `path` - Full path to the data file containing inline blob bytes.
    /// * `position` - Byte offset of the blob payload inside the data file.
    /// * `size` - Blob payload length in bytes.
    pub fn new_inline(
        object_store: Arc<ObjectStore>,
        path: Path,
        position: u64,
        size: u64,
    ) -> Self {
        Self::with_source(
            Arc::new(BlobSource::new(object_store, path)),
            position,
            size,
            BlobKind::Inline,
            None,
        )
    }

    /// Create a dedicated blob reader backed by a sidecar `.blob` file.
    ///
    /// Dedicated blobs occupy an entire sidecar file, so the logical read starts
    /// at offset `0` and spans `size` bytes.
    ///
    /// # Parameters
    ///
    /// * `object_store` - The store that owns `path`; reads are issued against this store.
    /// * `path` - Full path to the dedicated sidecar blob file.
    /// * `size` - Total byte length to expose from the sidecar file.
    pub fn new_dedicated(object_store: Arc<ObjectStore>, path: Path, size: u64) -> Self {
        Self::with_source(
            Arc::new(BlobSource::new(object_store, path)),
            0,
            size,
            BlobKind::Dedicated,
            None,
        )
    }

    /// Create a packed blob reader for a slice inside a shared sidecar `.blob` file.
    ///
    /// Packed blobs share one sidecar file; this constructor exposes only the
    /// `[position, position + size)` range that belongs to a single row.
    ///
    /// # Parameters
    ///
    /// * `object_store` - The store that owns `path`; reads are issued against this store.
    /// * `path` - Full path to the packed sidecar blob file.
    /// * `position` - Start offset of this blob within the packed sidecar.
    /// * `size` - Blob payload length in bytes.
    pub fn new_packed(
        object_store: Arc<ObjectStore>,
        path: Path,
        position: u64,
        size: u64,
    ) -> Self {
        Self::with_source(
            Arc::new(BlobSource::new(object_store, path)),
            position,
            size,
            BlobKind::Packed,
            None,
        )
    }

    /// Create an external blob reader backed by a caller-resolved object location.
    ///
    /// External blobs are identified by a URI in metadata, but actual reads happen
    /// against a concrete store/path pair resolved by the caller. This keeps URI
    /// resolution (which may be async) outside of the constructor.
    ///
    /// # Parameters
    ///
    /// * `object_store` - The resolved store used to open and read `path`.
    /// * `path` - The resolved object path that contains external blob bytes.
    /// * `uri` - The original URI recorded in blob metadata for round-tripping.
    /// * `position` - Start offset of the blob payload in the external object.
    /// * `size` - Number of bytes exposed from `position`.
    pub fn new_external(
        object_store: Arc<ObjectStore>,
        path: Path,
        uri: String,
        position: u64,
        size: u64,
    ) -> Self {
        Self::with_source(
            Arc::new(BlobSource::new(object_store, path)),
            position,
            size,
            BlobKind::External,
            Some(uri),
        )
    }

    /// Close the blob file, releasing any associated resources
    pub async fn close(&self) -> Result<()> {
        let mut state = self.state.lock().await;
        *state = BlobFileState::Closed;
        Ok(())
    }

    /// Returns true if the blob file is closed
    pub async fn is_closed(&self) -> bool {
        matches!(*self.state.lock().await, BlobFileState::Closed)
    }

    async fn do_with_cursor<T, Fut: Future<Output = Result<(u64, T)>>, Func: FnOnce(u64) -> Fut>(
        &self,
        func: Func,
    ) -> Result<T> {
        let mut state = self.state.lock().await;
        match state.deref_mut() {
            BlobFileState::Open(cursor) => {
                let (new_cursor, data) = func(*cursor).await?;
                *cursor = new_cursor;
                Ok(data)
            }
            BlobFileState::Closed => Err(Error::invalid_input(
                "Blob file is already closed".to_string(),
            )),
        }
    }

    async fn ensure_open(&self) -> Result<()> {
        let state = self.state.lock().await;
        match *state {
            BlobFileState::Open(_) => Ok(()),
            BlobFileState::Closed => Err(Error::invalid_input(
                "Blob file is already closed".to_string(),
            )),
        }
    }

    fn read_phys_range(&self, range: Range<u64>) -> Result<Range<u64>> {
        if range.start > range.end {
            return Err(Error::invalid_input(format!(
                "Blob range start {} must be <= end {}",
                range.start, range.end
            )));
        }
        if range.end > self.size {
            return Err(Error::invalid_input(format!(
                "Blob range end {} exceeds blob size {}",
                range.end, self.size
            )));
        }
        let start = self.position.checked_add(range.start).ok_or_else(|| {
            Error::invalid_input(format!(
                "Blob range start overflowed physical position: base={} offset={}",
                self.position, range.start
            ))
        })?;
        let end = self.position.checked_add(range.end).ok_or_else(|| {
            Error::invalid_input(format!(
                "Blob range end overflowed physical position: base={} offset={}",
                self.position, range.end
            ))
        })?;
        Ok(start..end)
    }

    /// Read a byte range relative to the beginning of this blob without changing the cursor.
    ///
    /// The provided range is interpreted in blob-local coordinates, not object
    /// coordinates. Empty ranges are allowed. This method is intended for random
    /// access callers that want deterministic range semantics instead of the
    /// stateful file-like cursor used by [`Self::read`] and [`Self::read_up_to`].
    pub async fn read_range(&self, range: Range<u64>) -> Result<Bytes> {
        let mut data = self.read_ranges(&[range]).await?;
        Ok(data.pop().unwrap_or_default())
    }

    /// Read multiple ranges relative to the beginning of this blob without changing the cursor.
    ///
    /// Empty ranges are allowed and yield empty buffers. The result order always
    /// matches the input order, even though the underlying physical requests may
    /// be reordered, coalesced, or split for efficiency.
    pub async fn read_ranges(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        self.ensure_open().await?;
        let physical_ranges = ranges
            .iter()
            .cloned()
            .map(|range| self.read_phys_range(range))
            .collect::<Result<Vec<_>>>()?;
        self.source.read_ranges(physical_ranges).await
    }

    /// Read the entire blob file from the current cursor position
    /// to the end of the file
    ///
    /// After this call the cursor will be pointing to the end of
    /// the file.
    pub async fn read(&self) -> Result<bytes::Bytes> {
        let size = self.size;
        let source = self.source.clone();
        let position = self.position;
        self.do_with_cursor(move |cursor| {
            let source = source.clone();
            async move {
                if cursor >= size {
                    return Ok((size, Bytes::new()));
                }
                let physical = (position + cursor)..(position + size);
                Ok((
                    size,
                    source.read_ranges(vec![physical]).await?.pop().unwrap(),
                ))
            }
        })
        .await
    }

    /// Read up to `len` bytes from the current cursor position
    ///
    /// After this call the cursor will be pointing to the end of
    /// the read data.
    pub async fn read_up_to(&self, len: usize) -> Result<bytes::Bytes> {
        let size = self.size;
        let source = self.source.clone();
        let position = self.position;
        self.do_with_cursor(move |cursor| {
            let source = source.clone();
            async move {
                if cursor >= size || len == 0 {
                    return Ok((size.min(cursor), Bytes::new()));
                }
                let read_size = len.min((size - cursor) as usize) as u64;
                let start = position + cursor;
                let end = start + read_size;
                let data = source.read_ranges(vec![start..end]).await?.pop().unwrap();
                Ok((cursor + read_size, data))
            }
        })
        .await
    }

    /// Seek to a new cursor position in the file
    pub async fn seek(&self, new_cursor: u64) -> Result<()> {
        let mut state = self.state.lock().await;
        match state.deref_mut() {
            BlobFileState::Open(cursor) => {
                *cursor = new_cursor;
                Ok(())
            }
            BlobFileState::Closed => Err(Error::invalid_input(
                "Blob file is already closed".to_string(),
            )),
        }
    }

    /// Return the current cursor position in the file
    pub async fn tell(&self) -> Result<u64> {
        let state = self.state.lock().await;
        match *state {
            BlobFileState::Open(cursor) => Ok(cursor),
            BlobFileState::Closed => Err(Error::invalid_input(
                "Blob file is already closed".to_string(),
            )),
        }
    }

    /// Return the size of the blob file in bytes
    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn position(&self) -> u64 {
        self.position
    }

    pub fn data_path(&self) -> &Path {
        &self.source.path
    }

    pub fn kind(&self) -> BlobKind {
        self.kind
    }

    pub fn uri(&self) -> Option<&str> {
        self.uri.as_deref()
    }
}

/// Blob bytes materialized by [`ReadBlobsBuilder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadBlob {
    /// Row address of the blob that was read.
    pub row_address: u64,
    /// Blob payload bytes.
    pub data: Bytes,
}

/// Stream returned by [`ReadBlobsBuilder::try_into_stream`].
pub type ReadBlobsStream = BoxStream<'static, Result<ReadBlob>>;

/// Row selector configured on [`ReadBlobsBuilder`].
#[derive(Debug, Clone)]
enum ReadBlobsSelection {
    None,
    RowIds(Vec<u64>),
    RowIndices(Vec<u64>),
    RowAddresses(Vec<u64>),
}

/// Planner knobs for [`ReadBlobsBuilder`].
///
/// Options that shape how `read_blobs` uses Lance's existing schedulers.
#[derive(Debug, Clone)]
struct ReadBlobsOptions {
    io_buffer_size_bytes: Option<u64>,
    preserve_order: bool,
}

impl Default for ReadBlobsOptions {
    fn default() -> Self {
        Self {
            io_buffer_size_bytes: None,
            preserve_order: true,
        }
    }
}

/// Builder for sequential / planned blob reads.
///
/// Unlike [`Dataset::take_blobs`], which returns [`BlobFile`] handles for
/// caller-driven random access, this builder plans object-store reads across a
/// selected row set and yields fully materialized blob payloads.
#[derive(Debug, Clone)]
pub struct ReadBlobsBuilder {
    dataset: Arc<Dataset>,
    column: String,
    blob_field_id: u32,
    selection: ReadBlobsSelection,
    options: ReadBlobsOptions,
}

impl ReadBlobsBuilder {
    pub(crate) fn new(dataset: Arc<Dataset>, column: String, blob_field_id: u32) -> Self {
        Self {
            dataset,
            column,
            blob_field_id,
            selection: ReadBlobsSelection::None,
            options: ReadBlobsOptions::default(),
        }
    }

    /// Read blobs for the provided stable row ids.
    pub fn with_row_ids(mut self, row_ids: impl Into<Vec<u64>>) -> Self {
        self.selection = ReadBlobsSelection::RowIds(row_ids.into());
        self
    }

    /// Read blobs for the provided row offsets in dataset order.
    pub fn with_row_indices(mut self, row_indices: impl Into<Vec<u64>>) -> Self {
        self.selection = ReadBlobsSelection::RowIndices(row_indices.into());
        self
    }

    /// Read blobs for the provided physical row addresses.
    pub fn with_row_addresses(mut self, row_addrs: impl Into<Vec<u64>>) -> Self {
        self.selection = ReadBlobsSelection::RowAddresses(row_addrs.into());
        self
    }

    /// Set the scheduler I/O buffer size used while materializing blobs.
    pub fn with_io_buffer_size_bytes(mut self, bytes: u64) -> Self {
        self.options.io_buffer_size_bytes = Some(bytes);
        self
    }

    /// Whether results must follow the caller's requested row order.
    pub fn preserve_order(mut self, preserve: bool) -> Self {
        self.options.preserve_order = preserve;
        self
    }

    /// Execute the planned blob read and return a stream of blob payloads.
    ///
    /// The stream yields one [`ReadBlob`] per selected non-null blob row.
    pub async fn try_into_stream(self) -> Result<ReadBlobsStream> {
        self.validate()?;
        let entries = collect_blob_entries_for_selection(
            &self.dataset,
            self.blob_field_id,
            &self.column,
            &self.selection,
        )
        .await?;
        let expected_selection_indices = entries
            .iter()
            .map(|entry| entry.selection_index)
            .collect::<VecDeque<_>>();
        let plans = plan_blob_read_plans(entries);
        let execution = Arc::new(ReadBlobsExecution::new(self.options.io_buffer_size_bytes));
        if plans.is_empty() {
            return Ok(stream::empty().boxed());
        }

        let plan_stream = stream::iter(plans.into_iter().map(move |plan| {
            let execution = execution.clone();
            execute_blob_read_plan(plan, execution)
        }))
        .buffer_unordered(self.dataset.object_store.io_parallelism().max(1));

        if !self.options.preserve_order {
            return Ok(plan_stream
                .map_ok(|blobs| {
                    stream::iter(blobs.into_iter().map(|blob| Ok(into_read_blob(blob))))
                })
                .try_flatten()
                .boxed());
        }

        let mut plan_stream = plan_stream.boxed();
        let mut expected_selection_indices = expected_selection_indices;
        let mut ready = BTreeMap::<usize, ReadBlob>::new();

        Ok(stream::poll_fn(move |cx| {
            loop {
                let Some(next_selection_index) = expected_selection_indices.front().copied() else {
                    return Poll::Ready(None);
                };

                if let Some(blob) = ready.remove(&next_selection_index) {
                    expected_selection_indices.pop_front();
                    return Poll::Ready(Some(Ok(blob)));
                }

                match plan_stream.poll_next_unpin(cx) {
                    Poll::Ready(Some(Ok(blobs))) => {
                        for blob in blobs {
                            ready.insert(blob.selection_index, into_read_blob(blob));
                        }
                    }
                    Poll::Ready(Some(Err(err))) => {
                        return Poll::Ready(Some(Err(err)));
                    }
                    Poll::Ready(None) => {
                        let err = Error::internal(format!(
                            "planned blob read stream completed before selection index {} was produced",
                            next_selection_index
                        ));
                        return Poll::Ready(Some(Err(err)));
                    }
                    Poll::Pending => return Poll::Pending,
                }
            }
        })
        .boxed())
    }

    /// Execute the planned blob read and collect the full result in memory.
    pub async fn execute(self) -> Result<Vec<ReadBlob>> {
        self.try_into_stream().await?.try_collect().await
    }

    fn validate(&self) -> Result<()> {
        match self.selection {
            ReadBlobsSelection::None => Err(Error::invalid_input(
                "ReadBlobsBuilder requires a row selection; call one of with_row_ids, with_row_indices, or with_row_addresses".to_string(),
            )),
            _ if self.options.io_buffer_size_bytes == Some(0) => Err(Error::invalid_input(
                "ReadBlobsBuilder io_buffer_size must be greater than 0".to_string(),
            )),
            _ => Ok(()),
        }
    }
}

/// One logical blob selected for planned reading.
#[derive(Debug)]
struct BlobEntry {
    selection_index: usize,
    row_address: u64,
    file: BlobFile,
}

/// Physical read input derived from one [`BlobEntry`].
#[derive(Debug)]
struct PlannedBlobRead {
    selection_index: usize,
    row_address: u64,
    physical_range: Range<u64>,
}

/// One per-source read plan emitted by `read_blobs`.
#[derive(Debug)]
struct BlobReadPlan {
    source_key: BlobSourceKey,
    source: Arc<BlobSource>,
    reads: Vec<PlannedBlobRead>,
}

/// Operation-scoped scheduler cache for one [`ReadBlobsBuilder`] execution.
///
/// We reuse one [`ScanScheduler`] per object store during a single `read_blobs`
/// operation and still submit exactly one request per physical file.
#[derive(Debug)]
struct ReadBlobsExecution {
    io_buffer_size_bytes: Option<u64>,
    schedulers: std::sync::Mutex<HashMap<String, Arc<ScanScheduler>>>,
}

impl ReadBlobsExecution {
    fn new(io_buffer_size_bytes: Option<u64>) -> Self {
        Self {
            io_buffer_size_bytes,
            schedulers: std::sync::Mutex::new(HashMap::new()),
        }
    }

    fn scheduler_for(&self, source: &BlobSource) -> Arc<ScanScheduler> {
        let mut schedulers = self.schedulers.lock().unwrap();
        schedulers
            .entry(source.object_store.store_prefix.clone())
            .or_insert_with(|| {
                let config = self
                    .io_buffer_size_bytes
                    .map(SchedulerConfig::new)
                    .unwrap_or_else(|| {
                        SchedulerConfig::max_bandwidth(source.object_store.as_ref())
                    });
                ScanScheduler::new(source.object_store.clone(), config)
            })
            .clone()
    }
}

/// Materialized blob bytes plus the original selection index used to restore
/// caller ordering after per-source reads complete.
#[derive(Debug)]
struct IndexedReadBlob {
    selection_index: usize,
    row_address: u64,
    data: Bytes,
}

fn into_read_blob(blob: IndexedReadBlob) -> ReadBlob {
    ReadBlob {
        row_address: blob.row_address,
        data: blob.data,
    }
}

/// Group selected blobs by physical source and sort each group's ranges by
/// physical offset before handing them to the file scheduler.
fn plan_blob_read_plans(entries: Vec<BlobEntry>) -> Vec<BlobReadPlan> {
    let mut plan_indices = HashMap::<BlobSourceKey, usize>::new();
    let mut plans = Vec::<BlobReadPlan>::new();

    for entry in entries {
        let source_key = BlobSourceKey::new(&entry.file.source);
        let plan_index = if let Some(plan_index) = plan_indices.get(&source_key) {
            *plan_index
        } else {
            let plan_index = plans.len();
            plans.push(BlobReadPlan {
                source_key: source_key.clone(),
                source: entry.file.source.clone(),
                reads: Vec::new(),
            });
            plan_indices.insert(source_key.clone(), plan_index);
            plan_index
        };

        plans[plan_index].reads.push(PlannedBlobRead {
            selection_index: entry.selection_index,
            row_address: entry.row_address,
            physical_range: entry.file.position..(entry.file.position + entry.file.size),
        });
    }

    plans.sort_by(|left, right| {
        left.source_key
            .store_prefix
            .cmp(&right.source_key.store_prefix)
            .then_with(|| left.source_key.path.cmp(&right.source_key.path))
    });

    for plan in &mut plans {
        plan.reads.sort_by(|left, right| {
            left.physical_range
                .start
                .cmp(&right.physical_range.start)
                .then_with(|| left.physical_range.end.cmp(&right.physical_range.end))
                .then_with(|| left.selection_index.cmp(&right.selection_index))
        });
    }

    plans
}

/// Execute one per-source blob read plan with a single scheduler submission.
async fn execute_blob_read_plan(
    task: BlobReadPlan,
    execution: Arc<ReadBlobsExecution>,
) -> Result<Vec<IndexedReadBlob>> {
    let ranges = task
        .reads
        .iter()
        .map(|read| read.physical_range.clone())
        .collect::<Vec<_>>();
    let scheduler = execution.scheduler_for(&task.source);
    let file_scheduler = scheduler
        .open_file(&task.source.path, &task.source.file_size)
        .await?;
    let priority = ranges[0].start;
    let bytes = file_scheduler.submit_request(ranges, priority).await?;

    Ok(task
        .reads
        .into_iter()
        .zip(bytes)
        .map(|(read, data)| IndexedReadBlob {
            selection_index: read.selection_index,
            row_address: read.row_address,
            data,
        })
        .collect())
}

pub(super) async fn take_blobs(
    dataset: &Arc<Dataset>,
    row_ids: &[u64],
    column: &str,
) -> Result<Vec<BlobFile>> {
    let blob_field_id = validate_blob_column(dataset, column)?;
    Ok(collect_blob_entries_for_selection(
        dataset,
        blob_field_id,
        column,
        &ReadBlobsSelection::RowIds(row_ids.to_vec()),
    )
    .await?
    .into_iter()
    .map(|entry| entry.file)
    .collect())
}

/// Take [BlobFile] by row addresses.
///
/// Row addresses are `u64` values encoding `(fragment_id << 32) | row_offset`.
/// Use this method when you already have row addresses, for example from
/// a scan with `with_row_address()`. For row IDs (stable identifiers), use
/// [`Dataset::take_blobs`]. For row indices (offsets), use
/// [`Dataset::take_blobs_by_indices`].
pub async fn take_blobs_by_addresses(
    dataset: &Arc<Dataset>,
    row_addrs: &[u64],
    column: &str,
) -> Result<Vec<BlobFile>> {
    let blob_field_id = validate_blob_column(dataset, column)?;
    Ok(collect_blob_entries_for_selection(
        dataset,
        blob_field_id,
        column,
        &ReadBlobsSelection::RowAddresses(row_addrs.to_vec()),
    )
    .await?
    .into_iter()
    .map(|entry| entry.file)
    .collect())
}

/// Validate that `column` exists and is a blob column, returning its field id.
pub(super) fn validate_blob_column(dataset: &Arc<Dataset>, column: &str) -> Result<u32> {
    let projection = dataset.schema().project(&[column])?;
    let blob_field = &projection.fields[0];
    if !blob_field.is_blob() {
        return Err(Error::invalid_input_source(
            format!("the column '{}' is not a blob column", column).into(),
        ));
    }
    Ok(blob_field.id as u32)
}

/// Load blob descriptor rows for a stable-row-id selection.
async fn take_blob_descriptions_by_row_ids(
    dataset: &Arc<Dataset>,
    row_ids: &[u64],
    column: &str,
) -> Result<RecordBatch> {
    let projection = dataset.schema().project(&[column])?;
    dataset
        .take_builder(row_ids, projection)?
        .with_row_address(true)
        .execute()
        .await
}

/// Load blob descriptor rows for a physical-row-address selection.
async fn take_blob_descriptions_by_row_addresses(
    dataset: &Arc<Dataset>,
    row_addrs: &[u64],
    column: &str,
) -> Result<RecordBatch> {
    let projection = dataset.schema().project(&[column])?;
    let projection_request = ProjectionRequest::from(projection);
    let projection_plan = Arc::new(projection_request.into_projection_plan(dataset.clone())?);
    TakeBuilder::try_new_from_addresses(dataset.clone(), row_addrs.to_vec(), projection_plan)?
        .with_row_address(true)
        .execute()
        .await
}

/// Resolve a caller selection into [`BlobEntry`] values that share `BlobSource`
/// instances by physical backing object.
async fn collect_blob_entries_for_selection(
    dataset: &Arc<Dataset>,
    blob_field_id: u32,
    column: &str,
    selection: &ReadBlobsSelection,
) -> Result<Vec<BlobEntry>> {
    let description_and_addr = match selection {
        ReadBlobsSelection::None => {
            return Err(Error::invalid_input(
                "Blob row selection is required".to_string(),
            ));
        }
        ReadBlobsSelection::RowIds(row_ids) => {
            take_blob_descriptions_by_row_ids(dataset, row_ids, column).await?
        }
        ReadBlobsSelection::RowIndices(row_indices) => {
            let row_addrs =
                super::take::row_offsets_to_row_addresses(&dataset.get_fragments(), row_indices)
                    .await?;
            take_blob_descriptions_by_row_addresses(dataset, &row_addrs, column).await?
        }
        ReadBlobsSelection::RowAddresses(row_addrs) => {
            take_blob_descriptions_by_row_addresses(dataset, row_addrs, column).await?
        }
    };

    if description_and_addr.num_rows() == 0 {
        return Ok(Vec::new());
    }

    let descriptions = description_and_addr.column(0).as_struct();
    let row_addrs = description_and_addr.column(1).as_primitive::<UInt64Type>();

    match blob_version_from_descriptions(descriptions)? {
        BlobVersion::V1 => collect_blob_entries_v1(dataset, blob_field_id, descriptions, row_addrs),
        BlobVersion::V2 => {
            collect_blob_entries_v2(dataset, blob_field_id, descriptions, row_addrs).await
        }
    }
}

fn blob_version_from_descriptions(descriptions: &StructArray) -> Result<BlobVersion> {
    let fields = descriptions.fields();
    if fields.len() == 2 && fields[0].name() == "position" && fields[1].name() == "size" {
        return Ok(BlobVersion::V1);
    }
    if fields.len() == 5
        && fields[0].name() == "kind"
        && fields[1].name() == "position"
        && fields[2].name() == "size"
        && fields[3].name() == "blob_id"
        && fields[4].name() == "blob_uri"
    {
        return Ok(BlobVersion::V2);
    }
    Err(Error::invalid_input_source(format!(
        "Unrecognized blob descriptions schema: expected v1 (position,size) or v2 (kind,position,size,blob_id,blob_uri) but got {:?}",
        fields.iter().map(|f| f.name().as_str()).collect::<Vec<_>>(),
    )
    .into()))
}

/// Convert blob v1 descriptors into logical blob entries.
fn collect_blob_entries_v1(
    dataset: &Arc<Dataset>,
    blob_field_id: u32,
    descriptions: &StructArray,
    row_addrs: &arrow::array::PrimitiveArray<UInt64Type>,
) -> Result<Vec<BlobEntry>> {
    let positions = descriptions.column(0).as_primitive::<UInt64Type>();
    let sizes = descriptions.column(1).as_primitive::<UInt64Type>();
    let mut source_cache = HashMap::<BlobSourceKey, Arc<BlobSource>>::new();
    row_addrs
        .values()
        .iter()
        .zip(positions.iter())
        .zip(sizes.iter())
        .enumerate()
        .filter_map(|(selection_index, ((row_addr, position), size))| {
            let position = position?;
            let size = size?;
            Some((selection_index, *row_addr, position, size))
        })
        .map(|(selection_index, row_addr, position, size)| {
            let frag_id = RowAddress::from(row_addr).fragment_id();
            let frag = dataset.get_fragment(frag_id as usize).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Blob row address {} references missing fragment {}",
                    row_addr, frag_id
                ))
            })?;
            let data_file = frag.data_file_for_field(blob_field_id).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Blob field {} has no data file in fragment {} for row address {}",
                    blob_field_id, frag_id, row_addr
                ))
            })?;
            let data_file_path = dataset.data_dir().child(data_file.path.as_str());
            Ok(BlobEntry {
                selection_index,
                row_address: row_addr,
                file: BlobFile::with_source(
                    shared_blob_source(
                        &mut source_cache,
                        dataset.object_store.clone(),
                        &data_file_path,
                    ),
                    position,
                    size,
                    BlobKind::Inline,
                    None,
                ),
            })
        })
        .collect()
}

/// Convert blob v2 descriptors into logical blob entries.
async fn collect_blob_entries_v2(
    dataset: &Arc<Dataset>,
    blob_field_id: u32,
    descriptions: &StructArray,
    row_addrs: &arrow::array::PrimitiveArray<UInt64Type>,
) -> Result<Vec<BlobEntry>> {
    let kinds = descriptions.column(0).as_primitive::<UInt8Type>();
    let positions = descriptions.column(1).as_primitive::<UInt64Type>();
    let sizes = descriptions.column(2).as_primitive::<UInt64Type>();
    let blob_ids = descriptions.column(3).as_primitive::<UInt32Type>();
    let blob_uris = descriptions.column(4).as_string::<i32>();

    let mut files = Vec::with_capacity(row_addrs.len());
    let mut fragment_cache = HashMap::<u32, BlobReadLocation>::new();
    let mut store_cache = HashMap::<u32, Arc<ObjectStore>>::new();
    let mut external_base_path_cache = HashMap::<u32, Path>::new();
    let mut source_cache = HashMap::<BlobSourceKey, Arc<BlobSource>>::new();
    for (selection_index, row_addr) in row_addrs.values().iter().enumerate() {
        let idx = selection_index;
        let kind = BlobKind::try_from(kinds.value(idx))?;

        // Struct is non-nullable; null rows are encoded as inline with zero position/size and empty uri
        if matches!(kind, BlobKind::Inline) && positions.value(idx) == 0 && sizes.value(idx) == 0 {
            continue;
        }

        match kind {
            BlobKind::Inline => {
                let position = positions.value(idx);
                let size = sizes.value(idx);
                let location = resolve_blob_read_location(
                    dataset,
                    blob_field_id,
                    *row_addr,
                    &mut fragment_cache,
                    &mut store_cache,
                )
                .await?;
                let source = shared_blob_source(
                    &mut source_cache,
                    location.object_store,
                    &location.data_file_path,
                );
                files.push(BlobEntry {
                    selection_index,
                    row_address: *row_addr,
                    file: BlobFile::with_source(source, position, size, BlobKind::Inline, None),
                });
            }
            BlobKind::Dedicated => {
                let blob_id = blob_ids.value(idx);
                let size = sizes.value(idx);
                let location = resolve_blob_read_location(
                    dataset,
                    blob_field_id,
                    *row_addr,
                    &mut fragment_cache,
                    &mut store_cache,
                )
                .await?;
                let path = blob_path(&location.data_file_dir, &location.data_file_key, blob_id);
                let source = shared_blob_source(&mut source_cache, location.object_store, &path);
                files.push(BlobEntry {
                    selection_index,
                    row_address: *row_addr,
                    file: BlobFile::with_source(source, 0, size, BlobKind::Dedicated, None),
                });
            }
            BlobKind::Packed => {
                let blob_id = blob_ids.value(idx);
                let size = sizes.value(idx);
                let position = positions.value(idx);
                let location = resolve_blob_read_location(
                    dataset,
                    blob_field_id,
                    *row_addr,
                    &mut fragment_cache,
                    &mut store_cache,
                )
                .await?;
                let path = blob_path(&location.data_file_dir, &location.data_file_key, blob_id);
                let source = shared_blob_source(&mut source_cache, location.object_store, &path);
                files.push(BlobEntry {
                    selection_index,
                    row_address: *row_addr,
                    file: BlobFile::with_source(source, position, size, BlobKind::Packed, None),
                });
            }
            BlobKind::External => {
                let uri_or_path = blob_uris.value(idx).to_string();
                let position = positions.value(idx);
                let size = sizes.value(idx);
                let base_id = blob_ids.value(idx);
                let (object_store, path) = if base_id == 0 {
                    let registry = dataset.session.store_registry();
                    let params = dataset
                        .store_params
                        .as_ref()
                        .map(|p| Arc::new((**p).clone()))
                        .unwrap_or_else(|| Arc::new(ObjectStoreParams::default()));
                    ObjectStore::from_uri_and_params(registry, &uri_or_path, &params).await?
                } else {
                    let object_store = if let Some(store) = store_cache.get(&base_id) {
                        store.clone()
                    } else {
                        let store = dataset.object_store_for_base(base_id).await?;
                        store_cache.insert(base_id, store.clone());
                        store
                    };
                    let base_root = if let Some(path) = external_base_path_cache.get(&base_id) {
                        path.clone()
                    } else {
                        let base = dataset.manifest.base_paths.get(&base_id).ok_or_else(|| {
                            Error::invalid_input(format!(
                                "External blob references unknown base_id {}",
                                base_id
                            ))
                        })?;
                        let path = base.extract_path(dataset.session.store_registry())?;
                        external_base_path_cache.insert(base_id, path.clone());
                        path
                    };
                    let path = join_base_and_relative_path(&base_root, &uri_or_path)?;
                    (object_store, path)
                };
                let size = if size > 0 {
                    size
                } else {
                    object_store.size(&path).await?
                };
                let source = shared_blob_source(&mut source_cache, object_store, &path);
                files.push(BlobEntry {
                    selection_index,
                    row_address: *row_addr,
                    file: BlobFile::with_source(
                        source,
                        position,
                        size,
                        BlobKind::External,
                        Some(uri_or_path),
                    ),
                });
            }
        }
    }

    Ok(files)
}

fn normalize_external_absolute_uri(uri: &str) -> Result<String> {
    let url = Url::parse(uri).map_err(|_| {
        Error::invalid_input(format!(
            "External URI '{}' is outside registered external bases and is not a valid absolute URI",
            uri
        ))
    })?;
    Ok(url.to_string())
}

fn join_base_and_relative_path(base: &Path, relative_path: &str) -> Result<Path> {
    let relative = Path::parse(relative_path).map_err(|e| {
        Error::invalid_input(format!(
            "Invalid relative external blob path '{}': {}",
            relative_path, e
        ))
    })?;
    Ok(Path::from_iter(base.parts().chain(relative.parts())))
}

/// Resolve the physical read location for a blob row in a base-aware way.
///
/// Given a `row_addr`, this helper locates the owning fragment and the blob field's
/// data file, then returns the concrete object store and paths needed to read blob
/// bytes correctly under multi-base datasets.
///
/// It uses two caller-provided caches:
/// - `fragment_cache` memoizes per-fragment path metadata (`data_file_dir`,
///   `data_file_path`, and `data_file_key`) plus the resolved store.
/// - `store_cache` memoizes `base_id -> ObjectStore` so multiple fragments that
///   share the same base do not repeat async store resolution.
async fn resolve_blob_read_location(
    dataset: &Arc<Dataset>,
    blob_field_id: u32,
    row_addr: u64,
    fragment_cache: &mut HashMap<u32, BlobReadLocation>,
    store_cache: &mut HashMap<u32, Arc<ObjectStore>>,
) -> Result<BlobReadLocation> {
    let frag_id = RowAddress::from(row_addr).fragment_id();
    if let Some(location) = fragment_cache.get(&frag_id) {
        return Ok(location.clone());
    }

    let frag = dataset
        .get_fragment(frag_id as usize)
        .ok_or_else(|| Error::internal("Fragment not found".to_string()))?;
    let data_file = frag
        .data_file_for_field(blob_field_id)
        .ok_or_else(|| Error::internal("Data file not found for blob field".to_string()))?;
    let data_file_dir = dataset.data_file_dir(data_file)?;
    let data_file_path = data_file_dir.child(data_file.path.as_str());
    let data_file_key = data_file_key_from_path(data_file.path.as_str()).to_string();

    let object_store = if let Some(base_id) = data_file.base_id {
        if let Some(store) = store_cache.get(&base_id) {
            store.clone()
        } else {
            let store = dataset.object_store_for_base(base_id).await?;
            store_cache.insert(base_id, store.clone());
            store
        }
    } else {
        dataset.object_store.clone()
    };

    let location = BlobReadLocation {
        object_store,
        data_file_dir,
        data_file_key,
        data_file_path,
    };
    fragment_cache.insert(frag_id, location.clone());
    Ok(location)
}

fn data_file_key_from_path(path: &str) -> &str {
    let filename = path.rsplit('/').next().unwrap_or(path);
    filename.strip_suffix(".lance").unwrap_or(filename)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::ops::Range;
    use std::sync::Arc;
    use std::time::Duration;

    use arrow::{
        array::AsArray,
        datatypes::{UInt8Type, UInt32Type, UInt64Type},
    };
    use arrow_array::RecordBatch;
    use arrow_array::{
        ArrayRef, RecordBatchIterator, StringArray, StructArray, UInt32Array, UInt64Array,
    };
    use arrow_schema::{DataType, Field, Schema};
    use async_trait::async_trait;
    use bytes::Bytes;
    use chrono::Utc;
    use futures::{StreamExt, TryStreamExt, future::try_join_all};
    use lance_arrow::{
        ARROW_EXT_NAME_KEY, BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY, BLOB_V2_EXT_NAME, DataTypeExt,
    };
    use lance_core::datatypes::BlobKind;
    use lance_io::object_store::{
        ObjectStore, ObjectStoreParams, ObjectStoreRegistry, StorageOptionsAccessor,
    };
    use lance_io::stream::RecordBatchStream;
    use lance_table::format::BasePath;
    use object_store::{
        Attributes, GetOptions, GetRange, GetResult, GetResultPayload, ListResult, MultipartUpload,
        ObjectMeta, PutMultipartOptions, PutOptions, PutPayload, PutResult, path::Path,
    };
    use tokio::sync::Notify;
    use url::Url;

    use lance_core::{
        Error, Result,
        utils::tempfile::{TempDir, TempStrDir},
    };
    use lance_datagen::{BatchCount, RowCount, array};
    use lance_file::version::LanceFileVersion;

    use super::{
        BlobEntry, BlobFile, BlobSource, ExternalBaseCandidate, ExternalBaseResolver,
        ReadBlobsExecution, collect_blob_entries_v1, data_file_key_from_path,
        execute_blob_read_plan, plan_blob_read_plans,
    };
    use crate::{
        Dataset,
        blob::{BlobArrayBuilder, blob_field},
        dataset::{ExternalBlobMode, WriteParams},
        utils::test::TestDatasetGenerator,
    };

    struct BlobTestFixture {
        _test_dir: TempStrDir,
        dataset: Arc<Dataset>,
        data: Vec<RecordBatch>,
    }

    struct MultiBaseBlobFixture {
        _test_dir: TempDir,
        dataset: Arc<Dataset>,
        expected: Vec<u8>,
    }

    #[cfg(feature = "azure")]
    fn azure_store_params(account_name: &str) -> ObjectStoreParams {
        ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([
                    ("account_name".to_string(), account_name.to_string()),
                    ("account_key".to_string(), "dGVzdA==".to_string()),
                ]),
            ))),
            ..Default::default()
        }
    }

    #[derive(Debug)]
    struct RejectEmptyRangeObjectStore;

    #[cfg(feature = "azure")]
    #[tokio::test]
    async fn test_external_base_resolver_uses_candidate_store_params() {
        let store_registry = Arc::new(ObjectStoreRegistry::default());
        let base_a = BasePath::new(
            1,
            "az://container/path-a".to_string(),
            Some("base-a".to_string()),
            false,
        );
        let base_b = BasePath::new(
            2,
            "az://container/path-b".to_string(),
            Some("base-b".to_string()),
            false,
        );

        let base_a_params = azure_store_params("account-a");
        let base_b_params = azure_store_params("account-b");

        let (store_a, extracted_a) =
            ObjectStore::from_uri_and_params(store_registry.clone(), &base_a.path, &base_a_params)
                .await
                .unwrap();
        let (store_b, extracted_b) =
            ObjectStore::from_uri_and_params(store_registry.clone(), &base_b.path, &base_b_params)
                .await
                .unwrap();

        let resolver = ExternalBaseResolver::new(
            vec![
                ExternalBaseCandidate {
                    base_id: base_a.id,
                    store_prefix: store_a.store_prefix.clone(),
                    base_path: extracted_a,
                    store_params: base_a_params,
                },
                ExternalBaseCandidate {
                    base_id: base_b.id,
                    store_prefix: store_b.store_prefix.clone(),
                    base_path: extracted_b,
                    store_params: base_b_params,
                },
            ],
            store_registry,
        );

        let resolved_a = resolver
            .resolve_external_uri("az://container/path-a/file.bin")
            .await
            .unwrap()
            .unwrap();
        let resolved_b = resolver
            .resolve_external_uri("az://container/path-b/file.bin")
            .await
            .unwrap()
            .unwrap();

        assert_eq!(resolved_a.base_id, 1);
        assert_eq!(resolved_a.relative_path, "file.bin");
        assert_eq!(resolved_b.base_id, 2);
        assert_eq!(resolved_b.relative_path, "file.bin");
    }

    impl std::fmt::Display for RejectEmptyRangeObjectStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "RejectEmptyRangeObjectStore")
        }
    }

    #[async_trait]
    impl object_store::ObjectStore for RejectEmptyRangeObjectStore {
        async fn put(
            &self,
            _location: &Path,
            _bytes: PutPayload,
        ) -> object_store::Result<PutResult> {
            unimplemented!("put is not used by these tests")
        }

        async fn put_opts(
            &self,
            _location: &Path,
            _bytes: PutPayload,
            _opts: PutOptions,
        ) -> object_store::Result<PutResult> {
            unimplemented!("put_opts is not used by these tests")
        }

        async fn put_multipart(
            &self,
            _location: &Path,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            unimplemented!("put_multipart is not used by these tests")
        }

        async fn put_multipart_opts(
            &self,
            _location: &Path,
            _opts: PutMultipartOptions,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            unimplemented!("put_multipart_opts is not used by these tests")
        }

        async fn get(&self, _location: &Path) -> object_store::Result<GetResult> {
            Err(object_store::Error::NotSupported {
                source: "get is not used by these tests".into(),
            })
        }

        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> object_store::Result<GetResult> {
            let Some(GetRange::Bounded(range)) = options.range else {
                unreachable!("blob reads should always request a bounded range")
            };
            if range.start == range.end {
                return Err(object_store::Error::Generic {
                    store: "RejectEmptyRangeObjectStore",
                    source: format!(
                        "Range started at {} and ended at {}",
                        range.start, range.end
                    )
                    .into(),
                });
            }
            Err(object_store::Error::NotSupported {
                source: format!("unexpected non-empty range {range:?} for {location}").into(),
            })
        }

        async fn delete(&self, _location: &Path) -> object_store::Result<()> {
            unimplemented!("delete is not used by these tests")
        }

        fn list(
            &self,
            _prefix: Option<&Path>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<ObjectMeta>> {
            unimplemented!("list is not used by these tests")
        }

        async fn list_with_delimiter(
            &self,
            _prefix: Option<&Path>,
        ) -> object_store::Result<ListResult> {
            unimplemented!("list_with_delimiter is not used by these tests")
        }

        async fn copy(&self, _from: &Path, _to: &Path) -> object_store::Result<()> {
            unimplemented!("copy is not used by these tests")
        }

        async fn copy_if_not_exists(&self, _from: &Path, _to: &Path) -> object_store::Result<()> {
            unimplemented!("copy_if_not_exists is not used by these tests")
        }
    }

    fn reject_empty_range_store() -> Arc<ObjectStore> {
        Arc::new(ObjectStore::new(
            Arc::new(RejectEmptyRangeObjectStore) as Arc<dyn object_store::ObjectStore>,
            Url::parse("mock:///blob-tests").unwrap(),
            None,
            None,
            false,
            true,
            lance_io::object_store::DEFAULT_LOCAL_IO_PARALLELISM,
            lance_io::object_store::DEFAULT_DOWNLOAD_RETRY_COUNT,
            None,
        ))
    }

    #[derive(Debug)]
    struct RecordingRangeObjectStore {
        data: Bytes,
        gate: Option<Arc<Notify>>,
        requested_ranges: std::sync::Mutex<Vec<Range<u64>>>,
    }

    impl RecordingRangeObjectStore {
        fn new(data: Bytes) -> Self {
            Self {
                data,
                gate: None,
                requested_ranges: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn with_gate(data: Bytes, gate: Arc<Notify>) -> Self {
            Self {
                data,
                gate: Some(gate),
                requested_ranges: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn requested_ranges(&self) -> Vec<Range<u64>> {
            self.requested_ranges.lock().unwrap().clone()
        }

        fn object_meta(&self, location: &Path) -> ObjectMeta {
            ObjectMeta {
                location: location.clone(),
                last_modified: Utc::now(),
                size: self.data.len() as u64,
                e_tag: None,
                version: None,
            }
        }
    }

    impl std::fmt::Display for RecordingRangeObjectStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "RecordingRangeObjectStore")
        }
    }

    #[async_trait]
    impl object_store::ObjectStore for RecordingRangeObjectStore {
        async fn put(
            &self,
            _location: &Path,
            _bytes: PutPayload,
        ) -> object_store::Result<PutResult> {
            unimplemented!("put is not used by these tests")
        }

        async fn put_opts(
            &self,
            _location: &Path,
            _bytes: PutPayload,
            _opts: PutOptions,
        ) -> object_store::Result<PutResult> {
            unimplemented!("put_opts is not used by these tests")
        }

        async fn put_multipart(
            &self,
            _location: &Path,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            unimplemented!("put_multipart is not used by these tests")
        }

        async fn put_multipart_opts(
            &self,
            _location: &Path,
            _opts: PutMultipartOptions,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            unimplemented!("put_multipart_opts is not used by these tests")
        }

        async fn get(&self, location: &Path) -> object_store::Result<GetResult> {
            self.get_opts(location, GetOptions::default()).await
        }

        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> object_store::Result<GetResult> {
            let range = match options.range {
                Some(GetRange::Bounded(range)) => range,
                None => 0..self.data.len() as u64,
                Some(other) => {
                    return Err(object_store::Error::NotSupported {
                        source: format!("unsupported range request {other:?}").into(),
                    });
                }
            };
            if let Some(gate) = &self.gate {
                gate.notified().await;
            }
            self.requested_ranges.lock().unwrap().push(range.clone());
            let bytes = self.data.slice(range.start as usize..range.end as usize);
            Ok(GetResult {
                payload: GetResultPayload::Stream(
                    futures::stream::once(async move { Ok(bytes) }).boxed(),
                ),
                meta: self.object_meta(location),
                range,
                attributes: Attributes::default(),
            })
        }

        async fn head(&self, location: &Path) -> object_store::Result<ObjectMeta> {
            Ok(self.object_meta(location))
        }

        async fn delete(&self, _location: &Path) -> object_store::Result<()> {
            unimplemented!("delete is not used by these tests")
        }

        fn list(
            &self,
            _prefix: Option<&Path>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<ObjectMeta>> {
            unimplemented!("list is not used by these tests")
        }

        async fn list_with_delimiter(
            &self,
            _prefix: Option<&Path>,
        ) -> object_store::Result<ListResult> {
            unimplemented!("list_with_delimiter is not used by these tests")
        }

        async fn copy(&self, _from: &Path, _to: &Path) -> object_store::Result<()> {
            unimplemented!("copy is not used by these tests")
        }

        async fn copy_if_not_exists(&self, _from: &Path, _to: &Path) -> object_store::Result<()> {
            unimplemented!("copy_if_not_exists is not used by these tests")
        }
    }

    fn recording_range_store_with_url(
        data: Bytes,
        url: &str,
    ) -> (Arc<ObjectStore>, Arc<RecordingRangeObjectStore>) {
        const TEST_RANGE_STORE_SIZE: usize = 128 * 1024;
        let mut padded = vec![0; TEST_RANGE_STORE_SIZE.max(data.len())];
        padded[..data.len()].copy_from_slice(data.as_ref());
        let inner = Arc::new(RecordingRangeObjectStore::new(Bytes::from(padded)));
        let store = Arc::new(ObjectStore::new(
            inner.clone() as Arc<dyn object_store::ObjectStore>,
            Url::parse(url).unwrap(),
            None,
            None,
            false,
            true,
            lance_io::object_store::DEFAULT_LOCAL_IO_PARALLELISM,
            lance_io::object_store::DEFAULT_DOWNLOAD_RETRY_COUNT,
            None,
        ));
        (store, inner)
    }

    fn recording_range_store(data: Bytes) -> (Arc<ObjectStore>, Arc<RecordingRangeObjectStore>) {
        recording_range_store_with_url(data, "mock://recording/blob-range-tests")
    }

    fn gated_range_store(
        data: Bytes,
        url: &str,
    ) -> (
        Arc<ObjectStore>,
        Arc<RecordingRangeObjectStore>,
        Arc<Notify>,
    ) {
        const TEST_RANGE_STORE_SIZE: usize = 128 * 1024;
        let mut padded = vec![0; TEST_RANGE_STORE_SIZE.max(data.len())];
        padded[..data.len()].copy_from_slice(data.as_ref());
        let gate = Arc::new(Notify::new());
        let inner = Arc::new(RecordingRangeObjectStore::with_gate(
            Bytes::from(padded),
            gate.clone(),
        ));
        let store = Arc::new(ObjectStore::new(
            inner.clone() as Arc<dyn object_store::ObjectStore>,
            Url::parse(url).unwrap(),
            None,
            None,
            false,
            true,
            lance_io::object_store::DEFAULT_LOCAL_IO_PARALLELISM,
            lance_io::object_store::DEFAULT_DOWNLOAD_RETRY_COUNT,
            None,
        ));
        (store, inner, gate)
    }

    impl BlobTestFixture {
        async fn new() -> Self {
            let test_dir = TempStrDir::default();

            let data = lance_datagen::gen_batch()
                .col("filterme", array::step::<UInt64Type>())
                .col("blobs", array::blob())
                .into_reader_rows(RowCount::from(10), BatchCount::from(10))
                .map(|batch| Ok(batch?))
                .collect::<Result<Vec<_>>>()
                .unwrap();

            let dataset = Arc::new(
                TestDatasetGenerator::new(data.clone(), LanceFileVersion::default())
                    .make_hostile(&test_dir)
                    .await,
            );

            Self {
                _test_dir: test_dir,
                dataset,
                data,
            }
        }
    }

    async fn create_multi_base_blob_v2_fixture(
        payload: Vec<u8>,
        dedicated_threshold: Option<usize>,
        is_dataset_root: bool,
    ) -> MultiBaseBlobFixture {
        let test_dir = TempDir::default();
        let primary_uri = test_dir.path_str();
        let base_dir = test_dir.std_path().join("blob_base");
        std::fs::create_dir_all(&base_dir).unwrap();
        let base_uri = format!("file://{}", base_dir.display());

        let mut blob_builder = BlobArrayBuilder::new(1);
        blob_builder.push_bytes(payload.clone()).unwrap();
        let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();

        let mut blob_column = blob_field("blob", true);
        if let Some(threshold) = dedicated_threshold {
            let mut metadata = blob_column.metadata().clone();
            metadata.insert(
                BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.to_string(),
                threshold.to_string(),
            );
            blob_column = blob_column.with_metadata(metadata);
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            blob_column,
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(UInt32Array::from(vec![0])), blob_array],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        let dataset = Arc::new(
            Dataset::write(
                reader,
                &primary_uri,
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    initial_bases: Some(vec![BasePath {
                        id: 1,
                        name: Some("blob_base".to_string()),
                        path: base_uri,
                        is_dataset_root,
                    }]),
                    target_bases: Some(vec![1]),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        assert!(
            dataset
                .fragments()
                .iter()
                .all(|frag| frag.files.iter().all(|file| file.base_id == Some(1)))
        );

        MultiBaseBlobFixture {
            _test_dir: test_dir,
            dataset,
            expected: payload,
        }
    }

    #[tokio::test]
    pub async fn test_take_blobs() {
        let fixture = BlobTestFixture::new().await;

        let row_ids = fixture
            .dataset
            .scan()
            .project::<String>(&[])
            .unwrap()
            .filter("filterme >= 50")
            .unwrap()
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();
        let row_ids = row_ids.column(0).as_primitive::<UInt64Type>().values();
        let row_ids = vec![row_ids[5], row_ids[17], row_ids[33]];

        let blobs = fixture.dataset.take_blobs(&row_ids, "blobs").await.unwrap();

        for (actual_idx, (expected_batch_idx, expected_row_idx)) in
            [(5, 5), (6, 7), (8, 3)].iter().enumerate()
        {
            let val = blobs[actual_idx].read().await.unwrap();
            let expected = fixture.data[*expected_batch_idx]
                .column(1)
                .as_binary::<i64>()
                .value(*expected_row_idx);

            assert_eq!(&val, expected);
        }
    }

    #[tokio::test]
    async fn test_read_blobs_requires_selection() {
        let fixture = BlobTestFixture::new().await;

        let err = fixture.dataset.read_blobs("blobs").unwrap().execute().await;

        assert!(matches!(err, Err(Error::InvalidInput { .. })));
        assert!(
            err.unwrap_err()
                .to_string()
                .contains("requires a row selection")
        );
    }

    #[tokio::test]
    async fn test_read_blobs_by_indices_execute() {
        let fixture = BlobTestFixture::new().await;
        let indices = vec![2, 12, 22];

        let blobs = fixture
            .dataset
            .read_blobs("blobs")
            .unwrap()
            .with_row_indices(indices)
            .execute()
            .await
            .unwrap();

        assert_eq!(blobs.len(), 3);
        for (actual_idx, (expected_batch_idx, expected_row_idx)) in
            [(0, 2), (1, 2), (2, 2)].iter().enumerate()
        {
            let expected = fixture.data[*expected_batch_idx]
                .column(1)
                .as_binary::<i64>()
                .value(*expected_row_idx);
            assert_eq!(blobs[actual_idx].data.as_ref(), expected);
        }
    }

    #[tokio::test]
    pub async fn test_take_blobs_by_indices() {
        let fixture = BlobTestFixture::new().await;

        let fragments = fixture.dataset.fragments();
        assert!(fragments.len() >= 2);
        let mut indices = Vec::with_capacity(fragments.len());
        let mut last = 2;

        for frag in fragments.iter() {
            indices.push(last as u64);
            last += frag.num_rows().unwrap_or(0);
        }
        indices.pop();

        // Row indices
        assert_eq!(indices, [2, 12, 22, 32, 42, 52, 62, 72, 82]);
        let blobs = fixture
            .dataset
            .take_blobs_by_indices(&indices, "blobs")
            .await
            .unwrap();

        // Row IDs
        let row_ids = fragments
            .iter()
            .map(|frag| (frag.id << 32) + 2)
            .collect::<Vec<_>>();
        let blobs2 = fixture.dataset.take_blobs(&row_ids, "blobs").await.unwrap();

        for (blob1, blob2) in blobs.iter().zip(blobs2.iter()) {
            assert_eq!(blob1.position(), blob2.position());
            assert_eq!(blob1.size(), blob2.size());
            assert_eq!(blob1.data_path(), blob2.data_path());
        }
    }

    #[tokio::test]
    pub async fn test_take_blob_id_not_exist() {
        let fixture = BlobTestFixture::new().await;

        let err = fixture.dataset.take_blobs(&[1000], "blobs").await;

        assert!(matches!(err, Err(Error::InvalidInput { .. })));
    }

    #[tokio::test]
    async fn test_collect_blob_entries_v1_rejects_missing_fragment() {
        let fixture = BlobTestFixture::new().await;
        let blob_field_id =
            fixture.dataset.schema().project(&["blobs"]).unwrap().fields[0].id as u32;
        let descriptions = StructArray::from(vec![
            (
                Arc::new(Field::new("position", DataType::UInt64, false)),
                Arc::new(UInt64Array::from(vec![1])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("size", DataType::UInt64, false)),
                Arc::new(UInt64Array::from(vec![3])) as ArrayRef,
            ),
        ]);
        let row_addrs = UInt64Array::from(vec![(999_u64 << 32) | 7]);

        let err =
            collect_blob_entries_v1(&fixture.dataset, blob_field_id, &descriptions, &row_addrs)
                .unwrap_err();

        assert!(err.to_string().contains("references missing fragment"));
    }

    #[tokio::test]
    pub async fn test_take_blob_not_blob_col() {
        let fixture = BlobTestFixture::new().await;

        let err = fixture.dataset.take_blobs(&[0], "filterme").await;

        assert!(matches!(err, Err(Error::InvalidInput { .. })));
        assert!(err.unwrap_err().to_string().contains("not a blob column"));
    }

    #[tokio::test]
    pub async fn test_scan_blobs() {
        let fixture = BlobTestFixture::new().await;

        // By default, scanning a blob column will load descriptions
        let batches = fixture
            .dataset
            .scan()
            .project(&["blobs"])
            .unwrap()
            .try_into_stream()
            .await
            .unwrap();

        let schema = batches.schema();

        assert!(schema.fields[0].data_type().is_struct());

        let batches = batches.try_collect::<Vec<_>>().await.unwrap();

        assert_eq!(batches.len(), 10);
        for batch in batches.iter() {
            assert_eq!(batch.num_columns(), 1);
            assert!(batch.column(0).data_type().is_struct());
        }

        // Should also be able to scan with filter
        let batches = fixture
            .dataset
            .scan()
            .project(&["blobs"])
            .unwrap()
            .filter("filterme = 50")
            .unwrap()
            .try_into_stream()
            .await
            .unwrap();

        let schema = batches.schema();

        assert!(schema.fields[0].data_type().is_struct());

        let batches = batches.try_collect::<Vec<_>>().await.unwrap();

        assert_eq!(batches.len(), 1);
        for batch in batches.iter() {
            assert_eq!(batch.num_columns(), 1);
            assert!(batch.column(0).data_type().is_struct());
        }
    }

    /// Test that take_blobs_by_indices works correctly with enable_stable_row_ids=true.
    ///
    /// This is a regression test for a bug where take_blobs_by_indices would fail
    /// with "index out of bounds" for fragment 1+ when stable row IDs are enabled.
    /// The bug was caused by passing row addresses (from row_offsets_to_row_addresses)
    /// to blob::take_blobs which expected row IDs. When stable row IDs are enabled,
    /// row addresses (fragment_id << 32 | offset) are different from row IDs
    /// (sequential integers), causing the row ID index lookup to fail for fragment 1+.
    #[tokio::test]
    pub async fn test_take_blobs_by_indices_with_stable_row_ids() {
        use crate::dataset::WriteParams;
        use arrow_array::RecordBatchIterator;

        let test_dir = TempStrDir::default();

        // Create test data with blob column
        let data = lance_datagen::gen_batch()
            .col("filterme", array::step::<UInt64Type>())
            .col("blobs", array::blob())
            .into_reader_rows(RowCount::from(6), BatchCount::from(1))
            .map(|batch| Ok(batch.unwrap()))
            .collect::<Result<Vec<_>>>()
            .unwrap();

        // Write with enable_stable_row_ids=true and force multiple fragments
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 3, // Force 2 fragments with 3 rows each
            ..Default::default()
        };

        let reader = RecordBatchIterator::new(data.clone().into_iter().map(Ok), data[0].schema());
        let dataset = Arc::new(
            Dataset::write(reader, &test_dir, Some(write_params))
                .await
                .unwrap(),
        );

        // Verify we have multiple fragments
        let fragments = dataset.fragments();
        assert!(
            fragments.len() >= 2,
            "Expected at least 2 fragments, got {}",
            fragments.len()
        );

        // Test first fragment (indices 0, 1, 2) - this always worked
        let blobs = dataset
            .take_blobs_by_indices(&[0, 1, 2], "blobs")
            .await
            .unwrap();
        assert_eq!(blobs.len(), 3, "First fragment blobs should have 3 items");

        // Verify we can read the blob content
        for blob in &blobs {
            let content = blob.read().await.unwrap();
            assert!(!content.is_empty(), "Blob content should not be empty");
        }

        // Test second fragment (indices 3, 4, 5) - this was failing before the fix
        let blobs = dataset
            .take_blobs_by_indices(&[3, 4, 5], "blobs")
            .await
            .unwrap();
        assert_eq!(blobs.len(), 3, "Second fragment blobs should have 3 items");

        // Verify we can read the blob content from second fragment
        for blob in &blobs {
            let content = blob.read().await.unwrap();
            assert!(!content.is_empty(), "Blob content should not be empty");
        }

        // Test mixed indices from both fragments
        let blobs = dataset
            .take_blobs_by_indices(&[1, 4], "blobs")
            .await
            .unwrap();
        assert_eq!(blobs.len(), 2, "Mixed fragment blobs should have 2 items");
    }

    #[test]
    fn test_data_file_key_from_path() {
        assert_eq!(data_file_key_from_path("data/abc.lance"), "abc");
        assert_eq!(data_file_key_from_path("abc.lance"), "abc");
        assert_eq!(data_file_key_from_path("nested/path/xyz"), "xyz");
    }

    #[tokio::test]
    async fn test_write_and_take_blobs_with_blob_array_builder() {
        let test_dir = TempStrDir::default();

        // Build a blob column with the new BlobArrayBuilder
        let mut blob_builder = BlobArrayBuilder::new(2);
        blob_builder.push_bytes(b"hello").unwrap();
        blob_builder.push_bytes(b"world").unwrap();
        let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();

        let id_array: arrow_array::ArrayRef = Arc::new(UInt32Array::from(vec![0, 1]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            blob_field("blob", true),
        ]));

        let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

        let params = WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        };
        let dataset = Arc::new(
            Dataset::write(reader, &test_dir, Some(params))
                .await
                .unwrap(),
        );

        let blobs = dataset
            .take_blobs_by_indices(&[0, 1], "blob")
            .await
            .unwrap();

        assert_eq!(blobs.len(), 2);
        let first = blobs[0].read().await.unwrap();
        let second = blobs[1].read().await.unwrap();
        assert_eq!(first.as_ref(), b"hello");
        assert_eq!(second.as_ref(), b"world");
    }

    #[tokio::test]
    async fn test_blob_file_read_empty_range_returns_empty_bytes() {
        let store = reject_empty_range_store();
        let path = Path::from("blobs/test.bin");

        let empty_blob = BlobFile::new_packed(store.clone(), path.clone(), 1, 0);
        assert!(empty_blob.read().await.unwrap().is_empty());
        assert!(empty_blob.read_up_to(16).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_blob_file_read_tracks_relative_cursor() {
        let test_dir = TempDir::default();
        let file_path = test_dir.std_path().join("blob.bin");
        std::fs::write(&file_path, b"abcd").unwrap();

        let path = Path::from_absolute_path(file_path).unwrap();
        let blob = BlobFile::new_packed(Arc::new(ObjectStore::local()), path, 1, 2);

        assert_eq!(blob.read().await.unwrap().as_ref(), b"bc");
        assert_eq!(blob.tell().await.unwrap(), 2);
        assert!(blob.read().await.unwrap().is_empty());
        assert!(blob.read_up_to(1).await.unwrap().is_empty());
        assert_eq!(blob.tell().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn test_blob_file_read_range_does_not_change_cursor() {
        let (store, _) = recording_range_store(Bytes::from_static(b"abcdefgh"));
        let path = Path::from("blobs/test.bin");
        let blob = BlobFile::new_packed(store, path, 1, 6);

        let bytes = blob.read_range(2..5).await.unwrap();
        assert_eq!(bytes.as_ref(), b"def");
        assert_eq!(blob.tell().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_blob_file_read_ranges_preserves_input_order() {
        let (store, inner) = recording_range_store(Bytes::from_static(b"abcdefghij"));
        let path = Path::from("blobs/test.bin");
        let blob = BlobFile::new_packed(store, path, 1, 6);

        let chunks = blob.read_ranges(&[4..6, 0..2, 2..4, 2..2]).await.unwrap();
        assert_eq!(chunks[0].as_ref(), b"fg");
        assert_eq!(chunks[1].as_ref(), b"bc");
        assert_eq!(chunks[2].as_ref(), b"de");
        assert!(chunks[3].is_empty());
        assert_eq!(inner.requested_ranges(), vec![1..7]);
    }

    #[tokio::test]
    async fn test_blob_file_read_range_rejects_out_of_bounds() {
        let (store, _) = recording_range_store(Bytes::from_static(b"abcdef"));
        let path = Path::from("blobs/test.bin");
        let blob = BlobFile::new_packed(store, path, 0, 4);

        let err = blob.read_range(1..5).await.unwrap_err();
        assert!(err.to_string().contains("exceeds blob size"));
    }

    #[tokio::test]
    async fn test_blob_files_share_source_and_coalesce() {
        let (store, inner) = recording_range_store(Bytes::from_static(b"abcdefghij"));
        let source = Arc::new(BlobSource::new(store, Path::from("blobs/test.bin")));
        let blob1 = BlobFile::with_source(source.clone(), 1, 3, BlobKind::Packed, None);
        let blob2 = BlobFile::with_source(source, 4, 3, BlobKind::Packed, None);

        let (data1, data2) = tokio::join!(blob1.read(), blob2.read());
        assert_eq!(data1.unwrap().as_ref(), b"bcd");
        assert_eq!(data2.unwrap().as_ref(), b"efg");
        assert_eq!(inner.requested_ranges(), vec![1..7]);
    }

    #[tokio::test]
    async fn test_read_blobs_plan_preserves_order_and_coalesces() {
        let (store, inner) = recording_range_store(Bytes::from_static(b"abcdefghij"));
        let source = Arc::new(BlobSource::new(store, Path::from("blobs/test.bin")));
        let entries = vec![
            BlobEntry {
                selection_index: 0,
                row_address: 10,
                file: BlobFile::with_source(source.clone(), 4, 3, BlobKind::Packed, None),
            },
            BlobEntry {
                selection_index: 1,
                row_address: 11,
                file: BlobFile::with_source(source, 1, 3, BlobKind::Packed, None),
            },
        ];
        let execution = Arc::new(ReadBlobsExecution::new(None));
        let blobs = try_join_all(
            plan_blob_read_plans(entries)
                .into_iter()
                .map(|plan| execute_blob_read_plan(plan, execution.clone())),
        )
        .await
        .unwrap();
        let mut blobs = blobs.into_iter().flatten().collect::<Vec<_>>();
        blobs.sort_by_key(|blob| blob.selection_index);

        assert_eq!(blobs.len(), 2);
        assert_eq!(blobs[0].row_address, 10);
        assert_eq!(blobs[0].data.as_ref(), b"efg");
        assert_eq!(blobs[1].row_address, 11);
        assert_eq!(blobs[1].data.as_ref(), b"bcd");
        assert_eq!(inner.requested_ranges(), vec![1..7]);
    }

    #[tokio::test]
    async fn test_read_blobs_stream_emits_ready_plan_without_waiting_for_slower_ones() {
        let (slow_store, _, slow_gate) = gated_range_store(
            Bytes::from_static(b"abcdef"),
            "mock://slow/blob-range-tests",
        );
        let (fast_store, _) = recording_range_store_with_url(
            Bytes::from_static(b"uvwxyz"),
            "mock://fast/blob-range-tests",
        );
        let entries = vec![
            BlobEntry {
                selection_index: 0,
                row_address: 10,
                file: BlobFile::with_source(
                    Arc::new(BlobSource::new(slow_store, Path::from("blobs/slow.bin"))),
                    0,
                    3,
                    BlobKind::Packed,
                    None,
                ),
            },
            BlobEntry {
                selection_index: 1,
                row_address: 11,
                file: BlobFile::with_source(
                    Arc::new(BlobSource::new(fast_store, Path::from("blobs/fast.bin"))),
                    0,
                    3,
                    BlobKind::Packed,
                    None,
                ),
            },
        ];
        let execution = Arc::new(ReadBlobsExecution::new(None));
        let mut stream: super::ReadBlobsStream = futures::stream::iter(
            plan_blob_read_plans(entries)
                .into_iter()
                .map(move |plan| execute_blob_read_plan(plan, execution.clone())),
        )
        .buffer_unordered(2)
        .map_ok(|blobs: Vec<super::IndexedReadBlob>| {
            futures::stream::iter(
                blobs
                    .into_iter()
                    .map(|blob| Ok::<super::ReadBlob, Error>(super::into_read_blob(blob))),
            )
        })
        .try_flatten()
        .boxed();

        let first = tokio::time::timeout(Duration::from_secs(1), stream.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(first.row_address, 11);
        assert_eq!(first.data.as_ref(), b"uvw");

        slow_gate.notify_one();
        let second = tokio::time::timeout(Duration::from_secs(1), stream.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(second.row_address, 10);
        assert_eq!(second.data.as_ref(), b"abc");
    }

    #[tokio::test]
    async fn test_take_blob_v2_from_non_default_base_inline() {
        let fixture = create_multi_base_blob_v2_fixture(b"inline".to_vec(), None, true).await;

        let blobs = fixture
            .dataset
            .take_blobs_by_indices(&[0], "blob")
            .await
            .unwrap();

        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs[0].kind(), BlobKind::Inline);
        assert_eq!(
            blobs[0].read().await.unwrap().as_ref(),
            fixture.expected.as_slice()
        );
    }

    #[tokio::test]
    async fn test_take_blob_v2_from_non_default_base_packed() {
        let fixture =
            create_multi_base_blob_v2_fixture(vec![0x5A; super::INLINE_MAX + 4096], None, true)
                .await;

        let blobs = fixture
            .dataset
            .take_blobs_by_indices(&[0], "blob")
            .await
            .unwrap();

        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs[0].kind(), BlobKind::Packed);
        assert_eq!(
            blobs[0].read().await.unwrap().as_ref(),
            fixture.expected.as_slice()
        );
    }

    #[tokio::test]
    async fn test_take_blob_v2_from_non_default_base_dedicated() {
        let fixture = create_multi_base_blob_v2_fixture(vec![0xA5; 4096], Some(1), true).await;

        let blobs = fixture
            .dataset
            .take_blobs_by_indices(&[0], "blob")
            .await
            .unwrap();

        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs[0].kind(), BlobKind::Dedicated);
        assert_eq!(
            blobs[0].read().await.unwrap().as_ref(),
            fixture.expected.as_slice()
        );
    }

    #[tokio::test]
    async fn test_take_blob_v2_from_data_only_base() {
        let fixture =
            create_multi_base_blob_v2_fixture(vec![0x6B; super::INLINE_MAX + 2048], None, false)
                .await;

        let blobs = fixture
            .dataset
            .take_blobs_by_indices(&[0], "blob")
            .await
            .unwrap();

        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs[0].kind(), BlobKind::Packed);
        assert_eq!(
            blobs[0].read().await.unwrap().as_ref(),
            fixture.expected.as_slice()
        );
    }

    #[tokio::test]
    async fn test_blob_v2_external_outside_base_denied_by_default() {
        let dataset_dir = TempDir::default();
        let external_dir = TempDir::default();
        let external_path = external_dir.std_path().join("external.bin");
        std::fs::write(&external_path, b"outside").unwrap();
        let external_uri = format!("file://{}", external_path.display());

        let mut blob_builder = BlobArrayBuilder::new(1);
        blob_builder.push_uri(external_uri).unwrap();
        let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();
        let schema = Arc::new(Schema::new(vec![blob_field("blob", true)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        let result = Dataset::write(
            reader,
            &dataset_dir.path_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await;

        let err = result.unwrap_err();
        assert!(
            err.to_string()
                .contains("outside registered external bases"),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn test_blob_v2_external_under_dataset_root_denied_by_default() {
        let test_dir = TempDir::default();
        let dataset_path = test_dir.std_path().join("dataset");
        std::fs::create_dir_all(dataset_path.join("media")).unwrap();
        let external_path = dataset_path.join("media").join("external.bin");
        std::fs::write(&external_path, b"root-local").unwrap();
        let external_uri = format!("file://{}", external_path.display());

        let mut blob_builder = BlobArrayBuilder::new(1);
        blob_builder.push_uri(external_uri).unwrap();
        let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();
        let schema = Arc::new(Schema::new(vec![blob_field("blob", true)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        let result = Dataset::write(
            reader,
            dataset_path.to_str().unwrap(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await;

        let err = result.unwrap_err();
        assert!(
            err.to_string()
                .contains("outside registered external bases"),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn test_blob_v2_external_outside_base_allowed() {
        let dataset_dir = TempDir::default();
        let external_dir = TempDir::default();
        let external_path = external_dir.std_path().join("external.bin");
        std::fs::write(&external_path, b"outside").unwrap();
        let external_uri = format!("file://{}", external_path.display());

        let mut blob_builder = BlobArrayBuilder::new(1);
        blob_builder.push_uri(external_uri.clone()).unwrap();
        let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();
        let schema = Arc::new(Schema::new(vec![blob_field("blob", true)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        let dataset = Arc::new(
            Dataset::write(
                reader,
                &dataset_dir.path_str(),
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    allow_external_blob_outside_bases: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let desc = dataset
            .scan()
            .project(&["blob"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
            .column(0)
            .as_struct()
            .to_owned();
        assert_eq!(
            desc.column(0).as_primitive::<UInt8Type>().value(0),
            BlobKind::External as u8
        );
        assert_eq!(desc.column(3).as_primitive::<UInt32Type>().value(0), 0);
        let expected_uri = super::normalize_external_absolute_uri(&external_uri).unwrap();
        assert_eq!(desc.column(4).as_string::<i32>().value(0), expected_uri);

        let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs[0].read().await.unwrap().as_ref(), b"outside");
    }

    #[tokio::test]
    async fn test_blob_v2_external_mapped_to_registered_base() {
        let test_dir = TempDir::default();
        let dataset_uri = test_dir.std_path().join("dataset");
        let external_base = test_dir.std_path().join("external_base");
        let external_obj_dir = external_base.join("objects");
        std::fs::create_dir_all(&external_obj_dir).unwrap();
        let external_path = external_obj_dir.join("mapped.bin");
        std::fs::write(&external_path, b"mapped").unwrap();
        let external_uri = format!("file://{}", external_path.display());
        let base_uri = format!("file://{}", external_base.display());

        let mut blob_builder = BlobArrayBuilder::new(1);
        blob_builder.push_uri(external_uri).unwrap();
        let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();
        let schema = Arc::new(Schema::new(vec![blob_field("blob", true)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        let dataset = Arc::new(
            Dataset::write(
                reader,
                dataset_uri.to_str().unwrap(),
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    initial_bases: Some(vec![BasePath {
                        id: 1,
                        name: Some("external".to_string()),
                        path: base_uri,
                        is_dataset_root: false,
                    }]),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let desc = dataset
            .scan()
            .project(&["blob"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
            .column(0)
            .as_struct()
            .to_owned();
        assert_eq!(
            desc.column(0).as_primitive::<UInt8Type>().value(0),
            BlobKind::External as u8
        );
        assert_eq!(desc.column(3).as_primitive::<UInt32Type>().value(0), 1);
        assert_eq!(
            desc.column(4).as_string::<i32>().value(0),
            "objects/mapped.bin"
        );

        let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs[0].read().await.unwrap().as_ref(), b"mapped");
    }

    #[tokio::test]
    async fn test_blob_v2_external_ingest_inline_slice() {
        let dataset_dir = TempDir::default();
        let external_dir = TempDir::default();
        let external_path = external_dir.std_path().join("external.bin");
        std::fs::write(&external_path, b"prefix-inline-suffix").unwrap();
        let external_uri = format!("file://{}", external_path.display());

        let metadata = [(ARROW_EXT_NAME_KEY.to_string(), BLOB_V2_EXT_NAME.to_string())]
            .into_iter()
            .collect();
        let blob_field = Field::new(
            "blob",
            DataType::Struct(
                vec![
                    Field::new("data", DataType::LargeBinary, true),
                    Field::new("uri", DataType::Utf8, true),
                    Field::new("position", DataType::UInt64, true),
                    Field::new("size", DataType::UInt64, true),
                ]
                .into(),
            ),
            true,
        )
        .with_metadata(metadata);
        let blob_array: ArrayRef = Arc::new(
            StructArray::try_new(
                vec![
                    Field::new("data", DataType::LargeBinary, true),
                    Field::new("uri", DataType::Utf8, true),
                    Field::new("position", DataType::UInt64, true),
                    Field::new("size", DataType::UInt64, true),
                ]
                .into(),
                vec![
                    Arc::new(arrow_array::LargeBinaryArray::from(vec![None::<&[u8]>])) as ArrayRef,
                    Arc::new(StringArray::from(vec![Some(external_uri.as_str())])) as ArrayRef,
                    Arc::new(UInt64Array::from(vec![Some(7)])) as ArrayRef,
                    Arc::new(UInt64Array::from(vec![Some(6)])) as ArrayRef,
                ],
                None,
            )
            .unwrap(),
        );
        let schema = Arc::new(Schema::new(vec![blob_field]));
        let batch = RecordBatch::try_new(schema.clone(), vec![blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        let dataset = Arc::new(
            Dataset::write(
                reader,
                &dataset_dir.path_str(),
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    external_blob_mode: ExternalBlobMode::Ingest,
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let desc = dataset
            .scan()
            .project(&["blob"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
            .column(0)
            .as_struct()
            .to_owned();
        assert_eq!(
            desc.column_by_name("kind")
                .unwrap()
                .as_primitive::<UInt8Type>()
                .value(0),
            BlobKind::Inline as u8
        );

        let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs[0].kind(), BlobKind::Inline);
        assert_eq!(blobs[0].read().await.unwrap().as_ref(), b"inline");
    }

    #[tokio::test]
    async fn test_blob_v2_external_ingest_packed() {
        let dataset_dir = TempDir::default();
        let external_dir = TempDir::default();
        let external_path = external_dir.std_path().join("external.bin");
        let payload = vec![0x5A; super::INLINE_MAX + 1024];
        std::fs::write(&external_path, &payload).unwrap();
        let external_uri = format!("file://{}", external_path.display());

        let mut blob_builder = BlobArrayBuilder::new(1);
        blob_builder.push_uri(external_uri).unwrap();
        let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();
        let schema = Arc::new(Schema::new(vec![blob_field("blob", true)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        let dataset = Arc::new(
            Dataset::write(
                reader,
                &dataset_dir.path_str(),
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    external_blob_mode: ExternalBlobMode::Ingest,
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let desc = dataset
            .scan()
            .project(&["blob"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
            .column(0)
            .as_struct()
            .to_owned();
        assert_eq!(
            desc.column_by_name("kind")
                .unwrap()
                .as_primitive::<UInt8Type>()
                .value(0),
            BlobKind::Packed as u8
        );

        let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs[0].kind(), BlobKind::Packed);
        assert_eq!(blobs[0].read().await.unwrap().as_ref(), payload.as_slice());
    }

    #[tokio::test]
    async fn test_blob_v2_external_ingest_dedicated() {
        let dataset_dir = TempDir::default();
        let external_dir = TempDir::default();
        let external_path = external_dir.std_path().join("external.bin");
        let payload = vec![0x7A; super::DEDICATED_THRESHOLD + 1024];
        std::fs::write(&external_path, &payload).unwrap();
        let external_uri = format!("file://{}", external_path.display());

        let mut blob_builder = BlobArrayBuilder::new(1);
        blob_builder.push_uri(external_uri).unwrap();
        let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();
        let schema = Arc::new(Schema::new(vec![blob_field("blob", true)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        let dataset = Arc::new(
            Dataset::write(
                reader,
                &dataset_dir.path_str(),
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    external_blob_mode: ExternalBlobMode::Ingest,
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let desc = dataset
            .scan()
            .project(&["blob"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
            .column(0)
            .as_struct()
            .to_owned();
        assert_eq!(
            desc.column_by_name("kind")
                .unwrap()
                .as_primitive::<UInt8Type>()
                .value(0),
            BlobKind::Dedicated as u8
        );

        let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs[0].kind(), BlobKind::Dedicated);
        assert_eq!(blobs[0].read().await.unwrap().as_ref(), payload.as_slice());
    }

    #[tokio::test]
    async fn test_blob_v2_requires_v2_2() {
        let test_dir = TempStrDir::default();

        let mut blob_builder = BlobArrayBuilder::new(1);
        blob_builder.push_bytes(b"hello").unwrap();
        let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();

        let id_array: arrow_array::ArrayRef = Arc::new(UInt32Array::from(vec![0]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            blob_field("blob", true),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        let result = Dataset::write(
            reader,
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await;

        assert!(
            result.is_err(),
            "Blob v2 should be rejected for file version 2.1"
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Blob v2 requires file version >= 2.2")
        );
    }

    async fn preprocess_kind_with_schema_metadata(metadata_value: &str, data_len: usize) -> u8 {
        let (object_store, base_path) = ObjectStore::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            "memory://blob_preprocessor",
            &ObjectStoreParams::default(),
        )
        .await
        .unwrap();
        let object_store = object_store.as_ref().clone();
        let data_dir = base_path.child("data");

        let mut field = blob_field("blob", true);
        let mut metadata = field.metadata().clone();
        metadata.insert(
            BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.to_string(),
            metadata_value.to_string(),
        );
        field = field.with_metadata(metadata);

        let writer_arrow_schema = Schema::new(vec![field.clone()]);
        let writer_schema = lance_core::datatypes::Schema::try_from(&writer_arrow_schema).unwrap();

        let mut preprocessor = super::BlobPreprocessor::new(
            object_store.clone(),
            data_dir,
            "data_file_key".to_string(),
            &writer_schema,
            None,
            false,
            ExternalBlobMode::Reference,
            Arc::new(ObjectStoreRegistry::default()),
            ObjectStoreParams::default(),
            None,
        );

        let mut blob_builder = BlobArrayBuilder::new(1);
        blob_builder.push_bytes(vec![0u8; data_len]).unwrap();
        let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();

        let field_without_metadata =
            Field::new("blob", field.data_type().clone(), field.is_nullable());
        let batch_schema = Arc::new(Schema::new(vec![field_without_metadata]));
        let batch = RecordBatch::try_new(batch_schema, vec![blob_array]).unwrap();

        let out = preprocessor.preprocess_batch(&batch).await.unwrap();
        let struct_arr = out
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .unwrap();
        struct_arr
            .column_by_name("kind")
            .unwrap()
            .as_primitive::<arrow::datatypes::UInt8Type>()
            .value(0)
    }

    #[tokio::test]
    async fn test_blob_v2_dedicated_threshold_ignores_non_positive_metadata() {
        let kind = preprocess_kind_with_schema_metadata("0", 256 * 1024).await;
        assert_eq!(kind, lance_core::datatypes::BlobKind::Packed as u8);
    }

    #[tokio::test]
    async fn test_blob_v2_dedicated_threshold_respects_smaller_metadata() {
        let kind = preprocess_kind_with_schema_metadata("131072", 256 * 1024).await;
        assert_eq!(kind, lance_core::datatypes::BlobKind::Dedicated as u8);
    }

    #[tokio::test]
    async fn test_blob_v2_dedicated_threshold_respects_larger_metadata() {
        let kind =
            preprocess_kind_with_schema_metadata("8388608", super::DEDICATED_THRESHOLD + 1024)
                .await;
        assert_eq!(kind, lance_core::datatypes::BlobKind::Packed as u8);
    }
}
