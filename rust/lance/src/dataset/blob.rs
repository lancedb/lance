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
use arrow_array::{
    Array, ArrayRef, GenericListArray, OffsetSizeTrait, RecordBatch, builder::LargeBinaryBuilder,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use bytes::Bytes;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt, TryStreamExt, stream};
use lance_arrow::{
    ARROW_EXT_NAME_KEY, BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY,
    BLOB_INLINE_SIZE_THRESHOLD_META_KEY, BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY, FieldExt,
    list::ListArrayExt, r#struct::StructArrayExt,
};
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use lance_io::scheduler::{FileScheduler, ScanScheduler, SchedulerConfig};
use object_store::path::Path;
use tokio::sync::{Mutex, OnceCell, oneshot};
use url::Url;

use super::take::TakeBuilder;
use super::write::ExternalBlobMode;
use super::{Dataset, ProjectionRequest};
use crate::blob::{
    BlobDescriptor, BlobDescriptorArrayBuilder, BlobIdAllocator, BlobRange, PackedBlobWriter,
    is_logical_blob_v2_field, is_prepared_blob_v2_field, validate_prepared_blob_array,
};
use arrow_array::StructArray;
use lance_core::datatypes::{BlobKind, BlobVersion, Field as LanceField, Schema, parse_field_path};
use lance_core::utils::blob::blob_path;
use lance_core::{Error, ROW_ADDR, Result, utils::address::RowAddress};
use lance_io::traits::Reader;
use lance_io::utils::CachedFileSize;

const INLINE_MAX: usize = 64 * 1024; // 64KB inline cutoff
const DEDICATED_THRESHOLD: usize = 4 * 1024 * 1024; // 4MB dedicated cutoff
const PACK_FILE_MAX_SIZE: usize = 1024 * 1024 * 1024; // 1GiB per .pack sidecar

pub(super) fn blob_inline_threshold_from_metadata(
    metadata: &HashMap<String, String>,
    field_name: &str,
) -> Result<usize> {
    blob_threshold_from_metadata(
        metadata,
        field_name,
        BLOB_INLINE_SIZE_THRESHOLD_META_KEY,
        INLINE_MAX,
        true,
    )
}

pub(super) fn blob_dedicated_threshold_from_metadata(
    metadata: &HashMap<String, String>,
    field_name: &str,
) -> Result<usize> {
    blob_threshold_from_metadata(
        metadata,
        field_name,
        BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY,
        DEDICATED_THRESHOLD,
        false,
    )
}

pub(super) fn blob_pack_file_threshold_from_metadata(
    metadata: &HashMap<String, String>,
    field_name: &str,
) -> Result<usize> {
    blob_threshold_from_metadata(
        metadata,
        field_name,
        BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY,
        PACK_FILE_MAX_SIZE,
        false,
    )
}

fn blob_threshold_from_metadata(
    metadata: &HashMap<String, String>,
    field_name: &str,
    key: &str,
    default_value: usize,
    allow_zero: bool,
) -> Result<usize> {
    let Some(value) = metadata.get(key) else {
        return Ok(default_value);
    };
    let threshold = value.parse::<usize>().map_err(|_| {
        Error::invalid_input(format!(
            "Invalid blob threshold metadata {key}={value:?} for field '{field_name}'; \
             expected a non-negative integer that fits in usize"
        ))
    })?;
    if !allow_zero && threshold == 0 {
        return Err(Error::invalid_input(format!(
            "Invalid blob threshold metadata {key}={value:?} for field '{field_name}'; \
             expected a positive integer"
        )));
    }
    Ok(threshold)
}

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

struct RollingPackedBlobWriter {
    current: Option<PackedBlobWriter>,
    current_size: usize,
    current_max_pack_size: Option<usize>,
}

impl RollingPackedBlobWriter {
    fn new() -> Self {
        Self {
            current: None,
            current_size: 0,
            current_max_pack_size: None,
        }
    }

    async fn start_new_pack(
        &mut self,
        object_store: ObjectStore,
        data_dir: Path,
        data_file_key: String,
        blob_id_allocator: BlobIdAllocator,
        max_pack_size: usize,
    ) -> Result<()> {
        self.finish().await?;
        let blob_id = blob_id_allocator.next()?;
        let data_file_path = data_dir.join(format!("{data_file_key}.lance"));
        self.current =
            Some(PackedBlobWriter::try_new(object_store, data_file_path, blob_id).await?);
        self.current_size = 0;
        self.current_max_pack_size = Some(max_pack_size);
        Ok(())
    }

    async fn write(
        &mut self,
        object_store: ObjectStore,
        data_dir: Path,
        data_file_key: String,
        blob_id_allocator: BlobIdAllocator,
        max_pack_size: usize,
        source: BlobWriteSource<'_>,
    ) -> Result<BlobDescriptor> {
        let len = source.size();
        let needs_new_pack = match (self.current.as_ref(), self.current_max_pack_size) {
            (Some(_), Some(current_max_pack_size)) => {
                current_max_pack_size != max_pack_size
                    || self.current_size + len > current_max_pack_size
            }
            _ => true,
        };
        if needs_new_pack {
            self.start_new_pack(
                object_store,
                data_dir,
                data_file_key,
                blob_id_allocator,
                max_pack_size,
            )
            .await?;
        }

        let writer = self.current.as_mut().expect("pack writer is initialized");
        let value = match source {
            BlobWriteSource::Bytes(data) => writer.write_blob_bytes(data).await?,
            BlobWriteSource::External(source) => {
                writer
                    .write_blob_from_reader(source.reader.as_ref(), source.reader_range()?)
                    .await?
            }
        };
        self.current_size += len;
        Ok(value)
    }

    async fn finish(&mut self) -> Result<()> {
        if let Some(writer) = self.current.take() {
            writer.finish().await?;
        }
        self.current_size = 0;
        self.current_max_pack_size = None;
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
    blob_id_allocator: BlobIdAllocator,
    pack_writer: RollingPackedBlobWriter,
    /// Write-param override for the pack-file roll size. When set, it takes
    /// precedence over each field's `blob-pack-file-size-threshold` metadata for
    /// this write job only; it is not persisted into the dataset schema.
    pack_file_size_override: Option<usize>,
    field_processors: Vec<BlobPreprocessField>,
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

#[derive(Clone, Debug)]
struct BlobPreprocessField {
    kind: BlobPreprocessFieldKind,
}

#[derive(Clone, Debug)]
enum BlobPreprocessFieldKind {
    BlobV2 {
        inline_threshold: usize,
        dedicated_threshold: usize,
        pack_file_threshold: usize,
        writer_metadata: HashMap<String, String>,
    },
    Struct {
        children: Vec<BlobPreprocessField>,
    },
    List {
        child: Box<BlobPreprocessField>,
    },
    Passthrough,
}

impl BlobPreprocessField {
    fn new(field: &ArrowField) -> Result<Self> {
        if field.is_blob_v2() {
            if is_prepared_blob_v2_field(field) {
                return Ok(Self {
                    kind: BlobPreprocessFieldKind::Passthrough,
                });
            }
            if !is_logical_blob_v2_field(field) {
                return Err(Error::invalid_input(format!(
                    "Blob v2 field '{}' must use either logical struct<data: LargeBinary?, \
                     uri: Utf8?> with optional position/size UInt64 fields or prepared \
                     struct<kind: UInt8?, data: LargeBinary?, uri: Utf8?, blob_id: UInt32?, \
                     blob_size: UInt64?, position: UInt64?>",
                    field.name()
                )));
            }
            return Ok(Self {
                kind: BlobPreprocessFieldKind::BlobV2 {
                    inline_threshold: blob_inline_threshold_from_metadata(
                        field.metadata(),
                        field.name(),
                    )?,
                    dedicated_threshold: blob_dedicated_threshold_from_metadata(
                        field.metadata(),
                        field.name(),
                    )?,
                    pack_file_threshold: blob_pack_file_threshold_from_metadata(
                        field.metadata(),
                        field.name(),
                    )?,
                    writer_metadata: field.metadata().clone(),
                },
            });
        }

        if let ArrowDataType::Struct(children) = field.data_type() {
            let children = children
                .iter()
                .map(|child| Self::new(child.as_ref()))
                .collect::<Result<Vec<_>>>()?;
            if children.iter().any(|child| child.requires_preprocessing()) {
                return Ok(Self {
                    kind: BlobPreprocessFieldKind::Struct { children },
                });
            }
        }

        if let ArrowDataType::List(child) | ArrowDataType::LargeList(child) = field.data_type() {
            let child = Self::new(child.as_ref())?;
            if child.requires_preprocessing() {
                return Ok(Self {
                    kind: BlobPreprocessFieldKind::List {
                        child: Box::new(child),
                    },
                });
            }
        }

        Ok(Self {
            kind: BlobPreprocessFieldKind::Passthrough,
        })
    }

    fn requires_preprocessing(&self) -> bool {
        !matches!(self.kind, BlobPreprocessFieldKind::Passthrough)
    }
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
}

impl BlobWriteSource<'_> {
    /// Return the payload size regardless of whether bytes come from memory or an external reader.
    fn size(&self) -> usize {
        match self {
            Self::Bytes(data) => data.len(),
            Self::External(source) => source
                .size()
                .try_into()
                .expect("packed and inline external blobs must fit into usize"),
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
        pack_file_size_override: Option<usize>,
    ) -> Result<Self> {
        let pack_writer = RollingPackedBlobWriter::new();
        let arrow_schema = arrow_schema::Schema::from(schema);
        let field_processors = arrow_schema
            .fields()
            .iter()
            .map(|field| BlobPreprocessField::new(field.as_ref()))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            object_store,
            data_dir,
            data_file_key,
            blob_id_allocator: BlobIdAllocator::new(1),
            pack_writer,
            pack_file_size_override,
            field_processors,
            external_base_resolver,
            allow_external_blob_outside_bases,
            external_blob_mode,
            source_store_registry,
            source_store_params,
        })
    }

    fn blob_writer_with_metadata(
        &self,
        field: &ArrowField,
        metadata: HashMap<String, String>,
    ) -> BlobDescriptorArrayBuilder {
        BlobDescriptorArrayBuilder::new_with_metadata(field.name(), field.is_nullable(), metadata)
    }

    async fn write_dedicated(
        object_store: ObjectStore,
        data_dir: Path,
        data_file_key: String,
        blob_id_allocator: BlobIdAllocator,
        source: BlobWriteSource<'_>,
    ) -> Result<BlobDescriptor> {
        let blob_id = blob_id_allocator.next()?;
        let data_file_path = data_dir.join(format!("{data_file_key}.lance"));
        let mut writer =
            crate::blob::DedicatedBlobWriter::try_new(object_store, data_file_path, blob_id)
                .await?;
        match source {
            BlobWriteSource::Bytes(data) => writer.write(data).await?,
            BlobWriteSource::External(source) => {
                writer
                    .write_from_reader(source.reader.as_ref(), source.reader_range()?)
                    .await?;
            }
        }
        writer.finish().await
    }

    async fn write_packed(
        &mut self,
        field_pack_file_threshold: usize,
        source: BlobWriteSource<'_>,
    ) -> Result<BlobDescriptor> {
        let max_pack_size = self
            .pack_file_size_override
            .unwrap_or(field_pack_file_threshold);
        self.pack_writer
            .write(
                self.object_store.clone(),
                self.data_dir.clone(),
                self.data_file_key.clone(),
                self.blob_id_allocator.clone(),
                max_pack_size,
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
        let expected_columns = self.field_processors.len();
        if batch.num_columns() != expected_columns {
            return Err(Error::invalid_input(format!(
                "Unexpected number of columns: expected {}, got {}",
                expected_columns,
                batch.num_columns()
            )));
        }

        let batch_schema = batch.schema();
        let batch_fields = batch_schema.fields();
        let field_processors = self.field_processors.clone();

        let mut new_columns = Vec::with_capacity(batch.num_columns());
        let mut new_fields = Vec::with_capacity(batch.num_columns());

        for ((processor, array), field) in field_processors
            .iter()
            .zip(batch.columns().iter())
            .zip(batch_fields.iter())
        {
            let (new_column, new_field) = self
                .preprocess_field(processor, array.clone(), field)
                .await?;
            new_columns.push(new_column);
            new_fields.push(new_field);
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

    fn preprocess_field<'a>(
        &'a mut self,
        processor: &'a BlobPreprocessField,
        array: ArrayRef,
        field: &'a Arc<ArrowField>,
    ) -> BoxFuture<'a, Result<(ArrayRef, Arc<ArrowField>)>> {
        async move {
            if is_prepared_blob_v2_field(field.as_ref()) {
                validate_prepared_blob_array(field.as_ref(), &array)?;
                return Ok((array, field.clone()));
            }

            match &processor.kind {
                BlobPreprocessFieldKind::Passthrough => Ok((array, field.clone())),
                BlobPreprocessFieldKind::BlobV2 {
                    inline_threshold,
                    dedicated_threshold,
                    pack_file_threshold,
                    writer_metadata,
                } => {
                    self.preprocess_blob_array(
                        array,
                        field.as_ref(),
                        *inline_threshold,
                        *dedicated_threshold,
                        *pack_file_threshold,
                        writer_metadata,
                    )
                    .await
                }
                BlobPreprocessFieldKind::Struct { children } => {
                    self.preprocess_struct_array(array, field.as_ref(), children)
                        .await
                }
                BlobPreprocessFieldKind::List { child } => match field.data_type() {
                    ArrowDataType::List(_) => {
                        self.preprocess_list_array::<i32>(array, field.as_ref(), child)
                            .await
                    }
                    ArrowDataType::LargeList(_) => {
                        self.preprocess_list_array::<i64>(array, field.as_ref(), child)
                            .await
                    }
                    _ => Err(Error::internal(format!(
                        "Blob list preprocessor received non-list field '{}'",
                        field.name()
                    ))),
                },
            }
        }
        .boxed()
    }

    async fn preprocess_struct_array(
        &mut self,
        array: ArrayRef,
        field: &ArrowField,
        children: &[BlobPreprocessField],
    ) -> Result<(ArrayRef, Arc<ArrowField>)> {
        let struct_arr = array
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| Error::invalid_input("Struct field was not a struct array"))?;
        if struct_arr.num_columns() != children.len() {
            return Err(Error::invalid_input(format!(
                "Struct field '{}' expected {} children, got {}",
                field.name(),
                children.len(),
                struct_arr.num_columns()
            )));
        }

        let struct_arr = struct_arr.normalize_slicing()?;
        let parent_nulls = struct_arr.nulls().cloned();
        let pushed_down = struct_arr.pushdown_nulls()?;
        let child_fields = pushed_down.fields().clone();
        let child_columns = pushed_down.columns().to_vec();

        let mut new_columns = Vec::with_capacity(children.len());
        let mut new_fields = Vec::with_capacity(children.len());
        for ((child_processor, child_array), child_field) in
            children.iter().zip(child_columns).zip(child_fields.iter())
        {
            let (new_column, new_field) = self
                .preprocess_field(child_processor, child_array, child_field)
                .await?;
            new_columns.push(new_column);
            new_fields.push(new_field);
        }

        let struct_array =
            StructArray::try_new(new_fields.clone().into(), new_columns, parent_nulls)?;
        let field = Arc::new(
            ArrowField::new(
                field.name(),
                ArrowDataType::Struct(new_fields.into()),
                field.is_nullable(),
            )
            .with_metadata(field.metadata().clone()),
        );
        Ok((Arc::new(struct_array), field))
    }

    async fn preprocess_list_array<O: OffsetSizeTrait>(
        &mut self,
        array: ArrayRef,
        field: &ArrowField,
        child: &BlobPreprocessField,
    ) -> Result<(ArrayRef, Arc<ArrowField>)> {
        let list_arr = array.as_list::<O>();
        let list_arr = if list_arr.null_count() > 0 {
            list_arr.filter_garbage_nulls()
        } else {
            list_arr.clone()
        };

        let first_offset = *list_arr
            .offsets()
            .first()
            .ok_or_else(|| Error::invalid_input("List offsets cannot be empty"))?;
        let last_offset = *list_arr
            .offsets()
            .last()
            .ok_or_else(|| Error::invalid_input("List offsets cannot be empty"))?;
        let values_len = list_arr.values().len();
        let needs_trim = first_offset != O::zero()
            || last_offset.to_usize().ok_or_else(|| {
                Error::invalid_input(format!(
                    "List field '{}' offset does not fit into usize",
                    field.name()
                ))
            })? != values_len;

        let (offsets, values) = if needs_trim {
            let values = list_arr.trimmed_values();
            let offsets = list_arr
                .offsets()
                .iter()
                .map(|offset| *offset - first_offset)
                .collect::<Vec<_>>();
            (OffsetBuffer::new(ScalarBuffer::from(offsets)), values)
        } else {
            (list_arr.offsets().clone(), list_arr.values().clone())
        };

        let child_field = match field.data_type() {
            ArrowDataType::List(child_field) | ArrowDataType::LargeList(child_field) => {
                child_field.clone()
            }
            other => {
                return Err(Error::invalid_input(format!(
                    "Blob list preprocessor expected list field '{}', got {other}",
                    field.name()
                )));
            }
        };
        let (new_values, new_child_field) =
            self.preprocess_field(child, values, &child_field).await?;

        let list_array = GenericListArray::<O>::try_new(
            new_child_field,
            offsets,
            new_values,
            list_arr.nulls().cloned(),
        )?;
        let field = Arc::new(
            ArrowField::new(
                field.name(),
                list_array.data_type().clone(),
                field.is_nullable(),
            )
            .with_metadata(field.metadata().clone()),
        );
        Ok((Arc::new(list_array), field))
    }

    async fn preprocess_blob_array(
        &mut self,
        array: ArrayRef,
        field: &ArrowField,
        inline_threshold: usize,
        dedicated_threshold: usize,
        pack_file_threshold: usize,
        writer_metadata: &HashMap<String, String>,
    ) -> Result<(ArrayRef, Arc<ArrowField>)> {
        let struct_arr = array
            .as_any()
            .downcast_ref::<StructArray>()
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

        let mut blob_writer = self.blob_writer_with_metadata(field, writer_metadata.clone());

        for i in 0..struct_arr.len() {
            if struct_arr.is_null(i) {
                blob_writer.push_null()?;
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

            if has_data && data_len > dedicated_threshold {
                let value = Self::write_dedicated(
                    self.object_store.clone(),
                    self.data_dir.clone(),
                    self.data_file_key.clone(),
                    self.blob_id_allocator.clone(),
                    BlobWriteSource::Bytes(data_col.value(i)),
                )
                .await?;
                blob_writer.push(value)?;
                continue;
            }

            if has_data && data_len > inline_threshold {
                let value = self
                    .write_packed(
                        pack_file_threshold,
                        BlobWriteSource::Bytes(data_col.value(i)),
                    )
                    .await?;
                blob_writer.push(value)?;
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
                        let value = Self::write_dedicated(
                            self.object_store.clone(),
                            self.data_dir.clone(),
                            self.data_file_key.clone(),
                            self.blob_id_allocator.clone(),
                            BlobWriteSource::External(&source),
                        )
                        .await?;
                        blob_writer.push(value)?;
                        continue;
                    }

                    if data_len > inline_threshold as u64 {
                        let value = self
                            .write_packed(pack_file_threshold, BlobWriteSource::External(&source))
                            .await?;
                        blob_writer.push(value)?;
                        continue;
                    }

                    let data = source.read_all().await?;
                    blob_writer.push_inline(data)?;
                    continue;
                }

                let (external_base_id, external_uri_or_path) =
                    self.resolve_external_reference(uri_val).await?;
                let range = if has_position && has_size {
                    BlobRange {
                        offset: position_col
                            .as_ref()
                            .expect("position column must exist")
                            .value(i),
                        size: size_col.as_ref().expect("size column must exist").value(i),
                    }
                } else {
                    BlobRange { offset: 0, size: 0 }
                };
                blob_writer.push(BlobDescriptor::External {
                    base_id: external_base_id,
                    uri: external_uri_or_path,
                    offset: range.offset,
                    size: range.size,
                })?;
                continue;
            }

            if has_data {
                blob_writer.push_inline(Bytes::copy_from_slice(data_col.value(i)))?;
            } else {
                blob_writer.push_null()?;
            }
        }

        let column = blob_writer.finish()?;
        let (field, array) = column.into_parts();
        Ok((array, Arc::new(field)))
    }

    pub(crate) async fn finish(&mut self) -> Result<()> {
        self.pack_writer.finish().await
    }
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

async fn execute_blob_entries(
    entries: Vec<BlobEntry>,
    io_parallelism: usize,
    io_buffer_size_bytes: Option<u64>,
) -> Result<Vec<IndexedReadBlob>> {
    let plans = plan_blob_read_plans(entries);
    if plans.is_empty() {
        return Ok(Vec::new());
    }

    let execution = Arc::new(ReadBlobsExecution::new(io_buffer_size_bytes));
    let batches = stream::iter(plans.into_iter().map(move |plan| {
        let execution = execution.clone();
        execute_blob_read_plan(plan, execution)
    }))
    .buffer_unordered(io_parallelism.max(1))
    .try_collect::<Vec<_>>()
    .await?;
    Ok(batches.into_iter().flatten().collect())
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
    let schema = dataset.schema();
    let blob_field = schema
        .field(column)
        .ok_or_else(|| Error::field_not_found(column, schema.field_paths()))?;
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

    let descriptions = leaf_descriptor_struct(&description_and_addr, column)?;
    let row_addrs = description_and_addr.column(1).as_primitive::<UInt64Type>();

    match blob_version_from_descriptions(descriptions)? {
        BlobVersion::V1 => collect_blob_entries_v1(dataset, blob_field_id, descriptions, row_addrs),
        BlobVersion::V2 => {
            collect_blob_entries_v2(dataset, blob_field_id, descriptions, row_addrs).await
        }
    }
}

/// Walk into the descriptor `RecordBatch` at `column` and return the leaf
/// descriptor `StructArray`, descending through nested struct children for
/// dotted paths.
fn leaf_descriptor_struct<'a>(batch: &'a RecordBatch, column: &str) -> Result<&'a StructArray> {
    let current = leaf_descriptor_array(batch, column)?;
    current
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            Error::invalid_input_source(
                format!(
                    "Blob column '{}' expected descriptor struct but got {}",
                    column,
                    current.data_type()
                )
                .into(),
            )
        })
}

fn leaf_descriptor_array<'a>(batch: &'a RecordBatch, column: &str) -> Result<&'a dyn Array> {
    let path = parse_field_path(column)?;
    let mut current: &dyn Array = batch
        .column_by_name(&path[0])
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "Blob column '{}' was not found in descriptor batch",
                column
            ))
        })?
        .as_ref();
    for segment in &path[1..] {
        let struct_array = current
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| {
                Error::invalid_input_source(
                    format!(
                        "Blob column path '{}' expected struct before segment '{}' but got {}",
                        column,
                        segment,
                        current.data_type()
                    )
                    .into(),
                )
            })?;
        current = struct_array
            .column_by_name(segment)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Blob column path '{}' missing segment '{}'",
                    column, segment
                ))
            })?
            .as_ref();
    }
    Ok(current)
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

struct BlobV2DescriptorColumns<'a> {
    descriptions: &'a StructArray,
    kinds: &'a arrow::array::PrimitiveArray<UInt8Type>,
    positions: &'a arrow::array::PrimitiveArray<UInt64Type>,
    sizes: &'a arrow::array::PrimitiveArray<UInt64Type>,
    blob_ids: &'a arrow::array::PrimitiveArray<UInt32Type>,
    blob_uris: &'a arrow::array::GenericStringArray<i32>,
}

impl<'a> BlobV2DescriptorColumns<'a> {
    fn new(descriptions: &'a StructArray) -> Self {
        Self {
            descriptions,
            kinds: descriptions.column(0).as_primitive::<UInt8Type>(),
            positions: descriptions.column(1).as_primitive::<UInt64Type>(),
            sizes: descriptions.column(2).as_primitive::<UInt64Type>(),
            blob_ids: descriptions.column(3).as_primitive::<UInt32Type>(),
            blob_uris: descriptions.column(4).as_string::<i32>(),
        }
    }

    fn is_null_blob(&self, idx: usize) -> Result<bool> {
        if self.descriptions.is_null(idx) || self.kinds.is_null(idx) {
            return Ok(true);
        }
        let kind = BlobKind::try_from(self.kinds.value(idx))?;
        Ok(matches!(kind, BlobKind::Inline)
            && self.positions.value(idx) == 0
            && self.sizes.value(idx) == 0)
    }
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
            let data_file_path = dataset.data_dir().join(data_file.path.as_str());
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
    let columns = BlobV2DescriptorColumns::new(descriptions);
    let mut files = Vec::with_capacity(row_addrs.len());
    let mut read_context = BlobV2ReadContext::new(dataset, blob_field_id);
    for (selection_index, row_addr) in row_addrs.values().iter().enumerate() {
        if let Some(entry) = read_context
            .collect_entry(&columns, selection_index, selection_index, *row_addr)
            .await?
        {
            files.push(entry);
        }
    }

    Ok(files)
}

fn is_blob_v2_binary_view(field: &LanceField) -> bool {
    field.is_blob_v2() && matches!(field.data_type(), ArrowDataType::LargeBinary)
}

fn public_blob_v2_binary_output_field(mut field: LanceField) -> LanceField {
    if is_blob_v2_binary_view(&field) {
        field.metadata.remove(ARROW_EXT_NAME_KEY);
    }
    field.children = field
        .children
        .into_iter()
        .map(public_blob_v2_binary_output_field)
        .collect();
    field
}

/// Return the public Arrow-facing schema for a blob v2 binary scan.
///
/// Scan planning uses a blob v2 extension marker on `LargeBinary` leaves to
/// identify payloads that need descriptor-based materialization. This helper
/// removes that internal marker before the schema is exposed to callers.
pub fn public_blob_v2_binary_output_schema(schema: &Schema) -> Schema {
    Schema {
        fields: schema
            .fields
            .iter()
            .cloned()
            .map(public_blob_v2_binary_output_field)
            .collect(),
        metadata: schema.metadata.clone(),
    }
}

fn field_has_blob_v2_binary_view(field: &LanceField) -> bool {
    is_blob_v2_binary_view(field) || field.children.iter().any(field_has_blob_v2_binary_view)
}

/// Return true if the schema contains a blob v2 leaf in binary payload view.
///
/// This detects the internal `LargeBinary` view created by
/// [`BlobHandling::AllBinary`](lance_core::datatypes::BlobHandling::AllBinary)
/// or selective binary blob handling.
pub fn schema_has_blob_v2_binary_view(schema: &Schema) -> bool {
    schema.fields.iter().any(field_has_blob_v2_binary_view)
}

fn blob_v2_descriptor_field(mut field: LanceField) -> LanceField {
    if is_blob_v2_binary_view(&field) {
        field.unloaded_mut();
        return field;
    }

    field.children = field
        .children
        .into_iter()
        .map(blob_v2_descriptor_field)
        .collect();
    field
}

/// Convert blob v2 binary-view leaves back to descriptor-view leaves.
///
/// Readers use this schema to fetch stored blob descriptors first. The scan
/// layer then materializes those descriptors into the caller's binary payload
/// view after row addresses are available.
pub fn blob_v2_descriptor_schema(schema: &Schema) -> Schema {
    Schema {
        fields: schema
            .fields
            .iter()
            .cloned()
            .map(blob_v2_descriptor_field)
            .collect(),
        metadata: schema.metadata.clone(),
    }
}

/// Materialize blob v2 descriptor arrays in a decoded batch into binary arrays.
///
/// The input batch must include `_rowaddr`, which is used to resolve packed,
/// dedicated, inline, and external blob payload locations. `output_schema`
/// defines the exact returned columns, including requested system columns, with
/// blob v2 binary leaves exposed as plain `LargeBinary` fields.
pub async fn materialize_blob_v2_binary_batch(
    dataset: &Arc<Dataset>,
    output_schema: &Schema,
    batch: RecordBatch,
) -> Result<RecordBatch> {
    let row_addr_idx = batch
        .schema()
        .column_with_name(ROW_ADDR)
        .ok_or_else(|| {
            Error::internal(format!(
                "_rowaddr column missing from blob v2 binary scan batch, columns: {:?}",
                batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|field| field.name())
                    .collect::<Vec<_>>()
            ))
        })?
        .0;
    let row_addrs = batch
        .column(row_addr_idx)
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let row_addrs: Arc<[u64]> = row_addrs.into();

    let mut columns = Vec::with_capacity(output_schema.fields.len());
    let mut fields = Vec::with_capacity(output_schema.fields.len());

    for field in &output_schema.fields {
        let input = batch
            .column_by_name(&field.name)
            .ok_or_else(|| {
                Error::internal(format!(
                    "blob v2 binary scan batch missing projected column '{}'",
                    field.name
                ))
            })?
            .clone();
        let materialized =
            materialize_blob_v2_binary_array(dataset, field, input, row_addrs.clone()).await?;
        columns.push(materialized);
        let output_field = public_blob_v2_binary_output_field(field.clone());
        fields.push(ArrowField::from(&output_field));
    }

    Ok(RecordBatch::try_new(
        Arc::new(ArrowSchema::new_with_metadata(
            fields,
            batch.schema().metadata().clone(),
        )),
        columns,
    )?)
}

fn materialize_blob_v2_binary_array<'a>(
    dataset: &'a Arc<Dataset>,
    field: &'a LanceField,
    array: ArrayRef,
    row_addrs: Arc<[u64]>,
) -> BoxFuture<'a, Result<ArrayRef>> {
    async move {
        if is_blob_v2_binary_view(field) {
            let descriptions = array.as_struct();
            return materialize_blob_v2_descriptors(
                dataset,
                field.id as u32,
                descriptions,
                row_addrs.as_ref(),
            )
            .await;
        }

        match field.data_type() {
            ArrowDataType::Struct(_) => {
                let struct_array = array.as_struct();
                let mut children = Vec::with_capacity(field.children.len());
                for (child_field, child_array) in
                    field.children.iter().zip(struct_array.columns().iter())
                {
                    children.push(
                        materialize_blob_v2_binary_array(
                            dataset,
                            child_field,
                            child_array.clone(),
                            row_addrs.clone(),
                        )
                        .await?,
                    );
                }
                let public_field = public_blob_v2_binary_output_field(field.clone());
                let ArrowDataType::Struct(fields) = public_field.data_type() else {
                    unreachable!("public output field preserved struct type")
                };
                Ok(Arc::new(StructArray::try_new(
                    fields,
                    children,
                    struct_array.nulls().cloned(),
                )?) as ArrayRef)
            }
            ArrowDataType::List(_) => {
                let list_array = array.as_list::<i32>();
                materialize_blob_v2_list_array::<i32>(dataset, field, list_array, row_addrs).await
            }
            ArrowDataType::LargeList(_) => {
                let list_array = array.as_list::<i64>();
                materialize_blob_v2_list_array::<i64>(dataset, field, list_array, row_addrs).await
            }
            _ => Ok(array),
        }
    }
    .boxed()
}

async fn materialize_blob_v2_list_array<O: OffsetSizeTrait>(
    dataset: &Arc<Dataset>,
    field: &LanceField,
    list_array: &GenericListArray<O>,
    row_addrs: Arc<[u64]>,
) -> Result<ArrayRef> {
    let offsets = list_array.value_offsets();
    let values_start = offsets[0].as_usize();
    let values_end = offsets[list_array.len()].as_usize();
    if values_end < values_start {
        return Err(Error::internal(format!(
            "List field '{}' has invalid offsets while materializing blob v2 binary scan",
            field.name
        )));
    }

    let values_len = values_end - values_start;
    let mut normalized_offsets = Vec::with_capacity(list_array.len() + 1);
    normalized_offsets.push(O::usize_as(0));
    let mut child_row_addrs = Vec::with_capacity(values_len);
    for row_idx in 0..list_array.len() {
        let start = offsets[row_idx].as_usize();
        let end = offsets[row_idx + 1].as_usize();
        if end < start {
            return Err(Error::internal(format!(
                "List field '{}' has decreasing offsets while materializing blob v2 binary scan",
                field.name
            )));
        }
        let row_addr = row_addrs.get(row_idx).copied().ok_or_else(|| {
            Error::internal(format!(
                "List field '{}' row address count {} did not match row count {}",
                field.name,
                row_addrs.len(),
                list_array.len()
            ))
        })?;
        for _ in start..end {
            child_row_addrs.push(row_addr);
        }
        normalized_offsets.push(O::usize_as(end - values_start));
    }
    let child_row_addrs: Arc<[u64]> = child_row_addrs.into();
    let child = field.children.first().ok_or_else(|| {
        Error::internal(format!(
            "List field '{}' missing child while materializing blob v2 binary scan",
            field.name
        ))
    })?;
    let values = list_array.values().slice(values_start, values_len);
    let values = materialize_blob_v2_binary_array(dataset, child, values, child_row_addrs).await?;
    let child_field = public_blob_v2_binary_output_field(child.clone());
    let list_array = GenericListArray::<O>::try_new(
        Arc::new(ArrowField::from(&child_field)),
        OffsetBuffer::new(ScalarBuffer::from(normalized_offsets)),
        values,
        list_array.nulls().cloned(),
    )?;
    Ok(Arc::new(list_array))
}

async fn materialize_blob_v2_descriptors(
    dataset: &Arc<Dataset>,
    blob_field_id: u32,
    descriptions: &StructArray,
    row_addrs: &[u64],
) -> Result<ArrayRef> {
    if descriptions.len() != row_addrs.len() {
        return Err(Error::internal(format!(
            "blob v2 descriptor count {} did not match row address count {}",
            descriptions.len(),
            row_addrs.len()
        )));
    }
    match blob_version_from_descriptions(descriptions)? {
        BlobVersion::V1 => {
            return Err(Error::not_supported(
                "Blob v2 binary materialization received a legacy blob descriptor".to_string(),
            ));
        }
        BlobVersion::V2 => {}
    }

    let columns = BlobV2DescriptorColumns::new(descriptions);
    let mut read_context = BlobV2ReadContext::new(dataset, blob_field_id);
    let mut entries = Vec::with_capacity(descriptions.len());
    let mut payloads = vec![None; descriptions.len()];

    for (idx, row_addr) in row_addrs.iter().copied().enumerate() {
        if descriptions.is_null(idx) || columns.kinds.is_null(idx) {
            continue;
        }

        let kind = BlobKind::try_from(columns.kinds.value(idx))?;
        if matches!(kind, BlobKind::Inline)
            && columns.positions.value(idx) == 0
            && columns.sizes.value(idx) == 0
        {
            payloads[idx] = Some(Bytes::new());
            continue;
        }

        let entry = read_context
            .collect_entry(&columns, idx, idx, row_addr)
            .await?
            .ok_or_else(|| {
                Error::internal(format!(
                    "blob v2 descriptor at index {idx} unexpectedly resolved to null"
                ))
            })?;
        entries.push(entry);
    }

    let blobs = execute_blob_entries(entries, dataset.object_store.io_parallelism(), None).await?;
    for blob in blobs {
        let payload = payloads.get_mut(blob.selection_index).ok_or_else(|| {
            Error::internal(format!(
                "blob result selection index {} exceeded descriptor count {}",
                blob.selection_index,
                descriptions.len()
            ))
        })?;
        if payload.replace(blob.data).is_some() {
            return Err(Error::internal(format!(
                "blob result selection index {} was produced more than once",
                blob.selection_index
            )));
        }
    }

    let payload_capacity = payloads.iter().flatten().map(Bytes::len).sum::<usize>();
    let mut builder = LargeBinaryBuilder::with_capacity(descriptions.len(), payload_capacity);
    for (idx, payload) in payloads.into_iter().enumerate() {
        if descriptions.is_null(idx) || columns.kinds.is_null(idx) {
            builder.append_null();
        } else {
            let payload = payload.ok_or_else(|| {
                Error::internal(format!(
                    "blob v2 descriptor at index {idx} did not produce a payload"
                ))
            })?;
            builder.append_value(payload);
        }
    }

    Ok(Arc::new(builder.finish()))
}

struct BlobV2ReadContext<'a> {
    dataset: &'a Arc<Dataset>,
    blob_field_id: u32,
    fragment_cache: HashMap<u32, BlobReadLocation>,
    store_cache: HashMap<u32, Arc<ObjectStore>>,
    external_base_path_cache: HashMap<u32, Path>,
    source_cache: HashMap<BlobSourceKey, Arc<BlobSource>>,
}

impl<'a> BlobV2ReadContext<'a> {
    fn new(dataset: &'a Arc<Dataset>, blob_field_id: u32) -> Self {
        Self {
            dataset,
            blob_field_id,
            fragment_cache: HashMap::new(),
            store_cache: HashMap::new(),
            external_base_path_cache: HashMap::new(),
            source_cache: HashMap::new(),
        }
    }

    async fn collect_entry(
        &mut self,
        columns: &BlobV2DescriptorColumns<'_>,
        idx: usize,
        selection_index: usize,
        row_addr: u64,
    ) -> Result<Option<BlobEntry>> {
        if columns.is_null_blob(idx)? {
            return Ok(None);
        }

        let kind = BlobKind::try_from(columns.kinds.value(idx))?;
        let entry = match kind {
            BlobKind::Inline => {
                self.collect_inline(columns, idx, selection_index, row_addr)
                    .await?
            }
            BlobKind::Dedicated => {
                self.collect_dedicated(columns, idx, selection_index, row_addr)
                    .await?
            }
            BlobKind::Packed => {
                self.collect_packed(columns, idx, selection_index, row_addr)
                    .await?
            }
            BlobKind::External => {
                self.collect_external(columns, idx, selection_index, row_addr)
                    .await?
            }
        };

        Ok(Some(entry))
    }

    async fn blob_read_location(&mut self, row_addr: u64) -> Result<BlobReadLocation> {
        resolve_blob_read_location(
            self.dataset,
            self.blob_field_id,
            row_addr,
            &mut self.fragment_cache,
            &mut self.store_cache,
        )
        .await
    }

    async fn collect_inline(
        &mut self,
        columns: &BlobV2DescriptorColumns<'_>,
        idx: usize,
        selection_index: usize,
        row_addr: u64,
    ) -> Result<BlobEntry> {
        let position = columns.positions.value(idx);
        let size = columns.sizes.value(idx);
        let location = self.blob_read_location(row_addr).await?;
        let source = shared_blob_source(
            &mut self.source_cache,
            location.object_store,
            &location.data_file_path,
        );
        Ok(BlobEntry {
            selection_index,
            row_address: row_addr,
            file: BlobFile::with_source(source, position, size, BlobKind::Inline, None),
        })
    }

    async fn collect_dedicated(
        &mut self,
        columns: &BlobV2DescriptorColumns<'_>,
        idx: usize,
        selection_index: usize,
        row_addr: u64,
    ) -> Result<BlobEntry> {
        let blob_id = columns.blob_ids.value(idx);
        let size = columns.sizes.value(idx);
        let location = self.blob_read_location(row_addr).await?;
        let path = blob_path(&location.data_file_dir, &location.data_file_key, blob_id);
        let source = shared_blob_source(&mut self.source_cache, location.object_store, &path);
        Ok(BlobEntry {
            selection_index,
            row_address: row_addr,
            file: BlobFile::with_source(source, 0, size, BlobKind::Dedicated, None),
        })
    }

    async fn collect_packed(
        &mut self,
        columns: &BlobV2DescriptorColumns<'_>,
        idx: usize,
        selection_index: usize,
        row_addr: u64,
    ) -> Result<BlobEntry> {
        let blob_id = columns.blob_ids.value(idx);
        let size = columns.sizes.value(idx);
        let position = columns.positions.value(idx);
        let location = self.blob_read_location(row_addr).await?;
        let path = blob_path(&location.data_file_dir, &location.data_file_key, blob_id);
        let source = shared_blob_source(&mut self.source_cache, location.object_store, &path);
        Ok(BlobEntry {
            selection_index,
            row_address: row_addr,
            file: BlobFile::with_source(source, position, size, BlobKind::Packed, None),
        })
    }

    async fn collect_external(
        &mut self,
        columns: &BlobV2DescriptorColumns<'_>,
        idx: usize,
        selection_index: usize,
        row_addr: u64,
    ) -> Result<BlobEntry> {
        let uri_or_path = columns.blob_uris.value(idx).to_string();
        let position = columns.positions.value(idx);
        let size = columns.sizes.value(idx);
        let base_id = columns.blob_ids.value(idx);
        let (object_store, path) = if base_id == 0 {
            let registry = self.dataset.session.store_registry();
            let params = self
                .dataset
                .store_params
                .as_ref()
                .map(|p| Arc::new((**p).clone()))
                .unwrap_or_else(|| Arc::new(ObjectStoreParams::default()));
            ObjectStore::from_uri_and_params(registry, &uri_or_path, &params).await?
        } else {
            let object_store = if let Some(store) = self.store_cache.get(&base_id) {
                store.clone()
            } else {
                let store = self.dataset.object_store(Some(base_id)).await?;
                self.store_cache.insert(base_id, store.clone());
                store
            };
            let base_root = if let Some(path) = self.external_base_path_cache.get(&base_id) {
                path.clone()
            } else {
                let base = self
                    .dataset
                    .manifest
                    .base_paths
                    .get(&base_id)
                    .ok_or_else(|| {
                        Error::invalid_input(format!(
                            "External blob references unknown base_id {}",
                            base_id
                        ))
                    })?;
                let path = base.extract_path(self.dataset.session.store_registry())?;
                self.external_base_path_cache.insert(base_id, path.clone());
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
        let source = shared_blob_source(&mut self.source_cache, object_store, &path);
        Ok(BlobEntry {
            selection_index,
            row_address: row_addr,
            file: BlobFile::with_source(
                source,
                position,
                size,
                BlobKind::External,
                Some(uri_or_path),
            ),
        })
    }
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
    let data_file_path = data_file_dir.clone().join(data_file.path.as_str());
    let data_file_key = data_file_key_from_path(data_file.path.as_str()).to_string();

    let object_store = if let Some(base_id) = data_file.base_id {
        if let Some(store) = store_cache.get(&base_id) {
            store.clone()
        } else {
            let store = dataset.object_store(Some(base_id)).await?;
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
        Array, ArrayRef, Int32Array, LargeBinaryArray, RecordBatchIterator, StringArray,
        StructArray, UInt8Array, UInt32Array, UInt64Array,
    };
    use arrow_schema::{DataType, Field, Schema};
    use async_trait::async_trait;
    use bytes::Bytes;
    use chrono::Utc;
    use futures::{StreamExt, TryStreamExt};
    use lance_arrow::{
        ARROW_EXT_NAME_KEY, BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY,
        BLOB_INLINE_SIZE_THRESHOLD_META_KEY, BLOB_META_KEY, BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY,
        BLOB_V2_EXT_NAME, DataTypeExt,
    };
    use lance_core::{
        datatypes::{BlobHandling, BlobKind},
        utils::blob::blob_path,
    };
    use lance_io::object_store::{
        ObjectStore, ObjectStoreParams, ObjectStoreRegistry, StorageOptionsAccessor,
    };
    use lance_io::stream::RecordBatchStream;
    use lance_table::format::BasePath;
    use object_store::{
        Attributes, CopyOptions, GetOptions, GetRange, GetResult, GetResultPayload, ListResult,
        MultipartUpload, ObjectMeta, PutMultipartOptions, PutOptions, PutPayload, PutResult,
        path::Path,
    };
    use tokio::sync::Notify;
    use url::Url;

    use lance_core::{
        Error, ROW_ADDR, ROW_CREATED_AT_VERSION, ROW_ID, ROW_LAST_UPDATED_AT_VERSION, Result,
        utils::tempfile::{TempDir, TempStrDir},
    };
    use lance_datagen::{BatchCount, RowCount, array};
    use lance_file::{
        version::LanceFileVersion,
        writer::{FileWriter, FileWriterOptions},
    };
    use uuid::Uuid;

    use super::{
        BlobEntry, BlobFile, BlobSource, ExternalBaseCandidate, ExternalBaseResolver,
        ReadBlobsExecution, collect_blob_entries_v1, data_file_key_from_path, execute_blob_entries,
        execute_blob_read_plan, plan_blob_read_plans,
    };
    use crate::{
        Dataset,
        blob::{BlobArrayBuilder, BlobDescriptorArrayBuilder, PackedBlobWriter, blob_field},
        dataset::{
            CommitBuilder, ExternalBlobMode, WriteMode, WriteParams,
            scanner::MaterializationStyle,
            transaction::{DataReplacementGroup, Operation, Transaction},
        },
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

    fn nested_blob_v2_batch(blob_array: ArrayRef) -> (Arc<Schema>, RecordBatch) {
        let blob_field = blob_field("blob", true);
        let info_fields = vec![Field::new("name", DataType::Utf8, false), blob_field];
        let info_array: ArrayRef = Arc::new(
            StructArray::try_new(
                info_fields.clone().into(),
                vec![
                    Arc::new(StringArray::from_iter_values(
                        (0..blob_array.len()).map(|idx| format!("name-{idx}")),
                    )) as ArrayRef,
                    blob_array,
                ],
                None,
            )
            .unwrap(),
        );

        let schema = Arc::new(Schema::new(vec![Field::new(
            "info",
            DataType::Struct(info_fields.into()),
            true,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![info_array]).unwrap();
        (schema, batch)
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
        async fn put_opts(
            &self,
            _location: &Path,
            _bytes: PutPayload,
            _opts: PutOptions,
        ) -> object_store::Result<PutResult> {
            unimplemented!("put_opts is not used by these tests")
        }

        async fn put_multipart_opts(
            &self,
            _location: &Path,
            _opts: PutMultipartOptions,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            unimplemented!("put_multipart_opts is not used by these tests")
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

        fn delete_stream(
            &self,
            _locations: futures::stream::BoxStream<'static, object_store::Result<Path>>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<Path>> {
            unimplemented!("delete_stream is not used by these tests")
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

        async fn copy_opts(
            &self,
            _from: &Path,
            _to: &Path,
            _options: CopyOptions,
        ) -> object_store::Result<()> {
            unimplemented!("copy_opts is not used by these tests")
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

        fn requested_blob_ranges(&self) -> Vec<Range<u64>> {
            let full_object = 0..self.data.len() as u64;
            self.requested_ranges()
                .into_iter()
                .filter(|range| range != &full_object)
                .collect()
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
        async fn put_opts(
            &self,
            _location: &Path,
            _bytes: PutPayload,
            _opts: PutOptions,
        ) -> object_store::Result<PutResult> {
            unimplemented!("put_opts is not used by these tests")
        }

        async fn put_multipart_opts(
            &self,
            _location: &Path,
            _opts: PutMultipartOptions,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            unimplemented!("put_multipart_opts is not used by these tests")
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
            let is_full_object_probe = range.start == 0 && range.end == self.data.len() as u64;
            if !is_full_object_probe && let Some(gate) = &self.gate {
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

        fn delete_stream(
            &self,
            _locations: futures::stream::BoxStream<'static, object_store::Result<Path>>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<Path>> {
            unimplemented!("delete_stream is not used by these tests")
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

        async fn copy_opts(
            &self,
            _from: &Path,
            _to: &Path,
            _options: CopyOptions,
        ) -> object_store::Result<()> {
            unimplemented!("copy_opts is not used by these tests")
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

    /// Regression test: scanning the same blob column twice on a single
    /// `Dataset` (and therefore a single cache scope) with different
    /// `BlobHandling` values must not panic in the structural decoder.
    ///
    /// Background: `FieldDataCacheKey` previously keyed cached page data only
    /// by `column_index`. A blob column has two valid decoder shapes — the
    /// descriptor view (`Struct<position, size>`) used when scanning with
    /// `BlobHandling::BlobsDescriptions`, and the bytes view (`LargeBinary`)
    /// used when scanning with `BlobHandling::AllBinary`. Both views go through
    /// the same `StructuralPrimitiveFieldScheduler` but instantiate different
    /// page-level schedulers, which cache different concrete `CachedPageData`
    /// types under the same column index. When the second view hit the cache
    /// populated by the first, it downcast the wrong state type and panicked.
    #[tokio::test]
    async fn test_blob_cache_key_distinguishes_views() {
        use crate::dataset::WriteParams;
        use arrow_array::RecordBatchIterator;
        use arrow_array::{LargeBinaryArray, UInt64Array};
        use arrow_schema::{Field, Schema};
        use lance_arrow::BLOB_META_KEY;
        use lance_core::datatypes::BlobHandling;

        let test_dir = TempStrDir::default();
        let blob_meta = HashMap::from([(BLOB_META_KEY.to_string(), "true".to_string())]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("blobs", DataType::LargeBinary, true).with_metadata(blob_meta),
            Field::new("idx", DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(LargeBinaryArray::from(vec![
                    Some(b"foo".as_slice()),
                    Some(b"bar".as_slice()),
                    Some(b"baz".as_slice()),
                ])),
                Arc::new(UInt64Array::from(vec![0u64, 1, 2])),
            ],
        )
        .unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        Dataset::write(
            reader,
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Open once and reuse the same Dataset (single Session, single cache)
        // across both scans. Reopening would defeat the test.
        let dataset = Arc::new(Dataset::open(test_dir.as_str()).await.unwrap());

        // Pass 1: descriptor view (Struct<position, size>). Default for scan().
        let mut scanner = dataset.scan();
        scanner.blob_handling(BlobHandling::BlobsDescriptions);
        let descriptors = scanner
            .project(&["blobs"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert!(descriptors.column(0).data_type().is_struct());

        // Pass 2: bytes view (LargeBinary). Used by compact_files.
        // Without the fix this used to panic in BlobPageScheduler::load
        // when it downcast the cached BlobDescriptionPageScheduler state.
        let mut scanner = dataset.scan();
        scanner.blob_handling(BlobHandling::AllBinary);
        let bytes = scanner
            .project(&["blobs"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(bytes.column(0).data_type(), &DataType::LargeBinary);
        let blobs = bytes.column(0).as_binary::<i64>();
        assert_eq!(blobs.value(0), b"foo");
        assert_eq!(blobs.value(1), b"bar");
        assert_eq!(blobs.value(2), b"baz");

        // Pass 3: back to descriptor view to verify both directions are safe.
        let mut scanner = dataset.scan();
        scanner.blob_handling(BlobHandling::BlobsDescriptions);
        let descriptors = scanner
            .project(&["blobs"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert!(descriptors.column(0).data_type().is_struct());
    }

    #[tokio::test]
    async fn test_v2_0_legacy_blob_descriptor_projection_and_reads() {
        let test_dir = TempStrDir::default();
        let blob_meta = HashMap::from([(BLOB_META_KEY.to_string(), "true".to_string())]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("blob", DataType::LargeBinary, true).with_metadata(blob_meta),
        ]));
        let payloads = [
            b"abc".as_slice(),
            b"defgh".as_slice(),
            b"ijklmnop".as_slice(),
        ];
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![0, 1, 2])) as ArrayRef,
                Arc::new(LargeBinaryArray::from_iter_values(payloads)) as ArrayRef,
            ],
        )
        .unwrap();
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch)], schema),
                &test_dir,
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_0),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        for blob_handling in [
            None,
            Some(BlobHandling::BlobsDescriptions),
            Some(BlobHandling::AllDescriptions),
        ] {
            let mut scanner = dataset.scan();
            if let Some(blob_handling) = blob_handling {
                scanner.blob_handling(blob_handling);
            }
            let descriptors = scanner
                .project(&["blob"])
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();
            let descriptor = descriptors.column(0).as_struct();
            assert_eq!(descriptor.fields().len(), 2);
            assert_eq!(descriptor.fields()[0].name(), "position");
            assert_eq!(descriptor.fields()[1].name(), "size");
            let sizes = descriptor
                .column_by_name("size")
                .unwrap()
                .as_primitive::<UInt64Type>();
            assert_eq!(sizes.values(), &[3, 5, 8]);
        }

        let mut scanner = dataset.scan();
        scanner.blob_handling(BlobHandling::AllBinary);
        let bytes = scanner
            .project(&["blob"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let bytes = bytes.column(0).as_binary::<i64>();
        for (idx, expected) in payloads.iter().enumerate() {
            assert_eq!(bytes.value(idx), *expected);
        }

        let blob_files = dataset
            .take_blobs_by_indices(&[0, 1, 2], "blob")
            .await
            .unwrap();
        assert_eq!(blob_files.len(), payloads.len());
        for (blob_file, expected) in blob_files.iter().zip(payloads) {
            assert_eq!(blob_file.read().await.unwrap().as_ref(), expected);
        }

        let read_blobs = dataset
            .read_blobs("blob")
            .unwrap()
            .with_row_indices(vec![0, 1, 2])
            .execute()
            .await
            .unwrap();
        assert_eq!(read_blobs.len(), payloads.len());
        for (read_blob, expected) in read_blobs.iter().zip(payloads) {
            assert_eq!(read_blob.data.as_ref(), expected);
        }
    }

    #[test]
    fn test_data_file_key_from_path() {
        assert_eq!(data_file_key_from_path("data/abc.lance"), "abc");
        assert_eq!(data_file_key_from_path("abc.lance"), "abc");
        assert_eq!(data_file_key_from_path("nested/path/xyz"), "xyz");
    }

    #[tokio::test]
    async fn test_write_and_take_blobs_with_blob_descriptor_array_builder() {
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
    async fn test_write_prepared_blob_column_normalizes_schema() {
        let test_dir = TempStrDir::default();
        let mut prepared_writer = BlobDescriptorArrayBuilder::new("blob");
        prepared_writer
            .push_inline(Bytes::from_static(b"prepared-inline"))
            .unwrap();
        let (prepared_field, prepared_array) = prepared_writer.finish().unwrap().into_parts();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            prepared_field,
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![0])) as ArrayRef,
                prepared_array,
            ],
        )
        .unwrap();

        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch)], schema),
                &test_dir,
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let dataset_schema = Schema::from(dataset.schema());
        let blob_field = dataset_schema.field_with_name("blob").unwrap();
        let DataType::Struct(fields) = blob_field.data_type() else {
            panic!("expected blob field to be a struct");
        };
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name(), "data");
        assert_eq!(fields[1].name(), "uri");

        let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs[0].read().await.unwrap().as_ref(), b"prepared-inline");
    }

    #[tokio::test]
    async fn test_data_replacement_uses_blob_descriptor_array_builder_prepared_packed_blob_column()
    {
        let test_dir = TempStrDir::default();
        let logical_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            blob_field("blob", true),
        ]));
        let mut initial_builder = BlobArrayBuilder::new(1);
        initial_builder.push_bytes(b"initial").unwrap();
        let initial_batch = RecordBatch::try_new(
            logical_schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![0])) as ArrayRef,
                initial_builder.finish().unwrap(),
            ],
        )
        .unwrap();
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(initial_batch)], logical_schema.clone()),
                &test_dir,
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let file_id = Uuid::new_v4().to_string();
        let data_file_name = format!("{file_id}.lance");
        let data_file_path = dataset.data_dir().join(data_file_name.as_str());
        let blob_id = 1;
        let packed_path = blob_path(&dataset.data_dir(), &file_id, blob_id);
        let mut blob_writer = BlobDescriptorArrayBuilder::new("blob");
        let mut packed = PackedBlobWriter::try_new(
            dataset.object_store.as_ref().clone(),
            data_file_path.clone(),
            blob_id,
        )
        .await
        .unwrap();
        assert_eq!(packed.path(), &packed_path);
        packed.write_blob(b"prepared-packed").await.unwrap();
        blob_writer.extend(packed.finish().await.unwrap()).unwrap();
        let prepared_field = blob_writer.field().clone();
        let prepared_array = blob_writer.finish().unwrap().into_parts().1;
        let append_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            prepared_field,
        ]));
        let replacement_batch = RecordBatch::try_new(
            append_schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![1])) as ArrayRef,
                prepared_array,
            ],
        )
        .unwrap();

        let object_writer = dataset.object_store.create(&data_file_path).await.unwrap();
        let mut file_writer = FileWriter::try_new(
            object_writer,
            crate::datatypes::Schema::try_from(append_schema.as_ref()).unwrap(),
            FileWriterOptions {
                format_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            },
        )
        .unwrap();
        file_writer.write_batch(&replacement_batch).await.unwrap();
        file_writer.finish().await.unwrap();

        let data_file = dataset
            .create_data_file(&data_file_name, None)
            .await
            .unwrap();
        let transaction = Transaction {
            read_version: dataset.manifest.version,
            uuid: Uuid::new_v4().hyphenated().to_string(),
            operation: Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, data_file)],
            },
            tag: None,
            transaction_properties: None,
        };
        let dataset = Arc::new(
            CommitBuilder::new(dataset)
                .execute(transaction)
                .await
                .unwrap(),
        );

        let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs[0].read().await.unwrap().as_ref(), b"prepared-packed");
        assert_eq!(blobs[0].kind(), BlobKind::Packed);
    }

    #[tokio::test]
    async fn test_append_uses_blob_descriptor_array_builder_prepared_blob_column() {
        let test_dir = TempStrDir::default();
        let logical_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            blob_field("blob", true),
        ]));
        let mut initial_builder = BlobArrayBuilder::new(1);
        initial_builder.push_bytes(b"initial").unwrap();
        let initial_batch = RecordBatch::try_new(
            logical_schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![0])) as ArrayRef,
                initial_builder.finish().unwrap(),
            ],
        )
        .unwrap();
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(initial_batch)], logical_schema.clone()),
                &test_dir,
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let mut blob_writer = BlobDescriptorArrayBuilder::new("blob");
        blob_writer
            .push_inline(Bytes::from_static(b"append"))
            .unwrap();
        let prepared_column = blob_writer.finish().unwrap();
        let (prepared_field, prepared_array) = prepared_column.into_parts();
        let append_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            prepared_field,
        ]));
        let append_batch = RecordBatch::try_new(
            append_schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![1])) as ArrayRef,
                prepared_array,
            ],
        )
        .unwrap();

        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(append_batch)], append_schema),
                dataset,
                Some(WriteParams {
                    mode: WriteMode::Append,
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let ids = dataset
            .scan()
            .project(&["id"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let id_values = ids.column(0).as_primitive::<UInt32Type>();
        assert_eq!(id_values.values(), &[0, 1]);

        let blobs = dataset
            .take_blobs_by_indices(&[0, 1], "blob")
            .await
            .unwrap();
        assert_eq!(blobs.len(), 2);
        assert_eq!(blobs[0].read().await.unwrap().as_ref(), b"initial");
        assert_eq!(blobs[1].read().await.unwrap().as_ref(), b"append");
    }

    #[tokio::test]
    async fn test_data_replacement_uses_blob_descriptor_array_builder_prepared_nested_blob_column()
    {
        let test_dir = TempStrDir::default();
        let mut initial_builder = BlobArrayBuilder::new(1);
        initial_builder.push_bytes(b"initial").unwrap();
        let (logical_schema, initial_batch) =
            nested_blob_v2_batch(initial_builder.finish().unwrap());
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(initial_batch)], logical_schema.clone()),
                &test_dir,
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let file_id = Uuid::new_v4().to_string();
        let data_file_name = format!("{file_id}.lance");
        let data_file_path = dataset.data_dir().join(data_file_name.as_str());
        let blob_id = 1;
        let mut blob_writer = BlobDescriptorArrayBuilder::new("blob");
        let mut packed = PackedBlobWriter::try_new(
            dataset.object_store.as_ref().clone(),
            data_file_path.clone(),
            blob_id,
        )
        .await
        .unwrap();
        packed.write_blob(b"nested-replacement").await.unwrap();
        blob_writer.extend(packed.finish().await.unwrap()).unwrap();
        let prepared_column = blob_writer.finish().unwrap();
        let (prepared_field, prepared_array) = prepared_column.into_parts();

        let info_fields = vec![Field::new("name", DataType::Utf8, false), prepared_field];
        let info_array = Arc::new(
            StructArray::try_new(
                info_fields.clone().into(),
                vec![
                    Arc::new(StringArray::from(vec!["replacement"])) as ArrayRef,
                    prepared_array,
                ],
                None,
            )
            .unwrap(),
        ) as ArrayRef;
        let replacement_schema = Arc::new(Schema::new(vec![Field::new(
            "info",
            DataType::Struct(info_fields.into()),
            true,
        )]));
        let replacement_batch =
            RecordBatch::try_new(replacement_schema.clone(), vec![info_array]).unwrap();

        let object_writer = dataset.object_store.create(&data_file_path).await.unwrap();
        let mut file_writer = FileWriter::try_new(
            object_writer,
            crate::datatypes::Schema::try_from(replacement_schema.as_ref()).unwrap(),
            FileWriterOptions {
                format_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            },
        )
        .unwrap();
        file_writer.write_batch(&replacement_batch).await.unwrap();
        file_writer.finish().await.unwrap();

        let data_file = dataset
            .create_data_file(&data_file_name, None)
            .await
            .unwrap();
        let transaction = Transaction {
            read_version: dataset.manifest.version,
            uuid: Uuid::new_v4().hyphenated().to_string(),
            operation: Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, data_file)],
            },
            tag: None,
            transaction_properties: None,
        };
        let dataset = Arc::new(
            CommitBuilder::new(dataset)
                .execute(transaction)
                .await
                .unwrap(),
        );

        let blobs = dataset
            .take_blobs_by_indices(&[0], "info.blob")
            .await
            .unwrap();
        assert_eq!(blobs.len(), 1);
        assert_eq!(
            blobs[0].read().await.unwrap().as_ref(),
            b"nested-replacement"
        );
        assert_eq!(blobs[0].kind(), BlobKind::Packed);
    }

    #[tokio::test]
    async fn test_create_data_file_skips_blob_orphans_with_top_level_name_collision() {
        let test_dir = TempStrDir::default();
        let mut initial_builder = BlobArrayBuilder::new(1);
        initial_builder.push_bytes(b"initial").unwrap();
        let (_, initial_info_batch) = nested_blob_v2_batch(initial_builder.finish().unwrap());
        let info_field = initial_info_batch.schema().field(0).as_ref().clone();
        let logical_schema = Arc::new(Schema::new(vec![
            info_field,
            Field::new("kind", DataType::UInt8, false),
        ]));
        let initial_batch = RecordBatch::try_new(
            logical_schema.clone(),
            vec![
                initial_info_batch.column(0).clone(),
                Arc::new(UInt8Array::from(vec![7])) as ArrayRef,
            ],
        )
        .unwrap();
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(initial_batch)], logical_schema.clone()),
                &test_dir,
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let file_id = Uuid::new_v4().to_string();
        let data_file_name = format!("{file_id}.lance");
        let data_file_path = dataset.data_dir().join(data_file_name.as_str());
        let mut blob_writer = BlobDescriptorArrayBuilder::new("blob");
        blob_writer
            .push_inline(Bytes::from_static(b"replacement"))
            .unwrap();
        let prepared_column = blob_writer.finish().unwrap();
        let (prepared_field, prepared_array) = prepared_column.into_parts();

        let info_fields = vec![Field::new("name", DataType::Utf8, false), prepared_field];
        let info_array = Arc::new(
            StructArray::try_new(
                info_fields.clone().into(),
                vec![
                    Arc::new(StringArray::from(vec!["replacement"])) as ArrayRef,
                    prepared_array,
                ],
                None,
            )
            .unwrap(),
        ) as ArrayRef;
        let replacement_schema = Arc::new(Schema::new(vec![Field::new(
            "info",
            DataType::Struct(info_fields.into()),
            true,
        )]));
        let replacement_batch =
            RecordBatch::try_new(replacement_schema.clone(), vec![info_array]).unwrap();

        let object_writer = dataset.object_store.create(&data_file_path).await.unwrap();
        let mut file_writer = FileWriter::try_new(
            object_writer,
            crate::datatypes::Schema::try_from(replacement_schema.as_ref()).unwrap(),
            FileWriterOptions {
                format_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            },
        )
        .unwrap();
        file_writer.write_batch(&replacement_batch).await.unwrap();
        file_writer.finish().await.unwrap();

        let data_file = dataset
            .create_data_file(&data_file_name, None)
            .await
            .unwrap();

        assert_eq!(data_file.fields.len(), 2);
        assert_eq!(data_file.column_indices.as_ref(), &[0, 1]);
    }

    #[tokio::test]
    async fn test_write_and_take_nested_blob_v2() {
        let test_dir = TempStrDir::default();
        let packed_payload = vec![0x4A; super::INLINE_MAX + 1024];

        let mut blob_builder = BlobArrayBuilder::new(3);
        blob_builder.push_bytes(b"hello").unwrap();
        blob_builder.push_bytes(&packed_payload).unwrap();
        blob_builder.push_null().unwrap();
        let blob_array: ArrayRef = blob_builder.finish().unwrap();

        let (schema, batch) = nested_blob_v2_batch(blob_array);
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        let dataset = Arc::new(
            Dataset::write(
                reader,
                &test_dir,
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let info_batch = dataset
            .scan()
            .project(&["info"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let blob_desc = info_batch
            .column(0)
            .as_struct()
            .column_by_name("blob")
            .unwrap()
            .as_struct();
        assert_eq!(
            blob_desc
                .column_by_name("kind")
                .unwrap()
                .as_primitive::<UInt8Type>()
                .value(0),
            BlobKind::Inline as u8
        );
        assert_eq!(
            blob_desc
                .column_by_name("kind")
                .unwrap()
                .as_primitive::<UInt8Type>()
                .value(1),
            BlobKind::Packed as u8
        );

        let blobs = dataset
            .take_blobs_by_indices(&[0, 1], "info.blob")
            .await
            .unwrap();
        assert_eq!(blobs.len(), 2);
        assert_eq!(blobs[0].read().await.unwrap().as_ref(), b"hello");
        assert_eq!(
            blobs[1].read().await.unwrap().as_ref(),
            packed_payload.as_slice()
        );

        let null_blobs = dataset
            .take_blobs_by_indices(&[2], "info.blob")
            .await
            .unwrap();
        assert!(null_blobs.is_empty());

        let filtered = dataset
            .scan()
            .project(&["info"])
            .unwrap()
            .filter("info.blob IS NOT NULL")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(filtered.num_rows(), 2);
    }

    #[tokio::test]
    async fn test_write_and_scan_list_blob_v2_descriptions() {
        let test_dir = TempStrDir::default();
        let packed_payload = vec![0x4B; super::INLINE_MAX + 1024];

        let mut blob_builder = BlobArrayBuilder::new(4);
        blob_builder.push_bytes(b"hello").unwrap();
        blob_builder.push_null().unwrap();
        blob_builder.push_bytes(&packed_payload).unwrap();
        blob_builder.push_bytes(b"tail").unwrap();
        let blob_values = blob_builder.finish().unwrap();

        let item_field = Arc::new(blob_field("item", true));
        let list_array: ArrayRef = Arc::new(
            arrow_array::ListArray::try_new(
                item_field.clone(),
                arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(vec![
                    0i32, 3, 3, 3, 4,
                ])),
                blob_values,
                Some(arrow_buffer::NullBuffer::from(vec![
                    true, true, false, true,
                ])),
            )
            .unwrap(),
        );

        let schema = Arc::new(Schema::new(vec![
            Field::new("blobs", DataType::List(item_field), true),
            Field::new("id", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![list_array, Arc::new(Int32Array::from(vec![0, 1, 2, 3]))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        let dataset = Arc::new(
            Dataset::write(
                reader,
                &test_dir,
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    enable_stable_row_ids: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let descriptions = dataset
            .scan()
            .project(&["blobs"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let lists = descriptions.column(0).as_list::<i32>();
        assert_eq!(lists.offsets().inner().as_ref(), &[0, 3, 3, 3, 4]);
        assert!(lists.is_valid(0));
        assert!(lists.is_valid(1));
        assert!(lists.is_null(2));
        assert!(lists.is_valid(3));

        let DataType::List(descriptor_field) = lists.data_type() else {
            panic!("unexpected list type: {}", lists.data_type());
        };
        assert!(matches!(descriptor_field.data_type(), DataType::Struct(_)));
        assert!(!descriptor_field.metadata().contains_key(ARROW_EXT_NAME_KEY));
        let descriptors = lists.values().as_struct();
        assert_eq!(descriptors.fields().len(), 5);
        assert_eq!(descriptors.fields()[0].name(), "kind");
        assert!(descriptors.is_valid(0));
        assert!(descriptors.is_null(1));
        assert!(descriptors.is_valid(2));
        assert!(descriptors.is_valid(3));
        let kinds = descriptors
            .column_by_name("kind")
            .unwrap()
            .as_primitive::<UInt8Type>();
        assert_eq!(kinds.value(0), BlobKind::Inline as u8);
        assert_eq!(kinds.value(2), BlobKind::Packed as u8);
        assert_eq!(kinds.value(3), BlobKind::Inline as u8);

        let filtered = dataset
            .scan()
            .project(&["blobs"])
            .unwrap()
            .filter("blobs IS NOT NULL")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(filtered.num_rows(), 3);

        let mut scanner = dataset.scan();
        scanner.blob_handling(BlobHandling::AllBinary);
        let bytes = scanner
            .project(&["blobs"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let lists = bytes.column(0).as_list::<i32>();
        assert_eq!(lists.offsets().inner().as_ref(), &[0, 3, 3, 3, 4]);
        assert!(lists.is_valid(0));
        assert!(lists.is_valid(1));
        assert!(lists.is_null(2));
        assert!(lists.is_valid(3));
        let DataType::List(value_field) = lists.data_type() else {
            panic!("unexpected list type: {}", lists.data_type());
        };
        assert_eq!(value_field.data_type(), &DataType::LargeBinary);
        assert!(!value_field.metadata().contains_key(ARROW_EXT_NAME_KEY));
        let values = lists.values().as_binary::<i64>();
        assert_eq!(values.value(0), b"hello");
        assert!(values.is_null(1));
        assert_eq!(values.value(2), packed_payload.as_slice());
        assert_eq!(values.value(3), b"tail");

        for (filter, materialization_style) in [
            (None, MaterializationStyle::Heuristic),
            (Some("id >= 2"), MaterializationStyle::Heuristic),
            (Some("id >= 2"), MaterializationStyle::AllEarly),
        ] {
            let mut scanner = dataset.scan();
            scanner.blob_handling(BlobHandling::AllBinary);
            scanner.materialization_style(materialization_style);
            scanner
                .project(&["blobs", ROW_LAST_UPDATED_AT_VERSION, ROW_CREATED_AT_VERSION])
                .unwrap()
                .with_row_id()
                .with_row_address();
            if let Some(filter) = filter {
                scanner.filter(filter).unwrap();
            }

            let expected_schema = scanner.schema().await.unwrap();
            let batch = scanner.try_into_batch().await.unwrap();
            assert_eq!(batch.schema().as_ref(), expected_schema.as_ref());
            assert_eq!(batch.num_rows(), if filter.is_some() { 2 } else { 4 });
            for column in [
                ROW_ID,
                ROW_ADDR,
                ROW_LAST_UPDATED_AT_VERSION,
                ROW_CREATED_AT_VERSION,
            ] {
                assert!(
                    batch.column_by_name(column).is_some(),
                    "requested system column {column} was missing"
                );
            }
        }
    }

    #[tokio::test]
    async fn test_write_and_scan_struct_nested_list_blob_v2() {
        let test_dir = TempStrDir::default();

        let mut blob_builder = BlobArrayBuilder::new(2);
        blob_builder.push_bytes(b"nested").unwrap();
        blob_builder.push_null().unwrap();
        let blob_values = blob_builder.finish().unwrap();

        let item_field = Arc::new(blob_field("item", true));
        let list_field = Field::new("blobs", DataType::List(item_field.clone()), true);
        let list_array: ArrayRef = Arc::new(
            arrow_array::ListArray::try_new(
                item_field,
                arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(vec![0i32, 2, 2])),
                blob_values,
                None,
            )
            .unwrap(),
        );
        let info_fields = vec![Field::new("name", DataType::Utf8, false), list_field];
        let info_array: ArrayRef = Arc::new(
            StructArray::try_new(
                info_fields.clone().into(),
                vec![
                    Arc::new(StringArray::from(vec!["row-0", "row-1"])) as ArrayRef,
                    list_array,
                ],
                None,
            )
            .unwrap(),
        );

        let schema = Arc::new(Schema::new(vec![Field::new(
            "info",
            DataType::Struct(info_fields.into()),
            true,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![info_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        let dataset = Arc::new(
            Dataset::write(
                reader,
                &test_dir,
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let descriptions = dataset
            .scan()
            .project(&["info"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let info = descriptions.column(0).as_struct();
        assert_eq!(
            info.column_by_name("name")
                .unwrap()
                .as_string::<i32>()
                .value(0),
            "row-0"
        );
        let lists = info.column_by_name("blobs").unwrap().as_list::<i32>();
        assert_eq!(lists.offsets().inner().as_ref(), &[0, 2, 2]);
        let DataType::List(descriptor_field) = lists.data_type() else {
            panic!("unexpected nested list type: {}", lists.data_type());
        };
        assert!(matches!(descriptor_field.data_type(), DataType::Struct(_)));
        assert!(!descriptor_field.metadata().contains_key(ARROW_EXT_NAME_KEY));
        let descriptors = lists.values().as_struct();
        assert_eq!(descriptors.fields().len(), 5);
        assert!(descriptors.is_valid(0));
        assert!(descriptors.is_null(1));

        let mut scanner = dataset.scan();
        scanner.blob_handling(BlobHandling::AllBinary);
        let bytes = scanner
            .project(&["info"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let info = bytes.column(0).as_struct();
        let lists = info.column_by_name("blobs").unwrap().as_list::<i32>();
        assert_eq!(lists.offsets().inner().as_ref(), &[0, 2, 2]);
        let DataType::List(value_field) = lists.data_type() else {
            panic!("unexpected nested list type: {}", lists.data_type());
        };
        assert_eq!(value_field.data_type(), &DataType::LargeBinary);
        assert!(!value_field.metadata().contains_key(ARROW_EXT_NAME_KEY));
        let values = lists.values().as_binary::<i64>();
        assert_eq!(values.value(0), b"nested");
        assert!(values.is_null(1));
    }

    #[tokio::test]
    async fn test_nested_blob_v2_requires_v2_2() {
        let test_dir = TempStrDir::default();

        let mut blob_builder = BlobArrayBuilder::new(1);
        blob_builder.push_bytes(b"hello").unwrap();
        let blob_array: ArrayRef = blob_builder.finish().unwrap();

        let (schema, batch) = nested_blob_v2_batch(blob_array);
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
            "Nested blob v2 should be rejected for file version 2.1"
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Blob v2 requires file version >= 2.2")
        );
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
        assert_eq!(inner.requested_blob_ranges(), vec![1..7]);
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
        assert_eq!(inner.requested_blob_ranges(), vec![1..7]);
    }

    #[tokio::test]
    async fn test_execute_blob_entries_preserves_order_and_coalesces() {
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
        let mut blobs = execute_blob_entries(entries, 2, None).await.unwrap();
        blobs.sort_by_key(|blob| blob.selection_index);

        assert_eq!(blobs.len(), 2);
        assert_eq!(blobs[0].row_address, 10);
        assert_eq!(blobs[0].data.as_ref(), b"efg");
        assert_eq!(blobs[1].row_address, 11);
        assert_eq!(blobs[1].data.as_ref(), b"bcd");
        assert_eq!(inner.requested_blob_ranges(), vec![1..7]);
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
    async fn test_blob_v2_external_ingest_respects_inline_threshold() {
        let dataset_dir = TempDir::default();
        let external_dir = TempDir::default();
        let external_path = external_dir.std_path().join("external.bin");
        let payload = vec![0x5A; 2048];
        std::fs::write(&external_path, &payload).unwrap();
        let external_uri = format!("file://{}", external_path.display());

        let mut blob_builder = BlobArrayBuilder::new(1);
        blob_builder.push_uri(external_uri).unwrap();
        let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();

        let mut field = blob_field("blob", true);
        let mut metadata = field.metadata().clone();
        metadata.insert(
            BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
            "1024".to_string(),
        );
        field = field.with_metadata(metadata);
        let schema = Arc::new(Schema::new(vec![field]));
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

    async fn try_preprocess_blobs_with_blob_metadata(
        metadata_entries: Vec<(&'static str, String)>,
        pack_file_size_override: Option<usize>,
        blob_len: usize,
        num_blobs: usize,
    ) -> Result<arrow_array::StructArray> {
        let (object_store, base_path) = ObjectStore::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            "memory://blob_preprocessor",
            &ObjectStoreParams::default(),
        )
        .await
        .unwrap();
        let object_store = object_store.as_ref().clone();
        let data_dir = base_path.clone().join("data");

        let mut field = blob_field("blob", true);
        let mut metadata = field.metadata().clone();
        for (key, value) in metadata_entries {
            metadata.insert(key.to_string(), value);
        }
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
            pack_file_size_override,
        )?;

        let mut blob_builder = BlobArrayBuilder::new(num_blobs);
        for _ in 0..num_blobs {
            blob_builder.push_bytes(vec![0u8; blob_len]).unwrap();
        }
        let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();

        let field_without_metadata =
            Field::new("blob", field.data_type().clone(), field.is_nullable());
        let batch_schema = Arc::new(Schema::new(vec![field_without_metadata]));
        let batch = RecordBatch::try_new(batch_schema, vec![blob_array]).unwrap();

        let out = preprocessor.preprocess_batch(&batch).await?;
        Ok(out
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .unwrap()
            .clone())
    }

    async fn try_preprocess_kind_with_blob_metadata(
        metadata_entries: Vec<(&'static str, String)>,
        data_len: usize,
    ) -> Result<u8> {
        let struct_arr =
            try_preprocess_blobs_with_blob_metadata(metadata_entries, None, data_len, 1).await?;
        Ok(struct_arr
            .column_by_name("kind")
            .unwrap()
            .as_primitive::<arrow::datatypes::UInt8Type>()
            .value(0))
    }

    async fn preprocess_kind_with_blob_metadata(
        metadata_entries: Vec<(&'static str, String)>,
        data_len: usize,
    ) -> u8 {
        try_preprocess_kind_with_blob_metadata(metadata_entries, data_len)
            .await
            .unwrap()
    }

    async fn packed_blobs_with_blob_metadata(
        metadata_entries: Vec<(&'static str, String)>,
        pack_file_size_override: Option<usize>,
        blob_len: usize,
        num_blobs: usize,
    ) -> Vec<u32> {
        let struct_arr = try_preprocess_blobs_with_blob_metadata(
            metadata_entries,
            pack_file_size_override,
            blob_len,
            num_blobs,
        )
        .await
        .unwrap();
        let blob_ids = struct_arr
            .column_by_name("blob_id")
            .unwrap()
            .as_primitive::<arrow::datatypes::UInt32Type>();
        (0..struct_arr.len()).map(|i| blob_ids.value(i)).collect()
    }

    #[tokio::test]
    async fn test_blob_v2_dedicated_threshold_rejects_non_positive_metadata() {
        let err = try_preprocess_kind_with_blob_metadata(
            vec![(BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY, "0".to_string())],
            256 * 1024,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("expected a positive integer"));
    }

    #[tokio::test]
    async fn test_blob_v2_inline_threshold_rejects_invalid_metadata() {
        let err = try_preprocess_kind_with_blob_metadata(
            vec![(
                BLOB_INLINE_SIZE_THRESHOLD_META_KEY,
                "not-a-number".to_string(),
            )],
            256 * 1024,
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("expected a non-negative integer that fits in usize")
        );
    }

    #[tokio::test]
    async fn test_blob_v2_write_rejects_invalid_inline_threshold_metadata() {
        let dataset_dir = TempDir::default();
        let mut field = blob_field("blob", true);
        let mut metadata = field.metadata().clone();
        metadata.insert(
            BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
            "not-a-number".to_string(),
        );
        field = field.with_metadata(metadata);
        let schema = Arc::new(Schema::new(vec![field]));

        let mut blob_builder = BlobArrayBuilder::new(1);
        blob_builder.push_bytes(vec![0u8; 256]).unwrap();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(blob_builder.finish().unwrap()) as ArrayRef],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);

        let result = Dataset::write(
            reader,
            &dataset_dir.path_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await;
        let Err(err) = result else {
            panic!("write with invalid blob threshold metadata should fail");
        };
        assert!(
            err.to_string()
                .contains("expected a non-negative integer that fits in usize")
        );
    }

    #[tokio::test]
    async fn test_blob_v2_dedicated_threshold_respects_smaller_metadata() {
        let kind = preprocess_kind_with_blob_metadata(
            vec![(BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY, "131072".to_string())],
            256 * 1024,
        )
        .await;
        assert_eq!(kind, lance_core::datatypes::BlobKind::Dedicated as u8);
    }

    #[tokio::test]
    async fn test_blob_v2_dedicated_threshold_respects_larger_metadata() {
        let kind = preprocess_kind_with_blob_metadata(
            vec![(
                BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY,
                "8388608".to_string(),
            )],
            super::DEDICATED_THRESHOLD + 1024,
        )
        .await;
        assert_eq!(kind, lance_core::datatypes::BlobKind::Packed as u8);
    }

    #[tokio::test]
    async fn test_blob_v2_pack_file_threshold_rolls_at_metadata_value() {
        // Blobs must exceed the inline cutoff to be packed at all. With a pack-file
        // threshold equal to a single blob's size, each of the three blobs rolls to its
        // own pack file (a distinct blob_id).
        let blob_len = super::INLINE_MAX + 1024;
        let blob_ids = packed_blobs_with_blob_metadata(
            vec![(BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY, blob_len.to_string())],
            None,
            blob_len,
            3,
        )
        .await;
        let distinct: std::collections::HashSet<u32> = blob_ids.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            3,
            "expected one pack file per blob: {blob_ids:?}"
        );
    }

    #[tokio::test]
    async fn test_blob_v2_pack_file_threshold_packs_within_metadata_value() {
        // A pack-file threshold large enough for all three blobs keeps them in a single
        // pack file (one shared blob_id).
        let blob_len = super::INLINE_MAX + 1024;
        let blob_ids = packed_blobs_with_blob_metadata(
            vec![(
                BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY,
                (blob_len * 3).to_string(),
            )],
            None,
            blob_len,
            3,
        )
        .await;
        let distinct: std::collections::HashSet<u32> = blob_ids.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            1,
            "expected a single shared pack file: {blob_ids:?}"
        );
    }

    #[tokio::test]
    async fn test_blob_v2_pack_file_threshold_rejects_non_positive_metadata() {
        let err = try_preprocess_kind_with_blob_metadata(
            vec![(BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY, "0".to_string())],
            1024,
        )
        .await
        .expect_err("zero pack-file threshold should be rejected");
        assert!(err.to_string().contains("expected a positive integer"));
    }

    #[tokio::test]
    async fn test_blob_v2_inline_threshold_respects_smaller_metadata() {
        let kind = preprocess_kind_with_blob_metadata(
            vec![(BLOB_INLINE_SIZE_THRESHOLD_META_KEY, "1024".to_string())],
            2048,
        )
        .await;
        assert_eq!(kind, lance_core::datatypes::BlobKind::Packed as u8);
    }

    #[tokio::test]
    async fn test_blob_v2_inline_threshold_respects_larger_metadata() {
        let kind = preprocess_kind_with_blob_metadata(
            vec![(
                BLOB_INLINE_SIZE_THRESHOLD_META_KEY,
                (super::INLINE_MAX + 8192).to_string(),
            )],
            super::INLINE_MAX + 4096,
        )
        .await;
        assert_eq!(kind, lance_core::datatypes::BlobKind::Inline as u8);
    }

    #[tokio::test]
    async fn test_blob_v2_inline_threshold_uses_strict_greater_than() {
        let kind = preprocess_kind_with_blob_metadata(
            vec![(BLOB_INLINE_SIZE_THRESHOLD_META_KEY, "1024".to_string())],
            1024,
        )
        .await;
        assert_eq!(kind, lance_core::datatypes::BlobKind::Inline as u8);
    }

    #[tokio::test]
    async fn test_blob_v2_dedicated_threshold_uses_strict_greater_than() {
        let kind = preprocess_kind_with_blob_metadata(
            vec![
                (BLOB_INLINE_SIZE_THRESHOLD_META_KEY, "2048".to_string()),
                (BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY, "1024".to_string()),
            ],
            1024,
        )
        .await;
        assert_eq!(kind, lance_core::datatypes::BlobKind::Inline as u8);
    }

    #[tokio::test]
    async fn test_blob_v2_inline_threshold_does_not_override_dedicated_threshold() {
        let kind = preprocess_kind_with_blob_metadata(
            vec![
                (BLOB_INLINE_SIZE_THRESHOLD_META_KEY, "8192".to_string()),
                (BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY, "4096".to_string()),
            ],
            6144,
        )
        .await;
        assert_eq!(kind, lance_core::datatypes::BlobKind::Dedicated as u8);
    }

    #[tokio::test]
    async fn test_blob_v2_inline_threshold_is_per_column() {
        let (object_store, base_path) = ObjectStore::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            "memory://blob_preprocessor",
            &ObjectStoreParams::default(),
        )
        .await
        .unwrap();
        let object_store = object_store.as_ref().clone();
        let data_dir = base_path.clone().join("data");

        let mut inline_field = blob_field("inline_blob", true);
        let mut inline_metadata = inline_field.metadata().clone();
        inline_metadata.insert(
            BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
            "4096".to_string(),
        );
        inline_field = inline_field.with_metadata(inline_metadata);

        let mut packed_field = blob_field("packed_blob", true);
        let mut packed_metadata = packed_field.metadata().clone();
        packed_metadata.insert(
            BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
            "1024".to_string(),
        );
        packed_field = packed_field.with_metadata(packed_metadata);

        let writer_arrow_schema = Schema::new(vec![inline_field.clone(), packed_field.clone()]);
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
        )
        .unwrap();

        let mut inline_builder = BlobArrayBuilder::new(1);
        inline_builder.push_bytes(vec![0u8; 2048]).unwrap();
        let inline_array: arrow_array::ArrayRef = inline_builder.finish().unwrap();

        let mut packed_builder = BlobArrayBuilder::new(1);
        packed_builder.push_bytes(vec![0u8; 2048]).unwrap();
        let packed_array: arrow_array::ArrayRef = packed_builder.finish().unwrap();

        let batch_schema = Arc::new(Schema::new(vec![
            Field::new(
                "inline_blob",
                inline_field.data_type().clone(),
                inline_field.is_nullable(),
            ),
            Field::new(
                "packed_blob",
                packed_field.data_type().clone(),
                packed_field.is_nullable(),
            ),
        ]));
        let batch = RecordBatch::try_new(batch_schema, vec![inline_array, packed_array]).unwrap();

        let out = preprocessor.preprocess_batch(&batch).await.unwrap();
        let inline_kind = out
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .unwrap()
            .column_by_name("kind")
            .unwrap()
            .as_primitive::<arrow::datatypes::UInt8Type>()
            .value(0);
        let packed_kind = out
            .column(1)
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .unwrap()
            .column_by_name("kind")
            .unwrap()
            .as_primitive::<arrow::datatypes::UInt8Type>()
            .value(0);

        assert_eq!(inline_kind, lance_core::datatypes::BlobKind::Inline as u8);
        assert_eq!(packed_kind, lance_core::datatypes::BlobKind::Packed as u8);
    }

    #[tokio::test]
    async fn test_blob_v2_pack_file_threshold_starts_new_packed_blob() {
        let (object_store, base_path) = ObjectStore::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            "memory://blob_preprocessor",
            &ObjectStoreParams::default(),
        )
        .await
        .unwrap();
        let object_store = object_store.as_ref().clone();
        let data_dir = base_path.clone().join("data");

        let mut field = blob_field("blob", true);
        let mut metadata = field.metadata().clone();
        metadata.insert(
            BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
            "0".to_string(),
        );
        metadata.insert(
            BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.to_string(),
            "1024".to_string(),
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
            Some(3),
        )
        .unwrap();

        let mut blob_builder = BlobArrayBuilder::new(3);
        blob_builder.push_bytes(vec![0x11; 2]).unwrap();
        blob_builder.push_bytes(vec![0x22; 2]).unwrap();
        blob_builder.push_bytes(vec![0x33; 1]).unwrap();
        let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();

        let batch_schema = Arc::new(Schema::new(vec![Field::new(
            "blob",
            field.data_type().clone(),
            field.is_nullable(),
        )]));
        let batch = RecordBatch::try_new(batch_schema, vec![blob_array]).unwrap();

        let out = preprocessor.preprocess_batch(&batch).await.unwrap();
        preprocessor.finish().await.unwrap();

        let struct_arr = out
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .unwrap();
        let kinds = struct_arr
            .column_by_name("kind")
            .unwrap()
            .as_primitive::<UInt8Type>();
        let blob_ids = struct_arr
            .column_by_name("blob_id")
            .unwrap()
            .as_primitive::<UInt32Type>();
        let sizes = struct_arr
            .column_by_name("blob_size")
            .unwrap()
            .as_primitive::<UInt64Type>();
        let positions = struct_arr
            .column_by_name("position")
            .unwrap()
            .as_primitive::<UInt64Type>();

        assert_eq!(kinds.values(), &[BlobKind::Packed as u8; 3]);
        assert_eq!(blob_ids.values(), &[1, 2, 2]);
        assert_eq!(sizes.values(), &[2, 2, 1]);
        assert_eq!(positions.values(), &[0, 0, 2]);
    }

    #[tokio::test]
    async fn test_blob_v2_append_rejects_explicit_inline_threshold_mismatch() {
        let dataset_dir = TempDir::default();
        let payload = vec![0u8; 2048];

        let schema = Arc::new(Schema::new(vec![blob_field("blob", true)]));
        let mut initial_builder = BlobArrayBuilder::new(1);
        initial_builder.push_bytes(payload.clone()).unwrap();
        let initial_batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(initial_builder.finish().unwrap()) as ArrayRef],
        )
        .unwrap();
        let initial_reader = RecordBatchIterator::new(vec![Ok(initial_batch)], schema);
        let dataset = Dataset::write(
            initial_reader,
            &dataset_dir.path_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let mut append_field = blob_field("blob", true);
        let mut append_metadata = append_field.metadata().clone();
        append_metadata.insert(
            BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
            "1024".to_string(),
        );
        append_field = append_field.with_metadata(append_metadata);
        let append_schema = Arc::new(Schema::new(vec![append_field]));
        let mut append_builder = BlobArrayBuilder::new(1);
        append_builder.push_bytes(payload).unwrap();
        let append_batch = RecordBatch::try_new(
            append_schema.clone(),
            vec![Arc::new(append_builder.finish().unwrap()) as ArrayRef],
        )
        .unwrap();
        let append_reader = RecordBatchIterator::new(vec![Ok(append_batch)], append_schema);

        let result = Dataset::write(
            append_reader,
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await;
        let Err(err) = result else {
            panic!("append with explicit blob threshold mismatch should fail");
        };
        let message = err.to_string();
        assert!(message.contains("Cannot append data with blob threshold metadata"));
        assert!(message.contains(BLOB_INLINE_SIZE_THRESHOLD_META_KEY));
    }

    #[tokio::test]
    async fn test_blob_v2_append_rejects_threshold_mismatch_with_non_blob_input_extension() {
        let dataset_dir = TempDir::default();
        let payload = vec![0u8; 2048];

        let schema = Arc::new(Schema::new(vec![blob_field("blob", true)]));
        let mut initial_builder = BlobArrayBuilder::new(1);
        initial_builder.push_bytes(payload.clone()).unwrap();
        let initial_batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(initial_builder.finish().unwrap()) as ArrayRef],
        )
        .unwrap();
        let initial_reader = RecordBatchIterator::new(vec![Ok(initial_batch)], schema);
        let dataset = Dataset::write(
            initial_reader,
            &dataset_dir.path_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let mut append_field = blob_field("blob", true);
        let mut append_metadata = append_field.metadata().clone();
        append_metadata.insert(
            ARROW_EXT_NAME_KEY.to_string(),
            "some.other.extension".to_string(),
        );
        append_metadata.insert(
            BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
            "1024".to_string(),
        );
        append_field = append_field.with_metadata(append_metadata);
        let append_schema = Arc::new(Schema::new(vec![append_field]));
        let mut append_builder = BlobArrayBuilder::new(1);
        append_builder.push_bytes(payload).unwrap();
        let append_batch = RecordBatch::try_new(
            append_schema.clone(),
            vec![Arc::new(append_builder.finish().unwrap()) as ArrayRef],
        )
        .unwrap();
        let append_reader = RecordBatchIterator::new(vec![Ok(append_batch)], append_schema);

        let result = Dataset::write(
            append_reader,
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await;
        let Err(err) = result else {
            panic!("append with ignored blob threshold metadata should fail");
        };
        let message = err.to_string();
        assert!(message.contains("Cannot append data with blob threshold metadata"));
        assert!(message.contains(BLOB_INLINE_SIZE_THRESHOLD_META_KEY));
    }

    #[tokio::test]
    async fn test_blob_v2_append_accepts_explicit_default_inline_threshold() {
        let dataset_dir = TempDir::default();
        let payload = vec![0u8; 2048];

        let schema = Arc::new(Schema::new(vec![blob_field("blob", true)]));
        let mut initial_builder = BlobArrayBuilder::new(1);
        initial_builder.push_bytes(payload.clone()).unwrap();
        let initial_batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(initial_builder.finish().unwrap()) as ArrayRef],
        )
        .unwrap();
        let initial_reader = RecordBatchIterator::new(vec![Ok(initial_batch)], schema);
        let dataset = Dataset::write(
            initial_reader,
            &dataset_dir.path_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let mut append_field = blob_field("blob", true);
        let mut append_metadata = append_field.metadata().clone();
        append_metadata.insert(
            BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
            super::INLINE_MAX.to_string(),
        );
        append_field = append_field.with_metadata(append_metadata);
        let append_schema = Arc::new(Schema::new(vec![append_field]));
        let mut append_builder = BlobArrayBuilder::new(1);
        append_builder.push_bytes(payload).unwrap();
        let append_batch = RecordBatch::try_new(
            append_schema.clone(),
            vec![Arc::new(append_builder.finish().unwrap()) as ArrayRef],
        )
        .unwrap();
        let append_reader = RecordBatchIterator::new(vec![Ok(append_batch)], append_schema);

        let dataset = Dataset::write(
            append_reader,
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 2);
    }

    #[tokio::test]
    async fn test_blob_v2_pack_file_threshold_write_param_overrides_metadata() {
        let blob_len = super::INLINE_MAX + 1024;
        let blob_ids = packed_blobs_with_blob_metadata(
            vec![(
                BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY,
                (blob_len * 3).to_string(),
            )],
            Some(blob_len),
            blob_len,
            3,
        )
        .await;
        let distinct: std::collections::HashSet<u32> = blob_ids.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            3,
            "write-param override should force one pack file per blob: {blob_ids:?}"
        );
    }
}
