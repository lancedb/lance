// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Columnar Lance-file container for dataset manifests.

use std::fmt::{Debug, Formatter};
use std::ops::Range;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll};

use arrow_array::builder::{BinaryBuilder, Int32Builder, ListBuilder};
use arrow_array::{
    Array, BinaryArray, FixedSizeBinaryArray, Int32Array, ListArray, StringArray, StructArray,
    UInt8Array, UInt32Array, UInt64Array,
};
use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema};
use byteorder::{ByteOrder, LittleEndian};
use bytes::{Bytes, BytesMut};
use futures::{FutureExt, TryStreamExt, future::BoxFuture};
use lance_core::cache::LanceCache;
use lance_core::datatypes::Schema;
use lance_core::deepsize::{Context as DeepSizeContext, DeepSizeOf};
use lance_core::{Error, Result};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::reader::{FileReader, FileReaderOptions, ReaderProjection};
use lance_file::version::LanceFileVersion;
use lance_file::writer::{FileWriter, FileWriterOptions};
use lance_io::ReadBatchParams;
use lance_io::object_store::ObjectStore;
use lance_io::object_writer::WriteResult;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::traits::{ByteStream, Reader, Writer};
use lance_io::utils::CachedFileSize;
use object_store::path::Path;
use prost::Message;
use tokio::io::AsyncWrite;

use crate::format::{
    DataFile, DataFileFieldInterner, DeletionFile, DeletionFileType, ExternalFile, Fragment,
    IndexMetadata, Manifest, RowDatasetVersionMeta, RowIdMeta, Transaction, pb,
};

const FORMAT_IDENTITY: &str = "lance-columnar-manifest";
const FORMAT_METADATA_KEY: &str = "lance:manifest_format";
const FORMAT_VERSION_METADATA_KEY: &str = "lance:manifest_schema_version";
const MANIFEST_SCHEMA_VERSION: u32 = 1;
const SECTION_BATCH_SIZE: usize = 64 * 1024;
const SECTION_VARIABLE_BYTES: usize = 32 * 1024 * 1024;
const SMALL_MANIFEST_PREFETCH_BYTES: usize = 512 * 1024;
const MAX_ARROW_OFFSET_BYTES: usize = i32::MAX as usize;

const HEADER_COLUMN: usize = 0;
const FRAGMENTS_COLUMN: usize = 1;
const DATA_FILES_COLUMN: usize = 2;
const INDICES_COLUMN: usize = 3;
const TRANSACTION_COLUMN: usize = 4;

fn checked_variable_bytes(current: usize, additional: usize, context: &str) -> Result<usize> {
    let total = current.checked_add(additional).ok_or_else(|| {
        Error::invalid_input(format!("{context} variable-width data overflows usize"))
    })?;
    if total > MAX_ARROW_OFFSET_BYTES {
        return Err(Error::invalid_input(format!(
            "{context} variable-width data exceeds Arrow's i32 offset limit"
        )));
    }
    Ok(total)
}

fn should_flush_batch(row_count: usize, variable_bytes: usize, next_bytes: usize) -> Result<bool> {
    checked_variable_bytes(0, next_bytes, "manifest row")?;
    let combined = variable_bytes.checked_add(next_bytes).ok_or_else(|| {
        Error::invalid_input("manifest batch variable-width data overflows usize")
    })?;
    Ok(row_count != 0 && (row_count >= SECTION_BATCH_SIZE || combined > SECTION_VARIABLE_BYTES))
}

fn sequence_variable_bytes(value: Option<SequenceRef<'_>>) -> usize {
    match value {
        Some(SequenceRef::Inline(data)) => data.len(),
        Some(SequenceRef::External(file)) => file.path.len(),
        None => 0,
    }
}

fn fragment_variable_bytes(fragment: &Fragment) -> Result<usize> {
    let mut bytes = 0;
    bytes = checked_variable_bytes(
        bytes,
        sequence_variable_bytes(fragment.row_id_meta.as_ref().map(|meta| match meta {
            RowIdMeta::Inline(data) => SequenceRef::Inline(data),
            RowIdMeta::External(file) => SequenceRef::External(file),
        })),
        "manifest fragment",
    )?;
    bytes = checked_variable_bytes(
        bytes,
        sequence_variable_bytes(fragment.last_updated_at_version_meta.as_ref().map(
            |meta| match meta {
                RowDatasetVersionMeta::Inline(data) => SequenceRef::Inline(data),
                RowDatasetVersionMeta::External(file) => SequenceRef::External(file),
            },
        )),
        "manifest fragment",
    )?;
    checked_variable_bytes(
        bytes,
        sequence_variable_bytes(
            fragment
                .created_at_version_meta
                .as_ref()
                .map(|meta| match meta {
                    RowDatasetVersionMeta::Inline(data) => SequenceRef::Inline(data),
                    RowDatasetVersionMeta::External(file) => SequenceRef::External(file),
                }),
        ),
        "manifest fragment",
    )
}

fn fragment_batch_ranges(fragments: &[Fragment]) -> Result<Vec<Range<usize>>> {
    let mut ranges = Vec::with_capacity(fragments.len().div_ceil(SECTION_BATCH_SIZE));
    let mut start = 0;
    let mut variable_bytes = 0;
    for (end, fragment) in fragments.iter().enumerate() {
        let next_bytes = fragment_variable_bytes(fragment)?;
        if should_flush_batch(end - start, variable_bytes, next_bytes)? {
            ranges.push(start..end);
            start = end;
            variable_bytes = 0;
        }
        variable_bytes =
            checked_variable_bytes(variable_bytes, next_bytes, "manifest fragment batch")?;
    }
    if start < fragments.len() {
        ranges.push(start..fragments.len());
    }
    Ok(ranges)
}

fn data_file_variable_bytes(file: &DataFile) -> Result<usize> {
    let list_bytes = file
        .fields
        .len()
        .checked_add(file.column_indices.len())
        .and_then(|values| values.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::invalid_input("manifest DataFile list data overflows usize"))?;
    checked_variable_bytes(file.path.len(), list_bytes, "manifest DataFile")
}

fn sequence_fields() -> Fields {
    vec![
        Field::new("kind", DataType::UInt8, false),
        Field::new("inline", DataType::Binary, true),
        Field::new("external_path", DataType::Utf8, true),
        Field::new("external_offset", DataType::UInt64, true),
        Field::new("external_size", DataType::UInt64, true),
    ]
    .into()
}

fn deletion_fields() -> Fields {
    vec![
        Field::new("read_version", DataType::UInt64, false),
        Field::new("id", DataType::UInt64, false),
        Field::new("file_type", DataType::UInt8, false),
        Field::new("num_deleted_rows", DataType::UInt64, true),
        Field::new("base_id", DataType::UInt32, true),
    ]
    .into()
}

fn fragment_page_fields() -> Fields {
    vec![
        Field::new("first_fragment_id", DataType::UInt64, false),
        Field::new("last_fragment_id", DataType::UInt64, false),
        Field::new("fragment_row_start", DataType::UInt64, false),
        Field::new("fragment_row_count", DataType::UInt32, false),
    ]
    .into()
}

fn logical_page_fields() -> Fields {
    vec![
        Field::new("logical_row_start", DataType::UInt64, false),
        Field::new("logical_row_end", DataType::UInt64, false),
        Field::new("fragment_row_start", DataType::UInt64, false),
        Field::new("fragment_row_count", DataType::UInt32, false),
    ]
    .into()
}

fn header_fields() -> Fields {
    let fragment_page_item = Arc::new(Field::new(
        "item",
        DataType::Struct(fragment_page_fields()),
        false,
    ));
    let logical_page_item = Arc::new(Field::new(
        "item",
        DataType::Struct(logical_page_fields()),
        false,
    ));
    vec![
        Field::new("format_identity", DataType::Utf8, false),
        Field::new("manifest_schema_version", DataType::UInt32, false),
        Field::new("manifest_payload", DataType::Binary, false),
        Field::new("fragment_count", DataType::UInt64, false),
        Field::new("data_file_count", DataType::UInt64, false),
        Field::new("index_count", DataType::UInt64, false),
        Field::new("transaction_count", DataType::UInt64, false),
        Field::new(
            "fragment_page_index",
            DataType::List(fragment_page_item),
            false,
        ),
        Field::new(
            "logical_row_page_index",
            DataType::List(logical_page_item),
            false,
        ),
    ]
    .into()
}

fn fragment_fields() -> Fields {
    vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("logical_row_start", DataType::UInt64, false),
        Field::new("physical_rows", DataType::UInt64, false),
        Field::new("data_file_start", DataType::UInt64, false),
        Field::new("data_file_count", DataType::UInt32, false),
        Field::new("deletion", DataType::Struct(deletion_fields()), true),
        Field::new("row_id_meta", DataType::Struct(sequence_fields()), true),
        Field::new(
            "last_updated_version_meta",
            DataType::Struct(sequence_fields()),
            true,
        ),
        Field::new(
            "created_at_version_meta",
            DataType::Struct(sequence_fields()),
            true,
        ),
    ]
    .into()
}

fn data_file_fields() -> Fields {
    let item = Arc::new(Field::new("item", DataType::Int32, true));
    vec![
        Field::new("managed_file_id", DataType::FixedSizeBinary(16), true),
        Field::new("explicit_path", DataType::Utf8, true),
        Field::new("fields", DataType::List(item.clone()), false),
        Field::new("column_indices", DataType::List(item), false),
        Field::new("file_major_version", DataType::UInt32, false),
        Field::new("file_minor_version", DataType::UInt32, false),
        Field::new("known_size_bytes", DataType::UInt64, true),
        Field::new("base_id", DataType::UInt32, true),
    ]
    .into()
}

fn payload_fields() -> Fields {
    vec![Field::new("payload", DataType::Binary, false)].into()
}

fn arrow_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("header", DataType::Struct(header_fields()), false),
        Field::new("fragments", DataType::Struct(fragment_fields()), false),
        Field::new("data_files", DataType::Struct(data_file_fields()), false),
        Field::new("indices", DataType::Struct(payload_fields()), false),
        Field::new("transaction", DataType::Struct(payload_fields()), false),
    ]))
}

/// Returns whether the manifest has enough normalized row statistics for the
/// columnar format. Invalid ordering and arithmetic remain hard errors.
pub(super) fn can_write(manifest: &Manifest) -> Result<bool> {
    let mut previous_id = None;
    let mut logical_rows = 0_u64;
    let mut data_files = 0_u64;
    for fragment in manifest.fragments.iter() {
        if previous_id.is_some_and(|id| fragment.id <= id) {
            return Err(Error::invalid_input(format!(
                "fragment IDs must be strictly increasing, found {} after {:?}",
                fragment.id, previous_id
            )));
        }
        previous_id = Some(fragment.id);

        let Some(physical_rows) = fragment.physical_rows else {
            return Ok(false);
        };
        let deleted_rows = match fragment.deletion_file.as_ref() {
            Some(deletion) => {
                let Some(num_deleted_rows) = deletion.num_deleted_rows else {
                    return Ok(false);
                };
                num_deleted_rows
            }
            None => 0,
        };
        if deleted_rows > physical_rows {
            return Err(Error::invalid_input(format!(
                "fragment {} has {} deleted rows but only {} physical rows",
                fragment.id, deleted_rows, physical_rows
            )));
        }
        logical_rows = logical_rows
            .checked_add((physical_rows - deleted_rows) as u64)
            .ok_or_else(|| Error::invalid_input("manifest logical row count overflows u64"))?;
        u32::try_from(fragment.files.len()).map_err(|_| {
            Error::invalid_input(format!(
                "fragment {} has {} data files, exceeding u32::MAX",
                fragment.id,
                fragment.files.len()
            ))
        })?;
        data_files = data_files
            .checked_add(fragment.files.len() as u64)
            .ok_or_else(|| Error::invalid_input("manifest data file count overflows u64"))?;
    }
    let _ = (logical_rows, data_files);
    Ok(true)
}

pub(super) fn is_columnar_footer(tail: &Bytes) -> Result<bool> {
    if tail.len() < 8 {
        return Err(Error::invalid_input(format!(
            "manifest footer requires at least 8 bytes, found {}",
            tail.len()
        )));
    }
    if &tail[tail.len() - 4..] != lance_file::format::MAGIC {
        return Ok(false);
    }
    let major = LittleEndian::read_u16(&tail[tail.len() - 8..tail.len() - 6]);
    let minor = LittleEndian::read_u16(&tail[tail.len() - 6..tail.len() - 4]);
    match (major, minor) {
        (2, 1) => Ok(true),
        (0, _) => Ok(false),
        _ => Err(Error::not_supported(format!(
            "unsupported manifest container version {}.{}",
            major, minor
        ))),
    }
}

struct CapturingWriter {
    inner: Box<dyn Writer>,
    result: Arc<Mutex<Option<WriteResult>>>,
}

impl AsyncWrite for CapturingWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(self.inner.as_mut()).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(self.inner.as_mut()).poll_flush(cx)
    }

    fn poll_shutdown(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(self.inner.as_mut()).poll_shutdown(cx)
    }

    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        bufs: &[std::io::IoSlice<'_>],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(self.inner.as_mut()).poll_write_vectored(cx, bufs)
    }

    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }
}

#[async_trait::async_trait]
impl Writer for CapturingWriter {
    async fn tell(&mut self) -> Result<usize> {
        self.inner.tell().await
    }

    async fn shutdown(&mut self) -> Result<WriteResult> {
        let result = self.inner.shutdown().await?;
        let mut captured = self
            .result
            .lock()
            .map_err(|_| Error::internal("manifest writer result lock was poisoned"))?;
        *captured = Some(result.clone());
        Ok(result)
    }
}

struct TailCachedReader {
    inner: Arc<dyn Reader>,
    tail: Bytes,
    tail_start: usize,
    size: usize,
}

impl Debug for TailCachedReader {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TailCachedReader")
            .field("path", &self.inner.path())
            .field("tail_start", &self.tail_start)
            .field("size", &self.size)
            .finish()
    }
}

impl DeepSizeOf for TailCachedReader {
    fn deep_size_of_children(&self, context: &mut DeepSizeContext) -> usize {
        self.tail.deep_size_of_children(context)
    }
}

impl Reader for TailCachedReader {
    fn path(&self) -> &Path {
        self.inner.path()
    }

    fn block_size(&self) -> usize {
        self.tail.len()
    }

    fn io_parallelism(&self) -> usize {
        self.inner.io_parallelism()
    }

    fn size(&self) -> BoxFuture<'_, object_store::Result<usize>> {
        futures::future::ready(Ok(self.size)).boxed()
    }

    fn get_range(&self, range: Range<usize>) -> BoxFuture<'static, object_store::Result<Bytes>> {
        if range.start >= self.tail_start && range.end <= self.size {
            let start = range.start - self.tail_start;
            let end = range.end - self.tail_start;
            return futures::future::ready(Ok(self.tail.slice(start..end))).boxed();
        }
        if range.start < self.tail_start && range.end > self.tail_start && range.end <= self.size {
            let inner = self.inner.clone();
            let tail = self.tail.slice(..range.end - self.tail_start);
            let prefix = range.start..self.tail_start;
            return async move {
                let prefix = inner.get_range(prefix).await?;
                let mut combined = BytesMut::with_capacity(prefix.len() + tail.len());
                combined.extend_from_slice(&prefix);
                combined.extend_from_slice(&tail);
                Ok(combined.freeze())
            }
            .boxed();
        }
        self.inner.get_range(range)
    }

    fn get_all(&self) -> BoxFuture<'_, object_store::Result<Bytes>> {
        if self.tail_start == 0 {
            return futures::future::ready(Ok(self.tail.clone())).boxed();
        }
        self.inner.get_all()
    }

    fn get_stream(&self) -> BoxFuture<'_, object_store::Result<ByteStream>> {
        self.inner.get_stream()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FragmentPage {
    first_fragment_id: u64,
    last_fragment_id: u64,
    fragment_row_start: u64,
    fragment_row_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LogicalPage {
    logical_row_start: u64,
    logical_row_end: u64,
    fragment_row_start: u64,
    fragment_row_count: u32,
}

struct ManifestLayout {
    data_file_count: u64,
    fragment_pages: Vec<FragmentPage>,
    logical_pages: Vec<LogicalPage>,
}

fn manifest_layout(manifest: &Manifest) -> Result<ManifestLayout> {
    if !can_write(manifest)? {
        return Err(Error::invalid_input(
            "columnar manifests require physical_rows and deletion counts for every fragment",
        ));
    }

    let mut data_file_count = 0_u64;
    let mut logical_row_start = 0_u64;
    let fragment_ranges = fragment_batch_ranges(&manifest.fragments)?;
    let mut fragment_pages = Vec::with_capacity(fragment_ranges.len());
    let mut logical_pages = Vec::with_capacity(fragment_pages.capacity());
    for range in fragment_ranges {
        let fragments = &manifest.fragments[range.clone()];
        let Some(first) = fragments.first() else {
            continue;
        };
        let last = fragments.last().ok_or_else(|| {
            Error::internal("a non-empty manifest fragment page had no last fragment")
        })?;
        let fragment_row_start = u64::try_from(range.start)
            .map_err(|_| Error::invalid_input("fragment page row start exceeds u64::MAX"))?;
        let fragment_row_count = u32::try_from(fragments.len())
            .map_err(|_| Error::invalid_input("fragment page row count exceeds u32::MAX"))?;
        let page_logical_start = logical_row_start;
        for fragment in fragments {
            let rows = fragment.num_rows().ok_or_else(|| {
                Error::invalid_input(format!(
                    "fragment {} is missing normalized row statistics",
                    fragment.id
                ))
            })?;
            logical_row_start = logical_row_start
                .checked_add(rows as u64)
                .ok_or_else(|| Error::invalid_input("manifest logical row count overflows u64"))?;
            data_file_count = data_file_count
                .checked_add(fragment.files.len() as u64)
                .ok_or_else(|| Error::invalid_input("manifest data file count overflows u64"))?;
        }
        fragment_pages.push(FragmentPage {
            first_fragment_id: first.id,
            last_fragment_id: last.id,
            fragment_row_start,
            fragment_row_count,
        });
        logical_pages.push(LogicalPage {
            logical_row_start: page_logical_start,
            logical_row_end: logical_row_start,
            fragment_row_start,
            fragment_row_count,
        });
    }

    Ok(ManifestLayout {
        data_file_count,
        fragment_pages,
        logical_pages,
    })
}

fn fragment_page_list(pages: &[FragmentPage]) -> Result<ListArray> {
    let values = StructArray::try_new(
        fragment_page_fields(),
        vec![
            Arc::new(UInt64Array::from(
                pages
                    .iter()
                    .map(|page| page.first_fragment_id)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(UInt64Array::from(
                pages
                    .iter()
                    .map(|page| page.last_fragment_id)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(UInt64Array::from(
                pages
                    .iter()
                    .map(|page| page.fragment_row_start)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(UInt32Array::from(
                pages
                    .iter()
                    .map(|page| page.fragment_row_count)
                    .collect::<Vec<_>>(),
            )),
        ],
        None,
    )?;
    let end = i32::try_from(pages.len())
        .map_err(|_| Error::invalid_input("fragment page index exceeds i32::MAX"))?;
    Ok(ListArray::new(
        Arc::new(Field::new(
            "item",
            DataType::Struct(fragment_page_fields()),
            false,
        )),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, end])),
        Arc::new(values),
        None,
    ))
}

fn logical_page_list(pages: &[LogicalPage]) -> Result<ListArray> {
    let values = StructArray::try_new(
        logical_page_fields(),
        vec![
            Arc::new(UInt64Array::from(
                pages
                    .iter()
                    .map(|page| page.logical_row_start)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(UInt64Array::from(
                pages
                    .iter()
                    .map(|page| page.logical_row_end)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(UInt64Array::from(
                pages
                    .iter()
                    .map(|page| page.fragment_row_start)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(UInt32Array::from(
                pages
                    .iter()
                    .map(|page| page.fragment_row_count)
                    .collect::<Vec<_>>(),
            )),
        ],
        None,
    )?;
    let end = i32::try_from(pages.len())
        .map_err(|_| Error::invalid_input("logical page index exceeds i32::MAX"))?;
    Ok(ListArray::new(
        Arc::new(Field::new(
            "item",
            DataType::Struct(logical_page_fields()),
            false,
        )),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, end])),
        Arc::new(values),
        None,
    ))
}

fn header_array(
    manifest: &Manifest,
    layout: &ManifestLayout,
    index_count: usize,
    transaction_count: usize,
) -> Result<StructArray> {
    let payload = manifest.to_protobuf_header().encode_to_vec();
    checked_variable_bytes(0, payload.len(), "manifest header payload")?;
    StructArray::try_new(
        header_fields(),
        vec![
            Arc::new(StringArray::from(vec![FORMAT_IDENTITY])),
            Arc::new(UInt32Array::from(vec![MANIFEST_SCHEMA_VERSION])),
            Arc::new(BinaryArray::from(vec![payload.as_slice()])),
            Arc::new(UInt64Array::from(vec![manifest.fragments.len() as u64])),
            Arc::new(UInt64Array::from(vec![layout.data_file_count])),
            Arc::new(UInt64Array::from(vec![index_count as u64])),
            Arc::new(UInt64Array::from(vec![transaction_count as u64])),
            Arc::new(fragment_page_list(&layout.fragment_pages)?),
            Arc::new(logical_page_list(&layout.logical_pages)?),
        ],
        None,
    )
    .map_err(Into::into)
}

fn deletion_array(fragments: &[Fragment]) -> Result<StructArray> {
    let mut valid = Vec::with_capacity(fragments.len());
    let mut read_versions = Vec::with_capacity(fragments.len());
    let mut ids = Vec::with_capacity(fragments.len());
    let mut file_types = Vec::with_capacity(fragments.len());
    let mut deleted_rows = Vec::with_capacity(fragments.len());
    let mut base_ids = Vec::with_capacity(fragments.len());
    for deletion in fragments
        .iter()
        .map(|fragment| fragment.deletion_file.as_ref())
    {
        match deletion {
            Some(deletion) => {
                valid.push(true);
                read_versions.push(deletion.read_version);
                ids.push(deletion.id);
                file_types.push(match deletion.file_type {
                    DeletionFileType::Array => 0,
                    DeletionFileType::Bitmap => 1,
                });
                deleted_rows.push(deletion.num_deleted_rows.map(|rows| rows as u64));
                base_ids.push(deletion.base_id);
            }
            None => {
                valid.push(false);
                read_versions.push(0);
                ids.push(0);
                file_types.push(0);
                deleted_rows.push(None);
                base_ids.push(None);
            }
        }
    }
    StructArray::try_new(
        deletion_fields(),
        vec![
            Arc::new(UInt64Array::from(read_versions)),
            Arc::new(UInt64Array::from(ids)),
            Arc::new(UInt8Array::from(file_types)),
            Arc::new(UInt64Array::from(deleted_rows)),
            Arc::new(UInt32Array::from(base_ids)),
        ],
        Some(NullBuffer::from(valid)),
    )
    .map_err(Into::into)
}

enum SequenceRef<'a> {
    Inline(&'a [u8]),
    External(&'a ExternalFile),
}

fn sequence_array<'a>(
    values: impl Iterator<Item = Option<SequenceRef<'a>>>,
    len: usize,
) -> Result<StructArray> {
    let mut valid = Vec::with_capacity(len);
    let mut kinds = Vec::with_capacity(len);
    let mut inline = Vec::with_capacity(len);
    let mut paths = Vec::with_capacity(len);
    let mut offsets = Vec::with_capacity(len);
    let mut sizes = Vec::with_capacity(len);
    for value in values {
        match value {
            Some(SequenceRef::Inline(data)) => {
                valid.push(true);
                kinds.push(1);
                inline.push(Some(data));
                paths.push(None);
                offsets.push(None);
                sizes.push(None);
            }
            Some(SequenceRef::External(file)) => {
                valid.push(true);
                kinds.push(2);
                inline.push(None);
                paths.push(Some(file.path.as_str()));
                offsets.push(Some(file.offset));
                sizes.push(Some(file.size));
            }
            None => {
                valid.push(false);
                kinds.push(0);
                inline.push(None);
                paths.push(None);
                offsets.push(None);
                sizes.push(None);
            }
        }
    }
    StructArray::try_new(
        sequence_fields(),
        vec![
            Arc::new(UInt8Array::from(kinds)),
            Arc::new(BinaryArray::from(inline)),
            Arc::new(StringArray::from(paths)),
            Arc::new(UInt64Array::from(offsets)),
            Arc::new(UInt64Array::from(sizes)),
        ],
        Some(NullBuffer::from(valid)),
    )
    .map_err(Into::into)
}

fn fragment_array(
    fragments: &[Fragment],
    logical_row_start: &mut u64,
    data_file_start: &mut u64,
) -> Result<StructArray> {
    let mut ids = Vec::with_capacity(fragments.len());
    let mut logical_starts = Vec::with_capacity(fragments.len());
    let mut physical_rows = Vec::with_capacity(fragments.len());
    let mut file_starts = Vec::with_capacity(fragments.len());
    let mut file_counts = Vec::with_capacity(fragments.len());
    for fragment in fragments {
        ids.push(fragment.id);
        logical_starts.push(*logical_row_start);
        let physical = fragment.physical_rows.ok_or_else(|| {
            Error::invalid_input(format!("fragment {} is missing physical_rows", fragment.id))
        })?;
        physical_rows.push(physical as u64);
        file_starts.push(*data_file_start);
        let file_count = u32::try_from(fragment.files.len()).map_err(|_| {
            Error::invalid_input(format!("fragment {} has too many data files", fragment.id))
        })?;
        file_counts.push(file_count);
        *data_file_start = data_file_start
            .checked_add(u64::from(file_count))
            .ok_or_else(|| Error::invalid_input("manifest data file range overflows u64"))?;
        let logical_rows = fragment.num_rows().ok_or_else(|| {
            Error::invalid_input(format!(
                "fragment {} is missing normalized row statistics",
                fragment.id
            ))
        })?;
        *logical_row_start = logical_row_start
            .checked_add(logical_rows as u64)
            .ok_or_else(|| Error::invalid_input("manifest logical row range overflows u64"))?;
    }

    let row_ids = sequence_array(
        fragments.iter().map(|fragment| {
            fragment.row_id_meta.as_ref().map(|meta| match meta {
                RowIdMeta::Inline(data) => SequenceRef::Inline(data),
                RowIdMeta::External(file) => SequenceRef::External(file),
            })
        }),
        fragments.len(),
    )?;
    let last_updated = sequence_array(
        fragments.iter().map(|fragment| {
            fragment
                .last_updated_at_version_meta
                .as_ref()
                .map(|meta| match meta {
                    RowDatasetVersionMeta::Inline(data) => SequenceRef::Inline(data),
                    RowDatasetVersionMeta::External(file) => SequenceRef::External(file),
                })
        }),
        fragments.len(),
    )?;
    let created = sequence_array(
        fragments.iter().map(|fragment| {
            fragment
                .created_at_version_meta
                .as_ref()
                .map(|meta| match meta {
                    RowDatasetVersionMeta::Inline(data) => SequenceRef::Inline(data),
                    RowDatasetVersionMeta::External(file) => SequenceRef::External(file),
                })
        }),
        fragments.len(),
    )?;

    StructArray::try_new(
        fragment_fields(),
        vec![
            Arc::new(UInt64Array::from(ids)),
            Arc::new(UInt64Array::from(logical_starts)),
            Arc::new(UInt64Array::from(physical_rows)),
            Arc::new(UInt64Array::from(file_starts)),
            Arc::new(UInt32Array::from(file_counts)),
            Arc::new(deletion_array(fragments)?),
            Arc::new(row_ids),
            Arc::new(last_updated),
            Arc::new(created),
        ],
        None,
    )
    .map_err(Into::into)
}

fn data_file_array(files: &[&DataFile]) -> Result<StructArray> {
    let total_fields = files.iter().map(|file| file.fields.len()).sum();
    let total_indices = files.iter().map(|file| file.column_indices.len()).sum();
    let mut fields =
        ListBuilder::with_capacity(Int32Builder::with_capacity(total_fields), files.len());
    let mut column_indices =
        ListBuilder::with_capacity(Int32Builder::with_capacity(total_indices), files.len());
    for file in files {
        for value in file.fields.iter() {
            fields.values().append_value(*value);
        }
        fields.append(true);
        for value in file.column_indices.iter() {
            column_indices.values().append_value(*value);
        }
        column_indices.append(true);
    }
    StructArray::try_new(
        data_file_fields(),
        vec![
            Arc::new(FixedSizeBinaryArray::new_null(16, files.len())),
            Arc::new(StringArray::from_iter_values(
                files.iter().map(|file| file.path.as_str()),
            )),
            Arc::new(fields.finish()),
            Arc::new(column_indices.finish()),
            Arc::new(UInt32Array::from(
                files
                    .iter()
                    .map(|file| file.file_major_version)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(UInt32Array::from(
                files
                    .iter()
                    .map(|file| file.file_minor_version)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(UInt64Array::from(
                files
                    .iter()
                    .map(|file| file.file_size_bytes.get().map(|size| size.get()))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(UInt32Array::from(
                files.iter().map(|file| file.base_id).collect::<Vec<_>>(),
            )),
        ],
        None,
    )
    .map_err(Into::into)
}

fn payload_array(payloads: &[Vec<u8>]) -> Result<StructArray> {
    let total_bytes = payloads.iter().try_fold(0, |total, payload| {
        checked_variable_bytes(total, payload.len(), "manifest opaque payload")
    })?;
    let mut builder = BinaryBuilder::with_capacity(payloads.len(), total_bytes);
    for payload in payloads {
        builder.append_value(payload);
    }
    StructArray::try_new(payload_fields(), vec![Arc::new(builder.finish())], None)
        .map_err(Into::into)
}

/// Writes a columnar manifest to an already-created object writer.
pub(super) async fn write(
    writer: Box<dyn Writer>,
    manifest: &mut Manifest,
    indices: Option<Vec<IndexMetadata>>,
    transaction: Option<Transaction>,
) -> Result<WriteResult> {
    if manifest.data_storage_format.lance_file_version()?.resolve() != LanceFileVersion::V2_3 {
        return Err(Error::invalid_input(
            "the columnar manifest container requires storage version 2.3",
        ));
    }
    let layout = manifest_layout(manifest)?;
    let transaction_pb = transaction.map(pb::Transaction::from);
    manifest.clear_section_locations();

    let captured = Arc::new(Mutex::new(None));
    let capturing_writer = CapturingWriter {
        inner: writer,
        result: captured.clone(),
    };
    let lance_schema = Schema::try_from(arrow_schema().as_ref())?;
    let mut file_writer = FileWriter::try_new(
        Box::new(capturing_writer),
        lance_schema,
        FileWriterOptions {
            data_cache_bytes: Some(8 * 1024 * 1024),
            format_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        },
    )?;
    file_writer.add_schema_metadata(FORMAT_METADATA_KEY, FORMAT_IDENTITY);
    file_writer.add_schema_metadata(
        FORMAT_VERSION_METADATA_KEY,
        MANIFEST_SCHEMA_VERSION.to_string(),
    );
    let index_count = indices.as_ref().map_or(0, Vec::len);
    let transaction_count = usize::from(transaction_pb.is_some());
    file_writer
        .write_column(
            HEADER_COLUMN,
            Arc::new(header_array(
                manifest,
                &layout,
                index_count,
                transaction_count,
            )?),
        )
        .await?;

    let mut logical_row_start = 0_u64;
    let mut data_file_start = 0_u64;
    for range in fragment_batch_ranges(&manifest.fragments)? {
        let fragments = &manifest.fragments[range];
        file_writer
            .write_column(
                FRAGMENTS_COLUMN,
                Arc::new(fragment_array(
                    fragments,
                    &mut logical_row_start,
                    &mut data_file_start,
                )?),
            )
            .await?;
    }

    let mut files = Vec::with_capacity(SECTION_BATCH_SIZE);
    let mut file_batch_bytes = 0;
    for file in manifest
        .fragments
        .iter()
        .flat_map(|fragment| fragment.files.iter())
    {
        let next_bytes = data_file_variable_bytes(file)?;
        if should_flush_batch(files.len(), file_batch_bytes, next_bytes)? {
            file_writer
                .write_column(DATA_FILES_COLUMN, Arc::new(data_file_array(&files)?))
                .await?;
            files.clear();
            file_batch_bytes = 0;
        }
        files.push(file);
        file_batch_bytes =
            checked_variable_bytes(file_batch_bytes, next_bytes, "manifest DataFile batch")?;
    }
    if !files.is_empty() {
        file_writer
            .write_column(DATA_FILES_COLUMN, Arc::new(data_file_array(&files)?))
            .await?;
    }

    if let Some(indices) = indices.as_ref() {
        let mut payloads = Vec::with_capacity(indices.len().min(SECTION_BATCH_SIZE));
        let mut payload_batch_bytes = 0;
        for index in indices {
            let payload = pb::IndexMetadata::from(index).encode_to_vec();
            if should_flush_batch(payloads.len(), payload_batch_bytes, payload.len())? {
                file_writer
                    .write_column(INDICES_COLUMN, Arc::new(payload_array(&payloads)?))
                    .await?;
                payloads.clear();
                payload_batch_bytes = 0;
            }
            payload_batch_bytes = checked_variable_bytes(
                payload_batch_bytes,
                payload.len(),
                "manifest index payload batch",
            )?;
            payloads.push(payload);
        }
        if !payloads.is_empty() {
            file_writer
                .write_column(INDICES_COLUMN, Arc::new(payload_array(&payloads)?))
                .await?;
        }
    }
    if let Some(transaction) = transaction_pb {
        file_writer
            .write_column(
                TRANSACTION_COLUMN,
                Arc::new(payload_array(&[transaction.encode_to_vec()])?),
            )
            .await?;
    }

    let summary = file_writer.finish().await?;
    let result = captured
        .lock()
        .map_err(|_| Error::internal("manifest writer result lock was poisoned"))?
        .clone()
        .ok_or_else(|| Error::internal("manifest writer did not report a shutdown result"))?;
    if result.size as u64 != summary.size_bytes {
        return Err(Error::internal(format!(
            "manifest writer reported size {} but Lance summary reported {}",
            result.size, summary.size_bytes
        )));
    }
    manifest.set_columnar_section_presence(index_count != 0, transaction_count != 0);
    Ok(result)
}

fn corrupt(path: &Path, message: impl Into<String>) -> Error {
    Error::corrupt_file(path.clone(), message.into())
}

fn try_reserve_manifest_capacity<T>(
    values: &mut Vec<T>,
    additional: usize,
    path: &Path,
    context: &str,
) -> Result<()> {
    values.try_reserve_exact(additional).map_err(|error| {
        corrupt(
            path,
            format!("manifest {context} count {additional} cannot be allocated safely: {error}"),
        )
    })
}

fn try_reserve_manifest_rows<T>(
    values: &mut Vec<T>,
    row_count: u64,
    path: &Path,
    context: &str,
) -> Result<usize> {
    let row_count = usize::try_from(row_count)
        .map_err(|_| corrupt(path, format!("manifest {context} count exceeds usize")))?;
    try_reserve_manifest_capacity(values, row_count, path, context)?;
    Ok(row_count)
}

fn child_array<'a, T: Array + 'static>(
    parent: &'a StructArray,
    name: &str,
    path: &Path,
) -> Result<&'a T> {
    let child = parent
        .column_by_name(name)
        .ok_or_else(|| corrupt(path, format!("manifest section is missing child '{name}'")))?;
    child.as_any().downcast_ref::<T>().ok_or_else(|| {
        corrupt(
            path,
            format!(
                "manifest child '{name}' has type {}, expected {}",
                child.data_type(),
                std::any::type_name::<T>()
            ),
        )
    })
}

fn struct_section<'a>(
    batch: &'a arrow_array::RecordBatch,
    name: &str,
    path: &Path,
) -> Result<&'a StructArray> {
    let column = batch
        .column_by_name(name)
        .ok_or_else(|| corrupt(path, format!("manifest is missing section '{name}'")))?;
    column
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            corrupt(
                path,
                format!(
                    "manifest section '{name}' has type {}, expected struct",
                    column.data_type()
                ),
            )
        })
}

fn required_u64(array: &UInt64Array, row: usize, name: &str, path: &Path) -> Result<u64> {
    if array.is_null(row) {
        Err(corrupt(
            path,
            format!("manifest child '{name}' is null at row {row}"),
        ))
    } else {
        Ok(array.value(row))
    }
}

fn required_u32(array: &UInt32Array, row: usize, name: &str, path: &Path) -> Result<u32> {
    if array.is_null(row) {
        Err(corrupt(
            path,
            format!("manifest child '{name}' is null at row {row}"),
        ))
    } else {
        Ok(array.value(row))
    }
}

fn required_u8(array: &UInt8Array, row: usize, name: &str, path: &Path) -> Result<u8> {
    if array.is_null(row) {
        Err(corrupt(
            path,
            format!("manifest child '{name}' is null at row {row}"),
        ))
    } else {
        Ok(array.value(row))
    }
}

fn decode_fragment_pages(array: &ListArray, row: usize, path: &Path) -> Result<Vec<FragmentPage>> {
    if array.is_null(row) {
        return Err(corrupt(path, "fragment page index is null"));
    }
    let values = array.value(row);
    let values = values
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| corrupt(path, "fragment page index values are not structs"))?;
    let first_ids = child_array::<UInt64Array>(values, "first_fragment_id", path)?;
    let last_ids = child_array::<UInt64Array>(values, "last_fragment_id", path)?;
    let starts = child_array::<UInt64Array>(values, "fragment_row_start", path)?;
    let counts = child_array::<UInt32Array>(values, "fragment_row_count", path)?;
    (0..values.len())
        .map(|row| {
            Ok(FragmentPage {
                first_fragment_id: required_u64(first_ids, row, "first_fragment_id", path)?,
                last_fragment_id: required_u64(last_ids, row, "last_fragment_id", path)?,
                fragment_row_start: required_u64(starts, row, "fragment_row_start", path)?,
                fragment_row_count: required_u32(counts, row, "fragment_row_count", path)?,
            })
        })
        .collect()
}

fn decode_logical_pages(array: &ListArray, row: usize, path: &Path) -> Result<Vec<LogicalPage>> {
    if array.is_null(row) {
        return Err(corrupt(path, "logical row page index is null"));
    }
    let values = array.value(row);
    let values = values
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| corrupt(path, "logical row page index values are not structs"))?;
    let logical_starts = child_array::<UInt64Array>(values, "logical_row_start", path)?;
    let logical_ends = child_array::<UInt64Array>(values, "logical_row_end", path)?;
    let fragment_starts = child_array::<UInt64Array>(values, "fragment_row_start", path)?;
    let fragment_counts = child_array::<UInt32Array>(values, "fragment_row_count", path)?;
    (0..values.len())
        .map(|row| {
            Ok(LogicalPage {
                logical_row_start: required_u64(logical_starts, row, "logical_row_start", path)?,
                logical_row_end: required_u64(logical_ends, row, "logical_row_end", path)?,
                fragment_row_start: required_u64(fragment_starts, row, "fragment_row_start", path)?,
                fragment_row_count: required_u32(fragment_counts, row, "fragment_row_count", path)?,
            })
        })
        .collect()
}

struct DecodedHeader {
    manifest: Manifest,
    fragment_count: u64,
    data_file_count: u64,
    index_count: u64,
    transaction_count: u64,
    fragment_pages: Vec<FragmentPage>,
    logical_pages: Vec<LogicalPage>,
}

fn decode_header(array: &StructArray, row: usize, path: &Path) -> Result<DecodedHeader> {
    if array.is_null(row) {
        return Err(corrupt(path, "manifest header row is null"));
    }
    let identities = child_array::<StringArray>(array, "format_identity", path)?;
    let versions = child_array::<UInt32Array>(array, "manifest_schema_version", path)?;
    let payloads = child_array::<BinaryArray>(array, "manifest_payload", path)?;
    let fragment_counts = child_array::<UInt64Array>(array, "fragment_count", path)?;
    let data_file_counts = child_array::<UInt64Array>(array, "data_file_count", path)?;
    let index_counts = child_array::<UInt64Array>(array, "index_count", path)?;
    let transaction_counts = child_array::<UInt64Array>(array, "transaction_count", path)?;
    let fragment_pages = child_array::<ListArray>(array, "fragment_page_index", path)?;
    let logical_pages = child_array::<ListArray>(array, "logical_row_page_index", path)?;

    if identities.is_null(row) || identities.value(row) != FORMAT_IDENTITY {
        return Err(corrupt(
            path,
            "manifest header format identity does not match",
        ));
    }
    let version = required_u32(versions, row, "manifest_schema_version", path)?;
    if version != MANIFEST_SCHEMA_VERSION {
        return Err(Error::not_supported(format!(
            "unsupported manifest schema version {version}"
        )));
    }
    if payloads.is_null(row) {
        return Err(corrupt(path, "manifest header payload is null"));
    }
    let payload = pb::Manifest::decode(payloads.value(row)).map_err(|error| {
        corrupt(
            path,
            format!("manifest header payload is invalid protobuf: {error}"),
        )
    })?;
    if !payload.fragments.is_empty() {
        return Err(corrupt(
            path,
            "manifest header payload unexpectedly contains fragments",
        ));
    }
    if payload.index_section.is_some() || payload.transaction_section.is_some() {
        return Err(corrupt(
            path,
            "columnar manifest header contains a protobuf auxiliary section offset",
        ));
    }
    let mut manifest = Manifest::try_from(payload)?;
    if manifest.data_storage_format.lance_file_version()?.resolve() != LanceFileVersion::V2_3 {
        return Err(corrupt(
            path,
            "columnar manifest header does not declare storage version 2.3",
        ));
    }
    let index_count = required_u64(index_counts, row, "index_count", path)?;
    let transaction_count = required_u64(transaction_counts, row, "transaction_count", path)?;
    if transaction_count > 1 {
        return Err(corrupt(
            path,
            format!("manifest contains {transaction_count} transaction rows"),
        ));
    }
    manifest.set_columnar_section_presence(index_count != 0, transaction_count != 0);
    Ok(DecodedHeader {
        manifest,
        fragment_count: required_u64(fragment_counts, row, "fragment_count", path)?,
        data_file_count: required_u64(data_file_counts, row, "data_file_count", path)?,
        index_count,
        transaction_count,
        fragment_pages: decode_fragment_pages(fragment_pages, row, path)?,
        logical_pages: decode_logical_pages(logical_pages, row, path)?,
    })
}

enum DecodedSequence {
    Inline(Vec<u8>),
    External(ExternalFile),
}

fn decode_sequence(
    array: &StructArray,
    row: usize,
    name: &str,
    path: &Path,
) -> Result<Option<DecodedSequence>> {
    if array.is_null(row) {
        return Ok(None);
    }
    let kinds = child_array::<UInt8Array>(array, "kind", path)?;
    let inline = child_array::<BinaryArray>(array, "inline", path)?;
    let external_paths = child_array::<StringArray>(array, "external_path", path)?;
    let external_offsets = child_array::<UInt64Array>(array, "external_offset", path)?;
    let external_sizes = child_array::<UInt64Array>(array, "external_size", path)?;
    match required_u8(kinds, row, "kind", path)? {
        1 if !inline.is_null(row)
            && external_paths.is_null(row)
            && external_offsets.is_null(row)
            && external_sizes.is_null(row) =>
        {
            Ok(Some(DecodedSequence::Inline(inline.value(row).to_vec())))
        }
        2 if inline.is_null(row)
            && !external_paths.is_null(row)
            && !external_offsets.is_null(row)
            && !external_sizes.is_null(row) =>
        {
            Ok(Some(DecodedSequence::External(ExternalFile {
                path: external_paths.value(row).to_string(),
                offset: external_offsets.value(row),
                size: external_sizes.value(row),
            })))
        }
        kind => Err(corrupt(
            path,
            format!("manifest {name} has invalid kind/fields combination {kind} at row {row}"),
        )),
    }
}

fn decode_deletion(array: &StructArray, row: usize, path: &Path) -> Result<Option<DeletionFile>> {
    if array.is_null(row) {
        return Ok(None);
    }
    let read_versions = child_array::<UInt64Array>(array, "read_version", path)?;
    let ids = child_array::<UInt64Array>(array, "id", path)?;
    let file_types = child_array::<UInt8Array>(array, "file_type", path)?;
    let deleted_rows = child_array::<UInt64Array>(array, "num_deleted_rows", path)?;
    let base_ids = child_array::<UInt32Array>(array, "base_id", path)?;
    let file_type = match required_u8(file_types, row, "file_type", path)? {
        0 => DeletionFileType::Array,
        1 => DeletionFileType::Bitmap,
        value => {
            return Err(corrupt(
                path,
                format!("manifest deletion file has unknown type {value} at row {row}"),
            ));
        }
    };
    let num_deleted_rows = if deleted_rows.is_null(row) {
        None
    } else {
        Some(usize::try_from(deleted_rows.value(row)).map_err(|_| {
            corrupt(
                path,
                format!("manifest deletion count does not fit usize at row {row}"),
            )
        })?)
    };
    Ok(Some(DeletionFile {
        read_version: required_u64(read_versions, row, "read_version", path)?,
        id: required_u64(ids, row, "id", path)?,
        file_type,
        num_deleted_rows,
        base_id: (!base_ids.is_null(row)).then(|| base_ids.value(row)),
    }))
}

struct DecodedFragment {
    fragment: Fragment,
    logical_row_start: u64,
    data_file_start: u64,
    data_file_count: u32,
}

fn decode_fragment_batch(
    array: &StructArray,
    path: &Path,
    interner: &mut DataFileFieldInterner,
) -> Result<Vec<DecodedFragment>> {
    let ids = child_array::<UInt64Array>(array, "id", path)?;
    let logical_starts = child_array::<UInt64Array>(array, "logical_row_start", path)?;
    let physical_rows = child_array::<UInt64Array>(array, "physical_rows", path)?;
    let data_file_starts = child_array::<UInt64Array>(array, "data_file_start", path)?;
    let data_file_counts = child_array::<UInt32Array>(array, "data_file_count", path)?;
    let deletions = child_array::<StructArray>(array, "deletion", path)?;
    let row_ids = child_array::<StructArray>(array, "row_id_meta", path)?;
    let last_updated = child_array::<StructArray>(array, "last_updated_version_meta", path)?;
    let created = child_array::<StructArray>(array, "created_at_version_meta", path)?;

    (0..array.len())
        .map(|row| {
            if array.is_null(row) {
                return Err(corrupt(
                    path,
                    format!("manifest fragment row {row} is null"),
                ));
            }
            let physical_rows_u64 = required_u64(physical_rows, row, "physical_rows", path)?;
            let physical_rows = usize::try_from(physical_rows_u64).map_err(|_| {
                corrupt(
                    path,
                    format!("manifest physical row count does not fit usize at row {row}"),
                )
            })?;
            let deletion_file = decode_deletion(deletions, row, path)?;
            let row_id_sequence =
                decode_sequence(row_ids, row, "row ID metadata", path)?.map(|sequence| {
                    match sequence {
                        DecodedSequence::Inline(data) => {
                            pb::data_fragment::RowIdSequence::InlineRowIds(data)
                        }
                        DecodedSequence::External(file) => {
                            pb::data_fragment::RowIdSequence::ExternalRowIds(pb::ExternalFile {
                                path: file.path,
                                offset: file.offset,
                                size: file.size,
                            })
                        }
                    }
                });
            let last_updated_at_version_sequence =
                decode_sequence(last_updated, row, "last-updated version metadata", path)?.map(
                    |sequence| {
                        match sequence {
                DecodedSequence::Inline(data) => {
                    pb::data_fragment::LastUpdatedAtVersionSequence::InlineLastUpdatedAtVersions(
                        data,
                    )
                }
                DecodedSequence::External(file) => {
                    pb::data_fragment::LastUpdatedAtVersionSequence::ExternalLastUpdatedAtVersions(
                        pb::ExternalFile {
                            path: file.path,
                            offset: file.offset,
                            size: file.size,
                        },
                    )
                }
            }
                    },
                );
            let created_at_version_sequence =
                decode_sequence(created, row, "created-at version metadata", path)?.map(
                    |sequence| match sequence {
                        DecodedSequence::Inline(data) => {
                            pb::data_fragment::CreatedAtVersionSequence::InlineCreatedAtVersions(
                                data,
                            )
                        }
                        DecodedSequence::External(file) => {
                            pb::data_fragment::CreatedAtVersionSequence::ExternalCreatedAtVersions(
                                pb::ExternalFile {
                                    path: file.path,
                                    offset: file.offset,
                                    size: file.size,
                                },
                            )
                        }
                    },
                );
            let mut fragment = interner.intern_fragment(pb::DataFragment {
                id: required_u64(ids, row, "id", path)?,
                files: Vec::new(),
                deletion_file: None,
                physical_rows: physical_rows_u64,
                row_id_sequence,
                last_updated_at_version_sequence,
                created_at_version_sequence,
            })?;
            fragment.physical_rows = Some(physical_rows);
            fragment.deletion_file = deletion_file;
            Ok(DecodedFragment {
                fragment,
                logical_row_start: required_u64(logical_starts, row, "logical_row_start", path)?,
                data_file_start: required_u64(data_file_starts, row, "data_file_start", path)?,
                data_file_count: required_u32(data_file_counts, row, "data_file_count", path)?,
            })
        })
        .collect()
}

fn managed_file_path(id: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut path = String::with_capacity(56);
    for byte in &id[..3] {
        for bit in (0..8).rev() {
            path.push(if (byte >> bit) & 1 == 1 { '1' } else { '0' });
        }
    }
    for byte in &id[3..] {
        path.push(HEX[(byte >> 4) as usize] as char);
        path.push(HEX[(byte & 0x0f) as usize] as char);
    }
    path.push_str(".lance");
    path
}

/// Resolves DataFile child arrays once per batch instead of once per row.
struct DataFileBatchDecoder<'a> {
    array: &'a StructArray,
    managed_ids: &'a FixedSizeBinaryArray,
    explicit_paths: &'a StringArray,
    fields: &'a ListArray,
    field_values: &'a Int32Array,
    column_indices: &'a ListArray,
    column_index_values: &'a Int32Array,
    major_versions: &'a UInt32Array,
    minor_versions: &'a UInt32Array,
    known_sizes: &'a UInt64Array,
    base_ids: &'a UInt32Array,
    path: &'a Path,
}

impl<'a> DataFileBatchDecoder<'a> {
    fn new(array: &'a StructArray, path: &'a Path) -> Result<Self> {
        let fields = child_array::<ListArray>(array, "fields", path)?;
        let column_indices = child_array::<ListArray>(array, "column_indices", path)?;
        let field_values = fields
            .values()
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| corrupt(path, "manifest DataFile fields values are not int32"))?;
        let column_index_values = column_indices
            .values()
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| {
                corrupt(
                    path,
                    "manifest DataFile column_indices values are not int32",
                )
            })?;
        Ok(Self {
            array,
            managed_ids: child_array::<FixedSizeBinaryArray>(array, "managed_file_id", path)?,
            explicit_paths: child_array::<StringArray>(array, "explicit_path", path)?,
            fields,
            field_values,
            column_indices,
            column_index_values,
            major_versions: child_array::<UInt32Array>(array, "file_major_version", path)?,
            minor_versions: child_array::<UInt32Array>(array, "file_minor_version", path)?,
            known_sizes: child_array::<UInt64Array>(array, "known_size_bytes", path)?,
            base_ids: child_array::<UInt32Array>(array, "base_id", path)?,
            path,
        })
    }

    fn list_values<'b>(
        &self,
        array: &'b ListArray,
        values: &'b Int32Array,
        row: usize,
        name: &str,
    ) -> Result<&'b [i32]> {
        if array.is_null(row) {
            return Err(corrupt(
                self.path,
                format!("manifest DataFile {name} is null at row {row}"),
            ));
        }
        let offsets = array.value_offsets();
        let start = usize::try_from(offsets[row]).map_err(|_| {
            corrupt(
                self.path,
                format!("manifest DataFile {name} has a negative start offset at row {row}"),
            )
        })?;
        let end = usize::try_from(offsets[row + 1]).map_err(|_| {
            corrupt(
                self.path,
                format!("manifest DataFile {name} has a negative end offset at row {row}"),
            )
        })?;
        if end < start || end > values.len() {
            return Err(corrupt(
                self.path,
                format!(
                    "manifest DataFile {name} range {start}..{end} is invalid for {} values at row {row}",
                    values.len()
                ),
            ));
        }
        if values.null_count() != 0
            && let Some(index) = (start..end).find(|index| values.is_null(*index))
        {
            return Err(corrupt(
                self.path,
                format!(
                    "manifest DataFile {name}[{}] is null at row {row}",
                    index - start
                ),
            ));
        }
        Ok(&values.values()[start..end])
    }

    fn decode(&self, row: usize, interner: &mut DataFileFieldInterner) -> Result<DataFile> {
        if self.array.is_null(row) {
            return Err(corrupt(
                self.path,
                format!("manifest DataFile row {row} is null"),
            ));
        }
        let file_path = match (
            self.managed_ids.is_null(row),
            self.explicit_paths.is_null(row),
        ) {
            (false, true) => managed_file_path(self.managed_ids.value(row)),
            (true, false) => self.explicit_paths.value(row).to_string(),
            _ => {
                return Err(corrupt(
                    self.path,
                    format!(
                        "manifest DataFile row {row} must set exactly one of managed_file_id and explicit_path"
                    ),
                ));
            }
        };
        let known_size = if self.known_sizes.is_null(row) {
            0
        } else {
            let size = self.known_sizes.value(row);
            if size == 0 {
                return Err(corrupt(
                    self.path,
                    format!("manifest DataFile row {row} has a zero known size"),
                ));
            }
            size
        };
        let (fields, column_indices) = interner.intern_data_file_fields(
            self.list_values(self.fields, self.field_values, row, "fields")?,
            self.list_values(
                self.column_indices,
                self.column_index_values,
                row,
                "column_indices",
            )?,
        );
        Ok(DataFile {
            path: file_path,
            fields,
            column_indices,
            file_major_version: required_u32(
                self.major_versions,
                row,
                "file_major_version",
                self.path,
            )?,
            file_minor_version: required_u32(
                self.minor_versions,
                row,
                "file_minor_version",
                self.path,
            )?,
            file_size_bytes: CachedFileSize::new(known_size),
            base_id: (!self.base_ids.is_null(row)).then(|| self.base_ids.value(row)),
        })
    }
}

fn section_projection(
    reader: &FileReader,
    section: &str,
    path: &Path,
) -> Result<(ReaderProjection, u64)> {
    let projection = ReaderProjection::from_column_names(
        reader.metadata().version(),
        reader.schema(),
        &[section],
    )?;
    let mut lengths = projection
        .column_indices
        .iter()
        .map(|column| reader.column_num_rows(*column as usize));
    let length = lengths.next().transpose()?.ok_or_else(|| {
        corrupt(
            path,
            format!("manifest section '{section}' has no leaf columns"),
        )
    })?;
    for other in lengths {
        let other = other?;
        if other != length {
            return Err(corrupt(
                path,
                format!(
                    "manifest section '{section}' has leaf columns with differing lengths {length} and {other}"
                ),
            ));
        }
    }
    Ok((projection, length))
}

fn validate_file_reader(file_reader: &FileReader, path: &Path) -> Result<()> {
    if file_reader.metadata().version() != LanceFileVersion::V2_1 {
        return Err(Error::not_supported(format!(
            "unsupported manifest container version {:?}",
            file_reader.metadata().version()
        )));
    }

    let schema_metadata = &file_reader.schema().metadata;
    if schema_metadata.get(FORMAT_METADATA_KEY).map(String::as_str) != Some(FORMAT_IDENTITY) {
        return Err(corrupt(
            path,
            "manifest schema format identity does not match",
        ));
    }
    let semantic_version = schema_metadata
        .get(FORMAT_VERSION_METADATA_KEY)
        .ok_or_else(|| corrupt(path, "manifest schema version metadata is missing"))?
        .parse::<u32>()
        .map_err(|error| corrupt(path, format!("manifest schema version is invalid: {error}")))?;
    if semantic_version != MANIFEST_SCHEMA_VERSION {
        return Err(Error::not_supported(format!(
            "unsupported manifest schema version {semantic_version}"
        )));
    }
    let file_arrow_schema = ArrowSchema::from(file_reader.schema().as_ref());
    if file_arrow_schema.fields() != arrow_schema().fields() {
        return Err(corrupt(
            path,
            "manifest schema does not match semantic version 1",
        ));
    }
    Ok(())
}

async fn open_file_reader(
    object_store: &ObjectStore,
    path: &Path,
    size: Option<u64>,
) -> Result<FileReader> {
    let reader: Arc<dyn Reader> = if let Some(size) = size {
        object_store
            .open_with_size(path, size as usize)
            .await?
            .into()
    } else {
        object_store.open(path).await?.into()
    };
    let scheduler = ScanScheduler::new(
        Arc::new(object_store.clone()),
        SchedulerConfig::max_bandwidth(object_store),
    );
    let file_reader = FileReader::try_open(
        scheduler.open_reader(reader),
        None,
        Arc::<DecoderPlugins>::default(),
        &LanceCache::no_cache(),
        FileReaderOptions {
            batch_size_bytes: Some(SECTION_VARIABLE_BYTES as u64),
            ..Default::default()
        },
    )
    .await?;
    validate_file_reader(&file_reader, path)?;
    Ok(file_reader)
}

async fn decode_payload_section<T>(
    file_reader: &FileReader,
    section_name: &str,
    path: &Path,
    mut decode: impl FnMut(&[u8]) -> Result<T>,
) -> Result<Vec<T>> {
    let (projection, row_count) = section_projection(file_reader, section_name, path)?;
    let mut decoded = Vec::new();
    let row_count = try_reserve_manifest_rows(&mut decoded, row_count, path, section_name)?;
    let mut stream = file_reader
        .read_stream_projected(
            ReadBatchParams::RangeFull,
            SECTION_BATCH_SIZE as u32,
            1,
            projection,
            FilterExpression::no_filter(),
        )
        .await?;
    while let Some(batch) = stream.try_next().await? {
        let section = struct_section(&batch, section_name, path)?;
        let values = child_array::<BinaryArray>(section, "payload", path)?;
        for row in 0..section.len() {
            if section.is_null(row) || values.is_null(row) {
                return Err(corrupt(
                    path,
                    format!("manifest {section_name} payload is null at row {row}"),
                ));
            }
            decoded.push(decode(values.value(row))?);
        }
    }
    if decoded.len() != row_count {
        return Err(corrupt(
            path,
            format!(
                "manifest decoded {} {section_name} rows, expected {row_count}",
                decoded.len()
            ),
        ));
    }
    Ok(decoded)
}

pub(super) async fn read_indexes(
    object_store: &ObjectStore,
    path: &Path,
    size: Option<u64>,
) -> Result<Vec<IndexMetadata>> {
    let file_reader = open_file_reader(object_store, path, size).await?;
    decode_payload_section(&file_reader, "indices", path, |payload| {
        let index = pb::IndexMetadata::decode(payload).map_err(|error| {
            corrupt(
                path,
                format!("manifest index payload is invalid protobuf: {error}"),
            )
        })?;
        IndexMetadata::try_from(index)
    })
    .await
}

pub(super) async fn read_transaction(
    object_store: &ObjectStore,
    path: &Path,
    size: Option<u64>,
) -> Result<Option<Transaction>> {
    let file_reader = open_file_reader(object_store, path, size).await?;
    let transactions = decode_payload_section(&file_reader, "transaction", path, |payload| {
        let transaction = pb::Transaction::decode(payload).map_err(|error| {
            corrupt(
                path,
                format!("manifest transaction payload is invalid protobuf: {error}"),
            )
        })?;
        Ok(Transaction::from(transaction))
    })
    .await?;
    if transactions.len() > 1 {
        return Err(corrupt(
            path,
            format!(
                "manifest contains {} transaction rows, expected at most 1",
                transactions.len()
            ),
        ));
    }
    Ok(transactions.into_iter().next())
}

pub(super) async fn read_with_prefetched_tail(
    object_store: &ObjectStore,
    reader: Arc<dyn Reader>,
    tail: Bytes,
) -> Result<Manifest> {
    if !is_columnar_footer(&tail)? {
        return Err(Error::invalid_input(
            "attempted to open a protobuf manifest as a columnar manifest",
        ));
    }
    let path = reader.path().clone();
    let size = reader.size().await?;
    if tail.len() > size {
        return Err(corrupt(
            &path,
            format!(
                "prefetched manifest tail has {} bytes but file size is {size}",
                tail.len()
            ),
        ));
    }
    let tail_start = size - tail.len();
    // Phase 1 materializes every fragment. Caching a small object after the
    // initial tail read preserves the protobuf tail-plus-remainder request shape
    // and prevents each projected section from issuing its own range request.
    let (tail, tail_start) = if size <= SMALL_MANIFEST_PREFETCH_BYTES && tail_start != 0 {
        let leading_bytes = reader.get_range(0..tail_start).await?;
        if leading_bytes.len() != tail_start {
            return Err(corrupt(
                &path,
                format!(
                    "manifest leading-byte read returned {} bytes, expected {tail_start}",
                    leading_bytes.len()
                ),
            ));
        }
        let mut complete = BytesMut::with_capacity(size);
        complete.extend_from_slice(&leading_bytes);
        complete.extend_from_slice(&tail);
        (complete.freeze(), 0)
    } else {
        (tail, tail_start)
    };
    let cached_reader: Arc<dyn Reader> = Arc::new(TailCachedReader {
        inner: reader,
        tail_start,
        tail,
        size,
    });
    let scheduler = ScanScheduler::new(
        Arc::new(object_store.clone()),
        SchedulerConfig::max_bandwidth(object_store),
    );
    let file_reader = FileReader::try_open(
        scheduler.open_reader(cached_reader),
        None,
        Arc::<DecoderPlugins>::default(),
        &LanceCache::no_cache(),
        FileReaderOptions {
            batch_size_bytes: Some(SECTION_VARIABLE_BYTES as u64),
            ..Default::default()
        },
    )
    .await?;
    validate_file_reader(&file_reader, &path)?;

    let (header_projection, header_rows) = section_projection(&file_reader, "header", &path)?;
    let (fragment_projection, fragment_rows) =
        section_projection(&file_reader, "fragments", &path)?;
    let (data_file_projection, data_file_rows) =
        section_projection(&file_reader, "data_files", &path)?;
    let (_, index_rows) = section_projection(&file_reader, "indices", &path)?;
    let (_, transaction_rows) = section_projection(&file_reader, "transaction", &path)?;
    if header_rows != 1 {
        return Err(corrupt(
            &path,
            format!("manifest has {header_rows} header rows, expected 1"),
        ));
    }

    let mut header_stream = file_reader
        .read_stream_projected(
            ReadBatchParams::RangeFull,
            SECTION_BATCH_SIZE as u32,
            1,
            header_projection,
            FilterExpression::no_filter(),
        )
        .await?;
    let mut decoded_header = None;
    while let Some(batch) = header_stream.try_next().await? {
        let section = struct_section(&batch, "header", &path)?;
        for row in 0..section.len() {
            if decoded_header.is_some() {
                return Err(corrupt(&path, "manifest contains multiple header rows"));
            }
            decoded_header = Some(decode_header(section, row, &path)?);
        }
    }
    let DecodedHeader {
        mut manifest,
        fragment_count,
        data_file_count,
        index_count,
        transaction_count,
        fragment_pages,
        logical_pages,
    } = decoded_header.ok_or_else(|| corrupt(&path, "manifest header is empty"))?;
    for (name, recorded, actual) in [
        ("fragments", fragment_count, fragment_rows),
        ("data_files", data_file_count, data_file_rows),
        ("indices", index_count, index_rows),
        ("transaction", transaction_count, transaction_rows),
    ] {
        if recorded != actual {
            return Err(corrupt(
                &path,
                format!("manifest header records {recorded} {name} rows but section has {actual}"),
            ));
        }
    }
    let mut fragments = Vec::new();
    let fragment_capacity =
        try_reserve_manifest_rows(&mut fragments, fragment_count, &path, "fragment")?;
    let mut data_file_counts = Vec::new();
    try_reserve_manifest_capacity(
        &mut data_file_counts,
        fragment_capacity,
        &path,
        "fragment DataFile-count",
    )?;
    let mut interner = DataFileFieldInterner::default();
    let mut previous_id = None;
    let mut expected_logical_start = 0_u64;
    let mut expected_data_file_start = 0_u64;
    let mut fragment_stream = file_reader
        .read_stream_projected(
            ReadBatchParams::RangeFull,
            SECTION_BATCH_SIZE as u32,
            4,
            fragment_projection,
            FilterExpression::no_filter(),
        )
        .await?;
    while let Some(batch) = fragment_stream.try_next().await? {
        let section = struct_section(&batch, "fragments", &path)?;
        for decoded in decode_fragment_batch(section, &path, &mut interner)? {
            if fragments.len() == fragment_capacity {
                return Err(corrupt(
                    &path,
                    format!("manifest decoded more than the recorded {fragment_count} fragments"),
                ));
            }
            if previous_id.is_some_and(|id| decoded.fragment.id <= id) {
                return Err(corrupt(
                    &path,
                    format!(
                        "manifest fragment IDs are not strictly increasing at {}",
                        decoded.fragment.id
                    ),
                ));
            }
            previous_id = Some(decoded.fragment.id);
            if decoded.logical_row_start != expected_logical_start {
                return Err(corrupt(
                    &path,
                    format!(
                        "manifest fragment {} starts at logical row {}, expected {}",
                        decoded.fragment.id, decoded.logical_row_start, expected_logical_start
                    ),
                ));
            }
            let physical_rows = decoded.fragment.physical_rows.ok_or_else(|| {
                corrupt(
                    &path,
                    format!(
                        "manifest fragment {} has no physical row count",
                        decoded.fragment.id
                    ),
                )
            })?;
            let deleted_rows = decoded
                .fragment
                .deletion_file
                .as_ref()
                .and_then(|deletion| deletion.num_deleted_rows)
                .unwrap_or(0);
            if deleted_rows > physical_rows {
                return Err(corrupt(
                    &path,
                    format!(
                        "manifest fragment {} has {deleted_rows} deleted rows but only {physical_rows} physical rows",
                        decoded.fragment.id
                    ),
                ));
            }
            expected_logical_start = expected_logical_start
                .checked_add((physical_rows - deleted_rows) as u64)
                .ok_or_else(|| corrupt(&path, "manifest logical row count overflows u64"))?;
            if decoded.data_file_start != expected_data_file_start {
                return Err(corrupt(
                    &path,
                    format!(
                        "manifest fragment {} starts at DataFile row {}, expected {}",
                        decoded.fragment.id, decoded.data_file_start, expected_data_file_start
                    ),
                ));
            }
            expected_data_file_start = expected_data_file_start
                .checked_add(u64::from(decoded.data_file_count))
                .ok_or_else(|| corrupt(&path, "manifest DataFile count overflows u64"))?;

            let mut fragment = decoded.fragment;
            try_reserve_manifest_capacity(
                &mut fragment.files,
                decoded.data_file_count as usize,
                &path,
                "fragment DataFile",
            )?;
            data_file_counts.push(decoded.data_file_count);
            fragments.push(fragment);
        }
    }
    if fragments.len() != fragment_capacity {
        return Err(corrupt(
            &path,
            format!(
                "manifest decoded {} fragments, expected {fragment_count}",
                fragments.len()
            ),
        ));
    }
    if expected_data_file_start != data_file_count {
        return Err(corrupt(
            &path,
            format!(
                "manifest fragment DataFile ranges cover {expected_data_file_start} rows, expected {data_file_count}"
            ),
        ));
    }

    let mut next_fragment = 0_usize;
    let mut decoded_data_files = 0_u64;
    let mut data_file_stream = file_reader
        .read_stream_projected(
            ReadBatchParams::RangeFull,
            SECTION_BATCH_SIZE as u32,
            4,
            data_file_projection,
            FilterExpression::no_filter(),
        )
        .await?;
    while let Some(batch) = data_file_stream.try_next().await? {
        let section = struct_section(&batch, "data_files", &path)?;
        let decoder = DataFileBatchDecoder::new(section, &path)?;
        for row in 0..section.len() {
            while next_fragment < fragments.len() && data_file_counts[next_fragment] == 0 {
                next_fragment += 1;
            }
            let data_file_count =
                data_file_counts
                    .get(next_fragment)
                    .copied()
                    .ok_or_else(|| {
                        corrupt(
                            &path,
                            format!("manifest has an unclaimed DataFile row {decoded_data_files}"),
                        )
                    })?;
            let fragment = fragments.get_mut(next_fragment).ok_or_else(|| {
                corrupt(
                    &path,
                    format!("manifest has an unclaimed DataFile row {decoded_data_files}"),
                )
            })?;
            fragment.files.push(decoder.decode(row, &mut interner)?);
            decoded_data_files = decoded_data_files
                .checked_add(1)
                .ok_or_else(|| corrupt(&path, "decoded manifest DataFile count overflows u64"))?;
            if fragment.files.len() == data_file_count as usize {
                next_fragment += 1;
            }
        }
    }
    if decoded_data_files != data_file_count {
        return Err(corrupt(
            &path,
            format!("manifest decoded {decoded_data_files} DataFiles, expected {data_file_count}"),
        ));
    }
    for (fragment, data_file_count) in fragments.iter().zip(&data_file_counts) {
        if fragment.files.len() != *data_file_count as usize {
            return Err(corrupt(
                &path,
                format!(
                    "manifest fragment {} decoded {} DataFiles, expected {}",
                    fragment.id,
                    fragment.files.len(),
                    data_file_count
                ),
            ));
        }
    }

    drop(data_file_counts);
    manifest.replace_fragments(Arc::new(fragments), &path)?;
    let reconstructed = manifest_layout(&manifest)?;
    if reconstructed.data_file_count != data_file_count
        || reconstructed.fragment_pages != fragment_pages
        || reconstructed.logical_pages != logical_pages
    {
        return Err(corrupt(
            &path,
            "manifest navigation metadata does not match fragment contents",
        ));
    }
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::num::NonZero;

    use arrow_schema::{DataType, Field};
    use lance_core::utils::tempfile::TempObjDir;
    use lance_io::utils::read_last_block;
    use uuid::Uuid;

    use crate::format::DataStorageFormat;

    use super::*;

    fn test_manifest() -> Manifest {
        let schema = Schema::try_from(&ArrowSchema::new(vec![Field::new(
            "value",
            DataType::Int32,
            true,
        )]))
        .unwrap();
        let fragments = vec![
            Fragment {
                id: 2,
                files: vec![
                    DataFile::new(
                        "data/legacy-name.lance",
                        vec![0],
                        vec![0],
                        2,
                        1,
                        NonZero::new(1234),
                        None,
                    ),
                    DataFile::new(
                        "imported/file.parquet",
                        vec![1, 2],
                        vec![3, -1],
                        2,
                        1,
                        NonZero::new(5678),
                        Some(7),
                    ),
                ],
                deletion_file: Some(DeletionFile {
                    read_version: 11,
                    id: 99,
                    file_type: DeletionFileType::Bitmap,
                    num_deleted_rows: Some(2),
                    base_id: Some(7),
                }),
                row_id_meta: Some(RowIdMeta::Inline(vec![1, 2, 3])),
                physical_rows: Some(10),
                last_updated_at_version_meta: Some(RowDatasetVersionMeta::Inline(Arc::from([
                    4_u8, 5, 6,
                ]))),
                created_at_version_meta: Some(RowDatasetVersionMeta::External(ExternalFile {
                    path: "metadata/created.bin".to_string(),
                    offset: 8,
                    size: 13,
                })),
            },
            Fragment {
                id: 8,
                files: Vec::new(),
                deletion_file: None,
                row_id_meta: Some(RowIdMeta::External(ExternalFile {
                    path: "metadata/row_ids.bin".to_string(),
                    offset: 21,
                    size: 34,
                })),
                physical_rows: Some(0),
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
            },
            Fragment {
                id: 9,
                files: Vec::new(),
                deletion_file: Some(DeletionFile {
                    read_version: 12,
                    id: 100,
                    file_type: DeletionFileType::Array,
                    num_deleted_rows: Some(0),
                    base_id: None,
                }),
                row_id_meta: Some(RowIdMeta::Inline(Vec::new())),
                physical_rows: Some(5),
                last_updated_at_version_meta: Some(RowDatasetVersionMeta::Inline(Arc::from(
                    Vec::<u8>::new(),
                ))),
                created_at_version_meta: None,
            },
        ];
        let mut manifest = Manifest::new(
            schema,
            Arc::new(fragments),
            DataStorageFormat::new(LanceFileVersion::V2_3),
            HashMap::new(),
        );
        manifest.version = 42;
        manifest.tag = Some("roundtrip".to_string());
        manifest
            .config
            .insert("manifest-test".to_string(), "true".to_string());
        manifest
    }

    fn test_index() -> IndexMetadata {
        IndexMetadata {
            uuid: Uuid::from_u128(1),
            fields: vec![0],
            name: "test-index".to_string(),
            dataset_version: 42,
            fragment_bitmap: None,
            index_details: None,
            index_version: 1,
            created_at: None,
            base_id: None,
            files: None,
        }
    }

    async fn write_raw_index_payload(
        store: &ObjectStore,
        path: &Path,
        manifest: &Manifest,
        payload: Vec<u8>,
    ) {
        let writer = store.create(path).await.unwrap();
        let captured = Arc::new(Mutex::new(None));
        let capturing_writer = CapturingWriter {
            inner: writer,
            result: captured,
        };
        let lance_schema = Schema::try_from(arrow_schema().as_ref()).unwrap();
        let mut file_writer = FileWriter::try_new(
            Box::new(capturing_writer),
            lance_schema,
            FileWriterOptions {
                format_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            },
        )
        .unwrap();
        file_writer.add_schema_metadata(FORMAT_METADATA_KEY, FORMAT_IDENTITY);
        file_writer.add_schema_metadata(
            FORMAT_VERSION_METADATA_KEY,
            MANIFEST_SCHEMA_VERSION.to_string(),
        );
        let layout = manifest_layout(manifest).unwrap();
        file_writer
            .write_column(
                HEADER_COLUMN,
                Arc::new(header_array(manifest, &layout, 1, 0).unwrap()),
            )
            .await
            .unwrap();
        file_writer
            .write_column(INDICES_COLUMN, Arc::new(payload_array(&[payload]).unwrap()))
            .await
            .unwrap();
        file_writer.finish().await.unwrap();
    }

    #[tokio::test]
    async fn round_trip_columnar_manifest() {
        let store = ObjectStore::memory();
        let path = Path::from("/columnar.manifest");
        let writer = store.create(&path).await.unwrap();
        let mut expected = test_manifest();
        let index = test_index();
        let transaction = Transaction::from(pb::Transaction::default());

        let write_result = write(
            writer,
            &mut expected,
            Some(vec![index.clone()]),
            Some(transaction.clone()),
        )
        .await
        .unwrap();
        assert!(write_result.size > 0);
        assert_eq!(expected.index_section, None);
        assert_eq!(expected.transaction_section, None);
        assert_eq!(
            expected.index_section_source().unwrap(),
            Some(crate::format::ManifestSectionSource::ColumnarProjection)
        );
        assert_eq!(
            expected.transaction_section_source().unwrap(),
            Some(crate::format::ManifestSectionSource::ColumnarProjection)
        );

        let reader: Arc<dyn Reader> = store.open(&path).await.unwrap().into();
        let tail = read_last_block(reader.as_ref()).await.unwrap();
        assert!(is_columnar_footer(&tail).unwrap());
        let actual = read_with_prefetched_tail(&store, reader.clone(), tail)
            .await
            .unwrap();
        assert_eq!(actual, expected);

        assert_eq!(read_indexes(&store, &path, None).await.unwrap(), [index]);
        assert_eq!(
            read_transaction(&store, &path, None).await.unwrap(),
            Some(transaction)
        );
        let file_reader = open_file_reader(&store, &path, None).await.unwrap();
        let first_page_buffer = file_reader
            .metadata()
            .column_metadatas
            .iter()
            .flat_map(|column| &column.pages)
            .flat_map(|page| page.buffer_offsets.iter().copied())
            .min();
        assert_eq!(first_page_buffer, Some(0));
        assert!(
            !file_reader
                .schema()
                .metadata
                .contains_key("lance:manifest_prefix_bytes")
        );
    }

    #[tokio::test]
    async fn round_trip_empty_columnar_manifest() {
        let store = ObjectStore::memory();
        let path = Path::from("/empty-columnar.manifest");
        let writer = store.create(&path).await.unwrap();
        let mut expected = Manifest::new(
            Schema::default(),
            Arc::new(Vec::new()),
            DataStorageFormat::new(LanceFileVersion::V2_3),
            HashMap::new(),
        );
        write(writer, &mut expected, None, None).await.unwrap();
        assert_eq!(expected.index_section, None);
        assert_eq!(expected.transaction_section, None);
        assert_eq!(expected.index_section_source().unwrap(), None);
        assert_eq!(expected.transaction_section_source().unwrap(), None);
        let reader: Arc<dyn Reader> = store.open(&path).await.unwrap().into();
        let tail = read_last_block(reader.as_ref()).await.unwrap();
        let actual = read_with_prefetched_tail(&store, reader, tail)
            .await
            .unwrap();
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn rejects_invalid_projected_index_payload() {
        let store = ObjectStore::memory();
        let path = Path::from("/invalid-index-payload.manifest");
        let manifest = Manifest::new(
            Schema::default(),
            Arc::new(Vec::new()),
            DataStorageFormat::new(LanceFileVersion::V2_3),
            HashMap::new(),
        );
        write_raw_index_payload(&store, &path, &manifest, vec![0xff]).await;

        let reader: Arc<dyn Reader> = store.open(&path).await.unwrap().into();
        let tail = read_last_block(reader.as_ref()).await.unwrap();
        let decoded = read_with_prefetched_tail(&store, reader, tail)
            .await
            .unwrap();
        assert_eq!(decoded.index_section, None);
        assert_eq!(
            decoded.index_section_source().unwrap(),
            Some(crate::format::ManifestSectionSource::ColumnarProjection)
        );
        let error = read_indexes(&store, &path, None).await.unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(
            error
                .to_string()
                .contains("index payload is invalid protobuf")
        );
    }

    #[test]
    fn rejects_unallocatable_row_count_without_panicking() {
        let path = Path::from("/huge-row-count.manifest");
        let result = std::panic::catch_unwind(|| {
            let mut rows = Vec::<u8>::new();
            try_reserve_manifest_rows(&mut rows, u64::MAX, &path, "indices")
        });
        let error = result
            .expect("huge manifest row count must not panic")
            .unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("indices count"));
    }

    #[test]
    fn rejects_protobuf_section_offsets_in_columnar_header() {
        let path = Path::from("/invalid-columnar-header.manifest");
        let mut manifest = test_manifest();
        manifest.index_section = Some(17);
        let layout = manifest_layout(&manifest).unwrap();
        let header = header_array(&manifest, &layout, 1, 0).unwrap();

        let Err(error) = decode_header(&header, 0, &path) else {
            panic!("columnar header unexpectedly accepted a protobuf offset");
        };
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(
            error
                .to_string()
                .contains("protobuf auxiliary section offset")
        );
    }

    #[tokio::test]
    async fn small_manifest_prefetches_remaining_bytes_once() {
        let store = ObjectStore::local();
        let temporary = TempObjDir::default();
        let path = (*temporary)
            .clone()
            .join("small-prefetched-columnar.manifest");
        let writer = store.create(&path).await.unwrap();
        let schema = test_manifest().schema;
        let fragments = (0..1_500_u64)
            .map(|id| Fragment {
                id,
                files: vec![DataFile::new(
                    format!("data/{id:032x}.lance"),
                    vec![0],
                    vec![0],
                    2,
                    3,
                    NonZero::new(1_024),
                    None,
                )],
                deletion_file: None,
                row_id_meta: None,
                physical_rows: Some(1),
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
            })
            .collect();
        let mut manifest = Manifest::new(
            schema,
            Arc::new(fragments),
            DataStorageFormat::new(LanceFileVersion::V2_3),
            HashMap::new(),
        );

        let result = write(writer, &mut manifest, None, None).await.unwrap();
        assert!(result.size <= SMALL_MANIFEST_PREFETCH_BYTES);
        let reader: Arc<dyn Reader> = store.open(&path).await.unwrap().into();
        let tail = read_last_block(reader.as_ref()).await.unwrap();
        assert!(tail.len() < result.size);

        let _ = store.io_stats_incremental();
        let actual = read_with_prefetched_tail(&store, reader, tail)
            .await
            .unwrap();
        let stats = store.io_stats_incremental();
        assert_eq!(stats.read_iops, 1);
        assert_eq!(actual, manifest);
    }

    #[test]
    fn builds_sparse_navigation_per_section_page() {
        let fragments = (0..=SECTION_BATCH_SIZE)
            .map(|id| Fragment {
                id: id as u64,
                files: Vec::new(),
                deletion_file: None,
                row_id_meta: None,
                physical_rows: Some(1),
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
            })
            .collect();
        let manifest = Manifest::new(
            Schema::default(),
            Arc::new(fragments),
            DataStorageFormat::default(),
            HashMap::new(),
        );

        let layout = manifest_layout(&manifest).unwrap();
        assert_eq!(layout.fragment_pages.len(), 2);
        assert_eq!(layout.logical_pages.len(), 2);
        assert_eq!(layout.fragment_pages[0].fragment_row_count, 65_536);
        assert_eq!(layout.fragment_pages[1].fragment_row_start, 65_536);
        assert_eq!(layout.fragment_pages[1].fragment_row_count, 1);
        assert_eq!(layout.logical_pages[1].logical_row_start, 65_536);
        assert_eq!(layout.logical_pages[1].logical_row_end, 65_537);
    }

    #[test]
    fn bounds_fragment_batches_by_variable_width_bytes() {
        let shared = Arc::<[u8]>::from(vec![0_u8; 512 * 1024]);
        let fragments = (0..70)
            .map(|id| Fragment {
                id,
                files: Vec::new(),
                deletion_file: None,
                row_id_meta: None,
                physical_rows: Some(1),
                last_updated_at_version_meta: Some(RowDatasetVersionMeta::Inline(shared.clone())),
                created_at_version_meta: None,
            })
            .collect::<Vec<_>>();

        assert_eq!(fragment_batch_ranges(&fragments).unwrap(), [0..64, 64..70]);
        let manifest = Manifest::new(
            Schema::default(),
            Arc::new(fragments),
            DataStorageFormat::new(LanceFileVersion::V2_3),
            HashMap::new(),
        );
        let layout = manifest_layout(&manifest).unwrap();
        assert_eq!(layout.fragment_pages.len(), 2);
        assert_eq!(layout.fragment_pages[0].fragment_row_count, 64);
        assert_eq!(layout.fragment_pages[1].fragment_row_start, 64);
        assert_eq!(layout.logical_pages[1].logical_row_start, 64);
    }

    #[test]
    fn validates_write_preconditions() {
        let mut missing_rows = test_manifest();
        Arc::make_mut(&mut missing_rows.fragments)[0].physical_rows = None;
        assert!(!can_write(&missing_rows).unwrap());

        let mut invalid_order = test_manifest();
        Arc::make_mut(&mut invalid_order.fragments)[1].id = 1;
        assert!(
            can_write(&invalid_order)
                .unwrap_err()
                .to_string()
                .contains("strictly increasing")
        );

        let mut invalid_deletion = test_manifest();
        Arc::make_mut(&mut invalid_deletion.fragments)[0]
            .deletion_file
            .as_mut()
            .unwrap()
            .num_deleted_rows = Some(11);
        assert!(
            can_write(&invalid_deletion)
                .unwrap_err()
                .to_string()
                .contains("deleted rows")
        );
    }

    #[test]
    fn detects_container_footer_versions() {
        let mut columnar = vec![0; 8];
        LittleEndian::write_u16(&mut columnar[0..2], 2);
        LittleEndian::write_u16(&mut columnar[2..4], 1);
        columnar[4..].copy_from_slice(lance_file::format::MAGIC);
        assert!(is_columnar_footer(&Bytes::from(columnar)).unwrap());

        let mut protobuf = vec![0; 8];
        LittleEndian::write_u16(&mut protobuf[0..2], 0);
        LittleEndian::write_u16(&mut protobuf[2..4], 2);
        protobuf[4..].copy_from_slice(lance_file::format::MAGIC);
        assert!(!is_columnar_footer(&Bytes::from(protobuf)).unwrap());
    }

    #[test]
    fn derives_managed_file_path() {
        let id = [
            0b1010_0101,
            0b0101_1010,
            0b1111_0000,
            0x01,
            0x23,
            0x45,
            0x67,
            0x89,
            0xab,
            0xcd,
            0xef,
            0x10,
            0x32,
            0x54,
            0x76,
            0x98,
        ];
        assert_eq!(
            managed_file_path(&id),
            "1010010101011010111100000123456789abcdef1032547698.lance"
        );
    }

    #[test]
    fn decodes_data_file_lists_from_a_sliced_batch() {
        let files = [
            DataFile::new("first.lance", vec![1], vec![10], 2, 3, None, None),
            DataFile::new(
                "second.lance",
                vec![2, 3],
                vec![20, 30],
                2,
                3,
                NonZero::new(7),
                Some(4),
            ),
        ];
        let array = data_file_array(&[&files[0], &files[1]]).unwrap();
        let sliced = array.slice(1, 1);
        let sliced = sliced.as_any().downcast_ref::<StructArray>().unwrap();
        let path = Path::from("/sliced.manifest");
        let decoder = DataFileBatchDecoder::new(sliced, &path).unwrap();

        assert_eq!(
            decoder
                .decode(0, &mut DataFileFieldInterner::default())
                .unwrap(),
            files[1]
        );
    }

    #[test]
    fn rejects_null_data_file_list_values() {
        let file = DataFile::new("null-list.lance", vec![1], vec![10], 2, 3, None, None);
        let array = data_file_array(&[&file]).unwrap();
        let mut columns = array.columns().to_vec();
        let mut null_fields = ListBuilder::new(Int32Builder::new());
        null_fields.values().append_null();
        null_fields.append(true);
        columns[2] = Arc::new(null_fields.finish());
        let array = StructArray::try_new(data_file_fields(), columns, None).unwrap();
        let path = Path::from("/null-list.manifest");
        let decoder = DataFileBatchDecoder::new(&array, &path).unwrap();

        let error = decoder
            .decode(0, &mut DataFileFieldInterner::default())
            .unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(
            error
                .to_string()
                .contains("DataFile fields[0] is null at row 0")
        );
    }
}
