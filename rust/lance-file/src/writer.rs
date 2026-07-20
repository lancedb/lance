// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use core::panic;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use arrow_array::{Array, ArrayRef, RecordBatch, StructArray, make_array};

use arrow_buffer::NullBuffer;
use bytes::{Buf, BufMut, Bytes, BytesMut};
use futures::StreamExt;
use futures::stream::FuturesOrdered;
use lance_core::datatypes::{Field, Schema as LanceSchema};
use lance_core::utils::bit::pad_bytes;
use lance_core::{Error, Result};
use lance_encoding::decoder::PageEncoding;
use lance_encoding::encoder::{
    BatchEncoder, EncodeTask, EncodedBatch, EncodedPage, EncodingOptions, FieldEncoder,
    FieldEncodingStrategy, OutOfLineBuffers, default_encoding_strategy,
};
use lance_encoding::repdef::RepDefBuilder;
use lance_encoding::version::LanceFileVersion;
use lance_io::object_store::ObjectStore;
use lance_io::traits::Writer;
use log::{debug, warn};
use object_store::path::Path;
use prost::Message;
use prost_types::Any;
use tokio::io::AsyncWrite;
use tokio::io::AsyncWriteExt;
use tracing::instrument;

use crate::datatypes::FieldsWithMeta;
use crate::format::MAGIC;
use crate::format::pb;
use crate::format::pbfile;
use crate::format::pbfile::DirectEncoding;

/// Pages buffers are aligned to 64 bytes
pub(crate) const PAGE_BUFFER_ALIGNMENT: usize = 64;
const PAD_BUFFER: [u8; PAGE_BUFFER_ALIGNMENT] = [72; PAGE_BUFFER_ALIGNMENT];
// In 2.1+, we split large pages on read instead of write to avoid empty pages
// and small pages issues. However, we keep the write-time limit at 32MB to avoid
// potential regressions in 2.0 format readers.
//
// This limit is not applied in the 2.1 writer
const MAX_PAGE_BYTES: usize = 32 * 1024 * 1024;
const ENV_LANCE_FILE_WRITER_MAX_PAGE_BYTES: &str = "LANCE_FILE_WRITER_MAX_PAGE_BYTES";

/// Summary of a completed Lance file write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileWriteSummary {
    /// The number of rows written to the file.
    pub num_rows: u64,
    /// The final size of the file in bytes.
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Default)]
pub struct FileWriterOptions {
    /// How many bytes to use for buffering column data
    ///
    /// When data comes in small batches the writer will buffer column data so that
    /// larger pages can be created.  This value will be divided evenly across all of the
    /// columns.  Generally you want this to be at least large enough to match your
    /// filesystem's ideal read size per column.
    ///
    /// In some cases you might want this value to be even larger if you have highly
    /// compressible data.  However, if this is too large, then the writer could require
    /// a lot of memory and write performance may suffer if the CPU-expensive encoding
    /// falls behind and can't be interleaved with the I/O expensive flushing.
    ///
    /// The default will use 8MiB per column which should be reasonable for most cases.
    // TODO: Do we need to be able to set this on a per-column basis?
    pub data_cache_bytes: Option<u64>,
    /// A hint to indicate the max size of a page
    ///
    /// This hint can't always be respected.  A single value could be larger than this value
    /// and we never slice single values.  In addition, there are some cases where it can be
    /// difficult to know size up-front and so we might not be able to respect this value.
    pub max_page_bytes: Option<u64>,
    /// The file writer buffers columns until enough data has arrived to flush a page
    /// to disk.
    ///
    /// Some columns with small data types may not flush very often.  These arrays can
    /// stick around for a long time.  These arrays might also be keeping larger data
    /// structures alive.  By default, the writer will make a deep copy of this array
    /// to avoid any potential memory leaks.  However, this can be disabled for a
    /// (probably minor) performance boost if you are sure that arrays are not keeping
    /// any sibling structures alive (this typically means the array was allocated in
    /// the same language / runtime as the writer)
    ///
    /// Do not enable this if your data is arriving from the C data interface.
    /// Data typically arrives one "batch" at a time (encoded in the C data interface
    /// as a struct array).  Each array in that batch keeps the entire batch alive.
    /// This means a small boolean array (which we will buffer in memory for quite a
    /// while) might keep a much larger record batch around in memory (even though most
    /// of that batch's data has been written to disk)
    pub keep_original_array: Option<bool>,
    pub encoding_strategy: Option<Arc<dyn FieldEncodingStrategy>>,
    /// The format version to use when writing the file
    ///
    /// This controls which encodings will be used when encoding the data.  Newer
    /// versions may have more efficient encodings.  However, newer format versions will
    /// require more up-to-date readers to read the data.
    pub format_version: Option<LanceFileVersion>,
}

// Total in-memory budget for buffering serialized page metadata before flushing
// to the spill file. Divided evenly across columns (with a floor of 64 bytes).
const DEFAULT_SPILL_BUFFER_LIMIT: usize = 256 * 1024;

/// Spills serialized page metadata to a temporary file to bound memory usage.
///
/// The spill file is an unstructured sequence of "chunks". Each chunk is a
/// contiguous run of length-delimited protobuf `Page` messages belonging to a
/// single column. Chunks from different columns are interleaved in the order
/// they are flushed (i.e. whenever a column's in-memory buffer exceeds
/// `per_column_limit`). The `column_chunks` index records the (offset, length)
/// of every chunk so each column's pages can be read back and reassembled in
/// order.
struct PageMetadataSpill {
    writer: Box<dyn Writer>,
    object_store: Arc<ObjectStore>,
    path: Path,
    /// Current write position in the spill file.
    position: u64,
    /// Per-column buffer of serialized (length-delimited protobuf) page metadata
    /// that has not yet been flushed to the spill file.
    column_buffers: Vec<Vec<u8>>,
    /// Per-column list of chunks that have been flushed to the spill file.
    /// Each entry is (offset, length) pointing into the spill file.
    column_chunks: Vec<Vec<(u64, u32)>>,
    /// Maximum bytes to buffer per column before flushing to the spill file.
    per_column_limit: usize,
}

impl PageMetadataSpill {
    async fn new(object_store: Arc<ObjectStore>, path: Path, num_columns: usize) -> Result<Self> {
        let writer = object_store.create(&path).await?;
        let per_column_limit = (DEFAULT_SPILL_BUFFER_LIMIT / num_columns.max(1)).max(64);
        Ok(Self {
            writer,
            object_store,
            path,
            position: 0,
            column_buffers: vec![Vec::new(); num_columns],
            column_chunks: vec![Vec::new(); num_columns],
            per_column_limit,
        })
    }

    async fn append_page(
        &mut self,
        column_idx: usize,
        page: &pbfile::column_metadata::Page,
    ) -> Result<()> {
        page.encode_length_delimited(&mut self.column_buffers[column_idx])
            .map_err(|e| {
                Error::io_source(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    e,
                )))
            })?;
        if self.column_buffers[column_idx].len() >= self.per_column_limit {
            self.flush_column(column_idx).await?;
        }
        Ok(())
    }

    async fn flush_column(&mut self, column_idx: usize) -> Result<()> {
        let buf = &self.column_buffers[column_idx];
        if buf.is_empty() {
            return Ok(());
        }
        let len = buf.len();
        self.writer.write_all(buf).await?;
        self.column_chunks[column_idx].push((self.position, len as u32));
        self.position += len as u64;
        self.column_buffers[column_idx].clear();
        Ok(())
    }

    async fn shutdown_writer(&mut self) -> Result<()> {
        for col_idx in 0..self.column_buffers.len() {
            self.flush_column(col_idx).await?;
        }
        Writer::shutdown(self.writer.as_mut()).await?;
        Ok(())
    }
}

fn decode_spilled_chunk(data: &Bytes) -> Result<Vec<pbfile::column_metadata::Page>> {
    let mut pages = Vec::new();
    let mut cursor = data.clone();
    while cursor.has_remaining() {
        let page =
            pbfile::column_metadata::Page::decode_length_delimited(&mut cursor).map_err(|e| {
                Error::io_source(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    e,
                )))
            })?;
        pages.push(page);
    }
    Ok(pages)
}

enum PageSpillState {
    Pending(Arc<ObjectStore>, Path),
    Active(PageMetadataSpill),
}

pub struct FileWriter {
    writer: Box<dyn Writer>,
    schema: Option<LanceSchema>,
    column_writers: Vec<Box<dyn FieldEncoder>>,
    column_metadata: Vec<pbfile::ColumnMetadata>,
    field_id_to_column_indices: Vec<(u32, u32)>,
    num_columns: u32,
    rows_written: u64,
    // The number of rows written for each top-level field (i.e. each entry in
    // `column_writers`). With `write_batch` every field advances together and
    // these are all equal, but `write_column` advances one field at a time, so
    // a single file may end up with columns of differing item counts.
    field_rows_written: Vec<u64>,
    global_buffers: Vec<(u64, u64)>,
    schema_metadata: HashMap<String, String>,
    options: FileWriterOptions,
    page_spill: Option<PageSpillState>,
}

fn initial_column_metadata() -> pbfile::ColumnMetadata {
    pbfile::ColumnMetadata {
        pages: Vec::new(),
        buffer_offsets: Vec::new(),
        buffer_sizes: Vec::new(),
        encoding: None,
    }
}

static WARNED_ON_UNSTABLE_API: AtomicBool = AtomicBool::new(false);

/// Verify that a non-nullable field contains no *unmasked* nulls.
///
/// When `allow_masked_child_nulls` is true, nulls in a non-nullable child of a
/// struct are accepted as long as every one of them is masked by a null
/// ancestor struct, matching Arrow's own `StructArray::try_new` validation.
/// This is required for Lance to be able to rewrite its own output: the
/// encoder pushes struct validity down into child arrays (`pushdown_nulls`)
/// and the decoder materializes the children of null structs as nulls, so a
/// read→write roundtrip (e.g. compaction) of a null struct with non-nullable
/// children produces exactly this shape.
///
/// `allow_masked_child_nulls` must only be set for file versions that encode
/// struct validity (2.1+).  The 2.0 encoding discards it, so a masked child
/// null would read back unmasked and fail `StructArray::try_new` at decode
/// time.
pub(crate) fn verify_field_nullability(
    arr: &dyn Array,
    field: &Field,
    ancestor_nulls: Option<&NullBuffer>,
    allow_masked_child_nulls: bool,
) -> Result<()> {
    // Logical nulls, not the physical buffer: read-back validation
    // (`StructArray::try_new`) uses `logical_nulls()`, and indirection types
    // (dictionary, run-end, union) carry logical nulls with no physical buffer
    // — e.g. a non-null key pointing at a null dictionary value. A physical-only
    // check would accept data that then fails to read back.
    let logical = arr.logical_nulls();
    if !field.nullable {
        let unmasked_null_count = match (logical.as_ref(), ancestor_nulls) {
            (None, _) => 0,
            (Some(nulls), Some(mask)) => NullBuffer::union(Some(nulls), Some(mask))
                .map(|combined| {
                    debug_assert!(
                        combined.null_count() >= mask.null_count(),
                        "the union of two null buffers cannot have fewer nulls than either input"
                    );
                    combined.null_count() - mask.null_count()
                })
                .unwrap_or(0),
            (Some(nulls), None) => nulls.null_count(),
        };
        if unmasked_null_count > 0 {
            return Err(Error::invalid_input(format!(
                "The field `{}` contained null values even though the field is marked non-null in the schema ({} null value(s) not masked by a null ancestor struct)",
                field.name, unmasked_null_count
            )));
        }
    }

    if let Some(struct_arr) = arr
        .as_any()
        .downcast_ref::<StructArray>()
        .filter(|_| allow_masked_child_nulls)
    {
        let child_mask = NullBuffer::union(ancestor_nulls, struct_arr.nulls());
        for (child_field, child) in field.children.iter().zip(struct_arr.columns()) {
            // Struct columns aren't sliced by the struct's own offset; align
            // them with the parent's validity window before recursing.
            let child = child.slice(struct_arr.offset(), struct_arr.len());
            verify_field_nullability(child.as_ref(), child_field, child_mask.as_ref(), true)?;
        }
        return Ok(());
    }

    let data = arr.to_data();
    for (child_field, child_data) in field.children.iter().zip(data.child_data()) {
        let child = make_array(child_data.clone());
        verify_field_nullability(child.as_ref(), child_field, None, allow_masked_child_nulls)?;
    }

    Ok(())
}

impl FileWriter {
    /// Create a new FileWriter with a desired output schema
    pub fn try_new(
        object_writer: Box<dyn Writer>,
        schema: LanceSchema,
        options: FileWriterOptions,
    ) -> Result<Self> {
        let mut writer = Self::new_lazy(object_writer, options);
        writer.initialize(schema)?;
        Ok(writer)
    }

    /// Create a new FileWriter without a desired output schema
    ///
    /// The output schema will be set based on the first batch of data to arrive.
    /// If no data arrives and the writer is finished then the write will fail.
    pub fn new_lazy(object_writer: Box<dyn Writer>, options: FileWriterOptions) -> Self {
        if let Some(format_version) = options.format_version
            && format_version.is_unstable()
            && WARNED_ON_UNSTABLE_API
                .compare_exchange(
                    false,
                    true,
                    std::sync::atomic::Ordering::Relaxed,
                    std::sync::atomic::Ordering::Relaxed,
                )
                .is_ok()
        {
            warn!(
                "You have requested an unstable format version.  Files written with this format version may not be readable in the future!  This is a development feature and should only be used for experimentation and never for production data."
            );
        }
        Self {
            writer: object_writer,
            schema: None,
            column_writers: Vec::new(),
            column_metadata: Vec::new(),
            num_columns: 0,
            rows_written: 0,
            field_rows_written: Vec::new(),
            field_id_to_column_indices: Vec::new(),
            global_buffers: Vec::new(),
            schema_metadata: HashMap::new(),
            page_spill: None,
            options,
        }
    }

    /// Spill page metadata to a sidecar file instead of accumulating in memory.
    ///
    /// This can dramatically reduce memory usage when many writers are open
    /// concurrently (e.g. IVF shuffle with thousands of partition writers).
    /// The sidecar file is created lazily on the first page write. The caller
    /// is responsible for cleaning up `path` (e.g. by placing it in a temp
    /// directory that is removed via RAII).
    pub fn with_page_metadata_spill(mut self, object_store: Arc<ObjectStore>, path: Path) -> Self {
        self.page_spill = Some(PageSpillState::Pending(object_store, path));
        self
    }

    /// Write a series of record batches to a new file
    ///
    /// Returns the number of rows written
    pub async fn create_file_with_batches(
        store: &ObjectStore,
        path: &Path,
        schema: lance_core::datatypes::Schema,
        batches: impl Iterator<Item = RecordBatch> + Send,
        options: FileWriterOptions,
    ) -> Result<usize> {
        let writer = store.create(path).await?;
        let mut writer = Self::try_new(writer, schema, options)?;
        for batch in batches {
            writer.write_batch(&batch).await?;
        }
        Ok(writer.finish().await?.num_rows as usize)
    }

    async fn do_write_buffer(writer: &mut (impl AsyncWrite + Unpin), buf: &[u8]) -> Result<()> {
        writer.write_all(buf).await?;
        let pad_bytes = pad_bytes::<PAGE_BUFFER_ALIGNMENT>(buf.len());
        writer.write_all(&PAD_BUFFER[..pad_bytes]).await?;
        Ok(())
    }

    /// Returns the format version that will be used when writing the file
    pub fn version(&self) -> LanceFileVersion {
        self.options.format_version.unwrap_or_default()
    }

    async fn write_page(&mut self, encoded_page: EncodedPage) -> Result<()> {
        let buffers = encoded_page.data;
        let mut buffer_offsets = Vec::with_capacity(buffers.len());
        let mut buffer_sizes = Vec::with_capacity(buffers.len());
        for buffer in buffers {
            buffer_offsets.push(self.writer.tell().await? as u64);
            buffer_sizes.push(buffer.len() as u64);
            Self::do_write_buffer(&mut self.writer, &buffer).await?;
        }
        let encoded_encoding = match encoded_page.description {
            PageEncoding::Legacy(array_encoding) => Any::from_msg(&array_encoding)?.encode_to_vec(),
            PageEncoding::Structural(page_layout) => Any::from_msg(&page_layout)?.encode_to_vec(),
        };
        let page = pbfile::column_metadata::Page {
            buffer_offsets,
            buffer_sizes,
            encoding: Some(pbfile::Encoding {
                location: Some(pbfile::encoding::Location::Direct(DirectEncoding {
                    encoding: encoded_encoding,
                })),
            }),
            length: encoded_page.num_rows,
            priority: encoded_page.row_number,
        };
        let col_idx = encoded_page.column_idx as usize;
        if matches!(&self.page_spill, Some(PageSpillState::Pending(..))) {
            let Some(PageSpillState::Pending(store, path)) = self.page_spill.take() else {
                unreachable!()
            };
            self.page_spill = Some(PageSpillState::Active(
                PageMetadataSpill::new(store, path, self.num_columns as usize).await?,
            ));
        }
        match &mut self.page_spill {
            Some(PageSpillState::Active(spill)) => spill.append_page(col_idx, &page).await?,
            None => self.column_metadata[col_idx].pages.push(page),
            Some(PageSpillState::Pending(..)) => unreachable!(),
        }
        Ok(())
    }

    #[instrument(skip_all, level = "debug")]
    async fn write_pages(&mut self, mut encoding_tasks: FuturesOrdered<EncodeTask>) -> Result<()> {
        // As soon as an encoding task is done we write it.  There is no parallelism
        // needed here because "writing" is really just submitting the buffer to the
        // underlying write scheduler (either the OS or object_store's scheduler for
        // cloud writes).  The only time we might truly await on write_page is if the
        // scheduler's write queue is full.
        //
        // Also, there is no point in trying to make write_page parallel anyways
        // because we wouldn't want buffers getting mixed up across pages.
        while let Some(encoding_task) = encoding_tasks.next().await {
            let encoded_page = encoding_task?;
            self.write_page(encoded_page).await?;
        }
        // It's important to flush here, we don't know when the next batch will arrive
        // and the underlying cloud store could have writes in progress that won't advance
        // until we interact with the writer again.  These in-progress writes will time out
        // if we don't flush.
        self.writer.flush().await?;
        Ok(())
    }

    /// Schedule batches of data to be written to the file
    pub async fn write_batches(
        &mut self,
        batches: impl Iterator<Item = &RecordBatch>,
    ) -> Result<()> {
        for batch in batches {
            self.write_batch(batch).await?;
        }
        Ok(())
    }

    fn verify_nullability_constraints(&self, batch: &RecordBatch) -> Result<()> {
        let allow_masked_child_nulls = self.version().resolve() >= LanceFileVersion::V2_1;
        for (col, field) in batch
            .columns()
            .iter()
            .zip(self.schema.as_ref().unwrap().fields.iter())
        {
            verify_field_nullability(col.as_ref(), field, None, allow_masked_child_nulls)?;
        }
        Ok(())
    }

    fn initialize(&mut self, mut schema: LanceSchema) -> Result<()> {
        let cache_bytes_per_column = if let Some(data_cache_bytes) = self.options.data_cache_bytes {
            data_cache_bytes / schema.fields.len() as u64
        } else {
            8 * 1024 * 1024
        };

        let max_page_bytes = self.options.max_page_bytes.unwrap_or_else(|| {
            std::env::var(ENV_LANCE_FILE_WRITER_MAX_PAGE_BYTES)
                .map(|s| {
                    s.parse::<u64>().unwrap_or_else(|e| {
                        warn!(
                            "Failed to parse {}: {}, using default",
                            ENV_LANCE_FILE_WRITER_MAX_PAGE_BYTES, e
                        );
                        MAX_PAGE_BYTES as u64
                    })
                })
                .unwrap_or(MAX_PAGE_BYTES as u64)
        });

        schema.validate()?;

        let keep_original_array = self.options.keep_original_array.unwrap_or(false);
        let encoding_strategy = self.options.encoding_strategy.clone().unwrap_or_else(|| {
            let version = self.version();
            default_encoding_strategy(version).into()
        });

        let encoding_options = EncodingOptions {
            cache_bytes_per_column,
            max_page_bytes,
            keep_original_array,
            buffer_alignment: PAGE_BUFFER_ALIGNMENT as u64,
            version: self.version(),
        };
        let encoder =
            BatchEncoder::try_new(&schema, encoding_strategy.as_ref(), &encoding_options)?;
        self.num_columns = encoder.num_columns();

        self.field_rows_written = vec![0; encoder.field_encoders.len()];
        self.column_writers = encoder.field_encoders;
        self.column_metadata = vec![initial_column_metadata(); self.num_columns as usize];
        self.field_id_to_column_indices = encoder.field_id_to_column_index;
        self.schema_metadata
            .extend(std::mem::take(&mut schema.metadata));
        self.schema = Some(schema);
        Ok(())
    }

    fn ensure_initialized(&mut self, batch: &RecordBatch) -> Result<&LanceSchema> {
        if self.schema.is_none() {
            let schema = LanceSchema::try_from(batch.schema().as_ref())?;
            self.initialize(schema)?;
        }
        Ok(self.schema.as_ref().unwrap())
    }

    #[instrument(skip_all, level = "debug")]
    fn encode_batch(
        &mut self,
        batch: &RecordBatch,
        external_buffers: &mut OutOfLineBuffers,
    ) -> Result<Vec<Vec<EncodeTask>>> {
        let field_arrays = self
            .schema
            .as_ref()
            .unwrap()
            .fields
            .iter()
            .enumerate()
            .map(|(field_idx, field)| {
                let array =
                    batch
                        .column_by_name(&field.name)
                        .ok_or(Error::invalid_input_source(
                            format!(
                                "Cannot write batch.  The batch was missing the column `{}`",
                                field.name
                            )
                            .into(),
                        ))?;
                Ok((field_idx, array.clone()))
            })
            .collect::<Result<Vec<_>>>()?;
        self.encode_columns(&field_arrays, external_buffers)
    }

    // Encode a set of `(field index, array)` pairs, each advancing only its own
    // column. Each task captures its field's current row offset at encode time,
    // so `advance_columns` must run after this call (never before); the order of
    // the returned tasks relative to `write_pages` does not matter.
    fn encode_columns(
        &mut self,
        field_arrays: &[(usize, ArrayRef)],
        external_buffers: &mut OutOfLineBuffers,
    ) -> Result<Vec<Vec<EncodeTask>>> {
        // Snapshot the starting row number of each field before borrowing the
        // column writers mutably below.
        let row_numbers = field_arrays
            .iter()
            .map(|(field_idx, _)| self.field_rows_written[*field_idx])
            .collect::<Vec<_>>();
        field_arrays
            .iter()
            .zip(row_numbers)
            .map(|((field_idx, array), row_number)| {
                let repdef = RepDefBuilder::default();
                let num_rows = array.len() as u64;
                self.column_writers[*field_idx].maybe_encode(
                    array.clone(),
                    external_buffers,
                    repdef,
                    row_number,
                    num_rows,
                )
            })
            .collect::<Result<Vec<_>>>()
    }

    // Advance the per-field row counters after a set of columns has been
    // written, keeping `rows_written` (the file's logical length) in sync as the
    // longest column. Only the written fields move, so their new totals fold into
    // `rows_written` directly without rescanning every field. (`write_batch`
    // advances every field uniformly and tracks this inline instead.)
    fn advance_columns(&mut self, field_arrays: &[(usize, ArrayRef)]) {
        for (field_idx, array) in field_arrays {
            let new_total = self.field_rows_written[*field_idx] + array.len() as u64;
            self.field_rows_written[*field_idx] = new_total;
            self.rows_written = self.rows_written.max(new_total);
        }
    }

    /// Schedule a batch of data to be written to the file
    ///
    /// Note: the future returned by this method may complete before the data has been fully
    /// flushed to the file (some data may be in the data cache or the I/O cache)
    pub async fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        debug!(
            "write_batch called with {} rows, {} columns, and {} bytes of data",
            batch.num_rows(),
            batch.num_columns(),
            batch.get_array_memory_size()
        );
        self.ensure_initialized(batch)?;
        self.verify_nullability_constraints(batch)?;
        let num_rows = batch.num_rows() as u64;
        if num_rows == 0 {
            return Ok(());
        }
        if num_rows > u32::MAX as u64 {
            return Err(Error::invalid_input_source(
                "cannot write Lance files with more than 2^32 rows".into(),
            ));
        }
        // First we push each array into its column writer.  This may or may not generate enough
        // data to trigger an encoding task.  We collect any encoding tasks into a queue.
        let mut external_buffers =
            OutOfLineBuffers::new(self.tell().await?, PAGE_BUFFER_ALIGNMENT as u64);
        let encoding_tasks = self.encode_batch(batch, &mut external_buffers)?;
        // Next, write external buffers
        for external_buffer in external_buffers.take_buffers() {
            Self::do_write_buffer(&mut self.writer, &external_buffer).await?;
        }

        let encoding_tasks = encoding_tasks
            .into_iter()
            .flatten()
            .collect::<FuturesOrdered<_>>();

        // `write_batch` advances every field by the same amount, so the longest
        // column simply grows by `num_rows`. Guard against overflowing the row
        // counter.
        if self.rows_written.checked_add(num_rows).is_none() {
            return Err(Error::invalid_input_source(format!("cannot write batch with {} rows because {} rows have already been written and Lance files cannot contain more than 2^64 rows", num_rows, self.rows_written).into()));
        }
        for field_rows in self.field_rows_written.iter_mut() {
            *field_rows += num_rows;
        }
        self.rows_written += num_rows;

        self.write_pages(encoding_tasks).await?;

        Ok(())
    }

    /// Write a single column, advancing only that column's row counter.
    ///
    /// Unlike [`write_batch`](Self::write_batch), which advances every column
    /// from a single shared row counter, this method advances one column
    /// independently. Used across calls it produces a single file whose columns
    /// may have different item counts.
    ///
    /// `column_index` refers to a top-level field in the writer's schema (the
    /// same order as the schema's fields); a nested child cannot be targeted on
    /// its own. Because each call writes the whole field from a single array, the
    /// children of a struct field always advance together and stay equal-length;
    /// only different top-level fields can diverge in length. A column may be
    /// written across multiple calls; its values are appended. A field that is
    /// never written ends up as a zero-length column. The writer must have been
    /// created with an explicit schema (via [`try_new`](Self::try_new)); a lazy
    /// schema cannot be inferred here because individual calls need not cover
    /// every field.
    ///
    /// ```
    /// # use arrow_array::{ArrayRef, Int32Array};
    /// # use std::sync::Arc;
    /// # use lance_file::writer::FileWriter;
    /// # async fn example(writer: &mut FileWriter) -> lance_core::Result<()> {
    /// // Field 0 gets three values, field 1 gets one — a non-rectangular file.
    /// writer.write_column(0, Arc::new(Int32Array::from(vec![1, 2, 3]))).await?;
    /// writer.write_column(1, Arc::new(Int32Array::from(vec![10]))).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn write_column(&mut self, column_index: usize, array: ArrayRef) -> Result<()> {
        let schema = self.schema.as_ref().ok_or_else(|| {
            Error::invalid_input_source(
                "write_column requires the writer to be created with an explicit schema".into(),
            )
        })?;
        let field = schema.fields.get(column_index).ok_or_else(|| {
            Error::invalid_input_source(
                format!(
                    "write_column: field index {} is out of bounds (schema has {} fields)",
                    column_index,
                    schema.fields.len()
                )
                .into(),
            )
        })?;
        if array.len() as u64 > u32::MAX as u64 {
            return Err(Error::invalid_input_source(
                "cannot write Lance files with more than 2^32 rows".into(),
            ));
        }
        verify_field_nullability(
            array.as_ref(),
            field,
            None,
            self.version().resolve() >= LanceFileVersion::V2_1,
        )?;

        // A never-advanced field simply remains a zero-length column, which the
        // encoders handle at `finish` time.
        if array.is_empty() {
            return Ok(());
        }

        let columns = [(column_index, array)];
        let mut external_buffers =
            OutOfLineBuffers::new(self.tell().await?, PAGE_BUFFER_ALIGNMENT as u64);
        let encoding_tasks = self.encode_columns(&columns, &mut external_buffers)?;
        for external_buffer in external_buffers.take_buffers() {
            Self::do_write_buffer(&mut self.writer, &external_buffer).await?;
        }
        let encoding_tasks = encoding_tasks
            .into_iter()
            .flatten()
            .collect::<FuturesOrdered<_>>();

        self.advance_columns(&columns);
        self.write_pages(encoding_tasks).await?;
        Ok(())
    }

    async fn write_column_metadata(
        &mut self,
        metadata: pbfile::ColumnMetadata,
    ) -> Result<(u64, u64)> {
        let metadata_bytes = metadata.encode_to_vec();
        let position = self.writer.tell().await? as u64;
        let len = metadata_bytes.len() as u64;
        self.writer.write_all(&metadata_bytes).await?;
        Ok((position, len))
    }

    async fn write_column_metadatas(&mut self) -> Result<Vec<(u64, u64)>> {
        let metadatas = std::mem::take(&mut self.column_metadata);

        // If spilling, finalize the spill writer and reopen for reading.
        // The spill file itself is cleaned up by the caller (it lives in a
        // temp directory managed by the caller's RAII guard).
        let spill_state = self.page_spill.take();
        let (spill_chunks, spill_reader) =
            if let Some(PageSpillState::Active(mut spill)) = spill_state {
                spill.shutdown_writer().await?;
                let reader = spill.object_store.open(&spill.path).await?;
                let chunks = std::mem::take(&mut spill.column_chunks);
                (chunks, Some(reader))
            } else {
                (Vec::new(), None)
            };

        let mut metadata_positions = Vec::with_capacity(metadatas.len());
        for (col_idx, mut metadata) in metadatas.into_iter().enumerate() {
            if let Some(reader) = &spill_reader {
                let mut pages = Vec::new();
                for &(offset, len) in &spill_chunks[col_idx] {
                    let data = reader
                        .get_range(offset as usize..(offset as usize + len as usize))
                        .await
                        .map_err(|e| Error::io_source(Box::new(e)))?;
                    pages.extend(decode_spilled_chunk(&data)?);
                }
                metadata.pages = pages;
            }
            metadata_positions.push(self.write_column_metadata(metadata).await?);
        }

        Ok(metadata_positions)
    }

    fn make_file_descriptor(
        schema: &lance_core::datatypes::Schema,
        num_rows: u64,
    ) -> Result<pb::FileDescriptor> {
        let fields_with_meta = FieldsWithMeta::from(schema);
        Ok(pb::FileDescriptor {
            schema: Some(pb::Schema {
                fields: fields_with_meta.fields.0,
                metadata: fields_with_meta.metadata,
            }),
            length: num_rows,
        })
    }

    async fn write_global_buffers(&mut self) -> Result<Vec<(u64, u64)>> {
        let schema = self.schema.as_mut().ok_or(Error::invalid_input("No schema provided on writer open and no data provided.  Schema is unknown and file cannot be created"))?;
        schema.metadata = std::mem::take(&mut self.schema_metadata);
        // Use descriptor layout for blob v2 fields in the footer to avoid exposing logical child fields.
        schema
            .fields
            .iter_mut()
            .for_each(|f| f.unload_blobs_recursive());

        let file_descriptor = Self::make_file_descriptor(schema, self.rows_written)?;
        let file_descriptor_bytes = file_descriptor.encode_to_vec();
        let file_descriptor_len = file_descriptor_bytes.len() as u64;
        let file_descriptor_position = self.writer.tell().await? as u64;
        self.writer.write_all(&file_descriptor_bytes).await?;
        let mut gbo_table = Vec::with_capacity(1 + self.global_buffers.len());
        gbo_table.push((file_descriptor_position, file_descriptor_len));
        gbo_table.append(&mut self.global_buffers);
        Ok(gbo_table)
    }

    /// Add a metadata entry to the schema
    ///
    /// This method is useful because sometimes the metadata is not known until after the
    /// data has been written.  This method allows you to alter the schema metadata.  It
    /// must be called before `finish` is called.
    pub fn add_schema_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.schema_metadata.insert(key.into(), value.into());
    }

    /// Prepare the writer when column data and metadata were produced externally.
    ///
    /// This is useful for flows that copy already-encoded pages (e.g., binary copy
    /// during compaction) where the column buffers have been written directly and we
    /// only need to write the footer and schema metadata. The provided
    /// `column_metadata` must describe the buffers already persisted by the
    /// underlying `ObjectWriter`, and `rows_written` should reflect the total number
    /// of rows in those buffers.
    pub fn initialize_with_external_metadata(
        &mut self,
        schema: lance_core::datatypes::Schema,
        column_metadata: Vec<pbfile::ColumnMetadata>,
        rows_written: u64,
    ) {
        self.schema = Some(schema);
        self.num_columns = column_metadata.len() as u32;
        self.column_metadata = column_metadata;
        self.rows_written = rows_written;
    }

    /// Adds a global buffer to the file
    ///
    /// The global buffer can contain any arbitrary bytes.  It will be written to the disk
    /// immediately.  This method returns the index of the global buffer (this will always
    /// start at 1 and increment by 1 each time this method is called)
    pub async fn add_global_buffer(&mut self, buffer: Bytes) -> Result<u32> {
        let position = self.writer.tell().await? as u64;
        let len = buffer.len() as u64;
        Self::do_write_buffer(&mut self.writer, &buffer).await?;
        self.global_buffers.push((position, len));
        Ok(self.global_buffers.len() as u32)
    }

    async fn finish_writers(&mut self) -> Result<()> {
        let mut col_idx = 0;
        for mut writer in std::mem::take(&mut self.column_writers) {
            let mut external_buffers =
                OutOfLineBuffers::new(self.tell().await?, PAGE_BUFFER_ALIGNMENT as u64);
            let columns = writer.finish(&mut external_buffers).await?;
            for buffer in external_buffers.take_buffers() {
                self.writer.write_all(&buffer).await?;
            }
            debug_assert_eq!(
                columns.len(),
                writer.num_columns() as usize,
                "Expected {} columns from column at index {} and got {}",
                writer.num_columns(),
                col_idx,
                columns.len()
            );
            for column in columns {
                for page in column.final_pages {
                    self.write_page(page).await?;
                }
                let column_metadata = &mut self.column_metadata[col_idx];
                let mut buffer_pos = self.writer.tell().await? as u64;
                for buffer in column.column_buffers {
                    column_metadata.buffer_offsets.push(buffer_pos);
                    let mut size = 0;
                    Self::do_write_buffer(&mut self.writer, &buffer).await?;
                    size += buffer.len() as u64;
                    buffer_pos += size;
                    column_metadata.buffer_sizes.push(size);
                }
                let encoded_encoding = Any::from_msg(&column.encoding)?.encode_to_vec();
                column_metadata.encoding = Some(pbfile::Encoding {
                    location: Some(pbfile::encoding::Location::Direct(pbfile::DirectEncoding {
                        encoding: encoded_encoding,
                    })),
                });
                col_idx += 1;
            }
        }
        if col_idx != self.column_metadata.len() {
            panic!(
                "Column writers finished with {} columns but we expected {}",
                col_idx,
                self.column_metadata.len()
            );
        }
        Ok(())
    }

    /// Converts self.version (which is a mix of "software version" and
    /// "format version" into a format version)
    fn version_to_numbers(&self) -> (u16, u16) {
        let version = self.options.format_version.unwrap_or_default();
        match version.resolve() {
            LanceFileVersion::V2_0 => (0, 3),
            LanceFileVersion::V2_1 => (2, 1),
            LanceFileVersion::V2_2 => (2, 2),
            LanceFileVersion::V2_3 => (2, 3),
            _ => panic!("Unsupported version: {}", version),
        }
    }

    /// Finishes writing the file
    ///
    /// This method will wait until all data has been flushed to the file.  Then it
    /// will write the file metadata and the footer.  It will not return until all
    /// data has been flushed and the file has been closed.
    ///
    /// Returns a summary of the completed file write.
    pub async fn finish(&mut self) -> Result<FileWriteSummary> {
        // 1. flush any remaining data and write out those pages
        let mut external_buffers =
            OutOfLineBuffers::new(self.tell().await?, PAGE_BUFFER_ALIGNMENT as u64);
        let encoding_tasks = self
            .column_writers
            .iter_mut()
            .map(|writer| writer.flush(&mut external_buffers))
            .collect::<Result<Vec<_>>>()?;
        for external_buffer in external_buffers.take_buffers() {
            Self::do_write_buffer(&mut self.writer, &external_buffer).await?;
        }
        let encoding_tasks = encoding_tasks
            .into_iter()
            .flatten()
            .collect::<FuturesOrdered<_>>();
        self.write_pages(encoding_tasks).await?;

        if !self.column_writers.is_empty() {
            self.finish_writers().await?;
        }

        // 3. write global buffers (we write the schema here)
        let global_buffer_offsets = self.write_global_buffers().await?;
        let num_global_buffers = global_buffer_offsets.len() as u32;

        // 4. write the column metadatas
        let column_metadata_start = self.writer.tell().await? as u64;
        let metadata_positions = self.write_column_metadatas().await?;

        // 5. write the column metadata offset table
        let cmo_table_start = self.writer.tell().await? as u64;
        for (meta_pos, meta_len) in metadata_positions {
            self.writer.write_u64_le(meta_pos).await?;
            self.writer.write_u64_le(meta_len).await?;
        }

        // 6. write global buffers offset table
        let gbo_table_start = self.writer.tell().await? as u64;
        for (gbo_pos, gbo_len) in global_buffer_offsets {
            self.writer.write_u64_le(gbo_pos).await?;
            self.writer.write_u64_le(gbo_len).await?;
        }

        let (major, minor) = self.version_to_numbers();
        // 7. write the footer
        self.writer.write_u64_le(column_metadata_start).await?;
        self.writer.write_u64_le(cmo_table_start).await?;
        self.writer.write_u64_le(gbo_table_start).await?;
        self.writer.write_u32_le(num_global_buffers).await?;
        self.writer.write_u32_le(self.num_columns).await?;
        self.writer.write_u16_le(major).await?;
        self.writer.write_u16_le(minor).await?;
        self.writer.write_all(MAGIC).await?;

        // 7. close the writer
        let write_result = Writer::shutdown(self.writer.as_mut()).await?;

        Ok(FileWriteSummary {
            num_rows: self.rows_written,
            size_bytes: write_result.size as u64,
        })
    }

    pub async fn abort(&mut self) {
        // For multipart uploads, ObjectWriter's Drop impl will abort
        // the upload when the writer is dropped.
    }

    pub async fn tell(&mut self) -> Result<u64> {
        Ok(self.writer.tell().await? as u64)
    }

    pub fn field_id_to_column_indices(&self) -> &[(u32, u32)] {
        &self.field_id_to_column_indices
    }
}

/// Utility trait for converting EncodedBatch to Bytes using the
/// lance file format
pub trait EncodedBatchWriteExt {
    /// Serializes into a lance file, including the schema
    fn try_to_self_described_lance(&self, version: LanceFileVersion) -> Result<Bytes>;
    /// Serializes into a lance file, without the schema.
    ///
    /// The schema must be provided to deserialize the buffer
    fn try_to_mini_lance(&self, version: LanceFileVersion) -> Result<Bytes>;
}

// Creates a lance footer and appends it to the encoded data
//
// The logic here is very similar to logic in the FileWriter except we
// are using BufMut (put_xyz) instead of AsyncWrite (write_xyz).
fn concat_lance_footer(
    batch: &EncodedBatch,
    write_schema: bool,
    version: LanceFileVersion,
) -> Result<Bytes> {
    // Estimating 1MiB for file footer
    let mut data = BytesMut::with_capacity(batch.data.len() + 1024 * 1024);
    data.put(batch.data.clone());
    // write global buffers (we write the schema here)
    let global_buffers = if write_schema {
        let schema_start = data.len() as u64;
        let lance_schema = lance_core::datatypes::Schema::try_from(batch.schema.as_ref())?;
        let descriptor = FileWriter::make_file_descriptor(&lance_schema, batch.num_rows)?;
        let descriptor_bytes = descriptor.encode_to_vec();
        let descriptor_len = descriptor_bytes.len() as u64;
        data.put(descriptor_bytes.as_slice());

        vec![(schema_start, descriptor_len)]
    } else {
        vec![]
    };
    let col_metadata_start = data.len() as u64;

    let mut col_metadata_positions = Vec::new();
    // Write column metadata
    for col in &batch.page_table {
        let position = data.len() as u64;
        let pages = col
            .page_infos
            .iter()
            .map(|page_info| {
                let encoded_encoding = match &page_info.encoding {
                    PageEncoding::Legacy(array_encoding) => {
                        Any::from_msg(array_encoding)?.encode_to_vec()
                    }
                    PageEncoding::Structural(page_layout) => {
                        Any::from_msg(page_layout)?.encode_to_vec()
                    }
                };
                let (buffer_offsets, buffer_sizes): (Vec<_>, Vec<_>) = page_info
                    .buffer_offsets_and_sizes
                    .as_ref()
                    .iter()
                    .cloned()
                    .unzip();
                Ok(pbfile::column_metadata::Page {
                    buffer_offsets,
                    buffer_sizes,
                    encoding: Some(pbfile::Encoding {
                        location: Some(pbfile::encoding::Location::Direct(DirectEncoding {
                            encoding: encoded_encoding,
                        })),
                    }),
                    length: page_info.num_rows,
                    priority: page_info.priority,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let (buffer_offsets, buffer_sizes): (Vec<_>, Vec<_>) =
            col.buffer_offsets_and_sizes.iter().cloned().unzip();
        let encoded_col_encoding = Any::from_msg(&col.encoding)?.encode_to_vec();
        let column = pbfile::ColumnMetadata {
            pages,
            buffer_offsets,
            buffer_sizes,
            encoding: Some(pbfile::Encoding {
                location: Some(pbfile::encoding::Location::Direct(pbfile::DirectEncoding {
                    encoding: encoded_col_encoding,
                })),
            }),
        };
        let column_bytes = column.encode_to_vec();
        col_metadata_positions.push((position, column_bytes.len() as u64));
        data.put(column_bytes.as_slice());
    }
    // Write column metadata offsets table
    let cmo_table_start = data.len() as u64;
    for (meta_pos, meta_len) in col_metadata_positions {
        data.put_u64_le(meta_pos);
        data.put_u64_le(meta_len);
    }
    // Write global buffers offsets table
    let gbo_table_start = data.len() as u64;
    let num_global_buffers = global_buffers.len() as u32;
    for (gbo_pos, gbo_len) in global_buffers {
        data.put_u64_le(gbo_pos);
        data.put_u64_le(gbo_len);
    }

    let (major, minor) = version.to_numbers();

    // write the footer
    data.put_u64_le(col_metadata_start);
    data.put_u64_le(cmo_table_start);
    data.put_u64_le(gbo_table_start);
    data.put_u32_le(num_global_buffers);
    data.put_u32_le(batch.page_table.len() as u32);
    data.put_u16_le(major as u16);
    data.put_u16_le(minor as u16);
    data.put(MAGIC.as_slice());

    Ok(data.freeze())
}

impl EncodedBatchWriteExt for EncodedBatch {
    fn try_to_self_described_lance(&self, version: LanceFileVersion) -> Result<Bytes> {
        concat_lance_footer(self, true, version)
    }

    fn try_to_mini_lance(&self, version: LanceFileVersion) -> Result<Bytes> {
        concat_lance_footer(self, false, version)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::reader::{FileReader, FileReaderOptions, ReaderProjection, describe_encoding};
    use crate::testing::FsFixture;
    use crate::writer::{ENV_LANCE_FILE_WRITER_MAX_PAGE_BYTES, FileWriter, FileWriterOptions};
    use arrow_array::builder::{Float32Builder, Int32Builder};
    use arrow_array::{ArrayRef, Int32Array, RecordBatch, UInt64Array};
    use arrow_array::{RecordBatchReader, StringArray, types::Float64Type};
    use arrow_schema::{DataType, Field, Field as ArrowField, Schema, Schema as ArrowSchema};
    use lance_core::cache::LanceCache;
    use lance_core::datatypes::Schema as LanceSchema;
    use lance_core::utils::tempfile::TempObjFile;
    use lance_datagen::{BatchCount, RowCount, array, gen_batch};
    use lance_encoding::compression_config::{CompressionFieldParams, CompressionParams};
    use lance_encoding::decoder::DecoderPlugins;
    use lance_encoding::version::LanceFileVersion;
    use lance_io::object_store::ObjectStore;
    use lance_io::utils::CachedFileSize;
    use rstest::rstest;

    #[tokio::test]
    async fn test_basic_write() {
        let tmp_path = TempObjFile::default();
        let obj_store = Arc::new(ObjectStore::local());

        let reader = gen_batch()
            .col("score", array::rand::<Float64Type>())
            .into_reader_rows(RowCount::from(1000), BatchCount::from(10));

        let writer = obj_store.create(&tmp_path).await.unwrap();

        let lance_schema =
            lance_core::datatypes::Schema::try_from(reader.schema().as_ref()).unwrap();

        let mut file_writer =
            FileWriter::try_new(writer, lance_schema, FileWriterOptions::default()).unwrap();

        for batch in reader {
            file_writer.write_batch(&batch.unwrap()).await.unwrap();
        }
        file_writer.add_schema_metadata("foo", "bar");
        file_writer.finish().await.unwrap();
        // Tests asserting the contents of the written file are in reader.rs
    }

    #[tokio::test]
    async fn test_write_empty() {
        let tmp_path = TempObjFile::default();
        let obj_store = Arc::new(ObjectStore::local());

        let reader = gen_batch()
            .col("score", array::rand::<Float64Type>())
            .into_reader_rows(RowCount::from(0), BatchCount::from(0));

        let writer = obj_store.create(&tmp_path).await.unwrap();

        let lance_schema =
            lance_core::datatypes::Schema::try_from(reader.schema().as_ref()).unwrap();

        let mut file_writer =
            FileWriter::try_new(writer, lance_schema, FileWriterOptions::default()).unwrap();

        for batch in reader {
            file_writer.write_batch(&batch.unwrap()).await.unwrap();
        }
        file_writer.add_schema_metadata("foo", "bar");
        file_writer.finish().await.unwrap();
    }

    // Read a single column back at an explicit range/index set, returning its
    // `Int32` values. Reading one column (or an equal-length group) at a time is
    // how unequal-length files are consumed: a full scan across columns of
    // differing lengths cannot form a single rectangular batch.
    async fn read_int32_column(
        reader: &FileReader,
        schema: &LanceSchema,
        version: LanceFileVersion,
        name: &str,
        params: lance_io::ReadBatchParams,
    ) -> Vec<Option<i32>> {
        use futures::TryStreamExt;
        use lance_encoding::decoder::FilterExpression;

        let projection = ReaderProjection::from_column_names(version, schema, &[name]).unwrap();
        let batches: Vec<RecordBatch> = reader
            .read_stream_projected(params, 1024, 16, projection, FilterExpression::no_filter())
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// A single file may hold columns of differing item counts, written by
    /// advancing each column's row counter independently (no shared global
    /// counter).
    #[rstest]
    #[tokio::test]
    async fn test_write_columns_unequal_lengths(
        #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
    ) {
        use lance_io::ReadBatchParams;

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, true),
            ArrowField::new("b", DataType::Int32, true),
            ArrowField::new("c", DataType::Int32, true),
        ]));
        let lance_schema = LanceSchema::try_from(arrow_schema.as_ref()).unwrap();

        let fs = FsFixture::default();
        let options = FileWriterOptions {
            format_version: Some(version),
            ..Default::default()
        };
        let mut writer = FileWriter::try_new(
            fs.object_store.create(&fs.tmp_path).await.unwrap(),
            lance_schema.clone(),
            options,
        )
        .unwrap();

        // Field "a" gets 5 values across two calls (appending), field "b" gets a
        // single value, and field "c" is never written (a zero-length column).
        let a1: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let b: ArrayRef = Arc::new(Int32Array::from(vec![10]));
        writer.write_column(0, a1).await.unwrap();
        writer.write_column(1, b).await.unwrap();
        let a2: ArrayRef = Arc::new(Int32Array::from(vec![4, 5]));
        writer.write_column(0, a2).await.unwrap();
        // An empty array is a no-op whether or not the field already has rows:
        // field "a" keeps its 5 rows, field "c" stays a zero-length column.
        let empty: ArrayRef = Arc::new(Int32Array::from(Vec::<i32>::new()));
        writer.write_column(0, empty.clone()).await.unwrap();
        writer.write_column(2, empty).await.unwrap();

        let summary = writer.finish().await.unwrap();
        // The file's logical length is the longest column.
        assert_eq!(summary.num_rows, 5);

        let file_scheduler = fs
            .scheduler
            .open_file(&fs.tmp_path, &CachedFileSize::unknown())
            .await
            .unwrap();
        let reader = FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            &LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await
        .unwrap();

        // Per-column row counts are recorded in / derivable from file metadata.
        assert_eq!(reader.num_rows(), 5);
        assert_eq!(reader.column_num_rows(0).unwrap(), 5);
        assert_eq!(reader.column_num_rows(1).unwrap(), 1);
        assert_eq!(reader.column_num_rows(2).unwrap(), 0);
        assert!(reader.column_num_rows(3).is_err());

        // Each column reads back independently at its own length.
        assert_eq!(
            read_int32_column(
                &reader,
                &lance_schema,
                version,
                "a",
                ReadBatchParams::Range(0..5)
            )
            .await,
            vec![Some(1), Some(2), Some(3), Some(4), Some(5)],
        );
        assert_eq!(
            read_int32_column(
                &reader,
                &lance_schema,
                version,
                "b",
                ReadBatchParams::Range(0..1)
            )
            .await,
            vec![Some(10)],
        );

        // Random access by position within the longer column returns the right
        // value even though other columns are shorter. (The take path requires
        // strictly increasing indices.)
        assert_eq!(
            read_int32_column(
                &reader,
                &lance_schema,
                version,
                "a",
                ReadBatchParams::Indices(arrow_array::UInt32Array::from(vec![0, 2, 4])),
            )
            .await,
            vec![Some(1), Some(3), Some(5)],
        );
    }

    /// Reading an unequal-length file:
    /// - a projection whose columns are equal length full-scans normally;
    /// - a full scan across columns of differing length is rejected up front,
    ///   before any batch is produced (even though a prefix would be rectangular);
    /// - a bounded read is valid as long as every projected column covers it;
    /// - a single-column `RangeFull` resolves to that column's own length, not
    ///   the file's (maximum) length.
    #[rstest]
    #[tokio::test]
    async fn test_read_unequal_length_projection(
        #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
    ) {
        use futures::TryStreamExt;
        use lance_encoding::decoder::FilterExpression;
        use lance_io::ReadBatchParams;

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, true),
            ArrowField::new("b", DataType::Int32, true),
            ArrowField::new("c", DataType::Int32, true),
        ]));
        let lance_schema = LanceSchema::try_from(arrow_schema.as_ref()).unwrap();
        let fs = FsFixture::default();
        let options = FileWriterOptions {
            format_version: Some(version),
            ..Default::default()
        };
        let mut writer = FileWriter::try_new(
            fs.object_store.create(&fs.tmp_path).await.unwrap(),
            lance_schema.clone(),
            options,
        )
        .unwrap();
        // "a" and "b" are equal length (5); "c" is shorter (1).
        writer
            .write_column(0, Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])))
            .await
            .unwrap();
        writer
            .write_column(1, Arc::new(Int32Array::from(vec![6, 7, 8, 9, 10])))
            .await
            .unwrap();
        writer
            .write_column(2, Arc::new(Int32Array::from(vec![100])))
            .await
            .unwrap();
        writer.finish().await.unwrap();

        let file_scheduler = fs
            .scheduler
            .open_file(&fs.tmp_path, &CachedFileSize::unknown())
            .await
            .unwrap();
        let reader = FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            &LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await
        .unwrap();

        let read = |names: &'static [&'static str], params: ReadBatchParams| {
            let projection =
                ReaderProjection::from_column_names(version, &lance_schema, names).unwrap();
            async {
                match reader
                    .read_stream_projected(
                        params,
                        1024,
                        16,
                        projection,
                        FilterExpression::no_filter(),
                    )
                    .await
                {
                    Ok(stream) => stream.try_collect::<Vec<RecordBatch>>().await,
                    Err(e) => Err(e),
                }
            }
        };
        let col_values = |batches: &[RecordBatch], idx: usize| -> Vec<Option<i32>> {
            batches
                .iter()
                .flat_map(|b| {
                    b.column(idx)
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap()
                        .iter()
                        .collect::<Vec<_>>()
                })
                .collect()
        };

        // Equal-length projection [a, b] full-scans into rectangular batches.
        let batches = read(&["a", "b"], ReadBatchParams::RangeFull).await.unwrap();
        assert_eq!(
            col_values(&batches, 0),
            vec![Some(1), Some(2), Some(3), Some(4), Some(5)]
        );
        assert_eq!(
            col_values(&batches, 1),
            vec![Some(6), Some(7), Some(8), Some(9), Some(10)]
        );

        // A mismatched-length projection [a, c] (5 vs 1) is rejected before any
        // batch is yielded, regardless of the read params — its columns cannot
        // be combined into rectangular batches. The error names each column's
        // length so the caller can see which column is the odd one out.
        let err = read(&["a", "c"], ReadBatchParams::RangeFull)
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("a=5") && err.contains("c=1"),
            "error should name each column's length, got: {err}"
        );
        assert!(
            read(&["a", "c"], ReadBatchParams::Range(0..1))
                .await
                .is_err(),
            "even a common-prefix read of unequal-length columns must error"
        );

        // A single-column RangeFull resolves to that column's own length.
        let batches = read(&["c"], ReadBatchParams::RangeFull).await.unwrap();
        assert_eq!(col_values(&batches, 0), vec![Some(100)]);
        let batches = read(&["a"], ReadBatchParams::RangeFull).await.unwrap();
        assert_eq!(
            col_values(&batches, 0),
            vec![Some(1), Some(2), Some(3), Some(4), Some(5)]
        );

        // RangeFrom/RangeTo likewise resolve against the projected column's own
        // length rather than the file's longest column.
        let batches = read(&["a"], ReadBatchParams::RangeFrom(2..)).await.unwrap();
        assert_eq!(col_values(&batches, 0), vec![Some(3), Some(4), Some(5)]);
        // RangeFrom on the short column "c" resolves to length 1, not 5.
        let batches = read(&["c"], ReadBatchParams::RangeFrom(0..)).await.unwrap();
        assert_eq!(col_values(&batches, 0), vec![Some(100)]);
        let batches = read(&["a"], ReadBatchParams::RangeTo(..3)).await.unwrap();
        assert_eq!(col_values(&batches, 0), vec![Some(1), Some(2), Some(3)]);
        // A bound past the projected column's length errors.
        assert!(
            read(&["a"], ReadBatchParams::RangeTo(..6)).await.is_err(),
            "RangeTo past the column length must error"
        );
        assert!(
            read(&["c"], ReadBatchParams::RangeFrom(2..)).await.is_err(),
            "RangeFrom past the column length must error"
        );
    }

    /// A struct and a list column each map to multiple physical columns, and a
    /// list's item column is longer than its top-level row count. The
    /// projection-length check must partition `column_indices` by top-level
    /// field and use each field's root column, so an ordinary (rectangular) file
    /// with nested columns still reads under the new validation path.
    #[rstest]
    #[tokio::test]
    async fn test_read_nested_columns_under_validation(
        #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
    ) {
        use arrow_array::types::Int32Type;
        use arrow_array::{ListArray, StructArray};
        use futures::TryStreamExt;
        use lance_encoding::decoder::FilterExpression;
        use lance_io::ReadBatchParams;

        let struct_type = DataType::Struct(
            vec![
                ArrowField::new("x", DataType::Int32, true),
                ArrowField::new("y", DataType::Int32, true),
            ]
            .into(),
        );
        let list_type = DataType::List(Arc::new(ArrowField::new("item", DataType::Int32, true)));
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, true),
            ArrowField::new("s", struct_type, true),
            ArrowField::new("lst", list_type, true),
        ]));
        let lance_schema = LanceSchema::try_from(arrow_schema.as_ref()).unwrap();

        let a: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let s: ArrayRef = Arc::new(StructArray::from(vec![
            (
                Arc::new(ArrowField::new("x", DataType::Int32, true)),
                Arc::new(Int32Array::from(vec![10, 20, 30])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("y", DataType::Int32, true)),
                Arc::new(Int32Array::from(vec![11, 21, 31])) as ArrayRef,
            ),
        ]));
        // 3 lists, 6 items: the item column is longer than the top-level rows.
        let lst: ArrayRef = Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(1), Some(2)]),
            Some(vec![Some(3)]),
            Some(vec![Some(4), Some(5), Some(6)]),
        ]));
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![a, s, lst]).unwrap();

        let fs = FsFixture::default();
        let options = FileWriterOptions {
            format_version: Some(version),
            ..Default::default()
        };
        let mut writer = FileWriter::try_new(
            fs.object_store.create(&fs.tmp_path).await.unwrap(),
            lance_schema.clone(),
            options,
        )
        .unwrap();
        writer.write_batch(&batch).await.unwrap();
        writer.finish().await.unwrap();

        let file_scheduler = fs
            .scheduler
            .open_file(&fs.tmp_path, &CachedFileSize::unknown())
            .await
            .unwrap();
        let reader = FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            &LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await
        .unwrap();

        // If `validate_field_length` mispartitioned the physical columns, the
        // length check would read the wrong root column (e.g. the list's item
        // column, length 6) and spuriously reject this rectangular file.
        for names in [&["a", "s", "lst"][..], &["a", "lst"][..], &["a", "s"][..]] {
            let projection =
                ReaderProjection::from_column_names(version, &lance_schema, names).unwrap();
            let batches: Vec<RecordBatch> = reader
                .read_stream_projected(
                    ReadBatchParams::RangeFull,
                    1024,
                    16,
                    projection,
                    FilterExpression::no_filter(),
                )
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(
                total_rows, 3,
                "projection {names:?} should read 3 top-level rows"
            );
        }
    }

    /// `write_column` rejects invalid inputs at the API boundary with
    /// descriptive errors: a writer without an explicit schema, an
    /// out-of-bounds field index, and a null written into a non-nullable field.
    #[tokio::test]
    async fn test_write_column_validation_errors() {
        // A lazy-schema writer cannot infer the schema from a single column.
        let fs = FsFixture::default();
        let mut lazy_writer = FileWriter::new_lazy(
            fs.object_store.create(&fs.tmp_path).await.unwrap(),
            FileWriterOptions::default(),
        );
        let err = lazy_writer
            .write_column(0, Arc::new(Int32Array::from(vec![1, 2, 3])))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("explicit schema"),
            "expected explicit-schema error, got: {err}"
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new("b", DataType::Int32, true),
        ]));
        let lance_schema = LanceSchema::try_from(arrow_schema.as_ref()).unwrap();

        // An out-of-bounds field index is rejected, naming the index and count.
        let fs = FsFixture::default();
        let mut writer = FileWriter::try_new(
            fs.object_store.create(&fs.tmp_path).await.unwrap(),
            lance_schema.clone(),
            FileWriterOptions::default(),
        )
        .unwrap();
        let err = writer
            .write_column(5, Arc::new(Int32Array::from(vec![1])))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains('5') && err.contains('2'),
            "expected out-of-bounds error naming index 5 and 2 fields, got: {err}"
        );

        // A null in a non-nullable field ("a") is rejected.
        let err = writer
            .write_column(0, Arc::new(Int32Array::from(vec![Some(1), None, Some(3)])))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("non-null"),
            "expected nullability error, got: {err}"
        );
    }

    /// In 2.1+, nulls in a non-nullable struct child are accepted when every
    /// one of them is masked by the parent struct's validity (Arrow's own
    /// `StructArray::try_new` semantics).  This is the shape the decoder
    /// produces when it materializes null structs, so rejecting it would fail
    /// any read→write roundtrip (e.g. compaction) of data Lance itself wrote.
    /// Nulls the parent does not mask are still rejected, and so are masked
    /// nulls in 2.0, whose encoding discards struct validity and thus cannot
    /// roundtrip them.
    #[tokio::test]
    async fn test_masked_nulls_in_non_nullable_struct_child() {
        use crate::testing::read_lance_file;
        use arrow_array::{Array, Float64Array, StructArray, cast::AsArray};
        use arrow_buffer::NullBuffer;
        use arrow_schema::Fields;
        use lance_encoding::decoder::FilterExpression;

        let non_null_children = Fields::from(vec![
            ArrowField::new("x", DataType::Float64, false),
            ArrowField::new("y", DataType::Float64, false),
        ]);
        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "position",
            DataType::Struct(non_null_children.clone()),
            true,
        )]));
        let lance_schema = LanceSchema::try_from(arrow_schema.as_ref()).unwrap();

        let fs = FsFixture::default();
        let options = FileWriterOptions {
            format_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        };
        let mut writer = FileWriter::try_new(
            fs.object_store.create(&fs.tmp_path).await.unwrap(),
            lance_schema.clone(),
            options.clone(),
        )
        .unwrap();

        // Rows 1 and 3 are null structs and their children are null at exactly
        // those slots.
        let xs = Float64Array::from(vec![Some(1.0), None, Some(3.0), None, Some(5.0)]);
        let ys = Float64Array::from(vec![Some(10.0), None, Some(30.0), None, Some(50.0)]);
        let position = StructArray::try_new(
            non_null_children,
            vec![Arc::new(xs) as ArrayRef, Arc::new(ys) as ArrayRef],
            Some(NullBuffer::from(vec![true, false, true, false, true])),
        )
        .unwrap();
        let batch =
            RecordBatch::try_new(arrow_schema, vec![Arc::new(position) as ArrayRef]).unwrap();
        writer.write_batch(&batch).await.unwrap();
        writer.add_schema_metadata("foo", "bar");
        writer.finish().await.unwrap();

        // Roundtrip: the parent's validity and the valid rows' values are intact.
        let batches = read_lance_file(
            &fs,
            Arc::<DecoderPlugins>::default(),
            FilterExpression::no_filter(),
        )
        .await;
        assert_eq!(batches.len(), 1);
        let position = batches[0].column(0).as_struct();
        assert_eq!(position.len(), 5);
        let xs = position.column(0).as_primitive::<Float64Type>();
        let ys = position.column(1).as_primitive::<Float64Type>();
        for (idx, valid, x, y) in [
            (0, true, 1.0, 10.0),
            (1, false, 0.0, 0.0),
            (2, true, 3.0, 30.0),
            (3, false, 0.0, 0.0),
            (4, true, 5.0, 50.0),
        ] {
            assert_eq!(position.is_valid(idx), valid);
            if valid {
                assert_eq!((xs.value(idx), ys.value(idx)), (x, y));
            }
        }

        // 2.0 does not encode struct validity, so the same masked batch is
        // rejected there instead of writing a file that cannot be decoded.
        let fs = FsFixture::default();
        let mut writer = FileWriter::try_new(
            fs.object_store.create(&fs.tmp_path).await.unwrap(),
            lance_schema.clone(),
            FileWriterOptions {
                format_version: Some(LanceFileVersion::V2_0),
                ..Default::default()
            },
        )
        .unwrap();
        let err = writer.write_batch(&batch).await.unwrap_err();
        assert!(
            matches!(err, lance_core::Error::InvalidInput { .. }),
            "expected InvalidInput, got: {err:?}"
        );
        let err = err.to_string();
        assert!(
            err.contains("non-null"),
            "expected nullability error, got: {err}"
        );

        // A null in a non-nullable child that the parent does NOT mask is still
        // rejected.  Arrow's `StructArray::try_new` refuses to even build this
        // shape, so declare the children nullable on the array and let the
        // writer validate it against its schema, where they are non-null.
        let nullable_children = Fields::from(vec![
            ArrowField::new("x", DataType::Float64, true),
            ArrowField::new("y", DataType::Float64, true),
        ]);
        let xs = Float64Array::from(vec![Some(1.0), None]);
        let ys = Float64Array::from(vec![Some(10.0), Some(20.0)]);
        let bad_position = StructArray::try_new(
            nullable_children.clone(),
            vec![Arc::new(xs) as ArrayRef, Arc::new(ys) as ArrayRef],
            Some(NullBuffer::from(vec![true, true])),
        )
        .unwrap();
        let bad_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "position",
            DataType::Struct(nullable_children),
            true,
        )]));
        let bad_batch =
            RecordBatch::try_new(bad_schema, vec![Arc::new(bad_position) as ArrayRef]).unwrap();

        let fs = FsFixture::default();
        let mut writer = FileWriter::try_new(
            fs.object_store.create(&fs.tmp_path).await.unwrap(),
            lance_schema,
            options,
        )
        .unwrap();
        let err = writer.write_batch(&bad_batch).await.unwrap_err();
        assert!(
            matches!(err, lance_core::Error::InvalidInput { .. }),
            "expected InvalidInput, got: {err:?}"
        );
        let err = err.to_string();
        assert!(
            err.contains("non-null"),
            "expected nullability error, got: {err}"
        );
    }

    /// The masked-null acceptance recurses through nested structs: a null at
    /// any depth is accepted as long as some ancestor struct masks it, and the
    /// mask accumulates across levels (grandparent ∪ parent).  Nulls that no
    /// ancestor masks are still rejected, at any depth.
    #[tokio::test]
    async fn test_masked_nulls_in_nested_structs() {
        use crate::testing::read_lance_file;
        use arrow_array::{Array, Float64Array, StructArray, cast::AsArray};
        use arrow_buffer::NullBuffer;
        use arrow_schema::Fields;
        use lance_encoding::decoder::FilterExpression;

        // outer (nullable) > mid (non-null struct) > x (non-null f64)
        let leaf_fields = Fields::from(vec![ArrowField::new("x", DataType::Float64, false)]);
        let mid_field = ArrowField::new("mid", DataType::Struct(leaf_fields.clone()), false);
        let outer_fields = Fields::from(vec![mid_field]);
        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "outer",
            DataType::Struct(outer_fields.clone()),
            true,
        )]));
        let lance_schema = LanceSchema::try_from(arrow_schema.as_ref()).unwrap();

        // Rows 1 and 3 are null outer structs; mid and x carry nulls at exactly
        // those slots — masked only by the GRANDPARENT (outer), which exercises
        // the accumulated ancestor mask on the recursive path.
        let xs = Float64Array::from(vec![Some(1.0), None, Some(3.0), None, Some(5.0)]);
        let outer_nulls = NullBuffer::from(vec![true, false, true, false, true]);
        let mid = StructArray::try_new(
            leaf_fields.clone(),
            vec![Arc::new(xs) as ArrayRef],
            Some(outer_nulls.clone()),
        )
        .unwrap();
        let outer = StructArray::try_new(
            outer_fields,
            vec![Arc::new(mid) as ArrayRef],
            Some(outer_nulls),
        )
        .unwrap();
        let batch = RecordBatch::try_new(arrow_schema, vec![Arc::new(outer) as ArrayRef]).unwrap();

        let fs = FsFixture::default();
        let options = FileWriterOptions {
            format_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        };
        let mut writer = FileWriter::try_new(
            fs.object_store.create(&fs.tmp_path).await.unwrap(),
            lance_schema.clone(),
            options.clone(),
        )
        .unwrap();
        writer.write_batch(&batch).await.unwrap();
        writer.add_schema_metadata("foo", "bar");
        writer.finish().await.unwrap();

        let batches = read_lance_file(
            &fs,
            Arc::<DecoderPlugins>::default(),
            FilterExpression::no_filter(),
        )
        .await;
        assert_eq!(batches.len(), 1);
        let outer = batches[0].column(0).as_struct();
        let mid = outer.column(0).as_struct();
        let xs = mid.column(0).as_primitive::<Float64Type>();
        for (idx, valid, x) in [
            (0, true, 1.0),
            (1, false, 0.0),
            (2, true, 3.0),
            (3, false, 0.0),
            (4, true, 5.0),
        ] {
            assert_eq!(outer.is_valid(idx), valid);
            if valid {
                assert_eq!(xs.value(idx), x);
            }
        }

        // A leaf null under VALID outer and mid structs is unmasked at every
        // level and stays rejected.  (Arrow refuses to build that shape with
        // non-nullable fields, so declare them nullable on the arrays and let
        // the writer validate against its schema.)
        let leaf_nullable = Fields::from(vec![ArrowField::new("x", DataType::Float64, true)]);
        let mid_nullable = Fields::from(vec![ArrowField::new(
            "mid",
            DataType::Struct(leaf_nullable.clone()),
            true,
        )]);
        let xs = Float64Array::from(vec![Some(1.0), None]);
        let mid = StructArray::try_new(
            leaf_nullable,
            vec![Arc::new(xs) as ArrayRef],
            Some(NullBuffer::from(vec![true, true])),
        )
        .unwrap();
        let outer = StructArray::try_new(
            mid_nullable.clone(),
            vec![Arc::new(mid) as ArrayRef],
            Some(NullBuffer::from(vec![true, true])),
        )
        .unwrap();
        let bad_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "outer",
            DataType::Struct(mid_nullable),
            true,
        )]));
        let bad_batch =
            RecordBatch::try_new(bad_schema, vec![Arc::new(outer) as ArrayRef]).unwrap();

        let fs = FsFixture::default();
        let mut writer = FileWriter::try_new(
            fs.object_store.create(&fs.tmp_path).await.unwrap(),
            lance_schema,
            options,
        )
        .unwrap();
        let err = writer.write_batch(&bad_batch).await.unwrap_err();
        assert!(
            matches!(err, lance_core::Error::InvalidInput { .. }),
            "expected InvalidInput, got: {err:?}"
        );
        let err = err.to_string();
        assert!(
            err.contains("non-null"),
            "expected nullability error, got: {err}"
        );
    }

    /// The check reasons about *logical* nulls, not physical ones — matching the
    /// `StructArray::try_new` validation applied on read-back. A non-nullable
    /// dictionary child whose key is non-null but points at a null dictionary
    /// value carries a logical null with no physical null buffer on the keys;
    /// if the parent struct does not mask that row it must be rejected at write
    /// time, otherwise the file would be un-readable.
    /// Reproduces the review finding on <https://github.com/lance-format/lance/pull/7844>.
    #[tokio::test]
    async fn test_logical_nulls_in_non_nullable_dictionary_child() {
        use arrow_array::{Array, DictionaryArray, Int32Array, StructArray, types::Int32Type};
        use arrow_buffer::NullBuffer;
        use arrow_schema::Fields;

        // Dictionary child `d`: values=[10, null], keys=[0, 1, null].
        //   row 0 -> values[0]=10   (not null)
        //   row 1 -> values[1]=null (LOGICAL null, key itself is non-null)
        //   row 2 -> key null        (physical + logical null)
        let values = Int32Array::from(vec![Some(10), None]);
        let keys = Int32Array::from(vec![Some(0), Some(1), None]);
        let dict = DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values)).unwrap();

        // The child is declared nullable *on the array* so Arrow will build it
        // (StructArray::try_new is exactly the read-back validation we're
        // reproducing — it rejects the non-nullable form outright). The writer's
        // own schema below marks it non-null, which is what must be enforced.
        let dict_type = dict.data_type().clone();
        let batch_children = Fields::from(vec![ArrowField::new("d", dict_type.clone(), true)]);
        // Parent struct validity [true, true, false]: masks row 2 only.
        // Row 1's logical null is NOT masked -> must be rejected.
        let position = StructArray::try_new(
            batch_children.clone(),
            vec![Arc::new(dict) as ArrayRef],
            Some(NullBuffer::from(vec![true, true, false])),
        )
        .unwrap();
        let batch_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "s",
            DataType::Struct(batch_children),
            true,
        )]));
        let batch =
            RecordBatch::try_new(batch_schema, vec![Arc::new(position) as ArrayRef]).unwrap();

        // Writer schema: child `d` is non-nullable.
        let writer_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "s",
            DataType::Struct(Fields::from(vec![ArrowField::new("d", dict_type, false)])),
            true,
        )]));
        let lance_schema = LanceSchema::try_from(writer_schema.as_ref()).unwrap();

        let fs = FsFixture::default();
        let mut writer = FileWriter::try_new(
            fs.object_store.create(&fs.tmp_path).await.unwrap(),
            lance_schema,
            FileWriterOptions {
                format_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            },
        )
        .unwrap();
        let err = writer.write_batch(&batch).await.unwrap_err();
        assert!(
            matches!(err, lance_core::Error::InvalidInput { .. }),
            "expected InvalidInput (unmasked logical null), got: {err:?}"
        );
        assert!(
            err.to_string().contains("non-null"),
            "expected nullability error, got: {err}"
        );
    }

    /// A non-nullable struct holding a *nullable* child that legitimately
    /// contains nulls must be accepted: the child's nulls are its own right, not
    /// a violation of the struct's non-nullability. Guards against folding a
    /// child's logical nulls into the parent's node-level check.
    #[tokio::test]
    async fn test_non_null_struct_with_nullable_child_having_nulls() {
        use crate::testing::read_lance_file;
        use arrow_array::{Array, Int32Array, StructArray, cast::AsArray, types::Int32Type};
        use arrow_schema::Fields;
        use lance_encoding::decoder::FilterExpression;

        // struct `s` (non-null) { id: i32 non-null, name: i32 NULLABLE with nulls }
        let child_fields = Fields::from(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("name", DataType::Int32, true),
        ]);
        let ids = Int32Array::from(vec![1, 2, 3]);
        let names = Int32Array::from(vec![Some(10), None, Some(30)]);
        let s = StructArray::try_new(
            child_fields.clone(),
            vec![Arc::new(ids) as ArrayRef, Arc::new(names) as ArrayRef],
            None, // struct itself is never null
        )
        .unwrap();
        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "s",
            DataType::Struct(child_fields),
            false,
        )]));
        let lance_schema = LanceSchema::try_from(arrow_schema.as_ref()).unwrap();
        let batch = RecordBatch::try_new(arrow_schema, vec![Arc::new(s) as ArrayRef]).unwrap();

        let fs = FsFixture::default();
        let mut writer = FileWriter::try_new(
            fs.object_store.create(&fs.tmp_path).await.unwrap(),
            lance_schema,
            FileWriterOptions {
                format_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            },
        )
        .unwrap();
        writer.write_batch(&batch).await.unwrap();
        writer.add_schema_metadata("foo", "bar");
        writer.finish().await.unwrap();

        let batches = read_lance_file(
            &fs,
            Arc::<DecoderPlugins>::default(),
            FilterExpression::no_filter(),
        )
        .await;
        let s = batches[0].column(0).as_struct();
        let names = s.column(1).as_primitive::<Int32Type>();
        assert!(s.is_valid(0) && s.is_valid(1) && s.is_valid(2));
        assert_eq!(names.value(0), 10);
        assert!(names.is_null(1));
        assert_eq!(names.value(2), 30);
    }

    /// The blocking read path applies the same projection-length validation as
    /// the async path: a short single column resolves to its own length, and a
    /// mismatched-length projection errors up front.
    #[rstest]
    #[tokio::test]
    async fn test_blocking_read_unequal_length(
        #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
    ) {
        use lance_encoding::decoder::FilterExpression;
        use lance_io::ReadBatchParams;

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, true),
            ArrowField::new("c", DataType::Int32, true),
        ]));
        let lance_schema = LanceSchema::try_from(arrow_schema.as_ref()).unwrap();
        let fs = FsFixture::default();
        let options = FileWriterOptions {
            format_version: Some(version),
            ..Default::default()
        };
        let mut writer = FileWriter::try_new(
            fs.object_store.create(&fs.tmp_path).await.unwrap(),
            lance_schema.clone(),
            options,
        )
        .unwrap();
        writer
            .write_column(0, Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])))
            .await
            .unwrap();
        writer
            .write_column(1, Arc::new(Int32Array::from(vec![100])))
            .await
            .unwrap();
        writer.finish().await.unwrap();

        let file_scheduler = fs
            .scheduler
            .open_file(&fs.tmp_path, &CachedFileSize::unknown())
            .await
            .unwrap();
        let reader = Arc::new(
            FileReader::try_open(
                file_scheduler,
                None,
                Arc::<DecoderPlugins>::default(),
                &LanceCache::no_cache(),
                FileReaderOptions::default(),
            )
            .await
            .unwrap(),
        );

        // Single short column: RangeFull resolves to its own length (1).
        let proj_c = ReaderProjection::from_column_names(version, &lance_schema, &["c"]).unwrap();
        let reader_c = reader.clone();
        let batches = tokio::task::spawn_blocking(move || {
            reader_c
                .read_stream_projected_blocking(
                    ReadBatchParams::RangeFull,
                    1024,
                    Some(proj_c),
                    FilterExpression::no_filter(),
                )
                .unwrap()
                .collect::<std::result::Result<Vec<RecordBatch>, _>>()
                .unwrap()
        })
        .await
        .unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1);

        // A mismatched projection [a, c] errors on the blocking path too.
        let proj_ac =
            ReaderProjection::from_column_names(version, &lance_schema, &["a", "c"]).unwrap();
        let reader_ac = reader.clone();
        let is_err = tokio::task::spawn_blocking(move || {
            reader_ac
                .read_stream_projected_blocking(
                    ReadBatchParams::RangeFull,
                    1024,
                    Some(proj_ac),
                    FilterExpression::no_filter(),
                )
                .is_err()
        })
        .await
        .unwrap();
        assert!(
            is_err,
            "blocking full scan across unequal-length columns must error"
        );
    }

    /// Files written the ordinary (rectangular) way keep equal column lengths,
    /// so the unequal-length support is backwards compatible.
    #[tokio::test]
    async fn test_write_batch_keeps_equal_lengths() {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, true),
            ArrowField::new("b", DataType::Int32, true),
        ]));
        let lance_schema = LanceSchema::try_from(arrow_schema.as_ref()).unwrap();

        let fs = FsFixture::default();
        let mut writer = FileWriter::try_new(
            fs.object_store.create(&fs.tmp_path).await.unwrap(),
            lance_schema,
            FileWriterOptions::default(),
        )
        .unwrap();
        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(Int32Array::from(vec![4, 5, 6])),
            ],
        )
        .unwrap();
        writer.write_batch(&batch).await.unwrap();
        let summary = writer.finish().await.unwrap();
        assert_eq!(summary.num_rows, 3);

        let file_scheduler = fs
            .scheduler
            .open_file(&fs.tmp_path, &CachedFileSize::unknown())
            .await
            .unwrap();
        let reader = FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            &LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(reader.column_num_rows(0).unwrap(), 3);
        assert_eq!(reader.column_num_rows(1).unwrap(), 3);
    }

    #[tokio::test]
    async fn test_max_page_bytes_enforced() {
        let arrow_field = Field::new("data", DataType::UInt64, false);
        let arrow_schema = Schema::new(vec![arrow_field]);
        let lance_schema = LanceSchema::try_from(&arrow_schema).unwrap();

        // 8MiB
        let data: Vec<u64> = (0..1_000_000).collect();
        let array = UInt64Array::from(data);
        let batch =
            RecordBatch::try_new(arrow_schema.clone().into(), vec![Arc::new(array)]).unwrap();

        let options = FileWriterOptions {
            max_page_bytes: Some(1024 * 1024), // 1MB
            // This is a 2.0 only test because 2.1+ splits large pages on read instead of write
            format_version: Some(LanceFileVersion::V2_0),
            ..Default::default()
        };

        let path = TempObjFile::default();
        let object_store = ObjectStore::local();
        let mut writer = FileWriter::try_new(
            object_store.create(&path).await.unwrap(),
            lance_schema,
            options,
        )
        .unwrap();

        writer.write_batch(&batch).await.unwrap();
        writer.finish().await.unwrap();

        let fs = FsFixture::default();
        let file_scheduler = fs
            .scheduler
            .open_file(&path, &CachedFileSize::unknown())
            .await
            .unwrap();
        let file_reader = FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            &LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await
        .unwrap();

        let column_meta = file_reader.metadata();

        let mut total_page_num: u32 = 0;
        for (col_idx, col_metadata) in column_meta.column_metadatas.iter().enumerate() {
            assert!(
                !col_metadata.pages.is_empty(),
                "Column {} has no pages",
                col_idx
            );

            for (page_idx, page) in col_metadata.pages.iter().enumerate() {
                total_page_num += 1;
                let total_size: u64 = page.buffer_sizes.iter().sum();
                assert!(
                    total_size <= 1024 * 1024,
                    "Column {} Page {} size {} exceeds 1MB limit",
                    col_idx,
                    page_idx,
                    total_size
                );
            }
        }

        assert_eq!(total_page_num, 8)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_max_page_bytes_env_var() {
        let arrow_field = Field::new("data", DataType::UInt64, false);
        let arrow_schema = Schema::new(vec![arrow_field]);
        let lance_schema = LanceSchema::try_from(&arrow_schema).unwrap();
        // 4MiB
        let data: Vec<u64> = (0..500_000).collect();
        let array = UInt64Array::from(data);
        let batch =
            RecordBatch::try_new(arrow_schema.clone().into(), vec![Arc::new(array)]).unwrap();

        // 2MiB
        unsafe {
            std::env::set_var(ENV_LANCE_FILE_WRITER_MAX_PAGE_BYTES, "2097152");
        }

        let options = FileWriterOptions {
            max_page_bytes: None, // enforce env
            ..Default::default()
        };

        let path = TempObjFile::default();
        let object_store = ObjectStore::local();
        let mut writer = FileWriter::try_new(
            object_store.create(&path).await.unwrap(),
            lance_schema.clone(),
            options,
        )
        .unwrap();

        writer.write_batch(&batch).await.unwrap();
        writer.finish().await.unwrap();

        let fs = FsFixture::default();
        let file_scheduler = fs
            .scheduler
            .open_file(&path, &CachedFileSize::unknown())
            .await
            .unwrap();
        let file_reader = FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            &LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await
        .unwrap();

        for col_metadata in file_reader.metadata().column_metadatas.iter() {
            for page in col_metadata.pages.iter() {
                let total_size: u64 = page.buffer_sizes.iter().sum();
                assert!(
                    total_size <= 2 * 1024 * 1024,
                    "Page size {} exceeds 2MB limit",
                    total_size
                );
            }
        }

        unsafe {
            std::env::set_var(ENV_LANCE_FILE_WRITER_MAX_PAGE_BYTES, "");
        }
    }

    #[tokio::test]
    async fn test_compression_overrides_end_to_end() {
        // Create test schema with different column types
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("customer_id", DataType::Int32, false),
            ArrowField::new("product_id", DataType::Int32, false),
            ArrowField::new("quantity", DataType::Int32, false),
            ArrowField::new("price", DataType::Float32, false),
            ArrowField::new("description", DataType::Utf8, false),
        ]));

        let lance_schema = LanceSchema::try_from(arrow_schema.as_ref()).unwrap();

        // Create test data with patterns suitable for different compression
        let mut customer_ids = Int32Builder::new();
        let mut product_ids = Int32Builder::new();
        let mut quantities = Int32Builder::new();
        let mut prices = Float32Builder::new();
        let mut descriptions = Vec::new();

        // Generate data with specific patterns:
        // - customer_id: highly repetitive (good for RLE)
        // - product_id: moderately repetitive (good for RLE)
        // - quantity: random values (not good for RLE)
        // - price: some repetition
        // - description: long strings (good for Zstd)
        for i in 0..10000 {
            // Customer ID repeats every 100 rows (100 unique customers)
            // This creates runs of 100 identical values
            customer_ids.append_value(i / 100);

            // Product ID has only 5 unique values with long runs
            product_ids.append_value(i / 2000);

            // Quantity is mostly 1 with occasional other values
            quantities.append_value(if i % 10 == 0 { 5 } else { 1 });

            // Price has only 3 unique values
            prices.append_value(match i % 3 {
                0 => 9.99,
                1 => 19.99,
                _ => 29.99,
            });

            // Descriptions are repetitive but we'll keep them simple
            descriptions.push(format!("Product {}", i / 2000));
        }

        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(customer_ids.finish()),
                Arc::new(product_ids.finish()),
                Arc::new(quantities.finish()),
                Arc::new(prices.finish()),
                Arc::new(StringArray::from(descriptions)),
            ],
        )
        .unwrap();

        // Configure compression parameters
        let mut params = CompressionParams::new();

        // RLE for ID columns (ends with _id)
        params.columns.insert(
            "*_id".to_string(),
            CompressionFieldParams {
                rle_threshold: Some(0.5), // Lower threshold to trigger RLE more easily
                compression: None,        // Will use default compression if any
                compression_level: None,
                bss: Some(lance_encoding::compression_config::BssMode::Off), // Explicitly disable BSS to ensure RLE is used
                minichunk_size: None,
            },
        );

        // For now, we'll skip Zstd compression since it's not imported
        // In a real implementation, you could add other compression types here

        // Build encoding strategy with compression parameters
        let encoding_strategy = lance_encoding::encoder::default_encoding_strategy_with_params(
            LanceFileVersion::V2_1,
            params,
        )
        .unwrap();

        // Configure file writer options
        let options = FileWriterOptions {
            encoding_strategy: Some(Arc::from(encoding_strategy)),
            format_version: Some(LanceFileVersion::V2_1),
            max_page_bytes: Some(64 * 1024), // 64KB pages
            ..Default::default()
        };

        // Write the file
        let path = TempObjFile::default();
        let object_store = ObjectStore::local();

        let mut writer = FileWriter::try_new(
            object_store.create(&path).await.unwrap(),
            lance_schema.clone(),
            options,
        )
        .unwrap();

        writer.write_batch(&batch).await.unwrap();
        writer.add_schema_metadata("compression_test", "configured_compression");
        writer.finish().await.unwrap();

        // Now write the same data without compression overrides for comparison
        let path_no_compression = TempObjFile::default();
        let default_options = FileWriterOptions {
            format_version: Some(LanceFileVersion::V2_1),
            max_page_bytes: Some(64 * 1024),
            ..Default::default()
        };

        let mut writer_no_compression = FileWriter::try_new(
            object_store.create(&path_no_compression).await.unwrap(),
            lance_schema.clone(),
            default_options,
        )
        .unwrap();

        writer_no_compression.write_batch(&batch).await.unwrap();
        writer_no_compression.finish().await.unwrap();

        // Note: With our current data patterns and RLE compression, the compressed file
        // might actually be slightly larger due to compression metadata overhead.
        // This is expected and the test is mainly to verify the system works end-to-end.

        // Read back the compressed file and verify data integrity
        let fs = FsFixture::default();
        let file_scheduler = fs
            .scheduler
            .open_file(&path, &CachedFileSize::unknown())
            .await
            .unwrap();

        let file_reader = FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            &LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await
        .unwrap();

        // Verify metadata
        let metadata = file_reader.metadata();
        assert_eq!(metadata.major_version, 2);
        assert_eq!(metadata.minor_version, 1);

        let schema = file_reader.schema();
        assert_eq!(
            schema.metadata.get("compression_test"),
            Some(&"configured_compression".to_string())
        );

        // Verify the actual encodings used
        let column_metadatas = &metadata.column_metadatas;

        // Check customer_id column (index 0) - should use RLE due to our configuration
        assert!(!column_metadatas[0].pages.is_empty());
        let customer_id_encoding = describe_encoding(&column_metadatas[0].pages[0]);
        assert!(
            customer_id_encoding.contains("RLE") || customer_id_encoding.contains("Rle"),
            "customer_id column should use RLE encoding due to '*_id' pattern match, but got: {}",
            customer_id_encoding
        );

        // Check product_id column (index 1) - should use RLE due to our configuration
        assert!(!column_metadatas[1].pages.is_empty());
        let product_id_encoding = describe_encoding(&column_metadatas[1].pages[0]);
        assert!(
            product_id_encoding.contains("RLE") || product_id_encoding.contains("Rle"),
            "product_id column should use RLE encoding due to '*_id' pattern match, but got: {}",
            product_id_encoding
        );
    }

    #[tokio::test]
    async fn test_field_metadata_compression() {
        // Test that field metadata compression settings are respected
        let mut metadata = HashMap::new();
        metadata.insert(
            lance_encoding::constants::COMPRESSION_META_KEY.to_string(),
            "zstd".to_string(),
        );
        metadata.insert(
            lance_encoding::constants::COMPRESSION_LEVEL_META_KEY.to_string(),
            "6".to_string(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("text", DataType::Utf8, false).with_metadata(metadata.clone()),
            ArrowField::new("data", DataType::Int32, false).with_metadata(HashMap::from([(
                lance_encoding::constants::COMPRESSION_META_KEY.to_string(),
                "none".to_string(),
            )])),
        ]));

        let lance_schema = LanceSchema::try_from(arrow_schema.as_ref()).unwrap();

        // Create test data
        let id_array = Int32Array::from_iter_values(0..1000);
        let text_array = StringArray::from_iter_values(
            (0..1000).map(|i| format!("test string {} repeated text", i)),
        );
        let data_array = Int32Array::from_iter_values((0..1000).map(|i| i * 2));

        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(id_array),
                Arc::new(text_array),
                Arc::new(data_array),
            ],
        )
        .unwrap();

        let path = TempObjFile::default();
        let object_store = ObjectStore::local();

        // Create encoding strategy that will read from field metadata
        let params = CompressionParams::new();
        let encoding_strategy = lance_encoding::encoder::default_encoding_strategy_with_params(
            LanceFileVersion::V2_1,
            params,
        )
        .unwrap();

        let options = FileWriterOptions {
            encoding_strategy: Some(Arc::from(encoding_strategy)),
            format_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        };
        let mut writer = FileWriter::try_new(
            object_store.create(&path).await.unwrap(),
            lance_schema.clone(),
            options,
        )
        .unwrap();

        writer.write_batch(&batch).await.unwrap();
        writer.finish().await.unwrap();

        // Read back metadata
        let fs = FsFixture::default();
        let file_scheduler = fs
            .scheduler
            .open_file(&path, &CachedFileSize::unknown())
            .await
            .unwrap();
        let file_reader = FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            &LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await
        .unwrap();

        let column_metadatas = &file_reader.metadata().column_metadatas;

        // The text column (index 1) should use zstd compression based on metadata
        let text_encoding = describe_encoding(&column_metadatas[1].pages[0]);
        // For string columns, we expect Binary encoding with zstd compression
        assert!(
            text_encoding.contains("Zstd"),
            "text column should use zstd compression from field metadata, but got: {}",
            text_encoding
        );

        // The data column (index 2) should use no compression based on metadata
        let data_encoding = describe_encoding(&column_metadatas[2].pages[0]);
        // For Int32 columns with "none" compression, we expect Flat encoding without compression
        assert!(
            data_encoding.contains("Flat") && data_encoding.contains("compression: None"),
            "data column should use no compression from field metadata, but got: {}",
            data_encoding
        );
    }

    #[tokio::test]
    async fn test_field_metadata_rle_threshold() {
        // Test that RLE threshold from field metadata is respected
        let mut metadata = HashMap::new();
        metadata.insert(
            lance_encoding::constants::RLE_THRESHOLD_META_KEY.to_string(),
            "0.9".to_string(),
        );
        // Also set compression to ensure RLE is used
        metadata.insert(
            lance_encoding::constants::COMPRESSION_META_KEY.to_string(),
            "lz4".to_string(),
        );
        // Explicitly disable BSS to ensure RLE is tested
        metadata.insert(
            lance_encoding::constants::BSS_META_KEY.to_string(),
            "off".to_string(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("status", DataType::Int32, false).with_metadata(metadata),
        ]));

        let lance_schema = LanceSchema::try_from(arrow_schema.as_ref()).unwrap();

        // Create data with very high repetition (3 runs for 10000 values = 0.0003 ratio)
        let status_array = Int32Array::from_iter_values(
            std::iter::repeat_n(200, 8000)
                .chain(std::iter::repeat_n(404, 1500))
                .chain(std::iter::repeat_n(500, 500)),
        );

        let batch =
            RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(status_array)]).unwrap();

        let path = TempObjFile::default();
        let object_store = ObjectStore::local();

        // Create encoding strategy that will read from field metadata
        let params = CompressionParams::new();
        let encoding_strategy = lance_encoding::encoder::default_encoding_strategy_with_params(
            LanceFileVersion::V2_1,
            params,
        )
        .unwrap();

        let options = FileWriterOptions {
            encoding_strategy: Some(Arc::from(encoding_strategy)),
            format_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        };
        let mut writer = FileWriter::try_new(
            object_store.create(&path).await.unwrap(),
            lance_schema.clone(),
            options,
        )
        .unwrap();

        writer.write_batch(&batch).await.unwrap();
        writer.finish().await.unwrap();

        // Read back and check encoding
        let fs = FsFixture::default();
        let file_scheduler = fs
            .scheduler
            .open_file(&path, &CachedFileSize::unknown())
            .await
            .unwrap();
        let file_reader = FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            &LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await
        .unwrap();

        let column_metadatas = &file_reader.metadata().column_metadatas;
        let status_encoding = describe_encoding(&column_metadatas[0].pages[0]);
        assert!(
            status_encoding.contains("RLE") || status_encoding.contains("Rle"),
            "status column should use RLE encoding due to metadata threshold, but got: {}",
            status_encoding
        );
    }

    #[tokio::test]
    async fn test_large_page_split_on_read() {
        use arrow_array::Array;
        use futures::TryStreamExt;
        use lance_encoding::decoder::FilterExpression;
        use lance_io::ReadBatchParams;

        // Test that large pages written with relaxed limits can be split during read

        let arrow_field = ArrowField::new("data", DataType::Binary, false);
        let arrow_schema = ArrowSchema::new(vec![arrow_field]);
        let lance_schema = LanceSchema::try_from(&arrow_schema).unwrap();

        // Create a large binary value (40MB) to trigger large page creation
        let large_value = vec![42u8; 40 * 1024 * 1024];
        let array = arrow_array::BinaryArray::from(vec![
            Some(large_value.as_slice()),
            Some(b"small value"),
        ]);
        let batch = RecordBatch::try_new(Arc::new(arrow_schema), vec![Arc::new(array)]).unwrap();

        // Write with relaxed page size limit (128MB)
        let options = FileWriterOptions {
            max_page_bytes: Some(128 * 1024 * 1024),
            format_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        };

        let fs = FsFixture::default();
        let path = fs.tmp_path;

        let mut writer = FileWriter::try_new(
            fs.object_store.create(&path).await.unwrap(),
            lance_schema.clone(),
            options,
        )
        .unwrap();

        writer.write_batch(&batch).await.unwrap();
        let write_summary = writer.finish().await.unwrap();
        assert_eq!(write_summary.num_rows, 2);
        assert_eq!(
            write_summary.size_bytes,
            fs.object_store.size(&path).await.unwrap()
        );

        // Read back with split configuration
        let file_scheduler = fs
            .scheduler
            .open_file(&path, &CachedFileSize::unknown())
            .await
            .unwrap();

        // Configure reader to split pages larger than 10MB into chunks
        let reader_options = FileReaderOptions {
            read_chunk_size: 10 * 1024 * 1024, // 10MB chunks
            ..Default::default()
        };

        let file_reader = FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            &LanceCache::no_cache(),
            reader_options,
        )
        .await
        .unwrap();

        // Read the data back
        let stream = file_reader
            .read_stream(
                ReadBatchParams::RangeFull,
                1024,
                10, // batch_readahead
                FilterExpression::no_filter(),
            )
            .await
            .unwrap();

        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        assert_eq!(batches.len(), 1);

        // Verify the data is correctly read despite splitting
        let read_array = batches[0].column(0);
        let read_binary = read_array
            .as_any()
            .downcast_ref::<arrow_array::BinaryArray>()
            .unwrap();

        assert_eq!(read_binary.len(), 2);
        assert_eq!(read_binary.value(0).len(), 40 * 1024 * 1024);
        assert_eq!(read_binary.value(1), b"small value");

        // Verify first value matches what we wrote
        assert!(read_binary.value(0).iter().all(|&b| b == 42u8));
    }

    fn spill_config() -> (TempObjFile, Arc<ObjectStore>) {
        let spill_path = TempObjFile::default();
        (spill_path, Arc::new(ObjectStore::local()))
    }

    fn make_batches(num_batches: i32, num_cols: usize, rows_per_batch: i32) -> Vec<RecordBatch> {
        let fields: Vec<_> = (0..num_cols)
            .map(|c| ArrowField::new(format!("c{c}"), DataType::Int32, false))
            .collect();
        let schema = Arc::new(ArrowSchema::new(fields));
        (0..num_batches)
            .map(|i| {
                let cols: Vec<Arc<dyn arrow_array::Array>> = (0..num_cols)
                    .map(|c| {
                        let start = (i * rows_per_batch + c as i32) * 100;
                        Arc::new(Int32Array::from_iter_values(start..start + rows_per_batch))
                            as Arc<dyn arrow_array::Array>
                    })
                    .collect();
                RecordBatch::try_new(schema.clone(), cols).unwrap()
            })
            .collect()
    }

    async fn write_and_read_batches(
        batches: &[RecordBatch],
        spill: Option<(Arc<ObjectStore>, object_store::path::Path)>,
    ) -> Vec<RecordBatch> {
        let fs = FsFixture::default();
        let lance_schema = LanceSchema::try_from(batches[0].schema().as_ref()).unwrap();
        let writer = fs.object_store.create(&fs.tmp_path).await.unwrap();
        let mut file_writer =
            FileWriter::try_new(writer, lance_schema, FileWriterOptions::default()).unwrap();
        if let Some((store, path)) = spill {
            file_writer = file_writer.with_page_metadata_spill(store, path);
        }
        for batch in batches {
            file_writer.write_batch(batch).await.unwrap();
        }
        file_writer.add_schema_metadata("foo", "bar");
        file_writer.finish().await.unwrap();

        crate::testing::read_lance_file(
            &fs,
            Arc::<DecoderPlugins>::default(),
            lance_encoding::decoder::FilterExpression::no_filter(),
        )
        .await
    }

    #[rstest::rstest]
    #[case::multi_col(20, 2, 100)]
    #[case::many_batches(50, 2, 100)]
    #[tokio::test]
    async fn test_page_metadata_spill_roundtrip(
        #[case] num_batches: i32,
        #[case] num_cols: usize,
        #[case] rows_per_batch: i32,
    ) {
        let batches = make_batches(num_batches, num_cols, rows_per_batch);
        let baseline = write_and_read_batches(&batches, None).await;
        let (spill_path, spill_store) = spill_config();
        let spilled =
            write_and_read_batches(&batches, Some((spill_store, spill_path.as_ref().clone())))
                .await;
        assert_eq!(baseline, spilled);
    }

    #[tokio::test]
    async fn test_page_metadata_spill_many_columns() {
        // Many columns forces small per-column buffer limits, exercising mid-write flushing.
        let batches = make_batches(10, 500, 100);
        let baseline = write_and_read_batches(&batches, None).await;
        let (spill_path, spill_store) = spill_config();
        let spilled =
            write_and_read_batches(&batches, Some((spill_store, spill_path.as_ref().clone())))
                .await;
        assert_eq!(baseline, spilled);
    }
}
