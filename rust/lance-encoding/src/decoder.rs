// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities and traits for decoding data
//!
//! There are two types of decoders, logical decoders and physical decoders.
//! In addition, decoding data is broken into two steps, scheduling the I/O
//! and decoding the returned data.
//!
//! # Physical vs. Logical Decoding
//!
//! The physical traits are [`self::PhysicalPageScheduler`] and
//! [`self::PhysicalPageDecoder`].  These are lower level encodings.  They
//! have a few advantages:
//!
//!  * They do not need to decode into an Arrow array and so they don't need
//!    to be enveloped into the Arrow filesystem (e.g. Arrow doesn't have a
//!    bit-packed type.  We can use variable-length binary but that is kind
//!    of overkill)
//!  * They can decode into existing storage.  This can allow for "page
//!    bridging".  If we are trying to decode into a batch of 1024 rows and
//!    the rows 0..1024 are spread across two pages then we can avoid a memory
//!    copy by allocating once and decoding each page into the outer allocation.
//!
//! However, there are some limitations too:
//!
//!  * They are constrained to a single column
//!  * The API is more complex
//!
//! The logical traits are [`self::LogicalPageScheduler`] and [`self::LogicalPageDecoder`]
//! These are designed to map from Arrow fields into one or more columns of Lance
//! data.  They do not decode into existing buffers and instead they return an Arrow Array.
//!
//! Encodings are typically nested into each other to form a tree.  The top of the tree is
//! the user requested schema.  Each field in that schema is assigned to one top-level logical
//! encoding.  That encoding can then contain other logical encodings or physical encodings.
//! Physical encodings can also contain other physical encodings.
//!
//! So, for example, a single field in the Arrow schema might have the type List<UInt32>
//!
//! The encoding tree could then be:
//!
//! root: List (logical encoding)
//!  - indices: Primitive (logical encoding)
//!    - column: Basic (physical encoding)
//!      - validity: Bitmap (physical encoding)
//!      - values: RLE (physical encoding)
//!        - runs: Value (physical encoding)
//!        - values: Value (physical encoding)
//!  - items: Primitive (logical encoding)
//!    - column: Basic (physical encoding)
//!      - values: Value (phsyical encoding)
//!
//! Note that, in this example, root.items.column does not have a validity because there were
//! no nulls in the page.
//!
//! Note that the decoding API is not symmetric with the encoding API.  The encoding API contains
//! buffer encoders, array encoders, and field encoders.  Physical decoders correspond to both
//! buffer encoders and array encoders.  Logical decoders correspond to both array encoders and
//! field encoders.
//!
//! ## Multiple buffers or multiple columns?
//!
//! Note that there are many different ways we can write encodings.  For example, we might
//! store primitive fields in a single column with two buffers (one for validity and one for
//! values)
//!
//! On the other hand, we could also store a primitive field as two different columns.  One
//! that yields a non-nullable boolean array and one that yields a non-nullable array of items.
//! Then we could combine these two arrays into a single array where the boolean array is the
//! bitmap.  There are a few subtle differences between the approaches:
//!
//! * Storing things as multiple buffers within the same column is generally more efficient and
//!   easier to schedule.  For example, in-batch coalescing is very easy but can only be done
//!   on data that is in the same page.
//! * When things are stored in multiple columns you have to worry about their pages not being
//!   in sync.  In our previous validity / values example this means we might have to do some
//!   memory copies to get the validity array and values arrays to be the same length as
//!   decode.
//! * When things are stored in a single column, projection is impossible.  For example, if we
//!   tried to store all the struct fields in a single column with lots of buffers then we wouldn't
//!   be able to read back individual fields of the struct.
//! * When things are stored in a single column they must have the same length.  This is an issue
//!   for list fields.  The items column does not usually have the same length as the parent list
//!   column (in most cases it is much longer).
//!
//! The fixed size list decoding is an interesting example because it is actually both a physical
//! encoding and a logical encoding.  A fixed size list of a physical encoding is, itself, a physical
//! encoding (e.g. a fixed size list of doubles).  However, a fixed size list of a logical encoding
//! is a logical encoding (e.g. a fixed size list of structs).
//!
//! # The scheduling loop
//!
//! Reading a Lance file involves both scheduling and decoding.  Its generally expected that these
//! will run as two separate threads.
//!
//! ```text
//!
//!                                    I/O PARALLELISM
//!                       Issues
//!                       Requests   ┌─────────────────┐
//!                                  │                 │        Wait for
//!                       ┌──────────►   I/O Service   ├─────►  Enough I/O ◄─┐
//!                       │          │                 │        For batch    │
//!                       │          └─────────────────┘             │3      │
//!                       │                                          │       │
//!                       │                                          │       │2
//! ┌─────────────────────┴─┐                              ┌─────────▼───────┴┐
//! │                       │                              │                  │Poll
//! │       Batch Decode    │ Decode tasks sent via channel│   Batch Decode   │1
//! │       Scheduler       ├─────────────────────────────►│   Stream         ◄─────
//! │                       │                              │                  │
//! └─────▲─────────────┬───┘                              └─────────┬────────┘
//!       │             │                                            │4
//!       │             │                                            │
//!       └─────────────┘                                   ┌────────┴────────┐
//!  Caller of schedule_range                Buffer polling │                 │
//!  will be scheduler thread                to achieve CPU │ Decode Batch    ├────►
//!  and schedule one decode                 parallelism    │ Task            │
//!  task (and all needed I/O)               (thread per    │                 │
//!  per logical page                         batch)        └─────────────────┘
//! ```
//!
//! The scheduling thread will work through the file from the
//! start to the end as quickly as possible.  Data is scheduled one page at a time in a row-major
//! fashion.  For example, imagine we have a file with the following page structure:
//!
//! ```text
//! Score (Float32)     | C0P0 |
//! Id (16-byte UUID)   | C1P0 | C1P1 | C1P2 | C1P3 |
//! Vector (4096 bytes) | C2P0 | C2P1 | C2P2 | C2P3 | .. | C2P1024 |
//! ```
//!
//! This would be quite common as each of these pages has the same number of bytes.  Let's pretend
//! each page is 1MiB and so there are 256Ki rows of data.  Each page of `Score` has 256Ki rows.
//! Each page of `Id` has 64Ki rows.  Each page of `Vector` has 256 rows.  The scheduler would then
//! schedule in the following order:
//!
//! C0 P0
//! C1 P0
//! C2 P0
//! C2 P1
//! ... (254 pages omitted)
//! C2 P255
//! C1 P1
//! C2 P256
//! ... (254 pages omitted)
//! C2 P511
//! C1 P2
//! C2 P512
//! ... (254 pages omitted)
//! C2 P767
//! C1 P3
//! C2 P768
//! ... (254 pages omitted)
//! C2 P1024
//!
//! This is the ideal scheduling order because it means we can decode complete rows as quickly as possible.
//! Note that the scheduler thread does not need to wait for I/O to happen at any point.  As soon as it starts
//! it will start scheduling one page of I/O after another until it has scheduled the entire file's worth of
//! I/O.  This is slightly different than other file readers which have "row group parallelism" and will
//! typically only schedule X row groups worth of reads at a time.
//!
//! In the near future there will be a backpressure mechanism and so it may need to stop/pause if the compute
//! falls behind.
//!
//! ## Indirect I/O
//!
//! Regrettably, there are times where we cannot know exactly what data we need until we have partially decoded
//! the file.  This happens when we have variable sized list data.  In that case the scheduling task for that
//! page will only schedule the first part of the read (loading the list offsets).  It will then immediately
//! spawn a new tokio task to wait for that I/O and decode the list offsets.  That follow-up task is not part
//! of the scheduling loop or the decode loop.  It is a free task.  Once the list offsets are decoded we submit
//! a follow-up I/O task.  This task is scheduled at a high priority because the decoder is going to need it soon.
//!
//! # The decode loop
//!
//! As soon as the scheduler starts we can start decoding.  Each time we schedule a page we
//! push a decoder for that page's data into a channel.  The decode loop
//! ([`BatchDecodeStream`]) reads from that channel.  Each time it receives a decoder it
//! waits until the decoder has all of its data.  Then it grabs the next decoder.  Once it has
//! enough loaded decoders to complete a batch worth of rows it will spawn a "decode batch task".
//!
//! These batch decode tasks perform the actual CPU work of decoding the loaded data into Arrow
//! arrays.  This may involve signifciant CPU processing like decompression or arithmetic in order
//! to restore the data to its correct in-memory representation.
//!
//! ## Batch size
//!
//! The `BatchDecodeStream` is configured with a batch size.  This does not need to have any
//! relation to the page size(s) used to write the data.  This keeps our compute work completely
//! independent of our I/O work.  We suggest using small batch sizes:
//!
//!  * Batches should fit in CPU cache (at least L3)
//!  * More batches means more opportunity for parallelism
//!  * The "batch overhead" is very small in Lance compared to other formats because it has no
//!    relation to the way the data is stored.

use std::collections::VecDeque;
use std::{ops::Range, sync::Arc};

use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::{Bytes, BytesMut};
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};
use log::trace;
use snafu::{location, Location};
use tokio::sync::mpsc::{self, unbounded_channel};

use lance_core::{Error, Result};
use tracing::instrument;

use crate::encoder::EncodedBatch;
use crate::encodings::logical::binary::BinaryPageScheduler;
use crate::encodings::logical::fixed_size_list::FslPageScheduler;
use crate::encodings::logical::list::ListPageScheduler;
use crate::encodings::logical::primitive::PrimitivePageScheduler;
use crate::encodings::logical::r#struct::{SimpleStructDecoder, SimpleStructScheduler};
use crate::encodings::physical::{ColumnBuffers, FileBuffers};
use crate::format::pb;
use crate::{BufferScheduler, EncodingsIo};

/// Metadata describing a page in a file
///
/// This is typically created by reading the metadata section of a Lance file
#[derive(Debug)]
pub struct PageInfo {
    /// The number of rows in the page
    pub num_rows: u32,
    /// The encoding that explains the buffers in the page
    pub encoding: pb::ArrayEncoding,
    /// The offsets of the buffers in the file
    pub buffer_offsets: Arc<Vec<u64>>,
}

/// Metadata describing a column in a file
///
/// This is typically created by reading the metadata section of a Lance file
#[derive(Debug)]
pub struct ColumnInfo {
    /// The index of the column in the file
    pub index: u32,
    /// The metadata for each page in the column
    pub page_infos: Vec<Arc<PageInfo>>,
    /// File positions of the column-level buffers
    pub buffer_offsets: Vec<u64>,
}

impl ColumnInfo {
    /// Create a new instance
    pub fn new(index: u32, page_infos: Vec<Arc<PageInfo>>, buffer_offsets: Vec<u64>) -> Self {
        Self {
            index,
            page_infos,
            buffer_offsets,
        }
    }
}

/// The scheduler for decoding batches
///
/// Lance decoding is done in two steps, scheduling, and decoding.  The
/// scheduling tends to be lightweight and should quickly figure what data
/// is needed from the disk issue the appropriate I/O requests.  A decode task is
/// created to eventually decode the data (once it is loaded) and scheduling
/// moves on to scheduling the next page.
///
/// Meanwhile, it's expected that a decode stream will be setup to run at the
/// same time.  Decode tasks take the data that is loaded and turn it into
/// Arrow arrays.
///
/// This approach allows us to keep our I/O parallelism and CPU parallelism
/// completely separate since those are often two very different values.
///
/// Backpressure should be achieved via the I/O service.  Requests that are
/// issued will pile up if the decode stream is not polling quickly enough.
/// The [`crate::EncodingsIo::submit_request`] function should return a pending
/// future once there are too many I/O requests in flight.
///
/// TODO: Implement backpressure
pub struct DecodeBatchScheduler {
    pub root_scheduler: SimpleStructScheduler,
}

impl DecodeBatchScheduler {
    fn is_primitive(data_type: &DataType) -> bool {
        if data_type.is_primitive() {
            true
        } else {
            match data_type {
                // DataType::is_primitive doesn't consider these primitive but we do
                DataType::Boolean | DataType::Null | DataType::FixedSizeBinary(_) => true,
                DataType::FixedSizeList(inner, _) => Self::is_primitive(inner.data_type()),
                _ => false,
            }
        }
    }

    fn create_primitive_scheduler(
        data_type: &DataType,
        column: &ColumnInfo,
        buffers: FileBuffers,
    ) -> Vec<Arc<dyn LogicalPageScheduler>> {
        // Primitive fields map to a single column
        let column_buffers = ColumnBuffers {
            file_buffers: buffers,
            positions: &column.buffer_offsets,
        };
        column
            .page_infos
            .iter()
            .cloned()
            .map(|page_info| {
                Arc::new(PrimitivePageScheduler::new(
                    data_type.clone(),
                    page_info,
                    column_buffers,
                )) as Arc<dyn LogicalPageScheduler>
            })
            .collect::<Vec<_>>()
    }

    fn check_simple_struct(column_info: &ColumnInfo) -> Result<()> {
        if !column_info.page_infos.len() == 1 {
            return Err(Error::InvalidInput { source: format!("Due to schema we expected a struct column but we received a column with {} pages and right now we only support struct columns with 1 page", column_info.page_infos.len()).into(), location: location!() });
        }
        let encoding = &column_info.page_infos[0].encoding;
        match encoding.array_encoding.as_ref().unwrap() {
            pb::array_encoding::ArrayEncoding::Struct(_) => Ok(()),
            _ => Err(Error::InvalidInput { source: format!("Expected a struct encoding because we have a struct field in the schema but got the encoding {:?}", encoding).into(), location: location!() }),
        }
    }
    // This function is where the all important mapping from Arrow schema
    // to decoders happens.  Note that the decoders can only be figured out
    // using both the schema AND the column metadata.  In theory, one could
    // infer the decoder / column type using only the column metadata.  However,
    // field nullability would be missing / incorrect and its also not as easy
    // as it sounds since pages can have different encodings and those encodings
    // often have various layers.  Also, sometimes the inference is just impossible.
    // For example, both Timestamp, Float64, Int64, and UInt64 will be encoded
    // as 8-byte value encoding.  The only way to know the data type is to look
    // at the schema.
    //
    // We also can't just guess the encoding based on the schema.  This is because
    // there may be multiple different ways to encode a field and it may even
    // change on a page-by-page basis.
    //
    // For example, if a field is a struct field then we expect a header
    // column that could have one of a few different encodings.
    //
    // This could be encoded with "simple struct" and an empty header column
    // followed by the shredded child columns.  It could be encoded as a nullable
    // struct where the nulls are in a dense bitmap.  It could even be encoded
    // as a packed (row-major) struct where there is only a single column containing
    // all of the data!
    //
    // TODO: Still lots of research to do here in different ways that
    // we can map schemas to buffers.
    //
    // Example: repetition levels - the validity bitmaps for nested
    // fields are fatter (more than one bit per row) and contain
    // validity information about parent fields (e.g. is this a
    // struct-struct-null or struct-null-null or null-null-null?)
    //
    // Examples: sentinel-shredding - instead of creating a wider
    // validity bitmap we assign more sentinels to each column.  So
    // if the values of an int32 array have a max of 1000 then we can
    // use 1001 to mean null int32 and 1002 to mean null parent.
    //
    // Examples: Sparse structs - the struct column has a validity
    // bitmap that must be read if you plan on reading the struct
    // or any nested field.  However, this could be a compressed
    // bitmap stored in metadata.  A perk for this approach is that
    // the child fields can then have a smaller size than the parent
    // field.  E.g. if a struct is 1000 rows and 900 of them are
    // null then there is one validity bitmap of length 1000 and
    // 100 rows of each of the children.
    //
    // TODO: In the future, this will need to be more flexible if
    // we want to allow custom encodings.  E.g. if the field's encoding
    // is not an encoding we expect then we should delegate to a plugin.
    fn create_field_scheduler<'a>(
        data_type: &DataType,
        column_infos: &mut impl Iterator<Item = &'a ColumnInfo>,
        buffers: FileBuffers,
    ) -> Vec<Arc<dyn LogicalPageScheduler>> {
        if Self::is_primitive(data_type) {
            let primitive_col = column_infos.next().unwrap();
            return Self::create_primitive_scheduler(data_type, primitive_col, buffers);
        }
        match data_type {
            DataType::FixedSizeList(inner, dimension) => {
                // A fixed size list column could either be a physical or a logical decoder
                // depending on the child data type.
                if Self::is_primitive(inner.data_type()) {
                    let primitive_col = column_infos.next().unwrap();
                    Self::create_primitive_scheduler(data_type, primitive_col, buffers)
                } else {
                    let inner_schedulers =
                        Self::create_field_scheduler(inner.data_type(), column_infos, buffers);
                    inner_schedulers
                        .into_iter()
                        .map(|inner| {
                            Arc::new(FslPageScheduler::new(inner, *dimension as u32))
                                as Arc<dyn LogicalPageScheduler>
                        })
                        .collect::<Vec<_>>()
                }
            }
            DataType::List(items_field) | DataType::LargeList(items_field) => {
                let offsets_column = column_infos.next().unwrap();
                let offsets_column_buffers = ColumnBuffers {
                    file_buffers: buffers,
                    positions: &offsets_column.buffer_offsets,
                };
                let items =
                    Self::create_field_scheduler(items_field.data_type(), column_infos, buffers);
                let mut items = items.into_iter().peekable();
                let mut cur_items_page_offset = 0;
                offsets_column
                    .page_infos
                    .iter()
                    .map(|offsets_page| {
                        if let Some(pb::array_encoding::ArrayEncoding::List(list_encoding)) =
                            &offsets_page.encoding.array_encoding
                        {
                            let inner = Arc::new(PrimitivePageScheduler::new(
                                DataType::UInt64,
                                offsets_page.clone(),
                                offsets_column_buffers,
                            ))
                                as Arc<dyn LogicalPageScheduler>;
                            let mut items_schedulers = Vec::new();
                            let mut num_items = list_encoding.num_items;
                            let first_items_page_offset = cur_items_page_offset;
                            while num_items > 0 {
                                let next_items_page = items.peek().unwrap().clone();
                                if next_items_page.num_rows() as u64 <= num_items {
                                    num_items -= next_items_page.num_rows() as u64;
                                    cur_items_page_offset = 0;
                                    items.next().unwrap();
                                } else {
                                    cur_items_page_offset += num_items as u32;
                                    num_items = 0;
                                }
                                items_schedulers.push(next_items_page);
                            }
                            let offset_type = if matches!(data_type, DataType::List(_)) {
                                DataType::Int32
                            } else {
                                DataType::Int64
                            };
                            Arc::new(ListPageScheduler::new(
                                inner,
                                items_schedulers,
                                items_field.data_type().clone(),
                                offset_type,
                                list_encoding.null_offset_adjustment,
                                first_items_page_offset,
                            )) as Arc<dyn LogicalPageScheduler>
                        } else {
                            // TODO: Should probably return Err here
                            panic!("Expected a list column");
                        }
                    })
                    .collect::<Vec<_>>()
            }
            DataType::Utf8 | DataType::Binary | DataType::LargeBinary | DataType::LargeUtf8 => {
                let list_type = if matches!(data_type, DataType::Utf8 | DataType::Binary) {
                    DataType::List(Arc::new(Field::new("item", DataType::UInt8, true)))
                } else {
                    DataType::LargeList(Arc::new(Field::new("item", DataType::UInt8, true)))
                };
                let list_decoders = Self::create_field_scheduler(&list_type, column_infos, buffers);
                list_decoders
                    .into_iter()
                    .map(|list_decoder| {
                        Arc::new(BinaryPageScheduler::new(list_decoder, data_type.clone()))
                            as Arc<dyn LogicalPageScheduler>
                    })
                    .collect::<Vec<_>>()
            }
            DataType::Struct(fields) => {
                let column_info = column_infos.next().unwrap();
                Self::check_simple_struct(column_info).unwrap();
                let child_schedulers = fields
                    .iter()
                    .map(|field| {
                        Self::create_field_scheduler(field.data_type(), column_infos, buffers)
                    })
                    .collect::<Vec<_>>();
                // For now, we don't record nullability for structs.  As a result, there is always
                // only one "page" of struct data.  In the future, this will change.  A null-aware
                // struct scheduler will need to first calculate how many rows are in the struct page
                // and then find the child pages that overlap.  This should be doable.
                vec![Arc::new(SimpleStructScheduler::new(
                    child_schedulers,
                    fields.clone(),
                ))]
            }
            // Still need support for string / binary / dictionary / RLE
            _ => todo!("Decoder support for data type {:?}", data_type),
        }
    }

    /// Creates a new decode scheduler with the expected schema and the column
    /// metadata of the file.
    pub fn new<'a>(
        schema: &'a Schema,
        column_infos: impl IntoIterator<Item = &'a ColumnInfo>,
        file_buffer_positions: &'a Vec<u64>,
    ) -> Self {
        let mut col_info_iter = column_infos.into_iter();
        let buffers = FileBuffers {
            positions: file_buffer_positions,
        };
        let field_schedulers = schema
            .fields
            .iter()
            .map(|field| {
                Self::create_field_scheduler(field.data_type(), &mut col_info_iter, buffers)
            })
            .collect::<Vec<_>>();
        let root_scheduler =
            SimpleStructScheduler::new_root(field_schedulers, schema.fields.clone());
        Self { root_scheduler }
    }

    /// Schedules the load of a range of rows
    ///
    /// # Arguments
    ///
    /// * `range` - The range of rows to load
    /// * `sink` - A channel to send the decode tasks
    /// * `scheduler` An I/O scheduler to issue I/O requests
    #[instrument(skip_all)]
    pub async fn schedule_range(
        &mut self,
        range: Range<u64>,
        sink: mpsc::UnboundedSender<DecoderMessage>,
        scheduler: Arc<dyn EncodingsIo>,
    ) -> Result<()> {
        let rows_to_read = range.end - range.start;
        trace!("Scheduling range {:?} ({} rows)", range, rows_to_read);

        let mut context = SchedulerContext::new(sink, scheduler);
        self.root_scheduler
            .schedule_ranges_u64(&[range.clone()], &mut context, range.start)?;

        trace!("Finished scheduling of range {:?}", range);
        Ok(())
    }

    /// Schedules the load of selected rows
    ///
    /// # Arguments
    ///
    /// * `indices` - The row indices to load (these must be in ascending order!)
    /// * `sink` - A channel to send the decode tasks
    /// * `scheduler` An I/O scheduler to issue I/O requests
    pub async fn schedule_take(
        &mut self,
        indices: &[u64],
        sink: mpsc::UnboundedSender<DecoderMessage>,
        scheduler: Arc<dyn EncodingsIo>,
    ) -> Result<()> {
        debug_assert!(indices.windows(2).all(|w| w[0] < w[1]));
        if indices.is_empty() {
            return Ok(());
        }
        trace!(
            "Scheduling take of {} rows [{}]",
            indices.len(),
            if indices.len() == 1 {
                indices[0].to_string()
            } else {
                format!("{}, ..., {}", indices[0], indices[indices.len() - 1])
            }
        );
        if indices.is_empty() {
            return Ok(());
        }
        let mut context = SchedulerContext::new(sink, scheduler);

        self.root_scheduler
            .schedule_take_u64(indices, &mut context, indices[0])?;
        trace!("Finished scheduling take of {} rows", indices.len());
        Ok(())
    }
}

pub struct ReadBatchTask {
    pub task: BoxFuture<'static, Result<RecordBatch>>,
    pub num_rows: u32,
}

/// A stream that takes scheduled jobs and generates decode tasks from them.
pub struct BatchDecodeStream {
    context: DecoderContext,
    root_decoder: SimpleStructDecoder,
    rows_remaining: u64,
    rows_per_batch: u32,
    rows_scheduled: u64,
    rows_drained: u64,
}

impl BatchDecodeStream {
    /// Create a new instance of a batch decode stream
    ///
    /// # Arguments
    ///
    /// * `scheduled` - an incoming stream of decode tasks from a
    ///   [`crate::decode::DecodeBatchScheduler`]
    /// * `schema` - the scheam of the data to create
    /// * `rows_per_batch` the number of rows to create before making a batch
    /// * `num_rows` the total number of rows scheduled
    /// * `num_columns` the total number of columns in the file
    pub fn new(
        scheduled: mpsc::UnboundedReceiver<DecoderMessage>,
        rows_per_batch: u32,
        num_rows: u64,
        root_decoder: SimpleStructDecoder,
    ) -> Self {
        Self {
            context: DecoderContext::new(scheduled),
            root_decoder,
            rows_remaining: num_rows,
            rows_per_batch,
            rows_scheduled: 0,
            rows_drained: 0,
        }
    }

    fn accept_decoder(&mut self, decoder: DecoderReady) -> Result<()> {
        debug_assert!(!decoder.path.is_empty());
        self.root_decoder.accept_child(decoder)
    }

    async fn wait_for_scheduled(&mut self, scheduled_need: u64) -> Result<()> {
        while self.rows_scheduled < scheduled_need {
            let next_message = self.context.source.recv().await;
            match next_message {
                Some(DecoderMessage::ScanLine(rows_scheduled)) => {
                    self.rows_scheduled = rows_scheduled;
                }
                Some(DecoderMessage::Decoder(decoder)) => {
                    self.accept_decoder(decoder)?;
                }
                None => {
                    return Err(Error::Internal {
                        message:
                            "The scheduler finished while the decoder was still waiting for input"
                                .to_string(),
                        location: location!(),
                    });
                }
            }
        }
        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn next_batch_task(&mut self) -> Result<Option<NextDecodeTask>> {
        trace!(
            "Draining batch task (rows_remaining={} rows_drained={} rows_scheduled={})",
            self.rows_remaining,
            self.rows_drained,
            self.rows_scheduled,
        );
        if self.rows_remaining == 0 {
            return Ok(None);
        }

        let to_take = self.rows_remaining.min(self.rows_per_batch as u64) as u32;
        self.rows_remaining -= to_take as u64;

        let scheduled_need =
            (self.rows_drained + to_take as u64).saturating_sub(self.rows_scheduled);
        if scheduled_need > 0 {
            let desired_scheduled = scheduled_need + self.rows_scheduled;
            trace!(
                "Draining from scheduler (desire at least {} scheduled rows)",
                desired_scheduled
            );
            self.wait_for_scheduled(desired_scheduled).await?;
        }

        let avail = self.root_decoder.avail_u64();
        trace!("Top level page has {} rows already available", avail);
        if avail < to_take as u64 {
            trace!(
                "Top level page waiting for an additional {} rows",
                to_take as u64 - avail
            );
            self.root_decoder.wait(to_take).await?;
        }
        let next_task = self.root_decoder.drain(to_take)?;
        self.rows_drained += to_take as u64;
        Ok(Some(next_task))
    }

    #[instrument(level = "debug", skip_all)]
    fn task_to_batch(task: NextDecodeTask) -> Result<RecordBatch> {
        let struct_arr = task.task.decode();
        match struct_arr {
            Ok(struct_arr) => Ok(RecordBatch::from(struct_arr.as_struct())),
            Err(e) => {
                let e = Error::Internal {
                    message: format!("Error decoding batch: {}", e),
                    location: location!(),
                };
                Err(e)
            }
        }
    }

    pub fn into_stream(self) -> BoxStream<'static, ReadBatchTask> {
        let stream = futures::stream::unfold(self, |mut slf| async move {
            let next_task = slf.next_batch_task().await;
            let next_task = next_task.transpose().map(|next_task| {
                let num_rows = next_task.as_ref().map(|t| t.num_rows).unwrap_or(0);
                let task = tokio::spawn(async move {
                    let next_task = next_task?;
                    Self::task_to_batch(next_task)
                });
                (task, num_rows)
            });
            next_task.map(|(task, num_rows)| {
                let task = task.map(|join_wrapper| join_wrapper.unwrap()).boxed();
                let next_task = ReadBatchTask { task, num_rows };
                (next_task, slf)
            })
        });
        stream.boxed()
    }
}

/// A decoder for single-column encodings of primitive data (this includes fixed size
/// lists of primitive data)
///
/// Physical decoders are able to decode into existing buffers for zero-copy operation.
///
/// Instances should be stateless and `Send` / `Sync`.  This is because multiple decode
/// tasks could reference the same page.  For example, imagine a page covers rows 0-2000
/// and the decoder stream has a batch size of 1024.  The decoder will be needed by both
/// the decode task for batch 0 and the decode task for batch 1.
///
/// See [`crate::decoder`] for more information
pub trait PhysicalPageDecoder: Send + Sync {
    /// Calculates and updates the capacity required to represent the requested data
    ///
    /// Capacity is stored as a tuple of (num_bytes: u64, is_needed: bool).  The `is_needed`
    /// portion only needs to be updated if the encoding has some concept of an "optional"
    /// buffer.
    ///
    /// The decoder should look at `rows_to_skip` and `num_rows` and then calculate how
    /// many bytes of data are needed.  It should then update the first part of the tuple.
    ///
    /// Note: Most encodings deal with a single buffer.  They may have multiple input buffers
    /// but they only have a single output buffer.  The current exception to this rule is the
    /// `basic` encoding which has an output "validity" buffer and an output "values" buffers.
    /// We may find there are other such exceptions.
    ///
    /// # Arguments
    ///
    /// * `rows_to_skip` - how many rows to skip (within the page) before decoding
    /// * `num_rows` - how many rows to decode
    /// * `buffers` - A mutable slice of "capacities" (as described above), one per buffer
    /// * `all_null` - A mutable bool, set to true if a decoder determines all values are null
    fn update_capacity(
        &self,
        rows_to_skip: u32,
        num_rows: u32,
        buffers: &mut [(u64, bool)],
        all_null: &mut bool,
    );
    /// Decodes the data into the requested buffers.
    ///
    /// You can assume that the capacity will have already been configured on the `BytesMut`
    /// according to the capacity calculated in [`PhysicalPageDecoder::update_capacity`]
    ///
    /// # Arguments
    ///
    /// * `rows_to_skip` - how many rows to skip (within the page) before decoding
    /// * `num_rows` - how many rows to decode
    /// * `dest_buffers` - the output buffers to decode into
    fn decode_into(&self, rows_to_skip: u32, num_rows: u32, dest_buffers: &mut [BytesMut]);
    fn num_buffers(&self) -> u32;
}

/// A scheduler for single-column encodings of primitive data
///
/// The scheduler is responsible for calculating what I/O is needed for the requested rows
///
/// Instances should be stateless and `Send` and `Sync`.  This is because instances can
/// be shared in follow-up I/O tasks.
///
/// See [`crate::decoder`] for more information
pub trait PhysicalPageScheduler: Send + Sync + std::fmt::Debug {
    /// Schedules a batch of I/O to load the data needed for the requested ranges
    ///
    /// Returns a future that will yield a decoder once the data has been loaded
    ///
    /// # Arguments
    ///
    /// * `range` - the range of row offsets (relative to start of page) requested
    ///             these must be ordered and must not overlap
    /// * `scheduler` - a scheduler to submit the I/O request to
    /// * `top_level_row` - the row offset of the top level field currently being
    ///   scheduled.  This can be used to assign priority to I/O requests
    fn schedule_ranges(
        &self,
        ranges: &[Range<u32>],
        scheduler: &dyn EncodingsIo,
        top_level_row: u64,
    ) -> BoxFuture<'static, Result<Box<dyn PhysicalPageDecoder>>>;
}

struct FixedPriorityIo {
    io: Arc<dyn EncodingsIo>,
    priority: u64,
}

impl EncodingsIo for FixedPriorityIo {
    fn submit_request(
        &self,
        range: Vec<Range<u64>>,
        _priority: u64,
    ) -> BoxFuture<'static, Result<Vec<Bytes>>> {
        self.io.submit_request(range, self.priority)
    }
}

/// Contains the context for a scheduler
pub struct SchedulerContext {
    /// The sink that sends decodeable tasks to the decode stage
    pub(crate) sink: mpsc::UnboundedSender<DecoderMessage>,
    recv: Option<mpsc::UnboundedReceiver<DecoderMessage>>,
    io: Arc<dyn EncodingsIo>,
    name: String,
    path: Vec<u32>,
    path_names: Vec<String>,
}

pub struct ScopedSchedulerContext<'a> {
    pub context: &'a mut SchedulerContext,
}

impl<'a> ScopedSchedulerContext<'a> {
    pub fn pop(self) -> &'a mut SchedulerContext {
        self.context.pop();
        self.context
    }
}

impl SchedulerContext {
    pub fn new(sink: mpsc::UnboundedSender<DecoderMessage>, io: Arc<dyn EncodingsIo>) -> Self {
        Self {
            sink,
            io,
            recv: None,
            name: "".to_string(),
            path: Vec::new(),
            path_names: Vec::new(),
        }
    }

    pub fn io(&self) -> &dyn EncodingsIo {
        self.io.as_ref()
    }

    pub fn push(&mut self, name: &str, index: u32) -> ScopedSchedulerContext {
        self.path.push(index);
        self.path_names.push(name.to_string());
        ScopedSchedulerContext { context: self }
    }

    pub fn pop(&mut self) {
        self.path.pop();
        self.path_names.pop();
    }

    pub fn path_name(&self) -> String {
        let path = self.path_names.join("/");
        if self.recv.is_some() {
            format!("TEMP({}){}", self.name, path)
        } else {
            format!("ROOT{}", path)
        }
    }

    pub fn emit(&mut self, decoder: Box<dyn LogicalPageDecoder>) {
        trace!(
            "Scheduling decoder of type {:?} for {:?}",
            decoder.data_type(),
            self.path,
        );
        self.sink
            .send(DecoderMessage::Decoder(DecoderReady {
                decoder,
                path: VecDeque::from_iter(self.path.iter().copied()),
            }))
            .unwrap();
    }

    // Temporary might not be the best name for this.  We create a new context that
    // shared the same I/O scheduler and has a different name.  This is used in two
    // situations.
    //
    // 1. When we need to create a new indirect I/O phase.
    // 2. When we need to wrap a set of decoders and so we want to intercept them
    //    before they are emitted.
    pub fn temporary(&self, fixed_priority: Option<u64>) -> Self {
        let inner_io = if let Some(priority) = fixed_priority {
            Arc::new(FixedPriorityIo {
                io: self.io.clone(),
                priority,
            }) as Arc<dyn EncodingsIo>
        } else {
            self.io.clone()
        };
        let (tx, rx) = unbounded_channel();
        let mut name = self.name.clone();
        name.push_str(&self.path_names.join("/"));
        Self {
            sink: tx,
            io: inner_io,
            recv: Some(rx),
            name,
            path: Vec::new(),
            path_names: Vec::new(),
        }
    }

    // Consumes the temporary context returning the decoder messages
    //
    // Used when the temporary context is used to create a new indirect I/O phase
    // where all the messages need to be replayed
    pub fn into_messages(self) -> Vec<DecoderMessage> {
        let mut recv = self
            .recv
            .expect("Call to `finish` on a non-temporary scheduler context");
        let mut decoders = Vec::new();
        while let Ok(decoder) = recv.try_recv() {
            decoders.push(decoder);
        }
        decoders
    }

    // Consumes the temporary context returning only the decoders
    //
    // Used when the temporary context is used to wrap a set of decoders
    pub fn into_decoders(self) -> Vec<Box<dyn LogicalPageDecoder>> {
        self.into_messages()
            .into_iter()
            .filter_map(|msg| {
                match msg {
                    DecoderMessage::ScanLine(_) => None,
                    // Should we ignore path here?  Currently, all "wrapping" layers should not have
                    // children and so there should be no path.  We could maybe debug_assert this.
                    DecoderMessage::Decoder(decoder_ready) => Some(decoder_ready.decoder),
                }
            })
            .collect::<Vec<_>>()
    }
}

/// A scheduler for a field's worth of data
///
/// Each page of incoming data maps to one `LogicalPageScheduler` instance.  However, this
/// page may map to many pages transitively.  For example, one page of struct data may cover
/// many pages of primitive child data.  In fact, the entire file is treated as one page
/// of SimpleStruct data.
///
/// The scheduler is responsible for calculating the neccesary I/O.  One schedule_range
/// request could trigger mulitple batches of I/O across multiple columns.  The scheduler
/// should emit decoders into the sink as quickly as possible.
///
/// As soon as a batch of data that can decoded then the scheduler should emit a decoder
/// in the "unloaded" state.  The decode stream will pull the decoder and start decoding.
///
/// The order in which decoders are emitted is important.  Pages should be emitted in
/// row-major order allowing decode of complete rows as quickly as possible.
///
/// The `LogicalPageScheduler` should be stateless and `Send` and `Sync`.  This is
/// because it might need to be shared.  For example, a list page has a reference to
/// the page schedulers for its items column.  This is shared with the follow-up I/O
/// task created when the offsets are loaded.
///
/// See [`crate::decoder`] for more information
pub trait LogicalPageScheduler: Send + Sync + std::fmt::Debug {
    /// Schedules I/O for the requested portions of the page.
    ///
    /// Note: `ranges` must be ordered and non-overlapping
    /// TODO: Support unordered or overlapping ranges in file scheduler
    fn schedule_ranges(
        &self,
        ranges: &[Range<u32>],
        context: &mut SchedulerContext,
        top_level_row: u64,
    ) -> Result<()>;
    /// Schedules I/O for the requested rows (identified by row offsets from start of page)
    /// TODO: implement this using schedule_ranges
    fn schedule_take(
        &self,
        indices: &[u32],
        context: &mut SchedulerContext,
        top_level_row: u64,
    ) -> Result<()>;
    /// The number of rows covered by this page
    fn num_rows(&self) -> u32;
}

/// A trait for tasks that decode data into an Arrow array
pub trait DecodeArrayTask: Send {
    /// Decodes the data into an Arrow array
    fn decode(self: Box<Self>) -> Result<ArrayRef>;
}

/// A task to decode data into an Arrow array
pub struct NextDecodeTask {
    /// The decode task itself
    pub task: Box<dyn DecodeArrayTask>,
    /// The number of rows that will be created
    pub num_rows: u32,
    /// Whether or not the decoder that created this still has more rows to decode
    pub has_more: bool,
}

pub struct DecoderReady {
    // The decoder that is ready to be decoded
    pub decoder: Box<dyn LogicalPageDecoder>,
    // The path to the decoder, the first value is the column index
    // following values, if present, are nested child indices
    //
    // For example, a path of [1, 1, 0] would mean to grab the second
    // column, then the second child, and then the first child.
    //
    // It could represent x in the following schema:
    //
    // score: float64
    // points: struct
    //   color: string
    //   location: struct
    //     x: float64
    //
    // Currently, only struct decoders have "children" although other
    // decoders may at some point as well.  List children are only
    // handled through indirect I/O at the moment and so they don't
    // need to be represented (yet)
    pub path: VecDeque<u32>,
}

pub enum DecoderMessage {
    // Emitted whenever the scheduler has made another pass through the columns.
    // Contains the number of rows that have been scheduled so far.
    ScanLine(u64),
    // Emitted whenever a decoder has been emitted
    Decoder(DecoderReady),
}

pub struct DecoderContext {
    source: mpsc::UnboundedReceiver<DecoderMessage>,
}

impl DecoderContext {
    pub fn new(source: mpsc::UnboundedReceiver<DecoderMessage>) -> Self {
        Self { source }
    }
}

/// A decoder for a field's worth of data
///
/// The decoder is initially "unloaded" (doesn't have all its data).  The [`Self::wait`]
/// method should be called to wait for the needed I/O data before attempting to decode
/// any further.
///
/// Unlike the other decoder types it is assumed that `LogicalPageDecoder` is stateful
/// and only `Send`.  This is why we don't need a `rows_to_skip` argument in [`Self::drain`]
pub trait LogicalPageDecoder: std::fmt::Debug + Send {
    /// Add a newly scheduled child decoder
    ///
    /// The default implementation does not expect children and returns
    /// an error.
    fn accept_child(&mut self, _child: DecoderReady) -> Result<()> {
        Err(Error::Internal {
            message: format!(
                "The decoder {:?} does not expect children but received a child",
                self
            ),
            location: location!(),
        })
    }
    /// Waits for enough data to be loaded to decode `num_rows` of data
    fn wait(&mut self, num_rows: u32) -> BoxFuture<Result<()>>;
    /// Creates a task to decode `num_rows` of data into an array
    fn drain(&mut self, num_rows: u32) -> Result<NextDecodeTask>;
    /// The number of rows that are in the page but haven't yet been "waited"
    fn unawaited(&self) -> u32;
    /// The number of rows that have been "waited" but not yet decoded
    fn avail(&self) -> u32;
    /// The data type of the decoded data
    fn data_type(&self) -> &DataType;
}

/// Decodes a batch of data from an in-memory structure created by [`crate::encoder::encode_batch`]
pub async fn decode_batch(batch: &EncodedBatch) -> Result<RecordBatch> {
    let mut decode_scheduler =
        DecodeBatchScheduler::new(batch.schema.as_ref(), &batch.page_table, &vec![]);
    let (tx, rx) = unbounded_channel();
    let io_scheduler = Arc::new(BufferScheduler::new(batch.data.clone())) as Arc<dyn EncodingsIo>;
    decode_scheduler
        .schedule_range(0..batch.num_rows, tx, io_scheduler)
        .await?;
    #[allow(clippy::single_range_in_vec_init)]
    let root_decoder = decode_scheduler
        .root_scheduler
        .new_root_decoder_ranges(&[0..batch.num_rows]);
    let stream = BatchDecodeStream::new(rx, batch.num_rows as u32, batch.num_rows, root_decoder);
    stream.into_stream().next().await.unwrap().task.await
}
