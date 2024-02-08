//! Utilities and traits for decoding data
//!
//! There are two types of decoders, logical decoders and physical decoder.
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
//!    of clumsy)
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
//! ## Future Changes
//!
//! Today, the selection of logical encodings is completely deterministic.  Struct fields will
//! always have a SimpleStruct encoding.  List fields will always have a list encoding.  All
//! other fields will always have a primitive encoding.
//!
//! In the future, this will probably change.  We may want to sometimes store a struct with a
//! packed encoding.  Or maybe we want to store an RLE of structs (note that logical RLE is
//! different than physical RLE).  Since logical encodings are consistent they will be the
//! same for every page of data.  This also means that we don't write the logical structure
//! into the file in any way currently and there is no notion of logical encoders.  That also
//! will probably change in the future.
//!
//! While the logical encodings are based on the schema, the pysical encodings are based on
//! the columns and the way they were written.  They can change from page to page.  The writer
//! does write down the physical encoding that was used into the column metadata and we do
//! use that to determine the physical encodings.
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
//!   be able to read back individual fields of the struct.  This might be a library limitation
//!   and not a format limitation.
//! * When things are stored in a single column they must have the same length.  This is an issue
//!   for list fields.  The items column does not usually have the same length as the parent list
//!   column (in most cases it is much longer).
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
//! push a decoder for that pages data into a channel.  The decode loop
//! ([`self::BatchDecodeStream`]) reads from that channel.  Each time it receives a decoder it
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

use std::{collections::VecDeque, ops::Range, sync::Arc};

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Schema};
use bytes::BytesMut;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::StreamExt;

use lance_core::Result;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::encodings::logical::list::ListPageScheduler;
use crate::encodings::logical::primitive::PrimitivePageScheduler;
use crate::EncodingsIo;

/// Metadata describing a page in a file
///
/// This is typically created by reading the metadata section of a Lance file
pub struct PageInfo {
    /// The number of rows in the page
    pub num_rows: u32,
    /// The physical decoder that explains the buffers in the page
    pub decoder: Arc<dyn PhysicalPageScheduler>,
    /// The offsets of the buffers in the file
    pub buffer_offsets: Arc<Vec<u64>>,
}

/// Metadata describing a column in a file
///
/// This is typically created by reading the metadata section of a Lance file
pub struct ColumnInfo {
    /// The metadata for each page in the column
    pub page_infos: Vec<Arc<PageInfo>>,
}

impl ColumnInfo {
    /// Create a new instance
    pub fn new(page_infos: Vec<Arc<PageInfo>>) -> Self {
        Self { page_infos }
    }
}

/// The scheduler for decoding batches
///
/// Lance decoding is done in two steps, scheduling, and decoding.  The
/// scheduling tends to be lightweight and should quickly figure what data
/// is needed from the disk and I/O requests are issued.  A decode task is
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
pub struct DecodeBatchScheduler {
    field_schedulers: Vec<Vec<Box<dyn LogicalPageScheduler>>>,
}

// As we schedule we keep one of these per column so that we know
// how far into the column we have already scheduled.
#[derive(Debug, Clone, Copy)]
struct FieldWalkStatus {
    rows_to_skip: u64,
    rows_to_take: u64,
    page_offset: u32,
    rows_queued: u64,
}

impl FieldWalkStatus {
    fn new_from_range(range: Range<u64>) -> Self {
        Self {
            rows_to_skip: range.start,
            rows_to_take: range.end - range.start,
            page_offset: 0,
            rows_queued: 0,
        }
    }
}

impl DecodeBatchScheduler {
    // This function is where the all important mapping from Arrow schema
    // to expected decoders happens.  Decoders are created by using the
    // expected field and the encoding metadata in the page.
    //
    // For example, if a field is a struct field then we expect a header
    // column that could have one of a few different encodings.
    //
    // If the encoding for a page is "non-null shredded" then the header
    // column will be empty (null pages only) and we will recurse into
    // the children in the following columns.
    //
    // On the other hand, if the field is a list, then we expect the
    // header column to be an integer column of offsets.
    //
    // Finally, if the field is a primitive, then the column should
    // have the basic encoding.
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
    // Examples: How to do RLE or dictionary encoded structs?
    //
    // TODO: In the future, this will need to be more flexible if
    // we want to allow custom encodings.  E.g. if the field's encoding
    // is not an encoding we expect then we should delegate to a plugin.
    fn create_field_scheduler<'a>(
        data_type: &DataType,
        column_infos: &mut impl Iterator<Item = &'a ColumnInfo>,
    ) -> Vec<Box<dyn LogicalPageScheduler>> {
        match data_type {
            DataType::Boolean
            | DataType::Date32
            | DataType::Date64
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
            | DataType::Duration(_)
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Int8
            | DataType::Interval(_)
            | DataType::Null
            | DataType::RunEndEncoded(_, _)
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::UInt8 => {
                // Primitive fields map to a single column
                let column = column_infos.next().unwrap();
                column
                    .page_infos
                    .iter()
                    .cloned()
                    .map(|page_info| {
                        Box::new(PrimitivePageScheduler::new(data_type.clone(), page_info))
                            as Box<dyn LogicalPageScheduler>
                    })
                    .collect::<Vec<_>>()
            }
            DataType::List(items_field) => {
                let offsets = Self::create_field_scheduler(&DataType::Int32, column_infos);
                let items = Self::create_field_scheduler(items_field.data_type(), column_infos);
                // TODO: This won't work if the offsets/items pages are not 1:1.  We need to fix
                // this by using the num_items field in the encoding data but that will require
                // us to actually look at the encoding data to create field schedulers which we
                // don't do yet
                offsets
                    .into_iter()
                    .zip(items)
                    .map(|(offsets_page, items_page)| {
                        Box::new(ListPageScheduler::new(
                            offsets_page,
                            vec![items_page],
                            DataType::Int32,
                        )) as Box<dyn LogicalPageScheduler>
                    })
                    .collect::<Vec<_>>()
            }
            _ => todo!(),
        }
    }

    /// Creates a new decode scheduler with the expected schema and the column
    /// metadata of the file.
    ///
    /// TODO: How does this work when doing projection?  Need to add tests.  Can
    /// probably take care of this in lance-file by only passing in the appropriate
    /// columns with the projected schema.
    pub fn new(schema: &Schema, column_infos: &[ColumnInfo]) -> Self {
        let mut col_info_iter = column_infos.iter();
        let field_schedulers = schema
            .fields
            .iter()
            .map(|field| Self::create_field_scheduler(field.data_type(), &mut col_info_iter))
            .collect::<Vec<_>>();
        Self { field_schedulers }
    }

    /// Schedules the load of a range of rows
    ///
    /// # Arguments
    ///
    /// * `range` - The range of rows to load
    /// * `sink` - A channel to send the decode tasks
    /// * `scheduler` An I/O scheduler to issue I/O requests
    pub async fn schedule_range(
        &mut self,
        range: Range<u64>,
        sink: mpsc::Sender<Box<dyn LogicalPageDecoder>>,
        scheduler: &Arc<dyn EncodingsIo>,
    ) -> Result<()> {
        let mut rows_to_read = range.end - range.start;

        let mut field_status =
            vec![FieldWalkStatus::new_from_range(range); self.field_schedulers.len()];

        // NOTE: The order in which we are scheduling tasks here is very important.  We want to schedule the I/O so that
        // we can deliver completed rows as quickly as possible to the decoder.  This means we want to schedule in row-major
        // order from start to the end.  E.g. if we schedule one column at a time then the decoder is going to have to wait
        // until almost all the I/O is finished before it can return a single batch.
        //
        // Luckily, we can do this using a simple greedy algorithm.  We iterate through each column independently.  For each
        // pass through the metadata we look for any column that doesn't have any "queued rows".  Once we find it we schedule
        // the next page for that column and increase its queued rows.  After each pass we should have some data queued for
        // each column.  We take the column with the least amount of queued data and decrement that amount from the queued
        // rows total of all columns.
        while rows_to_read > 0 {
            let mut min_rows_added = u32::MAX;
            for (col_idx, field_scheduler) in self.field_schedulers.iter().enumerate() {
                let status = &mut field_status[col_idx];
                if status.rows_queued == 0 {
                    let mut next_page = &field_scheduler[status.page_offset as usize];

                    while status.rows_to_skip > next_page.num_rows() as u64 {
                        status.rows_to_skip -= next_page.num_rows() as u64;
                        status.page_offset += 1;
                        next_page = &field_scheduler[status.page_offset as usize];
                    }

                    let page_range_start = status.rows_to_skip as u32;
                    let page_rows_remaining = next_page.num_rows() - page_range_start;
                    let rows_to_take = status.rows_to_take.min(page_rows_remaining as u64) as u32;
                    let page_range = page_range_start..(page_range_start + rows_to_take);

                    let scheduled = next_page.schedule_range(page_range, scheduler)?;

                    status.rows_queued += rows_to_take as u64;
                    status.rows_to_take -= rows_to_take as u64;
                    status.page_offset += 1;
                    status.rows_to_skip = 0;

                    sink.send(scheduled).await.unwrap();

                    min_rows_added = min_rows_added.min(rows_to_take);
                }
            }
            if min_rows_added == 0 {
                panic!("Error in scheduling logic, panic to avoid infinite loop");
            }
            rows_to_read -= min_rows_added as u64;
            for field_status in &mut field_status {
                field_status.rows_queued -= min_rows_added as u64;
            }
        }
        Ok(())
    }
}

// Represents the work to decode a single batch of data.  This is a CPU
// driven task where we untangle any sophisticated encodings.
struct DecodeBatchTask {
    columns: Vec<Vec<Box<dyn DecodeArrayTask>>>,
    schema: Schema,
}

impl DecodeBatchTask {
    fn run(self) -> Result<RecordBatch> {
        let columns = self
            .columns
            .into_iter()
            .map(|col_tasks| {
                let arrays = col_tasks
                    .into_iter()
                    .map(|col_task| col_task.decode())
                    .collect::<Result<Vec<_>>>()?;
                let array_refs = arrays.iter().map(|arr| arr.as_ref()).collect::<Vec<_>>();
                // TODO: If this is a primtiive column we should be able to avoid this
                // allocation + copy with "page bridging" which could save us a few CPU
                // cycles.
                Ok(arrow_select::concat::concat(&array_refs)?)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(RecordBatch::try_new(Arc::new(self.schema), columns)?)
    }
}

struct PartiallyDecodedPage {
    decoder: Box<dyn LogicalPageDecoder>,
    field_idx: u32,
}

/// A stream that takes scheduled jobs and generates decode tasks from them.
pub struct BatchDecodeStream {
    scheduled: mpsc::Receiver<Box<dyn LogicalPageDecoder>>,
    partial_pages: VecDeque<PartiallyDecodedPage>,
    schema: Schema,
    rows_remaining: u64,
    rows_per_batch: u32,
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
        scheduled: mpsc::Receiver<Box<dyn LogicalPageDecoder>>,
        schema: Schema,
        rows_per_batch: u32,
        num_rows: u64,
    ) -> Self {
        Self {
            scheduled,
            partial_pages: VecDeque::new(),
            schema,
            rows_remaining: num_rows,
            rows_per_batch,
        }
    }

    async fn next_batch_task(&mut self) -> Result<Option<DecodeBatchTask>> {
        if self.rows_remaining == 0 {
            return Ok(None);
        }

        let mut batch_steps = Vec::new();
        let rows_in_batch = (self.rows_per_batch as u64).min(self.rows_remaining) as u32;
        self.rows_remaining -= rows_in_batch as u64;

        let num_fields = self.schema.fields.len() as u32;
        for field_idx in 0..num_fields {
            let mut col_steps = Vec::new();
            let mut rows_remaining = rows_in_batch;
            while rows_remaining > 0 {
                let mut next_page_for_field = if self
                    .partial_pages
                    .front()
                    .map(|partial_page| partial_page.field_idx)
                    .unwrap_or(u32::MAX)
                    == field_idx
                {
                    self.partial_pages.pop_front().unwrap()
                } else {
                    let mut decoder = self.scheduled.recv().await.unwrap();
                    decoder.wait().await?;
                    PartiallyDecodedPage { field_idx, decoder }
                };
                let next_step = next_page_for_field.decoder.drain(rows_remaining)?;
                rows_remaining -= next_step.num_rows;
                col_steps.push(next_step.task);
                if next_step.has_more {
                    self.partial_pages.push_back(next_page_for_field);
                }
            }
            batch_steps.push(col_steps);
        }
        Ok(Some(DecodeBatchTask {
            columns: batch_steps,
            schema: self.schema.clone(),
        }))
    }

    pub fn into_stream(self) -> BoxStream<'static, JoinHandle<Result<RecordBatch>>> {
        let stream = futures::stream::unfold(self, |mut slf| async move {
            let next_task = slf.next_batch_task().await;
            let next_task = next_task.transpose().map(|next_task| {
                tokio::spawn(async move {
                    let next_task = next_task?;
                    next_task.run()
                })
            });
            next_task.map(|next_task| (next_task, slf))
        });
        stream.boxed()
    }
}

/// A decoder for simple buffer-based encodings
///
/// Physical decoders are able to decode into existing buffers for zero-copy operation.
///
/// Instances should be stateless and `Send` / `Sync`.  This is because multiple decode
/// tasks could reference the same page.  For example, imagine a page covers rows 900-1200
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
    /// Note: Most encodings deal with a single buffer.  They may multiple input buffers
    /// but they only have a single output buffer.  The exception to this rule is the `basic`
    /// encoding which has an output "validity" buffer and an output "values" buffers.  We
    /// may find there are other such exceptions.
    ///
    /// # Arguments
    ///
    /// * `rows_to_skip` - how many rows to skip (within the page) before decoding
    /// * `num_rows` - how many rows to decode
    /// * `buffers` - A mutable slice of "capacities" (as described above), one per buffer
    fn update_capacity(&self, rows_to_skip: u32, num_rows: u32, buffers: &mut [(u64, bool)]);
    /// Decodes the data into the requested buffers.
    ///
    /// You can assume that the capacity will have already been configured on the `BytesMut`
    /// according to the capacity calculated in [`Self::update_capacity`]
    ///
    /// # Arguments
    ///
    /// * `rows_to_skip` - how many rows to skip (within the page) before decoding
    /// * `num_rows` - how many rows to decode
    /// * `dest_buffers` - the output buffers to decode into
    fn decode_into(&self, rows_to_skip: u32, num_rows: u32, dest_buffers: &mut [BytesMut]);
}

/// A scheduler for simple buffer-based encodings
///
/// The scheduler is responsible for calculating what I/O is needed for the requested rows
///
/// Instances should be stateless and `Send` and `Sync`.  This is because instances can
/// be shared in follow-up I/O tasks.
///
/// See [`crate::decoder`] for more information
pub trait PhysicalPageScheduler: Send + Sync + std::fmt::Debug {
    /// Schedules a batch of I/O to load the data needed for the requested range
    ///
    /// Returns a future that will yield a decoder once the data has been loaded
    ///
    /// # Arguments
    ///
    /// * `range` - the range of row offsets (relative to start of page) requested
    /// * `scheduler` - a scheduler to submit the I/O request to
    fn schedule_range(
        &self,
        range: Range<u32>,
        scheduler: &dyn EncodingsIo,
    ) -> BoxFuture<'static, Result<Box<dyn PhysicalPageDecoder>>>;
}

/// A scheduler for a field's worth of data
///
/// The scheduler is responsible for calculating the neccesary I/O.  One schedule_range
/// request could trigger mulitple batches of I/O across multiple columns.
///
/// Returns a page decoder in the "unloaded" state
///
/// The `LogicalPageScheduler` should be stateless and `Send` and `Sync`.  This is
/// because it might need to be shared.  For example, a list page has a reference to
/// the page schedulers for its items column.  This is shared with the follow-up I/O
/// task created when the offsets are loaded.
///
/// See [`crate::decoder`] for more information
pub trait LogicalPageScheduler: Send + Sync {
    fn schedule_range(
        &self,
        range: Range<u32>,
        scheduler: &Arc<dyn EncodingsIo>,
    ) -> Result<Box<dyn LogicalPageDecoder>>;
    fn num_rows(&self) -> u32;
}

/// A trait for tasks that decode data into an Arrow array
pub trait DecodeArrayTask: Send {
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

/// A decoder for a field's worth of data
///
/// The decoder is initially "unloaded" (doesn't have all its data).  The [`Self::wait`]
/// method should be called to wait for the needed I/O data before attempting to decode
/// any further.
///
/// Unlike the other decoder types it is assumed that `LogicalPageDecoder` is stateful
/// and only `Send`.  This is why we don't need a `rows_to_skip` argument in [`Self::drain`]
pub trait LogicalPageDecoder: Send {
    /// Waits for the data to be loaded
    fn wait(&mut self) -> BoxFuture<Result<()>>;
    /// Creates a task to decode `num_rows` of data into an array
    fn drain(&mut self, num_rows: u32) -> Result<NextDecodeTask>;
    /// The number of rows remaining in the task
    fn avail(&self) -> u32;
}
