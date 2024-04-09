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

use std::{ops::Range, sync::Arc};

use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::BytesMut;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::StreamExt;
use log::trace;
use snafu::{location, Location};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use lance_core::{Error, Result};

use crate::encodings::logical::fixed_size_list::FslPageScheduler;
use crate::encodings::logical::list::ListPageScheduler;
use crate::encodings::logical::primitive::PrimitivePageScheduler;
use crate::encodings::logical::r#struct::SimpleStructScheduler;
use crate::encodings::logical::utf8::Utf8PageScheduler;
use crate::encodings::physical::{ColumnBuffers, FileBuffers};
use crate::format::pb;
use crate::EncodingsIo;

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
pub struct ColumnInfo {
    /// The metadata for each page in the column
    pub page_infos: Vec<Arc<PageInfo>>,
    pub buffer_offsets: Vec<u64>,
}

impl ColumnInfo {
    /// Create a new instance
    pub fn new(page_infos: Vec<Arc<PageInfo>>, buffer_offsets: Vec<u64>) -> Self {
        Self {
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
    root_scheduler: SimpleStructScheduler,
}

impl DecodeBatchScheduler {
    fn is_primitive(data_type: &DataType) -> bool {
        if data_type.is_primitive() {
            true
        } else {
            match data_type {
                // DataType::is_primitive doesn't consider these primitive but we do
                DataType::Boolean | DataType::Null => true,
                DataType::FixedSizeList(inner, _) => Self::is_primitive(inner.data_type()),
                _ => false,
            }
        }
    }

    fn create_primitive_scheduler<'a>(
        data_type: &DataType,
        column_infos: &mut impl Iterator<Item = &'a ColumnInfo>,
        buffers: FileBuffers,
    ) -> Vec<Box<dyn LogicalPageScheduler>> {
        // Primitive fields map to a single column
        let column = column_infos.next().unwrap();
        let column_buffers = ColumnBuffers {
            file_buffers: buffers,
            positions: &column.buffer_offsets,
        };
        column
            .page_infos
            .iter()
            .cloned()
            .map(|page_info| {
                Box::new(PrimitivePageScheduler::new(
                    data_type.clone(),
                    page_info,
                    column_buffers,
                )) as Box<dyn LogicalPageScheduler>
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
    ) -> Vec<Box<dyn LogicalPageScheduler>> {
        if Self::is_primitive(data_type) {
            return Self::create_primitive_scheduler(data_type, column_infos, buffers);
        }
        match data_type {
            DataType::FixedSizeList(inner, dimension) => {
                // A fixed size list column could either be a physical or a logical decoder
                // depending on the child data type.
                if Self::is_primitive(inner.data_type()) {
                    Self::create_primitive_scheduler(data_type, column_infos, buffers)
                } else {
                    let inner_schedulers =
                        Self::create_field_scheduler(inner.data_type(), column_infos, buffers);
                    inner_schedulers
                        .into_iter()
                        .map(|inner| {
                            Box::new(FslPageScheduler::new(inner, *dimension as u32))
                                as Box<dyn LogicalPageScheduler>
                        })
                        .collect::<Vec<_>>()
                }
            }
            DataType::List(items_field) => {
                let offsets = Self::create_field_scheduler(&DataType::Int32, column_infos, buffers);
                let items =
                    Self::create_field_scheduler(items_field.data_type(), column_infos, buffers);
                // TODO: This may need to be more flexible in the future if an items page can
                // be shared by multiple offsets pages.
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
            DataType::Utf8 => {
                let list_decoders = Self::create_field_scheduler(
                    &DataType::List(Arc::new(Field::new("item", DataType::UInt8, true))),
                    column_infos,
                    buffers,
                );
                list_decoders
                    .into_iter()
                    .map(|list_decoder| {
                        Box::new(Utf8PageScheduler::new(list_decoder))
                            as Box<dyn LogicalPageScheduler>
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
                vec![Box::new(SimpleStructScheduler::new(
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
    ///
    /// TODO: How does this work when doing projection?  Need to add tests.  Can
    /// probably take care of this in lance-file by only passing in the appropriate
    /// columns with the projected schema.
    pub fn new(
        schema: &Schema,
        column_infos: &[ColumnInfo],
        file_buffer_positions: &Vec<u64>,
    ) -> Self {
        let mut col_info_iter = column_infos.iter();
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
        let root_scheduler = SimpleStructScheduler::new(field_schedulers, schema.fields.clone());
        Self { root_scheduler }
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
        sink: mpsc::UnboundedSender<Box<dyn LogicalPageDecoder>>,
        scheduler: &Arc<dyn EncodingsIo>,
    ) -> Result<()> {
        let rows_to_read = range.end - range.start;
        trace!("Scheduling range {:?} ({} rows)", range, rows_to_read);

        let range = range.start as u32..range.end as u32;

        self.root_scheduler
            .schedule_ranges(&[range.clone()], scheduler, &sink)?;

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
        indices: &[u32],
        sink: mpsc::UnboundedSender<Box<dyn LogicalPageDecoder>>,
        scheduler: &Arc<dyn EncodingsIo>,
    ) -> Result<()> {
        debug_assert!(indices.windows(2).all(|w| w[0] < w[1]));
        trace!(
            "Scheduling take of {} rows [{}]",
            indices.len(),
            if indices.len() == 1 {
                indices[0].to_string()
            } else {
                format!("{}, ..., {}", indices[0], indices[indices.len() - 1])
            }
        );
        self.root_scheduler
            .schedule_take(indices, scheduler, &sink)?;
        trace!("Finished scheduling take of {} rows", indices.len());
        Ok(())
    }
}

/// A stream that takes scheduled jobs and generates decode tasks from them.
pub struct BatchDecodeStream {
    scheduled: mpsc::UnboundedReceiver<Box<dyn LogicalPageDecoder>>,
    current: Option<Box<dyn LogicalPageDecoder>>,
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
        scheduled: mpsc::UnboundedReceiver<Box<dyn LogicalPageDecoder>>,
        rows_per_batch: u32,
        num_rows: u64,
    ) -> Self {
        Self {
            scheduled,
            current: None,
            rows_remaining: num_rows,
            rows_per_batch,
        }
    }

    async fn next_batch_task(&mut self) -> Result<Option<NextDecodeTask>> {
        trace!("Draining batch task");
        if self.rows_remaining == 0 {
            return Ok(None);
        }

        let to_take = self.rows_remaining.min(self.rows_per_batch as u64) as u32;
        self.rows_remaining -= to_take as u64;

        if self.current.is_none() {
            trace!("Loading new top-level page");
            self.current = Some(self.scheduled.recv().await.unwrap());
        }
        let current = self.current.as_mut().unwrap();
        let avail = current.avail();
        trace!("Top level page has {} rows already available", avail);
        if avail < to_take {
            current.wait(to_take, &mut self.scheduled).await?;
        }
        let next_task = current.drain(to_take)?;
        if !next_task.has_more {
            self.current = None;
        }
        Ok(Some(next_task))
    }

    fn task_to_batch(task: NextDecodeTask) -> Result<RecordBatch> {
        let struct_arr = task.task.decode()?;
        Ok(RecordBatch::from(struct_arr.as_struct()))
    }

    pub fn into_stream(self) -> BoxStream<'static, JoinHandle<Result<RecordBatch>>> {
        let stream = futures::stream::unfold(self, |mut slf| async move {
            let next_task = slf.next_batch_task().await;
            let next_task = next_task.transpose().map(|next_task| {
                tokio::spawn(async move {
                    let next_task = next_task?;
                    Self::task_to_batch(next_task)
                })
            });
            next_task.map(|next_task| (next_task, slf))
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
    /// according to the capacity calculated in [`Self::update_capacity`]
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
    fn schedule_ranges(
        &self,
        ranges: &[Range<u32>],
        scheduler: &dyn EncodingsIo,
    ) -> BoxFuture<'static, Result<Box<dyn PhysicalPageDecoder>>>;
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
        scheduler: &Arc<dyn EncodingsIo>,
        sink: &mpsc::UnboundedSender<Box<dyn LogicalPageDecoder>>,
    ) -> Result<()>;
    /// Schedules I/O for the requested rows (identified by row offsets from start of page)
    /// TODO: implement this using schedule_ranges
    fn schedule_take(
        &self,
        indices: &[u32],
        scheduler: &Arc<dyn EncodingsIo>,
        sink: &mpsc::UnboundedSender<Box<dyn LogicalPageDecoder>>,
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

/// A decoder for a field's worth of data
///
/// The decoder is initially "unloaded" (doesn't have all its data).  The [`Self::wait`]
/// method should be called to wait for the needed I/O data before attempting to decode
/// any further.
///
/// Unlike the other decoder types it is assumed that `LogicalPageDecoder` is stateful
/// and only `Send`.  This is why we don't need a `rows_to_skip` argument in [`Self::drain`]
pub trait LogicalPageDecoder: Send {
    /// Waits for enough data to be loaded to decode `num_rows` of data
    fn wait<'a>(
        &'a mut self,
        num_rows: u32,
        source: &'a mut mpsc::UnboundedReceiver<Box<dyn LogicalPageDecoder>>,
    ) -> BoxFuture<'a, Result<()>>;
    /// Creates a task to decode `num_rows` of data into an array
    fn drain(&mut self, num_rows: u32) -> Result<NextDecodeTask>;
    /// The number of rows that are in the page but haven't yet been "waited"
    fn unawaited(&self) -> u32;
    /// The number of rows that have been "waited" but not yet decoded
    fn avail(&self) -> u32;
}
