// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities and traits for scheduling & decoding data
//!
//! Reading data involves two steps: scheduling and decoding.  The
//! scheduling step is responsible for figuring out what data is needed
//! and issuing the appropriate I/O requests.  The decoding step is
//! responsible for taking the loaded data and turning it into Arrow
//! arrays.
//!
//! # Scheduling
//!
//! Scheduling is split into [`self::FieldScheduler`] and [`self::PageScheduler`].
//! There is one field scheduler for each output field, which may map to many
//! columns of actual data.  A field scheduler is responsible for figuring out
//! the order in which pages should be scheduled.  Field schedulers then delegate
//! to page schedulers to figure out the I/O requests that need to be made for
//! the page.
//!
//! Page schedulers also create the decoders that will be used to decode the
//! scheduled data.
//!
//! # Decoding
//!
//! Decoders are split into [`self::PhysicalPageDecoder`] and
//! [`self::LogicalPageDecoder`].  Note that both physical and logical decoding
//! happens on a per-page basis.  There is no concept of a "field decoder" or
//! "column decoder".
//!
//! The physical decoders handle lower level encodings.  They have a few advantages:
//!
//!  * They do not need to decode into an Arrow array and so they don't need
//!    to be enveloped into the Arrow filesystem (e.g. Arrow doesn't have a
//!    bit-packed type.  We can use variable-length binary but that is kind
//!    of awkward)
//!  * They can decode into an existing allocation.  This can allow for "page
//!    bridging".  If we are trying to decode into a batch of 1024 rows and
//!    the rows 0..1024 are spread across two pages then we can avoid a memory
//!    copy by allocating once and decoding each page into the outer allocation.
//!    (note: page bridging is not actually implemented yet)
//!
//! However, there are some limitations for physical decoders:
//!
//!  * They are constrained to a single column
//!  * The API is more complex
//!
//! The logical decoders are designed to map one or more columns of Lance
//! data into an Arrow array.
//!
//! Typically, a "logical encoding" will have both a logical decoder and a field scheduler.
//! Meanwhile, a "physical encoding" will have a physical decoder but no corresponding field
//! scheduler.git add --all
//!
//!
//! # General notes
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
use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema};
use bytes::{Bytes, BytesMut};
use futures::future::BoxFuture;
use futures::stream::{BoxStream, FuturesOrdered};
use futures::{FutureExt, StreamExt, TryStreamExt};
use lance_core::datatypes::{Field, Schema};
use log::trace;
use snafu::{location, Location};
use tokio::sync::mpsc::{self, unbounded_channel};

use lance_core::{Error, Result};
use tracing::instrument;

use crate::encoder::get_str_encoding_type;
use crate::encoder::{values_column_encoding, EncodedBatch};
use crate::encodings::logical::binary::BinaryFieldScheduler;
use crate::encodings::logical::list::{ListFieldScheduler, OffsetPageInfo};
use crate::encodings::logical::primitive::PrimitiveFieldScheduler;
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
    /// The offsets and sizes of the buffers in the file
    pub buffer_offsets_and_sizes: Arc<[(u64, u64)]>,
}

/// Metadata describing a column in a file
///
/// This is typically created by reading the metadata section of a Lance file
#[derive(Debug, Clone)]
pub struct ColumnInfo {
    /// The index of the column in the file
    pub index: u32,
    /// The metadata for each page in the column
    pub page_infos: Arc<[PageInfo]>,
    /// File positions and their sizes of the column-level buffers
    pub buffer_offsets_and_sizes: Arc<[(u64, u64)]>,
    pub encoding: pb::ColumnEncoding,
}

impl ColumnInfo {
    /// Create a new instance
    pub fn new(
        index: u32,
        page_infos: Arc<[PageInfo]>,
        buffer_offsets_and_sizes: Vec<(u64, u64)>,
        encoding: pb::ColumnEncoding,
    ) -> Self {
        Self {
            index,
            page_infos,
            buffer_offsets_and_sizes: buffer_offsets_and_sizes.into_boxed_slice().into(),
            encoding,
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
    pub root_scheduler: Arc<dyn FieldScheduler>,
    pub root_fields: Fields,
}

/// Represents a series of decoder strategies
///
/// These strategies will be applied, in order, to determine
/// which decoder to use for a field.
#[derive(Debug, Clone)]
pub struct DecoderMiddlewareChain {
    chain: Vec<Arc<dyn FieldDecoderStrategy>>,
}

impl Default for DecoderMiddlewareChain {
    fn default() -> Self {
        Self {
            chain: Default::default(),
        }
        .add_strategy(Arc::new(CoreFieldDecoderStrategy))
    }
}

impl DecoderMiddlewareChain {
    /// Creates an empty decoder chain
    pub fn new() -> Self {
        Self { chain: Vec::new() }
    }

    /// Adds a decoder to the end of the chain
    pub fn add_strategy(mut self, decoder: Arc<dyn FieldDecoderStrategy>) -> Self {
        self.chain.push(decoder);
        self
    }

    /// Obtain a cursor into the chain that can be used to create
    /// field schedulers
    pub(crate) fn cursor<'a>(
        &'a self,
        io: &'a Arc<dyn EncodingsIo>,
    ) -> DecoderMiddlewareChainCursor<'a> {
        DecoderMiddlewareChainCursor {
            chain: self,
            io,
            cur_idx: 0,
            path: VecDeque::new(),
        }
    }
}

/// A cursor into a decoder middleware chain
///
/// Each field scheduler is given a cursor during the create_field_scheduler
/// call.  This cursor can be used both to create child field schedulers and
/// to create a scheduler from an inner encoding.
pub struct DecoderMiddlewareChainCursor<'a> {
    chain: &'a DecoderMiddlewareChain,
    io: &'a Arc<dyn EncodingsIo>,
    path: VecDeque<u32>,
    cur_idx: usize,
}

pub type ChosenFieldScheduler<'a> = (
    DecoderMiddlewareChainCursor<'a>,
    BoxFuture<'static, Result<Arc<dyn FieldScheduler>>>,
);

impl<'a> DecoderMiddlewareChainCursor<'a> {
    /// Returns the current path into the field being decoded
    pub fn current_path(&self) -> &VecDeque<u32> {
        &self.path
    }

    /// Returns the I/O service which can be used to grab column metadata
    pub fn io(&self) -> &Arc<dyn EncodingsIo> {
        self.io
    }

    /// Delegates responsibilty to the next encoder in the chain
    ///
    /// Field schedulers should call this method when:
    ///
    /// * They do not understand or handle the encoding
    /// * They wrap an encoding and want a scheduler for the inner encoding
    pub fn next(
        mut self,
        field: &Field,
        column_infos: &mut VecDeque<ColumnInfo>,
        buffers: FileBuffers,
    ) -> Result<ChosenFieldScheduler<'a>> {
        if self.cur_idx >= self.chain.chain.len() {
            return Err(Error::invalid_input(
                "The user requested field {:?} from column {:?} but no decoders were registered to handle it",
                location!(),
            ));
        }
        let item = &self.chain.chain[self.cur_idx];
        self.cur_idx += 1;
        item.create_field_scheduler(field, column_infos, buffers, self)
    }

    /// Restarts the decoder chain without creating a new "child"
    ///
    /// This can be useful, for example, when a field scheduler has
    /// an inner scheduler, and the current / parent strategies might
    /// apply to the inner scheduler.
    ///
    /// If the current / parent strategies should not be consulted
    /// then call [`Self::next`] instead.
    pub fn restart_at_current(
        mut self,
        field: &Field,
        column_infos: &mut VecDeque<ColumnInfo>,
        buffers: FileBuffers,
    ) -> Result<ChosenFieldScheduler<'a>> {
        self.cur_idx = 0;
        self.next(field, column_infos, buffers)
    }

    /// Restarts the decoder chain for a new "child" field.  The main
    /// difference between this and [`Self::restart_at_current`] is that
    /// this method will modify [`Self::current_path`]
    pub fn new_child(
        mut self,
        child_idx: u32,
        field: &Field,
        column_infos: &mut VecDeque<ColumnInfo>,
        buffers: FileBuffers,
    ) -> Result<ChosenFieldScheduler<'a>> {
        self.path.push_back(child_idx);
        self.cur_idx = 0;
        self.next(field, column_infos, buffers)
    }

    /// Starts the decoding process for a field
    pub(crate) fn start(
        mut self,
        field: &Field,
        column_infos: &mut VecDeque<ColumnInfo>,
        buffers: FileBuffers,
    ) -> Result<ChosenFieldScheduler<'a>> {
        self.path.clear();
        self.cur_idx = 0;
        self.next(field, column_infos, buffers)
    }
}

// A trait that handles the mapping from Arrow schema to field decoders.
//
// Note that the decoders can only be figured out using both the schema AND
// the column metadata.  In theory, one could infer the decoder / column type
// using only the column metadata.  However, field nullability would be
// missing / incorrect and its also not as easy as it sounds since pages can
// have different encodings and those encodings often have various layers.
// Also, sometimes the inference is just impossible.  For example,
// Timestamp, Float64, Int64, and UInt64 will all be encoded as 8-byte value
// encoding.  The only way to know the data type is to look at the schema.
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
pub trait FieldDecoderStrategy: Send + Sync + std::fmt::Debug {
    /// Called to create a field scheduler for a field
    ///
    /// Stratgies can examine:
    /// * The target field
    /// * The column metadata (potentially consuming multiple columns)
    ///
    /// If a strategy does not handle an encoding it should call
    /// `chain.next` to delegate to the next strategy in the chain.
    ///
    /// The actual scheduler creation is asynchronous.  This is because
    /// the scheduler may need to read column metadata from disk.
    fn create_field_scheduler<'a>(
        &self,
        field: &Field,
        column_infos: &mut VecDeque<ColumnInfo>,
        buffers: FileBuffers,
        chain: DecoderMiddlewareChainCursor<'a>,
    ) -> Result<ChosenFieldScheduler<'a>>;
}

/// The core decoder strategy handles all the various Arrow types
#[derive(Debug, Default)]
pub struct CoreFieldDecoderStrategy;

impl CoreFieldDecoderStrategy {
    /// This is just a sanity check to ensure there is no "wrapped encodings"
    /// that haven't been handled.
    fn ensure_values_encoded(column_info: &ColumnInfo, path: &VecDeque<u32>) -> Result<()> {
        let column_encoding = column_info
            .encoding
            .column_encoding
            .as_ref()
            .ok_or_else(|| {
                Error::invalid_input(
                    format!(
                        "the column at index {} was missing a ColumnEncoding",
                        column_info.index
                    ),
                    location!(),
                )
            })?;
        if matches!(
            column_encoding,
            pb::column_encoding::ColumnEncoding::Values(_)
        ) {
            Ok(())
        } else {
            Err(Error::invalid_input(format!("the column at index {} mapping to the input field at {:?} has column encoding {:?} and no decoder is registered to handle it", column_info.index, path, column_encoding), location!()))
        }
    }

    fn is_primitive(data_type: &DataType) -> bool {
        if data_type.is_primitive() {
            true
        } else if get_str_encoding_type() {
            match data_type {
                // DataType::is_primitive doesn't consider these primitive but we do
                DataType::Boolean
                | DataType::Null
                | DataType::FixedSizeBinary(_)
                | DataType::Utf8 => true,
                DataType::FixedSizeList(inner, _) => Self::is_primitive(inner.data_type()),
                _ => false,
            }
        } else {
            match data_type {
                // DataType::is_primitive doesn't consider these primitive but we do
                DataType::Boolean
                | DataType::Null
                | DataType::FixedSizeBinary(_)
                // | DataType::Utf8 
                => true,
                DataType::FixedSizeList(inner, _) => Self::is_primitive(inner.data_type()),
                _ => false,
            }
        }
    }

    fn create_primitive_scheduler(
        data_type: &DataType,
        path: &VecDeque<u32>,
        column: &ColumnInfo,
        buffers: FileBuffers,
    ) -> Result<Arc<dyn FieldScheduler>> {
        Self::ensure_values_encoded(column, path)?;
        // Primitive fields map to a single column
        let column_buffers = ColumnBuffers {
            file_buffers: buffers,
            positions_and_sizes: &column.buffer_offsets_and_sizes,
        };
        Ok(Arc::new(PrimitiveFieldScheduler::new(
            data_type.clone(),
            column.page_infos.clone(),
            column_buffers,
        )))
    }

    /// Helper method to verify the page encoding of a struct header column
    fn check_simple_struct(column_info: &ColumnInfo, path: &VecDeque<u32>) -> Result<()> {
        Self::ensure_values_encoded(column_info, path)?;
        if !column_info.page_infos.len() == 1 {
            return Err(Error::InvalidInput { source: format!("Due to schema we expected a struct column but we received a column with {} pages and right now we only support struct columns with 1 page", column_info.page_infos.len()).into(), location: location!() });
        }
        let encoding = &column_info.page_infos[0].encoding;
        match encoding.array_encoding.as_ref().unwrap() {
            pb::array_encoding::ArrayEncoding::Struct(_) => Ok(()),
            _ => Err(Error::InvalidInput { source: format!("Expected a struct encoding because we have a struct field in the schema but got the encoding {:?}", encoding).into(), location: location!() }),
        }
    }
}

impl FieldDecoderStrategy for CoreFieldDecoderStrategy {
    fn create_field_scheduler<'a>(
        &self,
        field: &Field,
        column_infos: &mut VecDeque<ColumnInfo>,
        buffers: FileBuffers,
        chain: DecoderMiddlewareChainCursor<'a>,
    ) -> Result<ChosenFieldScheduler<'a>> {
        let data_type = field.data_type();
        if Self::is_primitive(&data_type) {
            let primitive_col = column_infos.pop_front().unwrap();
            let scheduler = Self::create_primitive_scheduler(
                &data_type,
                chain.current_path(),
                &primitive_col,
                buffers,
            )?;
            return Ok((chain, std::future::ready(Ok(scheduler)).boxed()));
        }
        match &data_type {
            DataType::FixedSizeList(inner, _dimension) => {
                // A fixed size list column could either be a physical or a logical decoder
                // depending on the child data type.
                if Self::is_primitive(inner.data_type()) {
                    let primitive_col = column_infos.pop_front().unwrap();
                    let scheduler = Self::create_primitive_scheduler(
                        &data_type,
                        chain.current_path(),
                        &primitive_col,
                        buffers,
                    )?;
                    return Ok((chain, std::future::ready(Ok(scheduler)).boxed()));
                } else {
                    todo!()
                }
            }
            DataType::List(items_field) | DataType::LargeList(items_field) => {
                let offsets_column = column_infos.pop_front().unwrap();
                Self::ensure_values_encoded(&offsets_column, chain.current_path())?;
                let offsets_column_buffers = ColumnBuffers {
                    file_buffers: buffers,
                    positions_and_sizes: &offsets_column.buffer_offsets_and_sizes,
                };
                let (chain, items_scheduler) = chain.new_child(
                    /*child_idx=*/ 0,
                    &field.children[0],
                    column_infos,
                    buffers,
                )?;

                let (inner_infos, null_offset_adjustments): (Vec<_>, Vec<_>) = offsets_column
                    .page_infos
                    .iter()
                    .map(|offsets_page| {
                        if let Some(pb::array_encoding::ArrayEncoding::List(list_encoding)) =
                            &offsets_page.encoding.array_encoding
                        {
                            let inner = PageInfo {
                                buffer_offsets_and_sizes: offsets_page
                                    .buffer_offsets_and_sizes
                                    .clone(),
                                encoding: list_encoding.offsets.as_ref().unwrap().as_ref().clone(),
                                num_rows: offsets_page.num_rows,
                            };
                            (
                                inner,
                                OffsetPageInfo {
                                    offsets_in_page: offsets_page.num_rows,
                                    null_offset_adjustment: list_encoding.null_offset_adjustment,
                                    num_items_referenced_by_page: list_encoding.num_items,
                                },
                            )
                        } else {
                            // TODO: Should probably return Err here
                            panic!("Expected a list column");
                        }
                    })
                    .unzip();
                let inner = Arc::new(PrimitiveFieldScheduler::new(
                    DataType::UInt64,
                    Arc::from(inner_infos.into_boxed_slice()),
                    offsets_column_buffers,
                )) as Arc<dyn FieldScheduler>;
                let offset_type = if matches!(data_type, DataType::List(_)) {
                    DataType::Int32
                } else {
                    DataType::Int64
                };
                let items_type = items_field.data_type().clone();
                let list_scheduler_fut = async move {
                    let items_scheduler = items_scheduler.await?;
                    Ok(Arc::new(ListFieldScheduler::new(
                        inner,
                        items_scheduler,
                        items_type,
                        offset_type,
                        null_offset_adjustments,
                    )) as Arc<dyn FieldScheduler>)
                }
                .boxed();
                Ok((chain, list_scheduler_fut))
            }
            DataType::Utf8 | DataType::Binary | DataType::LargeBinary | DataType::LargeUtf8 => {
                let list_type = if matches!(data_type, DataType::Utf8 | DataType::Binary) {
                    DataType::List(Arc::new(ArrowField::new("item", DataType::UInt8, true)))
                } else {
                    DataType::LargeList(Arc::new(ArrowField::new("item", DataType::UInt8, true)))
                };
                let list_field = ArrowField::new(&field.name, list_type, true);
                let list_field = Field::try_from(&list_field).unwrap();
                // We've changed the data type but are still decoding the same "field"
                let (chain, list_decoder) =
                    chain.restart_at_current(&list_field, column_infos, buffers)?;
                let data_type = data_type.clone();
                let binary_scheduler_fut = async move {
                    let list_decoder = list_decoder.await?;
                    Ok(
                        Arc::new(BinaryFieldScheduler::new(list_decoder, data_type.clone()))
                            as Arc<dyn FieldScheduler>,
                    )
                }
                .boxed();
                Ok((chain, binary_scheduler_fut))
            }
            DataType::Struct(fields) => {
                let column_info = column_infos.pop_front().unwrap();
                Self::check_simple_struct(&column_info, chain.current_path()).unwrap();
                let (chain, child_schedulers) = field.children.iter().enumerate().try_fold(
                    (chain, FuturesOrdered::new()),
                    |(chain, mut fields), (field_idx, field)| {
                        let (chain, field) =
                            chain.new_child(field_idx as u32, field, column_infos, buffers)?;
                        fields.push_back(field);
                        Result::Ok((chain, fields))
                    },
                )?;

                let fields = fields.clone();
                let struct_fut = async move {
                    let child_schedulers = child_schedulers.try_collect::<Vec<_>>().await?;
                    Ok(
                        Arc::new(SimpleStructScheduler::new(child_schedulers, fields))
                            as Arc<dyn FieldScheduler>,
                    )
                }
                .boxed();

                // For now, we don't record nullability for structs.  As a result, there is always
                // only one "page" of struct data.  In the future, this will change.  A null-aware
                // struct scheduler will need to first calculate how many rows are in the struct page
                // and then find the child pages that overlap.  This should be doable.
                Ok((chain, struct_fut))
            }
            // TODO: Still need support for dictionary / RLE
            _ => chain.next(field, column_infos, buffers),
        }
    }
}

/// Create's a dummy ColumnInfo for the root column
fn root_column(num_rows: u64) -> ColumnInfo {
    let num_root_pages = num_rows.div_ceil(u32::MAX as u64);
    let final_page_num_rows = num_rows % (u32::MAX as u64);
    let root_pages = (0..num_root_pages)
        .map(|i| PageInfo {
            num_rows: if i == num_root_pages - 1 {
                final_page_num_rows as u32
            } else {
                u32::MAX
            },
            encoding: pb::ArrayEncoding {
                array_encoding: Some(pb::array_encoding::ArrayEncoding::Struct(
                    pb::SimpleStruct {},
                )),
            },
            buffer_offsets_and_sizes: Arc::new([]),
        })
        .collect::<Vec<_>>();
    ColumnInfo {
        buffer_offsets_and_sizes: Arc::new([]),
        encoding: values_column_encoding(),
        index: u32::MAX,
        page_infos: Arc::from(root_pages),
    }
}

impl DecodeBatchScheduler {
    /// Creates a new decode scheduler with the expected schema and the column
    /// metadata of the file.
    pub async fn try_new<'a>(
        schema: &'a Schema,
        column_infos: &[Arc<ColumnInfo>],
        file_buffer_positions_and_sizes: &'a Vec<(u64, u64)>,
        num_rows: u64,
        decoder_strategy: &DecoderMiddlewareChain,
        io: &Arc<dyn EncodingsIo>,
    ) -> Result<Self> {
        let buffers = FileBuffers {
            positions_and_sizes: file_buffer_positions_and_sizes,
        };
        let arrow_schema = ArrowSchema::from(schema);
        let root_fields = arrow_schema.fields().clone();
        let mut columns = VecDeque::with_capacity(column_infos.len() + 1);
        columns.push_back(root_column(num_rows));
        columns.extend(column_infos.iter().map(|col| col.as_ref().clone()));
        let root_type = DataType::Struct(root_fields.clone());
        let root_field = Field::try_from(&ArrowField::new("root", root_type, false))?;
        let (_, root_scheduler_fut) =
            decoder_strategy
                .cursor(io)
                .start(&root_field, &mut columns, buffers)?;
        let root_scheduler = root_scheduler_fut.await?;
        Ok(Self {
            root_scheduler,
            root_fields,
        })
    }

    pub fn from_scheduler(root_scheduler: Arc<dyn FieldScheduler>, root_fields: Fields) -> Self {
        Self {
            root_scheduler,
            root_fields,
        }
    }

    fn do_schedule_ranges(
        &mut self,
        ranges: &[Range<u64>],
        filter: &FilterExpression,
        io: Arc<dyn EncodingsIo>,
        mut schedule_action: impl FnMut(Result<DecoderMessage>),
    ) {
        let rows_requested = ranges.iter().map(|r| r.end - r.start).sum::<u64>();
        trace!("Scheduling ranges {:?} ({} rows)", ranges, rows_requested);

        let mut context = SchedulerContext::new(io);
        let maybe_root_job = self.root_scheduler.schedule_ranges(ranges, filter);
        if let Err(schedule_ranges_err) = maybe_root_job {
            schedule_action(Err(schedule_ranges_err));
            return;
        }
        let mut root_job = maybe_root_job.unwrap();
        let mut num_rows_scheduled = 0;
        let mut rows_to_schedule = root_job.num_rows();
        trace!("Scheduled ranges refined to {} rows", rows_to_schedule);
        while rows_to_schedule > 0 {
            let maybe_next_scan_line = root_job.schedule_next(&mut context, num_rows_scheduled);
            if let Err(schedule_next_err) = maybe_next_scan_line {
                schedule_action(Err(schedule_next_err));
                return;
            }
            let next_scan_line = maybe_next_scan_line.unwrap();
            num_rows_scheduled += next_scan_line.rows_scheduled as u64;
            rows_to_schedule -= next_scan_line.rows_scheduled as u64;
            trace!(
                "Scheduled scan line of {} rows and {} decoders",
                next_scan_line.rows_scheduled,
                next_scan_line.decoders.len()
            );
            schedule_action(Ok(DecoderMessage {
                scheduled_so_far: num_rows_scheduled,
                decoders: next_scan_line.decoders,
            }));
        }

        trace!("Finished scheduling {} ranges", ranges.len());
    }

    // This method is similar to schedule_ranges but instead of
    // sending the decoders to a channel it collects them all into a vector
    pub fn schedule_ranges_to_vec(
        &mut self,
        ranges: &[Range<u64>],
        filter: &FilterExpression,
        io: Arc<dyn EncodingsIo>,
    ) -> Result<Vec<DecoderMessage>> {
        let mut decode_messages = Vec::new();
        self.do_schedule_ranges(ranges, filter, io, |msg| decode_messages.push(msg));
        decode_messages.into_iter().collect::<Result<Vec<_>>>()
    }

    /// Schedules the load of a multiple ranges of rows
    ///
    /// Ranges must be non-overlapping and in sorted order
    ///
    /// # Arguments
    ///
    /// * `ranges` - The ranges of rows to load
    /// * `sink` - A channel to send the decode tasks
    /// * `scheduler` An I/O scheduler to issue I/O requests
    #[instrument(skip_all)]
    pub fn schedule_ranges(
        &mut self,
        ranges: &[Range<u64>],
        filter: &FilterExpression,
        sink: mpsc::UnboundedSender<Result<DecoderMessage>>,
        scheduler: Arc<dyn EncodingsIo>,
    ) {
        self.do_schedule_ranges(ranges, filter, scheduler, |msg| {
            sink.send(msg).unwrap();
        })
    }

    /// Schedules the load of a range of rows
    ///
    /// # Arguments
    ///
    /// * `range` - The range of rows to load
    /// * `sink` - A channel to send the decode tasks
    /// * `scheduler` An I/O scheduler to issue I/O requests
    #[instrument(skip_all)]
    pub fn schedule_range(
        &mut self,
        range: Range<u64>,
        filter: &FilterExpression,
        sink: mpsc::UnboundedSender<Result<DecoderMessage>>,
        scheduler: Arc<dyn EncodingsIo>,
    ) {
        self.schedule_ranges(&[range.clone()], filter, sink, scheduler)
    }

    /// Schedules the load of selected rows
    ///
    /// # Arguments
    ///
    /// * `indices` - The row indices to load (these must be in ascending order!)
    /// * `sink` - A channel to send the decode tasks
    /// * `scheduler` An I/O scheduler to issue I/O requests
    pub fn schedule_take(
        &mut self,
        indices: &[u64],
        filter: &FilterExpression,
        sink: mpsc::UnboundedSender<Result<DecoderMessage>>,
        scheduler: Arc<dyn EncodingsIo>,
    ) {
        debug_assert!(indices.windows(2).all(|w| w[0] < w[1]));
        if indices.is_empty() {
            return;
        }
        trace!("Scheduling take of {} rows", indices.len());
        let ranges = indices
            .iter()
            .map(|&idx| idx..(idx + 1))
            .collect::<Vec<_>>();
        self.schedule_ranges(&ranges, filter, sink, scheduler)
    }

    pub fn new_root_decoder_ranges(&self, ranges: &[Range<u64>]) -> SimpleStructDecoder {
        let rows_to_read = ranges
            .iter()
            .map(|range| range.end - range.start)
            .sum::<u64>();
        SimpleStructDecoder::new(self.root_fields.clone(), rows_to_read)
    }

    pub fn new_root_decoder_indices(&self, indices: &[u64]) -> SimpleStructDecoder {
        SimpleStructDecoder::new(self.root_fields.clone(), indices.len() as u64)
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
    scheduler_exhuasted: bool,
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
        scheduled: mpsc::UnboundedReceiver<Result<DecoderMessage>>,
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
            scheduler_exhuasted: false,
        }
    }

    fn accept_decoder(&mut self, decoder: DecoderReady) -> Result<()> {
        if decoder.path.is_empty() {
            // The root decoder we can ignore
            Ok(())
        } else {
            self.root_decoder.accept_child(decoder)
        }
    }

    async fn wait_for_scheduled(&mut self, scheduled_need: u64) -> Result<u64> {
        if self.scheduler_exhuasted {
            return Ok(self.rows_scheduled);
        }
        while self.rows_scheduled < scheduled_need {
            let next_message = self.context.source.recv().await;
            match next_message {
                Some(scan_line) => {
                    let scan_line = scan_line?;
                    self.rows_scheduled = scan_line.scheduled_so_far;
                    for decoder in scan_line.decoders {
                        self.accept_decoder(decoder)?;
                    }
                }
                None => {
                    // Schedule ended before we got all the data we expected.  This probably
                    // means some kind of pushdown filter was applied and we didn't load as
                    // much data as we thought we would.
                    self.scheduler_exhuasted = true;
                    return Ok(self.rows_scheduled);
                }
            }
        }
        Ok(scheduled_need)
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

        let mut to_take = self.rows_remaining.min(self.rows_per_batch as u64) as u32;
        self.rows_remaining -= to_take as u64;

        let scheduled_need =
            (self.rows_drained + to_take as u64).saturating_sub(self.rows_scheduled);
        trace!("scheduled_need = {} because rows_drained = {} and to_take = {} and rows_scheduled = {}", scheduled_need, self.rows_drained, to_take, self.rows_scheduled);
        if scheduled_need > 0 {
            let desired_scheduled = scheduled_need + self.rows_scheduled;
            trace!(
                "Draining from scheduler (desire at least {} scheduled rows)",
                desired_scheduled
            );
            let actually_scheduled = self.wait_for_scheduled(desired_scheduled).await?;
            if actually_scheduled < desired_scheduled {
                let under_scheduled = desired_scheduled - actually_scheduled;
                to_take -= under_scheduled as u32;
            }
        }

        if to_take == 0 {
            return Ok(None);
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
pub trait PrimitivePageDecoder: Send + Sync {
    /// Decode data into buffers
    ///
    /// This may be a simple zero-copy from a disk buffer or could involve complex decoding
    /// such as decompressing from some compressed representation.
    ///
    /// Capacity is stored as a tuple of (num_bytes: u64, is_needed: bool).  The `is_needed`
    /// portion only needs to be updated if the encoding has some concept of an "optional"
    /// buffer.
    ///
    /// Encodings can have any number of input or output buffers.  For example, a dictionary
    /// decoding will convert two buffers (indices + dictionary) into a single buffer
    ///
    /// Binary decodings have two output buffers (one for values, one for offsets)
    ///
    /// Other decodings could even expand the # of output buffers.  For example, we could decode
    /// fixed size strings into variable length strings going from one input buffer to multiple output
    /// buffers.
    ///
    /// Each Arrow data type typically has a fixed structure of buffers and the encoding chain will
    /// generally end at one of these structures.  However, intermediate structures may exist which
    /// do not correspond to any Arrow type at all.  For example, a bitpacking encoding will deal
    /// with buffers that have bits-per-value that is not a multiple of 8.
    ///
    /// The `primitive_array_from_buffers` method has an expected buffer layout for each arrow
    /// type (order matters) and encodings that aim to decode into arrow types should respect
    /// this layout.
    /// # Arguments
    ///
    /// * `rows_to_skip` - how many rows to skip (within the page) before decoding
    /// * `num_rows` - how many rows to decode
    /// * `all_null` - A mutable bool, set to true if a decoder determines all values are null
    fn decode(
        &self,
        rows_to_skip: u32,
        num_rows: u32,
        all_null: &mut bool,
    ) -> Result<Vec<BytesMut>>;
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

pub trait PageScheduler: Send + Sync + std::fmt::Debug {
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
        scheduler: &Arc<dyn EncodingsIo>,
        top_level_row: u64,
    ) -> BoxFuture<'static, Result<Box<dyn PrimitivePageDecoder>>>;
}

/// Contains the context for a scheduler
pub struct SchedulerContext {
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
    pub fn new(io: Arc<dyn EncodingsIo>) -> Self {
        Self {
            io,
            recv: None,
            name: "".to_string(),
            path: Vec::new(),
            path_names: Vec::new(),
        }
    }

    pub fn io(&self) -> &Arc<dyn EncodingsIo> {
        &self.io
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

    pub fn locate_decoder(&mut self, decoder: Box<dyn LogicalPageDecoder>) -> DecoderReady {
        trace!(
            "Scheduling decoder of type {:?} for {:?}",
            decoder.data_type(),
            self.path,
        );
        DecoderReady {
            decoder,
            path: VecDeque::from_iter(self.path.iter().copied()),
        }
    }
}

#[derive(Debug)]
pub struct ScheduledScanLine {
    pub rows_scheduled: u32,
    pub decoders: Vec<DecoderReady>,
}

pub trait SchedulingJob: std::fmt::Debug {
    fn schedule_next(
        &mut self,
        context: &mut SchedulerContext,
        top_level_row: u64,
    ) -> Result<ScheduledScanLine>;

    fn num_rows(&self) -> u64;
}

/// A filter expression to apply to the data
///
/// The core decoders do not currently take advantage of filtering in
/// any way.  In order to maintain the abstraction we represent filters
/// as an arbitrary byte sequence.
///
/// We recommend that encodings use Substrait for filters.
pub struct FilterExpression(pub Bytes);

impl FilterExpression {
    /// Create a filter expression that does not filter any data
    ///
    /// This is currently represented by an empty byte array.  Encoders
    /// that are "filter aware" should make sure they handle this case.
    pub fn no_filter() -> Self {
        Self(Bytes::new())
    }
}

/// A scheduler for a field's worth of data
///
/// Each field in a reader's output schema maps to one field scheduler.  This scheduler may
/// map to more than one column.  For example, one field of struct data may
/// cover many columns of child data.  In fact, the entire file is treated as one
/// top-level struct field.
///
/// The scheduler is responsible for calculating the neccesary I/O.  One schedule_range
/// request could trigger mulitple batches of I/O across multiple columns.  The scheduler
/// should emit decoders into the sink as quickly as possible.
///
/// As soon as the scheduler encounters a batch of data that can decoded then the scheduler
/// should emit a decoder in the "unloaded" state.  The decode stream will pull the decoder
/// and start decoding.
///
/// The order in which decoders are emitted is important.  Pages should be emitted in
/// row-major order allowing decode of complete rows as quickly as possible.
///
/// The `FieldScheduler` should be stateless and `Send` and `Sync`.  This is
/// because it might need to be shared.  For example, a list page has a reference to
/// the field schedulers for its items column.  This is shared with the follow-up I/O
/// task created when the offsets are loaded.
///
/// See [`crate::decoder`] for more information
pub trait FieldScheduler: Send + Sync + std::fmt::Debug {
    /// Schedules I/O for the requested portions of the field.
    ///
    /// Note: `ranges` must be ordered and non-overlapping
    /// TODO: Support unordered or overlapping ranges in file scheduler
    fn schedule_ranges<'a>(
        &'a self,
        ranges: &[Range<u64>],
        filter: &FilterExpression,
    ) -> Result<Box<dyn SchedulingJob + 'a>>;
    /// The number of rows in this field
    fn num_rows(&self) -> u64;
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

#[derive(Debug)]
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

pub struct DecoderMessage {
    pub scheduled_so_far: u64,
    pub decoders: Vec<DecoderReady>,
}

pub struct DecoderContext {
    source: mpsc::UnboundedReceiver<Result<DecoderMessage>>,
}

impl DecoderContext {
    pub fn new(source: mpsc::UnboundedReceiver<Result<DecoderMessage>>) -> Self {
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
pub async fn decode_batch(
    batch: &EncodedBatch,
    filter: &FilterExpression,
    field_decoder_strategy: &DecoderMiddlewareChain,
) -> Result<RecordBatch> {
    let io_scheduler = Arc::new(BufferScheduler::new(batch.data.clone())) as Arc<dyn EncodingsIo>;
    let mut decode_scheduler = DecodeBatchScheduler::try_new(
        batch.schema.as_ref(),
        &batch.page_table,
        &vec![],
        batch.num_rows,
        field_decoder_strategy,
        &io_scheduler,
    )
    .await?;
    let (tx, rx) = unbounded_channel();
    decode_scheduler.schedule_range(0..batch.num_rows, filter, tx, io_scheduler);
    #[allow(clippy::single_range_in_vec_init)]
    let root_decoder = decode_scheduler.new_root_decoder_ranges(&[0..batch.num_rows]);
    let stream = BatchDecodeStream::new(rx, batch.num_rows as u32, batch.num_rows, root_decoder);
    stream.into_stream().next().await.unwrap().task.await
}
