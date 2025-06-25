// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::BinaryHeap, ops::Range, sync::Arc};

use crate::{
    decoder::{
        DecodedArray, FilterExpression, LoadedPage, NextDecodeTask, PageEncoding,
        ScheduledScanLine, SchedulerContext, StructuralDecodeArrayTask, StructuralFieldDecoder,
        StructuralFieldScheduler, StructuralSchedulingJob,
    },
    encoder::{EncodeTask, EncodedColumn, EncodedPage, FieldEncoder, OutOfLineBuffers},
    format::pb,
    repdef::RepDefBuilder,
};
use arrow_array::{cast::AsArray, Array, ArrayRef, StructArray};
use arrow_schema::{DataType, Fields};
use futures::{
    future::BoxFuture,
    stream::{FuturesOrdered, FuturesUnordered},
    FutureExt, StreamExt, TryStreamExt,
};
use itertools::Itertools;
use lance_arrow::deepcopy::deep_copy_nulls;
use lance_arrow::FieldExt;
use lance_core::Result;
use log::trace;

use super::{list::StructuralListDecoder, primitive::StructuralPrimitiveFieldDecoder};

#[derive(Debug)]
struct StructuralSchedulingJobWithStatus<'a> {
    col_idx: u32,
    col_name: &'a str,
    job: Box<dyn StructuralSchedulingJob + 'a>,
    rows_scheduled: u64,
    rows_remaining: u64,
}

impl PartialEq for StructuralSchedulingJobWithStatus<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.col_idx == other.col_idx
    }
}

impl Eq for StructuralSchedulingJobWithStatus<'_> {}

impl PartialOrd for StructuralSchedulingJobWithStatus<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for StructuralSchedulingJobWithStatus<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Note this is reversed to make it min-heap
        other.rows_scheduled.cmp(&self.rows_scheduled)
    }
}

/// Scheduling job for struct data
///
/// The order in which we schedule the children is important.  We want to schedule the child
/// with the least amount of data first.
///
/// This allows us to decode entire rows as quickly as possible
#[derive(Debug)]
struct RepDefStructSchedulingJob<'a> {
    /// A min-heap whose key is the # of rows currently scheduled
    children: BinaryHeap<StructuralSchedulingJobWithStatus<'a>>,
    rows_scheduled: u64,
}

impl<'a> RepDefStructSchedulingJob<'a> {
    fn new(
        scheduler: &'a StructuralStructScheduler,
        children: Vec<Box<dyn StructuralSchedulingJob + 'a>>,
        num_rows: u64,
    ) -> Self {
        let children = children
            .into_iter()
            .enumerate()
            .map(|(idx, job)| StructuralSchedulingJobWithStatus {
                col_idx: idx as u32,
                col_name: scheduler.child_fields[idx].name(),
                job,
                rows_scheduled: 0,
                rows_remaining: num_rows,
            })
            .collect::<BinaryHeap<_>>();
        Self {
            children,
            rows_scheduled: 0,
        }
    }
}

impl StructuralSchedulingJob for RepDefStructSchedulingJob<'_> {
    fn schedule_next(
        &mut self,
        mut context: &mut SchedulerContext,
    ) -> Result<Option<ScheduledScanLine>> {
        let mut decoders = Vec::new();
        let old_rows_scheduled = self.rows_scheduled;
        // Schedule as many children as we need to until we have scheduled at least one
        // complete row
        while old_rows_scheduled == self.rows_scheduled {
            let mut next_child = self.children.pop().unwrap();
            let scoped = context.push(next_child.col_name, next_child.col_idx);
            let child_scan = next_child.job.schedule_next(scoped.context)?;
            // next_child is the least-scheduled child and, if it's done, that
            // means we are completely done.
            if child_scan.is_none() {
                return Ok(None);
            }
            let child_scan = child_scan.unwrap();

            trace!(
                "Scheduled {} rows for child {}",
                child_scan.rows_scheduled,
                next_child.col_idx
            );
            next_child.rows_scheduled += child_scan.rows_scheduled;
            next_child.rows_remaining -= child_scan.rows_scheduled;
            decoders.extend(child_scan.decoders);
            self.children.push(next_child);
            self.rows_scheduled = self.children.peek().unwrap().rows_scheduled;
            context = scoped.pop();
        }
        let struct_rows_scheduled = self.rows_scheduled - old_rows_scheduled;
        Ok(Some(ScheduledScanLine {
            decoders,
            rows_scheduled: struct_rows_scheduled,
        }))
    }
}

/// A scheduler for structs
///
/// The implementation is actually a bit more tricky than one might initially think.  We can't just
/// go through and schedule each column one after the other.  This would mean our decode can't start
/// until nearly all the data has arrived (since we need data from each column to yield a batch)
///
/// Instead, we schedule in row-major fashion
///
/// Note: this scheduler is the starting point for all decoding.  This is because we treat the top-level
/// record batch as a non-nullable struct.
#[derive(Debug)]
pub struct StructuralStructScheduler {
    children: Vec<Box<dyn StructuralFieldScheduler>>,
    child_fields: Fields,
}

impl StructuralStructScheduler {
    pub fn new(children: Vec<Box<dyn StructuralFieldScheduler>>, child_fields: Fields) -> Self {
        debug_assert!(!children.is_empty());
        Self {
            children,
            child_fields,
        }
    }
}

impl StructuralFieldScheduler for StructuralStructScheduler {
    fn schedule_ranges<'a>(
        &'a self,
        ranges: &[Range<u64>],
        filter: &FilterExpression,
    ) -> Result<Box<dyn StructuralSchedulingJob + 'a>> {
        let num_rows = ranges.iter().map(|r| r.end - r.start).sum();

        let child_schedulers = self
            .children
            .iter()
            .map(|child| child.schedule_ranges(ranges, filter))
            .collect::<Result<Vec<_>>>()?;

        Ok(Box::new(RepDefStructSchedulingJob::new(
            self,
            child_schedulers,
            num_rows,
        )))
    }

    fn initialize<'a>(
        &'a mut self,
        filter: &'a FilterExpression,
        context: &'a SchedulerContext,
    ) -> BoxFuture<'a, Result<()>> {
        let children_initialization = self
            .children
            .iter_mut()
            .map(|child| child.initialize(filter, context))
            .collect::<FuturesUnordered<_>>();
        async move {
            children_initialization
                .map(|res| res.map(|_| ()))
                .try_collect::<Vec<_>>()
                .await?;
            Ok(())
        }
        .boxed()
    }
}

#[derive(Debug)]
pub struct StructuralStructDecoder {
    children: Vec<Box<dyn StructuralFieldDecoder>>,
    data_type: DataType,
    child_fields: Fields,
    // The root decoder is slightly different because it cannot have nulls
    is_root: bool,
}

impl StructuralStructDecoder {
    pub fn new(fields: Fields, should_validate: bool, is_root: bool) -> Self {
        let children = fields
            .iter()
            .map(|field| Self::field_to_decoder(field, should_validate))
            .collect();
        let data_type = DataType::Struct(fields.clone());
        Self {
            data_type,
            children,
            child_fields: fields,
            is_root,
        }
    }

    fn field_to_decoder(
        field: &Arc<arrow_schema::Field>,
        should_validate: bool,
    ) -> Box<dyn StructuralFieldDecoder> {
        match field.data_type() {
            DataType::Struct(fields) => {
                if field.is_packed_struct() {
                    let decoder =
                        StructuralPrimitiveFieldDecoder::new(&field.clone(), should_validate);
                    Box::new(decoder)
                } else {
                    Box::new(Self::new(fields.clone(), should_validate, false))
                }
            }
            DataType::List(child_field) | DataType::LargeList(child_field) => {
                let child_decoder = Self::field_to_decoder(child_field, should_validate);
                Box::new(StructuralListDecoder::new(
                    child_decoder,
                    field.data_type().clone(),
                ))
            }
            DataType::RunEndEncoded(_, _) => todo!(),
            DataType::ListView(_) | DataType::LargeListView(_) => todo!(),
            DataType::Map(_, _) => todo!(),
            DataType::Union(_, _) => todo!(),
            _ => Box::new(StructuralPrimitiveFieldDecoder::new(field, should_validate)),
        }
    }

    pub fn drain_batch_task(&mut self, num_rows: u64) -> Result<NextDecodeTask> {
        let array_drain = self.drain(num_rows)?;
        Ok(NextDecodeTask {
            num_rows,
            task: Box::new(array_drain),
        })
    }
}

impl StructuralFieldDecoder for StructuralStructDecoder {
    fn accept_page(&mut self, mut child: LoadedPage) -> Result<()> {
        // children with empty path should not be delivered to this method
        let child_idx = child.path.pop_front().unwrap();
        // This decoder is intended for one of our children
        self.children[child_idx as usize].accept_page(child)?;
        Ok(())
    }

    fn drain(&mut self, num_rows: u64) -> Result<Box<dyn StructuralDecodeArrayTask>> {
        let child_tasks = self
            .children
            .iter_mut()
            .map(|child| child.drain(num_rows))
            .collect::<Result<Vec<_>>>()?;
        Ok(Box::new(RepDefStructDecodeTask {
            children: child_tasks,
            child_fields: self.child_fields.clone(),
            is_root: self.is_root,
        }))
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }
}

#[derive(Debug)]
struct RepDefStructDecodeTask {
    children: Vec<Box<dyn StructuralDecodeArrayTask>>,
    child_fields: Fields,
    is_root: bool,
}

impl StructuralDecodeArrayTask for RepDefStructDecodeTask {
    fn decode(self: Box<Self>) -> Result<DecodedArray> {
        let arrays = self
            .children
            .into_iter()
            .map(|task| task.decode())
            .collect::<Result<Vec<_>>>()?;
        let mut children = Vec::with_capacity(arrays.len());
        let mut arrays_iter = arrays.into_iter();
        let first_array = arrays_iter.next().unwrap();
        let length = first_array.array.len();

        // The repdef should be identical across all children at this point
        let mut repdef = first_array.repdef;
        children.push(first_array.array);

        for array in arrays_iter {
            debug_assert_eq!(length, array.array.len());
            children.push(array.array);
        }

        let validity = if self.is_root {
            None
        } else {
            repdef.unravel_validity(length)
        };
        let array = StructArray::new(self.child_fields, children, validity);
        Ok(DecodedArray {
            array: Arc::new(array),
            repdef,
        })
    }
}

/// A structural encoder for struct fields
///
/// The struct's validity is added to the rep/def builder
/// and the builder is cloned to all children.
pub struct StructStructuralEncoder {
    keep_original_array: bool,
    children: Vec<Box<dyn FieldEncoder>>,
}

impl StructStructuralEncoder {
    pub fn new(keep_original_array: bool, children: Vec<Box<dyn FieldEncoder>>) -> Self {
        Self {
            keep_original_array,
            children,
        }
    }
}

impl FieldEncoder for StructStructuralEncoder {
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
        external_buffers: &mut OutOfLineBuffers,
        mut repdef: RepDefBuilder,
        row_number: u64,
        num_rows: u64,
    ) -> Result<Vec<EncodeTask>> {
        let struct_array = array.as_struct();
        if let Some(validity) = struct_array.nulls() {
            if self.keep_original_array {
                repdef.add_validity_bitmap(validity.clone())
            } else {
                repdef.add_validity_bitmap(deep_copy_nulls(Some(validity)).unwrap())
            }
        } else {
            repdef.add_no_null(struct_array.len());
        }
        let child_tasks = self
            .children
            .iter_mut()
            .zip(struct_array.columns().iter())
            .map(|(encoder, arr)| {
                encoder.maybe_encode(
                    arr.clone(),
                    external_buffers,
                    repdef.clone(),
                    row_number,
                    num_rows,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(child_tasks.into_iter().flatten().collect::<Vec<_>>())
    }

    fn flush(&mut self, external_buffers: &mut OutOfLineBuffers) -> Result<Vec<EncodeTask>> {
        self.children
            .iter_mut()
            .map(|encoder| encoder.flush(external_buffers))
            .flatten_ok()
            .collect::<Result<Vec<_>>>()
    }

    fn num_columns(&self) -> u32 {
        self.children
            .iter()
            .map(|child| child.num_columns())
            .sum::<u32>()
    }

    fn finish(
        &mut self,
        external_buffers: &mut OutOfLineBuffers,
    ) -> BoxFuture<'_, Result<Vec<crate::encoder::EncodedColumn>>> {
        let mut child_columns = self
            .children
            .iter_mut()
            .map(|child| child.finish(external_buffers))
            .collect::<FuturesOrdered<_>>();
        async move {
            let mut encoded_columns = Vec::with_capacity(child_columns.len());
            while let Some(child_cols) = child_columns.next().await {
                encoded_columns.extend(child_cols?);
            }
            Ok(encoded_columns)
        }
        .boxed()
    }
}

pub struct StructFieldEncoder {
    children: Vec<Box<dyn FieldEncoder>>,
    column_index: u32,
    num_rows_seen: u64,
}

impl StructFieldEncoder {
    #[allow(dead_code)]
    pub fn new(children: Vec<Box<dyn FieldEncoder>>, column_index: u32) -> Self {
        Self {
            children,
            column_index,
            num_rows_seen: 0,
        }
    }
}

impl FieldEncoder for StructFieldEncoder {
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
        external_buffers: &mut OutOfLineBuffers,
        repdef: RepDefBuilder,
        row_number: u64,
        num_rows: u64,
    ) -> Result<Vec<EncodeTask>> {
        self.num_rows_seen += array.len() as u64;
        let struct_array = array.as_struct();
        let child_tasks = self
            .children
            .iter_mut()
            .zip(struct_array.columns().iter())
            .map(|(encoder, arr)| {
                encoder.maybe_encode(
                    arr.clone(),
                    external_buffers,
                    repdef.clone(),
                    row_number,
                    num_rows,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(child_tasks.into_iter().flatten().collect::<Vec<_>>())
    }

    fn flush(&mut self, external_buffers: &mut OutOfLineBuffers) -> Result<Vec<EncodeTask>> {
        let child_tasks = self
            .children
            .iter_mut()
            .map(|encoder| encoder.flush(external_buffers))
            .collect::<Result<Vec<_>>>()?;
        Ok(child_tasks.into_iter().flatten().collect::<Vec<_>>())
    }

    fn num_columns(&self) -> u32 {
        self.children
            .iter()
            .map(|child| child.num_columns())
            .sum::<u32>()
            + 1
    }

    fn finish(
        &mut self,
        external_buffers: &mut OutOfLineBuffers,
    ) -> BoxFuture<'_, Result<Vec<crate::encoder::EncodedColumn>>> {
        let mut child_columns = self
            .children
            .iter_mut()
            .map(|child| child.finish(external_buffers))
            .collect::<FuturesOrdered<_>>();
        let num_rows_seen = self.num_rows_seen;
        let column_index = self.column_index;
        async move {
            let mut columns = Vec::new();
            // Add a column for the struct header
            let mut header = EncodedColumn::default();
            header.final_pages.push(EncodedPage {
                data: Vec::new(),
                description: PageEncoding::Legacy(pb::ArrayEncoding {
                    array_encoding: Some(pb::array_encoding::ArrayEncoding::Struct(
                        pb::SimpleStruct {},
                    )),
                }),
                num_rows: num_rows_seen,
                column_idx: column_index,
                row_number: 0, // Not used by legacy encoding
            });
            columns.push(header);
            // Now run finish on the children
            while let Some(child_cols) = child_columns.next().await {
                columns.extend(child_cols?);
            }
            Ok(columns)
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {

    use std::{collections::HashMap, sync::Arc};

    use arrow_array::{
        builder::{Int32Builder, ListBuilder},
        Array, ArrayRef, Int32Array, StructArray,
    };
    use arrow_buffer::NullBuffer;
    use arrow_schema::{DataType, Field, Fields};

    use crate::{
        testing::{check_round_trip_encoding_of_data, check_round_trip_encoding_random, TestCases},
        version::LanceFileVersion,
    };

    #[test_log::test(tokio::test)]
    async fn test_simple_struct() {
        let data_type = DataType::Struct(Fields::from(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let field = Field::new("", data_type, false);
        check_round_trip_encoding_random(field, LanceFileVersion::V2_0).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_nullable_struct() {
        // Test data struct<score: int32, location: struct<x: int32, y: int32>>
        // - score: null
        //   location:
        //     x: 1
        //     y: 6
        // - score: 12
        //   location:
        //     x: 2
        //     y: null
        // - score: 13
        //   location:
        //     x: 3
        //     y: 8
        // - score: 14
        //   location: null
        // - null
        //
        let inner_fields = Fields::from(vec![
            Field::new("x", DataType::Int32, false),
            Field::new("y", DataType::Int32, true),
        ]);
        let inner_struct = DataType::Struct(inner_fields.clone());
        let outer_fields = Fields::from(vec![
            Field::new("score", DataType::Int32, true),
            Field::new("location", inner_struct, true),
        ]);

        let x_vals = Int32Array::from(vec![Some(1), Some(2), Some(3), Some(4), Some(5)]);
        let y_vals = Int32Array::from(vec![Some(6), None, Some(8), Some(9), Some(10)]);
        let scores = Int32Array::from(vec![None, Some(12), Some(13), Some(14), Some(15)]);

        let location_validity = NullBuffer::from(vec![true, true, true, false, true]);
        let locations = StructArray::new(
            inner_fields,
            vec![Arc::new(x_vals), Arc::new(y_vals)],
            Some(location_validity),
        );

        let rows_validity = NullBuffer::from(vec![true, true, true, true, false]);
        let rows = StructArray::new(
            outer_fields,
            vec![Arc::new(scores), Arc::new(locations)],
            Some(rows_validity),
        );

        let test_cases = TestCases::default().with_file_version(LanceFileVersion::V2_1);

        check_round_trip_encoding_of_data(vec![Arc::new(rows)], &test_cases, HashMap::new()).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_struct_list() {
        let data_type = DataType::Struct(Fields::from(vec![
            Field::new(
                "inner_list",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                true,
            ),
            Field::new("outer_int", DataType::Int32, true),
        ]));
        let field = Field::new("row", data_type, false);
        check_round_trip_encoding_random(field, LanceFileVersion::V2_0).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_empty_struct() {
        // It's technically legal for a struct to have 0 children, need to
        // make sure we support that
        let data_type = DataType::Struct(Fields::from(Vec::<Field>::default()));
        let field = Field::new("row", data_type, false);
        check_round_trip_encoding_random(field, LanceFileVersion::V2_0).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_complicated_struct() {
        let data_type = DataType::Struct(Fields::from(vec![
            Field::new("int", DataType::Int32, true),
            Field::new(
                "inner",
                DataType::Struct(Fields::from(vec![
                    Field::new("inner_int", DataType::Int32, true),
                    Field::new(
                        "inner_list",
                        DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                        true,
                    ),
                ])),
                true,
            ),
            Field::new("outer_binary", DataType::Binary, true),
        ]));
        let field = Field::new("row", data_type, false);
        check_round_trip_encoding_random(field, LanceFileVersion::V2_0).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_ragged_scheduling() {
        // This test covers scheduling when batches straddle page boundaries

        // Create a list with 10k nulls
        let items_builder = Int32Builder::new();
        let mut list_builder = ListBuilder::new(items_builder);
        for _ in 0..10000 {
            list_builder.append_null();
        }
        let list_array = Arc::new(list_builder.finish());
        let int_array = Arc::new(Int32Array::from_iter_values(0..10000));
        let fields = vec![
            Field::new("", list_array.data_type().clone(), true),
            Field::new("", int_array.data_type().clone(), true),
        ];
        let struct_array = Arc::new(StructArray::new(
            Fields::from(fields),
            vec![list_array, int_array],
            None,
        )) as ArrayRef;
        let struct_arrays = (0..10000)
            // Intentionally skip in some randomish amount to create more ragged scheduling
            .step_by(437)
            .map(|offset| struct_array.slice(offset, 437.min(10000 - offset)))
            .collect::<Vec<_>>();
        check_round_trip_encoding_of_data(struct_arrays, &TestCases::default(), HashMap::new())
            .await;
    }
}
