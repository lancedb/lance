// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::VecDeque, sync::Arc, vec};

use arrow::{array::AsArray, datatypes::UInt64Type};
use arrow_array::{Array, ArrayRef, LargeBinaryArray, PrimitiveArray, StructArray, UInt64Array};
use arrow_buffer::{
    BooleanBuffer, BooleanBufferBuilder, Buffer, NullBuffer, OffsetBuffer, ScalarBuffer,
};
use arrow_schema::{DataType, Field, Fields};
use bytes::Bytes;
use futures::{future::BoxFuture, FutureExt};
use snafu::{location, Location};

use lance_core::{Error, Result};

use crate::{
    buffer::LanceBuffer,
    decoder::{
        DecodeArrayTask, DecoderReady, FieldScheduler, FilterExpression, LogicalPageDecoder,
        MessageType, NextDecodeTask, PriorityRange, ScheduledScanLine, SchedulerContext,
        SchedulingJob,
    },
    encoder::{EncodeTask, FieldEncoder, OutOfLineBuffers},
    format::pb::{column_encoding, Blob, ColumnEncoding},
    repdef::RepDefBuilder,
    EncodingsIo,
};

lazy_static::lazy_static! {
    static ref DESC_FIELDS: Fields =
        Fields::from(vec![
            Field::new("position", DataType::UInt64, false),
            Field::new("size", DataType::UInt64, false),
        ]);
    pub static ref DESC_FIELD: Field =
        Field::new("description", DataType::Struct(DESC_FIELDS.clone()), false);
    pub static ref DESC_FIELD_LANCE: lance_core::datatypes::Field = (&*DESC_FIELD).try_into().unwrap();
}

/// A field scheduler for large binary data
///
/// Large binary data (1MiB+) can be inefficient if we store as a regular primtive.  We
/// essentially end up with 1 page per row (or a few rows) and the overhead of the
/// metadata can be significant.
///
/// At the same time the benefits of using pages (contiguous arrays) are pretty small since
/// we can generally perform random access at these sizes without much penalty.
///
/// This encoder gives up the random access and stores the large binary data out of line.  This
/// keeps the metadata small.
#[derive(Debug)]
pub struct BlobFieldScheduler {
    descriptions_scheduler: Arc<dyn FieldScheduler>,
}

impl BlobFieldScheduler {
    pub fn new(descriptions_scheduler: Arc<dyn FieldScheduler>) -> Self {
        Self {
            descriptions_scheduler,
        }
    }
}

#[derive(Debug)]
struct BlobFieldSchedulingJob<'a> {
    descriptions_job: Box<dyn SchedulingJob + 'a>,
}

impl<'a> SchedulingJob for BlobFieldSchedulingJob<'a> {
    fn schedule_next(
        &mut self,
        context: &mut SchedulerContext,
        priority: &dyn PriorityRange,
    ) -> Result<ScheduledScanLine> {
        let next_descriptions = self.descriptions_job.schedule_next(context, priority)?;
        let mut priority = priority.current_priority();
        let decoders = next_descriptions.decoders.into_iter().map(|decoder| {
            let decoder = decoder.into_legacy();
            let path = decoder.path;
            let mut decoder = decoder.decoder;
            let num_rows = decoder.num_rows();
            let descriptions_fut = async move {
                decoder
                    .wait_for_loaded(decoder.num_rows() - 1)
                    .await
                    .unwrap();
                let descriptions_task = decoder.drain(decoder.num_rows()).unwrap();
                descriptions_task.task.decode()
            }
            .boxed();
            let decoder = Box::new(BlobFieldDecoder {
                io: context.io().clone(),
                unloaded_descriptions: Some(descriptions_fut),
                positions: PrimitiveArray::<UInt64Type>::from_iter_values(vec![]),
                sizes: PrimitiveArray::<UInt64Type>::from_iter_values(vec![]),
                num_rows,
                loaded: VecDeque::new(),
                validity: VecDeque::new(),
                rows_loaded: 0,
                rows_drained: 0,
                base_priority: priority,
            });
            priority += num_rows;
            MessageType::DecoderReady(DecoderReady { decoder, path })
        });
        Ok(ScheduledScanLine {
            decoders: decoders.collect(),
            rows_scheduled: next_descriptions.rows_scheduled,
        })
    }

    fn num_rows(&self) -> u64 {
        self.descriptions_job.num_rows()
    }
}

impl FieldScheduler for BlobFieldScheduler {
    fn schedule_ranges<'a>(
        &'a self,
        ranges: &[std::ops::Range<u64>],
        filter: &FilterExpression,
    ) -> Result<Box<dyn SchedulingJob + 'a>> {
        let descriptions_job = self
            .descriptions_scheduler
            .schedule_ranges(ranges, filter)?;
        Ok(Box::new(BlobFieldSchedulingJob { descriptions_job }))
    }

    fn num_rows(&self) -> u64 {
        self.descriptions_scheduler.num_rows()
    }

    fn initialize<'a>(
        &'a self,
        filter: &'a FilterExpression,
        context: &'a SchedulerContext,
    ) -> BoxFuture<'a, Result<()>> {
        self.descriptions_scheduler.initialize(filter, context)
    }
}

pub struct BlobFieldDecoder {
    io: Arc<dyn EncodingsIo>,
    unloaded_descriptions: Option<BoxFuture<'static, Result<ArrayRef>>>,
    positions: PrimitiveArray<UInt64Type>,
    sizes: PrimitiveArray<UInt64Type>,
    num_rows: u64,
    loaded: VecDeque<Bytes>,
    validity: VecDeque<BooleanBuffer>,
    rows_loaded: u64,
    rows_drained: u64,
    base_priority: u64,
}

impl BlobFieldDecoder {
    fn drain_validity(&mut self, num_values: usize) -> Result<Option<NullBuffer>> {
        let mut validity = BooleanBufferBuilder::new(num_values);
        let mut remaining = num_values;
        while remaining > 0 {
            let next = self.validity.front_mut().unwrap();
            if remaining < next.len() {
                let slice = next.slice(0, remaining);
                validity.append_buffer(&slice);
                *next = next.slice(remaining, next.len() - remaining);
                remaining = 0;
            } else {
                validity.append_buffer(next);
                remaining -= next.len();
                self.validity.pop_front();
            }
        }
        let nulls = NullBuffer::new(validity.finish());
        if nulls.null_count() == 0 {
            Ok(None)
        } else {
            Ok(Some(nulls))
        }
    }
}

impl std::fmt::Debug for BlobFieldDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlobFieldDecoder")
            .field("num_rows", &self.num_rows)
            .field("rows_loaded", &self.rows_loaded)
            .field("rows_drained", &self.rows_drained)
            .finish()
    }
}

impl LogicalPageDecoder for BlobFieldDecoder {
    fn wait_for_loaded(&mut self, loaded_need: u64) -> BoxFuture<Result<()>> {
        async move {
            if self.unloaded_descriptions.is_some() {
                let descriptions = self.unloaded_descriptions.take().unwrap().await?;
                let descriptions = descriptions.as_struct();
                self.positions = descriptions.column(0).as_primitive().clone();
                self.sizes = descriptions.column(1).as_primitive().clone();
            }
            let start = self.rows_loaded as usize;
            let end = (loaded_need + 1).min(self.num_rows) as usize;
            let positions = self.positions.values().slice(start, end - start);
            let sizes = self.sizes.values().slice(start, end - start);
            let ranges = positions
                .iter()
                .zip(sizes.iter())
                .map(|(position, size)| *position..(*position + *size))
                .collect::<Vec<_>>();
            let validity = positions
                .iter()
                .zip(sizes.iter())
                .map(|(p, s)| *p != 1 || *s != 0)
                .collect::<BooleanBuffer>();
            self.validity.push_back(validity);
            self.rows_loaded = end as u64;
            let bytes = self
                .io
                .submit_request(ranges, self.base_priority + start as u64)
                .await?;
            self.loaded.extend(bytes);
            Ok(())
        }
        .boxed()
    }

    fn rows_loaded(&self) -> u64 {
        self.rows_loaded
    }

    fn num_rows(&self) -> u64 {
        self.num_rows
    }

    fn rows_drained(&self) -> u64 {
        self.rows_drained
    }

    fn drain(&mut self, num_rows: u64) -> Result<NextDecodeTask> {
        let bytes = self.loaded.drain(0..num_rows as usize).collect::<Vec<_>>();
        let validity = self.drain_validity(num_rows as usize)?;
        self.rows_drained += num_rows;
        Ok(NextDecodeTask {
            has_more: self.rows_drained < self.num_rows,
            num_rows,
            task: Box::new(BlobArrayDecodeTask::new(bytes, validity)),
        })
    }

    fn data_type(&self) -> &DataType {
        &DataType::LargeBinary
    }
}

struct BlobArrayDecodeTask {
    bytes: Vec<Bytes>,
    validity: Option<NullBuffer>,
}

impl BlobArrayDecodeTask {
    fn new(bytes: Vec<Bytes>, validity: Option<NullBuffer>) -> Self {
        Self { bytes, validity }
    }
}

impl DecodeArrayTask for BlobArrayDecodeTask {
    fn decode(self: Box<Self>) -> Result<ArrayRef> {
        let num_bytes = self.bytes.iter().map(|b| b.len()).sum::<usize>();
        let offsets = self
            .bytes
            .iter()
            .scan(0, |state, b| {
                let start = *state;
                *state += b.len();
                Some(start as i64)
            })
            .chain(std::iter::once(num_bytes as i64))
            .collect::<Vec<_>>();
        let offsets = OffsetBuffer::new(ScalarBuffer::from(offsets));
        let mut buffer = Vec::with_capacity(num_bytes);
        for bytes in self.bytes {
            buffer.extend_from_slice(&bytes);
        }
        let data_buf = Buffer::from_vec(buffer);
        Ok(Arc::new(LargeBinaryArray::new(
            offsets,
            data_buf,
            self.validity,
        )))
    }
}

// impl DecodeArrayTask for BlobFieldDecodeTask {
//     fn decode(self: Box<Self>) -> Result<ArrayRef> {
//     }
// }

// impl LogicalPageDecoder for PrimitiveFieldDecoder {
//     // TODO: In the future, at some point, we may consider partially waiting for primitive pages by
//     // breaking up large I/O into smaller I/O as a way to accelerate the "time-to-first-decode"
//     fn wait_for_loaded(&mut self, loaded_need: u64) -> BoxFuture<Result<()>> {
//     }

//     fn drain(&mut self, num_rows: u64) -> Result<NextDecodeTask> {
//     }

//     fn rows_loaded(&self) -> u64 {
//     }

//     fn rows_drained(&self) -> u64 {
//     }

//     fn num_rows(&self) -> u64 {
//     }

//     fn data_type(&self) -> &DataType {
//     }
// }

pub struct BlobFieldEncoder {
    description_encoder: Box<dyn FieldEncoder>,
}

impl BlobFieldEncoder {
    pub fn new(description_encoder: Box<dyn FieldEncoder>) -> Self {
        Self {
            description_encoder,
        }
    }

    fn write_bins(array: ArrayRef, external_buffers: &mut OutOfLineBuffers) -> Result<ArrayRef> {
        let binarray = array
            .as_binary_opt::<i64>()
            .ok_or_else(|| Error::InvalidInput {
                source: format!("Expected large_binary and received {}", array.data_type()).into(),
                location: location!(),
            })?;
        let mut positions = Vec::with_capacity(array.len());
        let mut sizes = Vec::with_capacity(array.len());
        let data = binarray.values();
        let nulls = binarray
            .nulls()
            .cloned()
            .unwrap_or(NullBuffer::new_valid(binarray.len()));
        for (w, is_valid) in binarray.value_offsets().windows(2).zip(nulls.into_iter()) {
            if is_valid {
                let start = w[0] as u64;
                let end = w[1] as u64;
                let size = end - start;
                if size > 0 {
                    let val = data.slice_with_length(start as usize, size as usize);
                    let position = external_buffers.add_buffer(LanceBuffer::Borrowed(val));
                    positions.push(position);
                    sizes.push(size);
                } else {
                    // Empty values are always (0,0)
                    positions.push(0);
                    sizes.push(0);
                }
            } else {
                // Null values are always (1, 0)
                positions.push(1);
                sizes.push(0);
            }
        }
        let positions = Arc::new(UInt64Array::from(positions));
        let sizes = Arc::new(UInt64Array::from(sizes));
        let descriptions = Arc::new(StructArray::new(
            DESC_FIELDS.clone(),
            vec![positions, sizes],
            None,
        ));
        Ok(descriptions)
    }
}

impl FieldEncoder for BlobFieldEncoder {
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
        external_buffers: &mut OutOfLineBuffers,
        repdef: RepDefBuilder,
        row_number: u64,
    ) -> Result<Vec<EncodeTask>> {
        let descriptions = Self::write_bins(array, external_buffers)?;
        self.description_encoder
            .maybe_encode(descriptions, external_buffers, repdef, row_number)
    }

    // If there is any data left in the buffer then create an encode task from it
    fn flush(&mut self, external_buffers: &mut OutOfLineBuffers) -> Result<Vec<EncodeTask>> {
        self.description_encoder.flush(external_buffers)
    }

    fn num_columns(&self) -> u32 {
        self.description_encoder.num_columns()
    }

    fn finish(
        &mut self,
        external_buffers: &mut OutOfLineBuffers,
    ) -> BoxFuture<'_, Result<Vec<crate::encoder::EncodedColumn>>> {
        let inner_finished = self.description_encoder.finish(external_buffers);
        async move {
            let mut cols = inner_finished.await?;
            assert_eq!(cols.len(), 1);
            let encoding = std::mem::take(&mut cols[0].encoding);
            let wrapped_encoding = ColumnEncoding {
                column_encoding: Some(column_encoding::ColumnEncoding::Blob(Box::new(Blob {
                    inner: Some(Box::new(encoding)),
                }))),
            };
            cols[0].encoding = wrapped_encoding;
            Ok(cols)
        }
        .boxed()
    }
}

#[cfg(test)]
pub mod tests {
    use std::{collections::HashMap, sync::Arc};

    use arrow_array::LargeBinaryArray;
    use arrow_schema::{DataType, Field};

    use crate::{
        encoder::BLOB_META_KEY,
        format::pb::column_encoding,
        testing::{check_round_trip_encoding_of_data, check_round_trip_encoding_random, TestCases},
        version::LanceFileVersion,
    };

    lazy_static::lazy_static! {
    static ref BLOB_META: HashMap<String, String> =
        [(BLOB_META_KEY.to_string(), "true".to_string())]
            .iter()
            .cloned()
            .collect::<HashMap<_, _>>();
    }

    #[test_log::test(tokio::test)]
    async fn test_blob() {
        let field = Field::new("", DataType::LargeBinary, false).with_metadata(BLOB_META.clone());
        check_round_trip_encoding_random(field, LanceFileVersion::V2_0).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_simple_blob() {
        let val1: &[u8] = &[1, 2, 3];
        let val2: &[u8] = &[7, 8, 9];
        let array = Arc::new(LargeBinaryArray::from(vec![Some(val1), None, Some(val2)]));
        let test_cases = TestCases::default().with_verify_encoding(Arc::new(|cols| {
            assert_eq!(cols.len(), 1);
            let col = &cols[0];
            assert!(matches!(
                col.encoding.column_encoding.as_ref().unwrap(),
                column_encoding::ColumnEncoding::Blob(_)
            ));
        }));
        // Use blob encoding if requested
        check_round_trip_encoding_of_data(vec![array.clone()], &test_cases, BLOB_META.clone())
            .await;

        let test_cases = TestCases::default().with_verify_encoding(Arc::new(|cols| {
            assert_eq!(cols.len(), 1);
            let col = &cols[0];
            assert!(!matches!(
                col.encoding.column_encoding.as_ref().unwrap(),
                column_encoding::ColumnEncoding::Blob(_)
            ));
        }));
        // Don't use blob encoding if not requested
        check_round_trip_encoding_of_data(vec![array], &test_cases, Default::default()).await;
    }
}
