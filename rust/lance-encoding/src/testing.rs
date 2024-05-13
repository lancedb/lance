// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{ops::Range, sync::Arc};

use arrow_array::{Array, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use arrow_select::concat::concat;
use bytes::{Bytes, BytesMut};
use futures::{future::BoxFuture, FutureExt, StreamExt};
use log::{debug, trace};
use tokio::sync::mpsc::{self, UnboundedSender};

use lance_core::Result;
use lance_datagen::{array, gen, RowCount, Seed};

use crate::{
    decoder::{BatchDecodeStream, ColumnInfo, DecodeBatchScheduler, LogicalPageDecoder, PageInfo},
    encoder::{BatchEncoder, EncodedPage, FieldEncoder},
    EncodingsIo,
};

pub(crate) struct SimulatedScheduler {
    data: Bytes,
}

impl SimulatedScheduler {
    pub fn new(data: Vec<EncodedPage>) -> Self {
        let mut bytes = BytesMut::new();
        for page in data.into_iter() {
            for buf in page.array.buffers {
                for part in buf.parts.into_iter() {
                    bytes.extend_from_slice(&part)
                }
            }
        }
        Self {
            data: bytes.freeze(),
        }
    }

    fn satisfy_request(&self, req: Range<u64>) -> Bytes {
        self.data.slice(req.start as usize..req.end as usize)
    }
}

impl EncodingsIo for SimulatedScheduler {
    fn submit_request(
        &self,
        ranges: Vec<Range<u64>>,
        _priority: u64,
    ) -> BoxFuture<'static, Result<Vec<Bytes>>> {
        std::future::ready(Ok(ranges
            .into_iter()
            .map(|range| self.satisfy_request(range))
            .collect::<Vec<_>>()))
        .boxed()
    }
}

async fn test_decode(
    num_rows: u64,
    schema: &Schema,
    column_infos: &[ColumnInfo],
    expected: Option<Arc<dyn Array>>,
    schedule_fn: impl FnOnce(
        DecodeBatchScheduler,
        UnboundedSender<Box<dyn LogicalPageDecoder>>,
    ) -> BoxFuture<'static, Result<()>>,
) {
    let decode_scheduler = DecodeBatchScheduler::new(schema, column_infos, &Vec::new());

    let (tx, rx) = mpsc::unbounded_channel();

    schedule_fn(decode_scheduler, tx).await.unwrap();

    const BATCH_SIZE: u32 = 100;
    let mut decode_stream = BatchDecodeStream::new(rx, BATCH_SIZE, num_rows).into_stream();

    let mut offset = 0;
    while let Some(batch) = decode_stream.next().await {
        let batch = batch.task.await.unwrap();
        if let Some(expected) = expected.as_ref() {
            let actual = batch.column(0);
            let expected_size = (BATCH_SIZE as usize).min(expected.len() - offset);
            let expected = expected.slice(offset, expected_size);
            assert_eq!(expected.data_type(), actual.data_type());
            assert_eq!(&expected, actual);
        }
        offset += BATCH_SIZE as usize;
    }
}

/// Given a field this will test the round trip encoding and decoding of random data
pub async fn check_round_trip_encoding_random(field: Field) {
    let lance_field = lance_core::datatypes::Field::try_from(&field).unwrap();
    for page_size in [4096, 1024 * 1024] {
        debug!("Testing random data with a page size of {}", page_size);
        let encoder_factory = || {
            let mut col_idx = 0;
            let mut field_id_to_col_index = Vec::new();
            BatchEncoder::get_encoder_for_field(
                &lance_field,
                page_size,
                /*keep_original_array=*/ true,
                &mut col_idx,
                &mut field_id_to_col_index,
            )
            .unwrap()
        };
        check_round_trip_field_encoding_random(encoder_factory, field.clone()).await
    }
}

fn supports_nulls(data_type: &DataType) -> bool {
    // We don't yet have nullability support for all types.  Don't test nullability for the
    // types we don't support.
    !matches!(data_type, DataType::Struct(_))
}

// The default will just test the full read
#[derive(Clone, Default)]
pub struct TestCases {
    ranges: Vec<Range<u64>>,
    indices: Vec<Vec<u32>>,
    skip_validation: bool,
}

impl TestCases {
    pub fn with_range(mut self, range: Range<u64>) -> Self {
        self.ranges.push(range);
        self
    }

    pub fn with_indices(mut self, indices: Vec<u32>) -> Self {
        self.indices.push(indices);
        self
    }

    pub fn without_validation(mut self) -> Self {
        self.skip_validation = true;
        self
    }
}

/// Given specific data and test cases we check round trip encoding and decoding
///
/// Note that the input `data` is a `Vec` to simulate multiple calls to `maybe_encode`.
/// In other words, these are multiple chunks of one long array and not multiple columns
/// in a record batch.  To feed a "record batch" you should first convert the record batch
/// to a struct array.
pub async fn check_round_trip_encoding_of_data(data: Vec<Arc<dyn Array>>, test_cases: &TestCases) {
    let example_data = data.first().expect("Data must have at least one array");
    let field = Field::new("", example_data.data_type().clone(), true);
    let lance_field = lance_core::datatypes::Field::try_from(&field).unwrap();
    for page_size in [4096, 1024 * 1024] {
        let mut col_idx = 0;
        let mut field_id_to_col_index = Vec::new();
        let encoder = BatchEncoder::get_encoder_for_field(
            &lance_field,
            page_size,
            /*keep_original=*/ true,
            &mut col_idx,
            &mut field_id_to_col_index,
        )
        .unwrap();
        check_round_trip_encoding_inner(encoder, &field, data.clone(), test_cases).await
    }
}

/// This is the inner-most check function that actually runs the round trip and tests it
async fn check_round_trip_encoding_inner(
    mut encoder: Box<dyn FieldEncoder>,
    field: &Field,
    data: Vec<Arc<dyn Array>>,
    test_cases: &TestCases,
) {
    let mut all_encoded_pages = Vec::new();
    let mut page_infos: Vec<Vec<Arc<PageInfo>>> = vec![Vec::new(); encoder.num_columns() as usize];
    let mut buffer_offset = 0;

    let mut simulate_write = |mut encoded_page: EncodedPage| {
        trace!("Encoded page {:?}", encoded_page);
        encoded_page.array.buffers.sort_by_key(|b| b.index);
        let buffer_offsets = encoded_page
            .array
            .buffers
            .iter()
            .map(|buf| {
                let offset = buffer_offset;
                buffer_offset += buf.parts.iter().map(|part| part.len() as u64).sum::<u64>();
                offset
            })
            .collect::<Vec<_>>();

        let page_info = PageInfo {
            num_rows: encoded_page.num_rows,
            encoding: encoded_page.array.encoding.clone(),
            buffer_offsets: Arc::new(buffer_offsets.clone()),
        };

        let col_idx = encoded_page.column_idx as usize;
        all_encoded_pages.push(encoded_page);
        page_infos[col_idx].push(Arc::new(page_info));
    };

    for arr in &data {
        for encode_task in encoder.maybe_encode(arr.clone()).unwrap() {
            let encoded_page = encode_task.await.unwrap();
            simulate_write(encoded_page);
        }
    }

    for encode_task in encoder.flush().unwrap() {
        let encoded_page = encode_task.await.unwrap();
        simulate_write(encoded_page);
    }

    let scheduler = Arc::new(SimulatedScheduler::new(all_encoded_pages)) as Arc<dyn EncodingsIo>;

    let column_infos = page_infos
        .into_iter()
        .enumerate()
        .map(|(col_idx, page_infos)| ColumnInfo::new(col_idx as u32, page_infos, Vec::new()))
        .collect::<Vec<_>>();
    let schema = Schema::new(vec![field.clone()]);

    let num_rows = data.iter().map(|arr| arr.len() as u64).sum::<u64>();
    let concat_data = if test_cases.skip_validation {
        None
    } else {
        Some(concat(&data.iter().map(|arr| arr.as_ref()).collect::<Vec<_>>()).unwrap())
    };

    // We always try a full decode, regardless of the test cases provided
    debug!("Testing full decode");
    let scheduler_copy = scheduler.clone();
    test_decode(
        num_rows,
        &schema,
        &column_infos,
        concat_data.clone(),
        |mut decode_scheduler, tx| {
            async move {
                decode_scheduler
                    .schedule_range(0..num_rows, tx, &scheduler_copy)
                    .await
            }
            .boxed()
        },
    )
    .await;

    // Test range scheduling
    for range in &test_cases.ranges {
        debug!("Testing decode of range {:?}", range);
        let num_rows = range.end - range.start;
        let expected = concat_data
            .as_ref()
            .map(|concat_data| concat_data.slice(range.start as usize, num_rows as usize));
        let scheduler = scheduler.clone();
        let range = range.clone();
        test_decode(
            num_rows,
            &schema,
            &column_infos,
            expected,
            |mut decode_scheduler, tx| {
                async move { decode_scheduler.schedule_range(range, tx, &scheduler).await }.boxed()
            },
        )
        .await;
    }

    // Test take scheduling
    for indices in &test_cases.indices {
        if indices.len() == 1 {
            debug!("Testing decode of index {}", indices[0]);
        } else {
            debug!(
                "Testing decode of {} indices spread across range [{}..{}]",
                indices.len(),
                indices[0],
                indices[indices.len() - 1]
            );
        }
        let num_rows = indices.len() as u64;
        let indices_arr = UInt32Array::from(indices.clone());
        let expected = concat_data
            .as_ref()
            .map(|concat_data| arrow_select::take::take(&concat_data, &indices_arr, None).unwrap());
        let scheduler = scheduler.clone();
        let indices = indices.clone();
        test_decode(
            num_rows,
            &schema,
            &column_infos,
            expected,
            |mut decode_scheduler, tx| {
                async move {
                    decode_scheduler
                        .schedule_take(&indices, tx, &scheduler)
                        .await
                }
                .boxed()
            },
        )
        .await;
    }
}

const NUM_RANDOM_ROWS: u32 = 10000;

/// Generates random data (parameterized by null rate, slicing, and # ingest batches)
/// and tests with that.
async fn check_round_trip_field_encoding_random(
    encoder_factory: impl Fn() -> Box<dyn FieldEncoder>,
    field: Field,
) {
    for null_rate in [None, Some(0.5), Some(1.0)] {
        for use_slicing in [false, true] {
            let field = if null_rate.is_some() {
                if !supports_nulls(field.data_type()) {
                    continue;
                }
                field.clone().with_nullable(true)
            } else {
                field.clone().with_nullable(false)
            };

            let test_cases = TestCases::default()
                .with_range(0..500)
                .with_range(100..1100)
                .with_range(8000..8500)
                .with_indices(vec![100])
                .with_indices(vec![0])
                .with_indices(vec![9999])
                .with_indices(vec![100, 1100, 5000])
                .with_indices(vec![1000, 2000, 3000])
                .with_indices(vec![2000, 2001, 2002, 2003, 2004])
                // Big take that spans multiple pages and generates multiple output batches
                .with_indices((100..500).map(|i| i * 3).collect::<Vec<_>>());

            for num_ingest_batches in [1, 5, 10] {
                let rows_per_batch = NUM_RANDOM_ROWS / num_ingest_batches;
                let mut data = Vec::new();

                // Test both ingesting one big array sliced into smaller arrays and smaller
                // arrays independently generated.  These behave slightly differently.  For
                // example, a list array sliced into smaller arrays will have arrays whose
                // starting offset is not 0.
                if use_slicing {
                    let mut generator = gen().anon_col(array::rand_type(field.data_type()));
                    if let Some(null_rate) = null_rate {
                        generator.with_random_nulls(null_rate);
                    }
                    let all_data = generator
                        .into_batch_rows(RowCount::from(10000))
                        .unwrap()
                        .column(0)
                        .clone();
                    let mut offset = 0;
                    for _ in 0..num_ingest_batches {
                        data.push(all_data.slice(offset, rows_per_batch as usize));
                        offset += rows_per_batch as usize;
                    }
                } else {
                    for i in 0..num_ingest_batches {
                        let mut generator = gen()
                            .with_seed(Seed::from(i as u64))
                            .anon_col(array::rand_type(field.data_type()));
                        if let Some(null_rate) = null_rate {
                            generator.with_random_nulls(null_rate);
                        }
                        let arr = generator
                            .into_batch_rows(RowCount::from(rows_per_batch as u64))
                            .unwrap()
                            .column(0)
                            .clone();
                        data.push(arr);
                    }
                }

                debug!(
                    "Testing with {} rows divided across {} batches for {} rows per batch with null_rate={:?} and use_slicing={}",
                    NUM_RANDOM_ROWS,
                    num_ingest_batches,
                    rows_per_batch,
                    null_rate,
                    use_slicing
                );
                check_round_trip_encoding_inner(encoder_factory(), &field, data, &test_cases).await
            }
        }
    }
}
