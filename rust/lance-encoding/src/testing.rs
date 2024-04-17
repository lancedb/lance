// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{ops::Range, sync::Arc};

use arrow_array::{Array, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use bytes::{Bytes, BytesMut};
use futures::{future::BoxFuture, FutureExt, StreamExt};
use log::trace;
use tokio::sync::mpsc::{self, UnboundedSender};

use lance_core::Result;
use lance_datagen::{array, gen, RowCount};

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
                    bytes.extend(part.iter())
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
    fn submit_request(&self, ranges: Vec<Range<u64>>) -> BoxFuture<'static, Result<Vec<Bytes>>> {
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
    expected: Arc<dyn Array>,
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
        let actual = batch.column(0);
        let expected_size = (BATCH_SIZE as usize).min(expected.len() - offset);
        let expected = expected.slice(offset, expected_size);
        assert_eq!(expected.data_type(), actual.data_type());
        assert_eq!(&expected, actual);
        offset += BATCH_SIZE as usize;
    }
}

pub async fn check_round_trip_encoding(field: Field) {
    let mut col_idx = 0;
    let mut field_id_to_col_index = Vec::new();
    let lance_field = lance_core::datatypes::Field::try_from(&field).unwrap();
    let encoder = BatchEncoder::get_encoder_for_field(
        &lance_field,
        4096,
        &mut col_idx,
        &mut field_id_to_col_index,
    )
    .unwrap();
    check_round_trip_field_encoding(encoder, field).await
}

fn supports_nulls(data_type: &DataType) -> bool {
    // We don't yet have nullability support for all types.  Don't test nullability for the
    // types we don't support.
    !matches!(
        data_type,
        DataType::List(_) | DataType::Struct(_) | DataType::Utf8 | DataType::Binary
    )
}

async fn check_round_trip_field_encoding(mut encoder: Box<dyn FieldEncoder>, field: Field) {
    for null_rate in [None, Some(0.5), Some(1.0)] {
        let field = if null_rate.is_some() {
            if !supports_nulls(field.data_type()) {
                continue;
            }
            field.clone().with_nullable(true)
        } else {
            field.clone().with_nullable(false)
        };
        let mut generator = gen().col(None, array::rand_type(field.data_type()));
        if let Some(null_rate) = null_rate {
            generator.with_random_nulls(null_rate);
        }
        let data = generator
            .into_batch_rows(RowCount::from(10000))
            .unwrap()
            .column(0)
            .clone();

        let num_rows = data.len();

        for num_ingest_batches in [1, 5, 10] {
            let rows_per_batch = num_rows / num_ingest_batches;
            trace!(
                "Testing with {} rows divided across {} batches for {} rows per batch",
                num_rows,
                num_ingest_batches,
                rows_per_batch
            );

            let mut offset = 0;
            let mut all_encoded_pages = Vec::new();
            let mut page_infos: Vec<Vec<Arc<PageInfo>>> =
                vec![Vec::new(); encoder.num_columns() as usize];
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
                        buffer_offset +=
                            buf.parts.iter().map(|part| part.len() as u64).sum::<u64>();
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

            for _ in 0..num_ingest_batches {
                let data = data.slice(offset, rows_per_batch);

                for encode_task in encoder.maybe_encode(data).unwrap() {
                    let encoded_page = encode_task.await.unwrap();
                    simulate_write(encoded_page);
                }

                offset += rows_per_batch;
            }

            for encode_task in encoder.flush().unwrap() {
                let encoded_page = encode_task.await.unwrap();
                simulate_write(encoded_page);
            }

            let scheduler =
                Arc::new(SimulatedScheduler::new(all_encoded_pages)) as Arc<dyn EncodingsIo>;

            let column_infos = page_infos
                .into_iter()
                .map(|page_infos| ColumnInfo::new(page_infos, Vec::new()))
                .collect::<Vec<_>>();
            let schema = Schema::new(vec![field.clone()]);

            // Test range scheduling
            for range in [0..500, 100..1100, 8000..8500] {
                let range = range.start as u64..range.end as u64;
                let num_rows = range.end - range.start;
                let expected = data.slice(range.start as usize, num_rows as usize);
                let scheduler = scheduler.clone();
                test_decode(
                    num_rows,
                    &schema,
                    &column_infos,
                    expected,
                    |mut decode_scheduler, tx| {
                        async move { decode_scheduler.schedule_range(range, tx, &scheduler).await }
                            .boxed()
                    },
                )
                .await;
            }

            // Test take scheduling
            for indices in [
                vec![100],
                vec![0],
                vec![9999],
                vec![100, 1100, 5000],
                vec![1000, 2000, 3000],
                vec![2000, 2001, 2002, 2003, 2004],
                // Big take that spans multiple pages and generates multiple output batches
                (100..500).map(|i| i * 3).collect::<Vec<_>>(),
            ] {
                let num_rows = indices.len() as u64;
                let indices_arr = UInt32Array::from(indices.clone());
                let expected = arrow_select::take::take(&data, &indices_arr, None).unwrap();
                let scheduler = scheduler.clone();
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
    }
}
