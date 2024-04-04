use std::{ops::Range, sync::Arc};

use arrow_schema::{Field, Schema};
use bytes::{Bytes, BytesMut};
use futures::{future::BoxFuture, FutureExt, StreamExt};
use log::trace;
use tokio::sync::mpsc;

use lance_core::Result;
use lance_datagen::{array, gen, RowCount};

use crate::{
    decoder::{BatchDecodeStream, ColumnInfo, DecodeBatchScheduler, PageInfo},
    encoder::{ArrayEncoder, EncodedPage, FieldEncoder},
    encodings::logical::primitive::PrimitiveFieldEncoder,
    EncodingsIo,
};

pub(crate) struct SimulatedScheduler {
    data: Bytes,
}

impl SimulatedScheduler {
    pub fn new(data: Vec<EncodedPage>) -> Self {
        let mut bytes = BytesMut::new();
        for arr in data.into_iter() {
            for buf in arr.buffers.into_iter() {
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

pub async fn check_round_trip_array_encoding(encoder: impl ArrayEncoder + 'static, field: Field) {
    let array_encoder = Arc::new(encoder);
    for page_size_bytes in [1024, 4 * 1024, 1024 * 1024] {
        let field_encoder = PrimitiveFieldEncoder::new(page_size_bytes, array_encoder.clone());
        check_round_trip_field_encoding(field_encoder, field.clone()).await
    }
}

pub async fn check_round_trip_field_encoding(mut encoder: impl FieldEncoder, field: Field) {
    let data = gen()
        .col(None, array::rand_type(field.data_type()))
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

        let mut simulate_write = |encoded_page: EncodedPage| {
            trace!("Encoded page {:?}", encoded_page);
            let buffer_offsets = encoded_page
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
                encoding: encoded_page.encoding.clone(),
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

        for range in [0..500, 100..1100, 8000..8500] {
            let num_rows = range.end - range.start;
            let schema = schema.clone();
            let mut decode_scheduler =
                DecodeBatchScheduler::new(&schema, &column_infos, &Vec::new());

            let (tx, rx) = mpsc::unbounded_channel();

            decode_scheduler
                .schedule_range(range.clone(), tx, &scheduler)
                .await
                .unwrap();

            const BATCH_SIZE: u32 = 100;
            let mut decode_stream = BatchDecodeStream::new(rx, BATCH_SIZE, num_rows).into_stream();

            let mut offset = range.start as usize;
            while let Some(batch) = decode_stream.next().await {
                let batch = batch.await.unwrap().unwrap();
                let actual = batch.column(0);
                let expected = data.slice(offset, BATCH_SIZE as usize);
                assert_eq!(expected.data_type(), actual.data_type());
                assert_eq!(&expected, actual);
                offset += BATCH_SIZE as usize;
            }
        }
    }
}
