use std::{ops::Range, sync::Arc};

use arrow_schema::{Field, Schema};
use bytes::{Bytes, BytesMut};
use futures::{future::BoxFuture, FutureExt, StreamExt};
use tokio::sync::mpsc;

use lance_core::Result;
use lance_datagen::{array, gen, RowCount};

use crate::{
    decoder::{
        BatchDecodeStream, ColumnInfo, DecodeBatchScheduler, PageInfo, PhysicalPageScheduler,
    },
    encoder::{ArrayEncoder, EncodedPage},
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

pub async fn check_round_trip_encoding(
    encoder: &dyn ArrayEncoder,
    decoders: &[Arc<dyn PhysicalPageScheduler>],
    field: Field,
) {
    let data = gen()
        .col(None, array::rand_type(field.data_type()))
        .into_batch_rows(RowCount::from(10000))
        .unwrap()
        .column(0)
        .clone();

    let num_rows = data.len();

    for num_pages in [1, 5, 10] {
        println!("Testing with {} pages", num_pages);
        let rows_per_page = num_rows / num_pages;

        let mut offset = 0;
        let mut encoded_arrays = Vec::new();
        let mut page_infos = vec![Vec::new(); decoders.len()];
        let mut buffer_offset = 0;

        for _ in 0..num_pages {
            let data = data.slice(offset, rows_per_page);

            let encoded_arrs = encoder.encode(&[data]).unwrap();
            assert_eq!(encoded_arrs.len(), decoders.len());
            for (idx, (decoder, encoded_arr)) in
                decoders.iter().zip(encoded_arrs.into_iter()).enumerate()
            {
                let buffer_offsets = encoded_arr
                    .buffers
                    .iter()
                    .map(|buf| {
                        let offset = buffer_offset;
                        buffer_offset +=
                            buf.parts.iter().map(|part| part.len() as u64).sum::<u64>();
                        offset
                    })
                    .collect::<Vec<_>>();
                encoded_arrays.push(encoded_arr);

                let page_info = PageInfo {
                    num_rows: rows_per_page as u32,
                    decoder: decoder.clone(),
                    buffer_offsets: Arc::new(buffer_offsets.clone()),
                };
                page_infos[idx].push(Arc::new(page_info));
            }

            offset += rows_per_page;
        }

        let scheduler = Arc::new(SimulatedScheduler::new(encoded_arrays)) as Arc<dyn EncodingsIo>;

        let column_infos = page_infos
            .into_iter()
            .map(ColumnInfo::new)
            .collect::<Vec<_>>();
        let schema = Schema::new(vec![field.clone()]);
        let mut decode_scheduler = DecodeBatchScheduler::new(&schema, &column_infos);

        let (tx, rx) = mpsc::channel(1024);

        decode_scheduler
            .schedule_range(100..1100, tx, &scheduler)
            .await
            .unwrap();

        let mut decode_stream = BatchDecodeStream::new(rx, schema, 100, 1000).into_stream();

        while let Some(batch) = decode_stream.next().await {
            batch.await.unwrap().unwrap();
            // TODO: Assert it matches range
        }
    }
}
