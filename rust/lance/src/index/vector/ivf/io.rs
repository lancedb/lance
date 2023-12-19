// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::cast::AsArray;
use arrow_array::{Array, FixedSizeListArray, RecordBatch, UInt32Array, UInt64Array};
use futures::{Stream, StreamExt};
use lance_arrow::*;
use lance_core::io::Writer;
use lance_core::Error;
use lance_index::vector::{PART_ID_COLUMN, PQ_CODE_COLUMN};
use snafu::{location, Location};

use super::{IVFIndex, Ivf};
use crate::dataset::ROW_ID;
use crate::encodings::plain::PlainEncoder;
use crate::index::vector::pq::PQIndex;
use crate::Result;

/// Write each partition of IVF_PQ index to the index file.
///
/// `batches`: RecordBatch stream of PQ codes and row ids, sorted by PQ code.
pub(super) async fn write_index_partitions(
    writer: &mut dyn Writer,
    ivf: &mut Ivf,
    streams: Vec<impl Stream<Item = Result<RecordBatch>>>,
    existing_partitions: Option<&IVFIndex>,
) -> Result<()> {
    // build the inital heap
    let mut streams_heap = BinaryHeap::new();
    let mut new_streams = vec![];

    for stream in streams {
        let mut stream = Box::pin(stream.peekable());

        match stream.as_mut().peek().await {
            Some(Ok(batch)) => {
                let part_ids: &UInt32Array = batch
                    .column_by_name(PART_ID_COLUMN)
                    .expect("part id column not found")
                    .as_primitive();
                let part_id = part_ids.values()[0];
                streams_heap.push((part_id, new_streams.len()));
                new_streams.push(stream);
            }
            Some(Err(e)) => {
                return Err(Error::IO {
                    message: format!("failed to read batch: {}", e),
                    location: location!(),
                });
            }
            None => {
                return Err(Error::IO {
                    message: "failed to read batch: end of stream".to_string(),
                    location: location!(),
                });
            }
        }
    }

    for part_id in 0..ivf.num_partitions() as u32 {
        let start = Instant::now();
        let mut pq_array = Vec::<Arc<dyn Array>>::new();
        let mut row_id_array = Vec::<Arc<dyn Array>>::new();

        if let Some(existing_idx) = existing_partitions.as_ref() {
            let part = existing_idx.load_partition(part_id as usize, true).await?;
            let pq_idx = part.as_any().downcast_ref::<PQIndex>().unwrap();
            if pq_idx.code.is_some() {
                let pq_code_arr = pq_idx.code.as_ref().unwrap().clone();
                let pq_code_fixed_size_arr = FixedSizeListArray::try_new_from_values(
                    pq_code_arr.as_ref().clone(),
                    pq_idx.pq.num_sub_vectors() as i32,
                )?;
                pq_array.push(Arc::new(pq_code_fixed_size_arr));
                row_id_array.push(pq_idx.row_ids.as_ref().unwrap().clone());
            }
        }

        // Merge all streams with the same partition id.
        while let Some((stream_part_id, stream_idx)) = streams_heap.pop() {
            if stream_part_id != part_id {
                streams_heap.push((stream_part_id, stream_idx));
                break;
            }

            let mut stream = new_streams[stream_idx].as_mut();
            let batch = match stream.next().await {
                Some(Ok(batch)) => batch,
                Some(Err(e)) => {
                    return Err(Error::IO {
                        message: format!("failed to read batch: {}", e),
                        location: location!(),
                    });
                }
                None => {
                    return Err(Error::IO {
                        message: "failed to read batch: unexpected end of stream".to_string(),
                        location: location!(),
                    });
                }
            };

            let pq_codes = batch
                .column_by_name(PQ_CODE_COLUMN)
                .expect("pq code column not found")
                .as_fixed_size_list()
                .clone();

            let row_ids: UInt64Array = batch
                .column_by_name(ROW_ID)
                .expect("row id column not found")
                .as_primitive()
                .clone();

            pq_array.push(Arc::new(pq_codes));
            row_id_array.push(Arc::new(row_ids));

            match stream.peek().await {
                Some(Ok(batch)) => {
                    let part_ids: &UInt32Array = batch
                        .column_by_name(PART_ID_COLUMN)
                        .expect("part id column not found")
                        .as_primitive();
                    if !part_ids.is_empty() {
                        streams_heap.push((part_ids.value(0), stream_idx));
                    }
                }
                Some(Err(e)) => {
                    return Err(Error::IO {
                        message: format!("IVF Shuffler::failed to read batch: {}", e),
                        location: location!(),
                    });
                }
                None => {}
            }
        }

        let total_records = row_id_array.iter().map(|a| a.len()).sum::<usize>();
        ivf.add_partition(writer.tell().await?, total_records as u32);
        if total_records > 0 {
            let pq_refs = pq_array.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
            PlainEncoder::write(writer, &pq_refs).await?;

            let row_ids_refs = row_id_array.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
            PlainEncoder::write(writer, row_ids_refs.as_slice()).await?;
        }
        log::info!(
            "Wrote partition {} in {} ms",
            part_id,
            start.elapsed().as_millis()
        );
    }
    Ok(())
}
