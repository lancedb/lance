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

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::{Array, FixedSizeListArray, RecordBatch};
use datafusion::error::Result as DFResult;
use datafusion::scalar::ScalarValue;
use futures::{Stream, StreamExt};
use lance_arrow::*;
use lance_core::io::Writer;
use lance_index::vector::PQ_CODE_COLUMN;

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
    new_data: impl Stream<Item = DFResult<(Vec<ScalarValue>, Vec<RecordBatch>)>> + Unpin,
    existing_partitions: Option<&IVFIndex>,
) -> Result<()> {
    let mut new_data = new_data.peekable();
    let mut new_data_ref = Pin::new(&mut new_data);

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

        // The new data is sorted by partition id, but it's not guaranteed that
        // every partition id has data, so we peek into the next batch to check
        // if the partition id matches.
        match new_data_ref.as_mut().peek().await {
            Some(Ok((part_values, _batch)))
                if part_values[0] == ScalarValue::UInt32(Some(part_id)) =>
            {
                let batches = new_data_ref.as_mut().next().await.unwrap().unwrap().1;
                for batch in batches {
                    let arr = batch.column_by_name(PQ_CODE_COLUMN).unwrap();
                    pq_array.push(arr.clone());
                    let arr = batch.column_by_name(ROW_ID).unwrap();
                    row_id_array.push(arr.clone());
                }
            }
            Some(Err(_)) => {
                return Err(new_data_ref
                    .as_mut()
                    .next()
                    .await
                    .unwrap()
                    .unwrap_err()
                    .into())
            }
            _ => {}
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
