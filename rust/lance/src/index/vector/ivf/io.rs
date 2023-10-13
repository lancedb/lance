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

use std::sync::Arc;

use arrow_array::{Array, FixedSizeListArray};
use futures::StreamExt;
use lance_arrow::*;
use lance_core::io::Writer;
use lance_index::vector::PQ_CODE_COLUMN;

use super::{shuffler::Shuffler, IVFIndex, Ivf};
use crate::dataset::ROW_ID;
use crate::encodings::plain::PlainEncoder;
use crate::index::vector::pq::PQIndex;
use crate::Result;

/// Write each partition of IVF_PQ index to the index file.
///
/// Partitioned index data is already sorted in the [Shuffler].
pub(super) async fn write_index_partitions(
    writer: &mut dyn Writer,
    ivf: &mut Ivf,
    shuffler: &Shuffler,
    existing_partitions: Option<&IVFIndex>,
) -> Result<()> {
    for part_id in 0..ivf.num_partitions() as u32 {
        let mut pq_array = Vec::<Arc<dyn Array>>::new();
        let mut row_id_array = Vec::<Arc<dyn Array>>::new();

        if let Some(existing_idx) = existing_partitions.as_ref() {
            let part = existing_idx.load_partition(part_id as usize).await?;
            let pq_idx = part.as_any().downcast_ref::<PQIndex>().unwrap();
            if pq_idx.code.is_some() {
                let pq_code_arr = pq_idx.code.as_ref().unwrap().clone();
                let pq_code_fixed_size_arr = FixedSizeListArray::try_new_from_values(
                    pq_code_arr.as_ref().clone(),
                    pq_idx.num_sub_vectors as i32,
                )?;
                pq_array.push(Arc::new(pq_code_fixed_size_arr));
                println!("pq_idx.row_ids: {:#?}", pq_idx.row_ids.as_ref());
                row_id_array.push(pq_idx.row_ids.as_ref().unwrap().clone());
            }
        }

        if let Some(mut stream) = shuffler.key_iter(part_id).await? {
            while let Some(batch) = stream.next().await {
                let batch = batch?;
                let arr = batch.column_by_name(PQ_CODE_COLUMN).unwrap();
                pq_array.push(arr.clone());
                let arr = batch.column_by_name(ROW_ID).unwrap();
                row_id_array.push(arr.clone());
            }
        }

        let total_records = row_id_array.iter().map(|a| a.len()).sum::<usize>();
        ivf.add_partition(writer.tell(), total_records as u32);
        if total_records > 0 {
            let pq_refs = pq_array.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
            PlainEncoder::write(writer, &pq_refs).await?;

            let row_ids_refs = row_id_array.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
            PlainEncoder::write(writer, row_ids_refs.as_slice()).await?;
        }
    }
    Ok(())
}
