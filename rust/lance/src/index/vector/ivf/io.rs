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

use futures::StreamExt;

use super::{shuffler::Shuffler, Ivf};
use crate::dataset::ROW_ID;
use crate::index::vector::ivf::PQ_CODE_COLUMN;
use crate::io::object_writer::ObjectWriter;
use crate::Result;

/// Write each partition of IVF_PQ index to the index file.
///
/// Partitioned index data is already sorted in the [Shuffler].
pub(super) async fn write_index_partitions(
    writer: &mut ObjectWriter,
    ivf: &mut Ivf,
    shuffler: &Shuffler<'_>,
) -> Result<()> {
    for part_id in 0..ivf.num_partitions() as u32 {
        if let Some(mut stream) = shuffler.key_iter(part_id).await? {
            let mut pq_array = vec![];
            let mut row_id_array = vec![];
            while let Some(batch) = stream.next().await {
                let batch = batch?;
                let arr = batch.column_by_name(PQ_CODE_COLUMN).unwrap();
                pq_array.push(arr.clone());
                let arr = batch.column_by_name(ROW_ID).unwrap();
                row_id_array.push(arr.clone());
            }
            let total_records = row_id_array.iter().map(|a| a.len()).sum::<usize>();

            ivf.add_partition(writer.tell(), total_records as u32);

            let pq_refs = pq_array.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
            writer.write_plain_encoded_array(pq_refs.as_slice()).await?;

            let row_ids_refs = row_id_array.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
            writer
                .write_plain_encoded_array(row_ids_refs.as_slice())
                .await?;
        } else {
            ivf.add_partition(writer.tell(), 0);
        }
    }
    Ok(())
}
