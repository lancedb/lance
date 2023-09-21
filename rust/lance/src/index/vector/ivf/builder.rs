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

use std::ops::Range;
use std::sync::Arc;

use arrow_array::{
    cast::AsArray,
    types::{Float32Type, UInt32Type},
    Array, BooleanArray, FixedSizeListArray, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use arrow_select::filter::filter_record_batch;
use futures::{
    stream::{self, repeat_with},
    StreamExt, TryStreamExt,
};
use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
use lance_linalg::{distance::MetricType, MatrixView};
use snafu::{location, Location};

use crate::dataset::ROW_ID;
use crate::index::vector::ivf::{
    io::write_index_partitions, shuffler::ShufflerBuilder, Ivf, PARTITION_ID_COLUMN, PQ_CODE_COLUMN,
};
use crate::index::vector::pq::ProductQuantizer;
use crate::io::object_writer::ObjectWriter;
use crate::{io::RecordBatchStream, Error, Result};

use super::RESIDUAL_COLUMN;

/// Filter a batch by a range of partition IDs, specified by `part_range`.
///
/// Expect the input batch has schema as
///
/// ```json
/// {
///     "_rowid": "uint32",
///     "<column>": "fixed_size_list<float16/32/64>",
/// }
/// ```
///
/// And output batch has schema as
///
/// ```json
/// {
///     "_rowid": "uint32",
///     "<column>": "fixed_size_list<float16/32/64>",
///     "__ivf_part_id": "uint32",
/// }
/// ```
fn filter_batch_by_partition(
    batch: &RecordBatch,
    column: &str,
    ivf: &Ivf,
    metric_type: MetricType,
    part_range: Range<u32>,
) -> Result<RecordBatch> {
    let arr = batch
        .column_by_name(column)
        .expect("The caller already checked column exist")
        .as_fixed_size_list();
    let dim = arr.value_length() as usize;

    let matrix = MatrixView::new(
        Arc::new(arr.values().as_primitive::<Float32Type>().clone()),
        dim,
    );
    let part_ids = ivf.compute_partitions(&matrix, metric_type);
    let selected: BooleanArray = BooleanArray::from_unary(&part_ids, |p| part_range.contains(&p));
    let partition_field = Field::new(PARTITION_ID_COLUMN, DataType::UInt32, false);
    let batch = batch.try_with_column(partition_field, Arc::new(part_ids))?;
    let filtered = filter_record_batch(&batch, &selected)?;

    // Filtered rows.
    let arr = filtered
        .column_by_name(column)
        .expect("The caller already checked the column exist")
        .as_fixed_size_list();
    let origin = MatrixView::new(
        Arc::new(arr.values().as_primitive::<Float32Type>().clone()),
        dim,
    );
    let part_ids = filtered
        .column_by_name(PARTITION_ID_COLUMN)
        .unwrap()
        .as_primitive::<UInt32Type>();
    let residual = ivf.compute_residual(&origin, part_ids);
    let residual_field = Field::new(
        RESIDUAL_COLUMN,
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
        ),
        true,
    );
    let residual_arr =
        FixedSizeListArray::try_new_from_values(residual.data().as_ref().clone(), dim as i32)?;
    let filtered = filtered.try_with_column(residual_field, Arc::new(residual_arr))?;

    Ok(filtered)
}

/// Build specific partitions of IVF index.
///
///
pub(super) async fn build_partitions(
    writer: &mut ObjectWriter,
    data: impl RecordBatchStream + Unpin,
    column: &str,
    ivf: &mut Ivf,
    pq: &ProductQuantizer,
    metric_type: MetricType,
    part_range: Range<u32>,
) -> Result<()> {
    let schema = data.schema();
    if schema.column_with_name(column).is_none() {
        return Err(Error::Schema {
            message: format!("column {} does not exist in data stream", column),
            location: location!(),
        });
    }
    if schema.column_with_name(ROW_ID).is_none() {
        return Err(Error::Schema {
            message: "ROW ID is not set when building index partitions".to_string(),
            location: location!(),
        });
    }

    let ivf_immutable = Arc::new(ivf.clone());
    let mut stream = data
        .zip(repeat_with(|| (part_range.clone(), ivf_immutable.clone())))
        .map(move |(b, (range, ivf_ref))| async move {
            let batch = b?;
            let col = column.to_string();
            let range_copy = range.clone();
            // Filter out the rows that are not in the partition range.
            let batch = tokio::task::spawn_blocking(move || {
                filter_batch_by_partition(&batch, &col, &ivf_ref, metric_type, range_copy)
            })
            .await??;
            // Run product quantization.
            let arr = batch
                .column_by_name(RESIDUAL_COLUMN)
                .expect("The caller already checked column exist")
                .as_fixed_size_list();
            let data = MatrixView::<Float32Type>::new(
                Arc::new(arr.values().as_primitive::<Float32Type>().clone()),
                arr.value_length() as usize,
            );
            let pq_code = pq.transform(&data, metric_type).await?;
            let pq_field = Field::new(PQ_CODE_COLUMN, pq_code.data_type().clone(), false);
            let batch = batch.try_with_column(pq_field, Arc::new(pq_code))?;
            // Do not need to serialize original vector
            let batch = batch.drop_column(column)?.drop_column(RESIDUAL_COLUMN)?;
            Ok::<(arrow_array::RecordBatch, std::ops::Range<u32>), Error>((batch, range))
        })
        .buffer_unordered(num_cpus::get())
        .and_then(|batch_and_range| async move {
            // Split batch into per-partition batches
            let (batch, range) = batch_and_range;
            Ok(stream::iter(range).map(move |part_id| {
                let predictions = BooleanArray::from_unary(
                    batch
                        .column_by_name(PARTITION_ID_COLUMN)
                        .unwrap()
                        .as_primitive::<UInt32Type>(),
                    |pid| pid == part_id,
                );
                let parted_batch =
                    filter_record_batch(&batch, &predictions)?.drop_column(PARTITION_ID_COLUMN)?;
                Ok::<(u32, RecordBatch), Error>((part_id, parted_batch))
            }))
        })
        .try_flatten_unordered(num_cpus::get())
        .boxed();

    const FLUSH_THRESHOLD: usize = 2 * 1024;

    let schema = Schema::new(vec![
        Field::new(ROW_ID, DataType::UInt64, false),
        Field::new(
            PQ_CODE_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, false)),
                pq.num_sub_vectors as i32,
            ),
            false,
        ),
    ]);
    let mut shuffler_builder = ShufflerBuilder::try_new(&schema, FLUSH_THRESHOLD).await?;
    while let Some(result) = stream.next().await {
        let (part_id, batch) = result?;
        if batch.num_rows() == 0 {
            continue;
        }
        shuffler_builder.insert(part_id, batch).await?;
    }

    let shuffler = shuffler_builder.finish().await?;
    write_index_partitions(writer, ivf, &shuffler).await?;

    Ok(())
}

#[cfg(test)]
mod tests {}
