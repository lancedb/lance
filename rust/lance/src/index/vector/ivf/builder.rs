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
use std::{collections::BTreeMap, ops::Range};

use arrow_array::types::UInt32Type;
use arrow_array::{cast::AsArray, types::Float32Type, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use futures::{stream::repeat_with, StreamExt};
use lance_arrow::RecordBatchExt;
use lance_core::{io::Writer, ROW_ID, ROW_ID_FIELD};
use lance_index::vector::{
    pq::{transform::PQTransformer, ProductQuantizer},
    residual::ResidualTransform,
    PART_ID_COLUMN, PQ_CODE_COLUMN,
};
use lance_linalg::{distance::MetricType, MatrixView};
use snafu::{location, Location};
use tracing::instrument;

use crate::index::vector::ivf::{
    io::write_index_partitions,
    shuffler::{Shuffler, ShufflerBuilder},
    Ivf,
};
use crate::{io::RecordBatchStream, Error, Result};

use super::RESIDUAL_COLUMN;

/// Disk-based shuffle a stream of [RecordBatch] into each IVF partition.
/// Sub-quantizer will be applied if provided.
///
/// Parameters
/// ----------
///   *data*: input data stream.
///   *ivf*: IVF model.
///
/// Returns
/// -------
///   Shuffler: a shuffler that stored the shuffled data.
///
/// TODO: move this to `lance-index` crate.
pub async fn shuffle_dataset(
    data: impl RecordBatchStream + Unpin,
    column: &str,
    ivf: Arc<lance_index::vector::ivf::Ivf>,
    // TODO: Once the transformer can generate schema automatically,
    // we can remove `num_sub_vectors`.
    num_sub_vectors: usize,
) -> Result<Shuffler> {
    let mut stream = data
        .zip(repeat_with(|| ivf.clone()))
        .map(|(b, ivf)| async move {
            let batch = b?;
            // TODO: Make CPU bound to a future.
            ivf.partition_transform(&batch, column).await
        })
        .buffer_unordered(num_cpus::get() * 2)
        .map(|batch| async move {
            let batch = batch?;
            // Collecting partition ID and row ID.
            tokio::task::spawn_blocking(move || {
                let part_id = batch
                    .column_by_name(PART_ID_COLUMN)
                    .expect("The caller already checked column exist");
                let part_id_arr = part_id.as_primitive::<UInt32Type>();
                let mut cnt_map = BTreeMap::<u32, Vec<u32>>::new();
                for (idx, part_id) in part_id_arr.values().iter().enumerate() {
                    cnt_map.entry(*part_id).or_default().push(idx as u32);
                }
                cnt_map
                    .into_iter()
                    .map(|(part_id, row_ids)| {
                        let indices = UInt32Array::from(row_ids);
                        let batch = batch.take(&indices)?;
                        Ok((part_id, batch))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .await
            .map_err(|e| Error::Index {
                message: e.to_string(),
                location: location!(),
            })
        })
        .buffer_unordered(num_cpus::get())
        .boxed();

    // TODO: dynamically detect schema from the transforms.
    let schema = Schema::new(vec![
        ROW_ID_FIELD.clone(),
        Field::new(
            PQ_CODE_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, false)),
                num_sub_vectors as i32,
            ),
            false,
        ),
    ]);
    const FLUSH_THRESHOLD: usize = 40 * 1024;

    let mut shuffler_builder = ShufflerBuilder::try_new(&schema, FLUSH_THRESHOLD).await?;
    while let Some(result) = stream.next().await {
        let batches = result??;
        if batches.is_empty() {
            continue;
        }
        for (part_id, batch) in batches {
            shuffler_builder.insert(part_id, batch).await?;
        }
    }
    shuffler_builder.finish().await
}

/// Build specific partitions of IVF index.
///
///
#[instrument(level = "debug", skip(writer, data, ivf, pq))]
pub(super) async fn build_partitions(
    writer: &mut dyn Writer,
    data: impl RecordBatchStream + Unpin,
    column: &str,
    ivf: &mut Ivf,
    pq: Arc<ProductQuantizer>,
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

    let centroids: MatrixView<Float32Type> = ivf.centroids.as_ref().try_into()?;
    let ivf_model = lance_index::vector::ivf::Ivf::new_with_range(
        centroids.clone(),
        metric_type,
        vec![
            Arc::new(ResidualTransform::new(
                ivf.centroids.as_ref().try_into()?,
                PART_ID_COLUMN,
                column,
            )),
            Arc::new(PQTransformer::new(
                pq.clone(),
                RESIDUAL_COLUMN,
                PQ_CODE_COLUMN,
            )),
        ],
        part_range.clone(),
    );
    let shuffler = shuffle_dataset(data, column, Arc::new(ivf_model), pq.num_sub_vectors).await?;
    write_index_partitions(writer, ivf, &shuffler, None).await?;

    Ok(())
}
