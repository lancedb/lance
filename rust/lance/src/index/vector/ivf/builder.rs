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

use arrow_schema::{DataType, Field, Schema};
use datafusion::error::DataFusionError;
use datafusion::execution::context::SessionContext;
use datafusion::execution::memory_pool::GreedyMemoryPool;
use datafusion::execution::runtime_env::{RuntimeConfig, RuntimeEnv};
use datafusion::logical_expr::col;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::{stream::repeat_with, StreamExt};
use lance_core::{io::Writer, ROW_ID, ROW_ID_FIELD};
use lance_datafusion::dataframe::{BatchStreamGrouper, DataFrameExt};
use lance_datafusion::exec::SessionContextExt;
use lance_index::vector::pq::ProductQuantizer;
use lance_index::vector::{PART_ID_COLUMN, PQ_CODE_COLUMN};
use lance_linalg::distance::MetricType;
use log::info;
use nohash_hasher::IntMap;
use snafu::{location, Location};
use tracing::instrument;

use crate::index::vector::ivf::{io::write_index_partitions, Ivf};
use crate::{io::RecordBatchStream, Error, Result};

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
///   BatchStreamGrouper: a stream of `Vec<RecordBatch>` each associated with
///   a partition id. The stream is sorted by partition id.
///
/// TODO: move this to `lance-index` crate.
pub async fn shuffle_dataset(
    data: impl RecordBatchStream + Unpin + 'static,
    column: &str,
    ivf: Arc<dyn lance_index::vector::ivf::Ivf>,
    // TODO: Once the transformer can generate schema automatically,
    // we can remove `num_sub_vectors`.
    num_sub_vectors: usize,
    memory_limit: usize,
) -> Result<BatchStreamGrouper> {
    let column: Arc<str> = column.into();
    let stream = data
        .zip(repeat_with(move || ivf.clone()))
        .map(move |(b, ivf)| {
            let col_ref = column.clone();

            tokio::task::spawn(async move {
                let batch = b?;
                ivf.partition_transform(&batch, col_ref.as_ref()).await
            })
        })
        .buffer_unordered(num_cpus::get())
        .map(|res| match res {
            Ok(Ok(batch)) => Ok(batch),
            Ok(Err(err)) => Err(DataFusionError::External(Box::new(err))),
            Err(err) => Err(DataFusionError::Execution(err.to_string())),
        })
        .boxed();

    // TODO: dynamically detect schema from the transforms.
    let schema = Arc::new(Schema::new(vec![
        ROW_ID_FIELD.clone(),
        Field::new(PART_ID_COLUMN, DataType::UInt32, false),
        Field::new(
            PQ_CODE_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, true)),
                num_sub_vectors as i32,
            ),
            false,
        ),
    ]));
    let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream));

    info!("Building IVF shuffler");

    let runtime_config =
        RuntimeConfig::new().with_memory_pool(Arc::new(GreedyMemoryPool::new(memory_limit)));
    let runtime_env = RuntimeEnv::new(runtime_config)?;
    let context = SessionContext::new_with_config_rt(Default::default(), Arc::new(runtime_env));

    Ok(context
        .read_one_shot(stream)?
        .sort(vec![col(PART_ID_COLUMN).sort(true, true)])?
        .group_by_stream(&[PART_ID_COLUMN])
        .await?)
}

/// Build specific partitions of IVF index.
///
///
#[allow(clippy::too_many_arguments)]
#[instrument(level = "debug", skip(writer, data, ivf, pq))]
pub(super) async fn build_partitions(
    writer: &mut dyn Writer,
    data: impl RecordBatchStream + Unpin + 'static,
    column: &str,
    ivf: &mut Ivf,
    pq: Arc<dyn ProductQuantizer>,
    metric_type: MetricType,
    part_range: Range<u32>,
    precomputed_partitons: Option<IntMap<u64, u32>>,
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

    let ivf_model = lance_index::vector::ivf::new_ivf_with_pq(
        ivf.centroids.values(),
        ivf.centroids.value_length() as usize,
        metric_type,
        column,
        pq.clone(),
        Some(part_range),
        precomputed_partitons,
    )?;

    let memory_limit = 4 * 1024 * 1024 * 1024; // 4GB

    let shuffled =
        shuffle_dataset(data, column, ivf_model, pq.num_sub_vectors(), memory_limit).await?;

    write_index_partitions(writer, ivf, shuffled, None).await?;

    Ok(())
}
