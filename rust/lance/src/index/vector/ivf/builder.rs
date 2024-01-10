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

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use futures::Stream;
use futures::{stream::repeat_with, StreamExt};
use lance_core::datatypes::Schema as LanceSchema;
use lance_core::{io::Writer, ROW_ID, ROW_ID_FIELD};
use lance_index::vector::ivf::shuffler::IvfShuffler;
use lance_index::vector::pq::ProductQuantizer;
use lance_index::vector::{PART_ID_COLUMN, PQ_CODE_COLUMN};
use lance_linalg::distance::MetricType;
use log::info;
use snafu::{location, Location};
use tracing::instrument;

use crate::index::vector::ivf::{io::write_index_partitions, Ivf};
use crate::{io::RecordBatchStream, Error, Result};

/// Disk-based shuffle for a stream of [RecordBatch] into each IVF partition.
/// Sub-quantizer will be applied if provided.
///
/// Parameters
/// ----------
///   *data*: input data stream.
///   *column*: column name of the vector column.
///   *ivf*: IVF model.
///   *num_partitions*: number of IVF partitions.
///   *num_sub_vectors*: number of PQ sub-vectors.
///
/// Returns
/// -------
///   Result<Vec<impl Stream<Item = Result<RecordBatch>>>>: a vector of streams
///   of shuffled partitioned data. Each stream corresponds to a partition and
///   is sorted within the stream. Consumer of these streams is expected to merge
///   the streams into a single stream by k-list mergo algo.
///
pub async fn shuffle_dataset(
    data: impl RecordBatchStream + Unpin + 'static,
    column: &str,
    ivf: Arc<dyn lance_index::vector::ivf::Ivf>,
    num_partitions: u32,
    num_sub_vectors: usize,
    shuffle_partition_batches: usize,
    shuffle_partition_concurrency: usize,
) -> Result<Vec<impl Stream<Item = Result<RecordBatch>>>> {
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
            Ok(Err(err)) => Err(Error::IO {
                message: err.to_string(),
                location: location!(),
            }),
            Err(err) => Err(Error::IO {
                message: err.to_string(),
                location: location!(),
            }),
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

    let stream = lance_core::io::RecordBatchStreamAdapter::new(schema.clone(), stream);

    let mut shuffler = IvfShuffler::try_new(
        num_partitions,
        num_sub_vectors,
        None,
        LanceSchema::try_from(schema.as_ref())?,
    )?;

    let start = std::time::Instant::now();
    shuffler.write_unsorted_stream(stream).await?;
    info!("wrote raw stream: {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let partition_files = shuffler
        .write_partitioned_shuffles(shuffle_partition_batches, shuffle_partition_concurrency)
        .await?;
    info!("counted partition sizes: {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let stream = shuffler.load_partitioned_shuffles(partition_files).await?;
    info!("merged partitioned shuffles: {:?}", start.elapsed());

    Ok(stream)
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
    precomputed_partitons: Option<HashMap<u64, u32>>,
    shuffle_partition_batches: usize,
    shuffle_partition_concurrency: usize,
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

    let stream = shuffle_dataset(
        data,
        column,
        ivf_model,
        ivf.num_partitions() as u32,
        pq.num_sub_vectors(),
        shuffle_partition_batches,
        shuffle_partition_concurrency,
    )
    .await?;

    write_index_partitions(writer, ivf, stream, None).await?;

    Ok(())
}
