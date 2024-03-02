// Copyright 2024 Lance Developers.
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

//! Transform of a Vector Input with partition IDs.

use std::ops::Range;
use std::sync::Arc;

use arrow_array::types::UInt32Type;
use arrow_array::{cast::AsArray, Array, ArrowPrimitiveType, RecordBatch, UInt32Array};
use arrow_schema::Field;
use futures::{stream, StreamExt};
use log::info;
use snafu::{location, Location};
use tracing::{instrument, Instrument};

use lance_arrow::{ArrowFloatType, RecordBatchExt};
use lance_core::Result;
use lance_linalg::distance::{Dot, MetricType, L2};
use lance_linalg::MatrixView;

use crate::vector::transform::Transformer;

use super::PART_ID_COLUMN;

/// Ivf Transformer
///
/// It transforms a Vector column, specified by the input data, into a column of partition IDs.
///
/// If the partition ID ("__ivf_part_id") column is already present in the Batch,
/// this transform is a Noop.
///
#[derive(Debug)]
pub struct IvfTransformer<T: ArrowFloatType + L2 + Dot> {
    centroids: MatrixView<T>,
    metric_type: MetricType,
    input_column: String,
    output_column: String,
}

impl<T: ArrowFloatType + L2 + Dot> IvfTransformer<T> {
    pub fn new(
        centroids: MatrixView<T>,
        metric_type: MetricType,
        input_column: impl AsRef<str>,
    ) -> Self {
        Self {
            centroids,
            metric_type,
            input_column: input_column.as_ref().to_owned(),
            output_column: PART_ID_COLUMN.to_owned(),
        }
    }

    /// Compute the partition for each row in the input Matrix.
    ///
    #[instrument(level = "debug", skip(data))]
    pub(super) async fn compute_partitions(&self, data: &MatrixView<T>) -> UInt32Array {
        use lance_linalg::kmeans::compute_partitions;

        let dimension = data.ndim();
        let centroids = self.centroids.data();
        let data = data.data();
        let metric_type = self.metric_type;

        let num_centroids = centroids.len() / dimension;
        let num_rows = data.len() / dimension;

        let chunks = std::cmp::min(num_cpus::get(), num_rows);

        info!(
            "computing partition on {} chunks, out of {} centroids, and {} vectors",
            chunks, num_centroids, num_rows,
        );
        // TODO: when usize::div_ceil() comes to stable Rust, we can use it here.
        let chunk_size = num_rows / chunks + if num_rows % chunks > 0 { 1 } else { 0 };
        let stride = chunk_size * dimension;

        let result: Vec<Vec<Option<u32>>> = stream::iter(0..chunks)
            .map(|chunk_id| stride * chunk_id..std::cmp::min(stride * (chunk_id + 1), data.len()))
            // When there are a large number of CPUs and a small number of rows,
            // it's possible there isn't an split of rows that there isn't
            // an even split of rows that both covers all CPUs and all rows.
            // For example, for 400 rows and 32 CPUs, 12-element chunks (12 * 32 = 384)
            // wouldn't cover all rows but 13-element chunks (13 * 32 = 416) would
            // have one empty chunk at the end. This filter removes those empty chunks.
            .filter(|range| futures::future::ready(range.start < range.end))
            .map(|range| async {
                let range: Range<usize> = range;
                let centroids = centroids.clone();
                let data = Arc::new(
                    data.slice(range.start, range.end - range.start)
                        .as_any()
                        .downcast_ref::<T::ArrayType>()
                        .unwrap()
                        .clone(),
                );

                compute_partitions::<T>(centroids, data, dimension, metric_type)
                    .in_current_span()
                    .await
            })
            .buffered(chunks)
            .collect::<Vec<_>>()
            .await;

        UInt32Array::from_iter(result.iter().flatten().copied())
    }
}

#[async_trait::async_trait]
impl<T: ArrowFloatType + L2 + Dot + ArrowPrimitiveType> Transformer for IvfTransformer<T> {
    async fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        if batch.column_by_name(&self.output_column).is_some() {
            // If the partition ID column is already present, we don't need to compute it again.
            return Ok(batch.clone());
        }
        let arr =
            batch
                .column_by_name(&self.input_column)
                .ok_or_else(|| lance_core::Error::Index {
                    message: format!(
                        "IvfTransformer: column {} not found in the RecordBatch",
                        self.input_column
                    ),
                    location: location!(),
                })?;
        let fsl = arr
            .as_fixed_size_list_opt()
            .ok_or_else(|| lance_core::Error::Index {
                message: format!(
                    "IvfTransformer: column {} is not a FixedSizeListArray: {}",
                    self.input_column,
                    arr.data_type(),
                ),
                location: location!(),
            })?;

        let mat = MatrixView::<T>::try_from(fsl)?;
        let part_ids = self.compute_partitions(&mat).await;
        let field = Field::new(PART_ID_COLUMN, part_ids.data_type().clone(), true);
        Ok(batch.try_with_column(field, Arc::new(part_ids))?)
    }
}

#[derive(Debug)]
pub(super) struct PartitionFilter {
    /// The partition column name.
    column: String,
    /// The partition range to filter.
    partition_range: Range<u32>,
}

impl PartitionFilter {
    pub fn new(column: impl AsRef<str>, partition_range: Range<u32>) -> Self {
        Self {
            column: column.as_ref().to_owned(),
            partition_range,
        }
    }

    fn filter_row_ids(&self, partition_ids: &[u32]) -> Vec<u32> {
        partition_ids
            .iter()
            .enumerate()
            .filter_map(|(idx, &part_id)| {
                if self.partition_range.contains(&part_id) {
                    Some(idx as u32)
                } else {
                    None
                }
            })
            .collect()
    }
}

#[async_trait::async_trait]
impl Transformer for PartitionFilter {
    async fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        // TODO: use datafusion execute?
        let arr = batch
            .column_by_name(&self.column)
            .ok_or_else(|| lance_core::Error::Index {
                message: format!(
                    "PartitionFilter: column {} not found in the RecordBatch",
                    self.column
                ),
                location: location!(),
            })?;
        let part_ids = arr.as_primitive::<UInt32Type>();
        let indices = UInt32Array::from(self.filter_row_ids(part_ids.values()));
        Ok(batch.take(&indices)?)
    }
}
