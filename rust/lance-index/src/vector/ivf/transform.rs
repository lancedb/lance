// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Transform of a Vector Input with partition IDs.

use std::ops::Range;
use std::sync::Arc;

use arrow::array::{ArrayBuilder, ListBuilder, UInt32Builder, UInt64Builder};
use arrow_array::types::UInt32Type;
use arrow_array::{cast::AsArray, Array, ArrowPrimitiveType, RecordBatch, UInt32Array};
use arrow_array::{GenericListArray, ListArray, UInt64Array};
use arrow_schema::Field;
use futures::{stream, StreamExt};
use lance_linalg::kmeans::compute_multiple_partitions;
use log::info;
use snafu::{location, Location};
use tracing::{instrument, Instrument};

use lance_arrow::{ArrowFloatType, RecordBatchExt};
use lance_core::{Result, ROW_ID};
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
    replicate_factor: f32,
    max_replica: usize,
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
            replicate_factor: 1.02,
            max_replica: 8,
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

    /// Compute the partition for each row in the input Matrix.
    ///
    #[instrument(level = "debug", skip(data))]
    pub(super) async fn compute_multiple_partitions(
        &self,
        data: &MatrixView<T>,
    ) -> GenericListArray<i32> {
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

        let chunk_size = num_rows.div_ceil(chunks);
        let stride = chunk_size * dimension;

        let result: Vec<Vec<Option<Vec<u32>>>> = stream::iter(0..chunks)
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

                compute_multiple_partitions::<T>(
                    centroids,
                    data,
                    dimension,
                    metric_type,
                    self.replicate_factor,
                    self.max_replica,
                )
                .in_current_span()
                .await
            })
            .buffered(chunks)
            .collect::<Vec<_>>()
            .await;
        let result = result.into_iter().flatten().collect::<Vec<_>>();

        let mut builder = ListBuilder::new(UInt32Builder::new());
        for part_ids in result {
            match part_ids {
                Some(part_ids) => {
                    builder.append_value(part_ids.into_iter().map(|x| Some(x)));
                }
                None => builder.append_null(),
            }
        }

        builder.finish()
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
        let part_ids = self.compute_multiple_partitions(&mat).await;
        let row_ids = batch
            .column_by_name(ROW_ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();

        let total_num_rows = part_ids.values().len();
        let mut row_id_builder = UInt64Builder::with_capacity(total_num_rows);
        let mut part_id_builder = UInt32Builder::with_capacity(total_num_rows);
        for i in 0..row_ids.len() {
            let part_ids = part_ids.value(i);
            let part_ids: &UInt32Array = part_ids.as_primitive();
            for part_id in part_ids {
                row_id_builder.append_value(row_ids.value(i));
                part_id_builder.append_value(part_id.unwrap());
            }
        }

        let field = Field::new(PART_ID_COLUMN, part_ids.value_type().clone(), true);
        Ok(batch
            .drop_column(&self.input_column)?
            .replace_column_by_name(ROW_ID, Arc::new(row_id_builder.finish()))?
            .try_with_column(field, Arc::new(part_id_builder.finish()))?)
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
