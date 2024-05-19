// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Transform of a Vector Input with partition IDs.

use std::ops::Range;
use std::sync::Arc;

use arrow_array::{
    cast::AsArray, types::UInt32Type, Array, FixedSizeListArray, RecordBatch, UInt32Array,
};
use arrow_schema::Field;
use snafu::{location, Location};
use tracing::instrument;

use lance_arrow::RecordBatchExt;
use lance_core::Result;
use lance_linalg::distance::DistanceType;
use lance_linalg::kmeans::compute_partitions_arrow_array;

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
pub struct IvfTransformer {
    centroids: FixedSizeListArray,
    distance_type: DistanceType,
    input_column: String,
    output_column: String,
}

impl IvfTransformer {
    pub fn new(
        centroids: FixedSizeListArray,
        distance_type: DistanceType,
        input_column: impl AsRef<str>,
    ) -> Self {
        Self {
            centroids,
            distance_type,
            input_column: input_column.as_ref().to_owned(),
            output_column: PART_ID_COLUMN.to_owned(),
        }
    }

    /// Compute the partition for each row in the input Matrix.
    ///
    #[instrument(level = "debug", skip(data))]
    pub(super) fn compute_partitions(&self, data: &FixedSizeListArray) -> UInt32Array {
        compute_partitions_arrow_array(&self.centroids, data, self.distance_type)
            .expect("failed to compute partitions")
            .into()
    }
}
impl Transformer for IvfTransformer {
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
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

        let part_ids = self.compute_partitions(fsl);
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

impl Transformer for PartitionFilter {
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
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
