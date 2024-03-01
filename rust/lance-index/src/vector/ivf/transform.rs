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

use arrow_array::cast::AsArray;
use std::sync::Arc;

use arrow_array::{Array, RecordBatch};
use arrow_schema::Field;
use lance_arrow::RecordBatchExt;
use lance_core::Result;
use snafu::{location, Location};

use super::Ivf;
use super::PART_ID_COLUMN;
use crate::vector::transform::Transformer;

/// Ivf Transformer
///
/// It transforms a Vector column, specified by the input data, into a column of partition IDs.
///
/// If the partition ID ("__ivf_part_id") column is already present in the Batch,
/// this transform is a Noop.
///
#[derive(Debug)]
pub struct IvfTransformer {
    ivf: Arc<dyn Ivf>,
    input_column: String,
    output_column: String,
}

impl IvfTransformer {
    pub fn new(ivf: Arc<dyn Ivf>, input_column: impl AsRef<str>) -> Self {
        Self {
            ivf,
            input_column: input_column.as_ref().to_owned(),
            output_column: PART_ID_COLUMN.to_owned(),
        }
    }
}

#[async_trait::async_trait]
impl Transformer for IvfTransformer {
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

        let part_ids = self.ivf.compute_partitions(fsl).await?;
        let field = Field::new(PART_ID_COLUMN, part_ids.data_type().clone(), true);
        Ok(batch.try_with_column(field, Arc::new(part_ids))?)
    }
}
