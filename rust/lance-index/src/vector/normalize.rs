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

use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::RecordBatch;
use lance_arrow::RecordBatchExt;
use lance_core::Result;
use lance_linalg::kernels::normalize_fsl;
use snafu::{location, Location};

use crate::vector::transform::Transformer;

#[derive(Debug)]
pub(crate) struct NormalizeTransformer {
    input_column: String,
}

impl NormalizeTransformer {
    pub fn new(col: &str) -> Self {
        Self {
            input_column: col.to_owned(),
        }
    }
}

#[async_trait::async_trait]
impl Transformer for NormalizeTransformer {
    async fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let column =
            batch
                .column_by_name(&self.input_column)
                .ok_or_else(|| lance_core::Error::Schema {
                    message: format!(
                        "column {} does not exist in the record batch",
                        self.input_column
                    ),
                    location: location!(),
                })?;

        let fsl = column
            .as_fixed_size_list_opt()
            .ok_or_else(|| lance_core::Error::Schema {
                message: format!("column {} is not a fixed size list", self.input_column),
                location: location!(),
            })?;
        let norm = normalize_fsl(fsl)?;
        Ok(batch.replace_column_by_name(&self.input_column, Arc::new(norm))?)
    }
}
