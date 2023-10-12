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

use std::fmt::{Debug, Formatter};

use arrow_array::{cast::AsArray, types::Float32Type, Array, RecordBatch};
use async_trait::async_trait;
use lance_core::{Error, Result};
use lance_linalg::MatrixView;

use super::ProductQuantizer;
use crate::vector::transform::Transformer;

pub struct PQTransformer {
    quantizer: ProductQuantizer,
    input_column: String,
    output_column: String,
}

impl PQTransformer {
    pub fn new(quantizer: ProductQuantizer, input_column: &str, output_column: &str) -> Self {
        Self {
            quantizer,
            input_column: input_column.to_owned(),
            output_column: output_column.to_owned(),
        }
    }
}

impl Debug for PQTransformer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PQTransformer(input={}, output={})",
            self.input_column, self.output_column
        )
    }
}

#[async_trait]
impl Transformer for PQTransformer {
    async fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let input_arr = batch
            .column_by_name(&self.input_column)
            .ok_or(Error::Index {
                message: format!(
                    "PQ Transform: column {} not found in batch",
                    self.input_column
                ),
            })?;
        let data: MatrixView<Float32Type> = input_arr
            .as_fixed_size_list_opt()
            .ok_or(Error::Index {
                message: format!(
                    "PQ Transform: column {} is not a fixed size list, got {}",
                    self.input_column,
                    input_arr.data_type(),
                ),
            })?
            .try_into()?;
        let pq_code = self.quantizer.transform(&data).await?;
        todo!()
    }
}

#[cfg(test)]
mod tests {}
