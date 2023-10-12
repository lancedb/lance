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

use arrow_array::RecordBatch;
use lance_core::{Error, Result};

use super::transform::Transformer;


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

impl Transformer for PQTransformer {
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let input_arr = batch.column_by_name(&self.input_column)
        todo!()
    }
}
