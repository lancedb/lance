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

use arrow_array::{Array, ArrayRef};
use lance_arrow::ArrowFloatType;
use lance_core::Result;

#[async_trait::async_trait]
pub trait Quantizer<T: ArrowFloatType> {
    /// Transform a vector column to quantized code column.
    ///
    /// Parameters
    /// ----------
    /// *data*: vector array, must be a `FixedSizeListArray`
    ///
    /// Returns
    /// -------
    ///   quantized code column
    async fn quantize(&self, data: &dyn Array) -> Result<ArrayRef>;

    /// Get the dimension of the quantized codes.
    fn code_dim(&self) -> usize;

    /// Get the number of bits used to represent each code.
    fn code_bits(&self) -> u16;
}
