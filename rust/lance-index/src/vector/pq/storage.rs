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

//! Product Quantization storage
//!
//! Used in graphs.

use std::sync::Arc;

use arrow_array::{
    cast::AsArray,
    types::{Float32Type, UInt8Type},
    FixedSizeListArray, Float32Array, UInt64Array, UInt8Array,
};

use super::{ProductQuantizer, ProductQuantizerImpl};
use lance_core::Result;

/// Product Quantization Storage
pub struct ProductQuantizationStorage {
    codebook: Arc<Float32Array>,

    pq_code: Arc<UInt8Array>,
    row_ids: Arc<UInt64Array>,
}

impl ProductQuantizationStorage {
    pub async fn new(
        quantizer: &ProductQuantizerImpl<Float32Type>,
        vectors: &FixedSizeListArray,
        row_ids: Arc<UInt64Array>,
    ) -> Result<Self> {
        let codebook = quantizer.codebook.clone();
        let pq_code = quantizer.transform(&vectors).await?;
        let pq_code = pq_code.as_primitive::<UInt8Type>().clone();
        Ok(Self {
            codebook,
            pq_code: pq_code.into(),
            row_ids,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}