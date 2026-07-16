// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::vector::quantizer::QuantizerBuildParams;

#[derive(Debug, Clone)]
pub struct SQBuildParams {
    /// Number of bits of scaling range.
    pub num_bits: u16,

    /// Sample rate for training.
    pub sample_rate: usize,

    /// User-provided, globally-trained quantization bounds.
    ///
    /// When set, per-build training is skipped and these exact bounds are used,
    /// so codes are comparable across independently built shards of a
    /// distributed index build. Mirrors `PQBuildParams::codebook`.
    pub bounds: Option<std::ops::Range<f64>>,
}

impl From<&SQBuildParams> for crate::pb::vector_index_details::ScalarQuantization {
    fn from(params: &SQBuildParams) -> Self {
        Self {
            num_bits: params.num_bits as u32,
        }
    }
}

impl Default for SQBuildParams {
    fn default() -> Self {
        Self {
            num_bits: 8,
            sample_rate: 256,
            bounds: None,
        }
    }
}

impl SQBuildParams {
    /// Create build params carrying pre-trained bounds, skipping training.
    pub fn with_bounds(num_bits: u16, bounds: std::ops::Range<f64>) -> Self {
        Self {
            num_bits,
            bounds: Some(bounds),
            ..Default::default()
        }
    }
}

impl QuantizerBuildParams for SQBuildParams {
    fn sample_size(&self) -> usize {
        self.sample_rate * 2usize.pow(self.num_bits as u32)
    }
}
