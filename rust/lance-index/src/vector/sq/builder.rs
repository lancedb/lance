// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::vector::quantizer::QuantizerBuildParams;

#[derive(Debug, Clone)]
pub struct SQBuildParams {
    /// Number of bits of scaling range.
    pub num_bits: u16,

    /// Sample rate for training.
    pub sample_rate: usize,
}

impl Default for SQBuildParams {
    fn default() -> Self {
        Self {
            num_bits: 8,
            sample_rate: 256,
        }
    }
}

impl QuantizerBuildParams for SQBuildParams {
    fn sample_size(&self) -> usize {
        self.sample_rate * 2usize.pow(self.num_bits as u32)
    }
}
