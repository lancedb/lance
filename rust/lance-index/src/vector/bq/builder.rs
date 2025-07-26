// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::Result;
use lance_file::v2::reader::FileReader;

use crate::vector::bq::storage::{RabbitQuantizationMetadata, RabbitQuantizationStorage};
use crate::vector::bq::RabbitQuantizationBuildParams;
use crate::vector::quantizer::{Quantization, Quantizer, QuantizerBuildParams, QuantizerMetadata};

pub struct RabbitQuantizer {
    metadata: RabbitQuantizationMetadata,
}

impl RabbitQuantizer {
    pub fn new(num_bits: u16, dim: usize) -> Self {
        Self {
            metadata: RabbitQuantizationMetadata::new(num_bits, dim),
        }
    }
}

impl Quantization for RabbitQuantizer {
    type BuildParams = RabbitQuantizationBuildParams;
    type Metadata = RabbitQuantizationMetadata;
    type Storage = RabbitQuantizationStorage;
}
