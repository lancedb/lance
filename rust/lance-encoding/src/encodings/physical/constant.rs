// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Routines for compressing and decompressing constant-encoded data

use crate::{
    buffer::LanceBuffer,
    compression::BlockDecompressor,
    data::{ConstantDataBlock, DataBlock},
};

use lance_core::Result;

/// A decompressor for constant-encoded data
#[derive(Debug)]
pub struct ConstantDecompressor {
    scalar: LanceBuffer,
}

impl ConstantDecompressor {
    pub fn new(scalar: LanceBuffer) -> Self {
        Self {
            scalar: scalar.into_borrowed(),
        }
    }
}

impl BlockDecompressor for ConstantDecompressor {
    fn decompress(&self, _data: LanceBuffer, num_values: u64) -> Result<DataBlock> {
        Ok(DataBlock::Constant(ConstantDataBlock {
            data: self.scalar.try_clone().unwrap(),
            num_values,
        }))
    }
}
