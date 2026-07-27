// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_schema::{DataType, Field as ArrowField};
use lance_core::{Result, datatypes::Field};
use lance_encoding::{
    buffer::LanceBuffer,
    compression::{BlockCompressor, CompressionStrategy, DefaultCompressionStrategy},
    data::{BlockInfo, DataBlock, FixedWidthDataBlock},
    encodings::logical::primitive::{fullzip::PerValueCompressor, miniblock::MiniBlockCompressor},
    format::ProtobufUtils21,
};

#[derive(Debug)]
struct IdentityBlockCompressor;

impl BlockCompressor for IdentityBlockCompressor {
    fn compress(&self, data: DataBlock) -> Result<Option<LanceBuffer>> {
        let DataBlock::FixedWidth(data) = data else {
            panic!("test compressor only accepts fixed-width data");
        };
        Ok(Some(data.data))
    }
}

#[derive(Debug, Default)]
struct CustomCompressionStrategy {
    fallback: DefaultCompressionStrategy,
}

impl CompressionStrategy for CustomCompressionStrategy {
    fn create_block_compressor(
        &self,
        _field: &Field,
        data: &DataBlock,
    ) -> Result<(
        Box<dyn BlockCompressor>,
        lance_encoding::format::pb21::CompressiveEncoding,
    )> {
        let DataBlock::FixedWidth(data) = data else {
            panic!("test strategy only accepts fixed-width data");
        };
        Ok((
            Box::new(IdentityBlockCompressor),
            ProtobufUtils21::flat(data.bits_per_value, None),
        ))
    }

    fn create_per_value(
        &self,
        field: &Field,
        data: &DataBlock,
    ) -> Result<Box<dyn PerValueCompressor>> {
        self.fallback.create_per_value(field, data)
    }

    fn create_miniblock_compressor(
        &self,
        field: &Field,
        data: &DataBlock,
    ) -> Result<Box<dyn MiniBlockCompressor>> {
        self.fallback.create_miniblock_compressor(field, data)
    }
}

#[test]
fn public_strategy_returns_the_frozen_block_compressor() {
    let field = Field::try_from(&ArrowField::new("values", DataType::UInt32, false)).unwrap();
    let values = vec![3_u32, 5, 8, 13];
    let data = DataBlock::FixedWidth(FixedWidthDataBlock {
        data: LanceBuffer::reinterpret_vec(values.clone()),
        bits_per_value: 32,
        num_values: values.len() as u64,
        block_info: BlockInfo::default(),
    });
    let strategy: Box<dyn CompressionStrategy> = Box::new(CustomCompressionStrategy::default());

    let (compressor, _) = strategy.create_block_compressor(&field, &data).unwrap();
    let payload = compressor.compress(data).unwrap().unwrap();
    assert_eq!(payload.borrow_to_typed_slice::<u32>().as_ref(), values);
}
