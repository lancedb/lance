// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_schema::DataType;
use futures::{future::BoxFuture, FutureExt};
use lance_core::Result;
use log::trace;

use crate::{
    data::{BlockInfo, DataBlock, FixedSizeListBlock, FixedWidthDataBlock},
    decoder::{PageScheduler, PerValueDecompressor, PrimitivePageDecoder},
    encoder::{ArrayEncoder, EncodedArray, PerValueCompressor, PerValueDataBlock},
    format::{pb, ProtobufUtils},
    EncodingsIo,
};

/// A scheduler for fixed size lists of primitive values
///
/// This scheduler is, itself, primitive
#[derive(Debug)]
pub struct FixedListScheduler {
    items_scheduler: Box<dyn PageScheduler>,
    dimension: u32,
}

impl FixedListScheduler {
    pub fn new(items_scheduler: Box<dyn PageScheduler>, dimension: u32) -> Self {
        Self {
            items_scheduler,
            dimension,
        }
    }
}

impl PageScheduler for FixedListScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u64>],
        scheduler: &Arc<dyn EncodingsIo>,
        top_level_row: u64,
    ) -> BoxFuture<'static, Result<Box<dyn PrimitivePageDecoder>>> {
        let expanded_ranges = ranges
            .iter()
            .map(|range| (range.start * self.dimension as u64)..(range.end * self.dimension as u64))
            .collect::<Vec<_>>();
        trace!(
            "Expanding {} fsl ranges across {}..{} to item ranges across {}..{}",
            ranges.len(),
            ranges[0].start,
            ranges[ranges.len() - 1].end,
            expanded_ranges[0].start,
            expanded_ranges[expanded_ranges.len() - 1].end
        );
        let inner_page_decoder =
            self.items_scheduler
                .schedule_ranges(&expanded_ranges, scheduler, top_level_row);
        let dimension = self.dimension;
        async move {
            let items_decoder = inner_page_decoder.await?;
            Ok(Box::new(FixedListDecoder {
                items_decoder,
                dimension: dimension as u64,
            }) as Box<dyn PrimitivePageDecoder>)
        }
        .boxed()
    }
}

pub struct FixedListDecoder {
    items_decoder: Box<dyn PrimitivePageDecoder>,
    dimension: u64,
}

impl PrimitivePageDecoder for FixedListDecoder {
    fn decode(&self, rows_to_skip: u64, num_rows: u64) -> Result<DataBlock> {
        let rows_to_skip = rows_to_skip * self.dimension;
        let num_child_rows = num_rows * self.dimension;
        let child_data = self.items_decoder.decode(rows_to_skip, num_child_rows)?;
        Ok(DataBlock::FixedSizeList(FixedSizeListBlock {
            child: Box::new(child_data),
            dimension: self.dimension,
        }))
    }
}

#[derive(Debug)]
pub struct FslEncoder {
    items_encoder: Box<dyn ArrayEncoder>,
    dimension: u32,
}

impl FslEncoder {
    pub fn new(items_encoder: Box<dyn ArrayEncoder>, dimension: u32) -> Self {
        Self {
            items_encoder,
            dimension,
        }
    }
}

impl ArrayEncoder for FslEncoder {
    fn encode(
        &self,
        data: DataBlock,
        data_type: &DataType,
        buffer_index: &mut u32,
    ) -> Result<EncodedArray> {
        let inner_type = match data_type {
            DataType::FixedSizeList(inner_field, _) => inner_field.data_type().clone(),
            _ => panic!("Expected fixed size list data type and got {}", data_type),
        };
        let data = data.as_fixed_size_list().unwrap();
        let child = *data.child;

        let encoded_data = self
            .items_encoder
            .encode(child, &inner_type, buffer_index)?;

        let data = DataBlock::FixedSizeList(FixedSizeListBlock {
            child: Box::new(encoded_data.data),
            dimension: self.dimension as u64,
        });

        let encoding = ProtobufUtils::fixed_size_list(encoded_data.encoding, self.dimension as u64);
        Ok(EncodedArray { data, encoding })
    }
}

/// A compressor for primitive FSLs that flattens each list into a
/// single value.  If the inner list has validity then the validity
/// is zipped in with the values.
///
/// In other words, if the list is FSL<u8?, 2> [[0, NULL], [4, 10]] then the
/// two buffers start as:
///
/// values: 0x00 0x?? 0x04 0x0A
/// validity: 0b1011
///
/// The output will be:
///
/// zipped: 0x01 0x00 0x00 0x?? 0x01 0x04 0x01 0x0A
///
/// Note that we expand validity to be at least a byte per value so this
/// approach is not ideal for small lists, though we should be using mini-block
/// for small lists anyways.
#[derive(Debug)]
pub struct FslPerValueCompressor {
    items_compressor: Box<dyn PerValueCompressor>,
    dimension: u64,
}

impl FslPerValueCompressor {
    pub fn new(items_compressor: Box<dyn PerValueCompressor>, dimension: u64) -> Self {
        Self {
            items_compressor,
            dimension,
        }
    }
}

impl PerValueCompressor for FslPerValueCompressor {
    fn compress(&self, data: DataBlock) -> Result<(PerValueDataBlock, pb::ArrayEncoding)> {
        let mut data = data.as_fixed_size_list().unwrap();
        let flattened = match data.child.as_mut() {
            DataBlock::FixedWidth(fixed_width) => DataBlock::FixedWidth(FixedWidthDataBlock {
                bits_per_value: fixed_width.bits_per_value * self.dimension,
                data: fixed_width.data.borrow_and_clone(),
                block_info: BlockInfo::new(),
                num_values: fixed_width.num_values / self.dimension,
            }),
            DataBlock::VariableWidth(_) => todo!("GH-3111: FSL with variable inner type"),
            DataBlock::Nullable(_) => todo!("GH-3112: FSL with nullable inner type"),
            DataBlock::FixedSizeList(_) => todo!("GH-3113: Nested FSLs"),
            _ => unreachable!(),
        };
        let (compressed, encoding) = self.items_compressor.compress(flattened)?;
        let wrapped_encoding = ProtobufUtils::fixed_size_list(encoding, self.dimension);

        Ok((compressed, wrapped_encoding))
    }
}

/// Reversed the process described in [`FslPerValueCompressor`]
#[derive(Debug)]
pub struct FslPerValueDecompressor {
    items_decompressor: Box<dyn PerValueDecompressor>,
    dimension: u64,
}

impl FslPerValueDecompressor {
    pub fn new(items_decompressor: Box<dyn PerValueDecompressor>, dimension: u64) -> Self {
        Self {
            items_decompressor,
            dimension,
        }
    }
}

impl PerValueDecompressor for FslPerValueDecompressor {
    fn decompress(&self, data: crate::buffer::LanceBuffer, num_values: u64) -> Result<DataBlock> {
        let decompressed = self.items_decompressor.decompress(data, num_values)?;
        let unflattened = match decompressed {
            DataBlock::FixedWidth(fixed_width) => DataBlock::FixedWidth(FixedWidthDataBlock {
                bits_per_value: fixed_width.bits_per_value / self.dimension,
                data: fixed_width.data,
                block_info: BlockInfo::new(),
                num_values: fixed_width.num_values * self.dimension,
            }),
            _ => todo!(),
        };
        Ok(DataBlock::FixedSizeList(FixedSizeListBlock {
            child: Box::new(unflattened),
            dimension: self.dimension,
        }))
    }

    fn bits_per_value(&self) -> u64 {
        self.items_decompressor.bits_per_value()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field};

    use crate::{testing::check_round_trip_encoding_random, version::LanceFileVersion};

    const PRIMITIVE_TYPES: &[DataType] = &[DataType::Int8, DataType::Float32, DataType::Float64];

    #[test_log::test(tokio::test)]
    async fn test_value_fsl_primitive() {
        for data_type in PRIMITIVE_TYPES {
            let inner_field = Field::new("item", data_type.clone(), true);
            let data_type = DataType::FixedSizeList(Arc::new(inner_field), 16);
            let field = Field::new("", data_type, false);
            check_round_trip_encoding_random(field, LanceFileVersion::V2_0).await;
        }
    }
}
