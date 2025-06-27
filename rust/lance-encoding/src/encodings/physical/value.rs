// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_buffer::{bit_util, BooleanBufferBuilder};
use snafu::location;

use crate::buffer::LanceBuffer;
use crate::compression::{
    BlockCompressor, BlockDecompressor, FixedPerValueDecompressor, MiniBlockDecompressor,
};
use crate::data::{
    BlockInfo, DataBlock, FixedSizeListBlock, FixedWidthDataBlock, NullableDataBlock,
};
use crate::encodings::logical::primitive::fullzip::{PerValueCompressor, PerValueDataBlock};
use crate::encodings::logical::primitive::miniblock::{
    MiniBlockChunk, MiniBlockCompressed, MiniBlockCompressor, MAX_MINIBLOCK_BYTES,
    MAX_MINIBLOCK_VALUES,
};
use crate::format::pb::{self, ArrayEncoding};
use crate::format::ProtobufUtils;

use lance_core::{Error, Result};

/// A compression strategy that writes fixed-width data as-is (no compression)
#[derive(Debug, Default)]
pub struct ValueEncoder {}

impl ValueEncoder {
    /// Use the largest chunk we can smaller than 4KiB
    fn find_log_vals_per_chunk(bytes_per_word: u64, values_per_word: u64) -> (u64, u64) {
        let mut size_bytes = 2 * bytes_per_word;
        let (mut log_num_vals, mut num_vals) = match values_per_word {
            1 => (1, 2),
            8 => (3, 8),
            _ => unreachable!(),
        };

        // If the type is so wide that we can't even fit 2 values we shouldn't be here
        assert!(size_bytes < MAX_MINIBLOCK_BYTES);

        while 2 * size_bytes < MAX_MINIBLOCK_BYTES && 2 * num_vals <= MAX_MINIBLOCK_VALUES {
            log_num_vals += 1;
            size_bytes *= 2;
            num_vals *= 2;
        }

        (log_num_vals, num_vals)
    }

    fn chunk_data(data: FixedWidthDataBlock) -> MiniBlockCompressed {
        // Usually there are X bytes per value.  However, when working with boolean
        // or FSL<boolean> we might have some number of bits per value that isn't
        // divisible by 8.  In this case, to avoid chunking in the middle of a byte
        // we calculate how many 8-value words we can fit in a chunk.
        let (bytes_per_word, values_per_word) = if data.bits_per_value % 8 == 0 {
            (data.bits_per_value / 8, 1)
        } else {
            (data.bits_per_value, 8)
        };

        // Aim for 4KiB chunks
        let (log_vals_per_chunk, vals_per_chunk) =
            Self::find_log_vals_per_chunk(bytes_per_word, values_per_word);
        let num_chunks = bit_util::ceil(data.num_values as usize, vals_per_chunk as usize);
        debug_assert_eq!(vals_per_chunk % values_per_word, 0);
        let bytes_per_chunk = bytes_per_word * (vals_per_chunk / values_per_word);
        let bytes_per_chunk = u16::try_from(bytes_per_chunk).unwrap();

        let data_buffer = data.data;

        let mut row_offset = 0;
        let mut chunks = Vec::with_capacity(num_chunks);

        let mut bytes_counter = 0;
        loop {
            if row_offset + vals_per_chunk <= data.num_values {
                chunks.push(MiniBlockChunk {
                    log_num_values: log_vals_per_chunk as u8,
                    buffer_sizes: vec![bytes_per_chunk],
                });
                row_offset += vals_per_chunk;
                bytes_counter += bytes_per_chunk as u64;
            } else {
                // Final chunk, special values
                let num_bytes = data_buffer.len() as u64 - bytes_counter;
                let num_bytes = u16::try_from(num_bytes).unwrap();
                chunks.push(MiniBlockChunk {
                    log_num_values: 0,
                    buffer_sizes: vec![num_bytes],
                });
                break;
            }
        }

        MiniBlockCompressed {
            chunks,
            data: vec![data_buffer],
            num_values: data.num_values,
        }
    }
}

#[derive(Debug)]
struct MiniblockFslLayer {
    validity: Option<LanceBuffer>,
    dimension: u64,
}

/// This impl deals with encoding FSL<FSL<...<FSL<FixedWidth>>>> data as a mini-block compressor.
/// The tricky part of FSL data is that we want to include inner validity buffers (we don't want these
/// to be part of the rep-def because that usually ends up being more expensive).
///
/// The resulting mini-block will, instead of having a single buffer, have X + 1 buffers where X is
/// the number of FSL layers that contain validity.
///
/// In the simple case where there is no validity inside the FSL layers, all we are doing here is flattening
/// the FSL layers into a single buffer.
///
/// Also: We don't allow a row to be broken across chunks.  This typically isn't too big of a deal since we
/// are usually dealing with relatively small vectors if we are using mini-block.
///
/// Note: when we do have validity we have to make copies of the validity buffers because they are bit buffers
/// and we need to bit slice them which requires copies or offsets.  Paying the price at write time to make
/// the copies is better than paying the price at read time to do the bit slicing.
impl ValueEncoder {
    fn make_fsl_encoding(layers: &[MiniblockFslLayer], bits_per_value: u64) -> ArrayEncoding {
        let mut encoding = ProtobufUtils::flat_encoding(bits_per_value, 0, None);
        for layer in layers.iter().rev() {
            let has_validity = layer.validity.is_some();
            let dimension = layer.dimension;
            encoding = ProtobufUtils::fsl_encoding(dimension, encoding, has_validity);
        }
        encoding
    }

    fn extract_fsl_chunk(
        data: &FixedWidthDataBlock,
        layers: &[MiniblockFslLayer],
        row_offset: usize,
        num_rows: usize,
        validity_buffers: &mut [Vec<u8>],
    ) -> Vec<u16> {
        let mut row_offset = row_offset;
        let mut num_values = num_rows;
        let mut buffer_counter = 0;
        let mut buffer_sizes = Vec::with_capacity(validity_buffers.len() + 1);
        for layer in layers {
            row_offset *= layer.dimension as usize;
            num_values *= layer.dimension as usize;
            if let Some(validity) = &layer.validity {
                let validity_slice = validity
                    .try_clone()
                    .unwrap()
                    .bit_slice_le_with_length(row_offset, num_values);
                validity_buffers[buffer_counter].extend_from_slice(&validity_slice);
                buffer_sizes.push(validity_slice.len() as u16);
                buffer_counter += 1;
            }
        }

        let bits_in_chunk = data.bits_per_value * num_values as u64;
        let bytes_in_chunk = bits_in_chunk.div_ceil(8);
        let bytes_in_chunk = u16::try_from(bytes_in_chunk).unwrap();
        buffer_sizes.push(bytes_in_chunk);

        buffer_sizes
    }

    fn chunk_fsl(
        data: FixedWidthDataBlock,
        layers: Vec<MiniblockFslLayer>,
        num_rows: u64,
    ) -> (MiniBlockCompressed, ArrayEncoding) {
        // Count size to calculate rows per chunk
        let mut ceil_bytes_validity = 0;
        let mut cum_dim = 1;
        let mut num_validity_buffers = 0;
        for layer in &layers {
            cum_dim *= layer.dimension;
            if layer.validity.is_some() {
                ceil_bytes_validity += cum_dim.div_ceil(8);
                num_validity_buffers += 1;
            }
        }
        // It's an estimate because validity buffers may have some padding bits
        let cum_bits_per_value = data.bits_per_value * cum_dim;
        let (cum_bytes_per_word, vals_per_word) = if cum_bits_per_value % 8 == 0 {
            (cum_bits_per_value / 8, 1)
        } else {
            (cum_bits_per_value, 8)
        };
        let est_bytes_per_word = (ceil_bytes_validity * vals_per_word) + cum_bytes_per_word;
        let (log_rows_per_chunk, rows_per_chunk) =
            Self::find_log_vals_per_chunk(est_bytes_per_word, vals_per_word);

        let num_chunks = num_rows.div_ceil(rows_per_chunk) as usize;

        // Allocate buffers for validity, these will be slightly bigger than the input validity buffers
        let mut chunks = Vec::with_capacity(num_chunks);
        let mut validity_buffers: Vec<Vec<u8>> = Vec::with_capacity(num_validity_buffers);
        cum_dim = 1;
        for layer in &layers {
            cum_dim *= layer.dimension;
            if let Some(validity) = &layer.validity {
                let layer_bytes_validity = cum_dim.div_ceil(8);
                let validity_with_padding =
                    layer_bytes_validity as usize * num_chunks * rows_per_chunk as usize;
                debug_assert!(validity_with_padding >= validity.len());
                validity_buffers.push(Vec::with_capacity(
                    layer_bytes_validity as usize * num_chunks,
                ));
            }
        }

        // Now go through and extract validity buffers
        let mut row_offset = 0;
        while row_offset + rows_per_chunk <= num_rows {
            let buffer_sizes = Self::extract_fsl_chunk(
                &data,
                &layers,
                row_offset as usize,
                rows_per_chunk as usize,
                &mut validity_buffers,
            );
            row_offset += rows_per_chunk;
            chunks.push(MiniBlockChunk {
                log_num_values: log_rows_per_chunk as u8,
                buffer_sizes,
            })
        }
        let rows_in_chunk = num_rows - row_offset;
        if rows_in_chunk > 0 {
            let buffer_sizes = Self::extract_fsl_chunk(
                &data,
                &layers,
                row_offset as usize,
                rows_in_chunk as usize,
                &mut validity_buffers,
            );
            chunks.push(MiniBlockChunk {
                log_num_values: 0,
                buffer_sizes,
            });
        }

        let encoding = Self::make_fsl_encoding(&layers, data.bits_per_value);
        // Finally, add the data buffer
        let buffers = validity_buffers
            .into_iter()
            .map(LanceBuffer::Owned)
            .chain(std::iter::once(data.data))
            .collect::<Vec<_>>();

        (
            MiniBlockCompressed {
                chunks,
                data: buffers,
                num_values: num_rows,
            },
            encoding,
        )
    }

    fn miniblock_fsl(data: DataBlock) -> (MiniBlockCompressed, ArrayEncoding) {
        let num_rows = data.num_values();
        let fsl = data.as_fixed_size_list().unwrap();
        let mut layers = Vec::new();
        let mut child = *fsl.child;
        let mut cur_layer = MiniblockFslLayer {
            validity: None,
            dimension: fsl.dimension,
        };
        loop {
            if let DataBlock::Nullable(nullable) = child {
                cur_layer.validity = Some(nullable.nulls);
                child = *nullable.data;
            }
            match child {
                DataBlock::FixedSizeList(inner) => {
                    layers.push(cur_layer);
                    cur_layer = MiniblockFslLayer {
                        validity: None,
                        dimension: inner.dimension,
                    };
                    child = *inner.child;
                }
                DataBlock::FixedWidth(inner) => {
                    layers.push(cur_layer);
                    return Self::chunk_fsl(inner, layers, num_rows);
                }
                _ => unreachable!("Unexpected data block type in value encoder's miniblock_fsl"),
            }
        }
    }
}

struct PerValueFslValidityIter {
    buffer: LanceBuffer,
    bits_per_row: usize,
    offset: usize,
}

/// In this section we deal with per-value encoding of FSL<FSL<...<FSL<FixedWidth>>>> data.
///
/// It's easier than mini-block.  All we need to do is flatten the FSL layers into a single buffer.
/// This includes any validity buffers we encounter on the way.
impl ValueEncoder {
    fn fsl_to_encoding(fsl: &FixedSizeListBlock) -> ArrayEncoding {
        let mut inner = fsl.child.as_ref();
        let mut has_validity = false;
        inner = match inner {
            DataBlock::Nullable(nullable) => {
                has_validity = true;
                nullable.data.as_ref()
            }
            _ => inner,
        };
        let inner_encoding = match inner {
            DataBlock::FixedWidth(fixed_width) => {
                ProtobufUtils::flat_encoding(fixed_width.bits_per_value, 0, None)
            }
            DataBlock::FixedSizeList(inner) => Self::fsl_to_encoding(inner),
            _ => unreachable!("Unexpected data block type in value encoder's fsl_to_encoding"),
        };
        ProtobufUtils::fsl_encoding(fsl.dimension, inner_encoding, has_validity)
    }

    fn simple_per_value_fsl(fsl: FixedSizeListBlock) -> (PerValueDataBlock, ArrayEncoding) {
        // The simple case is zero-copy, we just return the flattened inner buffer
        let encoding = Self::fsl_to_encoding(&fsl);
        let num_values = fsl.num_values();
        let mut child = *fsl.child;
        let mut cum_dim = 1;
        loop {
            cum_dim *= fsl.dimension;
            match child {
                DataBlock::Nullable(nullable) => {
                    child = *nullable.data;
                }
                DataBlock::FixedSizeList(inner) => {
                    child = *inner.child;
                }
                DataBlock::FixedWidth(inner) => {
                    let data = FixedWidthDataBlock {
                        bits_per_value: inner.bits_per_value * cum_dim,
                        num_values,
                        data: inner.data,
                        block_info: BlockInfo::new(),
                    };
                    return (PerValueDataBlock::Fixed(data), encoding);
                }
                _ => unreachable!(
                    "Unexpected data block type in value encoder's simple_per_value_fsl"
                ),
            }
        }
    }

    fn nullable_per_value_fsl(fsl: FixedSizeListBlock) -> (PerValueDataBlock, ArrayEncoding) {
        // If there are nullable inner values then we need to zip the validity with the values
        let encoding = Self::fsl_to_encoding(&fsl);
        let num_values = fsl.num_values();
        let mut bytes_per_row = 0;
        let mut cum_dim = 1;
        let mut current = fsl;
        let mut validity_iters: Vec<PerValueFslValidityIter> = Vec::new();
        let data_bytes_per_row: usize;
        let data_buffer: LanceBuffer;
        loop {
            cum_dim *= current.dimension;
            let mut child = *current.child;
            if let DataBlock::Nullable(nullable) = child {
                // Each item will need this many bytes of validity prepended to it
                bytes_per_row += cum_dim.div_ceil(8) as usize;
                validity_iters.push(PerValueFslValidityIter {
                    buffer: nullable.nulls,
                    bits_per_row: cum_dim as usize,
                    offset: 0,
                });
                child = *nullable.data;
            };
            match child {
                DataBlock::FixedSizeList(inner) => {
                    current = inner;
                }
                DataBlock::FixedWidth(fixed_width) => {
                    data_bytes_per_row =
                        (fixed_width.bits_per_value.div_ceil(8) * cum_dim) as usize;
                    bytes_per_row += data_bytes_per_row;
                    data_buffer = fixed_width.data;
                    break;
                }
                _ => unreachable!(
                    "Unexpected data block type in value encoder's nullable_per_value_fsl: {:?}",
                    child
                ),
            }
        }

        let bytes_needed = bytes_per_row * num_values as usize;
        let mut zipped = Vec::with_capacity(bytes_needed);
        let data_slice = &data_buffer;
        // Hopefully values are pretty large so we don't iterate this loop _too_ many times
        for i in 0..num_values as usize {
            for validity in validity_iters.iter_mut() {
                let validity_slice = validity
                    .buffer
                    .bit_slice_le_with_length(validity.offset, validity.bits_per_row);
                zipped.extend_from_slice(&validity_slice);
                validity.offset += validity.bits_per_row;
            }
            let start = i * data_bytes_per_row;
            let end = start + data_bytes_per_row;
            zipped.extend_from_slice(&data_slice[start..end]);
        }

        let zipped = LanceBuffer::Owned(zipped);
        let data = PerValueDataBlock::Fixed(FixedWidthDataBlock {
            bits_per_value: bytes_per_row as u64 * 8,
            num_values,
            data: zipped,
            block_info: BlockInfo::new(),
        });
        (data, encoding)
    }

    fn per_value_fsl(fsl: FixedSizeListBlock) -> (PerValueDataBlock, ArrayEncoding) {
        if !fsl.child.is_nullable() {
            Self::simple_per_value_fsl(fsl)
        } else {
            Self::nullable_per_value_fsl(fsl)
        }
    }
}

impl BlockCompressor for ValueEncoder {
    fn compress(&self, data: DataBlock) -> Result<LanceBuffer> {
        let data = match data {
            DataBlock::FixedWidth(fixed_width) => fixed_width.data,
            _ => unimplemented!(
                "Cannot compress block of type {} with ValueEncoder",
                data.name()
            ),
        };
        Ok(data)
    }
}

impl MiniBlockCompressor for ValueEncoder {
    fn compress(&self, chunk: DataBlock) -> Result<(MiniBlockCompressed, pb::ArrayEncoding)> {
        match chunk {
            DataBlock::FixedWidth(fixed_width) => {
                let encoding = ProtobufUtils::flat_encoding(fixed_width.bits_per_value, 0, None);
                Ok((Self::chunk_data(fixed_width), encoding))
            }
            DataBlock::FixedSizeList(_) => Ok(Self::miniblock_fsl(chunk)),
            _ => Err(Error::InvalidInput {
                source: format!(
                    "Cannot compress a data block of type {} with ValueEncoder",
                    chunk.name()
                )
                .into(),
                location: location!(),
            }),
        }
    }
}

#[derive(Debug)]
struct ValueFslDesc {
    dimension: u64,
    has_validity: bool,
}

/// A decompressor for fixed-width data that has
/// been written, as-is, to disk in single contiguous array
#[derive(Debug)]
pub struct ValueDecompressor {
    /// How many bits are in each inner-most item (e.g. FSL<Int32, 100> would be 32)
    bits_per_item: u64,
    /// How many bits are in each value (e.g. FSL<Int32, 100> would be 3200)
    ///
    /// This number is a little trickier to compute because we also have to include bytes
    /// of any inner validity
    bits_per_value: u64,
    /// How many items are in each value (e.g. FSL<Int32, 100> would be 100)
    items_per_value: u64,
    layers: Vec<ValueFslDesc>,
}

impl ValueDecompressor {
    pub fn from_flat(description: &pb::Flat) -> Self {
        Self {
            bits_per_item: description.bits_per_value,
            bits_per_value: description.bits_per_value,
            items_per_value: 1,
            layers: Vec::default(),
        }
    }

    pub fn from_fsl(mut description: &pb::FixedSizeList) -> Self {
        let mut layers = Vec::new();
        let mut cum_dim = 1;
        let mut bytes_per_value = 0;
        loop {
            layers.push(ValueFslDesc {
                has_validity: description.has_validity,
                dimension: description.dimension as u64,
            });
            cum_dim *= description.dimension as u64;
            if description.has_validity {
                bytes_per_value += cum_dim.div_ceil(8);
            }
            match description
                .items
                .as_ref()
                .unwrap()
                .array_encoding
                .as_ref()
                .unwrap()
            {
                pb::array_encoding::ArrayEncoding::FixedSizeList(inner) => {
                    description = inner;
                }
                pb::array_encoding::ArrayEncoding::Flat(flat) => {
                    let mut bits_per_value = bytes_per_value * 8;
                    bits_per_value += flat.bits_per_value * cum_dim;
                    return Self {
                        bits_per_item: flat.bits_per_value,
                        bits_per_value,
                        items_per_value: cum_dim,
                        layers,
                    };
                }
                _ => unreachable!(),
            }
        }
    }

    fn buffer_to_block(&self, data: LanceBuffer, num_values: u64) -> DataBlock {
        DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: self.bits_per_item,
            num_values,
            data,
            block_info: BlockInfo::new(),
        })
    }
}

impl BlockDecompressor for ValueDecompressor {
    fn decompress(&self, data: LanceBuffer, num_values: u64) -> Result<DataBlock> {
        let block = self.buffer_to_block(data, num_values);
        assert_eq!(block.num_values(), num_values);
        Ok(block)
    }
}

impl MiniBlockDecompressor for ValueDecompressor {
    fn decompress(&self, data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        let num_items = num_values * self.items_per_value;
        let mut buffer_iter = data.into_iter().rev();

        // Always at least 1 buffer
        let data_buf = buffer_iter.next().unwrap();
        let items = self.buffer_to_block(data_buf, num_items);
        let mut lists = items;

        for layer in self.layers.iter().rev() {
            if layer.has_validity {
                let validity_buf = buffer_iter.next().unwrap();
                lists = DataBlock::Nullable(NullableDataBlock {
                    data: Box::new(lists),
                    nulls: validity_buf,
                    block_info: BlockInfo::default(),
                });
            }
            lists = DataBlock::FixedSizeList(FixedSizeListBlock {
                child: Box::new(lists),
                dimension: layer.dimension,
            })
        }

        assert_eq!(lists.num_values(), num_values);
        Ok(lists)
    }
}

struct FslDecompressorValidityBuilder {
    buffer: BooleanBufferBuilder,
    bits_per_row: usize,
    bytes_per_row: usize,
}

// Helper methods for per-value decompression
impl ValueDecompressor {
    fn has_validity(&self) -> bool {
        self.layers.iter().any(|layer| layer.has_validity)
    }

    // If there is no validity then decompression is zero-copy, we just need to restore any FSL layers
    fn simple_decompress(&self, data: FixedWidthDataBlock, num_rows: u64) -> DataBlock {
        let mut cum_dim = 1;
        for layer in &self.layers {
            cum_dim *= layer.dimension;
        }
        debug_assert_eq!(self.bits_per_item, data.bits_per_value / cum_dim);
        let mut block = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: self.bits_per_item,
            num_values: num_rows * cum_dim,
            data: data.data,
            block_info: BlockInfo::new(),
        });
        for layer in self.layers.iter().rev() {
            block = DataBlock::FixedSizeList(FixedSizeListBlock {
                child: Box::new(block),
                dimension: layer.dimension,
            });
        }
        debug_assert_eq!(num_rows, block.num_values());
        block
    }

    // If there is validity then it has been zipped in with the values and we must unzip it
    fn unzip_decompress(&self, data: FixedWidthDataBlock, num_rows: usize) -> DataBlock {
        // No support for full-zip on per-value encodings
        assert_eq!(self.bits_per_item % 8, 0);
        let bytes_per_item = self.bits_per_item / 8;
        let mut buffer_builders = Vec::with_capacity(self.layers.len());
        let mut cum_dim = 1;
        let mut total_size_bytes = 0;
        // First, go through the layers, setup our builders, allocate space
        for layer in &self.layers {
            cum_dim *= layer.dimension as usize;
            if layer.has_validity {
                let validity_size_bits = cum_dim;
                let validity_size_bytes = validity_size_bits.div_ceil(8);
                total_size_bytes += num_rows * validity_size_bytes;
                buffer_builders.push(FslDecompressorValidityBuilder {
                    buffer: BooleanBufferBuilder::new(validity_size_bits * num_rows),
                    bits_per_row: cum_dim,
                    bytes_per_row: validity_size_bytes,
                })
            }
        }
        let num_items = num_rows * cum_dim;
        let data_size = num_items * bytes_per_item as usize;
        total_size_bytes += data_size;
        let mut data_buffer = Vec::with_capacity(data_size);

        assert_eq!(data.data.len(), total_size_bytes);

        let bytes_per_value = bytes_per_item as usize;
        let data_bytes_per_row = bytes_per_value * cum_dim;

        // Next, unzip
        let mut data_offset = 0;
        while data_offset < total_size_bytes {
            for builder in buffer_builders.iter_mut() {
                let start = data_offset * 8;
                let end = start + builder.bits_per_row;
                builder.buffer.append_packed_range(start..end, &data.data);
                data_offset += builder.bytes_per_row;
            }
            let end = data_offset + data_bytes_per_row;
            data_buffer.extend_from_slice(&data.data[data_offset..end]);
            data_offset += data_bytes_per_row;
        }

        // Finally, restore the structure
        let mut block = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: self.bits_per_value,
            num_values: num_items as u64,
            data: LanceBuffer::Owned(data_buffer),
            block_info: BlockInfo::new(),
        });

        let mut validity_bufs = buffer_builders
            .into_iter()
            .rev()
            .map(|mut b| LanceBuffer::Borrowed(b.buffer.finish().into_inner()));
        for layer in self.layers.iter().rev() {
            if layer.has_validity {
                let nullable = NullableDataBlock {
                    data: Box::new(block),
                    nulls: validity_bufs.next().unwrap(),
                    block_info: BlockInfo::new(),
                };
                block = DataBlock::Nullable(nullable);
            }
            block = DataBlock::FixedSizeList(FixedSizeListBlock {
                child: Box::new(block),
                dimension: layer.dimension,
            });
        }

        assert_eq!(num_rows, block.num_values() as usize);

        block
    }
}

impl FixedPerValueDecompressor for ValueDecompressor {
    fn decompress(&self, data: FixedWidthDataBlock, num_rows: u64) -> Result<DataBlock> {
        if self.has_validity() {
            Ok(self.unzip_decompress(data, num_rows as usize))
        } else {
            Ok(self.simple_decompress(data, num_rows))
        }
    }

    fn bits_per_value(&self) -> u64 {
        self.bits_per_value
    }
}

impl PerValueCompressor for ValueEncoder {
    fn compress(&self, data: DataBlock) -> Result<(PerValueDataBlock, ArrayEncoding)> {
        let (data, encoding) = match data {
            DataBlock::FixedWidth(fixed_width) => {
                let encoding = ProtobufUtils::flat_encoding(fixed_width.bits_per_value, 0, None);
                (PerValueDataBlock::Fixed(fixed_width), encoding)
            }
            DataBlock::FixedSizeList(fixed_size_list) => Self::per_value_fsl(fixed_size_list),
            _ => unimplemented!(
                "Cannot compress block of type {} with ValueEncoder",
                data.name()
            ),
        };
        Ok((data, encoding))
    }
}

// public tests module because we share the PRIMITIVE_TYPES constant with fixed_size_list
#[cfg(test)]
pub(crate) mod tests {
    use std::{collections::HashMap, sync::Arc};

    use arrow_array::{
        make_array, Array, ArrayRef, Decimal128Array, FixedSizeListArray, Int32Array,
    };
    use arrow_buffer::{BooleanBuffer, NullBuffer};
    use arrow_schema::{DataType, Field, TimeUnit};
    use lance_datagen::{array, gen, ArrayGeneratorExt, Dimension, RowCount};
    use rstest::rstest;

    use crate::{
        compression::{FixedPerValueDecompressor, MiniBlockDecompressor},
        data::DataBlock,
        encodings::{
            logical::primitive::{
                fullzip::{PerValueCompressor, PerValueDataBlock},
                miniblock::MiniBlockCompressor,
            },
            physical::value::ValueDecompressor,
        },
        format::pb,
        testing::{check_round_trip_encoding_of_data, check_round_trip_encoding_random, TestCases},
        version::LanceFileVersion,
    };

    use super::ValueEncoder;

    const PRIMITIVE_TYPES: &[DataType] = &[
        DataType::Null,
        DataType::FixedSizeBinary(2),
        DataType::Date32,
        DataType::Date64,
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
        DataType::Float16,
        DataType::Float32,
        DataType::Float64,
        DataType::Decimal128(10, 10),
        DataType::Decimal256(10, 10),
        DataType::Timestamp(TimeUnit::Nanosecond, None),
        DataType::Time32(TimeUnit::Second),
        DataType::Time64(TimeUnit::Nanosecond),
        DataType::Duration(TimeUnit::Second),
        // The Interval type is supported by the reader but the writer works with Lance schema
        // at the moment and Lance schema can't parse interval
        // DataType::Interval(IntervalUnit::DayTime),
    ];

    #[test_log::test(tokio::test)]
    async fn test_simple_value() {
        let items = Arc::new(Int32Array::from(vec![
            Some(0),
            None,
            Some(2),
            Some(3),
            Some(4),
            Some(5),
        ]));

        let test_cases = TestCases::default()
            .with_range(0..3)
            .with_range(0..2)
            .with_range(1..3)
            .with_indices(vec![0, 1, 2])
            .with_indices(vec![1])
            .with_indices(vec![2])
            .with_file_version(LanceFileVersion::V2_1);

        check_round_trip_encoding_of_data(vec![items], &test_cases, HashMap::default()).await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_value_primitive(
        #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
    ) {
        for data_type in PRIMITIVE_TYPES {
            log::info!("Testing encoding for {:?}", data_type);
            let field = Field::new("", data_type.clone(), false);
            check_round_trip_encoding_random(field, version).await;
        }
    }

    lazy_static::lazy_static! {
        static ref LARGE_TYPES: Vec<DataType> = vec![DataType::FixedSizeList(
            Arc::new(Field::new("", DataType::Int32, false)),
            128,
        )];
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_large_primitive(
        #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
    ) {
        for data_type in LARGE_TYPES.iter() {
            log::info!("Testing encoding for {:?}", data_type);
            let field = Field::new("", data_type.clone(), false);
            check_round_trip_encoding_random(field, version).await;
        }
    }

    #[test_log::test(tokio::test)]
    async fn test_decimal128_dictionary_encoding() {
        let test_cases = TestCases::default().with_file_version(LanceFileVersion::V2_1);
        let decimals: Vec<i32> = (0..100).collect();
        let repeated_strings: Vec<_> = decimals
            .iter()
            .cycle()
            .take(decimals.len() * 10000)
            .map(|&v| Some(v as i128))
            .collect();
        let decimal_array = Arc::new(Decimal128Array::from(repeated_strings)) as ArrayRef;
        check_round_trip_encoding_of_data(vec![decimal_array], &test_cases, HashMap::new()).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_miniblock_stress() {
        // Tests for strange page sizes and batch sizes and validity scenarios for miniblock

        // 10K integers, 100 per array, all valid
        let data1 = (0..100)
            .map(|_| Arc::new(Int32Array::from_iter_values(0..100)) as Arc<dyn Array>)
            .collect::<Vec<_>>();

        // Same as above but with mixed validity
        let data2 = (0..100)
            .map(|_| {
                Arc::new(Int32Array::from_iter((0..100).map(|i| {
                    if i % 2 == 0 {
                        Some(i)
                    } else {
                        None
                    }
                }))) as Arc<dyn Array>
            })
            .collect::<Vec<_>>();

        // Same as above but with all null for first half then all valid
        // TODO: Re-enable once the all-null path is complete
        let _data3 = (0..100)
            .map(|chunk_idx| {
                Arc::new(Int32Array::from_iter((0..100).map(|i| {
                    if chunk_idx < 50 {
                        None
                    } else {
                        Some(i)
                    }
                }))) as Arc<dyn Array>
            })
            .collect::<Vec<_>>();

        for data in [data1, data2 /*data3*/] {
            for batch_size in [10, 100, 1500, 15000] {
                // 40000 bytes of data
                let test_cases = TestCases::default()
                    .with_page_sizes(vec![1000, 2000, 3000, 60000])
                    .with_batch_size(batch_size)
                    .with_file_version(LanceFileVersion::V2_1);

                check_round_trip_encoding_of_data(data.clone(), &test_cases, HashMap::new()).await;
            }
        }
    }

    fn create_simple_fsl() -> FixedSizeListArray {
        // [[0, 1], NULL], [NULL, NULL], [[8, 9], [NULL, 11]]
        let items = Arc::new(Int32Array::from(vec![
            Some(0),
            Some(1),
            Some(2),
            Some(3),
            None,
            None,
            None,
            None,
            Some(8),
            Some(9),
            None,
            Some(11),
        ]));
        let items_field = Arc::new(Field::new("item", DataType::Int32, true));
        let inner_list_nulls = BooleanBuffer::from(vec![true, false, false, false, true, true]);
        let inner_list = Arc::new(FixedSizeListArray::new(
            items_field.clone(),
            2,
            items,
            Some(NullBuffer::new(inner_list_nulls)),
        ));
        let inner_list_field = Arc::new(Field::new(
            "item",
            DataType::FixedSizeList(items_field, 2),
            true,
        ));
        FixedSizeListArray::new(inner_list_field, 2, inner_list, None)
    }

    #[test]
    fn test_fsl_value_compression_miniblock() {
        let sample_list = create_simple_fsl();

        let starting_data = DataBlock::from_array(sample_list.clone());

        let encoder = ValueEncoder::default();
        let (data, compression) = MiniBlockCompressor::compress(&encoder, starting_data).unwrap();

        assert_eq!(data.num_values, 3);
        assert_eq!(data.data.len(), 3);
        assert_eq!(data.chunks.len(), 1);
        assert_eq!(data.chunks[0].buffer_sizes, vec![1, 2, 48]);
        assert_eq!(data.chunks[0].log_num_values, 0);

        let pb::array_encoding::ArrayEncoding::FixedSizeList(fsl) =
            compression.array_encoding.unwrap()
        else {
            panic!()
        };

        let decompressor = ValueDecompressor::from_fsl(fsl.as_ref());

        let decompressed =
            MiniBlockDecompressor::decompress(&decompressor, data.data, data.num_values).unwrap();

        let decompressed = make_array(
            decompressed
                .into_arrow(sample_list.data_type().clone(), true)
                .unwrap(),
        );

        assert_eq!(decompressed.as_ref(), &sample_list);
    }

    #[test]
    fn test_fsl_value_compression_per_value() {
        let sample_list = create_simple_fsl();

        let starting_data = DataBlock::from_array(sample_list.clone());

        let encoder = ValueEncoder::default();
        let (data, compression) = PerValueCompressor::compress(&encoder, starting_data).unwrap();

        let PerValueDataBlock::Fixed(data) = data else {
            panic!()
        };

        assert_eq!(data.bits_per_value, 144);
        assert_eq!(data.num_values, 3);
        assert_eq!(data.data.len(), 18 * 3);

        let pb::array_encoding::ArrayEncoding::FixedSizeList(fsl) =
            compression.array_encoding.unwrap()
        else {
            panic!()
        };

        let decompressor = ValueDecompressor::from_fsl(fsl.as_ref());

        let num_values = data.num_values;
        let decompressed =
            FixedPerValueDecompressor::decompress(&decompressor, data, num_values).unwrap();

        let decompressed = make_array(
            decompressed
                .into_arrow(sample_list.data_type().clone(), true)
                .unwrap(),
        );

        assert_eq!(decompressed.as_ref(), &sample_list);
    }

    fn create_random_fsl() -> Arc<dyn Array> {
        // Several levels of def and multiple pages
        let inner = array::rand_type(&DataType::Int32).with_random_nulls(0.1);
        let list_one = array::cycle_vec(inner, Dimension::from(4)).with_random_nulls(0.1);
        let list_two = array::cycle_vec(list_one, Dimension::from(4)).with_random_nulls(0.1);
        let list_three = array::cycle_vec(list_two, Dimension::from(2));

        // Should be 256Ki rows ~ 1MiB of data
        let batch = gen()
            .anon_col(list_three)
            .into_batch_rows(RowCount::from(8 * 1024))
            .unwrap();
        batch.column(0).clone()
    }

    #[test]
    fn fsl_value_miniblock_stress() {
        let sample_array = create_random_fsl();

        let starting_data =
            DataBlock::from_arrays(&[sample_array.clone()], sample_array.len() as u64);

        let encoder = ValueEncoder::default();
        let (data, compression) = MiniBlockCompressor::compress(&encoder, starting_data).unwrap();

        let pb::array_encoding::ArrayEncoding::FixedSizeList(fsl) =
            compression.array_encoding.unwrap()
        else {
            panic!()
        };

        let decompressor = ValueDecompressor::from_fsl(fsl.as_ref());

        let decompressed =
            MiniBlockDecompressor::decompress(&decompressor, data.data, data.num_values).unwrap();

        let decompressed = make_array(
            decompressed
                .into_arrow(sample_array.data_type().clone(), true)
                .unwrap(),
        );

        assert_eq!(decompressed.as_ref(), sample_array.as_ref());
    }

    #[test]
    fn fsl_value_per_value_stress() {
        let sample_array = create_random_fsl();

        let starting_data =
            DataBlock::from_arrays(&[sample_array.clone()], sample_array.len() as u64);

        let encoder = ValueEncoder::default();
        let (data, compression) = PerValueCompressor::compress(&encoder, starting_data).unwrap();

        let pb::array_encoding::ArrayEncoding::FixedSizeList(fsl) =
            compression.array_encoding.unwrap()
        else {
            panic!()
        };

        let decompressor = ValueDecompressor::from_fsl(fsl.as_ref());

        let PerValueDataBlock::Fixed(data) = data else {
            panic!()
        };

        let num_values = data.num_values;
        let decompressed =
            FixedPerValueDecompressor::decompress(&decompressor, data, num_values).unwrap();

        let decompressed = make_array(
            decompressed
                .into_arrow(sample_array.data_type().clone(), true)
                .unwrap(),
        );

        assert_eq!(decompressed.as_ref(), sample_array.as_ref());
    }
}
