// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dictionary codec for bounded unsigned block sequences.

#[cfg(test)]
use std::cell::Cell;
use std::{collections::BTreeMap, sync::Arc};

use crate::{
    buffer::LanceBuffer,
    compression::{
        BlockCompressor, BlockDecompressor, BlockValueType,
        block::{
            fixed_block, fixed_from_u64_values, read_unsigned_values, validate_fixed_payload_len,
            visit_unsigned_values,
        },
    },
    data::{BlockInfo, DataBlock, FixedWidthDataBlock},
    encodings::physical::try_vec_with_capacity,
};
use lance_core::{Error, Result};

pub(crate) const BLOCK_FRAME_BYTES: u64 = 16;

#[cfg(test)]
thread_local! {
    pub(crate) static BLOCK_MATERIALIZATION_COUNT: Cell<usize> = const { Cell::new(0) };
}

/// Dictionary compressor that owns its indices and items compressors.
#[derive(Debug)]
pub(crate) struct DictionaryBlockCompressor {
    value_type: BlockValueType,
    dictionary_items: Arc<[u64]>,
    indices: Box<dyn BlockCompressor>,
    items: Box<dyn BlockCompressor>,
}

impl DictionaryBlockCompressor {
    pub(crate) fn new(
        value_type: BlockValueType,
        dictionary_items: Arc<[u64]>,
        indices: Box<dyn BlockCompressor>,
        items: Box<dyn BlockCompressor>,
    ) -> Self {
        Self {
            value_type,
            dictionary_items,
            indices,
            items,
        }
    }
}

impl BlockCompressor for DictionaryBlockCompressor {
    fn compress(&self, data: DataBlock) -> Result<Option<LanceBuffer>> {
        #[cfg(test)]
        BLOCK_MATERIALIZATION_COUNT.with(|count| count.set(count.get().saturating_add(1)));

        let DataBlock::FixedWidth(data) = data else {
            return Err(Error::invalid_input(
                "Dictionary block compression requires fixed-width data",
            ));
        };
        if data.bits_per_value != self.value_type.bits_per_value() {
            return Err(Error::invalid_input(format!(
                "Dictionary compressor expects {}-bit values, got {}",
                self.value_type.bits_per_value(),
                data.bits_per_value
            )));
        }
        let dictionary_index = self
            .dictionary_items
            .iter()
            .enumerate()
            .map(|(index, value)| (*value, index as u32))
            .collect::<BTreeMap<_, _>>();
        let mut encoded_indices =
            try_vec_with_capacity::<u32>(data.num_values, "Dictionary indices")?;
        let mut position = 0_u64;
        visit_unsigned_values(&data, self.value_type, |value| {
            encoded_indices.push(*dictionary_index.get(&value).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Dictionary compressor does not contain input value {value} at position {position}"
                ))
            })?);
            position += 1;
            Ok(())
        })?;

        let indices_block = FixedWidthDataBlock {
            bits_per_value: 32,
            data: LanceBuffer::reinterpret_vec(encoded_indices),
            num_values: data.num_values,
            block_info: BlockInfo::default(),
        };
        let items_block =
            fixed_from_u64_values(&self.dictionary_items, self.value_type, "Dictionary items")?;
        let indices_payload = self
            .indices
            .compress(DataBlock::FixedWidth(indices_block))?;
        let items_payload = self.items.compress(DataBlock::FixedWidth(items_block))?;
        if indices_payload.is_none() && items_payload.is_none() {
            return Ok(None);
        }
        let indices_payload = indices_payload.unwrap_or_else(LanceBuffer::empty);
        let items_payload = items_payload.unwrap_or_else(LanceBuffer::empty);

        let mut output = try_frame(indices_payload.len(), items_payload.len())?;
        output.extend_from_slice(&(indices_payload.len() as u64).to_le_bytes());
        output.extend_from_slice(&(items_payload.len() as u64).to_le_bytes());
        output.extend_from_slice(&indices_payload);
        output.extend_from_slice(&items_payload);
        Ok(Some(LanceBuffer::from(output)))
    }
}

/// Dictionary decompressor that owns its indices and items decompressors.
#[derive(Debug)]
pub(crate) struct DictionaryBlockDecompressor {
    value_type: BlockValueType,
    num_dictionary_items: u32,
    indices: Box<dyn BlockDecompressor>,
    items: Box<dyn BlockDecompressor>,
    indices_have_payload: bool,
    items_have_payload: bool,
}

impl DictionaryBlockDecompressor {
    pub(crate) fn new(
        value_type: BlockValueType,
        num_dictionary_items: u32,
        indices: Box<dyn BlockDecompressor>,
        items: Box<dyn BlockDecompressor>,
        indices_have_payload: bool,
        items_have_payload: bool,
    ) -> Self {
        Self {
            value_type,
            num_dictionary_items,
            indices,
            items,
            indices_have_payload,
            items_have_payload,
        }
    }
}

impl BlockDecompressor for DictionaryBlockDecompressor {
    fn decompress(&self, data: Option<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        let has_payload = self.indices_have_payload || self.items_have_payload;
        let (indices_payload, items_payload) = if let Some(data) = data {
            if !has_payload {
                return Err(Error::invalid_input(
                    "Metadata-only Dictionary expects no payload",
                ));
            }
            if data.len() < BLOCK_FRAME_BYTES as usize {
                return Err(Error::invalid_input(format!(
                    "Dictionary payload has {} bytes, shorter than its {BLOCK_FRAME_BYTES}-byte header",
                    data.len()
                )));
            }
            let indices_size =
                u64::from_le_bytes(data[..8].try_into().expect("header length was checked"));
            let items_size =
                u64::from_le_bytes(data[8..16].try_into().expect("header length was checked"));
            let indices_size = usize::try_from(indices_size).map_err(|_| {
                Error::invalid_input("Dictionary indices payload length does not fit usize")
            })?;
            let items_size = usize::try_from(items_size).map_err(|_| {
                Error::invalid_input("Dictionary items payload length does not fit usize")
            })?;
            let indices_start = BLOCK_FRAME_BYTES as usize;
            let items_start = indices_start
                .checked_add(indices_size)
                .ok_or_else(|| Error::invalid_input("Dictionary indices payload end overflows"))?;
            let end = items_start
                .checked_add(items_size)
                .ok_or_else(|| Error::invalid_input("Dictionary items payload end overflows"))?;
            if end != data.len() {
                return Err(Error::invalid_input(format!(
                    "Dictionary framing describes {end} bytes, payload has {}",
                    data.len()
                )));
            }
            if !self.indices_have_payload && indices_size != 0 {
                return Err(Error::invalid_input(format!(
                    "Metadata-only Dictionary indices child has {indices_size} framed payload bytes"
                )));
            }
            if !self.items_have_payload && items_size != 0 {
                return Err(Error::invalid_input(format!(
                    "Metadata-only Dictionary items child has {items_size} framed payload bytes"
                )));
            }
            (
                self.indices_have_payload
                    .then(|| data.slice_with_length(indices_start, indices_size)),
                self.items_have_payload
                    .then(|| data.slice_with_length(items_start, items_size)),
            )
        } else {
            if has_payload {
                return Err(Error::invalid_input("Dictionary requires one payload"));
            }
            (None, None)
        };

        let indices = self.indices.decompress(indices_payload, num_values)?;
        let items = self
            .items
            .decompress(items_payload, u64::from(self.num_dictionary_items))?;
        let DataBlock::FixedWidth(indices) = indices else {
            return Err(Error::invalid_input(
                "Dictionary indices decoded to a non fixed-width block",
            ));
        };
        let DataBlock::FixedWidth(items) = items else {
            return Err(Error::invalid_input(
                "Dictionary items decoded to a non fixed-width block",
            ));
        };
        validate_fixed_payload_len(
            &indices.data,
            BlockValueType::UInt32,
            num_values,
            "Dictionary indices",
        )?;
        validate_fixed_payload_len(
            &items.data,
            self.value_type,
            u64::from(self.num_dictionary_items),
            "Dictionary items",
        )?;
        let indices = indices.data.borrow_to_typed_slice::<u32>();
        let items = read_unsigned_values(&items, self.value_type)?;
        let output = match self.value_type {
            BlockValueType::UInt8 => {
                let mut output = try_vec_with_capacity::<u8>(num_values, "Dictionary output")?;
                append_items(
                    &mut output,
                    &indices,
                    &items,
                    self.num_dictionary_items,
                    |value| value as u8,
                )?;
                LanceBuffer::reinterpret_vec(output)
            }
            BlockValueType::UInt16 => {
                let mut output = try_vec_with_capacity::<u16>(num_values, "Dictionary output")?;
                append_items(
                    &mut output,
                    &indices,
                    &items,
                    self.num_dictionary_items,
                    |value| value as u16,
                )?;
                LanceBuffer::reinterpret_vec(output)
            }
            BlockValueType::UInt32 => {
                let mut output = try_vec_with_capacity::<u32>(num_values, "Dictionary output")?;
                append_items(
                    &mut output,
                    &indices,
                    &items,
                    self.num_dictionary_items,
                    |value| value as u32,
                )?;
                LanceBuffer::reinterpret_vec(output)
            }
            BlockValueType::UInt64 => {
                let mut output = try_vec_with_capacity::<u64>(num_values, "Dictionary output")?;
                append_items(
                    &mut output,
                    &indices,
                    &items,
                    self.num_dictionary_items,
                    |value| value,
                )?;
                LanceBuffer::reinterpret_vec(output)
            }
        };
        Ok(fixed_block(self.value_type, num_values, output))
    }
}

fn append_items<T>(
    output: &mut Vec<T>,
    indices: &[u32],
    items: &[u64],
    num_dictionary_items: u32,
    convert: impl Fn(u64) -> T,
) -> Result<()> {
    for (position, index) in indices.iter().enumerate() {
        let value = *items.get(*index as usize).ok_or_else(|| {
            Error::invalid_input(format!(
                "Dictionary index {index} at position {position} is out of bounds for {num_dictionary_items} items"
            ))
        })?;
        output.push(convert(value));
    }
    Ok(())
}

fn try_frame(indices_payload_bytes: usize, items_payload_bytes: usize) -> Result<Vec<u8>> {
    let capacity = (BLOCK_FRAME_BYTES as usize)
        .checked_add(indices_payload_bytes)
        .and_then(|capacity| capacity.checked_add(items_payload_bytes))
        .ok_or_else(|| Error::invalid_input("Dictionary frame length overflows usize"))?;
    let mut output = Vec::new();
    output.try_reserve_exact(capacity).map_err(|error| {
        Error::invalid_input(format!(
            "Dictionary could not reserve {capacity} frame bytes: {error}"
        ))
    })?;
    Ok(output)
}
