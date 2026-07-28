// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fallible construction of concrete generic block decompressors.

#[cfg(test)]
use bytes::Bytes;

use super::*;
use crate::{
    buffer::LanceBuffer,
    compression::BlockDecompressor,
    encodings::physical::{
        block::{CompressionConfig, CompressionScheme},
        constant::ConstantBlockDecompressor,
        general::GenericGeneralBlockDecompressor,
        rle::{BlockRleDecompressor, BlockRunCount, MetadataRunLengths},
        value::FixedWidthBlockDecompressor,
    },
    format::pb21::{self, CompressiveEncoding, compressive_encoding::Compression},
};

#[cfg(feature = "bitpacking")]
use crate::encodings::physical::bitpacking::{InlineBitpacking, OutOfLineBitpacking};

fn checked_payload_bytes(
    value_type: BlockValueType,
    num_values: u64,
    label: &str,
) -> Result<usize> {
    let bytes = usize::try_from(num_values)
        .ok()
        .and_then(|num_values| num_values.checked_mul(value_type.bytes_per_value()))
        .ok_or_else(|| Error::invalid_input(format!("{label} payload length overflows usize")))?;
    if bytes > isize::MAX as usize {
        return Err(Error::invalid_input(format!(
            "{label} payload length {bytes} exceeds isize::MAX"
        )));
    }
    Ok(bytes)
}

pub fn validate_fixed_payload_len(
    payload: &LanceBuffer,
    value_type: BlockValueType,
    num_values: u64,
    label: &str,
) -> Result<()> {
    let expected = checked_payload_bytes(value_type, num_values, label)?;
    if payload.len() != expected {
        return Err(Error::invalid_input(format!(
            "{label} payload has {} bytes, expected {expected} for {num_values} {}-bit values",
            payload.len(),
            value_type.bits_per_value()
        )));
    }
    Ok(())
}

#[cfg(feature = "bitpacking")]
pub fn validate_inline_bitpacking_payload(
    payload: &LanceBuffer,
    value_type: BlockValueType,
    num_values: u64,
) -> Result<()> {
    if num_values == 0 || num_values > BITPACK_CHUNK_VALUES {
        return Err(Error::invalid_input(format!(
            "Inline bitpacking cardinality {num_values} is outside 1..={BITPACK_CHUNK_VALUES}"
        )));
    }
    let word_bytes = value_type.bytes_per_value();
    if payload.len() < word_bytes {
        return Err(Error::invalid_input(format!(
            "Inline bitpacking payload has {} bytes, shorter than its {word_bytes}-byte header",
            payload.len()
        )));
    }
    let bit_width = decode_scalar(&payload[..word_bytes], value_type, "Inline bitpacking")?;
    if bit_width > value_type.bits_per_value() {
        return Err(Error::invalid_input(format!(
            "Inline bitpacking width {bit_width} exceeds {}",
            value_type.bits_per_value()
        )));
    }
    let packed_words = (BITPACK_CHUNK_VALUES * bit_width) / value_type.bits_per_value();
    let expected_words = 1_u64
        .checked_add(packed_words)
        .ok_or_else(|| Error::invalid_input("Inline bitpacking payload word count overflows"))?;
    let expected_bytes = usize::try_from(expected_words)
        .ok()
        .and_then(|words| words.checked_mul(word_bytes))
        .ok_or_else(|| Error::invalid_input("Inline bitpacking payload length overflows"))?;
    if payload.len() != expected_bytes {
        return Err(Error::invalid_input(format!(
            "Inline bitpacking payload has {} bytes, expected {expected_bytes}",
            payload.len()
        )));
    }
    Ok(())
}

#[cfg(feature = "bitpacking")]
fn out_of_line_payload_bytes(
    value_type: BlockValueType,
    num_values: u64,
    compressed_bits_per_value: u64,
) -> Result<u64> {
    if compressed_bits_per_value >= value_type.bits_per_value() {
        return Err(Error::invalid_input(format!(
            "Invalid out-of-line bit width {compressed_bits_per_value} for {}-bit values",
            value_type.bits_per_value()
        )));
    }
    let full_chunks = num_values / BITPACK_CHUNK_VALUES;
    let tail_values = num_values % BITPACK_CHUNK_VALUES;
    let words_per_chunk =
        (BITPACK_CHUNK_VALUES * compressed_bits_per_value).div_ceil(value_type.bits_per_value());
    let mut words = full_chunks
        .checked_mul(words_per_chunk)
        .ok_or_else(|| Error::invalid_input("Out-of-line bitpacking word count overflows"))?;
    if tail_values > 0 {
        let tail_bit_savings = value_type.bits_per_value() - compressed_bits_per_value;
        let padding_cost = compressed_bits_per_value * (BITPACK_CHUNK_VALUES - tail_values);
        let tail_pack_savings = tail_bit_savings * tail_values;
        words = words
            .checked_add(if padding_cost < tail_pack_savings {
                words_per_chunk
            } else {
                tail_values
            })
            .ok_or_else(|| {
                Error::invalid_input("Out-of-line bitpacking tail word count overflows")
            })?;
    }
    words
        .checked_mul(value_type.bytes_per_value() as u64)
        .ok_or_else(|| Error::invalid_input("Out-of-line bitpacking byte length overflows"))
}

#[cfg(feature = "bitpacking")]
pub fn validate_out_of_line_payload(
    payload: &LanceBuffer,
    value_type: BlockValueType,
    num_values: u64,
    compressed_bits_per_value: u64,
) -> Result<()> {
    let expected = out_of_line_payload_bytes(value_type, num_values, compressed_bits_per_value)?;
    if payload.len() as u64 != expected {
        return Err(Error::invalid_input(format!(
            "Out-of-line bitpacking payload has {} bytes, expected {expected}",
            payload.len()
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Position {
    Root,
    Child,
}

/// Builds the concrete decoder tree while validating the bounded block grammar.
pub fn create_block_decompressor(
    encoding: &CompressiveEncoding,
    expected_type: BlockValueType,
) -> Result<(Box<dyn BlockDecompressor>, bool)> {
    create_inner(encoding, expected_type, Position::Root, true)
}

fn create_inner(
    encoding: &CompressiveEncoding,
    expected_type: BlockValueType,
    position: Position,
    allow_general: bool,
) -> Result<(Box<dyn BlockDecompressor>, bool)> {
    let compression = encoding
        .compression
        .as_ref()
        .ok_or_else(|| Error::invalid_input("Block encoding is missing its compression variant"))?;
    let expected_bits = expected_type.bits_per_value();
    match compression {
        Compression::Flat(flat) => {
            validate_flat(flat, expected_type, "Flat")?;
            Ok((
                Box::new(FixedWidthBlockDecompressor::new(expected_type)),
                true,
            ))
        }
        Compression::Constant(constant) => {
            let value = constant
                .value
                .as_deref()
                .map(|value| decode_scalar(value, expected_type, "Constant"))
                .transpose()?;
            Ok((
                Box::new(ConstantBlockDecompressor::new(expected_type, value)),
                false,
            ))
        }
        Compression::InlineBitpacking(bitpacking) => {
            validate_declared_bits(
                bitpacking.uncompressed_bits_per_value,
                expected_type,
                "Inline bitpacking",
            )?;
            if bitpacking.values.is_some() {
                return Err(Error::invalid_input(
                    "Inline bitpacking leaf buffer compression is unsupported",
                ));
            }
            #[cfg(feature = "bitpacking")]
            {
                Ok((Box::new(InlineBitpacking::new(expected_bits)), true))
            }
            #[cfg(not(feature = "bitpacking"))]
            {
                Err(Error::not_supported_source(
                    "this runtime was not built with bitpacking support".into(),
                ))
            }
        }
        Compression::OutOfLineBitpacking(bitpacking) => {
            validate_declared_bits(
                bitpacking.uncompressed_bits_per_value,
                expected_type,
                "Out-of-line bitpacking",
            )?;
            let values = bitpacking.values.as_deref().ok_or_else(|| {
                Error::invalid_input("Out-of-line bitpacking is missing its values encoding")
            })?;
            let Some(Compression::Flat(flat)) = values.compression.as_ref() else {
                return Err(Error::invalid_input(
                    "Out-of-line bitpacking values must use Flat encoding",
                ));
            };
            if flat.data.is_some() || flat.bits_per_value >= expected_bits {
                return Err(Error::invalid_input(format!(
                    "Out-of-line bitpacking width {} must be between 0 and {}",
                    flat.bits_per_value,
                    expected_bits - 1
                )));
            }
            #[cfg(feature = "bitpacking")]
            {
                Ok((
                    Box::new(OutOfLineBitpacking::new(flat.bits_per_value, expected_bits)),
                    true,
                ))
            }
            #[cfg(not(feature = "bitpacking"))]
            {
                Err(Error::not_supported_source(
                    "this runtime was not built with bitpacking support".into(),
                ))
            }
        }
        Compression::General(general) => {
            if position != Position::Root || !allow_general {
                return Err(Error::invalid_input(
                    "General compression is only supported as the single outer block transform",
                ));
            }
            let config = validate_compression_config(general.compression.as_ref(), "General")?;
            let child_encoding = general.values.as_deref().ok_or_else(|| {
                Error::invalid_input("General compression is missing its child encoding")
            })?;
            if !matches!(
                child_encoding.compression.as_ref(),
                Some(Compression::Flat(_))
            ) {
                return Err(Error::invalid_input(
                    "Outer General block compression only supports a Flat child",
                ));
            }
            let (child, child_has_payload) =
                create_inner(child_encoding, expected_type, Position::Child, false)?;
            if !child_has_payload {
                return Err(Error::invalid_input(
                    "Outer General block compression requires a payload-bearing child",
                ));
            }
            Ok((
                Box::new(GenericGeneralBlockDecompressor::new(
                    child,
                    config,
                    expected_type,
                )),
                true,
            ))
        }
        Compression::Rle(rle) => {
            if position != Position::Root {
                return Err(Error::invalid_input(
                    "RLE is not supported as a block codec child",
                ));
            }
            let values_encoding = rle
                .values
                .as_deref()
                .ok_or_else(|| Error::invalid_input("RLE is missing its values encoding"))?;
            let run_lengths_encoding = rle
                .run_lengths
                .as_deref()
                .ok_or_else(|| Error::invalid_input("RLE is missing its run lengths encoding"))?;
            let run_length_type = infer_inner(run_lengths_encoding, Position::Child)?;
            if !matches!(
                run_length_type,
                BlockValueType::UInt8 | BlockValueType::UInt16 | BlockValueType::UInt32
            ) {
                return Err(Error::invalid_input(format!(
                    "RLE run lengths must use 8, 16, or 32-bit values, got {}",
                    run_length_type.bits_per_value()
                )));
            }
            let (values, values_have_payload) =
                create_inner(values_encoding, expected_type, Position::Child, false)?;
            let (run_lengths, run_lengths_have_payload) = create_inner(
                run_lengths_encoding,
                run_length_type,
                Position::Child,
                false,
            )?;
            let metadata_run_lengths = match run_lengths_encoding.compression.as_ref() {
                Some(Compression::Constant(constant)) => {
                    let value = decode_scalar(
                        constant.value.as_deref().ok_or_else(|| {
                            Error::invalid_input("RLE run lengths Constant is missing its scalar")
                        })?,
                        run_length_type,
                        "RLE run lengths Constant",
                    )?;
                    Some(MetadataRunLengths::Constant(value))
                }
                _ => None,
            };
            let run_count = if let Some(metadata) = metadata_run_lengths {
                BlockRunCount::Metadata(metadata)
            } else if matches!(
                run_lengths_encoding.compression.as_ref(),
                Some(Compression::Flat(_))
            ) {
                BlockRunCount::RunLengthsPayload
            } else if matches!(
                values_encoding.compression.as_ref(),
                Some(Compression::Flat(_))
            ) {
                BlockRunCount::ValuesPayload
            } else {
                return Err(Error::invalid_input(
                    "RLE requires metadata run lengths or a Flat child to determine the run count",
                ));
            };
            Ok((
                Box::new(BlockRleDecompressor::new(
                    expected_type,
                    run_length_type,
                    values,
                    run_lengths,
                    values_have_payload,
                    run_lengths_have_payload,
                    run_count,
                )),
                values_have_payload || run_lengths_have_payload,
            ))
        }
        other => Err(Error::invalid_input(format!(
            "Unsupported block sequence encoding: {}",
            compression_name(other)
        ))),
    }
}

fn validate_flat(flat: &pb21::Flat, expected_type: BlockValueType, label: &str) -> Result<()> {
    validate_declared_bits(flat.bits_per_value, expected_type, label)?;
    if flat.data.is_some() {
        return Err(Error::invalid_input(format!(
            "{label} leaf buffer compression is unsupported"
        )));
    }
    Ok(())
}

fn validate_declared_bits(actual: u64, expected_type: BlockValueType, label: &str) -> Result<()> {
    if actual != expected_type.bits_per_value() {
        return Err(Error::invalid_input(format!(
            "{label} declares {actual}-bit values, expected {}",
            expected_type.bits_per_value()
        )));
    }
    Ok(())
}

fn decode_scalar(bytes: &[u8], value_type: BlockValueType, label: &str) -> Result<u64> {
    if bytes.len() != value_type.bytes_per_value() {
        return Err(Error::invalid_input(format!(
            "{label} scalar has {} bytes, expected {}",
            bytes.len(),
            value_type.bytes_per_value()
        )));
    }
    Ok(match value_type {
        BlockValueType::UInt8 => u64::from(bytes[0]),
        BlockValueType::UInt16 => u64::from(u16::from_le_bytes([bytes[0], bytes[1]])),
        BlockValueType::UInt32 => u64::from(u32::from_le_bytes(
            bytes.try_into().expect("scalar length was checked"),
        )),
        BlockValueType::UInt64 => {
            u64::from_le_bytes(bytes.try_into().expect("scalar length was checked"))
        }
    })
}

#[cfg(test)]
pub fn encode_scalar(value: u64, value_type: BlockValueType) -> Result<Bytes> {
    if value > value_type.max_value() {
        return Err(Error::invalid_input(format!(
            "Scalar value {value} exceeds the {}-bit value range",
            value_type.bits_per_value()
        )));
    }
    Ok(match value_type {
        BlockValueType::UInt8 => vec![value as u8],
        BlockValueType::UInt16 => (value as u16).to_le_bytes().to_vec(),
        BlockValueType::UInt32 => (value as u32).to_le_bytes().to_vec(),
        BlockValueType::UInt64 => value.to_le_bytes().to_vec(),
    }
    .into())
}

fn validate_compression_config(
    compression: Option<&pb21::BufferCompression>,
    label: &str,
) -> Result<CompressionConfig> {
    let compression = compression
        .ok_or_else(|| Error::invalid_input(format!("{label} is missing compression config")))?;
    let scheme = pb21::CompressionScheme::try_from(compression.scheme).map_err(|_| {
        Error::invalid_input(format!(
            "{label} has unknown compression scheme {}",
            compression.scheme
        ))
    })?;
    let scheme = CompressionScheme::try_from(scheme)?;
    Ok(CompressionConfig::new(scheme, compression.level))
}

fn infer_inner(encoding: &CompressiveEncoding, position: Position) -> Result<BlockValueType> {
    let compression = encoding
        .compression
        .as_ref()
        .ok_or_else(|| Error::invalid_input("Block encoding is missing its compression variant"))?;
    match compression {
        Compression::Flat(flat) => BlockValueType::from_bits(flat.bits_per_value),
        Compression::Constant(constant) => {
            let value = constant.value.as_ref().ok_or_else(|| {
                Error::invalid_input(
                    "Cannot infer an empty Constant block type without its typed container role",
                )
            })?;
            BlockValueType::from_bits((value.len() * 8) as u64)
        }
        Compression::InlineBitpacking(bitpacking) => {
            BlockValueType::from_bits(bitpacking.uncompressed_bits_per_value)
        }
        Compression::OutOfLineBitpacking(bitpacking) => {
            BlockValueType::from_bits(bitpacking.uncompressed_bits_per_value)
        }
        Compression::General(general) if position == Position::Root => infer_inner(
            general
                .values
                .as_deref()
                .ok_or_else(|| Error::invalid_input("General is missing its child encoding"))?,
            Position::Child,
        ),
        Compression::Rle(rle) if position == Position::Root => infer_inner(
            rle.values
                .as_deref()
                .ok_or_else(|| Error::invalid_input("RLE is missing its values encoding"))?,
            Position::Child,
        ),
        other => Err(Error::invalid_input(format!(
            "Cannot infer bounded block value type from {} at {position:?}",
            compression_name(other)
        ))),
    }
}

fn compression_name(compression: &Compression) -> &'static str {
    match compression {
        Compression::Flat(_) => "flat",
        Compression::Variable(_) => "variable",
        Compression::Constant(_) => "constant",
        Compression::OutOfLineBitpacking(_) => "out-of-line bitpacking",
        Compression::InlineBitpacking(_) => "inline bitpacking",
        Compression::Fsst(_) => "fsst",
        Compression::Dictionary(_) => "dictionary",
        Compression::Rle(_) => "rle",
        Compression::ByteStreamSplit(_) => "byte stream split",
        Compression::General(_) => "general",
        Compression::FixedSizeList(_) => "fixed-size list",
        Compression::PackedStruct(_) => "packed struct",
        Compression::VariablePackedStruct(_) => "variable packed struct",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_nested_transform_trees() {
        let nested = crate::format::ProtobufUtils21::rle(
            crate::format::ProtobufUtils21::rle(
                crate::format::ProtobufUtils21::flat(64, None),
                crate::format::ProtobufUtils21::constant(Some(vec![1].into())),
            ),
            crate::format::ProtobufUtils21::constant(Some(vec![1].into())),
        );
        assert!(
            create_block_decompressor(&nested, BlockValueType::UInt64)
                .unwrap_err()
                .to_string()
                .contains("child")
        );
    }

    #[test]
    fn rejects_mistyped_flat_leaf() {
        let flat = crate::format::ProtobufUtils21::flat(32, None);
        let error = create_block_decompressor(&flat, BlockValueType::UInt64).unwrap_err();
        assert!(error.to_string().contains("expected 64"));
    }
}
