// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

use std::sync::Arc;

use crate::{
    buffer::LanceBuffer,
    compression::BlockCompressor,
    data::{BlockInfo, DataBlock, FixedWidthDataBlock},
    encodings::physical::{
        constant::ConstantBlockCompressor, dictionary::DictionaryBlockCompressor,
        range::RangeEncoder, rle::BlockRleCompressor, value::FixedWidthBlockCompressor,
    },
    format::{ProtobufUtils21, pb21::CompressiveEncoding},
};

fn fixed_u64(values: &[u64]) -> FixedWidthDataBlock {
    FixedWidthDataBlock {
        data: LanceBuffer::reinterpret_vec(values.to_vec()),
        bits_per_value: 64,
        num_values: values.len() as u64,
        block_info: BlockInfo::default(),
    }
}

fn decoded_u64(block: DataBlock) -> Vec<u64> {
    let DataBlock::FixedWidth(block) = block else {
        panic!("expected fixed-width output");
    };
    block.data.borrow_to_typed_slice::<u64>().to_vec()
}

fn decoded_u32(block: DataBlock) -> Vec<u32> {
    let DataBlock::FixedWidth(block) = block else {
        panic!("expected fixed-width output");
    };
    block.data.borrow_to_typed_slice::<u32>().to_vec()
}

fn round_trip_u64(
    values: &[u64],
    compressor: Box<dyn BlockCompressor>,
    encoding: CompressiveEncoding,
) {
    let payload = compressor
        .compress(DataBlock::FixedWidth(fixed_u64(values)))
        .unwrap();
    let (decoder, has_payload) =
        create_block_decompressor(&encoding, BlockValueType::UInt64).unwrap();
    assert_eq!(payload.is_some(), has_payload);
    assert_eq!(
        decoded_u64(decoder.decompress(payload, values.len() as u64).unwrap()),
        values
    );
}

#[test]
fn metadata_compressors_validate_and_decode() {
    round_trip_u64(
        &[],
        Box::new(ConstantBlockCompressor::new(BlockValueType::UInt64, None)),
        ProtobufUtils21::constant(None),
    );
    round_trip_u64(
        &[7_u64; 32],
        Box::new(ConstantBlockCompressor::new(
            BlockValueType::UInt64,
            Some(7),
        )),
        ProtobufUtils21::constant(Some(7_u64.to_le_bytes().to_vec().into())),
    );
    let values = (0..32_u64).map(|value| 5 + value * 3).collect::<Vec<_>>();
    round_trip_u64(
        &values,
        Box::new(RangeEncoder::new(64, 5, 3)),
        ProtobufUtils21::range(64, 5, 3),
    );
}

#[test]
fn payload_presence_distinguishes_metadata_from_empty_payload() {
    let flat = Box::new(FixedWidthBlockCompressor::new(BlockValueType::UInt64));
    let payload = flat
        .compress(DataBlock::FixedWidth(fixed_u64(&[])))
        .unwrap();
    assert!(payload.as_ref().is_some_and(|payload| payload.is_empty()));

    let (flat_decoder, flat_has_payload) =
        create_block_decompressor(&ProtobufUtils21::flat(64, None), BlockValueType::UInt64)
            .unwrap();
    assert!(flat_has_payload);
    assert!(flat_decoder.decompress(None, 0).is_err());
    assert!(
        decoded_u64(
            flat_decoder
                .decompress(Some(LanceBuffer::empty()), 0)
                .unwrap()
        )
        .is_empty()
    );

    let (constant_decoder, constant_has_payload) = create_block_decompressor(
        &ProtobufUtils21::constant(Some(7_u64.to_le_bytes().to_vec().into())),
        BlockValueType::UInt64,
    )
    .unwrap();
    assert!(!constant_has_payload);
    assert!(
        constant_decoder
            .decompress(Some(LanceBuffer::empty()), 1)
            .is_err()
    );
}

#[test]
fn rle_compressor_owns_and_reuses_children() {
    let mut values = Vec::new();
    for value in [11_u64, 91, 37, 123] {
        values.extend(std::iter::repeat_n(value, 256));
    }
    let compressor = Box::new(BlockRleCompressor::new(
        BlockValueType::UInt64,
        BlockValueType::UInt16,
        Box::new(FixedWidthBlockCompressor::new(BlockValueType::UInt64)),
        Box::new(ConstantBlockCompressor::new(
            BlockValueType::UInt16,
            Some(256),
        )),
    ));
    let encoding = ProtobufUtils21::rle(
        ProtobufUtils21::flat(64, None),
        ProtobufUtils21::constant(Some(256_u16.to_le_bytes().to_vec().into())),
    );
    round_trip_u64(&values, compressor, encoding);
}

#[test]
fn dictionary_materializes_once_when_built() {
    use crate::encodings::physical::dictionary::BLOCK_MATERIALIZATION_COUNT;

    let values = (0..4096_u64)
        .map(|index| [u64::MAX - 9, 17, 1_u64 << 50, 991][index as usize % 4])
        .collect::<Vec<_>>();
    let dictionary_items = Arc::from([17_u64, 991, 1_u64 << 50, u64::MAX - 9].as_slice());
    let compressor = Box::new(DictionaryBlockCompressor::new(
        BlockValueType::UInt64,
        dictionary_items,
        Box::new(FixedWidthBlockCompressor::new(BlockValueType::UInt32)),
        Box::new(FixedWidthBlockCompressor::new(BlockValueType::UInt64)),
    ));
    let encoding = ProtobufUtils21::dictionary(
        ProtobufUtils21::flat(32, None),
        ProtobufUtils21::flat(64, None),
        4,
    );
    BLOCK_MATERIALIZATION_COUNT.with(|count| count.set(0));
    BLOCK_MATERIALIZATION_COUNT.with(|count| assert_eq!(count.get(), 0));
    round_trip_u64(&values, compressor, encoding);
    BLOCK_MATERIALIZATION_COUNT.with(|count| assert_eq!(count.get(), 1));
}

#[cfg(feature = "bitpacking")]
#[test]
fn out_of_line_bitpacking_round_trip() {
    use crate::encodings::physical::bitpacking::OutOfLineBitpacking;

    let values = (0..4096_u64)
        .map(|index| (index * 37) % 1024)
        .collect::<Vec<_>>();
    round_trip_u64(
        &values,
        Box::new(OutOfLineBitpacking::new(10, 64)),
        ProtobufUtils21::out_of_line_bitpacking(64, ProtobufUtils21::flat(10, None)),
    );
}

#[cfg(any(feature = "lz4", feature = "zstd"))]
#[test]
fn general_compressor_owns_a_flat_child() {
    use crate::encodings::physical::{
        block::{CompressionConfig, CompressionScheme},
        general::GeneralBlockCompressor,
    };

    let scheme = if cfg!(feature = "lz4") {
        CompressionScheme::Lz4
    } else {
        CompressionScheme::Zstd
    };
    let config = CompressionConfig::new(scheme, None);
    let compressor = Box::new(GeneralBlockCompressor::new(
        Box::new(FixedWidthBlockCompressor::new(BlockValueType::UInt64)),
        config,
    ));
    let encoding = ProtobufUtils21::wrapped(config, ProtobufUtils21::flat(64, None)).unwrap();
    let values = (0..16_384_u64).map(|value| value % 17).collect::<Vec<_>>();
    round_trip_u64(&values, compressor, encoding);
}

#[test]
fn factory_rejects_unbounded_or_mistyped_trees() {
    let nested_delta = ProtobufUtils21::delta(
        64,
        0,
        ProtobufUtils21::delta(64, 0, ProtobufUtils21::flat(64, None)),
    );
    assert!(
        create_block_decompressor(&nested_delta, BlockValueType::UInt64)
            .unwrap_err()
            .to_string()
            .contains("child")
    );

    let wrong_width = ProtobufUtils21::range(32, 0, 1);
    assert!(
        create_block_decompressor(&wrong_width, BlockValueType::UInt64)
            .unwrap_err()
            .to_string()
            .contains("expected 64")
    );

    let oversized = ProtobufUtils21::dictionary(
        ProtobufUtils21::flat(32, None),
        ProtobufUtils21::flat(64, None),
        (MAX_DICTIONARY_ITEMS + 1) as u32,
    );
    assert!(
        create_block_decompressor(&oversized, BlockValueType::UInt64)
            .unwrap_err()
            .to_string()
            .contains("outside")
    );
}

#[test]
fn constant_cardinality_contract_is_checked_at_decode() {
    let (empty, has_payload) =
        create_block_decompressor(&ProtobufUtils21::constant(None), BlockValueType::UInt64)
            .unwrap();
    assert!(!has_payload);
    assert!(decoded_u64(empty.decompress(None, 0).unwrap()).is_empty());
    assert!(empty.decompress(None, 1).is_err());

    let (present, has_payload) = create_block_decompressor(
        &ProtobufUtils21::constant(Some(7_u64.to_le_bytes().to_vec().into())),
        BlockValueType::UInt64,
    )
    .unwrap();
    assert!(!has_payload);
    assert!(present.decompress(None, 0).is_err());
    assert_eq!(
        decoded_u64(present.decompress(None, 3).unwrap()),
        vec![7, 7, 7]
    );
}

#[test]
fn range_checks_cardinality_and_overflow() {
    let (decoder, has_payload) =
        create_block_decompressor(&ProtobufUtils21::range(32, 3, 5), BlockValueType::UInt32)
            .unwrap();
    assert!(!has_payload);
    assert_eq!(
        decoded_u32(decoder.decompress(None, 4).unwrap()),
        vec![3, 8, 13, 18]
    );
    assert!(decoder.decompress(None, 1).is_err());

    let (overflowing, has_payload) = create_block_decompressor(
        &ProtobufUtils21::range(32, u32::MAX as u64, 1),
        BlockValueType::UInt32,
    )
    .unwrap();
    assert!(!has_payload);
    assert!(overflowing.decompress(None, 2).is_err());
}

#[test]
fn delta_checks_prefix_sum_overflow() {
    let encoding = ProtobufUtils21::delta(
        64,
        u64::MAX,
        ProtobufUtils21::constant(Some(1_u64.to_le_bytes().to_vec().into())),
    );
    let (decoder, has_payload) =
        create_block_decompressor(&encoding, BlockValueType::UInt64).unwrap();
    assert!(!has_payload);
    assert!(decoder.decompress(None, 2).is_err());
}

#[test]
fn metadata_only_rle_round_trip() {
    let encoding = ProtobufUtils21::rle(
        ProtobufUtils21::range(64, 10, 1),
        ProtobufUtils21::range(32, 1, 1),
    );
    let (decoder, has_payload) =
        create_block_decompressor(&encoding, BlockValueType::UInt64).unwrap();
    assert!(!has_payload);
    assert_eq!(
        decoded_u64(decoder.decompress(None, 6).unwrap()),
        vec![10, 11, 11, 12, 12, 12]
    );
    assert!(decoder.decompress(None, 7).is_err());
}

#[test]
fn rle_framing_is_fallible() {
    let encoding = ProtobufUtils21::rle(
        ProtobufUtils21::flat(64, None),
        ProtobufUtils21::constant(Some(vec![2_u8].into())),
    );
    let (decoder, has_payload) =
        create_block_decompressor(&encoding, BlockValueType::UInt64).unwrap();
    assert!(has_payload);
    assert!(
        decoder
            .decompress(Some(LanceBuffer::from(vec![0; 7])), 4)
            .is_err()
    );

    let mut payload = 100_u64.to_le_bytes().to_vec();
    payload.extend_from_slice(&[0; 8]);
    assert!(
        decoder
            .decompress(Some(LanceBuffer::from(payload)), 4)
            .is_err()
    );

    let mut payload = 16_u64.to_le_bytes().to_vec();
    payload.extend_from_slice(&[0; 16]);
    payload.push(1);
    let error = decoder
        .decompress(Some(LanceBuffer::from(payload)), 4)
        .unwrap_err();
    assert!(error.to_string().contains("Metadata-only RLE run-length"));
}

#[test]
fn metadata_dictionary_checks_index_bounds() {
    let encoding = ProtobufUtils21::dictionary(
        ProtobufUtils21::range(32, 0, 1),
        ProtobufUtils21::range(64, 10, 1),
        2,
    );
    let (decoder, has_payload) =
        create_block_decompressor(&encoding, BlockValueType::UInt64).unwrap();
    assert!(!has_payload);
    let error = decoder.decompress(None, 3).unwrap_err();
    assert!(error.to_string().contains("out of bounds"));
}

#[test]
fn dictionary_framing_rejects_inconsistent_lengths() {
    let encoding = ProtobufUtils21::dictionary(
        ProtobufUtils21::flat(32, None),
        ProtobufUtils21::flat(64, None),
        2,
    );
    let (decoder, has_payload) =
        create_block_decompressor(&encoding, BlockValueType::UInt64).unwrap();
    assert!(has_payload);
    let mut payload = 4_u64.to_le_bytes().to_vec();
    payload.extend_from_slice(&16_u64.to_le_bytes());
    payload.extend_from_slice(&[0; 8]);
    assert!(
        decoder
            .decompress(Some(LanceBuffer::from(payload)), 1)
            .is_err()
    );
}

#[test]
fn dictionary_framing_rejects_payload_for_metadata_child() {
    let encoding = ProtobufUtils21::dictionary(
        ProtobufUtils21::flat(32, None),
        ProtobufUtils21::range(64, 10, 1),
        2,
    );
    let (decoder, has_payload) =
        create_block_decompressor(&encoding, BlockValueType::UInt64).unwrap();
    assert!(has_payload);
    let mut payload = 4_u64.to_le_bytes().to_vec();
    payload.extend_from_slice(&1_u64.to_le_bytes());
    payload.extend_from_slice(&0_u32.to_le_bytes());
    payload.push(0);
    let error = decoder
        .decompress(Some(LanceBuffer::from(payload)), 1)
        .unwrap_err();
    assert!(error.to_string().contains("Metadata-only Dictionary items"));
}

#[test]
fn decoder_allocation_overflow_is_fallible() {
    let (decoder, has_payload) = create_block_decompressor(
        &ProtobufUtils21::constant(Some(1_u64.to_le_bytes().to_vec().into())),
        BlockValueType::UInt64,
    )
    .unwrap();
    assert!(!has_payload);
    assert!(decoder.decompress(None, u64::MAX).is_err());
}

#[cfg(feature = "bitpacking")]
#[test]
fn frozen_out_of_line_compressor_rejects_wider_reuse() {
    use crate::encodings::physical::bitpacking::OutOfLineBitpacking;

    let compressor = OutOfLineBitpacking::new(3, 64);
    let error = compressor
        .compress(DataBlock::FixedWidth(fixed_u64(&[1, 2, 8])))
        .unwrap_err();
    assert!(error.to_string().contains("requires 4 bits"));
}
