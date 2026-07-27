// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Private bounded selector for concrete unsigned block compressors.

use super::{SelectedBlockCompressor, statistics::*};
use crate::{
    compression::{BlockValueType, block::encode_scalar},
    compression_config::CompressionFieldParams,
    encodings::physical::{
        block::{CompressedBufferEncoder, CompressionConfig, CompressionScheme},
        constant::ConstantBlockCompressor,
        general::{self, GeneralBlockCompressor},
        range::RangeEncoder,
        value::FixedWidthBlockCompressor,
    },
    format::ProtobufUtils21,
};
use lance_core::{Error, Result};

#[cfg(feature = "bitpacking")]
use crate::compression::{
    BlockCompressor,
    block::{BITPACK_CHUNK_VALUES, out_of_line_payload_bytes},
};
#[cfg(feature = "bitpacking")]
use crate::encodings::physical::bitpacking::{InlineBitpacking, OutOfLineBitpacking};

/// How a direct leaf candidate's payload is estimated for another sequence.
///
/// This cost metadata exists only while comparing concrete codecs.
#[derive(Debug, Clone, Copy)]
pub(super) enum PayloadEstimator {
    Flat,
    #[cfg(feature = "bitpacking")]
    InlineBitpacking,
    #[cfg(feature = "bitpacking")]
    OutOfLineBitpacking {
        compressed_bits: u64,
    },
    General {
        config: CompressionConfig,
    },
}

#[derive(Debug)]
pub(super) struct Candidate {
    pub(super) selection: SelectedBlockCompressor,
    pub(super) estimator: Option<PayloadEstimator>,
    pub(super) transform_depth: u8,
    pub(super) decode_cpu_rank: u8,
    pub(super) stable_rank: u8,
}

fn estimate_general_with_config(stats: &SequenceStats, config: CompressionConfig) -> Result<u64> {
    let sample = stats.sample_bytes();
    if sample.is_empty() && stats.raw_bytes() != 0 {
        return Err(Error::invalid_input(
            "Cannot estimate General block compression without a sample",
        ));
    }
    let compressed = general::compress_block(config, sample)?;
    Ok(
        if sample.is_empty() || sample.len() as u64 == stats.raw_bytes() {
            compressed.len() as u64
        } else {
            (compressed.len() as u64)
                .saturating_mul(stats.raw_bytes())
                .div_ceil(sample.len() as u64)
        },
    )
}

pub(super) fn direct_candidates(
    stats: &SequenceStats,
    field_params: &CompressionFieldParams,
    stable_rank_base: u8,
    allow_general: bool,
) -> Result<Vec<Candidate>> {
    let mut candidates = vec![Candidate {
        selection: flat_selection(stats.value_type),
        estimator: Some(PayloadEstimator::Flat),
        transform_depth: 0,
        decode_cpu_rank: 0,
        stable_rank: stable_rank_base,
    }];

    #[cfg(feature = "bitpacking")]
    if stats.len > 0 && stats.required_bits() < stats.value_type.bits_per_value() {
        let compressed_bits = stats.required_bits();
        let (compressor, encoding, estimator) = if stats.len <= BITPACK_CHUNK_VALUES {
            (
                Box::new(InlineBitpacking::new(stats.value_type.bits_per_value()))
                    as Box<dyn BlockCompressor>,
                ProtobufUtils21::inline_bitpacking(stats.value_type.bits_per_value(), None),
                PayloadEstimator::InlineBitpacking,
            )
        } else {
            (
                Box::new(OutOfLineBitpacking::new(
                    compressed_bits,
                    stats.value_type.bits_per_value(),
                )) as Box<dyn BlockCompressor>,
                ProtobufUtils21::out_of_line_bitpacking(
                    stats.value_type.bits_per_value(),
                    ProtobufUtils21::flat(compressed_bits, None),
                ),
                PayloadEstimator::OutOfLineBitpacking { compressed_bits },
            )
        };
        candidates.push(Candidate {
            selection: SelectedBlockCompressor::new(compressor, encoding, true),
            estimator: Some(estimator),
            transform_depth: 1,
            decode_cpu_rank: 1,
            stable_rank: stable_rank_base.saturating_add(1),
        });
    }

    if allow_general
        && let Some((config, estimated_payload)) = estimate_general_payload(stats, field_params)?
        && estimated_payload < stats.raw_bytes()
    {
        let child_encoding = ProtobufUtils21::flat(stats.value_type.bits_per_value(), None);
        let encoding = ProtobufUtils21::wrapped(config, child_encoding)?;
        let child = Box::new(FixedWidthBlockCompressor::new(stats.value_type));
        let compressor = Box::new(GeneralBlockCompressor::new(child, config));
        candidates.push(Candidate {
            selection: SelectedBlockCompressor::new(compressor, encoding, true),
            estimator: Some(PayloadEstimator::General { config }),
            transform_depth: 1,
            decode_cpu_rank: 4,
            stable_rank: stable_rank_base.saturating_add(2),
        });
    }
    Ok(candidates)
}

pub(super) fn flat_selection(value_type: BlockValueType) -> SelectedBlockCompressor {
    SelectedBlockCompressor::new(
        Box::new(FixedWidthBlockCompressor::new(value_type)),
        ProtobufUtils21::flat(value_type.bits_per_value(), None),
        true,
    )
}

pub(super) fn constant_selection(
    value_type: BlockValueType,
    value: Option<u64>,
) -> SelectedBlockCompressor {
    let encoded_value = value.map(|value| {
        encode_scalar(value, value_type)
            .expect("selector value was observed in the declared value type")
    });
    SelectedBlockCompressor::new(
        Box::new(ConstantBlockCompressor::new(value_type, value)),
        ProtobufUtils21::constant(encoded_value),
        false,
    )
}

pub(super) fn range_selection(
    value_type: BlockValueType,
    start: u64,
    step: u64,
) -> SelectedBlockCompressor {
    SelectedBlockCompressor::new(
        Box::new(RangeEncoder::new(value_type.bits_per_value(), start, step)),
        ProtobufUtils21::range(value_type.bits_per_value(), start, step),
        false,
    )
}

fn explicit_general_config(
    field_params: &CompressionFieldParams,
) -> Result<Option<CompressionConfig>> {
    let Some(raw) = field_params.compression.as_deref() else {
        return Ok(None);
    };
    if raw == "none" {
        return Ok(None);
    }
    let scheme: CompressionScheme = raw.parse()?;
    match scheme {
        CompressionScheme::Lz4 | CompressionScheme::Zstd => Ok(Some(CompressionConfig::new(
            scheme,
            field_params.compression_level,
        ))),
        CompressionScheme::None | CompressionScheme::Fsst => Err(Error::invalid_input(format!(
            "Compression scheme '{raw}' is not supported for fixed-width block sequences"
        ))),
    }
}

fn estimate_general_payload(
    stats: &SequenceStats,
    field_params: &CompressionFieldParams,
) -> Result<Option<(CompressionConfig, u64)>> {
    let config = match field_params.compression.as_deref() {
        Some("none") => return Ok(None),
        Some(_) => explicit_general_config(field_params)?,
        None if stats.raw_bytes() > 32 * 1024 => {
            Some(CompressedBufferEncoder::default().compressor.config())
        }
        None => None,
    };
    let Some(config) = config else {
        return Ok(None);
    };
    let sample = stats.sample_bytes();
    if sample.is_empty() {
        return Ok(None);
    }
    Ok(Some((config, estimate_general_with_config(stats, config)?)))
}

pub(super) fn estimate_payload(estimator: PayloadEstimator, stats: &SequenceStats) -> Result<u64> {
    match estimator {
        PayloadEstimator::Flat => Ok(stats.raw_bytes()),
        #[cfg(feature = "bitpacking")]
        PayloadEstimator::InlineBitpacking => Ok((1
            + (BITPACK_CHUNK_VALUES * stats.required_bits()) / stats.value_type.bits_per_value())
        .saturating_mul(stats.value_type.bytes_per_value() as u64)),
        #[cfg(feature = "bitpacking")]
        PayloadEstimator::OutOfLineBitpacking { compressed_bits } => {
            if stats.required_bits() > compressed_bits {
                return Ok(u64::MAX);
            }
            out_of_line_payload_bytes(stats.value_type, stats.len, compressed_bits)
        }
        PayloadEstimator::General { config } => estimate_general_with_config(stats, config),
    }
}
