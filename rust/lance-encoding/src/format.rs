// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

/// Protobuf definitions for encodings
///
/// These are the messages used for describing encoding in the 2.0 format
pub mod pb {
    #![allow(clippy::all)]
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(unused)]
    #![allow(improper_ctypes)]
    #![allow(clippy::upper_case_acronyms)]
    #![allow(clippy::use_self)]
    include!(concat!(env!("OUT_DIR"), "/lance.encodings.rs"));
}

/// Protobuf definitions for encodings21
///
/// These are the messages used for describing encoding in the 2.1 format
/// and any newer formats.
pub mod pb21 {
    #![allow(clippy::all)]
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(unused)]
    #![allow(improper_ctypes)]
    #![allow(clippy::upper_case_acronyms)]
    #![allow(clippy::use_self)]
    include!(concat!(env!("OUT_DIR"), "/lance.encodings21.rs"));
}

use pb::{
    array_encoding::ArrayEncoding as ArrayEncodingEnum,
    buffer::BufferType,
    nullable::{AllNull, NoNull, Nullability, SomeNull},
    ArrayEncoding, Binary, Bitpacked, BitpackedForNonNeg, Block, Dictionary, FixedSizeBinary,
    FixedSizeList, Flat, Fsst, InlineBitpacking, Nullable, OutOfLineBitpacking, PackedStruct,
    PackedStructFixedWidthMiniBlock, Rle, Variable,
};

use crate::{
    encodings::physical::block::CompressionConfig,
    format::pb21::{compressive_encoding::Compression, CompressiveEncoding},
    repdef::DefinitionInterpretation,
};

use self::pb::Constant;
use lance_core::Result;

// Utility functions for creating complex protobuf objects
pub struct ProtobufUtils {}

impl ProtobufUtils {
    pub fn constant(value: Vec<u8>) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Constant(Constant {
                value: value.into(),
            })),
        }
    }

    pub fn basic_all_null_encoding() -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Nullable(Box::new(Nullable {
                nullability: Some(Nullability::AllNulls(AllNull {})),
            }))),
        }
    }

    pub fn basic_some_null_encoding(
        validity: ArrayEncoding,
        values: ArrayEncoding,
    ) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Nullable(Box::new(Nullable {
                nullability: Some(Nullability::SomeNulls(Box::new(SomeNull {
                    validity: Some(Box::new(validity)),
                    values: Some(Box::new(values)),
                }))),
            }))),
        }
    }

    pub fn basic_no_null_encoding(values: ArrayEncoding) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Nullable(Box::new(Nullable {
                nullability: Some(Nullability::NoNulls(Box::new(NoNull {
                    values: Some(Box::new(values)),
                }))),
            }))),
        }
    }

    pub fn block(scheme: &str) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Block(Block {
                scheme: scheme.to_string(),
            })),
        }
    }

    pub fn flat_encoding(
        bits_per_value: u64,
        buffer_index: u32,
        compression: Option<CompressionConfig>,
    ) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Flat(Flat {
                bits_per_value,
                buffer: Some(pb::Buffer {
                    buffer_index,
                    buffer_type: BufferType::Page as i32,
                }),
                compression: compression.map(|compression_config| pb::Compression {
                    scheme: compression_config.scheme.to_string(),
                    level: compression_config.level,
                }),
            })),
        }
    }

    pub fn fsl_encoding(dimension: u64, items: ArrayEncoding, has_validity: bool) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::FixedSizeList(Box::new(FixedSizeList {
                dimension: dimension.try_into().unwrap(),
                items: Some(Box::new(items)),
                has_validity,
            }))),
        }
    }

    pub fn bitpacked_encoding(
        compressed_bits_per_value: u64,
        uncompressed_bits_per_value: u64,
        buffer_index: u32,
        signed: bool,
    ) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Bitpacked(Bitpacked {
                compressed_bits_per_value,
                buffer: Some(pb::Buffer {
                    buffer_index,
                    buffer_type: BufferType::Page as i32,
                }),
                uncompressed_bits_per_value,
                signed,
            })),
        }
    }

    pub fn bitpacked_for_non_neg_encoding(
        compressed_bits_per_value: u64,
        uncompressed_bits_per_value: u64,
        buffer_index: u32,
    ) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::BitpackedForNonNeg(BitpackedForNonNeg {
                compressed_bits_per_value,
                buffer: Some(pb::Buffer {
                    buffer_index,
                    buffer_type: BufferType::Page as i32,
                }),
                uncompressed_bits_per_value,
            })),
        }
    }
    pub fn inline_bitpacking(uncompressed_bits_per_value: u64) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::InlineBitpacking(InlineBitpacking {
                uncompressed_bits_per_value,
            })),
        }
    }
    pub fn out_of_line_bitpacking(
        uncompressed_bits_per_value: u64,
        compressed_bits_per_value: u64,
    ) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::OutOfLineBitpacking(
                OutOfLineBitpacking {
                    uncompressed_bits_per_value,
                    compressed_bits_per_value,
                },
            )),
        }
    }

    pub fn variable(bits_per_offset: u8) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Variable(Variable {
                bits_per_offset: bits_per_offset as u32,
            })),
        }
    }

    // Construct a `FsstMiniBlock` ArrayEncoding, the inner `binary_mini_block` encoding is actually
    // not used and `FsstMiniBlockDecompressor` constructs a `binary_mini_block` in a `hard-coded` fashion.
    // This can be an optimization later.
    pub fn fsst(data: ArrayEncoding, symbol_table: Vec<u8>) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Fsst(Box::new(Fsst {
                binary: Some(Box::new(data)),
                symbol_table: symbol_table.into(),
            }))),
        }
    }

    pub fn rle(bits_per_value: u64) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Rle(Rle { bits_per_value })),
        }
    }

    pub fn byte_stream_split(bits_per_value: u64) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::ByteStreamSplit(pb::ByteStreamSplit {
                bits_per_value,
            })),
        }
    }

    pub fn general_mini_block(
        inner: ArrayEncoding,
        compression: CompressionConfig,
    ) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::GeneralMiniBlock(Box::new(
                pb::GeneralMiniBlock {
                    inner: Some(Box::new(inner)),
                    compression: Some(pb::Compression {
                        scheme: compression.scheme.to_string(),
                        level: compression.level,
                    }),
                },
            ))),
        }
    }

    pub fn packed_struct(
        child_encodings: Vec<ArrayEncoding>,
        packed_buffer_index: u32,
    ) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::PackedStruct(PackedStruct {
                inner: child_encodings,
                buffer: Some(pb::Buffer {
                    buffer_index: packed_buffer_index,
                    buffer_type: BufferType::Page as i32,
                }),
            })),
        }
    }

    pub fn packed_struct_fixed_width_mini_block(
        data: ArrayEncoding,
        bits_per_values: Vec<u32>,
    ) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::PackedStructFixedWidthMiniBlock(
                Box::new(PackedStructFixedWidthMiniBlock {
                    flat: Some(Box::new(data)),
                    bits_per_values,
                }),
            )),
        }
    }

    pub fn binary(
        indices_encoding: ArrayEncoding,
        bytes_encoding: ArrayEncoding,
        null_adjustment: u64,
    ) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Binary(Box::new(Binary {
                bytes: Some(Box::new(bytes_encoding)),
                indices: Some(Box::new(indices_encoding)),
                null_adjustment,
            }))),
        }
    }

    pub fn dict_encoding(
        indices: ArrayEncoding,
        items: ArrayEncoding,
        num_items: u32,
    ) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Dictionary(Box::new(Dictionary {
                indices: Some(Box::new(indices)),
                items: Some(Box::new(items)),
                num_dictionary_items: num_items,
            }))),
        }
    }

    pub fn fixed_size_binary(data: ArrayEncoding, byte_width: u32) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::FixedSizeBinary(Box::new(
                FixedSizeBinary {
                    bytes: Some(Box::new(data)),
                    byte_width,
                },
            ))),
        }
    }
}

pub struct ProtobufUtils21 {}

impl ProtobufUtils21 {
    pub fn flat(
        bits_per_value: u64,
        values_compression: Option<pb21::BufferCompression>,
    ) -> CompressiveEncoding {
        CompressiveEncoding {
            compression: Some(Compression::Flat(pb21::Flat {
                bits_per_value,
                data: values_compression,
            })),
        }
    }

    pub fn fsl(
        items_per_value: u64,
        has_validity: bool,
        values: CompressiveEncoding,
    ) -> CompressiveEncoding {
        CompressiveEncoding {
            compression: Some(Compression::FixedSizeList(Box::new(pb21::FixedSizeList {
                items_per_value,
                has_validity,
                values: Some(Box::new(values)),
            }))),
        }
    }

    pub fn variable(
        offsets_desc: CompressiveEncoding,
        values_compression: Option<pb21::BufferCompression>,
    ) -> CompressiveEncoding {
        CompressiveEncoding {
            compression: Some(Compression::Variable(Box::new(pb21::Variable {
                offsets: Some(Box::new(offsets_desc)),
                values: values_compression,
            }))),
        }
    }

    pub fn inline_bitpacking(
        uncompressed_bits_per_value: u64,
        values_compression: Option<pb21::BufferCompression>,
    ) -> CompressiveEncoding {
        CompressiveEncoding {
            compression: Some(Compression::InlineBitpacking(pb21::InlineBitpacking {
                uncompressed_bits_per_value,
                values: values_compression,
            })),
        }
    }

    pub fn out_of_line_bitpacking(
        uncompressed_bits_per_value: u64,
        values: CompressiveEncoding,
    ) -> CompressiveEncoding {
        CompressiveEncoding {
            compression: Some(Compression::OutOfLineBitpacking(Box::new(
                pb21::OutOfLineBitpacking {
                    uncompressed_bits_per_value,
                    values: Some(Box::new(values)),
                },
            ))),
        }
    }

    pub fn buffer_compression(compression: CompressionConfig) -> Result<pb21::BufferCompression> {
        Ok(pb21::BufferCompression {
            scheme: pb21::CompressionScheme::try_from(compression.scheme)? as i32,
            level: compression.level,
        })
    }

    pub fn wrapped(
        compression: CompressionConfig,
        values: CompressiveEncoding,
    ) -> Result<CompressiveEncoding> {
        Ok(CompressiveEncoding {
            compression: Some(Compression::General(Box::new(pb21::General {
                compression: Some(Self::buffer_compression(compression)?),
                values: Some(Box::new(values)),
            }))),
        })
    }

    pub fn rle(
        values: CompressiveEncoding,
        run_lengths: CompressiveEncoding,
    ) -> CompressiveEncoding {
        CompressiveEncoding {
            compression: Some(Compression::Rle(Box::new(pb21::Rle {
                values: Some(Box::new(values)),
                run_lengths: Some(Box::new(run_lengths)),
            }))),
        }
    }

    pub fn byte_stream_split(values: CompressiveEncoding) -> CompressiveEncoding {
        CompressiveEncoding {
            compression: Some(Compression::ByteStreamSplit(Box::new(
                pb21::ByteStreamSplit {
                    values: Some(Box::new(values)),
                },
            ))),
        }
    }

    pub fn fsst(data: CompressiveEncoding, symbol_table: Vec<u8>) -> CompressiveEncoding {
        CompressiveEncoding {
            compression: Some(Compression::Fsst(Box::new(pb21::Fsst {
                symbol_table: symbol_table.into(),
                values: Some(Box::new(data)),
            }))),
        }
    }

    pub fn packed_struct(
        values: CompressiveEncoding,
        bits_per_values: Vec<u64>,
    ) -> CompressiveEncoding {
        CompressiveEncoding {
            compression: Some(Compression::PackedStruct(Box::new(pb21::PackedStruct {
                values: Some(Box::new(values)),
                bits_per_value: bits_per_values,
            }))),
        }
    }

    fn def_inter_to_repdef_layer(def: DefinitionInterpretation) -> i32 {
        match def {
            DefinitionInterpretation::AllValidItem => pb21::RepDefLayer::RepdefAllValidItem as i32,
            DefinitionInterpretation::AllValidList => pb21::RepDefLayer::RepdefAllValidList as i32,
            DefinitionInterpretation::NullableItem => pb21::RepDefLayer::RepdefNullableItem as i32,
            DefinitionInterpretation::NullableList => pb21::RepDefLayer::RepdefNullableList as i32,
            DefinitionInterpretation::EmptyableList => {
                pb21::RepDefLayer::RepdefEmptyableList as i32
            }
            DefinitionInterpretation::NullableAndEmptyableList => {
                pb21::RepDefLayer::RepdefNullAndEmptyList as i32
            }
        }
    }

    pub fn repdef_layer_to_def_interp(layer: i32) -> DefinitionInterpretation {
        let layer = pb21::RepDefLayer::try_from(layer).unwrap();
        match layer {
            pb21::RepDefLayer::RepdefAllValidItem => DefinitionInterpretation::AllValidItem,
            pb21::RepDefLayer::RepdefAllValidList => DefinitionInterpretation::AllValidList,
            pb21::RepDefLayer::RepdefNullableItem => DefinitionInterpretation::NullableItem,
            pb21::RepDefLayer::RepdefNullableList => DefinitionInterpretation::NullableList,
            pb21::RepDefLayer::RepdefEmptyableList => DefinitionInterpretation::EmptyableList,
            pb21::RepDefLayer::RepdefNullAndEmptyList => {
                DefinitionInterpretation::NullableAndEmptyableList
            }
            pb21::RepDefLayer::RepdefUnspecified => panic!("Unspecified repdef layer"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn miniblock_layout(
        rep_encoding: Option<CompressiveEncoding>,
        def_encoding: Option<CompressiveEncoding>,
        value_encoding: CompressiveEncoding,
        repetition_index_depth: u32,
        num_buffers: u64,
        dictionary_encoding: Option<(CompressiveEncoding, u64)>,
        def_meaning: &[DefinitionInterpretation],
        num_items: u64,
    ) -> pb21::PageLayout {
        assert!(!def_meaning.is_empty());
        let (dictionary, num_dictionary_items) = dictionary_encoding
            .map(|(d, i)| (Some(d), i))
            .unwrap_or((None, 0));
        pb21::PageLayout {
            layout: Some(pb21::page_layout::Layout::MiniBlockLayout(
                pb21::MiniBlockLayout {
                    def_compression: def_encoding,
                    rep_compression: rep_encoding,
                    value_compression: Some(value_encoding),
                    repetition_index_depth,
                    num_buffers,
                    dictionary,
                    num_dictionary_items,
                    layers: def_meaning
                        .iter()
                        .map(|&def| Self::def_inter_to_repdef_layer(def))
                        .collect(),
                    num_items,
                },
            )),
        }
    }

    fn full_zip_layout(
        bits_rep: u8,
        bits_def: u8,
        details: pb21::full_zip_layout::Details,
        value_encoding: CompressiveEncoding,
        def_meaning: &[DefinitionInterpretation],
        num_items: u32,
        num_visible_items: u32,
    ) -> pb21::PageLayout {
        pb21::PageLayout {
            layout: Some(pb21::page_layout::Layout::FullZipLayout(
                pb21::FullZipLayout {
                    bits_rep: bits_rep as u32,
                    bits_def: bits_def as u32,
                    details: Some(details),
                    value_compression: Some(value_encoding),
                    num_items,
                    num_visible_items,
                    layers: def_meaning
                        .iter()
                        .map(|&def| Self::def_inter_to_repdef_layer(def))
                        .collect(),
                },
            )),
        }
    }

    pub fn fixed_full_zip_layout(
        bits_rep: u8,
        bits_def: u8,
        bits_per_value: u32,
        value_encoding: CompressiveEncoding,
        def_meaning: &[DefinitionInterpretation],
        num_items: u32,
        num_visible_items: u32,
    ) -> pb21::PageLayout {
        Self::full_zip_layout(
            bits_rep,
            bits_def,
            pb21::full_zip_layout::Details::BitsPerValue(bits_per_value),
            value_encoding,
            def_meaning,
            num_items,
            num_visible_items,
        )
    }

    pub fn variable_full_zip_layout(
        bits_rep: u8,
        bits_def: u8,
        bits_per_offset: u32,
        value_encoding: CompressiveEncoding,
        def_meaning: &[DefinitionInterpretation],
        num_items: u32,
        num_visible_items: u32,
    ) -> pb21::PageLayout {
        Self::full_zip_layout(
            bits_rep,
            bits_def,
            pb21::full_zip_layout::Details::BitsPerOffset(bits_per_offset),
            value_encoding,
            def_meaning,
            num_items,
            num_visible_items,
        )
    }

    pub fn all_null_layout(def_meaning: &[DefinitionInterpretation]) -> pb21::PageLayout {
        pb21::PageLayout {
            layout: Some(pb21::page_layout::Layout::AllNullLayout(
                pb21::AllNullLayout {
                    layers: def_meaning
                        .iter()
                        .map(|&def| Self::def_inter_to_repdef_layer(def))
                        .collect(),
                },
            )),
        }
    }

    pub fn simple_all_null_layout() -> pb21::PageLayout {
        Self::all_null_layout(&[DefinitionInterpretation::NullableItem])
    }
}
