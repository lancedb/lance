// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

/// Protobuf definitions for encodings
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

use pb::{
    array_encoding::ArrayEncoding as ArrayEncodingEnum,
    buffer::BufferType,
    full_zip_layout,
    nullable::{AllNull, NoNull, Nullability, SomeNull},
    page_layout::Layout,
    AllNullLayout, ArrayEncoding, Binary, Bitpack2, Bitpacked, BitpackedForNonNeg, Dictionary,
    FixedSizeBinary, FixedSizeList, Flat, Fsst, MiniBlockLayout, Nullable, PackedStruct,
    PackedStructFixedWidthMiniBlock, PageLayout, RepDefLayer, Variable,
};

use crate::{
    encodings::physical::block_compress::CompressionConfig, repdef::DefinitionInterpretation,
};

use self::pb::Constant;

// Utility functions for creating complex protobuf objects
pub struct ProtobufUtils {}

impl ProtobufUtils {
    pub fn constant(value: Vec<u8>, num_values: u64) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Constant(Constant {
                value: value.into(),
                num_values,
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

    pub fn fsl_encoding(dimension: u64, items: ArrayEncoding) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::FixedSizeList(Box::new(FixedSizeList {
                dimension: dimension.try_into().unwrap(),
                items: Some(Box::new(items)),
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
    pub fn bitpack2(uncompressed_bits_per_value: u64) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Bitpack2(Bitpack2 {
                uncompressed_bits_per_value,
            })),
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

    pub fn fixed_size_list(data: ArrayEncoding, dimension: u64) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::FixedSizeList(Box::new(FixedSizeList {
                dimension: dimension.try_into().unwrap(),
                items: Some(Box::new(data)),
            }))),
        }
    }

    fn def_inter_to_repdef_layer(def: DefinitionInterpretation) -> i32 {
        match def {
            DefinitionInterpretation::AllValidItem => RepDefLayer::RepdefAllValidItem as i32,
            DefinitionInterpretation::AllValidList => RepDefLayer::RepdefAllValidList as i32,
            DefinitionInterpretation::NullableItem => RepDefLayer::RepdefNullableItem as i32,
            DefinitionInterpretation::NullableList => RepDefLayer::RepdefNullableList as i32,
            DefinitionInterpretation::EmptyableList => RepDefLayer::RepdefEmptyableList as i32,
            DefinitionInterpretation::NullableAndEmptyableList => {
                RepDefLayer::RepdefNullAndEmptyList as i32
            }
        }
    }

    pub fn repdef_layer_to_def_interp(layer: i32) -> DefinitionInterpretation {
        let layer = RepDefLayer::try_from(layer).unwrap();
        match layer {
            RepDefLayer::RepdefAllValidItem => DefinitionInterpretation::AllValidItem,
            RepDefLayer::RepdefAllValidList => DefinitionInterpretation::AllValidList,
            RepDefLayer::RepdefNullableItem => DefinitionInterpretation::NullableItem,
            RepDefLayer::RepdefNullableList => DefinitionInterpretation::NullableList,
            RepDefLayer::RepdefEmptyableList => DefinitionInterpretation::EmptyableList,
            RepDefLayer::RepdefNullAndEmptyList => {
                DefinitionInterpretation::NullableAndEmptyableList
            }
            RepDefLayer::RepdefUnspecified => panic!("Unspecified repdef layer"),
        }
    }

    pub fn miniblock_layout(
        rep_encoding: ArrayEncoding,
        def_encoding: ArrayEncoding,
        value_encoding: ArrayEncoding,
        repetition_index_depth: u32,
        dictionary_encoding: Option<ArrayEncoding>,
        def_meaning: &[DefinitionInterpretation],
        num_items: u64,
    ) -> PageLayout {
        assert!(!def_meaning.is_empty());
        PageLayout {
            layout: Some(Layout::MiniBlockLayout(MiniBlockLayout {
                def_compression: Some(def_encoding),
                rep_compression: Some(rep_encoding),
                value_compression: Some(value_encoding),
                repetition_index_depth,
                dictionary: dictionary_encoding,
                layers: def_meaning
                    .iter()
                    .map(|&def| Self::def_inter_to_repdef_layer(def))
                    .collect(),
                num_items,
            })),
        }
    }

    fn full_zip_layout(
        bits_rep: u8,
        bits_def: u8,
        details: full_zip_layout::Details,
        value_encoding: ArrayEncoding,
        def_meaning: &[DefinitionInterpretation],
        num_items: u32,
        num_visible_items: u32,
    ) -> PageLayout {
        PageLayout {
            layout: Some(Layout::FullZipLayout(pb::FullZipLayout {
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
            })),
        }
    }

    pub fn fixed_full_zip_layout(
        bits_rep: u8,
        bits_def: u8,
        bits_per_value: u32,
        value_encoding: ArrayEncoding,
        def_meaning: &[DefinitionInterpretation],
        num_items: u32,
        num_visible_items: u32,
    ) -> PageLayout {
        Self::full_zip_layout(
            bits_rep,
            bits_def,
            full_zip_layout::Details::BitsPerValue(bits_per_value),
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
        value_encoding: ArrayEncoding,
        def_meaning: &[DefinitionInterpretation],
        num_items: u32,
        num_visible_items: u32,
    ) -> PageLayout {
        Self::full_zip_layout(
            bits_rep,
            bits_def,
            full_zip_layout::Details::BitsPerOffset(bits_per_offset),
            value_encoding,
            def_meaning,
            num_items,
            num_visible_items,
        )
    }

    pub fn all_null_layout(def_meaning: &[DefinitionInterpretation]) -> PageLayout {
        PageLayout {
            layout: Some(Layout::AllNullLayout(AllNullLayout {
                layers: def_meaning
                    .iter()
                    .map(|&def| Self::def_inter_to_repdef_layer(def))
                    .collect(),
            })),
        }
    }

    pub fn simple_all_null_layout() -> PageLayout {
        Self::all_null_layout(&[DefinitionInterpretation::NullableItem])
    }
}
