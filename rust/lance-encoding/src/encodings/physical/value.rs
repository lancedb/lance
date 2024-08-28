// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::ArrayRef;
use bytes::Bytes;
use futures::{future::BoxFuture, FutureExt};
use log::trace;
use snafu::{location, Location};
use std::fmt;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use crate::buffer::LanceBuffer;
use crate::data::{DataBlock, FixedWidthDataBlock};
use crate::encoder::BufferEncodingStrategy;
use crate::{
    decoder::{PageScheduler, PrimitivePageDecoder},
    encoder::{ArrayEncoder, EncodedArray, EncodedArrayBuffer},
    format::pb,
    EncodingsIo,
};

use lance_core::{Error, Result};

use super::buffers::GeneralBufferCompressor;

pub const COMPRESSION_META_KEY: &str = "lance:compression";

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionScheme {
    None,
    Zstd,
}

impl fmt::Display for CompressionScheme {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let scheme_str = match self {
            Self::Zstd => "zstd",
            Self::None => "none",
        };
        write!(f, "{}", scheme_str)
    }
}

pub fn parse_compression_scheme(scheme: &str) -> Result<CompressionScheme> {
    match scheme {
        "none" => Ok(CompressionScheme::None),
        "zstd" => Ok(CompressionScheme::Zstd),
        _ => Err(Error::invalid_input(
            format!("Unknown compression scheme: {}", scheme),
            location!(),
        )),
    }
}

/// Scheduler for a simple encoding where buffers of fixed-size items are stored as-is on disk
#[derive(Debug, Clone, Copy)]
pub struct ValuePageScheduler {
    // TODO: do we really support values greater than 2^32 bytes per value?
    // I think we want to, in theory, but will need to test this case.
    bytes_per_value: u64,
    buffer_offset: u64,
    buffer_size: u64,
    compression_scheme: CompressionScheme,
}

impl ValuePageScheduler {
    pub fn new(
        bytes_per_value: u64,
        buffer_offset: u64,
        buffer_size: u64,
        compression_scheme: CompressionScheme,
    ) -> Self {
        Self {
            bytes_per_value,
            buffer_offset,
            buffer_size,
            compression_scheme,
        }
    }
}

impl PageScheduler for ValuePageScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u64>],
        scheduler: &Arc<dyn EncodingsIo>,
        top_level_row: u64,
    ) -> BoxFuture<'static, Result<Box<dyn PrimitivePageDecoder>>> {
        let (mut min, mut max) = (u64::MAX, 0);
        let byte_ranges = if self.compression_scheme == CompressionScheme::None {
            ranges
                .iter()
                .map(|range| {
                    let start = self.buffer_offset + (range.start * self.bytes_per_value);
                    let end = self.buffer_offset + (range.end * self.bytes_per_value);
                    min = min.min(start);
                    max = max.max(end);
                    start..end
                })
                .collect::<Vec<_>>()
        } else {
            min = self.buffer_offset;
            max = self.buffer_offset + self.buffer_size;
            // for compressed page, the ranges are always the entire page,
            // and it is guaranteed that only one range is passed
            vec![Range {
                start: min,
                end: max,
            }]
        };

        trace!(
            "Scheduling I/O for {} ranges spread across byte range {}..{}",
            byte_ranges.len(),
            min,
            max
        );
        let bytes = scheduler.submit_request(byte_ranges, top_level_row);
        let bytes_per_value = self.bytes_per_value;

        let range_offsets = if self.compression_scheme != CompressionScheme::None {
            ranges
                .iter()
                .map(|range| {
                    let start = (range.start * bytes_per_value) as usize;
                    let end = (range.end * bytes_per_value) as usize;
                    start..end
                })
                .collect::<Vec<_>>()
        } else {
            vec![]
        };

        async move {
            let bytes = bytes.await?;

            Ok(Box::new(ValuePageDecoder {
                bytes_per_value,
                data: bytes,
                uncompressed_data: Arc::new(Mutex::new(None)),
                uncompressed_range_offsets: range_offsets,
            }) as Box<dyn PrimitivePageDecoder>)
        }
        .boxed()
    }
}

struct ValuePageDecoder {
    bytes_per_value: u64,
    data: Vec<Bytes>,
    uncompressed_data: Arc<Mutex<Option<Vec<Bytes>>>>,
    uncompressed_range_offsets: Vec<std::ops::Range<usize>>,
}

impl ValuePageDecoder {
    fn decompress(&self) -> Result<Vec<Bytes>> {
        // for compressed page, it is guaranteed that only one range is passed
        let bytes_u8: Vec<u8> = self.data[0].to_vec();
        let buffer_compressor = GeneralBufferCompressor::get_compressor("");
        let mut uncompressed_bytes: Vec<u8> = Vec::new();
        buffer_compressor.decompress(&bytes_u8, &mut uncompressed_bytes)?;

        let mut bytes_in_ranges: Vec<Bytes> =
            Vec::with_capacity(self.uncompressed_range_offsets.len());
        for range in &self.uncompressed_range_offsets {
            let start = range.start;
            let end = range.end;
            bytes_in_ranges.push(Bytes::from(uncompressed_bytes[start..end].to_vec()));
        }
        Ok(bytes_in_ranges)
    }

    fn get_uncompressed_bytes(&self) -> Result<Arc<Mutex<Option<Vec<Bytes>>>>> {
        let mut uncompressed_bytes = self.uncompressed_data.lock().unwrap();
        if uncompressed_bytes.is_none() {
            *uncompressed_bytes = Some(self.decompress()?);
        }
        Ok(Arc::clone(&self.uncompressed_data))
    }

    fn is_compressed(&self) -> bool {
        !self.uncompressed_range_offsets.is_empty()
    }

    fn decode_buffers<'a>(
        &'a self,
        buffers: impl IntoIterator<Item = &'a Bytes>,
        mut bytes_to_skip: u64,
        mut bytes_to_take: u64,
    ) -> LanceBuffer {
        let mut dest: Option<Vec<u8>> = None;

        for buf in buffers.into_iter() {
            let buf_len = buf.len() as u64;
            if bytes_to_skip > buf_len {
                bytes_to_skip -= buf_len;
            } else {
                let bytes_to_take_here = (buf_len - bytes_to_skip).min(bytes_to_take);
                bytes_to_take -= bytes_to_take_here;
                let start = bytes_to_skip as usize;
                let end = start + bytes_to_take_here as usize;
                let slice = buf.slice(start..end);
                match (&mut dest, bytes_to_take) {
                    (None, 0) => {
                        // The entire request is contained in one buffer so we can maybe zero-copy
                        // if the slice is aligned properly
                        return LanceBuffer::from_bytes(slice, self.bytes_per_value);
                    }
                    (None, _) => {
                        dest.replace(Vec::with_capacity(bytes_to_take as usize));
                    }
                    _ => {}
                }
                dest.as_mut().unwrap().extend_from_slice(&slice);
                bytes_to_skip = 0;
            }
        }
        LanceBuffer::from(dest.unwrap_or_default())
    }
}

impl PrimitivePageDecoder for ValuePageDecoder {
    fn decode(&self, rows_to_skip: u64, num_rows: u64) -> Result<DataBlock> {
        let bytes_to_skip = rows_to_skip * self.bytes_per_value;
        let bytes_to_take = num_rows * self.bytes_per_value;

        let data_buffer = if self.is_compressed() {
            let decoding_data = self.get_uncompressed_bytes()?;
            let buffers = decoding_data.lock().unwrap();
            self.decode_buffers(buffers.as_ref().unwrap(), bytes_to_skip, bytes_to_take)
        } else {
            self.decode_buffers(&self.data, bytes_to_skip, bytes_to_take)
        };
        Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: self.bytes_per_value * 8,
            data: data_buffer,
            num_values: num_rows,
        }))
    }
}

#[derive(Debug)]
pub struct ValueEncoder {
    buffer_encoding_strategy: Arc<dyn BufferEncodingStrategy>,
}

impl ValueEncoder {
    pub fn try_new(buffer_encoding_strategy: Arc<dyn BufferEncodingStrategy>) -> Result<Self> {
        Ok(Self {
            buffer_encoding_strategy,
        })
    }
}

impl ArrayEncoder for ValueEncoder {
    fn encode(&self, arrays: &[ArrayRef], buffer_index: &mut u32) -> Result<EncodedArray> {
        let index = *buffer_index;
        *buffer_index += 1;

        let buffer_encoder = self
            .buffer_encoding_strategy
            .create_buffer_encoder(arrays)?;
        let (encoded_buffer, encoded_buffer_meta) = buffer_encoder.encode(arrays)?;

        let array_encoding = if let Some(bitpacking_meta) = encoded_buffer_meta.bitpacking {
            pb::array_encoding::ArrayEncoding::Bitpacked(pb::Bitpacked {
                compressed_bits_per_value: bitpacking_meta.bits_per_value,
                uncompressed_bits_per_value: encoded_buffer_meta.bits_per_value,
                signed: bitpacking_meta.signed,
                buffer: Some(pb::Buffer {
                    buffer_index: index,
                    buffer_type: pb::buffer::BufferType::Page as i32,
                }),
            })
        } else {
            pb::array_encoding::ArrayEncoding::Flat(pb::Flat {
                bits_per_value: encoded_buffer_meta.bits_per_value,
                buffer: Some(pb::Buffer {
                    buffer_index: index,
                    buffer_type: pb::buffer::BufferType::Page as i32,
                }),
                compression: encoded_buffer_meta
                    .compression_scheme
                    .map(|compression_scheme| pb::Compression {
                        scheme: compression_scheme.to_string(),
                    }),
            })
        };

        let array_bufs = vec![EncodedArrayBuffer {
            parts: encoded_buffer.parts,
            index,
        }];

        Ok(EncodedArray {
            buffers: array_bufs,
            encoding: pb::ArrayEncoding {
                array_encoding: Some(array_encoding),
            },
        })
    }
}

// public tests module because we share the PRIMITIVE_TYPES constant with fixed_size_list
#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;

    use super::*;

    use std::marker::PhantomData;
    use std::sync::Arc;

    use arrow::datatypes::{Int16Type, Int32Type, Int64Type};
    use arrow_array::{
        types::{UInt32Type, UInt64Type, UInt8Type},
        ArrayRef, ArrowPrimitiveType, Float32Array, Int16Array, Int32Array, Int64Array, Int8Array,
        PrimitiveArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
    };
    use arrow_schema::{DataType, Field, TimeUnit};
    use rand::distributions::Uniform;

    use lance_arrow::DataTypeExt;
    use lance_datagen::{array::rand_with_distribution, ArrayGenerator};

    use crate::{
        encoder::{ArrayEncoder, CoreBufferEncodingStrategy},
        testing::{
            check_round_trip_encoding_generated, check_round_trip_encoding_random,
            ArrayGeneratorProvider,
        },
        version::LanceFileVersion,
    };

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
    async fn test_value_primitive() {
        for data_type in PRIMITIVE_TYPES {
            let field = Field::new("", data_type.clone(), false);
            check_round_trip_encoding_random(field, HashMap::new()).await;
        }
    }

    #[test_log::test(test)]
    fn test_will_bitpack_allowed_types_when_possible() {
        let test_cases: Vec<(DataType, ArrayRef, u64)> = vec![
            (
                DataType::UInt8,
                Arc::new(UInt8Array::from_iter_values(vec![0, 1, 2, 3, 4, 5])),
                3, // bits per value
            ),
            (
                DataType::UInt16,
                Arc::new(UInt16Array::from_iter_values(vec![0, 1, 2, 3, 4, 5 << 8])),
                11,
            ),
            (
                DataType::UInt32,
                Arc::new(UInt32Array::from_iter_values(vec![0, 1, 2, 3, 4, 5 << 16])),
                19,
            ),
            (
                DataType::UInt64,
                Arc::new(UInt64Array::from_iter_values(vec![0, 1, 2, 3, 4, 5 << 32])),
                35,
            ),
            (
                DataType::Int8,
                Arc::new(Int8Array::from_iter_values(vec![0, 2, 3, 4, -5])),
                4,
            ),
            (
                // check it will not pack with signed bit if all values of signed type are positive
                DataType::Int8,
                Arc::new(Int8Array::from_iter_values(vec![0, 2, 3, 4, 5])),
                3,
            ),
            (
                DataType::Int16,
                Arc::new(Int16Array::from_iter_values(vec![0, 1, 2, 3, -4, 5 << 8])),
                12,
            ),
            (
                DataType::Int32,
                Arc::new(Int32Array::from_iter_values(vec![0, 1, 2, 3, 4, -5 << 16])),
                20,
            ),
            (
                DataType::Int64,
                Arc::new(Int64Array::from_iter_values(vec![
                    0,
                    1,
                    2,
                    -3,
                    -4,
                    -5 << 32,
                ])),
                36,
            ),
        ];

        for (data_type, arr, bits_per_value) in test_cases {
            let arrs = vec![arr.clone() as _];
            let mut buffed_index = 1;
            let encoder = ValueEncoder::try_new(Arc::new(CoreBufferEncodingStrategy {
                compression_scheme: CompressionScheme::None,
                version: LanceFileVersion::V2_1,
            }))
            .unwrap();
            let result = encoder.encode(&arrs, &mut buffed_index).unwrap();
            let array_encoding = result.encoding.array_encoding.unwrap();

            match array_encoding {
                pb::array_encoding::ArrayEncoding::Bitpacked(bitpacked) => {
                    assert_eq!(bits_per_value, bitpacked.compressed_bits_per_value);
                    assert_eq!(
                        (data_type.byte_width() * 8) as u64,
                        bitpacked.uncompressed_bits_per_value
                    );
                }
                _ => {
                    panic!("Array did not use bitpacking encoding")
                }
            }
        }

        // check it will otherwise use flat encoding
        let test_cases: Vec<(DataType, ArrayRef)> = vec![
            // it should use flat encoding for datatypes that don't support bitpacking
            (
                DataType::Float32,
                Arc::new(Float32Array::from_iter_values(vec![0.1, 0.2, 0.3])),
            ),
            // it should still use flat encoding if bitpacked encoding would be packed
            // into the full byte range
            (
                DataType::UInt8,
                Arc::new(UInt8Array::from_iter_values(vec![0, 1, 2, 3, 4, 250])),
            ),
            (
                DataType::UInt16,
                Arc::new(UInt16Array::from_iter_values(vec![0, 1, 2, 3, 4, 250 << 8])),
            ),
            (
                DataType::UInt32,
                Arc::new(UInt32Array::from_iter_values(vec![
                    0,
                    1,
                    2,
                    3,
                    4,
                    250 << 24,
                ])),
            ),
            (
                DataType::UInt64,
                Arc::new(UInt64Array::from_iter_values(vec![
                    0,
                    1,
                    2,
                    3,
                    4,
                    250 << 56,
                ])),
            ),
            (
                DataType::Int8,
                Arc::new(Int8Array::from_iter_values(vec![-100])),
            ),
            (
                DataType::Int16,
                Arc::new(Int16Array::from_iter_values(vec![-100 << 8])),
            ),
            (
                DataType::Int32,
                Arc::new(Int32Array::from_iter_values(vec![-100 << 24])),
            ),
            (
                DataType::Int64,
                Arc::new(Int64Array::from_iter_values(vec![-100 << 56])),
            ),
        ];

        for (data_type, arr) in test_cases {
            let arrs = vec![arr.clone() as _];
            let mut buffed_index = 1;
            let encoder = ValueEncoder::try_new(Arc::new(CoreBufferEncodingStrategy {
                compression_scheme: CompressionScheme::None,
                version: LanceFileVersion::default_v2(),
            }))
            .unwrap();
            let result = encoder.encode(&arrs, &mut buffed_index).unwrap();
            let array_encoding = result.encoding.array_encoding.unwrap();

            match array_encoding {
                pb::array_encoding::ArrayEncoding::Flat(flat) => {
                    assert_eq!((data_type.byte_width() * 8) as u64, flat.bits_per_value);
                }
                _ => {
                    panic!("Array did not use bitpacking encoding")
                }
            }
        }
    }

    struct DistributionArrayGeneratorProvider<
        DataType,
        Dist: rand::distributions::Distribution<DataType::Native> + Clone + Send + Sync + 'static,
    >
    where
        DataType::Native: Copy + 'static,
        PrimitiveArray<DataType>: From<Vec<DataType::Native>> + 'static,
        DataType: ArrowPrimitiveType,
    {
        phantom: PhantomData<DataType>,
        distribution: Dist,
    }

    impl<DataType, Dist> DistributionArrayGeneratorProvider<DataType, Dist>
    where
        Dist: rand::distributions::Distribution<DataType::Native> + Clone + Send + Sync + 'static,
        DataType::Native: Copy + 'static,
        PrimitiveArray<DataType>: From<Vec<DataType::Native>> + 'static,
        DataType: ArrowPrimitiveType,
    {
        fn new(dist: Dist) -> Self {
            Self {
                distribution: dist,
                phantom: Default::default(),
            }
        }
    }

    impl<DataType, Dist> ArrayGeneratorProvider for DistributionArrayGeneratorProvider<DataType, Dist>
    where
        Dist: rand::distributions::Distribution<DataType::Native> + Clone + Send + Sync + 'static,
        DataType::Native: Copy + 'static,
        PrimitiveArray<DataType>: From<Vec<DataType::Native>> + 'static,
        DataType: ArrowPrimitiveType,
    {
        fn provide(&self) -> Box<dyn ArrayGenerator> {
            rand_with_distribution::<DataType, Dist>(self.distribution.clone())
        }

        fn copy(&self) -> Box<dyn ArrayGeneratorProvider> {
            Box::new(Self {
                phantom: self.phantom,
                distribution: self.distribution.clone(),
            })
        }
    }

    #[test_log::test(tokio::test)]
    async fn test_bitpack_primitive() {
        let bitpacked_test_cases: &Vec<(DataType, Box<dyn ArrayGeneratorProvider>)> = &vec![
            // check less than one byte for multi-byte type
            (
                DataType::UInt32,
                Box::new(
                    DistributionArrayGeneratorProvider::<UInt32Type, Uniform<u32>>::new(
                        Uniform::new(0, 19),
                    ),
                ),
            ),
            // // check that more than one byte for multi-byte type
            (
                DataType::UInt32,
                Box::new(
                    DistributionArrayGeneratorProvider::<UInt32Type, Uniform<u32>>::new(
                        Uniform::new(5 << 7, 6 << 7),
                    ),
                ),
            ),
            (
                DataType::UInt64,
                Box::new(
                    DistributionArrayGeneratorProvider::<UInt64Type, Uniform<u64>>::new(
                        Uniform::new(5 << 42, 6 << 42),
                    ),
                ),
            ),
            // check less than one byte for single-byte type
            (
                DataType::UInt8,
                Box::new(
                    DistributionArrayGeneratorProvider::<UInt8Type, Uniform<u8>>::new(
                        Uniform::new(0, 19),
                    ),
                ),
            ),
            // check less than one byte for single-byte type
            (
                DataType::UInt64,
                Box::new(
                    DistributionArrayGeneratorProvider::<UInt64Type, Uniform<u64>>::new(
                        Uniform::new(129, 259),
                    ),
                ),
            ),
            // check byte aligned for single byte
            (
                DataType::UInt32,
                Box::new(
                    DistributionArrayGeneratorProvider::<UInt32Type, Uniform<u32>>::new(
                        // this range should always give 8 bits
                        Uniform::new(200, 250),
                    ),
                ),
            ),
            // check where the num_bits divides evenly into the bit length of the type
            (
                DataType::UInt64,
                Box::new(
                    DistributionArrayGeneratorProvider::<UInt64Type, Uniform<u64>>::new(
                        Uniform::new(1, 3), // 2 bits
                    ),
                ),
            ),
            // check byte aligned for multiple bytes
            (
                DataType::UInt32,
                Box::new(
                    DistributionArrayGeneratorProvider::<UInt32Type, Uniform<u32>>::new(
                        // this range should always always give 16 bits
                        Uniform::new(200 << 8, 250 << 8),
                    ),
                ),
            ),
            // check byte aligned where the num bits doesn't divide evenly into the byte length
            (
                DataType::UInt64,
                Box::new(
                    DistributionArrayGeneratorProvider::<UInt64Type, Uniform<u64>>::new(
                        // this range should always give 24 hits
                        Uniform::new(200 << 16, 250 << 16),
                    ),
                ),
            ),
            // check that we can still encode an all-0 array
            (
                DataType::UInt32,
                Box::new(
                    DistributionArrayGeneratorProvider::<UInt32Type, Uniform<u32>>::new(
                        // this range should always always give 16 bits
                        Uniform::new(0, 1),
                    ),
                ),
            ),
            // check for signed types
            (
                DataType::Int16,
                Box::new(
                    DistributionArrayGeneratorProvider::<Int16Type, Uniform<i16>>::new(
                        Uniform::new(-5, 5),
                    ),
                ),
            ),
            (
                DataType::Int64,
                Box::new(
                    DistributionArrayGeneratorProvider::<Int64Type, Uniform<i64>>::new(
                        Uniform::new(-(5 << 42), 6 << 42),
                    ),
                ),
            ),
            (
                DataType::Int32,
                Box::new(
                    DistributionArrayGeneratorProvider::<Int32Type, Uniform<i32>>::new(
                        Uniform::new(-(5 << 7), 6 << 7),
                    ),
                ),
            ),
            // check signed where packed to < 1 byte for multi-byte type
            (
                DataType::Int32,
                Box::new(
                    DistributionArrayGeneratorProvider::<Int32Type, Uniform<i32>>::new(
                        Uniform::new(-19, 19),
                    ),
                ),
            ),
            // check signed byte aligned to single byte
            (
                DataType::Int32,
                Box::new(
                    DistributionArrayGeneratorProvider::<Int32Type, Uniform<i32>>::new(
                        // this range should always give 8 bits
                        Uniform::new(-120, 120),
                    ),
                ),
            ),
            // check signed byte aligned to multiple bytes
            (
                DataType::Int32,
                Box::new(
                    DistributionArrayGeneratorProvider::<Int32Type, Uniform<i32>>::new(
                        // this range should always give 16 bits
                        Uniform::new(-120 << 8, 120 << 8),
                    ),
                ),
            ),
            // check that it works for all positive integers even if type is signed
            (
                DataType::Int32,
                Box::new(
                    DistributionArrayGeneratorProvider::<Int32Type, Uniform<i32>>::new(
                        Uniform::new(10, 20),
                    ),
                ),
            ),
            // check that all 0 works for signed type
            (
                DataType::Int32,
                Box::new(
                    DistributionArrayGeneratorProvider::<Int32Type, Uniform<i32>>::new(
                        Uniform::new(0, 1),
                    ),
                ),
            ),
        ];

        for (data_type, array_gen_provider) in bitpacked_test_cases {
            let field = Field::new("", data_type.clone(), false);
            check_round_trip_encoding_generated(
                field,
                array_gen_provider.copy(),
                LanceFileVersion::V2_1,
            )
            .await;
        }
    }
}
