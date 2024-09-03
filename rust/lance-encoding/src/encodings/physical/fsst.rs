// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{ops::Range, sync::Arc};

use arrow_array::{cast::AsArray, Array, BinaryArray};
use arrow_buffer::{Buffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::DataType;
use arrow_select::concat::concat;
use futures::{future::BoxFuture, FutureExt};

use lance_core::Result;

use crate::{
    buffer::LanceBuffer,
    data::{DataBlock, NullableDataBlock, VariableWidthBlock},
    decoder::{PageScheduler, PrimitivePageDecoder},
    encoder::{ArrayEncoder, EncodedArray},
    format::pb,
    EncodingsIo,
};

#[derive(Debug)]
pub struct FsstPageScheduler {
    inner_scheduler: Box<dyn PageScheduler>,
    symbol_table: Vec<u8>,
}

impl FsstPageScheduler {
    pub fn new(inner_scheduler: Box<dyn PageScheduler>, symbol_table: Vec<u8>) -> Self {
        Self {
            inner_scheduler,
            symbol_table,
        }
    }
}

impl PageScheduler for FsstPageScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[Range<u64>],
        scheduler: &Arc<dyn EncodingsIo>,
        top_level_row: u64,
    ) -> BoxFuture<'static, Result<Box<dyn PrimitivePageDecoder>>> {
        let inner_decoder = self
            .inner_scheduler
            .schedule_ranges(ranges, scheduler, top_level_row);
        let symbol_table = self.symbol_table.clone();

        async move {
            let inner_decoder = inner_decoder.await?;
            Ok(Box::new(FsstPageDecoder {
                inner_decoder,
                symbol_table,
            }) as Box<dyn PrimitivePageDecoder>)
        }
        .boxed()
    }
}

struct FsstPageDecoder {
    inner_decoder: Box<dyn PrimitivePageDecoder>,
    symbol_table: Vec<u8>,
}

impl PrimitivePageDecoder for FsstPageDecoder {
    fn decode(&self, rows_to_skip: u64, num_rows: u64) -> Result<DataBlock> {
        let compressed_data = self.inner_decoder.decode(rows_to_skip, num_rows)?;
        let (string_data, nulls) = match compressed_data {
            DataBlock::Nullable(nullable) => {
                let data = nullable.data.as_variable_width()?;
                Result::Ok((data, Some(nullable.nulls)))
            }
            DataBlock::VariableWidth(variable) => Ok((variable, None)),
            _ => panic!("Received non-variable width data from inner decoder"),
        }?;

        let offsets = ScalarBuffer::<i32>::from(string_data.offsets.into_buffer());
        let bytes = string_data.data.into_buffer();

        let mut decompressed_offsets = vec![0_i32; offsets.len()];
        let mut decompressed_bytes = vec![0_u8; bytes.len() * 8];
        // Safety: Exposes uninitialized memory but we're about to clobber it
        unsafe {
            decompressed_bytes.set_len(decompressed_bytes.capacity());
        }
        fsst::fsst::decompress(
            &self.symbol_table,
            &bytes,
            &offsets,
            &mut decompressed_bytes,
            &mut decompressed_offsets,
        )?;

        // TODO: Change PrimitivePageDecoder to use Vec instead of BytesMut
        // since there is no way to get BytesMut from Vec but these copies should be avoidable
        // This is not the first time this has happened
        let mut offsets_as_bytes_mut = Vec::with_capacity(decompressed_offsets.len());
        let decompressed_offsets = ScalarBuffer::<i32>::from(decompressed_offsets);
        offsets_as_bytes_mut.extend_from_slice(decompressed_offsets.inner().as_slice());

        let mut bytes_as_bytes_mut = Vec::with_capacity(decompressed_bytes.len());
        bytes_as_bytes_mut.extend_from_slice(&decompressed_bytes);

        let new_string_data = DataBlock::VariableWidth(VariableWidthBlock {
            bits_per_offset: 32,
            data: LanceBuffer::from(bytes_as_bytes_mut),
            num_values: num_rows,
            offsets: LanceBuffer::from(offsets_as_bytes_mut),
        });

        if let Some(nulls) = nulls {
            Ok(DataBlock::Nullable(NullableDataBlock {
                data: Box::new(new_string_data),
                nulls,
            }))
        } else {
            Ok(new_string_data)
        }
    }
}

#[derive(Debug)]
pub struct FsstArrayEncoder {
    inner_encoder: Box<dyn ArrayEncoder>,
}

impl FsstArrayEncoder {
    pub fn new(inner_encoder: Box<dyn ArrayEncoder>) -> Self {
        Self { inner_encoder }
    }
}

impl ArrayEncoder for FsstArrayEncoder {
    fn encode(
        &self,
        arrays: &[arrow_array::ArrayRef],
        buffer_index: &mut u32,
    ) -> lance_core::Result<crate::encoder::EncodedArray> {
        // Currently, fsst encoder expects one buffer, so let us concatenate
        let concat_array = concat(&arrays.iter().map(|arr| arr.as_ref()).collect::<Vec<_>>())?;
        let (offsets, values) = match concat_array.data_type() {
            DataType::Utf8 => {
                let str_array = concat_array.as_string();
                (str_array.offsets().inner(), str_array.values())
            }
            DataType::Binary => {
                let bin_array = concat_array.as_binary();
                (bin_array.offsets().inner(), bin_array.values())
            }
            _ => panic!("Received neither Utf8 nor Binary array from inner encoder"),
        };

        let mut dest_offsets = vec![0_i32; offsets.len() * 2];
        let mut dest_values = vec![0_u8; values.len() * 2];
        let mut symbol_table = vec![0_u8; fsst::fsst::FSST_SYMBOL_TABLE_SIZE];

        fsst::fsst::compress(
            &mut symbol_table,
            values.as_slice(),
            offsets,
            &mut dest_values,
            &mut dest_offsets,
        )?;

        let dest_array = Arc::new(BinaryArray::new(
            OffsetBuffer::new(ScalarBuffer::from(dest_offsets)),
            Buffer::from(dest_values),
            concat_array.nulls().cloned(),
        ));

        let inner_encoded = self.inner_encoder.encode(&[dest_array], buffer_index)?;
        Ok(EncodedArray {
            buffers: inner_encoded.buffers,
            encoding: pb::ArrayEncoding {
                array_encoding: Some(pb::array_encoding::ArrayEncoding::Fsst(Box::new(
                    pb::Fsst {
                        binary: Some(Box::new(inner_encoded.encoding)),
                        symbol_table,
                    },
                ))),
            },
        })
    }
}

#[cfg(test)]
mod tests {

    use std::collections::HashMap;

    use lance_datagen::{ByteCount, RowCount};

    use crate::testing::{check_round_trip_encoding_of_data, TestCases};

    #[test_log::test(tokio::test)]
    async fn test_fsst() {
        let arr = lance_datagen::gen()
            .anon_col(lance_datagen::array::rand_utf8(ByteCount::from(32), false))
            .into_batch_rows(RowCount::from(1_000_000))
            .unwrap()
            .column(0)
            .clone();
        check_round_trip_encoding_of_data(vec![arr], &TestCases::default(), HashMap::new()).await;
    }
}
