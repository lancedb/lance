use std::{ops::Range, sync::Arc};

use arrow_array::{cast::AsArray, Array, BinaryArray};
use arrow_buffer::{Buffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::DataType;
use arrow_select::concat::concat;
use bytes::BytesMut;
use futures::{future::BoxFuture, FutureExt};

use lance_core::Result;

use crate::{
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
    fn decode(
        &self,
        rows_to_skip: u64,
        num_rows: u64,
        all_null: &mut bool,
    ) -> Result<Vec<BytesMut>> {
        let buffers = self
            .inner_decoder
            .decode(rows_to_skip, num_rows, all_null)?;

        let mut buffers_iter = buffers.into_iter();

        // Buffer order expected from inner binary decoder
        let validity = buffers_iter.next().unwrap();
        let offsets = buffers_iter.next().unwrap();
        let dummy = buffers_iter.next().unwrap();
        let bytes = buffers_iter.next().unwrap();

        // Reinterpret offsets as i32
        let offsets = ScalarBuffer::<i32>::new(
            Buffer::from_bytes(offsets.freeze().into()),
            0,
            num_rows as usize + 1,
        );

        // Need to adjust offsets to account for symbol table
        // TODO: Don't do this with a copy
        let mut compressed_offsets = Vec::with_capacity(offsets.len());
        compressed_offsets.extend(
            offsets
                .iter()
                .map(|val| *val + self.symbol_table.len() as i32),
        );

        // Need to insert symbol table back in front of compressed bytes
        let mut compressed_bytes = Vec::with_capacity(self.symbol_table.len() + bytes.len());
        compressed_bytes.extend_from_slice(&self.symbol_table);
        compressed_bytes.extend_from_slice(&bytes);

        let mut decompressed_offsets = vec![0_i32; compressed_offsets.len()];
        let mut decompressed_bytes = vec![0_u8; compressed_bytes.len() * 3];
        // Safety: Exposes uninitialized memory but we're about to clobber it
        unsafe {
            decompressed_bytes.set_len(decompressed_bytes.capacity());
        }

        fsst::fsst::decompress(
            &compressed_bytes,
            &compressed_offsets,
            &mut decompressed_bytes,
            &mut decompressed_offsets,
        )?;

        // TODO: Change PrimitivePageDecoder to use Vec instead of BytesMut
        // since there is no way to get BytesMut from Vec but these copies should be avoidable
        // This is not the first time this has happened
        let mut offsets_as_bytes_mut = BytesMut::with_capacity(decompressed_offsets.len());
        let decompressed_offsets = ScalarBuffer::<i32>::from(decompressed_offsets);
        offsets_as_bytes_mut.extend_from_slice(decompressed_offsets.inner().as_slice());

        let mut bytes_as_bytes_mut = BytesMut::with_capacity(decompressed_bytes.len());
        bytes_as_bytes_mut.extend_from_slice(&decompressed_bytes);

        Ok(vec![
            validity,
            offsets_as_bytes_mut,
            dummy,
            bytes_as_bytes_mut,
        ])
    }

    fn num_buffers(&self) -> u32 {
        self.inner_decoder.num_buffers()
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

        fsst::fsst::compress(
            values.as_slice(),
            &offsets,
            &mut dest_values,
            &mut dest_offsets,
        )?;

        let symbol_table_num_bytes = dest_offsets[0] as usize;
        let symbol_table = dest_values[0..symbol_table_num_bytes].to_vec();

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

    use arrow_schema::DataType;
    use lance_datagen::{ByteCount, RowCount};

    use crate::{
        encoder::ArrayEncoder,
        encodings::physical::{
            basic::BasicEncoder,
            binary::BinaryEncoder,
            value::{CompressionScheme, ValueEncoder},
        },
    };

    use super::FsstArrayEncoder;

    #[test]
    fn test_encode() {
        let indices_encoder = Box::new(BasicEncoder::new(Box::new(
            ValueEncoder::try_new(&DataType::UInt64, CompressionScheme::None).unwrap(),
        )));
        let bytes_encoder = Box::new(BasicEncoder::new(Box::new(
            ValueEncoder::try_new(&DataType::UInt8, CompressionScheme::None).unwrap(),
        )));
        let string_encoder = Box::new(BinaryEncoder::new(indices_encoder, bytes_encoder));
        let encoder = FsstArrayEncoder::new(string_encoder);

        let arr = lance_datagen::gen()
            .anon_col(lance_datagen::array::rand_utf8(ByteCount::from(16), false))
            .into_batch_rows(RowCount::from(10000))
            .unwrap()
            .column(0)
            .clone();
        let mut buf_index = 0;

        let encoded = encoder.encode(&[arr], &mut buf_index).unwrap();

        dbg!(encoded);
    }
}
