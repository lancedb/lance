// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{cast::AsArray, ArrayRef};
use bytes::BytesMut;
use futures::{future::BoxFuture, FutureExt};
use lance_core::Result;
use log::trace;

use crate::{
    decoder::{PageScheduler, PrimitivePageDecoder},
    encoder::{ArrayEncoder, EncodedArray},
    format::pb,
    EncodingsIo,
};

/// A scheduler for fixed size lists of primitive values
///
/// This scheduler is, itself, primitive
#[derive(Debug)]
pub struct FixedListScheduler {
    items_scheduler: Box<dyn PageScheduler>,
    dimension: u32,
}

impl FixedListScheduler {
    pub fn new(items_scheduler: Box<dyn PageScheduler>, dimension: u32) -> Self {
        Self {
            items_scheduler,
            dimension,
        }
    }
}

impl PageScheduler for FixedListScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u64>],
        scheduler: &Arc<dyn EncodingsIo>,
        top_level_row: u64,
    ) -> BoxFuture<'static, Result<Box<dyn PrimitivePageDecoder>>> {
        let expanded_ranges = ranges
            .iter()
            .map(|range| (range.start * self.dimension as u64)..(range.end * self.dimension as u64))
            .collect::<Vec<_>>();
        trace!(
            "Expanding {} fsl ranges across {}..{} to item ranges across {}..{}",
            ranges.len(),
            ranges[0].start,
            ranges[ranges.len() - 1].end,
            expanded_ranges[0].start,
            expanded_ranges[expanded_ranges.len() - 1].end
        );
        let inner_page_decoder =
            self.items_scheduler
                .schedule_ranges(&expanded_ranges, scheduler, top_level_row);
        let dimension = self.dimension;
        async move {
            let items_decoder = inner_page_decoder.await?;
            Ok(Box::new(FixedListDecoder {
                items_decoder,
                dimension: dimension as u64,
            }) as Box<dyn PrimitivePageDecoder>)
        }
        .boxed()
    }
}

pub struct FixedListDecoder {
    items_decoder: Box<dyn PrimitivePageDecoder>,
    dimension: u64,
}

impl PrimitivePageDecoder for FixedListDecoder {
    fn decode(
        &self,
        rows_to_skip: u64,
        num_rows: u64,
        all_null: &mut bool,
    ) -> Result<Vec<BytesMut>> {
        let rows_to_skip = rows_to_skip * self.dimension;
        let num_rows = num_rows * self.dimension;
        self.items_decoder.decode(rows_to_skip, num_rows, all_null)
    }

    fn num_buffers(&self) -> u32 {
        self.items_decoder.num_buffers()
    }
}

#[derive(Debug)]
pub struct FslEncoder {
    items_encoder: Box<dyn ArrayEncoder>,
    dimension: u32,
}

impl FslEncoder {
    pub fn new(items_encoder: Box<dyn ArrayEncoder>, dimension: u32) -> Self {
        Self {
            items_encoder,
            dimension,
        }
    }
}

impl ArrayEncoder for FslEncoder {
    fn encode(&self, arrays: &[ArrayRef], buffer_index: &mut u32) -> Result<EncodedArray> {
        let inner_arrays = arrays
            .iter()
            .map(|arr| arr.as_fixed_size_list().values().clone())
            .collect::<Vec<_>>();
        let items_page = self.items_encoder.encode(&inner_arrays, buffer_index)?;
        Ok(EncodedArray {
            buffers: items_page.buffers,
            encoding: pb::ArrayEncoding {
                array_encoding: Some(pb::array_encoding::ArrayEncoding::FixedSizeList(Box::new(
                    pb::FixedSizeList {
                        dimension: self.dimension,
                        items: Some(Box::new(items_page.encoding)),
                    },
                ))),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field};

    use crate::testing::check_round_trip_encoding_random;

    const PRIMITIVE_TYPES: &[DataType] = &[DataType::Int8, DataType::Float32, DataType::Float64];

    #[test_log::test(tokio::test)]
    async fn test_value_fsl_primitive() {
        for data_type in PRIMITIVE_TYPES {
            let inner_field = Field::new("item", data_type.clone(), true);
            let data_type = DataType::FixedSizeList(Arc::new(inner_field), 16);
            let field = Field::new("", data_type, false);
            check_round_trip_encoding_random(field).await;
        }
    }
}
