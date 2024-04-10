// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{cast::AsArray, ArrayRef};
use futures::{future::BoxFuture, FutureExt};
use lance_core::Result;
use log::trace;

use crate::{
    decoder::{PhysicalPageDecoder, PhysicalPageScheduler},
    encoder::{ArrayEncoder, EncodedPage},
    format::pb,
    EncodingsIo,
};

/// A scheduler for fixed size lists of primitive values
///
/// This scheduler is, itself, primitive
#[derive(Debug)]
pub struct FixedListScheduler {
    items_scheduler: Box<dyn PhysicalPageScheduler>,
    dimension: u32,
}

impl FixedListScheduler {
    pub fn new(items_scheduler: Box<dyn PhysicalPageScheduler>, dimension: u32) -> Self {
        Self {
            items_scheduler,
            dimension,
        }
    }
}

impl PhysicalPageScheduler for FixedListScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u32>],
        scheduler: &dyn EncodingsIo,
    ) -> BoxFuture<'static, Result<Box<dyn PhysicalPageDecoder>>> {
        let expanded_ranges = ranges
            .iter()
            .map(|range| (range.start * self.dimension)..(range.end * self.dimension))
            .collect::<Vec<_>>();
        trace!(
            "Expanding {} fsl ranges across {}..{} to item ranges across {}..{}",
            ranges.len(),
            ranges[0].start,
            ranges[ranges.len() - 1].end,
            expanded_ranges[0].start,
            expanded_ranges[expanded_ranges.len() - 1].end
        );
        let inner_page_decoder = self
            .items_scheduler
            .schedule_ranges(&expanded_ranges, scheduler);
        let dimension = self.dimension;
        async move {
            let items_decoder = inner_page_decoder.await?;
            Ok(Box::new(FixedListDecoder {
                items_decoder,
                dimension,
            }) as Box<dyn PhysicalPageDecoder>)
        }
        .boxed()
    }
}

pub struct FixedListDecoder {
    items_decoder: Box<dyn PhysicalPageDecoder>,
    dimension: u32,
}

impl PhysicalPageDecoder for FixedListDecoder {
    fn update_capacity(&self, rows_to_skip: u32, num_rows: u32, buffers: &mut [(u64, bool)]) {
        let rows_to_skip = rows_to_skip * self.dimension;
        let num_rows = num_rows * self.dimension;
        self.items_decoder
            .update_capacity(rows_to_skip, num_rows, buffers);
    }

    fn decode_into(&self, rows_to_skip: u32, num_rows: u32, dest_buffers: &mut [bytes::BytesMut]) {
        let rows_to_skip = rows_to_skip * self.dimension;
        let num_rows = num_rows * self.dimension;
        self.items_decoder
            .decode_into(rows_to_skip, num_rows, dest_buffers);
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
    fn encode(&self, arrays: &[ArrayRef]) -> Result<EncodedPage> {
        let inner_arrays = arrays
            .iter()
            .map(|arr| arr.as_fixed_size_list().values().clone())
            .collect::<Vec<_>>();
        let items_page = self.items_encoder.encode(&inner_arrays)?;
        Ok(EncodedPage {
            buffers: items_page.buffers,
            num_rows: items_page.num_rows / self.dimension,
            encoding: pb::ArrayEncoding {
                array_encoding: Some(pb::array_encoding::ArrayEncoding::FixedSizeList(Box::new(
                    pb::FixedSizeList {
                        dimension: self.dimension,
                        items: Some(Box::new(items_page.encoding)),
                    },
                ))),
            },
            column_idx: items_page.column_idx,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field};

    use crate::encodings::physical::basic::BasicEncoder;
    use crate::encodings::physical::fixed_size_list::FslEncoder;
    use crate::encodings::physical::value::tests::PRIMITIVE_TYPES;
    use crate::testing::check_round_trip_array_encoding;

    #[test_log::test(tokio::test)]
    async fn test_value_fsl_primitive() {
        for data_type in PRIMITIVE_TYPES {
            let encoder = FslEncoder::new(Box::new(BasicEncoder::new(0)), 16);
            let inner_field = Field::new("item", data_type.clone(), true);
            let field = Field::new(
                "",
                DataType::FixedSizeList(Arc::new(inner_field), 16),
                false,
            );
            check_round_trip_array_encoding(encoder, field).await;
        }
    }
}
