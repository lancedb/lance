// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_schema::DataType;
use futures::{future::BoxFuture, FutureExt};
use lance_core::Result;
use log::trace;

use crate::{
    data::{DataBlock, FixedSizeListBlock},
    decoder::{PageScheduler, PrimitivePageDecoder},
    encoder::{ArrayEncoder, EncodedArray},
    format::ProtobufUtils,
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
    fn decode(&self, rows_to_skip: u64, num_rows: u64) -> Result<DataBlock> {
        let rows_to_skip = rows_to_skip * self.dimension;
        let num_child_rows = num_rows * self.dimension;
        let child_data = self.items_decoder.decode(rows_to_skip, num_child_rows)?;
        Ok(DataBlock::FixedSizeList(FixedSizeListBlock {
            child: Box::new(child_data),
            dimension: self.dimension,
        }))
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
    fn encode(
        &self,
        data: DataBlock,
        data_type: &DataType,
        buffer_index: &mut u32,
    ) -> Result<EncodedArray> {
        let inner_type = match data_type {
            DataType::FixedSizeList(inner_field, _) => inner_field.data_type().clone(),
            _ => panic!("Expected fixed size list data type and got {}", data_type),
        };
        let data = data.as_fixed_size_list().unwrap();
        let child = *data.child;

        let encoded_data = self
            .items_encoder
            .encode(child, &inner_type, buffer_index)?;

        let data = DataBlock::FixedSizeList(FixedSizeListBlock {
            child: Box::new(encoded_data.data),
            dimension: self.dimension as u64,
        });

        let encoding = ProtobufUtils::fixed_size_list(encoded_data.encoding, self.dimension);
        Ok(EncodedArray { data, encoding })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field};

    use crate::{testing::check_round_trip_encoding_random, version::LanceFileVersion};

    const PRIMITIVE_TYPES: &[DataType] = &[DataType::Int8, DataType::Float32, DataType::Float64];

    #[test_log::test(tokio::test)]
    async fn test_value_fsl_primitive() {
        for data_type in PRIMITIVE_TYPES {
            let inner_field = Field::new("item", data_type.clone(), true);
            let data_type = DataType::FixedSizeList(Arc::new(inner_field), 16);
            let field = Field::new("", data_type, false);
            check_round_trip_encoding_random(field, LanceFileVersion::V2_0).await;
        }
    }
}
