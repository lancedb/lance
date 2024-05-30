// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, UInt32Array};
use arrow_buffer::Buffer;
use arrow_schema::{Field, Schema};
use datafusion_common::{arrow::datatypes::DataType, ScalarValue};
use datafusion_expr::Accumulator;
use datafusion_physical_expr::expressions::{MaxAccumulator, MinAccumulator};
use futures::{future::BoxFuture, FutureExt};
use lance_encoding::{
    decoder::{FieldScheduler, SchedulingJob},
    encoder::{
        encode_batch, CoreFieldEncodingStrategy, EncodedBuffer, EncodedColumn, FieldEncoder,
    },
    format::pb,
};

use lance_core::{Error, Result};
use lance_file::v2::writer::EncodedBatchWriteExt;
use snafu::{location, Location};

#[derive(Debug)]
struct CreatedZoneMap {
    min: ScalarValue,
    max: ScalarValue,
    null_count: u32,
}

#[derive(Debug)]
pub struct ZoneMapsFieldScheduler {
    inner: Arc<dyn FieldScheduler>,
    zone_maps: Vec<CreatedZoneMap>,
    rows_per_zone: u32,
}

impl FieldScheduler for ZoneMapsFieldScheduler {
    fn schedule_ranges<'a>(
        &'a self,
        ranges: &[std::ops::Range<u64>],
    ) -> Result<Box<dyn SchedulingJob + 'a>> {
        self.inner.schedule_ranges(ranges)
    }

    fn num_rows(&self) -> u64 {
        self.inner.num_rows()
    }
}

/// A field encoder that creates zone maps for the data it encodes
///
/// This encoder will create zone maps for the data it encodes.  The zone maps are created by
/// dividing the data into zones of a fixed size and calculating the min/max values for each
/// zone.  The zone maps are then encoded as metadata.
///
/// This metadata can be used by the reader to skip over zones that don't contain data that
/// matches the query.
pub struct ZoneMapsFieldEncoder {
    items_encoder: Box<dyn FieldEncoder>,
    items_type: DataType,

    rows_per_map: u32,

    maps: Vec<CreatedZoneMap>,
    cur_offset: u32,
    min: MinAccumulator,
    max: MaxAccumulator,
    null_count: u32,
}

impl ZoneMapsFieldEncoder {
    pub fn try_new(
        items_encoder: Box<dyn FieldEncoder>,
        items_type: DataType,
        rows_per_map: u32,
    ) -> Result<Self> {
        let min = MinAccumulator::try_new(&items_type)?;
        let max = MaxAccumulator::try_new(&items_type)?;
        Ok(Self {
            rows_per_map,
            items_encoder,
            items_type,
            min,
            max,
            null_count: 0,
            cur_offset: 0,
            maps: Vec::new(),
        })
    }
}

impl ZoneMapsFieldEncoder {
    fn new_map(&mut self) -> Result<()> {
        // TODO: We should be truncating the min/max values here
        let map = CreatedZoneMap {
            min: self.min.evaluate()?,
            max: self.max.evaluate()?,
            null_count: self.null_count,
        };
        self.maps.push(map);
        self.min = MinAccumulator::try_new(&self.items_type)?;
        self.max = MaxAccumulator::try_new(&self.items_type)?;
        self.null_count = 0;
        self.cur_offset = 0;
        Ok(())
    }

    fn update_stats(&mut self, array: &ArrayRef) -> Result<()> {
        self.null_count += array.null_count() as u32;
        self.min.update_batch(&[array.clone()])?;
        self.max.update_batch(&[array.clone()])?;
        Ok(())
    }

    fn update(&mut self, array: &ArrayRef) -> Result<()> {
        let mut remaining = array.len() as u32;
        let mut offset = 0;

        while remaining > 0 {
            let desired = self.rows_per_map - self.cur_offset;
            if desired > remaining {
                // Not enough data to fill a map, increment counts and return
                self.update_stats(&array.slice(offset, remaining as usize))?;
                self.cur_offset += remaining;
                break;
            } else {
                // We have enough data to fill a map
                self.update_stats(&array.slice(offset, desired as usize))?;
                self.new_map()?;
            }
            offset += desired as usize;
            remaining = remaining.saturating_sub(desired);
        }
        Ok(())
    }

    async fn maps_to_metadata(&mut self) -> Result<EncodedBuffer> {
        let maps = std::mem::take(&mut self.maps);
        let (mins, (maxes, null_counts)): (Vec<_>, (Vec<_>, Vec<_>)) = maps
            .into_iter()
            .map(|mp| (mp.min, (mp.max, mp.null_count)))
            .unzip();
        let mins = ScalarValue::iter_to_array(mins.into_iter())?;
        let maxes = ScalarValue::iter_to_array(maxes.into_iter())?;
        let null_counts = Arc::new(UInt32Array::from_iter_values(null_counts.into_iter()));
        let zone_map_schema = Arc::new(Schema::new(vec![
            Field::new("min", mins.data_type().clone(), true),
            Field::new("max", maxes.data_type().clone(), true),
            Field::new("null_count", DataType::UInt32, false),
        ]));
        let zone_maps = RecordBatch::try_new(zone_map_schema, vec![mins, maxes, null_counts])?;
        let encoding_strategy = CoreFieldEncodingStrategy::default();
        let encoded_zone_maps = encode_batch(&zone_maps, &encoding_strategy, u64::MAX).await?;
        let zone_maps_buffer = encoded_zone_maps.try_to_mini_lance()?;
        Ok(EncodedBuffer {
            parts: vec![Buffer::from(zone_maps_buffer)],
        })
    }
}

impl FieldEncoder for ZoneMapsFieldEncoder {
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
    ) -> Result<Vec<lance_encoding::encoder::EncodeTask>> {
        // TODO: If we do the zone map calculation as part of the encoding task then we can
        // parallelize statistics gathering.  Could be faster too since the encoding task is
        // going to need to access the same data (although the input to an encoding task is
        // probably too big for the CPU cache anyways).  We can worry about this if we need
        // to improve write speed.
        self.update(&array)?;
        self.items_encoder.maybe_encode(array)
    }

    fn flush(&mut self) -> Result<Vec<lance_encoding::encoder::EncodeTask>> {
        if self.cur_offset > 0 {
            // Create final map
            self.new_map()?;
        }
        self.items_encoder.flush()
    }

    fn finish(&mut self) -> BoxFuture<'_, Result<Vec<EncodedColumn>>> {
        async move {
            let items_columns = self.items_encoder.finish().await?;
            if items_columns.is_empty() {
                return Err(Error::invalid_input("attempt to apply zone maps to a field encoder that generated zero columns of data".to_string(), location!()))
            }
            let items_column = items_columns.into_iter().next().unwrap();
            let final_pages = items_column.final_pages;
            let mut column_buffers = items_column.column_buffers;
            let zone_buffer_index = column_buffers.len();
            column_buffers.push(self.maps_to_metadata().await?);
            let column_encoding = pb::ColumnEncoding {
                column_encoding: Some(pb::column_encoding::ColumnEncoding::ZoneIndex(Box::new(
                    pb::ZoneIndex {
                        inner: Some(Box::new(items_column.encoding)),
                        rows_per_zone: self.rows_per_map,
                        zone_map_buffer: Some(pb::Buffer {
                            buffer_index: zone_buffer_index as u32,
                            buffer_type: i32::from(pb::buffer::BufferType::Column),
                        }),
                    },
                ))),
            };
            Ok(vec![EncodedColumn {
                encoding: column_encoding,
                final_pages,
                column_buffers,
            }])
        }
        .boxed()
    }

    fn num_columns(&self) -> u32 {
        self.items_encoder.num_columns()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use arrow_array::types::Int32Type;
    use arrow_schema::DataType;
    use lance_datagen::{BatchCount, RowCount};
    use lance_encoding::encoder::{
        ColumnIndexSequence, CoreFieldEncodingStrategy, FieldEncoder, FieldEncodingStrategy,
    };

    #[tokio::test]
    async fn test_basic_stats() {
        let encoding_strategy = CoreFieldEncodingStrategy::default();
        let mut col_idx_seq = ColumnIndexSequence::default();
        let mock_field = lance_core::datatypes::Field::try_from(arrow_schema::Field::new(
            "foo",
            DataType::Int32,
            false,
        ))
        .unwrap();
        let inner = encoding_strategy
            .create_field_encoder(
                &encoding_strategy,
                &mock_field,
                &mut col_idx_seq,
                4096,
                true,
                &HashMap::new(),
            )
            .unwrap();
        let mut encoder =
            super::ZoneMapsFieldEncoder::try_new(inner, DataType::Int32, 100).unwrap();

        let gen = lance_datagen::gen()
            .anon_col(lance_datagen::array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(1024), BatchCount::from(7));

        for batch in gen {
            let batch = batch.unwrap();
            let array = batch.column(0);
            encoder.maybe_encode(array.clone()).unwrap();
        }

        encoder.finish().await.unwrap();
    }
}
