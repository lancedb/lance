// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::{HashMap, VecDeque},
    ops::Range,
    sync::{Arc, Mutex},
};

use arrow_array::{cast::AsArray, types::UInt32Type, ArrayRef, RecordBatch, UInt32Array};
use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
use bytes::Bytes;
use datafusion::functions_aggregate::min_max::{MaxAccumulator, MinAccumulator};
use datafusion_common::{arrow::datatypes::DataType, DFSchema, DFSchemaRef, ScalarValue};
use datafusion_expr::{
    col,
    execution_props::ExecutionProps,
    interval_arithmetic::{Interval, NullableInterval},
    simplify::SimplifyContext,
    Accumulator, Expr,
};
use datafusion_functions::core::expr_ext::FieldAccessor;
use datafusion_optimizer::simplify_expressions::ExprSimplifier;
use futures::{future::BoxFuture, FutureExt};
use lance_datafusion::planner::Planner;
use lance_encoding::{
    buffer::LanceBuffer,
    decoder::{
        decode_batch, ColumnInfoIter, DecoderPlugins, FieldScheduler, FilterExpression,
        PriorityRange, ScheduledScanLine, SchedulerContext, SchedulingJob,
    },
    encoder::{
        encode_batch, CoreFieldEncodingStrategy, EncodedBatch, EncodedColumn, EncodingOptions,
        FieldEncoder, OutOfLineBuffers,
    },
    format::pb,
    repdef::RepDefBuilder,
    EncodingsIo,
};

use lance_core::{cache::FileMetadataCache, datatypes::Schema, Error, Result};
use lance_file::{
    v2::{reader::EncodedBatchReaderExt, writer::EncodedBatchWriteExt},
    version::LanceFileVersion,
};
use snafu::{location, Location};

use crate::substrait::FilterExpressionExt;

#[derive(Debug)]
struct CreatedZoneMap {
    min: ScalarValue,
    max: ScalarValue,
    null_count: u32,
}

/// Builds up a vector of ranges from a series of sorted ranges that
/// may be adjacent (in which case we merge them) or disjoint (in
/// which case we create separate ranges).
#[derive(Default)]
struct RangesBuilder {
    ranges: Vec<Range<u64>>,
}

impl RangesBuilder {
    fn add_range(&mut self, range: Range<u64>) {
        if let Some(cur) = self.ranges.last_mut() {
            if cur.end == range.start {
                cur.end = range.end;
            } else {
                self.ranges.push(range);
            }
        } else {
            self.ranges.push(range);
        }
    }
}

struct ZoneMapsFilter<F: Fn(u64) -> bool> {
    filter: F,
    rows_per_zone: u64,
}

impl<F: Fn(u64) -> bool> ZoneMapsFilter<F> {
    fn new(filter: F, rows_per_zone: u64) -> Self {
        Self {
            filter,
            rows_per_zone,
        }
    }

    /// Given a requested range, and a filter telling us which zones
    /// could possibly include matching data, generate a smaller range
    /// (or ranges) that only include matching zones.
    fn refine_range(&self, mut range: std::ops::Range<u64>) -> Vec<std::ops::Range<u64>> {
        let mut ranges_builder = RangesBuilder::default();
        let mut zone_idx = range.start / self.rows_per_zone;
        while !range.is_empty() {
            let end = range.end.min((zone_idx + 1) * self.rows_per_zone);

            if (self.filter)(zone_idx) {
                let zone_range = range.start..end;
                ranges_builder.add_range(zone_range);
            }

            range.start = end;
            zone_idx += 1;
        }
        ranges_builder.ranges
    }

    fn refine_ranges(&self, ranges: &[Range<u64>]) -> Vec<Range<u64>> {
        ranges
            .iter()
            .flat_map(|r| self.refine_range(r.clone()))
            .collect()
    }
}

/// Substrait represents paths as a series of field indices
///
/// This method converts that into a datafusion expression
#[allow(unused)]
fn path_to_expr(path: &VecDeque<u32>) -> Expr {
    let mut parts_iter = path.iter().map(|path_num| path_num.to_string());
    let mut expr = col(parts_iter.next().unwrap());
    for part in parts_iter {
        expr = expr.field(part);
    }
    expr
}

/// If a column has zone info in the encoding description then extract it
#[allow(unused)]
pub(crate) fn extract_zone_info(
    column_info: &mut ColumnInfoIter,
    data_type: &DataType,
    cur_path: &VecDeque<u32>,
) -> Option<(u32, UnloadedPushdown)> {
    let mut result: Option<(u32, UnloadedPushdown)> = None;
    let result_ref = &mut result;
    column_info.peek_transform(|col_info| {
        let encoding = col_info.encoding.column_encoding.as_ref().unwrap();
        match *encoding {
            pb::column_encoding::ColumnEncoding::ZoneIndex(ref zone_index) => {
                let mut zone_index = zone_index.clone();
                let inner = zone_index.inner.take().unwrap();
                let rows_per_zone = zone_index.rows_per_zone;
                let zone_map_buffer = zone_index.zone_map_buffer.as_ref().unwrap().clone();
                assert_eq!(
                    zone_map_buffer.buffer_type,
                    i32::from(pb::buffer::BufferType::Column)
                );
                let (position, size) =
                    col_info.buffer_offsets_and_sizes[zone_map_buffer.buffer_index as usize];
                let column = path_to_expr(cur_path);
                let unloaded_pushdown = UnloadedPushdown {
                    data_type: data_type.clone(),
                    column,
                    position,
                    size,
                };
                *result_ref = Some((rows_per_zone, unloaded_pushdown));

                let mut col_info = col_info.as_ref().clone();
                col_info.encoding = *inner;
                Arc::new(col_info)
            }
            _ => col_info,
        }
    });
    result
}

/// Extracted pushdown information obtained from the column encoding
/// description.
///
/// This is "unloaded" because we haven't yet loaded the actual zone
/// maps from the file (though position and size tell us where they
/// are)
#[derive(Debug)]
pub struct UnloadedPushdown {
    data_type: DataType,
    column: Expr,
    position: u64,
    size: u64,
}

#[derive(Debug)]
struct ZoneMap {
    items: Vec<(Expr, NullableInterval)>,
}

#[derive(Debug)]
struct InitializedState {
    zone_maps: Vec<ZoneMap>,
    filter: Option<Expr>,
    df_schema: Option<DFSchemaRef>,
}

/// A top level scheduler that refines the requested range based on
/// pushdown filtering with zone maps
#[derive(Debug)]
pub struct ZoneMapsFieldScheduler {
    inner: Arc<dyn FieldScheduler>,
    schema: Arc<Schema>,
    // A map from field id to unloaded zone map for that field
    pushdown_buffers: HashMap<u32, UnloadedPushdown>,
    rows_per_zone: u32,
    num_rows: u64,
    initialized_state: Mutex<Option<InitializedState>>,
}

impl ZoneMapsFieldScheduler {
    pub fn new(
        inner: Arc<dyn FieldScheduler>,
        schema: Arc<Schema>,
        pushdown_buffers: HashMap<u32, UnloadedPushdown>,
        rows_per_zone: u32,
        num_rows: u64,
    ) -> Self {
        Self {
            inner,
            schema,
            pushdown_buffers,
            rows_per_zone,
            num_rows,
            // These are set during initialization
            initialized_state: Mutex::new(None),
        }
    }

    async fn load_pushdowns(
        &self,
        io: &dyn EncodingsIo,
        _cache: &FileMetadataCache,
        pushdowns: &[&UnloadedPushdown],
    ) -> Result<Vec<ZoneMap>> {
        // TODO: Use cache
        let ranges = pushdowns
            .iter()
            .map(|pushdown| pushdown.position..pushdown.position + pushdown.size)
            .collect();
        let buffers = io.submit_request(ranges, 0).await?;
        let mut maps = Vec::new();
        for (buffer, pushdown) in buffers.into_iter().zip(pushdowns.iter()) {
            // There's no point in running this in parallel since it's actually synchronous
            let map = self
                .parse_zone(buffer, &pushdown.data_type, &pushdown.column)
                .await?;
            maps.push(map);
        }
        // A this point each item in `maps` is a vector of guarantees for a single field
        // We need to transpose this so that each item is a vector of guarantees for a single zone
        let zone_maps = transpose2(maps)
            .into_iter()
            .map(|items| ZoneMap { items })
            .collect();
        Ok(zone_maps)
    }

    /// Load the zone maps from the file
    async fn load_maps(
        &self,
        io: &dyn EncodingsIo,
        cache: &FileMetadataCache,
        filter_schema: &Schema,
    ) -> Result<Vec<ZoneMap>> {
        let pushdowns_to_load = filter_schema
            .fields
            .iter()
            .filter_map(|field| {
                let field_id = field.id as u32;
                let unloaded = self.pushdown_buffers.get(&field_id)?;
                Some(unloaded)
            })
            .collect::<Vec<_>>();
        self.load_pushdowns(io, cache, &pushdowns_to_load).await
    }

    async fn do_initialize(
        &self,
        io: &dyn EncodingsIo,
        cache: &FileMetadataCache,
        filter: &FilterExpression,
    ) -> Result<()> {
        if filter.is_noop() {
            return Ok(());
        }

        let arrow_schema = ArrowSchema::from(self.schema.as_ref());
        let df_schema = DFSchema::try_from(arrow_schema.clone())?;
        let df_filter = filter.substrait_to_df(Arc::new(arrow_schema))?;

        let columns = Planner::column_names_in_expr(&df_filter);
        let referenced_schema = self.schema.project(&columns)?;

        let df_schema = Some(Arc::new(df_schema));
        let zone_maps = self.load_maps(io, cache, &referenced_schema).await?;
        let filter = Some(df_filter);

        let state = InitializedState {
            zone_maps,
            filter,
            df_schema,
        };
        let mut initialized_state = self.initialized_state.lock().unwrap();
        *initialized_state = Some(state);
        Ok(())
    }

    fn create_filter(&self) -> Result<impl Fn(u64) -> bool + '_> {
        Ok(move |zone_idx| {
            let state = self.initialized_state.lock().unwrap();
            let state = state.as_ref().unwrap();
            let zone_map = &state.zone_maps[zone_idx as usize];
            let props = ExecutionProps::new();
            let context =
                SimplifyContext::new(&props).with_schema(state.df_schema.as_ref().unwrap().clone());
            let mut simplifier = ExprSimplifier::new(context);
            simplifier = simplifier.with_guarantees(zone_map.items.clone());
            match simplifier.simplify(state.filter.as_ref().unwrap().clone()) {
                Ok(expr) => match expr {
                    // Predicate, given guarantees, is always false, we can skip the zone
                    Expr::Literal(ScalarValue::Boolean(Some(false))) => false,
                    // Predicate may be true, need to load the zone
                    _ => true,
                },
                Err(err) => {
                    // TODO: this logs on each iteration, but maybe should should
                    // only log once per call of this func?
                    log::debug!("Failed to simplify predicate: {}", err);
                    true
                }
            }
        })
    }

    /// Parse the statistics into a set of guarantees for each batch.
    fn extract_guarantees(
        stats: &RecordBatch,
        rows_per_zone: u32,
        num_rows: u64,
        data_type: &DataType,
        col: Expr,
    ) -> Vec<(Expr, NullableInterval)> {
        let min_values = stats.column(0);
        let max_values = stats.column(1);
        let null_counts = stats.column(2).as_primitive::<UInt32Type>();

        let mut guarantees = Vec::new();
        for zone_idx in 0..stats.num_rows() {
            let num_rows_in_zone = if zone_idx == stats.num_rows() - 1 {
                (num_rows % rows_per_zone as u64) as u32
            } else {
                rows_per_zone
            };
            let min_value = ScalarValue::try_from_array(&min_values, zone_idx).unwrap();
            let max_value = ScalarValue::try_from_array(&max_values, zone_idx).unwrap();
            let null_count = null_counts.values()[zone_idx];

            let values = Interval::try_new(min_value, max_value).unwrap();
            let interval = match (null_count, num_rows_in_zone) {
                (0, _) => NullableInterval::NotNull { values },
                (null_count, num_rows_in_zone) if null_count == num_rows_in_zone => {
                    NullableInterval::Null {
                        datatype: data_type.clone(),
                    }
                }
                _ => NullableInterval::MaybeNull { values },
            };
            guarantees.push((col.clone(), interval));
        }
        guarantees
    }

    async fn parse_zone(
        &self,
        buffer: Bytes,
        data_type: &DataType,
        col: &Expr,
    ) -> Result<Vec<(Expr, NullableInterval)>> {
        let zone_map_schema = Schema::try_from(&ArrowSchema::new(vec![
            ArrowField::new("min", data_type.clone(), true),
            ArrowField::new("max", data_type.clone(), true),
            ArrowField::new("null_count", DataType::UInt32, false),
        ]))
        .unwrap();
        let zone_maps_batch =
            EncodedBatch::try_from_mini_lance(buffer, &zone_map_schema, LanceFileVersion::V2_0)?;
        let zone_maps_batch = decode_batch(
            &zone_maps_batch,
            &FilterExpression::no_filter(),
            Arc::<DecoderPlugins>::default(),
            /*should_validate= */ false,
        )
        .await?;

        Ok(Self::extract_guarantees(
            &zone_maps_batch,
            self.rows_per_zone,
            self.num_rows,
            data_type,
            col.clone(),
        ))
    }
}

// Utility function to transpose Vec<Vec<...>> from Stack Overflow
// https://stackoverflow.com/questions/64498617/how-to-transpose-a-vector-of-vectors-in-rust
// Author: https://stackoverflow.com/users/1695172/netwave
fn transpose2<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}

// Schedulers don't always handle empty ranges well, so we need to provide a dummy job
#[derive(Debug)]
struct EmptySchedulingJob {}

impl SchedulingJob for EmptySchedulingJob {
    fn schedule_next(
        &mut self,
        _context: &mut SchedulerContext,
        _priority: &dyn PriorityRange,
    ) -> Result<ScheduledScanLine> {
        Ok(ScheduledScanLine {
            rows_scheduled: 0,
            decoders: vec![],
        })
    }

    fn num_rows(&self) -> u64 {
        0
    }
}

impl FieldScheduler for ZoneMapsFieldScheduler {
    fn initialize<'a>(
        &'a self,
        filter: &'a FilterExpression,
        context: &'a SchedulerContext,
    ) -> BoxFuture<'a, Result<()>> {
        async move {
            self.do_initialize(context.io().as_ref(), context.cache(), filter)
                .await
        }
        .boxed()
    }

    fn schedule_ranges<'a>(
        &'a self,
        ranges: &[std::ops::Range<u64>],
        filter: &FilterExpression,
    ) -> Result<Box<dyn SchedulingJob + 'a>> {
        if filter.is_noop() {
            return self.inner.schedule_ranges(ranges, filter);
        }
        let zone_filter_fn = self.create_filter()?;
        let zone_filter = ZoneMapsFilter::new(zone_filter_fn, self.rows_per_zone as u64);
        let ranges = zone_filter.refine_ranges(ranges);
        if ranges.is_empty() {
            Ok(Box::new(EmptySchedulingJob {}))
        } else {
            self.inner.schedule_ranges(&ranges, filter)
        }
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

    async fn maps_to_metadata(maps: Vec<CreatedZoneMap>) -> Result<LanceBuffer> {
        let (mins, (maxes, null_counts)): (Vec<_>, (Vec<_>, Vec<_>)) = maps
            .into_iter()
            .map(|mp| (mp.min, (mp.max, mp.null_count)))
            .unzip();
        let mins = ScalarValue::iter_to_array(mins.into_iter())?;
        let maxes = ScalarValue::iter_to_array(maxes.into_iter())?;
        let null_counts = Arc::new(UInt32Array::from_iter_values(null_counts.into_iter()));
        let zone_map_schema = ArrowSchema::new(vec![
            ArrowField::new("min", mins.data_type().clone(), true),
            ArrowField::new("max", maxes.data_type().clone(), true),
            ArrowField::new("null_count", DataType::UInt32, false),
        ]);
        let schema = Schema::try_from(&zone_map_schema)?;
        let zone_maps =
            RecordBatch::try_new(Arc::new(zone_map_schema), vec![mins, maxes, null_counts])?;
        let encoding_strategy = CoreFieldEncodingStrategy::default();
        let encoded_zone_maps = encode_batch(
            &zone_maps,
            Arc::new(schema),
            &encoding_strategy,
            &EncodingOptions {
                cache_bytes_per_column: u64::MAX,
                max_page_bytes: u64::MAX,
                keep_original_array: true,
            },
        )
        .await?;
        let zone_maps_buffer = encoded_zone_maps.try_to_mini_lance()?;

        Ok(LanceBuffer::from_bytes(zone_maps_buffer, 1))
    }
}

impl FieldEncoder for ZoneMapsFieldEncoder {
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
        external_buffers: &mut OutOfLineBuffers,
        repdef: RepDefBuilder,
        row_number: u64,
    ) -> Result<Vec<lance_encoding::encoder::EncodeTask>> {
        // TODO: If we do the zone map calculation as part of the encoding task then we can
        // parallelize statistics gathering.  Could be faster too since the encoding task is
        // going to need to access the same data (although the input to an encoding task is
        // probably too big for the CPU cache anyways).  We can worry about this if we need
        // to improve write speed.
        self.update(&array)?;
        self.items_encoder
            .maybe_encode(array, external_buffers, repdef, row_number)
    }

    fn flush(
        &mut self,
        external_buffers: &mut OutOfLineBuffers,
    ) -> Result<Vec<lance_encoding::encoder::EncodeTask>> {
        self.items_encoder.flush(external_buffers)
    }

    fn finish(
        &mut self,
        external_buffers: &mut OutOfLineBuffers,
    ) -> BoxFuture<'_, Result<Vec<EncodedColumn>>> {
        if self.cur_offset > 0 {
            // Create final map
            if let Err(err) = self.new_map() {
                return async move { Err(err) }.boxed();
            }
        }
        let maps = std::mem::take(&mut self.maps);
        let rows_per_zone = self.rows_per_map;
        let items_columns = self.items_encoder.finish(external_buffers);

        async move {
            let items_columns = items_columns.await?;
            if items_columns.is_empty() {
                return Err(Error::invalid_input("attempt to apply zone maps to a field encoder that generated zero columns of data".to_string(), location!()))
            }
            let items_column = items_columns.into_iter().next().unwrap();
            let final_pages = items_column.final_pages;
            let mut column_buffers = items_column.column_buffers;
            let zone_buffer_index = column_buffers.len();
            column_buffers.push(Self::maps_to_metadata(maps).await?);
            let column_encoding = pb::ColumnEncoding {
                column_encoding: Some(pb::column_encoding::ColumnEncoding::ZoneIndex(Box::new(
                    pb::ZoneIndex {
                        inner: Some(Box::new(items_column.encoding)),
                        rows_per_zone,
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
    use std::sync::Arc;

    use arrow_array::types::Int32Type;
    use datafusion_common::ScalarValue;
    use datafusion_expr::{col, BinaryExpr, Expr, Operator};
    use lance_datagen::{BatchCount, RowCount};
    use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
    use lance_file::v2::{
        testing::{count_lance_file, write_lance_file, FsFixture},
        writer::FileWriterOptions,
    };

    use crate::{substrait::FilterExpressionExt, LanceDfFieldEncodingStrategy};

    #[test_log::test(tokio::test)]
    async fn test_basic_stats() {
        let data = lance_datagen::gen()
            .col("0", lance_datagen::array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(1024), BatchCount::from(30));

        let fs = FsFixture::default();

        let options = FileWriterOptions {
            encoding_strategy: Some(Arc::new(LanceDfFieldEncodingStrategy::default())),
            ..Default::default()
        };

        let written_file = write_lance_file(data, &fs, options).await;

        let decoder_middleware: Arc<DecoderPlugins> = Arc::default();

        let num_rows = written_file
            .data
            .iter()
            .map(|rb| rb.num_rows())
            .sum::<usize>();

        let result = count_lance_file(
            &fs,
            decoder_middleware.clone(),
            FilterExpression::no_filter(),
        )
        .await;
        assert_eq!(num_rows, result);

        let result = count_lance_file(
            &fs,
            decoder_middleware.clone(),
            FilterExpression::df_to_substrait(
                Expr::BinaryExpr(BinaryExpr {
                    left: Box::new(col("0")),
                    op: Operator::Gt,
                    right: Box::new(Expr::Literal(ScalarValue::Int32(Some(50000)))),
                }),
                written_file.schema.as_ref(),
            )
            .unwrap(),
        )
        .await;
        assert_eq!(0, result);

        let result = count_lance_file(
            &fs,
            decoder_middleware,
            FilterExpression::df_to_substrait(
                Expr::BinaryExpr(BinaryExpr {
                    left: Box::new(col("0")),
                    op: Operator::Gt,
                    right: Box::new(Expr::Literal(ScalarValue::Int32(Some(20000)))),
                }),
                written_file.schema.as_ref(),
            )
            .unwrap(),
        )
        .await;
        assert_eq!(30 * 1024 - 20000, result);
    }
}
