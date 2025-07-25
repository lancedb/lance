// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Zone Map Index
//!
//! Zone maps are a columnar database technique for predicate pushdown and scan pruning.
//! They break data into fixed-size chunks called "zones" and maintain summary statistics
//! (min, max, null count) for each zone. This enables efficient filtering by eliminating
//! zones that cannot contain matching values.
//!
//! Zone maps are "inexact" filters - they can definitively exclude zones but may include
//! false positives that require rechecking.

use std::sync::Arc;

use super::{AnyQuery, IndexStore, SargableQuery, ScalarIndex, SearchResult};
use crate::{Index, IndexType};
use arrow_array::{Array, BinaryArray, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use datafusion::physical_plan::SendableRecordBatchStream;
use deepsize::DeepSizeOf;
use futures::TryStreamExt;
use lance_core::utils::mask::RowIdTreeMap;
use lance_core::{Error, Result, ROW_ID};
use roaring::RoaringTreemap;
use serde::{Deserialize, Serialize};
use snafu::{location, Location};

use super::btree::{OrderableScalarValue, TrainingSource};

// Zone map constants
const ZONE_MAP_ZONES_FILENAME: &str = "zones.lance";
const DEFAULT_ZONE_SIZE: usize = 1024; // Number of rows per zone

fn zones_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("zone_id", DataType::UInt64, false),
        Field::new("min_value", DataType::Binary, true),
        Field::new("max_value", DataType::Binary, true),
        Field::new("null_count", DataType::UInt64, false),
        Field::new("row_ids", DataType::Binary, false),
    ]))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneStatistics {
    pub zone_id: u64,
    pub min_value: Option<Vec<u8>>,
    pub max_value: Option<Vec<u8>>,
    pub null_count: u64,
    pub row_ids: RoaringTreemap,
}

#[derive(Debug, DeepSizeOf)]
pub struct ZoneMapIndex {
    zones: Vec<ZoneStatistics>,
    data_type: DataType,
}

impl ZoneMapIndex {
    pub fn new(zones: Vec<ZoneStatistics>, data_type: DataType) -> Self {
        Self { zones, data_type }
    }

    pub async fn load(store: Arc<dyn IndexStore>) -> Result<Self> {
        let reader = store.open_index_file(ZONE_MAP_ZONES_FILENAME).await?;
        let batches: Vec<_> = reader.read_all().await?;

        if batches.is_empty() {
            return Ok(Self::new(vec![], DataType::Null));
        }

        let batch = arrow_select::concat::concat_batches(&batches[0].schema(), batches.iter())?;

        let zone_id_col = batch
            .column(0)
            .as_primitive::<arrow_array::types::UInt64Type>();
        let min_value_col = batch.column(1).as_binary::<i32>();
        let max_value_col = batch.column(2).as_binary::<i32>();
        let null_count_col = batch
            .column(3)
            .as_primitive::<arrow_array::types::UInt64Type>();
        let row_ids_col = batch.column(4).as_binary::<i32>();

        let mut zones = Vec::with_capacity(batch.num_rows());

        for i in 0..batch.num_rows() {
            let row_ids_bytes = row_ids_col.value(i);
            let mut row_ids = RoaringTreemap::new();
            row_ids.deserialize_from(&row_ids_bytes[..])?;

            zones.push(ZoneStatistics {
                zone_id: zone_id_col.value(i),
                min_value: min_value_col
                    .is_valid(i)
                    .then(|| min_value_col.value(i).to_vec()),
                max_value: max_value_col
                    .is_valid(i)
                    .then(|| max_value_col.value(i).to_vec()),
                null_count: null_count_col.value(i),
                row_ids,
            });
        }

        // Infer data type from the first valid min/max value
        let data_type = if let Some(zone) = zones.iter().find(|z| z.min_value.is_some()) {
            // For now, assume all zone maps are for numeric types
            // This can be extended to support other orderable types
            DataType::Int64 // TODO: Should be determined from the actual column type
        } else {
            DataType::Null
        };

        Ok(Self::new(zones, data_type))
    }

    fn evaluate_zone_against_query(&self, zone: &ZoneStatistics, query: &SargableQuery) -> bool {
        // If all values in the zone are null, handle null queries specially
        if zone.min_value.is_none() && zone.max_value.is_none() {
            return match query {
                SargableQuery::IsNull => true,
                _ => false,
            };
        }

        let min_val = match &zone.min_value {
            Some(bytes) => OrderableScalarValue::try_from_bytes(bytes, &self.data_type),
            None => return false,
        };
        let max_val = match &zone.max_value {
            Some(bytes) => OrderableScalarValue::try_from_bytes(bytes, &self.data_type),
            None => return false,
        };

        let (min_val, max_val) = match (min_val, max_val) {
            (Ok(min), Ok(max)) => (min, max),
            _ => return false,
        };

        match query {
            SargableQuery::Equals(target) => {
                if let Ok(target) = OrderableScalarValue::try_from_scalar_value(target) {
                    target >= min_val && target <= max_val
                } else {
                    false
                }
            }
            SargableQuery::Range(start, end) => {
                use std::ops::Bound;

                let zone_overlaps_range = match (start, end) {
                    (Bound::Unbounded, Bound::Unbounded) => true,
                    (Bound::Included(s), Bound::Unbounded)
                    | (Bound::Excluded(s), Bound::Unbounded) => {
                        if let Ok(start_val) = OrderableScalarValue::try_from_scalar_value(s) {
                            let start_check = match start {
                                Bound::Included(_) => max_val >= start_val,
                                Bound::Excluded(_) => max_val > start_val,
                                _ => unreachable!(),
                            };
                            start_check
                        } else {
                            false
                        }
                    }
                    (Bound::Unbounded, Bound::Included(e))
                    | (Bound::Unbounded, Bound::Excluded(e)) => {
                        if let Ok(end_val) = OrderableScalarValue::try_from_scalar_value(e) {
                            let end_check = match end {
                                Bound::Included(_) => min_val <= end_val,
                                Bound::Excluded(_) => min_val < end_val,
                                _ => unreachable!(),
                            };
                            end_check
                        } else {
                            false
                        }
                    }
                    (start_bound, end_bound) => {
                        let start_val = match start_bound {
                            Bound::Included(s) | Bound::Excluded(s) => {
                                OrderableScalarValue::try_from_scalar_value(s)
                            }
                            _ => unreachable!(),
                        };
                        let end_val = match end_bound {
                            Bound::Included(e) | Bound::Excluded(e) => {
                                OrderableScalarValue::try_from_scalar_value(e)
                            }
                            _ => unreachable!(),
                        };

                        match (start_val, end_val) {
                            (Ok(s), Ok(e)) => {
                                let start_check = match start_bound {
                                    Bound::Included(_) => max_val >= s,
                                    Bound::Excluded(_) => max_val > s,
                                    _ => unreachable!(),
                                };
                                let end_check = match end_bound {
                                    Bound::Included(_) => min_val <= e,
                                    Bound::Excluded(_) => min_val < e,
                                    _ => unreachable!(),
                                };
                                start_check && end_check
                            }
                            _ => false,
                        }
                    }
                };
                zone_overlaps_range
            }
            SargableQuery::IsIn(values) => {
                // Zone overlaps if any value in the set could be in the zone
                values.iter().any(|val| {
                    if let Ok(target) = OrderableScalarValue::try_from_scalar_value(val) {
                        target >= min_val && target <= max_val
                    } else {
                        false
                    }
                })
            }
            SargableQuery::IsNull => zone.null_count > 0,
            SargableQuery::IsNotNull => {
                // Zone may contain non-null values if not all values are null
                zone.min_value.is_some()
                    || zone.max_value.is_some()
                    || zone.row_ids.len() as u64 > zone.null_count
            }
        }
    }
}

#[async_trait]
impl Index for ZoneMapIndex {
    fn index_type(&self) -> IndexType {
        IndexType::ZoneMap
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "index_type": "ZoneMap",
            "num_zones": self.zones.len(),
        }))
    }

    async fn calculate_included_frags(&self) -> Result<roaring::RoaringBitmap> {
        // Zone maps can theoretically be built on any fragment
        // For now, return all fragments as potentially included
        Ok(roaring::RoaringBitmap::new())
    }
}

#[async_trait]
impl ScalarIndex for ZoneMapIndex {
    async fn search(&self, query: &dyn AnyQuery) -> Result<SearchResult> {
        let sargable_query = query.as_any().downcast_ref::<SargableQuery>();
        if sargable_query.is_none() {
            return Err(Error::InvalidInput {
                source: "ZoneMapIndex can only handle SargableQuery".into(),
                location: location!(),
            });
        }
        let query = sargable_query.unwrap();

        let mut matching_row_ids = RoaringTreemap::new();

        for zone in &self.zones {
            if self.evaluate_zone_against_query(zone, query) {
                matching_row_ids |= &zone.row_ids;
            }
        }

        // Zone maps are inexact - they may return false positives
        Ok(SearchResult::AtMost(RowIdTreeMap::from(matching_row_ids)))
    }

    fn can_answer_exact(&self, _query: &dyn AnyQuery) -> bool {
        false // Zone maps are always inexact filters
    }

    async fn load(store: Arc<dyn IndexStore>) -> Result<Arc<Self>>
    where
        Self: Sized,
    {
        Ok(Arc::new(Self::load(store).await?))
    }
}

pub struct ZoneMapIndexBuilder {
    zone_size: usize,
    data_type: DataType,
    zones: Vec<ZoneStatistics>,
    current_zone_rows: Vec<(u64, Option<OrderableScalarValue>)>,
    current_zone_id: u64,
}

impl ZoneMapIndexBuilder {
    pub fn new(zone_size: usize, data_type: DataType) -> Self {
        Self {
            zone_size,
            data_type,
            zones: Vec::new(),
            current_zone_rows: Vec::new(),
            current_zone_id: 0,
        }
    }

    fn validate_schema(schema: &Schema) -> Result<()> {
        if schema.fields().len() != 2 {
            return Err(Error::InvalidInput {
                source: "Zone map index schema must have exactly two fields".into(),
                location: location!(),
            });
        }
        if *schema.field(1).data_type() != DataType::UInt64 {
            return Err(Error::InvalidInput {
                source: "Second field in zone map index schema must be of type UInt64".into(),
                location: location!(),
            });
        }
        Ok(())
    }

    fn finalize_current_zone(&mut self) {
        if self.current_zone_rows.is_empty() {
            return;
        }

        let mut min_val: Option<OrderableScalarValue> = None;
        let mut max_val: Option<OrderableScalarValue> = None;
        let mut null_count = 0u64;
        let mut row_ids = RoaringTreemap::new();

        for (row_id, value) in &self.current_zone_rows {
            row_ids.insert(*row_id);

            match value {
                Some(val) => {
                    if min_val.is_none() || val < min_val.as_ref().unwrap() {
                        min_val = Some(val.clone());
                    }
                    if max_val.is_none() || val > max_val.as_ref().unwrap() {
                        max_val = Some(val.clone());
                    }
                }
                None => null_count += 1,
            }
        }

        let zone_stats = ZoneStatistics {
            zone_id: self.current_zone_id,
            min_value: min_val.map(|v| v.to_bytes()),
            max_value: max_val.map(|v| v.to_bytes()),
            null_count,
            row_ids,
        };

        self.zones.push(zone_stats);
        self.current_zone_rows.clear();
        self.current_zone_id += 1;
    }

    fn process_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        let data_col = batch.column(0);
        let row_id_col = batch
            .column(1)
            .as_primitive::<arrow_array::types::UInt64Type>();

        for (i, row_id) in row_id_col.values().iter().enumerate() {
            let value = if data_col.is_valid(i) {
                Some(OrderableScalarValue::try_from_array_value(data_col, i)?)
            } else {
                None
            };

            self.current_zone_rows.push((*row_id, value));

            if self.current_zone_rows.len() >= self.zone_size {
                self.finalize_current_zone();
            }
        }

        Ok(())
    }

    pub async fn train(&mut self, mut data: SendableRecordBatchStream) -> Result<()> {
        let schema = data.schema();
        Self::validate_schema(schema.as_ref())?;

        while let Some(batch) = data.try_next().await? {
            self.process_batch(&batch)?;
        }

        // Finalize any remaining data in the current zone
        self.finalize_current_zone();

        Ok(())
    }

    pub async fn write(self, store: &dyn IndexStore) -> Result<()> {
        if self.zones.is_empty() {
            // Create an empty file
            let schema = zones_schema();
            let empty_batch = RecordBatch::new_empty(schema.clone());
            let mut writer = store
                .new_index_file(ZONE_MAP_ZONES_FILENAME, schema)
                .await?;
            writer.write_record_batch(empty_batch).await?;
            writer.finish().await?;
            return Ok(());
        }

        let zone_ids = UInt64Array::from_iter_values(self.zones.iter().map(|z| z.zone_id));

        let min_values = BinaryArray::from_opt_vec(
            self.zones
                .iter()
                .map(|z| z.min_value.as_ref().map(|v| v.as_slice()))
                .collect(),
        );

        let max_values = BinaryArray::from_opt_vec(
            self.zones
                .iter()
                .map(|z| z.max_value.as_ref().map(|v| v.as_slice()))
                .collect(),
        );

        let null_counts = UInt64Array::from_iter_values(self.zones.iter().map(|z| z.null_count));

        let row_ids = BinaryArray::from_iter_values(self.zones.iter().map(|zone| {
            let mut buf = Vec::new();
            zone.row_ids.serialize_into(&mut buf).unwrap();
            buf
        }));

        let schema = zones_schema();
        let zones_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(zone_ids),
                Arc::new(min_values),
                Arc::new(max_values),
                Arc::new(null_counts),
                Arc::new(row_ids),
            ],
        )?;

        let mut writer = store
            .new_index_file(ZONE_MAP_ZONES_FILENAME, schema)
            .await?;
        writer.write_record_batch(zones_batch).await?;
        writer.finish().await?;

        Ok(())
    }
}

pub async fn train_zone_map_index(
    data_source: Box<dyn TrainingSource + Send>,
    index_store: &dyn IndexStore,
    data_type: DataType,
) -> Result<()> {
    let batches_source = data_source.scan_unordered_chunks(4096).await?;
    let mut builder = ZoneMapIndexBuilder::new(DEFAULT_ZONE_SIZE, data_type);

    builder.train(batches_source).await?;
    builder.write(index_store).await
}
collections::HashMap;
use std::