// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    any::Any,
    collections::{HashMap, HashSet},
    fmt::Debug,
    sync::Arc,
};

use arrow::array::AsArray;
use arrow_array::{Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::physical_plan::{stream::RecordBatchStreamAdapter, SendableRecordBatchStream};
use datafusion_common::ScalarValue;
use deepsize::DeepSizeOf;
use futures::TryStreamExt;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;
use snafu::{location, Location};

use crate::{Index, IndexType};

use super::{bitmap::train_bitmap_index, SargableQuery};
use super::{
    bitmap::BitmapIndex, btree::BtreeTrainingSource, AnyQuery, IndexStore, ScalarIndex, TagQuery,
};

pub const BITMAP_LOOKUP_NAME: &str = "bitmap_page_lookup.lance";

trait TagSubIndex: ScalarIndex + DeepSizeOf {}

impl<T: ScalarIndex + DeepSizeOf> TagSubIndex for T {}

/// A scalar index that can be used on List<T> columns to
/// support queries with array_contains_all and array_contains_any
/// using an underlying bitmap index.
#[derive(Clone, Debug, DeepSizeOf)]
pub struct TagIndex {
    values_index: Arc<dyn TagSubIndex>,
}

impl TagIndex {
    fn new(values_index: Arc<dyn TagSubIndex>) -> Self {
        Self { values_index }
    }
}

#[async_trait]
impl Index for TagIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn crate::vector::VectorIndex>> {
        Err(Error::NotSupported {
            source: "TagIndex is not a vector index".into(),
            location: location!(),
        })
    }

    fn index_type(&self) -> IndexType {
        IndexType::Scalar
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        self.values_index.statistics()
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        unimplemented!()
    }
}

impl TagIndex {
    async fn search_values(&self, values: &Vec<ScalarValue>) -> Result<Vec<UInt64Array>> {
        let mut value_results = Vec::with_capacity(values.len());
        for value in values {
            let value_query = SargableQuery::Equals(value.clone());
            value_results.push(self.values_index.search(&value_query).await?);
        }
        Ok(value_results)
    }

    fn set_union(&self, sets: Vec<UInt64Array>) -> UInt64Array {
        if sets.len() == 1 {
            sets.into_iter().next().unwrap()
        } else {
            let combined = sets
                .iter()
                .flat_map(|arr| arr.values())
                .copied()
                .collect::<HashSet<_>>();
            UInt64Array::from_iter_values(combined)
        }
    }

    fn set_intersection(&self, sets: Vec<UInt64Array>) -> UInt64Array {
        let mut set_iter = sets.into_iter();
        let mut all: HashSet<_> = set_iter.next().unwrap().into_iter().collect();
        for next in set_iter {
            let next_set = next.into_iter().collect::<HashSet<_>>();
            all.retain(|item| !next_set.contains(item));
        }
        all.into_iter().collect()
    }
}

#[async_trait]
impl ScalarIndex for TagIndex {
    async fn search(&self, query: &dyn AnyQuery) -> Result<UInt64Array> {
        let query = query.as_any().downcast_ref::<TagQuery>().unwrap();

        match query {
            TagQuery::HasAllTags(tags) => {
                let values_results = self.search_values(tags).await?;
                Ok(self.set_union(values_results))
            }
            TagQuery::HasOneTag(tags) => {
                let values_results = self.search_values(tags).await?;
                Ok(self.set_intersection(values_results))
            }
        }
    }

    async fn load(store: Arc<dyn IndexStore>) -> Result<Arc<Self>> {
        BitmapIndex::load(store)
            .await
            .map(|index| Arc::new(Self::new(index)))
    }

    /// Remap the row ids, creating a new remapped version of this index in `dest_store`
    async fn remap(
        &self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        self.values_index.remap(mapping, dest_store).await
    }

    /// Add the new data into the index, creating an updated version of the index in `dest_store`
    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        self.values_index.update(new_data, dest_store).await
    }
}

fn extract_flatten_indices(list_arr: &dyn Array) -> UInt64Array {
    if let Some(list_arr) = list_arr.as_list_opt::<i32>() {
        let mut indices = Vec::with_capacity(list_arr.values().len());
        let offsets = list_arr.value_offsets();
        for (offset_idx, w) in offsets.windows(2).enumerate() {
            let size = (w[1] - w[0]) as u64;
            indices.extend((0..size).map(|_| offset_idx as u64));
        }
        UInt64Array::from(indices)
    } else if let Some(list_arr) = list_arr.as_list_opt::<i64>() {
        let mut indices = Vec::with_capacity(list_arr.values().len());
        let offsets = list_arr.value_offsets();
        for (offset_idx, w) in offsets.windows(2).enumerate() {
            let size = (w[1] - w[0]) as u64;
            indices.extend((0..size).map(|_| offset_idx as u64));
        }
        UInt64Array::from(indices)
    } else {
        unreachable!("Should verify that the first column is a list earlier")
    }
}

fn unnest_schema(schema: &Schema) -> SchemaRef {
    let mut fields_iter = schema.fields.iter().cloned();
    let key_field = fields_iter.next().unwrap();
    let remaining_fields = fields_iter.collect::<Vec<_>>();

    let new_key_field = if let DataType::List(item_field) = key_field.data_type() {
        Field::new(
            key_field.name(),
            item_field.data_type().clone(),
            item_field.is_nullable() || key_field.is_nullable(),
        )
    } else {
        unreachable!("Should verify that the first column is a list earlier")
    };

    let all_fields = vec![Arc::new(new_key_field)]
        .into_iter()
        .chain(remaining_fields)
        .collect::<Vec<_>>();

    Arc::new(Schema::new(Fields::from(all_fields)))
}

fn unnest_batch(
    batch: arrow::record_batch::RecordBatch,
    unnest_schema: SchemaRef,
) -> datafusion_common::Result<RecordBatch> {
    let mut columns_iter = batch.columns().iter().cloned();
    let key_col = columns_iter.next().unwrap();
    let remaining_cols = columns_iter.collect::<Vec<_>>();

    let remaining_fields = unnest_schema
        .fields
        .iter()
        .skip(1)
        .cloned()
        .collect::<Vec<_>>();

    let remaining_batch = RecordBatch::try_new(
        Arc::new(Schema::new(Fields::from(remaining_fields.clone()))),
        remaining_cols,
    )?;

    let flatten_indices = extract_flatten_indices(key_col.as_ref());

    let flattened_remaining =
        arrow_select::take::take_record_batch(&remaining_batch, &flatten_indices)?;

    let new_key_values = if let Some(key_list) = key_col.as_list_opt::<i32>() {
        let value_start = key_list.value_offsets()[key_list.offset()] as usize;
        let value_stop = key_list.value_offsets()[key_list.len()] as usize;
        key_list
            .values()
            .slice(value_start, value_stop - value_start)
            .clone()
    } else if let Some(key_list) = key_col.as_list_opt::<i64>() {
        let value_start = key_list.value_offsets()[key_list.offset()] as usize;
        let value_stop = key_list.value_offsets()[key_list.len()] as usize;
        key_list
            .values()
            .slice(value_start, value_stop - value_start)
            .clone()
    } else {
        unreachable!("Should verify that the first column is a list earlier")
    };

    let all_columns = vec![new_key_values]
        .into_iter()
        .chain(flattened_remaining.columns().iter().cloned())
        .collect::<Vec<_>>();

    datafusion_common::Result::Ok(arrow::record_batch::RecordBatch::try_new(
        unnest_schema,
        all_columns,
    )?)
}

struct UnnestTrainingSource {
    source: Box<dyn BtreeTrainingSource>,
}

#[async_trait]
impl BtreeTrainingSource for UnnestTrainingSource {
    async fn scan_ordered_chunks(
        self: Box<Self>,
        chunk_size: u32,
    ) -> Result<SendableRecordBatchStream> {
        let source = self.source.scan_ordered_chunks(chunk_size).await?;
        let unnest_schema = unnest_schema(source.schema().as_ref());
        let unnest_schema_copy = unnest_schema.clone();
        let source = source.try_filter_map(move |batch| {
            std::future::ready(Some(unnest_batch(batch, unnest_schema.clone())).transpose())
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            unnest_schema_copy.clone(),
            source,
        )))
    }
}

/// Trains a new tag index
pub async fn train_tag_index(
    data_source: Box<dyn BtreeTrainingSource + Send>,
    index_store: &dyn IndexStore,
) -> Result<()> {
    let unnest_source = Box::new(UnnestTrainingSource {
        source: data_source,
    });

    train_bitmap_index(unnest_source, index_store).await
}
