// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{any::Any, collections::HashMap, fmt::Debug, pin::Pin, sync::Arc};

use arrow::array::AsArray;
use arrow_array::{Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::execution::RecordBatchStream;
use datafusion::physical_plan::{stream::RecordBatchStreamAdapter, SendableRecordBatchStream};
use datafusion_common::ScalarValue;
use deepsize::DeepSizeOf;
use futures::{stream::BoxStream, StreamExt, TryStream, TryStreamExt};
use lance_core::cache::LanceCache;
use lance_core::{utils::mask::RowAddrTreeMap, Error, Result};
use roaring::RoaringBitmap;
use snafu::location;
use tracing::instrument;

use super::{bitmap::BitmapIndex, AnyQuery, IndexStore, LabelListQuery, ScalarIndex};
use super::{BuiltinIndexType, SargableQuery, ScalarIndexParams};
use super::{MetricsCollector, SearchResult};
use crate::frag_reuse::FragReuseIndex;
use crate::pbold;
use crate::scalar::bitmap::BitmapIndexPlugin;
use crate::scalar::expression::{LabelListQueryParser, ScalarQueryParser};
use crate::scalar::registry::{
    DefaultTrainingRequest, ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest,
    VALUE_COLUMN_NAME,
};
use crate::scalar::{CreatedIndex, UpdateCriteria};
use crate::{Index, IndexType};

pub const BITMAP_LOOKUP_NAME: &str = "bitmap_page_lookup.lance";
const LABEL_LIST_INDEX_VERSION: u32 = 0;

#[async_trait]
trait LabelListSubIndex: ScalarIndex + DeepSizeOf {
    async fn search_exact(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<RowAddrTreeMap> {
        let result = self.search(query, metrics).await?;
        match result {
            SearchResult::Exact(row_ids) => Ok(row_ids),
            _ => Err(Error::Internal {
                message: "Label list sub-index should return exact results".to_string(),
                location: location!(),
            }),
        }
    }
}

impl<T: ScalarIndex + DeepSizeOf> LabelListSubIndex for T {}

/// A scalar index that can be used on `List<T>` columns to
/// support queries with array_contains_all and array_contains_any
/// using an underlying bitmap index.
#[derive(Clone, Debug, DeepSizeOf)]
pub struct LabelListIndex {
    values_index: Arc<dyn LabelListSubIndex>,
}

impl LabelListIndex {
    fn new(values_index: Arc<dyn LabelListSubIndex>) -> Self {
        Self { values_index }
    }

    async fn load(
        store: Arc<dyn IndexStore>,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        index_cache: &LanceCache,
    ) -> Result<Arc<Self>> {
        BitmapIndex::load(store, frag_reuse_index, index_cache)
            .await
            .map(|index| Arc::new(Self::new(index)))
    }
}

#[async_trait]
impl Index for LabelListIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn crate::vector::VectorIndex>> {
        Err(Error::NotSupported {
            source: "LabeListIndex is not a vector index".into(),
            location: location!(),
        })
    }

    async fn prewarm(&self) -> Result<()> {
        self.values_index.prewarm().await
    }

    fn index_type(&self) -> IndexType {
        IndexType::LabelList
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        self.values_index.statistics()
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        unimplemented!()
    }
}

impl LabelListIndex {
    fn search_values<'a>(
        &'a self,
        values: &'a Vec<ScalarValue>,
        metrics: &'a dyn MetricsCollector,
    ) -> BoxStream<'a, Result<RowAddrTreeMap>> {
        futures::stream::iter(values)
            .then(move |value| {
                let value_query = SargableQuery::Equals(value.clone());
                async move { self.values_index.search_exact(&value_query, metrics).await }
            })
            .boxed()
    }

    async fn set_union<'a>(
        &'a self,
        mut sets: impl TryStream<Ok = RowAddrTreeMap, Error = Error> + 'a + Unpin,
        single_set: bool,
    ) -> Result<RowAddrTreeMap> {
        let mut union_bitmap = sets.try_next().await?.unwrap();
        if single_set {
            return Ok(union_bitmap);
        }
        while let Some(next) = sets.try_next().await? {
            union_bitmap |= next;
        }
        Ok(union_bitmap)
    }

    async fn set_intersection<'a>(
        &'a self,
        mut sets: impl TryStream<Ok = RowAddrTreeMap, Error = Error> + 'a + Unpin,
        single_set: bool,
    ) -> Result<RowAddrTreeMap> {
        let mut intersect_bitmap = sets.try_next().await?.unwrap();
        if single_set {
            return Ok(intersect_bitmap);
        }
        while let Some(next) = sets.try_next().await? {
            intersect_bitmap &= &next;
        }
        Ok(intersect_bitmap)
    }
}

#[async_trait]
impl ScalarIndex for LabelListIndex {
    #[instrument(skip_all, level = "debug")]
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let query = query.as_any().downcast_ref::<LabelListQuery>().unwrap();

        let row_ids = match query {
            LabelListQuery::HasAllLabels(labels) => {
                let values_results = self.search_values(labels, metrics);
                self.set_intersection(values_results, labels.len() == 1)
                    .await
            }
            LabelListQuery::HasAnyLabel(labels) => {
                let values_results = self.search_values(labels, metrics);
                self.set_union(values_results, labels.len() == 1).await
            }
        }?;
        Ok(SearchResult::Exact(row_ids))
    }

    fn can_remap(&self) -> bool {
        true
    }

    /// Remap the row ids, creating a new remapped version of this index in `dest_store`
    async fn remap(
        &self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        self.values_index.remap(mapping, dest_store).await?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::LabelListIndexDetails::default())
                .unwrap(),
            index_version: LABEL_LIST_INDEX_VERSION,
        })
    }

    /// Add the new data into the index, creating an updated version of the index in `dest_store`
    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        self.values_index
            .update(unnest_chunks(new_data)?, dest_store)
            .await?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::LabelListIndexDetails::default())
                .unwrap(),
            index_version: LABEL_LIST_INDEX_VERSION,
        })
    }

    fn update_criteria(&self) -> UpdateCriteria {
        UpdateCriteria::only_new_data(TrainingCriteria::new(TrainingOrdering::None).with_row_id())
    }

    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        Ok(ScalarIndexParams::for_builtin(BuiltinIndexType::LabelList))
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
        unreachable!(
            "Should verify that the first column is a list earlier. Got array of type: {}",
            list_arr.data_type()
        )
    }
}

fn unnest_schema(schema: &Schema) -> SchemaRef {
    let mut fields_iter = schema.fields.iter().cloned();
    let key_field = fields_iter.next().unwrap();
    let remaining_fields = fields_iter.collect::<Vec<_>>();

    let new_key_field = match key_field.data_type() {
        DataType::List(item_field) | DataType::LargeList(item_field) => Field::new(
            key_field.name(),
            item_field.data_type().clone(),
            item_field.is_nullable() || key_field.is_nullable(),
        ),
        other_type => {
            unreachable!(
                "The first field in the schema must be a List or LargeList type. \
                Found: {}. This should have been verified earlier in the code.",
                other_type
            )
        }
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
        Arc::new(Schema::new(Fields::from(remaining_fields))),
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

fn unnest_chunks(
    source: Pin<Box<dyn RecordBatchStream + Send>>,
) -> Result<SendableRecordBatchStream> {
    let unnest_schema = unnest_schema(source.schema().as_ref());
    let unnest_schema_copy = unnest_schema.clone();
    let source = source.try_filter_map(move |batch| {
        std::future::ready(Some(unnest_batch(batch, unnest_schema.clone())).transpose())
    });

    Ok(Box::pin(RecordBatchStreamAdapter::new(
        unnest_schema_copy,
        source,
    )))
}

#[derive(Debug, Default)]
pub struct LabelListIndexPlugin;

#[async_trait]
impl ScalarIndexPlugin for LabelListIndexPlugin {
    fn name(&self) -> &str {
        "LabelList"
    }

    fn new_training_request(
        &self,
        _params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        if !matches!(
            field.data_type(),
            DataType::List(_) | DataType::LargeList(_)
        ) {
            return Err(Error::InvalidInput {
                source: format!(
                    "LabelList index can only be created on List or LargeList type columns. Column has type {:?}",
                    field.data_type()
                )
                .into(),
                location: location!(),
            });
        }

        Ok(Box::new(DefaultTrainingRequest::new(
            TrainingCriteria::new(TrainingOrdering::None).with_row_id(),
        )))
    }

    fn provides_exact_answer(&self) -> bool {
        true
    }

    fn version(&self) -> u32 {
        LABEL_LIST_INDEX_VERSION
    }

    fn new_query_parser(
        &self,
        index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        Some(Box::new(LabelListQueryParser::new(index_name)))
    }

    /// Train a new index
    ///
    /// The provided data must fulfill all the criteria returned by `training_criteria`
    /// and the plugin can rely on this fact.
    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
        fragment_ids: Option<Vec<u32>>,
    ) -> Result<CreatedIndex> {
        if fragment_ids.is_some() {
            return Err(Error::InvalidInput {
                source: "LabelList index does not support fragment training".into(),
                location: location!(),
            });
        }

        let schema = data.schema();
        let field = schema
            .column_with_name(VALUE_COLUMN_NAME)
            .ok_or_else(|| Error::InvalidInput {
                source: "Index training data missing value column"
                    .to_string()
                    .into(),
                location: location!(),
            })?
            .1;

        if !matches!(
            field.data_type(),
            DataType::List(_) | DataType::LargeList(_)
        ) {
            return Err(Error::InvalidInput {
                source: format!(
                    "LabelList index can only be created on List or LargeList type columns. Column has type {:?}",
                    field.data_type()
                )
                .into(),
                location: location!(),
            });
        }

        let data = unnest_chunks(data)?;
        let bitmap_plugin = BitmapIndexPlugin;
        bitmap_plugin
            .train_index(data, index_store, request, fragment_ids)
            .await?;
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::LabelListIndexDetails::default())
                .unwrap(),
            index_version: LABEL_LIST_INDEX_VERSION,
        })
    }

    /// Load an index from storage
    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        _index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        Ok(
            LabelListIndex::load(index_store, frag_reuse_index, cache).await?
                as Arc<dyn ScalarIndex>,
        )
    }
}
