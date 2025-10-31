// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::frag_reuse::FragReuseIndex;
use crate::metrics::{MetricsCollector, NoOpMetricsCollector};
use crate::scalar::expression::{GeoQueryParser, ScalarQueryParser};
use crate::scalar::lance_format::LanceIndexStore;
use crate::scalar::registry::{
    ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest,
};
pub use crate::scalar::rtree::bbox::{
    bounding_box, bounding_box_single_scalar, total_bounds, update_total_bounds, BoundingBox,
};
use crate::scalar::rtree::sort::Sorter;
use crate::scalar::{
    AnyQuery, BuiltinIndexType, CreatedIndex, GeoQuery, IndexReader, IndexReaderStream, IndexStore,
    IndexWriter, ScalarIndex, ScalarIndexParams, SearchResult, UpdateCriteria,
};
use crate::vector::VectorIndex;
use crate::{pb, Index, IndexType};
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::UInt32Array;
use arrow_array::{Array, BinaryArray, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_common::DataFusionError;
use deepsize::DeepSizeOf;
use futures::{stream, StreamExt, TryFutureExt, TryStreamExt};
use geoarrow_array::array::{from_arrow_array, RectArray};
use geoarrow_array::builder::RectBuilder;
use geoarrow_array::{GeoArrowArray, GeoArrowArrayAccessor, IntoArrow};
use geoarrow_schema::{Dimension, RectType};
use lance_arrow::RecordBatchExt;
use lance_core::cache::{CacheKey, LanceCache, WeakLanceCache};
use lance_core::utils::address::RowAddress;
use lance_core::utils::mask::RowIdTreeMap;
use lance_core::utils::tempfile::TempDir;
use lance_core::{Error, Result, ROW_ID};
use lance_datafusion::chunker::chunk_concat_stream;
use lance_io::object_store::ObjectStore;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use snafu::location;
use sort::hilbert_sort::HilbertSorter;
use std::any::Any;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, LazyLock};

pub mod bbox;
mod sort;

pub const DEFAULT_RTREE_PAGE_SIZE: u32 = 4096;
const RTREE_INDEX_VERSION: u32 = 0;
const RTREE_PAGES_NAME: &str = "page_data.lance";
const RTREE_NULLS_NAME: &str = "nulls.lance";

static BBOX_SCHEMA: LazyLock<Arc<ArrowSchema>> = LazyLock::new(|| {
    let bbox_type = RectType::new(Dimension::XY, Default::default());
    Arc::new(ArrowSchema::new(vec![bbox_type.to_field("bbox", false)]))
});
static BBOX_ROWID_SCHEMA: LazyLock<Arc<ArrowSchema>> = LazyLock::new(|| {
    let mut fields = BBOX_SCHEMA.fields().iter().cloned().collect::<Vec<_>>();
    fields.push(Arc::new(ArrowField::new(ROW_ID, DataType::UInt64, true)));
    Arc::new(ArrowSchema::new(fields))
});
static RTREE_PAGE_SCHEMA: LazyLock<Arc<ArrowSchema>> = LazyLock::new(|| BBOX_ROWID_SCHEMA.clone());

static RTREE_NULLS_SCHEMA: LazyLock<Arc<ArrowSchema>> = LazyLock::new(|| {
    Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "nulls",
        DataType::Binary,
        false,
    )]))
});

#[derive(Debug, Clone, Serialize)]
pub struct RTreeMetadata {
    pub(crate) page_size: u32,
    pub(crate) num_pages: u64,
    pub(crate) page_offsets: Vec<usize>,
    pub(crate) num_items: usize,
    pub(crate) bbox: BoundingBox,
}

impl RTreeMetadata {
    pub fn new(
        page_size: u32,
        num_pages: u64,
        page_offsets: Vec<usize>,
        num_items: usize,
        bbox: BoundingBox,
    ) -> Self {
        Self {
            page_size,
            num_pages,
            page_offsets,
            num_items,
            bbox,
        }
    }

    fn into_map(self) -> HashMap<String, String> {
        HashMap::from_iter(vec![
            ("page_size".to_owned(), self.page_size.to_string()),
            ("num_pages".to_owned(), self.num_pages.to_string()),
            (
                "page_offsets".to_owned(),
                serde_json::json!(self.page_offsets).to_string(),
            ),
            ("num_items".to_owned(), self.num_items.to_string()),
            ("bbox".to_owned(), serde_json::json!(self.bbox).to_string()),
        ])
    }
}

impl From<&HashMap<String, String>> for RTreeMetadata {
    fn from(metadata: &HashMap<String, String>) -> Self {
        let page_size = metadata
            .get("page_size")
            .map(|bs| bs.parse().unwrap_or(DEFAULT_RTREE_PAGE_SIZE))
            .unwrap_or(DEFAULT_RTREE_PAGE_SIZE);
        let num_pages = metadata
            .get("num_pages")
            .map(|bs| bs.parse().unwrap_or(0))
            .unwrap_or(0);
        let page_offsets: Vec<usize> = metadata
            .get("page_offsets")
            .map(|bs| serde_json::from_str(bs).unwrap_or_default())
            .unwrap_or_default();
        let num_items = metadata
            .get("num_items")
            .map(|bs| bs.parse().unwrap_or(0))
            .unwrap_or(0);
        let bbox = metadata
            .get("bbox")
            .map(|bs| serde_json::from_str(bs).unwrap_or_default())
            .unwrap_or_default();
        Self::new(page_size, num_pages, page_offsets, num_items, bbox)
    }
}

/// Extract bounding boxes from geometry columns
pub fn extract_bounding_boxes(
    geometry_array: &dyn Array,
    geometry_field: &ArrowField,
) -> Result<RectArray> {
    let geo_array = from_arrow_array(geometry_array, geometry_field).map_err(|e| Error::Index {
        message: format!("Construct GeoArrowArray from an Arrow Array failed: {}", e),
        location: location!(),
    })?;
    let rect_array = bounding_box(geo_array.as_ref())?;

    Ok(rect_array)
}

struct BboxStreamStats {
    null_map: RowIdTreeMap,
    total_bbox: BoundingBox,
    // non-null
    num_items: usize,
}

#[derive(Debug, Clone)]
pub enum RTreeCacheKey {
    Page(u64),
    Nulls,
}

#[derive(Debug)]
pub struct RTreeCacheValue(Arc<RecordBatch>);

impl DeepSizeOf for RTreeCacheValue {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.0.get_array_memory_size()
    }
}

impl CacheKey for RTreeCacheKey {
    type ValueType = RTreeCacheValue;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        match self {
            Self::Page(page_id) => format!("page-{}", page_id).into(),
            Self::Nulls => "nulls".into(),
        }
    }
}

#[derive(Clone)]
pub struct RTreeIndex {
    pub(crate) metadata: Arc<RTreeMetadata>,
    store: Arc<dyn IndexStore>,
    frag_reuse_index: Option<Arc<FragReuseIndex>>,
    index_cache: WeakLanceCache,
    pages_reader: Arc<dyn IndexReader>,
    nulls_reader: Arc<dyn IndexReader>,
}

impl std::fmt::Debug for RTreeIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RTreeIndex")
            .field("metadata", &self.metadata)
            .field("store", &self.store)
            .finish()
    }
}

impl RTreeIndex {
    pub async fn load(
        store: Arc<dyn IndexStore>,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        index_cache: &LanceCache,
    ) -> Result<Arc<Self>> {
        let pages_reader = store.open_index_file(RTREE_PAGES_NAME).await?;
        let metadata = RTreeMetadata::from(&pages_reader.schema().metadata);
        let nulls_reader = store.open_index_file(RTREE_NULLS_NAME).await?;

        Ok(Arc::new(Self {
            metadata: Arc::new(metadata),
            store,
            frag_reuse_index,
            index_cache: WeakLanceCache::from(index_cache),
            pages_reader,
            nulls_reader,
        }))
    }

    async fn page_range(&self, page_idx: u64) -> Result<Range<usize>> {
        let start = match self.metadata.page_offsets.get(page_idx as usize) {
            None => self.pages_reader.num_rows(),
            Some(start) => *start,
        };
        let end = match self.metadata.page_offsets.get((page_idx + 1) as usize) {
            None => self.pages_reader.num_rows(),
            Some(end) => *end,
        };
        Ok(start..end)
    }

    async fn search_bbox(
        &self,
        bbox: BoundingBox,
        metrics: &dyn MetricsCollector,
    ) -> Result<RowIdTreeMap> {
        if self.metadata.num_pages == 0 {
            return Ok(RowIdTreeMap::default());
        }

        let mut row_ids = RowIdTreeMap::new();
        let mut outer_page_idx = Some(self.metadata.num_pages - 1);
        let mut queue = vec![];

        while let Some(page_idx) = outer_page_idx {
            let range = self.page_range(page_idx).await?;
            let is_leaf = range.start < self.metadata.num_items;
            let batch = self
                .index_cache
                .get_or_insert_with_key(RTreeCacheKey::Page(page_idx), move || async move {
                    let batch = self.pages_reader.read_range(range, None).await?;
                    metrics.record_part_load();
                    Ok(RTreeCacheValue(Arc::new(batch)))
                })
                .await
                .map(|v| v.0.clone())?;

            let bbox_array =
                extract_bounding_boxes(batch.column(0).as_ref(), batch.schema().field(0))?;
            let rowid_or_pageid_array = batch
                .column(1)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();

            for i in 0..bbox_array.len() {
                let rect = bbox_array.value(i).unwrap();
                if bbox.rect_intersects(&rect) {
                    if is_leaf {
                        let row_id = rowid_or_pageid_array.value(i);
                        row_ids.insert(row_id);
                    } else {
                        let page_id = rowid_or_pageid_array.value(i);
                        queue.push(page_id);
                    }
                }
            }

            outer_page_idx = queue.pop();
        }

        Ok(row_ids)
    }

    async fn search_null(&self, metrics: &dyn MetricsCollector) -> Result<RowIdTreeMap> {
        let batch = self
            .index_cache
            .get_or_insert_with_key(RTreeCacheKey::Nulls, move || async move {
                // Only one row
                let batch = self.nulls_reader.read_range(0..1, None).await?;
                metrics.record_part_load();
                Ok(RTreeCacheValue(Arc::new(batch)))
            })
            .await
            .map(|v| v.0.clone())?;

        let null_map = match batch.num_rows() {
            0 => RowIdTreeMap::default(),
            1 => {
                let bytes = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<BinaryArray>()
                    .unwrap()
                    .value(0);
                RowIdTreeMap::deserialize_from(bytes)?
            }
            _ => {
                unreachable!()
            }
        };
        Ok(null_map)
    }

    /// Create a stream of all the data in the index, in the same format used to train the index
    async fn into_data_stream(self) -> Result<SendableRecordBatchStream> {
        let reader = self.store.open_index_file(RTREE_PAGES_NAME).await?;
        let reader_stream = IndexReaderStream::new_with_limit(
            reader,
            self.metadata.page_size as u64,
            self.metadata.num_items as u64,
        )
        .await;
        let batches = reader_stream
            .map(|fut| fut.map_err(DataFusionError::from))
            .buffered(self.store.io_parallelism())
            .boxed();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            RTREE_PAGE_SCHEMA.clone(),
            batches,
        )))
    }

    async fn combine_old_new(
        self,
        new_input: SendableRecordBatchStream,
    ) -> Result<SendableRecordBatchStream> {
        let old_input = self.into_data_stream().await?;
        debug_assert_eq!(
            old_input.schema().flattened_fields().len(),
            new_input.schema().flattened_fields().len()
        );

        let merged = futures::stream::select(old_input, new_input);

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            RTREE_PAGE_SCHEMA.clone(),
            merged,
        )))
    }
}

impl DeepSizeOf for RTreeIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        let mut total_size = 0;

        total_size += self.store.deep_size_of_children(context);

        total_size
    }
}

#[async_trait]
impl Index for RTreeIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn VectorIndex>> {
        Err(Error::NotSupported {
            source: "RTreeIndex is not vector index".into(),
            location: location!(),
        })
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self.metadata.clone()).map_err(|e| Error::Internal {
            message: format!("Error serializing statistics: {}", e),
            location: location!(),
        })
    }

    async fn prewarm(&self) -> Result<()> {
        if self.metadata.num_pages > 0 {
            let mut outer_page_idx = Some(self.metadata.num_pages - 1);
            let mut queue = vec![];

            while let Some(page_idx) = outer_page_idx {
                let range = self.page_range(page_idx).await?;
                let is_leaf = range.start < self.metadata.num_items;
                let batch = Arc::new(self.pages_reader.read_range(range, None).await?);
                self.index_cache
                    .insert_with_key(
                        &RTreeCacheKey::Page(page_idx),
                        Arc::new(RTreeCacheValue(batch.clone())),
                    )
                    .await;

                let rowid_or_pageid_array = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap();

                if !is_leaf {
                    for i in 0..rowid_or_pageid_array.len() {
                        let page_id = rowid_or_pageid_array.value(i);
                        queue.push(page_id);
                    }
                }

                outer_page_idx = queue.pop();
            }
        }

        // Only one row
        let batch = self.nulls_reader.read_range(0..1, None).await?;
        self.index_cache
            .insert_with_key(
                &RTreeCacheKey::Nulls,
                Arc::new(RTreeCacheValue(Arc::new(batch))),
            )
            .await;

        Ok(())
    }

    fn index_type(&self) -> IndexType {
        IndexType::RTree
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        let mut frag_ids = RoaringBitmap::default();

        let mut reader_stream = self.clone().into_data_stream().await?;
        let mut read_rows = 0;
        while let Some(page) = reader_stream.try_next().await? {
            let mut page_frag_ids = page
                .column_by_name(ROW_ID)
                .ok_or_else(|| Error::Index {
                    message: format!("RTree page lacks {} column", ROW_ID),
                    location: location!(),
                })?
                .as_primitive::<UInt64Type>()
                .iter()
                .flatten()
                .map(|row_id| RowAddress::from(row_id).fragment_id())
                .collect::<Vec<_>>();
            page_frag_ids.sort();
            page_frag_ids.dedup();
            frag_ids |= RoaringBitmap::from_sorted_iter(page_frag_ids).unwrap();

            read_rows += page.num_rows();
            if read_rows >= self.metadata.num_items {
                break;
            }
        }
        Ok(frag_ids)
    }
}

#[async_trait]
impl ScalarIndex for RTreeIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let query = query.as_any().downcast_ref::<GeoQuery>().unwrap();
        match query {
            GeoQuery::IntersectQuery(query) => {
                let geo_array =
                    extract_bounding_boxes(query.value.to_array()?.as_ref(), &query.field)?;
                let bbox = bounding_box_single_scalar(&geo_array)?;
                let mut rowids = self.search_bbox(bbox, metrics).await?;

                if let Some(fri) = &self.frag_reuse_index {
                    rowids = fri.remap_row_ids_tree_map(&rowids);
                }
                Ok(SearchResult::AtMost(rowids))
            }
            GeoQuery::IsNull => {
                let mut null_map = self.search_null(metrics).await?;

                if let Some(fri) = &self.frag_reuse_index {
                    null_map = fri.remap_row_ids_tree_map(&null_map);
                }
                Ok(SearchResult::Exact(null_map))
            }
        }
    }

    fn can_remap(&self) -> bool {
        false
    }

    async fn remap(
        &self,
        _mapping: &HashMap<u64, Option<u64>>,
        _dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        Err(Error::InvalidInput {
            source: "RTree does not support remap".into(),
            location: location!(),
        })
    }

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        let bbox_data = RTreeIndexPlugin::convert_bbox_stream(new_data)?;
        let tmpdir = Arc::new(TempDir::default());
        let spill_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        ));
        let (new_bbox_data, analyze) = RTreeIndexPlugin::process_and_analyze_bbox_stream(
            bbox_data,
            self.metadata.page_size,
            spill_store.clone(),
        )
        .await?;

        let merged_bbox_data = self.clone().combine_old_new(new_bbox_data).await?;

        let null_map = self.search_null(&NoOpMetricsCollector).await?;

        let mut new_bbox = BoundingBox::new();
        new_bbox.add_rect(&analyze.total_bbox);
        new_bbox.add_rect(&self.metadata.bbox);

        let merge_analyze = BboxStreamStats {
            null_map: RowIdTreeMap::union_all(&[&null_map, &analyze.null_map]),
            total_bbox: new_bbox,
            num_items: self.metadata.num_items + analyze.num_items,
        };

        RTreeIndexPlugin::train_rtree_index(
            merged_bbox_data,
            merge_analyze,
            self.metadata.page_size,
            dest_store,
        )
        .await?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pb::RTreeIndexDetails::default())?,
            index_version: RTREE_INDEX_VERSION,
        })
    }

    fn update_criteria(&self) -> UpdateCriteria {
        UpdateCriteria::only_new_data(TrainingCriteria::new(TrainingOrdering::None).with_row_id())
    }

    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        let params = serde_json::to_value(RTreeParameters {
            page_size: Some(self.metadata.page_size),
        })?;
        Ok(ScalarIndexParams::for_builtin(BuiltinIndexType::RTree).with_params(&params))
    }
}

/// Parameters for a rtree index
#[derive(Debug, Serialize, Deserialize, Clone)]
struct RTreeParameters {
    /// The number of rows to include in each page
    pub page_size: Option<u32>,
}

pub struct RTreeTrainingRequest {
    parameters: RTreeParameters,
    criteria: TrainingCriteria,
}

impl RTreeTrainingRequest {
    fn new(parameters: RTreeParameters) -> Self {
        Self {
            parameters,
            criteria: TrainingCriteria::new(TrainingOrdering::None).with_row_id(),
        }
    }
}

impl Default for RTreeTrainingRequest {
    fn default() -> Self {
        Self::new(RTreeParameters {
            page_size: Some(DEFAULT_RTREE_PAGE_SIZE),
        })
    }
}

impl TrainingRequest for RTreeTrainingRequest {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn criteria(&self) -> &TrainingCriteria {
        &self.criteria
    }
}

#[derive(Debug, Default)]
pub struct RTreeIndexPlugin;

impl RTreeIndexPlugin {
    fn validate_schema(schema: &ArrowSchema) -> Result<()> {
        if schema.fields().len() != 2 {
            return Err(Error::InvalidInput {
                source: "RTree index schema must have exactly two fields".into(),
                location: location!(),
            });
        }

        let row_id_field = schema.field_with_name(ROW_ID)?;
        if *row_id_field.data_type() != DataType::UInt64 {
            return Err(Error::InvalidInput {
                source: "Second field in RTree index schema must be of type UInt64".into(),
                location: location!(),
            });
        }
        Ok(())
    }

    fn convert_bbox_stream(source: SendableRecordBatchStream) -> Result<SendableRecordBatchStream> {
        let bbox_stream = source
            .map_err(DataFusionError::into)
            .and_then(move |batch| async move {
                let schema = batch.schema();
                let geometry_field = schema.field(0);
                let geometry_array = batch.column(0);
                let bbox_array = extract_bounding_boxes(geometry_array, geometry_field)?;
                let bbox_field = bbox_array.extension_type().clone().to_field("bbox", true);

                let bbox_schema = Arc::new(ArrowSchema::new(vec![
                    bbox_field,
                    ArrowField::new(ROW_ID, DataType::UInt64, true),
                ]));
                RecordBatch::try_new(
                    bbox_schema,
                    vec![bbox_array.into_array_ref(), batch.column(1).clone()],
                )
                .map_err(DataFusionError::from)
            });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            BBOX_ROWID_SCHEMA.clone(),
            bbox_stream,
        )))
    }

    /// Processes a bounding box data stream, separating null and non-null elements, and collects
    /// statistics about non-null elements.
    async fn process_and_analyze_bbox_stream(
        mut data: SendableRecordBatchStream,
        page_size: u32,
        spill_store: Arc<LanceIndexStore>,
    ) -> Result<(SendableRecordBatchStream, BboxStreamStats)> {
        let mut null_rowids = RowIdTreeMap::new();
        let mut total_bbox = BoundingBox::new();
        let mut num_non_null_rows = 0;

        let schema = data.schema();

        // 1. Scan source data statistics bbox, and spill data to disk
        let mut writer = spill_store
            .new_index_file("analyze.tmp", BBOX_ROWID_SCHEMA.clone())
            .await?;

        while let Some(batch) = data.next().await {
            let batch = batch?;
            let bbox_array = extract_bounding_boxes(&batch.column(0), batch.schema().field(0))?;
            let rowid_array = batch
                .column(1)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();

            update_total_bounds(&mut total_bbox, &bbox_array).map_err(|e| Error::Index {
                message: format!("Construct GeoArrowArray from an Arrow Array failed: {}", e),
                location: location!(),
            })?;

            let num_rows = bbox_array.len();

            let mut non_null_indexes = vec![];

            for i in 0..num_rows {
                if bbox_array.is_null(i) {
                    let rowid = rowid_array.value(i);
                    null_rowids.insert(rowid);
                } else {
                    non_null_indexes.push(i as u32);
                }
            }

            let new_batch = if non_null_indexes.is_empty() {
                // all nulls, skip write
                continue;
            } else if non_null_indexes.len() == num_rows {
                batch
            } else {
                batch.take(&UInt32Array::from(non_null_indexes))?
            };

            num_non_null_rows += new_batch.num_rows();
            writer.write_record_batch(new_batch).await?;
        }
        writer.finish().await?;
        let reader = spill_store.open_index_file("analyze.tmp").await?;
        let stream = IndexReaderStream::new(reader, page_size as u64)
            .await
            .map(|fut| fut.map_err(DataFusionError::from))
            .buffered(spill_store.io_parallelism())
            .boxed();
        let new_data = RecordBatchStreamAdapter::new(schema.clone(), stream);

        Ok((
            Box::pin(new_data),
            BboxStreamStats {
                null_map: null_rowids,
                total_bbox,
                num_items: num_non_null_rows,
            },
        ))
    }

    pub async fn write_index(
        sorted_data: SendableRecordBatchStream,
        num_items: usize,
        bbox: BoundingBox,
        store: &dyn IndexStore,
        page_size: u32,
    ) -> Result<()> {
        let mut page_idx: u64 = 0;
        let mut writer = store
            .new_index_file(RTREE_PAGES_NAME, RTREE_PAGE_SCHEMA.clone())
            .await?;

        let mut page_offsets = vec![];
        let mut curr_offset = 0;

        if num_items > 0 {
            let mut current_level = Some((sorted_data, num_items));
            while let Some((mut data, num_items)) = current_level.take() {
                if num_items <= page_size as usize {
                    while let Some(batch) = data.next().await {
                        let batch = batch?;
                        page_offsets.push(curr_offset);
                        curr_offset += batch.num_rows();
                        train_rtree_page(batch, page_idx, writer.as_mut()).await?;
                        page_idx += 1;
                    }
                } else {
                    let mut next_level = vec![];
                    let mut paged_source = chunk_concat_stream(data, page_size as usize);
                    while let Some(batch) = paged_source.next().await {
                        let batch = batch?;
                        page_offsets.push(curr_offset);
                        curr_offset += batch.num_rows();
                        let encoded_batch =
                            train_rtree_page(batch, page_idx, writer.as_mut()).await?;
                        page_idx += 1;
                        next_level.push(encoded_batch);
                    }
                    if !next_level.is_empty() {
                        let next_num_items = next_level.len();
                        current_level = Some((
                            encoded_batches_into_batch_stream(next_level, page_size),
                            next_num_items,
                        ));
                    }
                }
            }
        }

        writer
            .finish_with_metadata(
                RTreeMetadata::new(page_size, page_idx, page_offsets, num_items, bbox).into_map(),
            )
            .await?;

        Ok(())
    }

    pub async fn write_nulls(store: &dyn IndexStore, null_map: RowIdTreeMap) -> Result<()> {
        let mut writer = store
            .new_index_file(RTREE_NULLS_NAME, RTREE_NULLS_SCHEMA.clone())
            .await?;
        let mut bytes = Vec::new();
        null_map.serialize_into(&mut bytes)?;
        let batch = RecordBatch::try_new(
            RTREE_NULLS_SCHEMA.clone(),
            vec![Arc::new(BinaryArray::from_vec(vec![&bytes]))],
        )?;

        writer.write_record_batch(batch).await?;
        writer.finish().await
    }

    async fn train_rtree_index(
        bbox_data: SendableRecordBatchStream,
        analyze: BboxStreamStats,
        page_size: u32,
        store: &dyn IndexStore,
    ) -> Result<()> {
        // new sorted stream
        let sorter = HilbertSorter::new(analyze.total_bbox);
        let sorted_data = sorter.sort(bbox_data).await?;

        Self::write_index(
            sorted_data,
            analyze.num_items,
            analyze.total_bbox,
            store,
            page_size,
        )
        .await?;

        Self::write_nulls(store, analyze.null_map).await?;

        Ok(())
    }
}

#[async_trait]
impl ScalarIndexPlugin for RTreeIndexPlugin {
    fn new_training_request(
        &self,
        params: &str,
        _field: &ArrowField,
    ) -> Result<Box<dyn TrainingRequest>> {
        let params = serde_json::from_str::<RTreeParameters>(params)?;
        Ok(Box::new(RTreeTrainingRequest::new(params)))
    }

    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
        fragment_ids: Option<Vec<u32>>,
    ) -> Result<CreatedIndex> {
        if fragment_ids.is_some() {
            return Err(Error::InvalidInput {
                source: "RTree index does not support fragment training".into(),
                location: location!(),
            });
        }

        Self::validate_schema(&data.schema())?;

        let request = request
            .as_any()
            .downcast_ref::<RTreeTrainingRequest>()
            .unwrap();
        let page_size = request
            .parameters
            .page_size
            .unwrap_or(DEFAULT_RTREE_PAGE_SIZE);

        let bbox_data = Self::convert_bbox_stream(data)?;
        let tmpdir = Arc::new(TempDir::default());
        let spill_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        ));
        let (bbox_data, analyze) =
            Self::process_and_analyze_bbox_stream(bbox_data, page_size, spill_store.clone())
                .await?;

        Self::train_rtree_index(bbox_data, analyze, page_size, index_store).await?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pb::RTreeIndexDetails::default())?,
            index_version: RTREE_INDEX_VERSION,
        })
    }

    fn provides_exact_answer(&self) -> bool {
        true
    }

    fn version(&self) -> u32 {
        RTREE_INDEX_VERSION
    }

    fn new_query_parser(
        &self,
        index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        Some(Box::new(GeoQueryParser::new(index_name)))
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        _index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        Ok(RTreeIndex::load(index_store, frag_reuse_index, cache).await? as Arc<dyn ScalarIndex>)
    }
}

struct EncodedBatch {
    bbox: BoundingBox,
    page_id: u64,
}

fn encoded_batches_into_batch_stream(
    batches: Vec<EncodedBatch>,
    batch_size: u32,
) -> SendableRecordBatchStream {
    let batches = batches
        .chunks(batch_size as usize)
        .map(|chunk| {
            let bbox_type = RectType::new(Dimension::XY, Default::default());
            let mut bbox_builder = RectBuilder::with_capacity(bbox_type, chunk.len());
            let mut page_ids = UInt64Array::builder(chunk.len());

            for item in chunk {
                bbox_builder.push_rect(Some(&item.bbox));
                page_ids.append_value(item.page_id);
            }

            RecordBatch::try_new(
                RTREE_PAGE_SCHEMA.clone(),
                vec![
                    bbox_builder.finish().into_array_ref(),
                    Arc::new(page_ids.finish()),
                ],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    Box::pin(RecordBatchStreamAdapter::new(
        RTREE_PAGE_SCHEMA.clone(),
        stream::iter(batches).map(Ok).boxed(),
    ))
}

async fn train_rtree_page(
    batch: RecordBatch,
    page_id: u64,
    writer: &mut dyn IndexWriter,
) -> Result<EncodedBatch> {
    let geo_array = extract_bounding_boxes(batch.column(0).as_ref(), batch.schema().field(0))?;
    let bbox = total_bounds(&geo_array)?;
    writer.write_record_batch(batch).await?;
    Ok(EncodedBatch { bbox, page_id })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::NoOpMetricsCollector;
    use crate::scalar::registry::VALUE_COLUMN_NAME;
    use arrow_array::ArrayRef;
    use arrow_schema::Schema;
    use geo_types::{coord, Rect};
    use geoarrow_array::builder::{PointBuilder, RectBuilder};
    use geoarrow_schema::{Dimension, PointType, RectType};
    use lance_core::utils::tempfile::TempObjDir;

    fn expected_page_offsets(num_items: usize, page_size: u32) -> Vec<usize> {
        let mut page_offsets = vec![];
        let mut cur_level_items = num_items;
        let mut cur_offset = 0;
        while cur_level_items > 0 {
            if cur_level_items <= page_size as usize {
                page_offsets.push(cur_offset);
                break;
            }
            for off in (0..cur_level_items).step_by(page_size as usize) {
                page_offsets.push(cur_offset + off);
            }
            cur_offset += cur_level_items;
            cur_level_items = cur_level_items.div_ceil(page_size as usize);
        }

        page_offsets
    }

    fn expected_num_pages(num_items: usize, page_size: u32) -> u64 {
        expected_page_offsets(num_items, page_size).len() as u64
    }

    fn convert_bbox_rowid_batch_stream(
        geo_array: &dyn GeoArrowArray,
        row_id_array: ArrayRef,
    ) -> SendableRecordBatchStream {
        let schema = Arc::new(Schema::new(vec![
            geo_array.data_type().to_field(VALUE_COLUMN_NAME, true),
            ArrowField::new(ROW_ID, DataType::UInt64, false),
        ]));

        let batch =
            RecordBatch::try_new(schema.clone(), vec![geo_array.to_array_ref(), row_id_array])
                .unwrap();

        let stream = stream::once(async move { Ok(batch) });
        Box::pin(RecordBatchStreamAdapter::new(schema, stream))
    }

    async fn train_index(
        geo_array: &dyn GeoArrowArray,
        page_size: Option<u32>,
    ) -> (Arc<RTreeIndex>, TempObjDir) {
        let page_size = page_size.unwrap_or(DEFAULT_RTREE_PAGE_SIZE);
        let mut num_items = 0;
        for i in 0..geo_array.len() {
            if !geo_array.is_null(i) {
                num_items += 1;
            }
        }

        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let stream = convert_bbox_rowid_batch_stream(
            geo_array,
            Arc::new(UInt64Array::from(
                (0..geo_array.len() as u64).collect::<Vec<_>>(),
            )),
        );

        let plugin = RTreeIndexPlugin;
        plugin
            .train_index(
                stream,
                store.as_ref(),
                Box::new(RTreeTrainingRequest::new(RTreeParameters {
                    page_size: Some(page_size),
                })),
                None,
            )
            .await
            .unwrap();

        let pages_reader = store.open_index_file(RTREE_PAGES_NAME).await.unwrap();
        let metadata = RTreeMetadata::from(&pages_reader.schema().metadata);
        assert_eq!(metadata.num_items, num_items);
        assert_eq!(metadata.num_pages, expected_num_pages(num_items, page_size));
        assert_eq!(
            metadata.page_offsets,
            expected_page_offsets(num_items, page_size)
        );

        (
            RTreeIndex::load(store, None, &LanceCache::no_cache())
                .await
                .unwrap(),
            tmpdir,
        )
    }

    #[tokio::test]
    async fn test_search_bbox() {
        let bbox_type = RectType::new(Dimension::XY, Default::default());

        let mut rect_builder = RectBuilder::new(bbox_type.clone());
        let num_items = 10000;
        let page_size = 16;
        for i in 0..num_items {
            let i = i as f64;
            rect_builder.push_rect(Some(&Rect::new(
                coord! { x: i, y: i },
                coord! { x: i + 1.0, y: i + 1.0 },
            )));
        }
        let rect_arr = rect_builder.finish();

        let (rtree_index, _tmpdir) = train_index(&rect_arr, Some(page_size)).await;

        let mut search_bbox = BoundingBox::new();
        search_bbox.add_rect(&Rect::new(
            coord! { x: 10.5, y: 1.5 },
            coord! { x: 99.5, y: 200.5 },
        ));
        let row_ids = rtree_index
            .search_bbox(search_bbox, &NoOpMetricsCollector)
            .await
            .unwrap();

        let mut expected_row_ids = RowIdTreeMap::new();
        for i in 0..rect_arr.len() {
            let mut bbox = BoundingBox::new();
            bbox.add_rect(&rect_arr.value(i).unwrap());
            if search_bbox.rect_intersects(&bbox) {
                expected_row_ids.insert(i as u64);
            }
        }
        assert_eq!(row_ids, expected_row_ids);
    }

    #[tokio::test]
    async fn test_search_null() {
        let point_type = PointType::new(Dimension::XY, Default::default());

        let mut point_builder = PointBuilder::new(point_type.clone());
        point_builder.push_point(Some(&geo_types::point!(x: -1.0, y: 1.0)));
        point_builder.push_null();
        point_builder.push_point(Some(&geo_types::point!(x: -2.0, y: 2.0)));
        point_builder.push_point(Some(&geo_types::point!(x: -3.0, y: 2.0)));
        point_builder.push_null();
        let point_arr = point_builder.finish();

        let (rtree_index, _tmpdir) = train_index(&point_arr, None).await;
        let row_ids = rtree_index
            .search_null(&NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            row_ids.row_ids().unwrap().collect::<Vec<_>>(),
            vec![
                RowAddress::new_from_parts(0, 1),
                RowAddress::new_from_parts(0, 4),
            ]
        );
    }

    #[tokio::test]
    async fn test_update_and_search() {
        let bbox_type = RectType::new(Dimension::XY, Default::default());

        let page_size = 16;
        let mut rect_builder = RectBuilder::new(bbox_type.clone());
        let num_items = 10000;
        for i in 0..num_items {
            let i = i as f64;
            rect_builder.push_rect(Some(&Rect::new(
                coord! { x: i, y: i },
                coord! { x: i + 1.0, y: i + 1.0 },
            )));
        }
        let rect_arr = rect_builder.finish();
        let (rtree_index, _tmpdir) = train_index(&rect_arr, Some(page_size)).await;

        let tmpdir = TempObjDir::default();
        let new_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let mut rect_builder = RectBuilder::new(bbox_type.clone());
        let num_items = 10000;
        for i in 0..num_items {
            let i = i as f64;
            rect_builder.push_rect(Some(&Rect::new(
                coord! { x: i + 0.5, y: i + 0.5 },
                coord! { x: i + 1.5, y: i + 1.5 },
            )));
        }
        let new_rect_arr = rect_builder.finish();
        let new_rowid_arr = (rect_arr.len() as u64..(rect_arr.len() + new_rect_arr.len()) as u64)
            .collect::<Vec<_>>();
        let stream = convert_bbox_rowid_batch_stream(
            &new_rect_arr,
            Arc::new(UInt64Array::from(new_rowid_arr.clone())),
        );
        rtree_index
            .update(stream, new_store.as_ref())
            .await
            .unwrap();

        let new_rtree_index = RTreeIndex::load(new_store.clone(), None, &LanceCache::no_cache())
            .await
            .unwrap();

        let mut search_bbox = BoundingBox::new();
        search_bbox.add_rect(&Rect::new(
            coord! { x: 10.5, y: 1.5 },
            coord! { x: 99.5, y: 200.5 },
        ));
        let row_ids = new_rtree_index
            .search_bbox(search_bbox, &NoOpMetricsCollector)
            .await
            .unwrap();

        let mut expected_row_ids = RowIdTreeMap::new();
        for i in 0..rect_arr.len() {
            let bbox = BoundingBox::new_with_rect(&rect_arr.value(i).unwrap());
            if search_bbox.rect_intersects(&bbox) {
                expected_row_ids.insert(i as u64);
            }
        }
        for i in 0..new_rect_arr.len() {
            let bbox = BoundingBox::new_with_rect(&new_rect_arr.value(i).unwrap());
            if search_bbox.rect_intersects(&bbox) {
                expected_row_ids.insert(new_rowid_arr.get(i).copied().unwrap());
            }
        }

        assert_eq!(row_ids, expected_row_ids);
    }
}
