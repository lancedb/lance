// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    any::Any, cmp::Ordering, collections::{BTreeMap, BinaryHeap, HashMap}, fmt::{Debug, Display}, ops::Bound, sync::Arc
};

use arrow::array::{BinaryBuilder, FixedSizeListBuilder, ListBuilder, UInt64Builder};
use arrow_array::{Array, BinaryArray, ListArray, RecordBatch, UInt64Array};
use arrow_array::types::UInt8Type;
use arrow_schema::{DataType, Field, Schema, SortOptions};
use async_trait::async_trait;
use datafusion::physical_plan::{
    sorts::sort_preserving_merge::SortPreservingMergeExec, stream::RecordBatchStreamAdapter,
    union::UnionExec, ExecutionPlan, RecordBatchStream, SendableRecordBatchStream,
};
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_expr::Accumulator;
use datafusion_physical_expr::{
    expressions::{Column, MaxAccumulator, MinAccumulator},
    PhysicalSortExpr,
};
use deepsize::DeepSizeOf;
use futures::{
    future::BoxFuture,
    stream::{self},
    FutureExt, Stream, StreamExt, TryFutureExt, TryStreamExt,
};
use lance_core::{Error, Result};
use lance_datafusion::{
    chunker::chunk_concat_stream,
    exec::{execute_plan, LanceExecutionOptions, OneShotExec},
};
use roaring::RoaringBitmap;
use roaring::treemap::RoaringTreemap;
use serde::{Serialize, Serializer};
use snafu::{location, Location};

use crate::{Index, IndexType};

use super::{
    btree::{BTreeSubIndex, BtreeTrainingSource}, flat::FlatIndexMetadata, IndexReader, IndexStore, IndexWriter, ScalarIndex, ScalarQuery
};
use std::collections::HashSet;

use super::btree::OrderableScalarValue;
use bytes::Bytes;


const BITMAP_LOOKUP_NAME: &str = "bitmap_page_lookup.lance";
const BITMAP_PAGES_NAME: &str = "bitmap_page_data.lance";

#[derive(Clone, Debug)]
pub struct LanceRoaringTreemap(RoaringTreemap);

impl DeepSizeOf for LanceRoaringTreemap {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        0
    }
}

#[derive(Clone, Debug, DeepSizeOf)]
pub struct BitmapIndex {
    index_map: BTreeMap<OrderableScalarValue, LanceRoaringTreemap>,
    page_map: BTreeMap<OrderableScalarValue, Vec<u64>>,
    store: Arc<dyn IndexStore>,
}

impl BitmapIndex {
    fn new(
        index_map: BTreeMap<OrderableScalarValue, LanceRoaringTreemap>,
        page_map: BTreeMap<OrderableScalarValue, Vec<u64>>,
        store: Arc<dyn IndexStore>,
    ) -> Self {
        Self {
            index_map,
            page_map,
            store,
        }
    }

    // creates a new BitmapIndex from a serialized RecordBatch
    fn try_from_serialized(data: RecordBatch, store: Arc<dyn IndexStore>) -> Result<Self> {

        if data.num_rows() == 0 {
            return Err(Error::Internal {
                message: "attempt to load bitmap index from empty record batch".into(),
                location: location!(),
            });
        }

        let dict_keys = data.column(0);
        let binary_bitmaps = data.column(1);
        let pages_column = data.column(2);
        let bitmap_binary_array = binary_bitmaps.as_any().downcast_ref::<BinaryArray>().unwrap();
        let pages_array = pages_column.as_any().downcast_ref::<ListArray>().unwrap();

        let mut index_map: BTreeMap<OrderableScalarValue, LanceRoaringTreemap> = BTreeMap::new();
        let mut page_map: BTreeMap<OrderableScalarValue, Vec<u64>> = BTreeMap::new();

        for idx in 0..data.num_rows() {
            let key = OrderableScalarValue(ScalarValue::try_from_array(dict_keys, idx)?);
            let bitmap_bytes = bitmap_binary_array.value(idx);
            let bitmap = LanceRoaringTreemap(RoaringTreemap::deserialize_from(bitmap_bytes).unwrap());

            let page_numbers = pages_array.value(idx).as_any().downcast_ref::<UInt64Array>().unwrap().values().to_vec();

            index_map.insert(key.clone(), bitmap);
            page_map.insert(key, page_numbers);
        }

        Ok(Self::new(index_map, page_map, store))
    }
}

#[async_trait]
impl Index for BitmapIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn crate::vector::VectorIndex>> {
        Err(Error::NotSupported {
            source: "BitmapIndex is not a vector index".into(),
            location: location!(),
        })
    }

    fn index_type(&self) -> IndexType {
        IndexType::Scalar
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        unimplemented!()
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        unimplemented!()
    }
}

#[async_trait]
impl ScalarIndex for BitmapIndex {
    async fn search(&self, query: &ScalarQuery) -> Result<UInt64Array> {

        unimplemented!()
    }

    async fn load(store: Arc<dyn IndexStore>) -> Result<Arc<Self>> {
        let page_lookup_file = store.open_index_file(BITMAP_LOOKUP_NAME).await?;
        let serialized_lookup = page_lookup_file.read_record_batch(0).await?;

        Ok(Arc::new(Self::try_from_serialized(
            serialized_lookup,
            store,
        )?))
    }

    /// Remap the row ids, creating a new remapped version of this index in `dest_store`
    async fn remap(
        &self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {

        unimplemented!()
    }

    /// Add the new data into the index, creating an updated version of the index in `dest_store`
    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {

        unimplemented!()
    }
}

fn get_batch_from_arrays(keys: Arc<dyn Array>, binary_bitmaps: Arc<dyn Array>, pages: Arc<dyn Array>) -> Result<RecordBatch> 
{
    let schema = Arc::new(Schema::new(vec![
        Field::new("keys", keys.data_type().clone(), true),
        Field::new("bitmaps", binary_bitmaps.data_type().clone(), true),
        Field::new("pages", pages.data_type().clone(), true),
    ]));

    let columns = vec![
        keys,
        binary_bitmaps,
        pages
    ];

    Ok(RecordBatch::try_new(schema, columns)?)
}

// Takes an iterator of Vec<u64> and processes each vector
// to turn it into a RoaringTreemap. Each RoaringTreeMap is
// serialized to bytes. The entire collection is converted to a BinaryArray
fn get_bitmaps_from_iter<I>(iter: I) -> Arc<dyn Array> 
where 
    I: Iterator<Item = Vec<u64>>
{
    let mut builder = BinaryBuilder::new();
    iter.for_each(|vec| {
        let mut bitmap = RoaringTreemap::new();
        bitmap.extend(vec.into_iter());
        let mut bytes = Vec::new();
        bitmap.serialize_into(&mut bytes).unwrap();
        builder.append_value(&bytes);
    });

    Arc::new(builder.finish())
}

// Takes an iterator of HashSet<u64> and processes each set
// to turn it into an Array UInt64Array. The entire collection is stored 
// as a variable length ListArray
fn get_pages_from_iter<I>(iter: I) -> Arc<dyn Array>
where
    I: Iterator<Item = HashSet<u64>>
{
    let mut builder = ListBuilder::new(UInt64Builder::new());
    iter.for_each(|set| {
        let pages_vec: Vec<u64> = set.into_iter().collect();
        builder.values().append_slice(&pages_vec);
        builder.append(true);
    });

    let pages_array = builder.finish();

    Arc::new(pages_array)
}

pub async fn train_bitmap_index(
    data_source: Box<dyn BtreeTrainingSource + Send>,
    index_store: &dyn IndexStore,
) -> Result<()> {
    let mut batch_idx = 0;
    let mut batches_source = data_source.scan_ordered_chunks(4096).await?;

    // mapping from item to list of row ids where it is present
    let mut dictionary: HashMap<ScalarValue, Vec<u64>> = HashMap::new();
    // mapping from item to set of pages where it is present
    let mut page_dictionary: HashMap<ScalarValue, HashSet<u64>> = HashMap::new();

    while let Some(batch) = batches_source.try_next().await? {

        debug_assert_eq!(batch.num_columns(), 2);
        debug_assert_eq!(*batch.column(1).data_type(), DataType::UInt64);

        let key_column = batch.column(0);

        for i in 0..key_column.len() {
            let key = ScalarValue::try_from_array(key_column.as_ref(), i)?;
            dictionary.entry(key.clone()).or_insert_with(Vec::new).push(i as u64);
            page_dictionary.entry(key).or_insert_with(HashSet::new).insert(batch_idx);
        }

        batch_idx += 1;
    }

    let keys_iter = dictionary.keys().cloned();
    let values_iter = dictionary.values().cloned();
    let pages_iter = page_dictionary.values().cloned();

    let keys_array = ScalarValue::iter_to_array(keys_iter)?;
    let binary_bitmap_array = get_bitmaps_from_iter(values_iter);
    let pages_array = get_pages_from_iter(pages_iter);

    let record_batch = get_batch_from_arrays(keys_array, binary_bitmap_array, pages_array)?;
    
    let mut bitmap_index_file = index_store
        .new_index_file(BITMAP_LOOKUP_NAME, record_batch.schema())
        .await?;
    bitmap_index_file.write_record_batch(record_batch).await?;
    bitmap_index_file.finish().await?;
    Ok(())
}