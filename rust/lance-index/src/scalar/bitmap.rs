// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    any::Any, collections::{BTreeMap, BinaryHeap, HashMap}, fmt::{Debug, Display}, ops::Bound, sync::Arc
};

use arrow::{array::{BinaryBuilder,  UInt64Builder}, datatypes::UInt64Type};
use arrow_array::{Array, BinaryArray, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use datafusion::physical_plan:: SendableRecordBatchStream;
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
use super::btree::OrderableScalarValue;

const BITMAP_LOOKUP_NAME: &str = "bitmap_page_lookup.lance";

#[derive(Clone, Debug)]
pub struct BitmapIndex {
    index_map: BTreeMap<OrderableScalarValue, UInt64Array>,
    null_rows: UInt64Array,
    bitmap_array_total_sz: usize,
    store: Arc<dyn IndexStore>,
}

impl BitmapIndex {
    fn new(
        index_map: BTreeMap<OrderableScalarValue, UInt64Array>,
        null_rows: UInt64Array,
        bitmap_array_total_sz: usize,
        store: Arc<dyn IndexStore>,
    ) -> Self {
        Self {
            index_map,
            null_rows,
            bitmap_array_total_sz,
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
        // let null_rows = data.column(2).as_any().downcast_ref::<UInt64Array>().unwrap().clone();
        let bitmap_binary_array = binary_bitmaps.as_any().downcast_ref::<BinaryArray>().unwrap();

        let null_rows: Vec<u64> = Vec::new();
        let mut index_map: BTreeMap<OrderableScalarValue, UInt64Array> = BTreeMap::new();

        let mut bitmap_array_total_sz = 0;
        for idx in 0..data.num_rows() {
            let key = OrderableScalarValue(ScalarValue::try_from_array(dict_keys, idx)?);
            let bitmap_bytes = bitmap_binary_array.value(idx);
            let bitmap = RoaringTreemap::deserialize_from(bitmap_bytes).unwrap();
            let bitmap_vec: Vec<u64> = bitmap.into_iter().collect();
            let bitmap_array = UInt64Array::from(bitmap_vec);

            index_map.insert(key.clone(), bitmap_array.clone());
            bitmap_array_total_sz += bitmap_array.len()*8;
        }

        let null_rows = UInt64Array::from(null_rows);
        Ok(Self::new(index_map, null_rows, bitmap_array_total_sz, store))
    }
}

impl DeepSizeOf for BitmapIndex {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        let mut total_size = 0;

        // Size of BTreeMap keys
        for (key, _) in &self.index_map {
            total_size += key.deep_size_of();
        }
        
        // Size of BTreeMap values
        total_size += self.bitmap_array_total_sz;

        // Size of Arc<dyn IndexStore> contents
        total_size += self.store.deep_size_of();

        total_size
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
        
        let empty_vec: Vec<u64> = Vec::new();
        let empty_array = UInt64Array::from(empty_vec);

        let row_ids = match query {
            ScalarQuery::Equals(val) => {
                let key = OrderableScalarValue(val.clone());
                self.index_map.get(&key).unwrap_or(&empty_array).clone()
            }
            ScalarQuery::Range(start, end) => {

                let range_start = match start {
                    Bound::Included(val) => Bound::Included(OrderableScalarValue(val.clone())),
                    Bound::Excluded(val) => Bound::Excluded(OrderableScalarValue(val.clone())),
                    Bound::Unbounded => Bound::Unbounded,
                };

                let range_end = match end {
                    Bound::Included(val) => Bound::Included(OrderableScalarValue(val.clone())),
                    Bound::Excluded(val) => Bound::Excluded(OrderableScalarValue(val.clone())),
                    Bound::Unbounded => Bound::Unbounded,
                };

                let range_iter = self.index_map.range((range_start, range_end));
                let total_len: usize = range_iter.clone().map(|(_, arr)| arr.len()).sum();
                let mut builder = UInt64Builder::with_capacity(total_len);
                
                for (_, array) in range_iter {
                    builder.append_slice(array.values());
                }

                builder.finish()
            }
            ScalarQuery::IsIn(values) => {
                let mut builder = UInt64Builder::new();
                for val in values {
                    let key = OrderableScalarValue(val.clone());
                    if let Some(array) = self.index_map.get(&key) {
                        builder.append_slice(array.values());
                    }
                }

                builder.finish()
            }
            ScalarQuery::IsNull() => {
                self.null_rows.clone()
            }
        };

        Ok(row_ids)
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

fn get_batch_from_arrays(keys: Arc<dyn Array>, binary_bitmaps: Arc<dyn Array>) -> Result<RecordBatch> 
{
    let schema = Arc::new(Schema::new(vec![
        Field::new("keys", keys.data_type().clone(), true),
        Field::new("bitmaps", binary_bitmaps.data_type().clone(), true),
        // Field::new("nulls", nulls.data_type().clone(), false),
    ]));

    let columns = vec![
        keys,
        binary_bitmaps,
        // nulls
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

pub async fn train_bitmap_index(
    data_source: Box<dyn BtreeTrainingSource + Send>,
    index_store: &dyn IndexStore,
) -> Result<()> {
    let mut batches_source = data_source.scan_ordered_chunks(4096).await?;

    // mapping from item to list of the row ids where it is present
    let mut dictionary: HashMap<ScalarValue, Vec<u64>> = HashMap::new();
    // let mut null_row_ids = Vec::new();

    while let Some(batch) = batches_source.try_next().await? {

        debug_assert_eq!(batch.num_columns(), 2);
        debug_assert_eq!(*batch.column(1).data_type(), DataType::UInt64);

        let key_column = batch.column(0);

        for i in 0..key_column.len() {
            let key = ScalarValue::try_from_array(key_column.as_ref(), i)?;
            dictionary.entry(key.clone()).or_insert_with(Vec::new).push(i as u64);
        }
    }

    let keys_iter = dictionary.keys().cloned();
    let values_iter = dictionary.values().cloned();

    let keys_array = ScalarValue::iter_to_array(keys_iter)?;
    let binary_bitmap_array = get_bitmaps_from_iter(values_iter);
    // let null_array = Arc::new(UInt64Array::from(null_row_ids));

    let record_batch = get_batch_from_arrays(keys_array, binary_bitmap_array)?;
    println!("Record batch: {:?}", record_batch);
    
    let mut bitmap_index_file = index_store
        .new_index_file(BITMAP_LOOKUP_NAME, record_batch.schema())
        .await?;
    bitmap_index_file.write_record_batch(record_batch).await?;
    bitmap_index_file.finish().await?;
    Ok(())
}