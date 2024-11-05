// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    any::Any,
    collections::{BTreeMap, HashMap},
    fmt::Debug,
    ops::Bound,
    sync::Arc,
};

use arrow::array::BinaryBuilder;
use arrow_array::{Array, BinaryArray, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion_common::ScalarValue;
use deepsize::DeepSizeOf;
use futures::TryStreamExt;
use lance_core::{utils::mask::RowIdTreeMap, Error, Result};
use roaring::RoaringBitmap;
use serde::Serialize;
use snafu::{location, Location};
use tracing::instrument;

use crate::{Index, IndexType};

use super::{btree::OrderableScalarValue, SargableQuery};
use super::{btree::TrainingSource, AnyQuery, IndexStore, ScalarIndex};

pub const BITMAP_LOOKUP_NAME: &str = "bitmap_page_lookup.lance";

/// A scalar index that stores a bitmap for each possible value
///
/// This index works best for low-cardinality columns, where the number of unique values is small.
/// The bitmap stores a list of row ids where the value is present.
#[derive(Clone, Debug)]
pub struct BitmapIndex {
    index_map: BTreeMap<OrderableScalarValue, RowIdTreeMap>,
    // Memoized index_map size for DeepSizeOf
    index_map_size_bytes: usize,
    store: Arc<dyn IndexStore>,
}

impl BitmapIndex {
    fn new(
        index_map: BTreeMap<OrderableScalarValue, RowIdTreeMap>,
        index_map_size_bytes: usize,
        store: Arc<dyn IndexStore>,
    ) -> Self {
        Self {
            index_map,
            index_map_size_bytes,
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
        let bitmap_binary_array = binary_bitmaps
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();

        let mut index_map: BTreeMap<OrderableScalarValue, RowIdTreeMap> = BTreeMap::new();

        let mut index_map_size_bytes = 0;
        for idx in 0..data.num_rows() {
            let key = OrderableScalarValue(ScalarValue::try_from_array(dict_keys, idx)?);
            let bitmap_bytes = bitmap_binary_array.value(idx);
            let bitmap = RowIdTreeMap::deserialize_from(bitmap_bytes).unwrap();

            index_map_size_bytes += key.deep_size_of();
            // This should be a reasonable approximation of the RowIdTreeMap size
            index_map_size_bytes += bitmap_bytes.len();
            index_map.insert(key, bitmap);
        }

        Ok(Self::new(index_map, index_map_size_bytes, store))
    }
}

impl DeepSizeOf for BitmapIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        let mut total_size = 0;

        // Size of BTreeMap values
        total_size += self.index_map_size_bytes;

        // Size of Arc<dyn IndexStore> contents
        total_size += self.store.deep_size_of_children(context);

        total_size
    }
}

#[derive(Serialize)]
struct BitmapStatistics {
    num_bitmaps: usize,
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
        IndexType::Bitmap
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let stats = BitmapStatistics {
            num_bitmaps: self.index_map.len(),
        };
        serde_json::to_value(stats).map_err(|e| Error::Internal {
            message: format!("failed to serialize bitmap index statistics: {}", e),
            location: location!(),
        })
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        unimplemented!()
    }
}

#[async_trait]
impl ScalarIndex for BitmapIndex {
    #[instrument(name = "bitmap_search", level = "debug", skip_all)]
    async fn search(&self, query: &dyn AnyQuery) -> Result<RowIdTreeMap> {
        let query = query.as_any().downcast_ref::<SargableQuery>().unwrap();

        let row_ids = match query {
            SargableQuery::Equals(val) => {
                let key = OrderableScalarValue(val.clone());
                self.index_map.get(&key).cloned().unwrap_or_default()
            }
            SargableQuery::Range(start, end) => {
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

                let maps = self
                    .index_map
                    .range((range_start, range_end))
                    .map(|(_, v)| v)
                    .collect::<Vec<_>>();

                RowIdTreeMap::union_all(&maps)
            }
            SargableQuery::IsIn(values) => {
                let mut union_bitmap = RowIdTreeMap::default();
                for val in values {
                    let key = OrderableScalarValue(val.clone());
                    if let Some(bitmap) = self.index_map.get(&key) {
                        union_bitmap |= bitmap.clone();
                    }
                }

                union_bitmap
            }
            SargableQuery::IsNull() => {
                if let Some(array) = self
                    .index_map
                    .iter()
                    .find(|(key, _)| key.0.is_null())
                    .map(|(_, value)| value)
                {
                    array.clone()
                } else {
                    RowIdTreeMap::default()
                }
            }
            SargableQuery::FullTextSearch(_) => {
                return Err(Error::NotSupported {
                    source: "full text search is not supported for bitmap indexes".into(),
                    location: location!(),
                });
            }
        };

        Ok(row_ids)
    }

    async fn load(store: Arc<dyn IndexStore>) -> Result<Arc<Self>> {
        let page_lookup_file = store.open_index_file(BITMAP_LOOKUP_NAME).await?;
        let serialized_lookup = page_lookup_file
            .read_range(0..page_lookup_file.num_rows(), None)
            .await?;

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
        let state = self
            .index_map
            .iter()
            .map(|(key, bitmap)| {
                let bitmap = RowIdTreeMap::from_iter(
                    bitmap
                        .row_ids()
                        .unwrap()
                        .filter_map(|addr| *mapping.get(&u64::from(addr))?),
                );
                (key.0.clone(), bitmap)
            })
            .collect::<HashMap<_, _>>();
        write_bitmap_index(state, dest_store).await
    }

    /// Add the new data into the index, creating an updated version of the index in `dest_store`
    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        let state = self
            .index_map
            .iter()
            .map(|(key, bitmap)| (key.0.clone(), bitmap.clone()))
            .collect::<HashMap<_, _>>();
        do_train_bitmap_index(new_data, state, dest_store).await
    }
}

fn get_batch_from_arrays(
    keys: Arc<dyn Array>,
    binary_bitmaps: Arc<dyn Array>,
) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("keys", keys.data_type().clone(), true),
        Field::new("bitmaps", binary_bitmaps.data_type().clone(), true),
    ]));

    let columns = vec![keys, binary_bitmaps];

    Ok(RecordBatch::try_new(schema, columns)?)
}

// Takes an iterator of Vec<u64> and processes each vector
// to turn it into a RoaringTreemap. Each RoaringTreeMap is
// serialized to bytes. The entire collection is converted to a BinaryArray
fn get_bitmaps_from_iter<I>(iter: I) -> Arc<dyn Array>
where
    I: Iterator<Item = RowIdTreeMap>,
{
    let mut builder = BinaryBuilder::new();
    iter.for_each(|bitmap| {
        let mut bytes = Vec::new();
        bitmap.serialize_into(&mut bytes).unwrap();
        builder.append_value(&bytes);
    });

    Arc::new(builder.finish())
}

async fn write_bitmap_index(
    state: HashMap<ScalarValue, RowIdTreeMap>,
    index_store: &dyn IndexStore,
) -> Result<()> {
    let keys_iter = state.keys().cloned();
    let keys_array = ScalarValue::iter_to_array(keys_iter)?;

    let values_iter = state.into_values();
    let binary_bitmap_array = get_bitmaps_from_iter(values_iter);

    let record_batch = get_batch_from_arrays(keys_array, binary_bitmap_array)?;

    let mut bitmap_index_file = index_store
        .new_index_file(BITMAP_LOOKUP_NAME, record_batch.schema())
        .await?;
    bitmap_index_file.write_record_batch(record_batch).await?;
    bitmap_index_file.finish().await?;
    Ok(())
}

async fn do_train_bitmap_index(
    mut data_source: SendableRecordBatchStream,
    mut state: HashMap<ScalarValue, RowIdTreeMap>,
    index_store: &dyn IndexStore,
) -> Result<()> {
    while let Some(batch) = data_source.try_next().await? {
        debug_assert_eq!(batch.num_columns(), 2);
        debug_assert_eq!(*batch.column(1).data_type(), DataType::UInt64);

        let key_column = batch.column(0);
        let row_id_column = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();

        for i in 0..key_column.len() {
            let row_id = row_id_column.value(i);
            let key = ScalarValue::try_from_array(key_column.as_ref(), i)?;
            state.entry(key.clone()).or_default().insert(row_id);
        }
    }

    write_bitmap_index(state, index_store).await
}

pub async fn train_bitmap_index(
    data_source: Box<dyn TrainingSource + Send>,
    index_store: &dyn IndexStore,
) -> Result<()> {
    let batches_source = data_source.scan_unordered_chunks(4096).await?;

    // mapping from item to list of the row ids where it is present
    let dictionary: HashMap<ScalarValue, RowIdTreeMap> = HashMap::new();

    do_train_bitmap_index(batches_source, dictionary, index_store).await
}
