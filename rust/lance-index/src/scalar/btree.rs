// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{
    collections::{BTreeMap, BinaryHeap},
    fmt::Debug,
    ops::Bound,
    sync::Arc,
};

use arrow_array::{Array, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::{Field, Schema};
use async_trait::async_trait;
use datafusion_common::ScalarValue;
use datafusion_expr::Accumulator;
use datafusion_physical_expr::expressions::{MaxAccumulator, MinAccumulator};
use futures::{stream, FutureExt, Stream, StreamExt, TryStreamExt};
use lance_core::Result;

use super::{
    flat::FlatIndexLoader, IndexReader, IndexStore, IndexWriter, ScalarIndex, ScalarQuery,
};

const BTREE_LOOKUP_NAME: &str = "page_lookup.lance";
const BTREE_PAGES_NAME: &str = "page_data.lance";

#[derive(Debug)]
struct OrderableScalarValue(ScalarValue);

impl PartialEq for OrderableScalarValue {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl Eq for OrderableScalarValue {}

impl PartialOrd for OrderableScalarValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for OrderableScalarValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // TODO: impl Ord for ScalarValue directly upstream in DataFusion
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Debug)]
pub struct BTreeLookup {
    tree: BTreeMap<OrderableScalarValue, Vec<u64>>,
    /// Pages where the value is null
    null_pages: Vec<u64>,
}

// impl<T: Ord + Send + Sync + Debug> BTreeLookup<T> {
impl BTreeLookup {
    fn new(tree: BTreeMap<OrderableScalarValue, Vec<u64>>, null_pages: Vec<u64>) -> Self {
        Self { tree, null_pages }
    }

    // All pages that could have a value equal to val
    fn pages_eq(&self, val: &ScalarValue) -> Vec<u64> {
        self.tree
            .range((
                Bound::Included(OrderableScalarValue(val.clone())),
                Bound::Unbounded,
            ))
            .next()
            .map(|(_, val)| val.clone())
            .unwrap_or_default()
    }

    // All pages that could have a value equal to one of the values
    fn pages_in(&self, values: &[ScalarValue]) -> Vec<u64> {
        let page_lists = values
            .iter()
            .map(|val| self.pages_eq(val))
            .collect::<Vec<_>>();
        let total_size = page_lists.iter().map(|set| set.len()).sum();
        let mut heap = BinaryHeap::with_capacity(total_size);
        for page_list in page_lists {
            heap.extend(page_list);
        }
        let mut all_pages = heap.into_sorted_vec();
        all_pages.dedup();
        all_pages
    }

    fn wrap_bound(bound: Bound<ScalarValue>) -> Bound<OrderableScalarValue> {
        match bound {
            Bound::Unbounded => Bound::Unbounded,
            Bound::Included(val) => Bound::Included(OrderableScalarValue(val)),
            Bound::Excluded(val) => Bound::Excluded(OrderableScalarValue(val)),
        }
    }

    // All pages that could have a value in the range
    fn pages_between(&self, range: (Bound<ScalarValue>, Bound<ScalarValue>)) -> Vec<u64> {
        self.tree
            .range((Self::wrap_bound(range.0), Self::wrap_bound(range.1)))
            .map(|(_, val)| val)
            .flatten()
            .copied()
            .collect::<Vec<_>>()
    }

    fn pages_null(&self) -> Vec<u64> {
        self.null_pages.clone()
    }
}

#[async_trait]
pub trait SubIndexLoader: std::fmt::Debug + Send + Sync {
    async fn load_subindex(
        &self,
        offset: u64,
        index_reader: Arc<dyn IndexReader>,
    ) -> Result<Arc<dyn ScalarIndex>>;
}

#[derive(Debug)]
pub struct BTreeIndex {
    page_lookup: BTreeLookup,
    store: Arc<dyn IndexStore>,
    sub_index: Arc<dyn SubIndexLoader>,
}

impl BTreeIndex {
    fn new(
        tree: BTreeMap<OrderableScalarValue, Vec<u64>>,
        null_pages: Vec<u64>,
        store: Arc<dyn IndexStore>,
        sub_index: Arc<dyn SubIndexLoader>,
    ) -> Self {
        let page_lookup = BTreeLookup::new(tree, null_pages);
        Self {
            page_lookup,
            store,
            sub_index,
        }
    }

    async fn search_page(
        &self,
        query: &ScalarQuery,
        page_offset: u64,
        index_reader: Arc<dyn IndexReader>,
    ) -> Result<UInt64Array> {
        let subindex = self
            .sub_index
            .load_subindex(page_offset, index_reader)
            .await?;
        // TODO: If this is an IN query we can perhaps simplify the subindex query by restricting it to the
        // values that might be in the page.  E.g. if we are searching for X IN [5, 3, 7] and five is in pages
        // 1 and 2 and three is in page 2 and seven is in pages 8 and 9 then when we search page 2 we only need
        // to search for X IN [5, 3]
        subindex.search(query).await
    }

    fn try_from_serialized(data: RecordBatch, store: Arc<dyn IndexStore>) -> Result<Self> {
        let mut map = BTreeMap::<OrderableScalarValue, Vec<u64>>::new();
        let mut null_pages = Vec::<u64>::new();

        let mins = data.column(0);
        let maxs = data.column(1);
        let null_counts = data
            .column(2)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let offsets = data
            .column(3)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();

        let first_min = ScalarValue::try_from_array(&mins, 0)?;
        map.entry(OrderableScalarValue(first_min)).or_default();

        for idx in 0..data.num_rows() {
            let max = ScalarValue::try_from_array(&maxs, idx)?;
            let null_count = null_counts.values()[idx];
            let offset = offsets.values()[idx];

            dbg!(&max, &null_count, &offset);

            map.entry(OrderableScalarValue(max))
                .or_default()
                .push(offset);
            if null_count > 0 {
                null_pages.push(offset);
            }
        }

        // TODO: Support other page types?
        let sub_index = Arc::new(FlatIndexLoader {});

        Ok(Self::new(map, null_pages, store, sub_index))
    }
}

#[async_trait]
impl ScalarIndex for BTreeIndex {
    async fn search(&self, query: &ScalarQuery) -> Result<UInt64Array> {
        let pages = match query {
            ScalarQuery::Equals(val) => self.page_lookup.pages_eq(val),
            ScalarQuery::Range(start, end) => {
                self.page_lookup.pages_between((start.clone(), end.clone()))
            }
            ScalarQuery::IsIn(values) => self.page_lookup.pages_in(values),
            ScalarQuery::IsNull() => self.page_lookup.pages_null(),
        };
        let sub_index_reader = self.store.open_index_file(BTREE_PAGES_NAME).await?;
        let page_tasks = pages
            .into_iter()
            .map(|page_index| {
                self.search_page(query, page_index, sub_index_reader.clone())
                    .boxed()
            })
            .collect::<Vec<_>>();
        let row_id_lists = stream::iter(page_tasks)
            .buffered(num_cpus::get())
            .try_collect::<Vec<UInt64Array>>()
            .await?;
        let total_size = row_id_lists
            .iter()
            .map(|row_id_list| row_id_list.len())
            .sum();
        let mut all_row_ids = Vec::with_capacity(total_size);
        for row_id_list in row_id_lists {
            all_row_ids.extend(row_id_list.values());
        }
        Ok(UInt64Array::from_iter_values(all_row_ids))
    }

    async fn load(store: Arc<dyn IndexStore>) -> Result<Arc<Self>> {
        let page_lookup_file = store.open_index_file(BTREE_LOOKUP_NAME).await?;
        let serialized_lookup = page_lookup_file.read_record_batch(0).await?;
        Ok(Arc::new(Self::try_from_serialized(
            serialized_lookup,
            store,
        )?))
    }
}

struct BatchStats {
    min: ScalarValue,
    max: ScalarValue,
    null_count: u32,
}

fn min_val(array: &Arc<dyn Array>) -> Result<ScalarValue> {
    let mut acc = MinAccumulator::try_new(array.data_type())?;
    acc.update_batch(&[array.clone()])?;
    Ok(acc.evaluate()?)
}

fn max_val(array: &Arc<dyn Array>) -> Result<ScalarValue> {
    let mut acc = MaxAccumulator::try_new(array.data_type())?;
    acc.update_batch(&[array.clone()])?;
    Ok(acc.evaluate()?)
}

fn analyze_batch(batch: &RecordBatch) -> Result<BatchStats> {
    let values = batch.column(0);
    let min = min_val(&values)?;
    let max = max_val(&values)?;
    Ok(BatchStats {
        min,
        max,
        null_count: values.null_count() as u32,
    })
}

#[async_trait]
pub trait SubIndexTrainer: Send + Sync {
    fn schema(&self) -> &Arc<Schema>;
    async fn train(&self, batch: RecordBatch, writer: &mut dyn IndexWriter) -> Result<u64>;
}

struct EncodedBatch {
    stats: BatchStats,
    offset: u64,
}

async fn train_btree_page(
    batch: RecordBatch,
    batch_idx: u32,
    sub_index_trainer: &dyn SubIndexTrainer,
    writer: &mut dyn IndexWriter,
) -> Result<EncodedBatch> {
    let stats = analyze_batch(&batch)?;
    sub_index_trainer.train(batch, writer).await?;
    Ok(EncodedBatch {
        stats,
        offset: batch_idx as u64,
    })
}

fn btree_stats_as_batch(stats: Vec<EncodedBatch>) -> Result<RecordBatch> {
    let mins = ScalarValue::iter_to_array(stats.iter().map(|stat| stat.stats.min.clone()))?;
    let maxs = ScalarValue::iter_to_array(stats.iter().map(|stat| stat.stats.max.clone()))?;
    let null_counts = UInt32Array::from_iter_values(stats.iter().map(|stat| stat.stats.null_count));
    let offsets = UInt64Array::from_iter_values(stats.iter().map(|stat| stat.offset));

    let schema = Arc::new(Schema::new(vec![
        Field::new("min", mins.data_type().clone(), false),
        Field::new("max", maxs.data_type().clone(), false),
        Field::new("null_count", null_counts.data_type().clone(), false),
        Field::new("idx_offset", offsets.data_type().clone(), false),
    ]));

    let columns = vec![
        mins,
        maxs,
        Arc::new(null_counts) as Arc<dyn Array>,
        Arc::new(offsets) as Arc<dyn Array>,
    ];

    Ok(RecordBatch::try_new(schema, columns)?)
}

pub async fn train_btree_index<S: Stream<Item = Result<RecordBatch>> + Unpin>(
    mut batches: S,
    sub_index_trainer: &dyn SubIndexTrainer,
    index_store: &dyn IndexStore,
) -> Result<()> {
    let mut sub_index_file = index_store
        .new_index_file(BTREE_PAGES_NAME, sub_index_trainer.schema().clone())
        .await?;
    let mut encoded_batches = Vec::new();
    let mut batch_idx = 0;
    while let Some(batch) = batches.try_next().await? {
        encoded_batches.push(
            train_btree_page(batch, batch_idx, sub_index_trainer, sub_index_file.as_mut()).await?,
        );
        batch_idx += 1;
    }
    sub_index_file.finish().await?;
    let record_batch = btree_stats_as_batch(encoded_batches)?;
    let mut btree_index_file = index_store
        .new_index_file(BTREE_LOOKUP_NAME, record_batch.schema())
        .await?;
    btree_index_file.write_record_batch(record_batch).await?;
    btree_index_file.finish().await?;
    Ok(())
}
