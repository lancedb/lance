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
    any::Any,
    cmp::Ordering,
    collections::{BTreeMap, BinaryHeap, HashMap},
    fmt::{Debug, Display},
    ops::Bound,
    sync::Arc,
};

use arrow_array::{Array, RecordBatch, UInt32Array, UInt64Array};
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
use serde::{Serialize, Serializer};
use snafu::{location, Location};

use crate::{Index, IndexType};

use super::{
    flat::FlatIndexMetadata, IndexReader, IndexStore, IndexWriter, ScalarIndex, ScalarQuery,
};

const BTREE_LOOKUP_NAME: &str = "page_lookup.lance";
const BTREE_PAGES_NAME: &str = "page_data.lance";

/// Wraps a ScalarValue and implements Ord (ScalarValue only implements PartialOrd)
#[derive(Clone, Debug)]
struct OrderableScalarValue(ScalarValue);

impl Display for OrderableScalarValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl PartialEq for OrderableScalarValue {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl Eq for OrderableScalarValue {}

impl PartialOrd for OrderableScalarValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// manual implementation of `Ord` that panics when asked to compare scalars of different type
// and always puts nulls before non-nulls (this is consistent with Option<T>'s implementation
// of Ord)
//
// TODO: Consider upstreaming this
impl Ord for OrderableScalarValue {
    fn cmp(&self, other: &Self) -> Ordering {
        use ScalarValue::*;
        // This purposely doesn't have a catch-all "(_, _)" so that
        // any newly added enum variant will require editing this list
        // or else face a compile error
        match (&self.0, &other.0) {
            (Decimal128(v1, p1, s1), Decimal128(v2, p2, s2)) => {
                if p1.eq(p2) && s1.eq(s2) {
                    v1.cmp(v2)
                } else {
                    // Two decimal values can only be compared if they have the same precision and scale.
                    panic!("Attempt to compare decimals with unequal precision / scale")
                }
            }
            (Decimal128(v1, _, _), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Decimal128(_, _, _), _) => panic!("Attempt to compare decimal with non-decimal"),
            (Decimal256(v1, p1, s1), Decimal256(v2, p2, s2)) => {
                if p1.eq(p2) && s1.eq(s2) {
                    v1.cmp(v2)
                } else {
                    // Two decimal values can only be compared if they have the same precision and scale.
                    panic!("Attempt to compare decimals with unequal precision / scale")
                }
            }
            (Decimal256(v1, _, _), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Decimal256(_, _, _), _) => panic!("Attempt to compare decimal with non-decimal"),
            (Boolean(v1), Boolean(v2)) => v1.cmp(v2),
            (Boolean(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Boolean(_), _) => panic!("Attempt to compare boolean with non-boolean"),
            (Float32(v1), Float32(v2)) => match (v1, v2) {
                (Some(f1), Some(f2)) => f1.total_cmp(f2),
                (None, Some(_)) => Ordering::Less,
                (Some(_), None) => Ordering::Greater,
                (None, None) => Ordering::Equal,
            },
            (Float32(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Float32(_), _) => panic!("Attempt to compare f32 with non-f32"),
            (Float64(v1), Float64(v2)) => match (v1, v2) {
                (Some(f1), Some(f2)) => f1.total_cmp(f2),
                (None, Some(_)) => Ordering::Less,
                (Some(_), None) => Ordering::Greater,
                (None, None) => Ordering::Equal,
            },
            (Float64(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Float64(_), _) => panic!("Attempt to compare f64 with non-f64"),
            (Int8(v1), Int8(v2)) => v1.cmp(v2),
            (Int8(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Int8(_), _) => panic!("Attempt to compare Int8 with non-Int8"),
            (Int16(v1), Int16(v2)) => v1.cmp(v2),
            (Int16(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Int16(_), _) => panic!("Attempt to compare Int16 with non-Int16"),
            (Int32(v1), Int32(v2)) => v1.cmp(v2),
            (Int32(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Int32(_), _) => panic!("Attempt to compare Int32 with non-Int32"),
            (Int64(v1), Int64(v2)) => v1.cmp(v2),
            (Int64(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Int64(_), _) => panic!("Attempt to compare Int16 with non-Int64"),
            (UInt8(v1), UInt8(v2)) => v1.cmp(v2),
            (UInt8(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (UInt8(_), _) => panic!("Attempt to compare UInt8 with non-UInt8"),
            (UInt16(v1), UInt16(v2)) => v1.cmp(v2),
            (UInt16(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (UInt16(_), _) => panic!("Attempt to compare UInt16 with non-UInt16"),
            (UInt32(v1), UInt32(v2)) => v1.cmp(v2),
            (UInt32(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (UInt32(_), _) => panic!("Attempt to compare UInt32 with non-UInt32"),
            (UInt64(v1), UInt64(v2)) => v1.cmp(v2),
            (UInt64(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (UInt64(_), _) => panic!("Attempt to compare Int16 with non-UInt64"),
            (Utf8(v1), Utf8(v2)) => v1.cmp(v2),
            (Utf8(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Utf8(_), _) => panic!("Attempt to compare Utf8 with non-Utf8"),
            (LargeUtf8(v1), LargeUtf8(v2)) => v1.cmp(v2),
            (LargeUtf8(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (LargeUtf8(_), _) => panic!("Attempt to compare LargeUtf8 with non-LargeUtf8"),
            (Binary(v1), Binary(v2)) => v1.cmp(v2),
            (Binary(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Binary(_), _) => panic!("Attempt to compare Binary with non-Binary"),
            (FixedSizeBinary(_, v1), FixedSizeBinary(_, v2)) => v1.cmp(v2),
            (FixedSizeBinary(_, v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (FixedSizeBinary(_, _), _) => {
                panic!("Attempt to compare FixedSizeBinary with non-FixedSizeBinary")
            }
            (LargeBinary(v1), LargeBinary(v2)) => v1.cmp(v2),
            (LargeBinary(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (LargeBinary(_), _) => panic!("Attempt to compare LargeBinary with non-LargeBinary"),
            (FixedSizeList(left), FixedSizeList(right)) => {
                if left.eq(right) {
                    todo!()
                } else {
                    panic!(
                        "Attempt to compare fixed size list elements with different widths/fields"
                    )
                }
            }
            (FixedSizeList(left), Null) => {
                if left.is_null(0) {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (FixedSizeList(_), _) => {
                panic!("Attempt to compare FixedSizeList with non-FixedSizeList")
            }
            (List(_), List(_)) => todo!(),
            (List(left), Null) => {
                if left.is_null(0) {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (List(_), _) => {
                panic!("Attempt to compare List with non-List")
            }
            (LargeList(_), _) => todo!(),
            (Date32(v1), Date32(v2)) => v1.cmp(v2),
            (Date32(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Date32(_), _) => panic!("Attempt to compare Date32 with non-Date32"),
            (Date64(v1), Date64(v2)) => v1.cmp(v2),
            (Date64(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Date64(_), _) => panic!("Attempt to compare Date64 with non-Date64"),
            (Time32Second(v1), Time32Second(v2)) => v1.cmp(v2),
            (Time32Second(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Time32Second(_), _) => panic!("Attempt to compare Time32Second with non-Time32Second"),
            (Time32Millisecond(v1), Time32Millisecond(v2)) => v1.cmp(v2),
            (Time32Millisecond(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Time32Millisecond(_), _) => {
                panic!("Attempt to compare Time32Millisecond with non-Time32Millisecond")
            }
            (Time64Microsecond(v1), Time64Microsecond(v2)) => v1.cmp(v2),
            (Time64Microsecond(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Time64Microsecond(_), _) => {
                panic!("Attempt to compare Time64Microsecond with non-Time64Microsecond")
            }
            (Time64Nanosecond(v1), Time64Nanosecond(v2)) => v1.cmp(v2),
            (Time64Nanosecond(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Time64Nanosecond(_), _) => {
                panic!("Attempt to compare Time64Nanosecond with non-Time64Nanosecond")
            }
            (TimestampSecond(v1, _), TimestampSecond(v2, _)) => v1.cmp(v2),
            (TimestampSecond(v1, _), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (TimestampSecond(_, _), _) => {
                panic!("Attempt to compare TimestampSecond with non-TimestampSecond")
            }
            (TimestampMillisecond(v1, _), TimestampMillisecond(v2, _)) => v1.cmp(v2),
            (TimestampMillisecond(v1, _), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (TimestampMillisecond(_, _), _) => {
                panic!("Attempt to compare TimestampMillisecond with non-TimestampMillisecond")
            }
            (TimestampMicrosecond(v1, _), TimestampMicrosecond(v2, _)) => v1.cmp(v2),
            (TimestampMicrosecond(v1, _), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (TimestampMicrosecond(_, _), _) => {
                panic!("Attempt to compare TimestampMicrosecond with non-TimestampMicrosecond")
            }
            (TimestampNanosecond(v1, _), TimestampNanosecond(v2, _)) => v1.cmp(v2),
            (TimestampNanosecond(v1, _), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (TimestampNanosecond(_, _), _) => {
                panic!("Attempt to compare TimestampNanosecond with non-TimestampNanosecond")
            }
            (IntervalYearMonth(v1), IntervalYearMonth(v2)) => v1.cmp(v2),
            (IntervalYearMonth(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (IntervalYearMonth(_), _) => {
                panic!("Attempt to compare IntervalYearMonth with non-IntervalYearMonth")
            }
            (IntervalDayTime(v1), IntervalDayTime(v2)) => v1.cmp(v2),
            (IntervalDayTime(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (IntervalDayTime(_), _) => {
                panic!("Attempt to compare IntervalDayTime with non-IntervalDayTime")
            }
            (IntervalMonthDayNano(v1), IntervalMonthDayNano(v2)) => v1.cmp(v2),
            (IntervalMonthDayNano(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (IntervalMonthDayNano(_), _) => {
                panic!("Attempt to compare IntervalMonthDayNano with non-IntervalMonthDayNano")
            }
            (DurationSecond(v1), DurationSecond(v2)) => v1.cmp(v2),
            (DurationSecond(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (DurationSecond(_), _) => {
                panic!("Attempt to compare DurationSecond with non-DurationSecond")
            }
            (DurationMillisecond(v1), DurationMillisecond(v2)) => v1.cmp(v2),
            (DurationMillisecond(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (DurationMillisecond(_), _) => {
                panic!("Attempt to compare DurationMillisecond with non-DurationMillisecond")
            }
            (DurationMicrosecond(v1), DurationMicrosecond(v2)) => v1.cmp(v2),
            (DurationMicrosecond(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (DurationMicrosecond(_), _) => {
                panic!("Attempt to compare DurationMicrosecond with non-DurationMicrosecond")
            }
            (DurationNanosecond(v1), DurationNanosecond(v2)) => v1.cmp(v2),
            (DurationNanosecond(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (DurationNanosecond(_), _) => {
                panic!("Attempt to compare DurationNanosecond with non-DurationNanosecond")
            }
            (Struct(_arr), Struct(_arr2)) => todo!(),
            (Struct(arr), Null) => {
                if arr.is_empty() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Struct(_arr), _) => panic!("Attempt to compare Struct with non-Struct"),
            (Dictionary(_k1, _v1), Dictionary(_k2, _v2)) => todo!(),
            (Dictionary(_, v1), Null) => Self(*v1.clone()).cmp(&Self(ScalarValue::Null)),
            (Dictionary(_, _), _) => panic!("Attempt to compare Dictionary with non-Dictionary"),
            (Null, Null) => Ordering::Equal,
            (Null, _) => todo!(),
            (Union(_, _, _), Union(_, _, _)) => panic!("Attempt to compare Union with Union"),
            (Union(_, _, _), _) => panic!("Attempt to compare Union with non-Union"),
        }
    }
}

#[derive(Debug)]
struct PageRecord {
    max: OrderableScalarValue,
    page_number: u32,
}

trait BTreeMapExt<K, V> {
    fn largest_node_less(&self, key: &K) -> Option<(&K, &V)>;
}

impl<K: Ord, V> BTreeMapExt<K, V> for BTreeMap<K, V> {
    fn largest_node_less(&self, key: &K) -> Option<(&K, &V)> {
        self.range((Bound::Unbounded, Bound::Excluded(key)))
            .next_back()
    }
}

/// An in-memory structure that can quickly satisfy scalar queries using a btree of ScalarValue
#[derive(Debug)]
pub struct BTreeLookup {
    tree: BTreeMap<OrderableScalarValue, Vec<PageRecord>>,
    /// Pages where the value may be null
    null_pages: Vec<u32>,
}

impl BTreeLookup {
    fn new(tree: BTreeMap<OrderableScalarValue, Vec<PageRecord>>, null_pages: Vec<u32>) -> Self {
        Self { tree, null_pages }
    }

    fn all_page_ids(&self) -> Vec<u32> {
        let mut ids = self
            .tree
            .iter()
            .flat_map(|(_, pages)| pages)
            .map(|page| page.page_number)
            .collect::<Vec<_>>();
        ids.dedup();
        ids
    }

    // All pages that could have a value equal to val
    fn pages_eq(&self, query: &OrderableScalarValue) -> Vec<u32> {
        self.pages_between((Bound::Included(query), Bound::Excluded(query)))
    }

    // All pages that could have a value equal to one of the values
    fn pages_in(&self, values: impl IntoIterator<Item = OrderableScalarValue>) -> Vec<u32> {
        let page_lists = values
            .into_iter()
            .map(|val| self.pages_eq(&val))
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

    // All pages that could have a value in the range
    fn pages_between(
        &self,
        range: (Bound<&OrderableScalarValue>, Bound<&OrderableScalarValue>),
    ) -> Vec<u32> {
        // We need to grab a little bit left of the given range because the query might be 7
        // and the first page might be something like 5-10.
        let lower_bound = match range.0 {
            Bound::Unbounded => Bound::Unbounded,
            // It doesn't matter if the bound is exclusive or inclusive.  We are going to grab
            // the first node whose min is strictly less than the given bound.  Then we grab
            // all nodes greater than or equal to that
            //
            // We have to peek a bit to the left because we might have something like a lower
            // bound of 7 and there is a page [5-10] we want to search for.
            Bound::Included(lower) => self
                .tree
                .largest_node_less(lower)
                .map(|val| Bound::Included(val.0))
                .unwrap_or(Bound::Unbounded),
            Bound::Excluded(lower) => self
                .tree
                .largest_node_less(lower)
                .map(|val| Bound::Included(val.0))
                .unwrap_or(Bound::Unbounded),
        };
        let upper_bound = match range.1 {
            Bound::Unbounded => Bound::Unbounded,
            Bound::Included(upper) => Bound::Included(upper),
            // Even if the upper bound is excluded we need to include it on an [x, x) query.  This is because the
            // query might be [x, x).  Our lower bound might find some [a-x] bucket and we still
            // want to include any [x, z] bucket.
            //
            // We could be slightly more accurate here and only include the upper bound if the lower bound
            // is defined, inclusive, and equal to the upper bound.  However, let's keep it simple for now.  This
            // should only affect the probably rare case that our query is a true range query and the value
            // matches an upper bound.  This will all be moot if/when we merge pages.
            Bound::Excluded(upper) => Bound::Included(upper),
        };
        let candidates = self
            .tree
            .range((lower_bound, upper_bound))
            .flat_map(|val| val.1);
        match lower_bound {
            Bound::Unbounded => candidates.map(|val| val.page_number).collect(),
            Bound::Included(lower_bound) => candidates
                .filter(|val| val.max.cmp(lower_bound) != Ordering::Less)
                .map(|val| val.page_number)
                .collect(),
            Bound::Excluded(lower_bound) => candidates
                .filter(|val| val.max.cmp(lower_bound) == Ordering::Greater)
                .map(|val| val.page_number)
                .collect(),
        }
    }

    fn pages_null(&self) -> Vec<u32> {
        self.null_pages.clone()
    }
}

/// A btree index satisfies scalar queries using a b tree
///
/// The upper layers of the btree are expected to be cached and, when unloaded,
/// are stored in a btree structure in memory.  The leaves of the btree are left
/// to be searched by some other kind of index (currently a flat search).
///
/// This strikes a balance between an expensive memory structure containing all
/// of the values and an expensive disk structure that can't be efficiently searched.
///
/// For example, given 1Bi values we can store 256Ki leaves of size 4Ki.  We only
/// need memory space for 256Ki leaves (depends on the data type but usually a few MiB
/// at most) and can narrow our search to 4Ki values.
///
/// Note: this is very similar to the IVF index except we store the IVF part in a btree
/// for faster lookup
#[derive(Clone, Debug)]
pub struct BTreeIndex {
    page_lookup: Arc<BTreeLookup>,
    store: Arc<dyn IndexStore>,
    sub_index: Arc<dyn BTreeSubIndex>,
}

impl BTreeIndex {
    fn new(
        tree: BTreeMap<OrderableScalarValue, Vec<PageRecord>>,
        null_pages: Vec<u32>,
        store: Arc<dyn IndexStore>,
        sub_index: Arc<dyn BTreeSubIndex>,
    ) -> Self {
        let page_lookup = Arc::new(BTreeLookup::new(tree, null_pages));
        Self {
            page_lookup,
            store,
            sub_index,
        }
    }

    async fn search_page(
        &self,
        query: &ScalarQuery,
        page_number: u32,
        index_reader: Arc<dyn IndexReader>,
    ) -> Result<UInt64Array> {
        let serialized_page = index_reader.read_record_batch(page_number).await?;
        let subindex = self.sub_index.load_subindex(serialized_page).await?;
        // TODO: If this is an IN query we can perhaps simplify the subindex query by restricting it to the
        // values that might be in the page.  E.g. if we are searching for X IN [5, 3, 7] and five is in pages
        // 1 and 2 and three is in page 2 and seven is in pages 8 and 9 then when we search page 2 we only need
        // to search for X IN [5, 3]
        subindex.search(query).await
    }

    fn try_from_serialized(data: RecordBatch, store: Arc<dyn IndexStore>) -> Result<Self> {
        let mut map = BTreeMap::<OrderableScalarValue, Vec<PageRecord>>::new();
        let mut null_pages = Vec::<u32>::new();

        if data.num_rows() == 0 {
            return Err(Error::Internal {
                message: "attempt to load btree index from empty stats batch".into(),
                location: location!(),
            });
        }

        let mins = data.column(0);
        let maxs = data.column(1);
        let null_counts = data
            .column(2)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let page_numbers = data
            .column(3)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();

        for idx in 0..data.num_rows() {
            let min = OrderableScalarValue(ScalarValue::try_from_array(&mins, idx)?);
            let max = OrderableScalarValue(ScalarValue::try_from_array(&maxs, idx)?);
            let null_count = null_counts.values()[idx];
            let page_number = page_numbers.values()[idx];

            map.entry(min)
                .or_default()
                .push(PageRecord { max, page_number });
            if null_count > 0 {
                null_pages.push(page_number);
            }
        }

        let last_max = ScalarValue::try_from_array(&maxs, data.num_rows() - 1)?;
        map.entry(OrderableScalarValue(last_max)).or_default();

        let data_type = mins.data_type();

        // TODO: Support other page types?
        let sub_index = Arc::new(FlatIndexMetadata::new(data_type.clone()));

        Ok(Self::new(map, null_pages, store, sub_index))
    }

    /// Create a stream of all the data in the index, in the same format used to train the index
    async fn into_data_stream(self) -> Result<impl RecordBatchStream> {
        let reader = self.store.open_index_file(BTREE_PAGES_NAME).await?;
        let pages = self.page_lookup.all_page_ids();
        let schema = self.sub_index.schema().clone();
        let batches = IndexReaderStream {
            reader,
            pages,
            idx: 0,
        }
        .map(|fut| fut.map_err(DataFusionError::from))
        .buffered(num_cpus::get())
        .boxed();
        Ok(RecordBatchStreamAdapter::new(schema, batches))
    }
}

fn wrap_bound(bound: &Bound<ScalarValue>) -> Bound<OrderableScalarValue> {
    match bound {
        Bound::Unbounded => Bound::Unbounded,
        Bound::Included(val) => Bound::Included(OrderableScalarValue(val.clone())),
        Bound::Excluded(val) => Bound::Excluded(OrderableScalarValue(val.clone())),
    }
}

fn serialize_with_display<T: Display, S: Serializer>(
    value: &Option<T>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    if let Some(value) = value {
        serializer.collect_str(value)
    } else {
        serializer.collect_str("N/A")
    }
}

#[derive(Serialize)]
struct BTreeStatistics {
    #[serde(serialize_with = "serialize_with_display")]
    min: Option<OrderableScalarValue>,
    #[serde(serialize_with = "serialize_with_display")]
    max: Option<OrderableScalarValue>,
    num_pages: u32,
}

#[async_trait]
impl Index for BTreeIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn index_type(&self) -> IndexType {
        IndexType::Scalar
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let min = self
            .page_lookup
            .tree
            .first_key_value()
            .map(|(k, _)| k.clone());
        let max = self
            .page_lookup
            .tree
            .last_key_value()
            .map(|(k, _)| k.clone());
        serde_json::to_value(&BTreeStatistics {
            num_pages: self.page_lookup.tree.len() as u32,
            min,
            max,
        })
        .map_err(|err| err.into())
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        let mut frag_ids = RoaringBitmap::default();

        let sub_index_reader = self.store.open_index_file(BTREE_PAGES_NAME).await?;
        for page_number in self.page_lookup.all_page_ids() {
            let serialized = sub_index_reader.read_record_batch(page_number).await?;
            let page = self.sub_index.load_subindex(serialized).await?;
            frag_ids |= page.calculate_included_frags().await?;
        }

        Ok(frag_ids)
    }
}

#[async_trait]
impl ScalarIndex for BTreeIndex {
    async fn search(&self, query: &ScalarQuery) -> Result<UInt64Array> {
        let pages = match query {
            ScalarQuery::Equals(val) => self
                .page_lookup
                .pages_eq(&OrderableScalarValue(val.clone())),
            ScalarQuery::Range(start, end) => self
                .page_lookup
                .pages_between((wrap_bound(start).as_ref(), wrap_bound(end).as_ref())),
            ScalarQuery::IsIn(values) => self
                .page_lookup
                .pages_in(values.iter().map(|val| OrderableScalarValue(val.clone()))),
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

    async fn remap(
        &self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        // Remap and write the pages
        let mut sub_index_file = dest_store
            .new_index_file(BTREE_PAGES_NAME, self.sub_index.schema().clone())
            .await?;

        let sub_index_reader = self.store.open_index_file(BTREE_PAGES_NAME).await?;

        for page_number in self.page_lookup.all_page_ids() {
            let old_serialized = sub_index_reader.read_record_batch(page_number).await?;
            let remapped = self
                .sub_index
                .remap_subindex(old_serialized, mapping)
                .await?;
            sub_index_file.write_record_batch(remapped).await?;
        }

        sub_index_file.finish().await?;

        // Copy the lookup file as-is
        self.store
            .copy_index_file(BTREE_LOOKUP_NAME, dest_store)
            .await
    }

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        // Merge the existing index data with the new data and then retrain the index on the merged stream
        let merged_data_source = Box::new(BTreeUpdater::new(self.clone(), new_data));
        train_btree_index(merged_data_source, self.sub_index.as_ref(), dest_store).await
    }
}

struct BatchStats {
    min: ScalarValue,
    max: ScalarValue,
    null_count: u32,
}

// See https://github.com/apache/arrow-datafusion/issues/8031 for the underlying issue.  We use
// MinAccumulator / MaxAccumulator to retrieve the min/max values and these are unreliable in the
// presence of NaN
fn check_for_nan(value: ScalarValue) -> Result<ScalarValue> {
    match value {
        ScalarValue::Float32(Some(val)) if val.is_nan() => Err(Error::NotSupported {
            source: "Scalar indices cannot currently be created on columns with NaN values".into(),
            location: location!(),
        }),
        ScalarValue::Float64(Some(val)) if val.is_nan() => Err(Error::NotSupported {
            source: "Scalar indices cannot currently be created on columns with NaN values".into(),
            location: location!(),
        }),
        _ => Ok(value),
    }
}

fn min_val(array: &Arc<dyn Array>) -> Result<ScalarValue> {
    let mut acc = MinAccumulator::try_new(array.data_type())?;
    acc.update_batch(&[array.clone()])?;
    check_for_nan(acc.evaluate()?)
}

fn max_val(array: &Arc<dyn Array>) -> Result<ScalarValue> {
    let mut acc = MaxAccumulator::try_new(array.data_type())?;
    acc.update_batch(&[array.clone()])?;
    check_for_nan(acc.evaluate()?)
}

fn analyze_batch(batch: &RecordBatch) -> Result<BatchStats> {
    let values = batch.column(0);
    let min = min_val(values)?;
    let max = max_val(values)?;
    Ok(BatchStats {
        min,
        max,
        null_count: values.null_count() as u32,
    })
}

/// A trait that must be implemented by anything that wishes to act as a btree subindex
#[async_trait]
pub trait BTreeSubIndex: Debug + Send + Sync {
    /// Trains the subindex on a single batch of data and serializes it to Arrow
    async fn train(&self, batch: RecordBatch) -> Result<RecordBatch>;

    /// Deserialize a subindex from Arrow
    async fn load_subindex(&self, serialized: RecordBatch) -> Result<Arc<dyn ScalarIndex>>;

    /// Retrieve the data used to originally train this page
    ///
    /// In order to perform an update we need to merge the old data in with the new data which
    /// means we need to access the new data.  Right now this is convenient for flat indices but
    /// we may need to take a different approach if we ever decide to use a sub-index other than
    /// flat
    async fn retrieve_data(&self, serialized: RecordBatch) -> Result<RecordBatch>;

    /// The schema of the subindex when serialized to Arrow
    fn schema(&self) -> &Arc<Schema>;

    /// Given a serialized page, deserialize it, remap the row ids, and re-serialize it
    async fn remap_subindex(
        &self,
        serialized: RecordBatch,
        mapping: &HashMap<u64, Option<u64>>,
    ) -> Result<RecordBatch>;
}

struct EncodedBatch {
    stats: BatchStats,
    page_number: u32,
}

async fn train_btree_page(
    batch: RecordBatch,
    batch_idx: u32,
    sub_index_trainer: &dyn BTreeSubIndex,
    writer: &mut dyn IndexWriter,
) -> Result<EncodedBatch> {
    let stats = analyze_batch(&batch)?;
    let trained = sub_index_trainer.train(batch).await?;
    writer.write_record_batch(trained).await?;
    Ok(EncodedBatch {
        stats,
        page_number: batch_idx,
    })
}

fn btree_stats_as_batch(stats: Vec<EncodedBatch>) -> Result<RecordBatch> {
    let mins = ScalarValue::iter_to_array(stats.iter().map(|stat| stat.stats.min.clone()))?;
    let maxs = ScalarValue::iter_to_array(stats.iter().map(|stat| stat.stats.max.clone()))?;
    let null_counts = UInt32Array::from_iter_values(stats.iter().map(|stat| stat.stats.null_count));
    let page_numbers = UInt32Array::from_iter_values(stats.iter().map(|stat| stat.page_number));

    let schema = Arc::new(Schema::new(vec![
        Field::new("min", mins.data_type().clone(), false),
        Field::new("max", maxs.data_type().clone(), false),
        Field::new("null_count", null_counts.data_type().clone(), false),
        Field::new("page_idx", page_numbers.data_type().clone(), false),
    ]));

    let columns = vec![
        mins,
        maxs,
        Arc::new(null_counts) as Arc<dyn Array>,
        Arc::new(page_numbers) as Arc<dyn Array>,
    ];

    Ok(RecordBatch::try_new(schema, columns)?)
}

#[async_trait]
pub trait BtreeTrainingSource: Send {
    /// Returns a stream of batches, ordered by the value column (in ascending order)
    ///
    /// Each batch should have chunk_size rows
    ///
    /// The schema for the batch is slightly flexible.
    /// The first column may have any name or type, these are the values to index
    /// The second column must be the row ids which must be UInt64Type
    async fn scan_ordered_chunks(
        self: Box<Self>,
        chunk_size: u32,
    ) -> Result<SendableRecordBatchStream>;
}

/// Train a btree index from a stream of sorted page-size batches of values and row ids
///
/// Note: This is likely to change.  It is unreasonable to expect the caller to do the sorting
/// and re-chunking into page-size batches.  This is left for simplicity as this feature is still
/// a work in progress
pub async fn train_btree_index(
    data_source: Box<dyn BtreeTrainingSource + Send>,
    sub_index_trainer: &dyn BTreeSubIndex,
    index_store: &dyn IndexStore,
) -> Result<()> {
    let mut sub_index_file = index_store
        .new_index_file(BTREE_PAGES_NAME, sub_index_trainer.schema().clone())
        .await?;
    let mut encoded_batches = Vec::new();
    let mut batch_idx = 0;
    let mut batches_source = data_source.scan_ordered_chunks(4096).await?;
    while let Some(batch) = batches_source.try_next().await? {
        debug_assert_eq!(batch.num_columns(), 2);
        debug_assert_eq!(*batch.column(1).data_type(), DataType::UInt64);
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

/// A source of training data created by merging existing data with new data
struct BTreeUpdater {
    index: BTreeIndex,
    new_data: SendableRecordBatchStream,
}

impl BTreeUpdater {
    fn new(index: BTreeIndex, new_data: SendableRecordBatchStream) -> Self {
        Self { index, new_data }
    }
}

impl BTreeUpdater {
    fn into_old_input(index: BTreeIndex) -> Arc<dyn ExecutionPlan> {
        let schema = index.sub_index.schema().clone();
        let batches = index.into_data_stream().into_stream().try_flatten().boxed();
        let stream = Box::pin(RecordBatchStreamAdapter::new(schema, batches));
        Arc::new(OneShotExec::new(stream))
    }
}

#[async_trait]
impl BtreeTrainingSource for BTreeUpdater {
    async fn scan_ordered_chunks(
        self: Box<Self>,
        chunk_size: u32,
    ) -> Result<SendableRecordBatchStream> {
        let new_input = Arc::new(OneShotExec::new(self.new_data));
        let old_input = Self::into_old_input(self.index);
        debug_assert_eq!(
            old_input.schema().all_fields().len(),
            new_input.schema().all_fields().len()
        );
        let sort_expr = PhysicalSortExpr {
            expr: Arc::new(Column::new("values", 0)),
            options: SortOptions {
                descending: false,
                nulls_first: true,
            },
        };
        // The UnionExec creates multiple partitions but the SortPreservingMergeExec merges
        // them back into a single partition.
        let all_data = Arc::new(UnionExec::new(vec![old_input, new_input]));
        let ordered = Arc::new(SortPreservingMergeExec::new(vec![sort_expr], all_data));
        let unchunked = execute_plan(
            ordered,
            LanceExecutionOptions {
                use_spilling: true,
                ..Default::default()
            },
        )?;
        Ok(chunk_concat_stream(unchunked, chunk_size as usize))
    }
}

/// A stream that reads the original training data back out of the index
///
/// This is used for updating the index
struct IndexReaderStream {
    reader: Arc<dyn IndexReader>,
    pages: Vec<u32>,
    idx: usize,
}

impl Stream for IndexReaderStream {
    type Item = BoxFuture<'static, Result<RecordBatch>>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let idx = this.idx;
        if idx >= this.pages.len() {
            return std::task::Poll::Ready(None);
        }
        let page_number = this.pages[idx];
        this.idx += 1;
        let reader_copy = this.reader.clone();
        let read_task = async move { reader_copy.read_record_batch(page_number).await }.boxed();
        std::task::Poll::Ready(Some(read_task))
    }
}
