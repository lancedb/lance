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
    cmp::Ordering,
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
use lance_core::{Error, Result};
use snafu::{location, Location};

use super::{
    flat::FlatIndexLoader, IndexReader, IndexStore, IndexWriter, ScalarIndex, ScalarQuery,
};

const BTREE_LOOKUP_NAME: &str = "page_lookup.lance";
const BTREE_PAGES_NAME: &str = "page_data.lance";

/// Wraps a ScalarValue and implements Ord (ScalarValue only implements PartialOrd)
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
            (Fixedsizelist(_v1, t1, l1), Fixedsizelist(_v2, t2, l2)) => {
                if t1.eq(t2) && l1.eq(l2) {
                    todo!()
                } else {
                    panic!(
                        "Attempt to compare fixed size list elements with different widths/fields"
                    )
                }
            }
            (Fixedsizelist(v1, _, _), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Fixedsizelist(_, _, _), _) => {
                panic!("Attempt to compare Fixedsizelist with non-Fixedsizelist")
            }
            (List(_, _), List(_, _)) => todo!(),
            (List(v1, _), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (List(_, _), _) => {
                panic!("Attempt to compare List with non-List")
            }
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
            (Struct(_v1, _t1), Struct(_v2, _t2)) => todo!(),
            (Struct(v1, _), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Struct(_, _), _) => panic!("Attempt to compare Struct with non-Struct"),
            (Dictionary(_k1, _v1), Dictionary(_k2, _v2)) => todo!(),
            (Dictionary(_, v1), Null) => Self(*v1.clone()).cmp(&Self(ScalarValue::Null)),
            (Dictionary(_, _), _) => panic!("Attempt to compare Dictionary with non-Dictionary"),
            (Null, Null) => Ordering::Equal,
            (Null, _) => todo!(),
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

/// A trait that must be implemented by a btree sub-index
///
/// A sub-index must be capable of indexing a single page of data.  We represent
/// pages as a single record batch.
#[async_trait]
pub trait SubIndexLoader: std::fmt::Debug + Send + Sync {
    async fn load_subindex(
        &self,
        page_number: u32,
        index_reader: Arc<dyn IndexReader>,
    ) -> Result<Arc<dyn ScalarIndex>>;
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
#[derive(Debug)]
pub struct BTreeIndex {
    page_lookup: BTreeLookup,
    store: Arc<dyn IndexStore>,
    sub_index: Arc<dyn SubIndexLoader>,
}

impl BTreeIndex {
    fn new(
        tree: BTreeMap<OrderableScalarValue, Vec<PageRecord>>,
        null_pages: Vec<u32>,
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
        page_number: u32,
        index_reader: Arc<dyn IndexReader>,
    ) -> Result<UInt64Array> {
        let subindex = self
            .sub_index
            .load_subindex(page_number, index_reader)
            .await?;
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

        // TODO: Support other page types?
        let sub_index = Arc::new(FlatIndexLoader {});

        Ok(Self::new(map, null_pages, store, sub_index))
    }
}

fn wrap_bound(bound: &Bound<ScalarValue>) -> Bound<OrderableScalarValue> {
    match bound {
        Bound::Unbounded => Bound::Unbounded,
        Bound::Included(val) => Bound::Included(OrderableScalarValue(val.clone())),
        Bound::Excluded(val) => Bound::Excluded(OrderableScalarValue(val.clone())),
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

#[async_trait]
pub trait SubIndexTrainer: Send + Sync {
    fn schema(&self) -> &Arc<Schema>;
    async fn train(&self, batch: RecordBatch, writer: &mut dyn IndexWriter) -> Result<u64>;
}

struct EncodedBatch {
    stats: BatchStats,
    page_number: u32,
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

/// Train a btree index from a stream of sorted page-size batches of values and row ids
///
/// Note: This is likely to change.  It is unreasonable to expect the caller to do the sorting
/// and re-chunking into page-size batches.  This is left for simplicity as this feature is still
/// a work in progress
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
