// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    any::Any,
    cmp::Ordering,
    collections::{BTreeMap, BinaryHeap, HashMap},
    fmt::{Debug, Display},
    ops::Bound,
    sync::Arc,
};

use super::{
    flat::FlatIndexMetadata, AnyQuery, IndexReader, IndexStore, IndexWriter, MetricsCollector,
    SargableQuery, ScalarIndex, SearchResult,
};
use crate::frag_reuse::FragReuseIndex;
use crate::{Index, IndexType};
use arrow_array::{new_empty_array, Array, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Schema, SortOptions};
use async_trait::async_trait;
use datafusion::physical_plan::{
    sorts::sort_preserving_merge::SortPreservingMergeExec, stream::RecordBatchStreamAdapter,
    union::UnionExec, ExecutionPlan, RecordBatchStream, SendableRecordBatchStream,
};
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_physical_expr::{expressions::Column, LexOrdering, PhysicalSortExpr};
use deepsize::{Context, DeepSizeOf};
use futures::{
    future::BoxFuture,
    stream::{self},
    FutureExt, Stream, StreamExt, TryFutureExt, TryStreamExt,
};
use lance_core::{
    utils::{
        mask::RowIdTreeMap,
        tokio::get_num_compute_intensive_cpus,
        tracing::{IO_TYPE_LOAD_SCALAR_PART, TRACE_IO_EVENTS},
    },
    Error, Result,
};
use lance_datafusion::{
    chunker::chunk_concat_stream,
    exec::{execute_plan, LanceExecutionOptions, OneShotExec},
};
use log::debug;
use moka::sync::Cache;
use roaring::RoaringBitmap;
use serde::{Serialize, Serializer};
use snafu::location;
use tracing::info;

const BTREE_LOOKUP_NAME: &str = "page_lookup.lance";
const BTREE_PAGES_NAME: &str = "page_data.lance";
pub const DEFAULT_BTREE_BATCH_SIZE: u64 = 4096;
const BATCH_SIZE_META_KEY: &str = "batch_size";

static CACHE_SIZE: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| {
    std::env::var("LANCE_BTREE_CACHE_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512 * 1024 * 1024)
});

/// Wraps a ScalarValue and implements Ord (ScalarValue only implements PartialOrd)
#[derive(Clone, Debug)]
pub struct OrderableScalarValue(pub ScalarValue);

impl DeepSizeOf for OrderableScalarValue {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        // deepsize and size both factor in the size of the ScalarValue
        self.0.size() - std::mem::size_of::<ScalarValue>()
    }
}

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
            (Float16(v1), Float16(v2)) => match (v1, v2) {
                (Some(f1), Some(f2)) => f1.total_cmp(f2),
                (None, Some(_)) => Ordering::Less,
                (Some(_), None) => Ordering::Greater,
                (None, None) => Ordering::Equal,
            },
            (Float16(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Float16(_), _) => panic!("Attempt to compare f16 with non-f16"),
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
            (Utf8(v1) | Utf8View(v1) | LargeUtf8(v1), Utf8(v2) | Utf8View(v2) | LargeUtf8(v2)) => {
                v1.cmp(v2)
            }
            (Utf8(v1) | Utf8View(v1) | LargeUtf8(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Utf8(_) | Utf8View(_) | LargeUtf8(_), _) => {
                panic!("Attempt to compare Utf8 with non-Utf8")
            }
            (
                Binary(v1) | LargeBinary(v1) | BinaryView(v1),
                Binary(v2) | LargeBinary(v2) | BinaryView(v2),
            ) => v1.cmp(v2),
            (Binary(v1) | LargeBinary(v1) | BinaryView(v1), Null) => {
                if v1.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Binary(_) | LargeBinary(_) | BinaryView(_), _) => {
                panic!("Attempt to compare Binary with non-Binary")
            }
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
            (Map(_), Map(_)) => todo!(),
            (Map(left), Null) => {
                if left.is_null(0) {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }
            (Map(_), _) => {
                panic!("Attempt to compare Map with non-Map")
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
            // What would a btree of unions even look like?  May not be possible.
            (Union(_, _, _), _) => todo!("Support for union scalars"),
            (Null, Null) => Ordering::Equal,
            (Null, _) => todo!(),
        }
    }
}

#[derive(Debug, DeepSizeOf, PartialEq, Eq)]
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
#[derive(Debug, DeepSizeOf, PartialEq, Eq)]
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
        if query.0.is_null() {
            self.pages_null()
        } else {
            self.pages_between((Bound::Included(query), Bound::Excluded(query)))
        }
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

        match (lower_bound, upper_bound) {
            (Bound::Excluded(lower), Bound::Excluded(upper))
            | (Bound::Excluded(lower), Bound::Included(upper))
            | (Bound::Included(lower), Bound::Excluded(upper)) => {
                // It's not really clear what (Included(5), Excluded(5)) would mean so we
                // interpret it as an empty range which matches rust's BTreeMap behavior
                if lower >= upper {
                    return vec![];
                }
            }
            (Bound::Included(lower), Bound::Included(upper)) => {
                if lower > upper {
                    return vec![];
                }
            }
            _ => {}
        }

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

// Caches btree pages in memory
#[derive(Debug)]
struct BTreeCache(Cache<u32, Arc<dyn ScalarIndex>>);

impl DeepSizeOf for BTreeCache {
    fn deep_size_of_children(&self, _: &mut Context) -> usize {
        self.0.iter().map(|(_, v)| v.deep_size_of()).sum()
    }
}

// We only need to open a file reader for pages if we need to load a page.  If all
// pages are cached we don't open it.  If we do open it we should only open it once.
#[derive(Clone)]
struct LazyIndexReader {
    index_reader: Arc<tokio::sync::Mutex<Option<Arc<dyn IndexReader>>>>,
    store: Arc<dyn IndexStore>,
}

impl LazyIndexReader {
    fn new(store: Arc<dyn IndexStore>) -> Self {
        Self {
            index_reader: Arc::new(tokio::sync::Mutex::new(None)),
            store,
        }
    }

    async fn get(&self) -> Result<Arc<dyn IndexReader>> {
        let mut reader = self.index_reader.lock().await;
        if reader.is_none() {
            let index_reader = self.store.open_index_file(BTREE_PAGES_NAME).await?;
            *reader = Some(index_reader);
        }
        Ok(reader.as_ref().unwrap().clone())
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
#[derive(Clone, Debug, DeepSizeOf)]
pub struct BTreeIndex {
    page_lookup: Arc<BTreeLookup>,
    page_cache: Arc<BTreeCache>,
    store: Arc<dyn IndexStore>,
    sub_index: Arc<dyn BTreeSubIndex>,
    batch_size: u64,
    fri: Option<Arc<FragReuseIndex>>,
}

impl BTreeIndex {
    fn new(
        tree: BTreeMap<OrderableScalarValue, Vec<PageRecord>>,
        null_pages: Vec<u32>,
        store: Arc<dyn IndexStore>,
        sub_index: Arc<dyn BTreeSubIndex>,
        batch_size: u64,
        fri: Option<Arc<FragReuseIndex>>,
    ) -> Self {
        let page_lookup = Arc::new(BTreeLookup::new(tree, null_pages));
        let page_cache = Arc::new(BTreeCache(
            Cache::builder()
                .max_capacity(*CACHE_SIZE)
                .weigher(|_, v: &Arc<dyn ScalarIndex>| v.deep_size_of() as u32)
                .build(),
        ));
        Self {
            page_lookup,
            page_cache,
            store,
            sub_index,
            batch_size,
            fri,
        }
    }

    async fn lookup_page(
        &self,
        page_number: u32,
        index_reader: LazyIndexReader,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn ScalarIndex>> {
        if let Some(cached) = self.page_cache.0.get(&page_number) {
            return Ok(cached);
        }
        metrics.record_part_load();
        info!(target: TRACE_IO_EVENTS, r#type=IO_TYPE_LOAD_SCALAR_PART, index_type="btree", part_id=page_number);
        let index_reader = index_reader.get().await?;
        let mut serialized_page = index_reader
            .read_record_batch(page_number as u64, self.batch_size)
            .await?;
        if let Some(fri_ref) = self.fri.as_ref() {
            serialized_page = fri_ref.remap_row_ids_record_batch(serialized_page, 1)?;
        }
        let subindex = self.sub_index.load_subindex(serialized_page).await?;
        self.page_cache.0.insert(page_number, subindex.clone());
        Ok(subindex)
    }

    async fn search_page(
        &self,
        query: &SargableQuery,
        page_number: u32,
        index_reader: LazyIndexReader,
        metrics: &dyn MetricsCollector,
    ) -> Result<RowIdTreeMap> {
        let subindex = self.lookup_page(page_number, index_reader, metrics).await?;
        // TODO: If this is an IN query we can perhaps simplify the subindex query by restricting it to the
        // values that might be in the page.  E.g. if we are searching for X IN [5, 3, 7] and five is in pages
        // 1 and 2 and three is in page 2 and seven is in pages 8 and 9 then when we search page 2 we only need
        // to search for X IN [5, 3]
        match subindex.search(query, metrics).await? {
            SearchResult::Exact(map) => Ok(map),
            _ => Err(Error::Internal {
                message: "BTree sub-indices need to return exact results".to_string(),
                location: location!(),
            }),
        }
    }

    fn try_from_serialized(
        data: RecordBatch,
        store: Arc<dyn IndexStore>,
        batch_size: u64,
        fri: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let mut map = BTreeMap::<OrderableScalarValue, Vec<PageRecord>>::new();
        let mut null_pages = Vec::<u32>::new();

        if data.num_rows() == 0 {
            let data_type = data.column(0).data_type().clone();
            let sub_index = Arc::new(FlatIndexMetadata::new(data_type));
            return Ok(Self::new(
                map, null_pages, store, sub_index, batch_size, fri,
            ));
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

            // If the page is entirely null don't even bother putting it in the tree
            if !max.0.is_null() {
                map.entry(min)
                    .or_default()
                    .push(PageRecord { max, page_number });
            }

            if null_count > 0 {
                null_pages.push(page_number);
            }
        }

        let last_max = ScalarValue::try_from_array(&maxs, data.num_rows() - 1)?;
        map.entry(OrderableScalarValue(last_max)).or_default();

        let data_type = mins.data_type();

        // TODO: Support other page types?
        let sub_index = Arc::new(FlatIndexMetadata::new(data_type.clone()));

        Ok(Self::new(
            map, null_pages, store, sub_index, batch_size, fri,
        ))
    }

    /// Create a stream of all the data in the index, in the same format used to train the index
    async fn into_data_stream(self) -> Result<impl RecordBatchStream> {
        let reader = self.store.open_index_file(BTREE_PAGES_NAME).await?;
        let schema = self.sub_index.schema().clone();
        let reader_stream = IndexReaderStream::new(reader, self.batch_size).await;
        let batches = reader_stream
            .map(|fut| fut.map_err(DataFusionError::from))
            .buffered(self.store.io_parallelism())
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

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn crate::vector::VectorIndex>> {
        Err(Error::NotSupported {
            source: "BTreeIndex is not vector index".into(),
            location: location!(),
        })
    }

    async fn prewarm(&self) -> Result<()> {
        // TODO: BTree can (and should) support pre-warming by loading the pages into memory
        Ok(())
    }

    fn index_type(&self) -> IndexType {
        IndexType::BTree
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
        let mut reader_stream = IndexReaderStream::new(sub_index_reader, self.batch_size)
            .await
            .buffered(self.store.io_parallelism());
        while let Some(serialized) = reader_stream.try_next().await? {
            let page = self.sub_index.load_subindex(serialized).await?;
            frag_ids |= page.calculate_included_frags().await?;
        }

        Ok(frag_ids)
    }
}

#[async_trait]
impl ScalarIndex for BTreeIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let query = query.as_any().downcast_ref::<SargableQuery>().unwrap();
        let pages = match query {
            SargableQuery::Equals(val) => self
                .page_lookup
                .pages_eq(&OrderableScalarValue(val.clone())),
            SargableQuery::Range(start, end) => self
                .page_lookup
                .pages_between((wrap_bound(start).as_ref(), wrap_bound(end).as_ref())),
            SargableQuery::IsIn(values) => self
                .page_lookup
                .pages_in(values.iter().map(|val| OrderableScalarValue(val.clone()))),
            SargableQuery::FullTextSearch(_) => return Err(Error::invalid_input(
                "full text search is not supported for BTree index, build a inverted index for it",
                location!(),
            )),
            SargableQuery::IsNull() => self.page_lookup.pages_null(),
        };
        let lazy_index_reader = LazyIndexReader::new(self.store.clone());
        let page_tasks = pages
            .into_iter()
            .map(|page_index| {
                self.search_page(query, page_index, lazy_index_reader.clone(), metrics)
                    .boxed()
            })
            .collect::<Vec<_>>();
        debug!("Searching {} btree pages", page_tasks.len());
        let row_ids = stream::iter(page_tasks)
            // I/O and compute mixed here but important case is index in cache so
            // use compute intensive thread count
            .buffered(get_num_compute_intensive_cpus())
            .try_collect::<RowIdTreeMap>()
            .await?;
        Ok(SearchResult::Exact(row_ids))
    }

    fn can_answer_exact(&self, _: &dyn AnyQuery) -> bool {
        true
    }

    async fn load(
        store: Arc<dyn IndexStore>,
        fri: Option<Arc<FragReuseIndex>>,
    ) -> Result<Arc<Self>> {
        let page_lookup_file = store.open_index_file(BTREE_LOOKUP_NAME).await?;
        let num_rows_in_lookup = page_lookup_file.num_rows();
        let serialized_lookup = page_lookup_file
            .read_range(0..num_rows_in_lookup, None)
            .await?;
        let file_schema = page_lookup_file.schema();
        let batch_size = file_schema
            .metadata
            .get(BATCH_SIZE_META_KEY)
            .map(|bs| bs.parse().unwrap_or(DEFAULT_BTREE_BATCH_SIZE))
            .unwrap_or(DEFAULT_BTREE_BATCH_SIZE);
        Ok(Arc::new(Self::try_from_serialized(
            serialized_lookup,
            store,
            batch_size,
            fri,
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
        let mut reader_stream = IndexReaderStream::new(sub_index_reader, self.batch_size)
            .await
            .buffered(self.store.io_parallelism());
        while let Some(serialized) = reader_stream.try_next().await? {
            let remapped = self.sub_index.remap_subindex(serialized, mapping).await?;
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
        train_btree_index(
            merged_data_source,
            self.sub_index.as_ref(),
            dest_store,
            DEFAULT_BTREE_BATCH_SIZE as u32,
        )
        .await
    }
}

struct BatchStats {
    min: ScalarValue,
    max: ScalarValue,
    null_count: u32,
}

fn analyze_batch(batch: &RecordBatch) -> Result<BatchStats> {
    let values = batch.column(0);
    if values.is_empty() {
        return Err(Error::Internal {
            message: "received an empty batch in btree training".to_string(),
            location: location!(),
        });
    }
    let min = ScalarValue::try_from_array(&values, 0).map_err(|e| Error::Internal {
        message: format!("failed to get min value from batch: {}", e),
        location: location!(),
    })?;
    let max =
        ScalarValue::try_from_array(&values, values.len() - 1).map_err(|e| Error::Internal {
            message: format!("failed to get max value from batch: {}", e),
            location: location!(),
        })?;

    Ok(BatchStats {
        min,
        max,
        null_count: values.null_count() as u32,
    })
}

/// A trait that must be implemented by anything that wishes to act as a btree subindex
#[async_trait]
pub trait BTreeSubIndex: Debug + Send + Sync + DeepSizeOf {
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

fn btree_stats_as_batch(stats: Vec<EncodedBatch>, value_type: &DataType) -> Result<RecordBatch> {
    let mins = if stats.is_empty() {
        new_empty_array(value_type)
    } else {
        ScalarValue::iter_to_array(stats.iter().map(|stat| stat.stats.min.clone()))?
    };
    let maxs = if stats.is_empty() {
        new_empty_array(value_type)
    } else {
        ScalarValue::iter_to_array(stats.iter().map(|stat| stat.stats.max.clone()))?
    };
    let null_counts = UInt32Array::from_iter_values(stats.iter().map(|stat| stat.stats.null_count));
    let page_numbers = UInt32Array::from_iter_values(stats.iter().map(|stat| stat.page_number));

    let schema = Arc::new(Schema::new(vec![
        // min and max can be null if the entire batch is null values
        Field::new("min", mins.data_type().clone(), true),
        Field::new("max", maxs.data_type().clone(), true),
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
pub trait TrainingSource: Send {
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

    /// Returns a stream of batches
    ///
    /// Each batch should have chunk_size rows
    ///
    /// The schema for the batch is slightly flexible.
    /// The first column may have any name or type, these are the values to index
    /// The second column must be the row ids which must be UInt64Type
    async fn scan_unordered_chunks(
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
    data_source: Box<dyn TrainingSource + Send>,
    sub_index_trainer: &dyn BTreeSubIndex,
    index_store: &dyn IndexStore,
    batch_size: u32,
) -> Result<()> {
    let mut sub_index_file = index_store
        .new_index_file(BTREE_PAGES_NAME, sub_index_trainer.schema().clone())
        .await?;
    let mut encoded_batches = Vec::new();
    let mut batch_idx = 0;
    let mut batches_source = data_source.scan_ordered_chunks(batch_size).await?;
    let value_type = batches_source.schema().field(0).data_type().clone();
    while let Some(batch) = batches_source.try_next().await? {
        debug_assert_eq!(batch.num_columns(), 2);
        debug_assert_eq!(*batch.column(1).data_type(), DataType::UInt64);
        encoded_batches.push(
            train_btree_page(batch, batch_idx, sub_index_trainer, sub_index_file.as_mut()).await?,
        );
        batch_idx += 1;
    }
    sub_index_file.finish().await?;
    let record_batch = btree_stats_as_batch(encoded_batches, &value_type)?;
    let mut file_schema = record_batch.schema().as_ref().clone();
    file_schema
        .metadata
        .insert(BATCH_SIZE_META_KEY.to_string(), batch_size.to_string());
    let mut btree_index_file = index_store
        .new_index_file(BTREE_LOOKUP_NAME, Arc::new(file_schema))
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
impl TrainingSource for BTreeUpdater {
    async fn scan_ordered_chunks(
        self: Box<Self>,
        chunk_size: u32,
    ) -> Result<SendableRecordBatchStream> {
        let data_type = self.new_data.schema().field(0).data_type().clone();
        // Datafusion currently has bugs with spilling on string columns
        // See https://github.com/apache/datafusion/issues/10073
        //
        // One we upgrade we can remove this
        let use_spilling = !matches!(data_type, DataType::Utf8 | DataType::LargeUtf8);

        let new_input = Arc::new(OneShotExec::new(self.new_data));
        let old_input = Self::into_old_input(self.index);
        debug_assert_eq!(
            old_input.schema().flattened_fields().len(),
            new_input.schema().flattened_fields().len()
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
        let ordered = Arc::new(SortPreservingMergeExec::new(
            LexOrdering::new(vec![sort_expr]),
            all_data,
        ));

        let unchunked = execute_plan(
            ordered,
            LanceExecutionOptions {
                use_spilling,
                ..Default::default()
            },
        )?;
        Ok(chunk_concat_stream(unchunked, chunk_size as usize))
    }

    async fn scan_unordered_chunks(
        self: Box<Self>,
        _chunk_size: u32,
    ) -> Result<SendableRecordBatchStream> {
        // BTree indices will never use unordered scans
        unimplemented!()
    }
}

/// A stream that reads the original training data back out of the index
///
/// This is used for updating the index
struct IndexReaderStream {
    reader: Arc<dyn IndexReader>,
    batch_size: u64,
    num_batches: u32,
    batch_idx: u32,
}

impl IndexReaderStream {
    async fn new(reader: Arc<dyn IndexReader>, batch_size: u64) -> Self {
        let num_batches = reader.num_batches(batch_size).await;
        Self {
            reader,
            batch_size,
            num_batches,
            batch_idx: 0,
        }
    }
}

impl Stream for IndexReaderStream {
    type Item = BoxFuture<'static, Result<RecordBatch>>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.batch_idx >= this.num_batches {
            return std::task::Poll::Ready(None);
        }
        let batch_num = this.batch_idx;
        this.batch_idx += 1;
        let reader_copy = this.reader.clone();
        let batch_size = this.batch_size;
        let read_task = async move {
            reader_copy
                .read_record_batch(batch_num as u64, batch_size)
                .await
        }
        .boxed();
        std::task::Poll::Ready(Some(read_task))
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use arrow::datatypes::{Float32Type, Float64Type, Int32Type, UInt64Type};
    use arrow_array::FixedSizeListArray;
    use arrow_schema::DataType;
    use datafusion::{
        execution::{SendableRecordBatchStream, TaskContext},
        physical_plan::{sorts::sort::SortExec, stream::RecordBatchStreamAdapter, ExecutionPlan},
    };
    use datafusion_common::{DataFusionError, ScalarValue};
    use datafusion_physical_expr::{expressions::col, LexOrdering, PhysicalSortExpr};
    use deepsize::DeepSizeOf;
    use futures::TryStreamExt;
    use lance_core::{cache::LanceCache, utils::mask::RowIdTreeMap};
    use lance_datafusion::{chunker::break_stream, datagen::DatafusionDatagenExt};
    use lance_datagen::{array, gen, ArrayGeneratorExt, BatchCount, RowCount};
    use lance_io::object_store::ObjectStore;
    use object_store::path::Path;
    use tempfile::tempdir;

    use crate::{
        metrics::NoOpMetricsCollector,
        scalar::{
            btree::{BTreeIndex, BTREE_PAGES_NAME, DEFAULT_BTREE_BATCH_SIZE},
            flat::FlatIndexMetadata,
            lance_format::{tests::MockTrainingSource, LanceIndexStore},
            IndexStore, SargableQuery, ScalarIndex, SearchResult,
        },
    };

    use super::{train_btree_index, OrderableScalarValue};

    #[test]
    fn test_scalar_value_size() {
        let size_of_i32 = OrderableScalarValue(ScalarValue::Int32(Some(0))).deep_size_of();
        let size_of_many_i32 = OrderableScalarValue(ScalarValue::FixedSizeList(Arc::new(
            FixedSizeListArray::from_iter_primitive::<Int32Type, _, _>(
                vec![Some(vec![Some(0); 128])],
                128,
            ),
        )))
        .deep_size_of();

        // deep_size_of should account for the rust type overhead
        assert!(size_of_i32 > 4);
        assert!(size_of_many_i32 > 128 * 4);
    }

    #[tokio::test]
    async fn test_null_ids() {
        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Generate 50,000 rows of random data with 80% nulls
        let stream = gen()
            .col(
                "value",
                array::rand::<Float32Type>().with_nulls(&[true, false, false, false, false]),
            )
            .col("_rowid", array::step::<UInt64Type>())
            .into_df_stream(RowCount::from(5000), BatchCount::from(10));
        let data_source = Box::new(MockTrainingSource::from(stream));
        let sub_index_trainer = FlatIndexMetadata::new(DataType::Float32);

        train_btree_index(
            data_source,
            &sub_index_trainer,
            test_store.as_ref(),
            DEFAULT_BTREE_BATCH_SIZE as u32,
        )
        .await
        .unwrap();

        let index = BTreeIndex::load(test_store.clone(), None).await.unwrap();

        assert_eq!(index.page_lookup.null_pages.len(), 10);

        let remap_dir = Arc::new(tempdir().unwrap());
        let remap_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(remap_dir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Remap with a no-op mapping.  The remapped index should be identical to the original
        index
            .remap(&HashMap::default(), remap_store.as_ref())
            .await
            .unwrap();

        let remap_index = BTreeIndex::load(remap_store.clone(), None).await.unwrap();

        assert_eq!(remap_index.page_lookup, index.page_lookup);

        let original_pages = test_store.open_index_file(BTREE_PAGES_NAME).await.unwrap();
        let remapped_pages = remap_store.open_index_file(BTREE_PAGES_NAME).await.unwrap();

        assert_eq!(original_pages.num_rows(), remapped_pages.num_rows());

        let original_data = original_pages
            .read_record_batch(0, original_pages.num_rows() as u64)
            .await
            .unwrap();
        let remapped_data = remapped_pages
            .read_record_batch(0, remapped_pages.num_rows() as u64)
            .await
            .unwrap();

        assert_eq!(original_data, remapped_data);
    }

    #[tokio::test]
    async fn test_nan_ordering() {
        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        let values = vec![
            0.0,
            1.0,
            2.0,
            3.0,
            f64::NAN,
            f64::NEG_INFINITY,
            f64::INFINITY,
        ];

        // This is a bit overkill but we've had bugs in the past where DF's sort
        // didn't agree with Arrow's sort so we do an end-to-end test here
        // and use DF to sort the data like we would in a real dataset.
        let data = gen()
            .col("value", array::cycle::<Float64Type>(values.clone()))
            .col("_rowid", array::step::<UInt64Type>())
            .into_df_exec(RowCount::from(10), BatchCount::from(100));
        let schema = data.schema();
        let sort_expr = PhysicalSortExpr::new_default(col("value", schema.as_ref()).unwrap());
        let plan = Arc::new(SortExec::new(LexOrdering::new(vec![sort_expr]), data));
        let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        let stream = break_stream(stream, 64);
        let stream = stream.map_err(DataFusionError::from);
        let stream =
            Box::pin(RecordBatchStreamAdapter::new(schema, stream)) as SendableRecordBatchStream;
        let data_source = Box::new(MockTrainingSource::from(stream));

        let sub_index_trainer = FlatIndexMetadata::new(DataType::Float64);

        train_btree_index(data_source, &sub_index_trainer, test_store.as_ref(), 64)
            .await
            .unwrap();

        let index = BTreeIndex::load(test_store, None).await.unwrap();

        for (idx, value) in values.into_iter().enumerate() {
            let query = SargableQuery::Equals(ScalarValue::Float64(Some(value)));
            let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
            assert_eq!(
                result,
                SearchResult::Exact(RowIdTreeMap::from_iter(((idx as u64)..1000).step_by(7)))
            );
        }
    }
}
