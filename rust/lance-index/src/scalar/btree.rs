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
    flat::FlatIndexMetadata, AnyQuery, BuiltinIndexType, IndexReader, IndexStore, IndexWriter,
    MetricsCollector, SargableQuery, ScalarIndex, ScalarIndexParams, SearchResult,
};
use crate::pbold;
use crate::{
    frag_reuse::FragReuseIndex,
    scalar::{
        expression::{SargableQueryParser, ScalarQueryParser},
        registry::{ScalarIndexPlugin, TrainingOrdering, TrainingRequest, VALUE_COLUMN_NAME},
        CreatedIndex, UpdateCriteria,
    },
};
use crate::{metrics::NoOpMetricsCollector, scalar::registry::TrainingCriteria};
use crate::{Index, IndexType};
use arrow_arith::numeric::add;
use arrow_array::{new_empty_array, Array, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Schema, SortOptions};
use async_trait::async_trait;
use datafusion::physical_plan::{
    sorts::sort_preserving_merge::SortPreservingMergeExec, stream::RecordBatchStreamAdapter,
    union::UnionExec, ExecutionPlan, SendableRecordBatchStream,
};
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_physical_expr::{expressions::Column, PhysicalSortExpr};
use deepsize::DeepSizeOf;
use futures::{
    future::BoxFuture,
    stream::{self},
    FutureExt, Stream, StreamExt, TryFutureExt, TryStreamExt,
};
use lance_core::{
    cache::{CacheKey, LanceCache, WeakLanceCache},
    error::LanceOptionExt,
    utils::{
        mask::RowIdTreeMap,
        tokio::get_num_compute_intensive_cpus,
        tracing::{IO_TYPE_LOAD_SCALAR_PART, TRACE_IO_EVENTS},
    },
    Error, Result, ROW_ID,
};
use lance_datafusion::{
    chunker::chunk_concat_stream,
    exec::{execute_plan, LanceExecutionOptions, OneShotExec},
};
use lance_io::object_store::ObjectStore;
use log::{debug, warn};
use object_store::path::Path;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize, Serializer};
use snafu::location;
use tracing::info;

const BTREE_LOOKUP_NAME: &str = "page_lookup.lance";
const BTREE_PAGES_NAME: &str = "page_data.lance";
pub const DEFAULT_BTREE_BATCH_SIZE: u64 = 4096;
const BATCH_SIZE_META_KEY: &str = "batch_size";
pub const DEFAULT_RANGE_PARTITIONED: bool = false;
const RANGE_PARTITIONED_META_KEY: &str = "range_partitioned";
const PAGE_NUM_PER_RANGE_PARTITION_META_KEY: &str = "page_num_per_range_partition";
const BTREE_INDEX_VERSION: u32 = 0;
pub(crate) const BTREE_VALUES_COLUMN: &str = "values";
pub(crate) const BTREE_IDS_COLUMN: &str = "ids";

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
// Cache key implementation for type-safe cache access
#[derive(Debug, Clone, DeepSizeOf)]
pub struct CachedScalarIndex(Arc<dyn ScalarIndex>);

impl CachedScalarIndex {
    pub fn new(index: Arc<dyn ScalarIndex>) -> Self {
        Self(index)
    }

    pub fn into_inner(self) -> Arc<dyn ScalarIndex> {
        self.0
    }
}

#[derive(Debug, Clone)]
pub struct BTreePageKey {
    pub page_number: u32,
}

impl CacheKey for BTreePageKey {
    type ValueType = CachedScalarIndex;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("page-{}", self.page_number).into()
    }
}

/// Note: this is very similar to the IVF index except we store the IVF part in a btree
/// for faster lookup
#[derive(Clone, Debug)]
pub struct BTreeIndex {
    page_lookup: Arc<BTreeLookup>,
    index_cache: WeakLanceCache,
    store: Arc<dyn IndexStore>,
    sub_index: Arc<dyn BTreeSubIndex>,
    batch_size: u64,
    frag_reuse_index: Option<Arc<FragReuseIndex>>,
}

impl DeepSizeOf for BTreeIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        // We don't include the index cache, or anything stored in it. For example:
        // sub_index and fri.
        self.page_lookup.deep_size_of_children(context) + self.store.deep_size_of_children(context)
    }
}

impl BTreeIndex {
    #[allow(clippy::too_many_arguments)]
    fn new(
        tree: BTreeMap<OrderableScalarValue, Vec<PageRecord>>,
        null_pages: Vec<u32>,
        store: Arc<dyn IndexStore>,
        index_cache: WeakLanceCache,
        sub_index: Arc<dyn BTreeSubIndex>,
        batch_size: u64,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Self {
        let page_lookup = Arc::new(BTreeLookup::new(tree, null_pages));
        Self {
            page_lookup,
            store,
            index_cache,
            sub_index,
            batch_size,
            frag_reuse_index,
        }
    }

    async fn lookup_page(
        &self,
        page_number: u32,
        index_reader: LazyIndexReader,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn ScalarIndex>> {
        self.index_cache
            .get_or_insert_with_key(BTreePageKey { page_number }, move || async move {
                let result = self.read_page(page_number, index_reader, metrics).await?;
                Ok(CachedScalarIndex::new(result))
            })
            .await
            .map(|v| v.as_ref().clone().into_inner())
    }

    async fn read_page(
        &self,
        page_number: u32,
        index_reader: LazyIndexReader,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn ScalarIndex>> {
        metrics.record_part_load();
        info!(target: TRACE_IO_EVENTS, r#type=IO_TYPE_LOAD_SCALAR_PART, index_type="btree", part_id=page_number);
        let index_reader = index_reader.get().await?;
        let mut serialized_page = index_reader
            .read_record_batch(page_number as u64, self.batch_size)
            .await?;
        if let Some(frag_reuse_index_ref) = self.frag_reuse_index.as_ref() {
            serialized_page =
                frag_reuse_index_ref.remap_row_ids_record_batch(serialized_page, 1)?;
        }
        let result = self.sub_index.load_subindex(serialized_page).await?;
        Ok(result)
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
        // 1 and 2 and three is in page 2 and seven is in pages 8 and 9, then when searching page 2 we only need
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
        index_cache: &LanceCache,
        batch_size: u64,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let mut map = BTreeMap::<OrderableScalarValue, Vec<PageRecord>>::new();
        let mut null_pages = Vec::<u32>::new();

        if data.num_rows() == 0 {
            let data_type = data.column(0).data_type().clone();
            let sub_index = Arc::new(FlatIndexMetadata::new(data_type));
            return Ok(Self::new(
                map,
                null_pages,
                store,
                WeakLanceCache::from(index_cache),
                sub_index,
                batch_size,
                frag_reuse_index,
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
            map,
            null_pages,
            store,
            WeakLanceCache::from(index_cache),
            sub_index,
            batch_size,
            frag_reuse_index,
        ))
    }

    async fn load(
        store: Arc<dyn IndexStore>,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        index_cache: &LanceCache,
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
            index_cache,
            batch_size,
            frag_reuse_index,
        )?))
    }

    /// Create a stream of all the data in the index, in the same format used to train the index
    async fn into_data_stream(self) -> Result<SendableRecordBatchStream> {
        let reader = self.store.open_index_file(BTREE_PAGES_NAME).await?;
        let schema = self.sub_index.schema().clone();
        let value_field = schema.field(0).clone().with_name(VALUE_COLUMN_NAME);
        let row_id_field = schema.field(1).clone().with_name(ROW_ID);
        let new_schema = Arc::new(Schema::new(vec![value_field, row_id_field]));
        let new_schema_clone = new_schema.clone();
        let reader_stream = IndexReaderStream::new(reader, self.batch_size).await;
        let batches = reader_stream
            .map(|fut| fut.map_err(DataFusionError::from))
            .buffered(self.store.io_parallelism())
            .map_ok(move |batch| {
                RecordBatch::try_new(
                    new_schema.clone(),
                    vec![batch.column(0).clone(), batch.column(1).clone()],
                )
                .unwrap()
            })
            .boxed();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            new_schema_clone,
            batches,
        )))
    }

    async fn into_old_data(self) -> Result<Arc<dyn ExecutionPlan>> {
        let stream = self.into_data_stream().await?;
        Ok(Arc::new(OneShotExec::new(stream)))
    }

    async fn combine_old_new(
        self,
        new_data: SendableRecordBatchStream,
        chunk_size: u64,
    ) -> Result<SendableRecordBatchStream> {
        let data_type = new_data.schema().field(0).data_type().clone();
        // Datafusion currently has bugs with spilling on string columns
        // See https://github.com/apache/datafusion/issues/10073
        //
        // One we upgrade we can remove this
        let use_spilling = !matches!(data_type, DataType::Utf8 | DataType::LargeUtf8);
        let value_column_index = new_data.schema().index_of(VALUE_COLUMN_NAME)?;

        let new_input = Arc::new(OneShotExec::new(new_data));
        let old_input = self.into_old_data().await?;
        debug_assert_eq!(
            old_input.schema().flattened_fields().len(),
            new_input.schema().flattened_fields().len()
        );

        let sort_expr = PhysicalSortExpr {
            expr: Arc::new(Column::new(VALUE_COLUMN_NAME, value_column_index)),
            options: SortOptions {
                descending: false,
                nulls_first: true,
            },
        };
        // The UnionExec creates multiple partitions but the SortPreservingMergeExec merges
        // them back into a single partition.
        let all_data = Arc::new(UnionExec::new(vec![old_input, new_input]));
        let ordered = Arc::new(SortPreservingMergeExec::new([sort_expr].into(), all_data));

        let unchunked = execute_plan(
            ordered,
            LanceExecutionOptions {
                use_spilling,
                ..Default::default()
            },
        )?;
        Ok(chunk_concat_stream(unchunked, chunk_size as usize))
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
        let index_reader = LazyIndexReader::new(self.store.clone());
        let reader = index_reader.get().await?;
        let num_rows = reader.num_rows();
        let batch_size = self.batch_size as usize;
        let num_pages = num_rows.div_ceil(batch_size);
        let mut pages = stream::iter(0..num_pages)
            .map(|page_idx| {
                let index_reader = index_reader.clone();
                let page_idx = page_idx as u32;
                async move {
                    let page = self
                        .read_page(page_idx, index_reader, &NoOpMetricsCollector)
                        .await?;
                    Result::Ok((page_idx, page))
                }
            })
            .buffer_unordered(get_num_compute_intensive_cpus());

        while let Some((page_idx, page)) = pages.try_next().await? {
            let inserted = self
                .index_cache
                .insert_with_key(
                    &BTreePageKey {
                        page_number: page_idx,
                    },
                    Arc::new(CachedScalarIndex::new(page)),
                )
                .await;

            if !inserted {
                return Err(Error::Internal {
                    message: "Failed to prewarm index: cache is no longer available".to_string(),
                    location: location!(),
                });
            }
        }

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

    fn can_remap(&self) -> bool {
        true
    }

    async fn remap(
        &self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
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
            .await?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::BTreeIndexDetails::default())
                .unwrap(),
            index_version: BTREE_INDEX_VERSION,
        })
    }

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        // Merge the existing index data with the new data and then retrain the index on the merged stream
        let merged_data_source = self
            .clone()
            .combine_old_new(new_data, DEFAULT_BTREE_BATCH_SIZE)
            .await?;
        train_btree_index(
            merged_data_source,
            self.sub_index.as_ref(),
            dest_store,
            DEFAULT_BTREE_BATCH_SIZE,
            None,
            None,
        )
        .await?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::BTreeIndexDetails::default())
                .unwrap(),
            index_version: BTREE_INDEX_VERSION,
        })
    }

    fn update_criteria(&self) -> UpdateCriteria {
        UpdateCriteria::only_new_data(TrainingCriteria::new(TrainingOrdering::Values).with_row_id())
    }

    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        let params = serde_json::to_value(BTreeParameters {
            zone_size: Some(self.batch_size),
            range_id: None,
        })?;
        Ok(ScalarIndexParams::for_builtin(BuiltinIndexType::BTree).with_params(&params))
    }
}

struct BatchStats {
    min: ScalarValue,
    max: ScalarValue,
    null_count: u32,
}

fn analyze_batch(batch: &RecordBatch) -> Result<BatchStats> {
    let values = batch.column_by_name(VALUE_COLUMN_NAME).expect_ok()?;
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

/// Train a btree index from a stream of sorted page-size batches of values and row ids
///
/// Note: This is likely to change.  It is unreasonable to expect the caller to do the sorting
/// and re-chunking into page-size batches.  This is left for simplicity as this feature is still
/// a work in progress
pub async fn train_btree_index(
    batches_source: SendableRecordBatchStream,
    sub_index_trainer: &dyn BTreeSubIndex,
    index_store: &dyn IndexStore,
    batch_size: u64,
    fragment_ids: Option<Vec<u32>>,
    range_id: Option<u32>,
) -> Result<()> {
    // Create `partition_id` for distributed index building.
    // This ID serves as a high-level mask (first 32 bits of a u64) to ensure
    // that index partitions generated by different workers do not conflict.
    // Lance supports two strategies for distributed training: fragment-based and range-based.
    let partition_id = fragment_ids
        .as_ref()
        // --- Fragment-based Partitioning ---
        // Used when training sub-indexes on a fragment-level-split basis. The `partition_id` is
        // derived from `fragment_ids` to associate the index pages with their source fragment.
        .and_then(|frag_ids| frag_ids.first())
        .map(|&first_frag_id| (first_frag_id as u64) << 32)
        // --- Range-based Partitioning ---
        // Built upon data globally sorted by an external compute engine. The `range_id` creates
        // a unique name for the index pages generated by each worker.
        .or_else(|| range_id.map(|id| (id as u64) << 32));

    let mut sub_index_file;
    if partition_id.is_none() {
        sub_index_file = index_store
            .new_index_file(BTREE_PAGES_NAME, sub_index_trainer.schema().clone())
            .await?;
    } else {
        sub_index_file = index_store
            .new_index_file(
                part_page_data_file_path(partition_id.unwrap()).as_str(),
                sub_index_trainer.schema().clone(),
            )
            .await?;
    }

    let mut encoded_batches = Vec::new();
    let mut batch_idx = 0;

    let value_type = batches_source
        .schema()
        .field_with_name(VALUE_COLUMN_NAME)?
        .data_type()
        .clone();

    let mut batches_source = chunk_concat_stream(batches_source, batch_size as usize);

    while let Some(batch) = batches_source.try_next().await? {
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
    file_schema.metadata.insert(
        RANGE_PARTITIONED_META_KEY.to_string(),
        range_id.is_some().to_string(),
    );
    let mut btree_index_file;
    if partition_id.is_none() {
        btree_index_file = index_store
            .new_index_file(BTREE_LOOKUP_NAME, Arc::new(file_schema))
            .await?;
    } else {
        btree_index_file = index_store
            .new_index_file(
                part_lookup_file_path(partition_id.unwrap()).as_str(),
                Arc::new(file_schema),
            )
            .await?;
    }
    btree_index_file.write_record_batch(record_batch).await?;
    btree_index_file.finish().await?;
    Ok(())
}

pub async fn merge_index_files(
    object_store: &ObjectStore,
    index_dir: &Path,
    store: Arc<dyn IndexStore>,
    batch_readhead: Option<usize>,
) -> Result<()> {
    // List all partition page / lookup files in the index directory
    let (part_page_files, part_lookup_files) =
        list_page_lookup_files(object_store, index_dir).await?;
    merge_metadata_files(store, &part_page_files, &part_lookup_files, batch_readhead).await
}

/// List and filter files from the index directory
/// Returns (page_files, lookup_files)
async fn list_page_lookup_files(
    object_store: &ObjectStore,
    index_dir: &Path,
) -> Result<(Vec<String>, Vec<String>)> {
    let mut part_page_files = Vec::new();
    let mut part_lookup_files = Vec::new();

    let mut list_stream = object_store.list(Some(index_dir.clone()));

    while let Some(item) = list_stream.next().await {
        match item {
            Ok(meta) => {
                let file_name = meta.location.filename().unwrap_or_default();
                // Filter files matching the pattern part_*_page_data.lance
                if file_name.starts_with("part_") && file_name.ends_with("_page_data.lance") {
                    part_page_files.push(file_name.to_string());
                }
                // Filter files matching the pattern part_*_page_lookup.lance
                if file_name.starts_with("part_") && file_name.ends_with("_page_lookup.lance") {
                    part_lookup_files.push(file_name.to_string());
                }
            }
            Err(_) => continue,
        }
    }

    if part_page_files.is_empty() || part_lookup_files.is_empty() {
        return Err(Error::Internal {
            message: format!(
                "No partition metadata files found in index directory: {} (page_files: {}, lookup_files: {})",
                index_dir, part_page_files.len(), part_lookup_files.len()
            ),
            location: location!(),
        });
    }

    Ok((part_page_files, part_lookup_files))
}

/// Merge multiple partition page / lookup files into a complete metadata file
///
/// In a distributed environment, each worker node writes partition page / lookup file for the partitions it processes,
/// and this function merges these files into a final metadata file.
/// - For **non-range-partitioned** indices, it performs a full K-way sort-merge of page files to create new global page and lookup files.
/// - For **range-partitioned** indices, it concatenates lookup files, as data is already globally sorted.
async fn merge_metadata_files(
    store: Arc<dyn IndexStore>,
    part_page_files: &[String],
    part_lookup_files: &[String],
    batch_readhead: Option<usize>,
) -> Result<()> {
    if part_lookup_files.is_empty() || part_page_files.is_empty() {
        return Err(Error::Internal {
            message: "No partition files provided for merging".to_string(),
            location: location!(),
        });
    }

    // Step 1: Create lookup map for page files by partition ID
    if part_lookup_files.len() != part_page_files.len() {
        return Err(Error::Internal {
            message: format!(
                "Number of partition lookup files ({}) does not match number of partition page files ({})",
                part_lookup_files.len(),
                part_page_files.len()
            ),
            location: location!(),
        });
    }
    let mut page_files_map = HashMap::new();
    for page_file in part_page_files {
        let partition_id = extract_partition_id(page_file)?;
        page_files_map.insert(partition_id, page_file);
    }

    // Step 2: Validate that all lookup files have corresponding page files
    for lookup_file in part_lookup_files {
        let partition_id = extract_partition_id(lookup_file)?;
        if !page_files_map.contains_key(&partition_id) {
            return Err(Error::Internal {
                message: format!(
                    "No corresponding page file found for lookup file: {} (partition_id: {})",
                    lookup_file, partition_id
                ),
                location: location!(),
            });
        }
    }

    // Step 3: Extract shared metadata and generate lookup_schema
    let first_lookup_reader = store.open_index_file(&part_lookup_files[0]).await?;
    let batch_size = first_lookup_reader
        .schema()
        .metadata
        .get(BATCH_SIZE_META_KEY)
        .map(|bs| bs.parse().unwrap_or(DEFAULT_BTREE_BATCH_SIZE))
        .unwrap_or(DEFAULT_BTREE_BATCH_SIZE);
    let range_partitioned = first_lookup_reader
        .schema()
        .metadata
        .get(RANGE_PARTITIONED_META_KEY)
        .map(|bs| bs.parse().unwrap_or(DEFAULT_RANGE_PARTITIONED))
        .unwrap_or(DEFAULT_RANGE_PARTITIONED);

    // Get the value type from lookup schema (min column)
    let value_type = first_lookup_reader
        .schema()
        .fields
        .first()
        .unwrap()
        .data_type();

    let mut metadata = HashMap::new();
    metadata.insert(BATCH_SIZE_META_KEY.to_string(), batch_size.to_string());
    let lookup_schema = Arc::new(Schema::new(vec![
        Field::new("min", value_type.clone(), true),
        Field::new("max", value_type.clone(), true),
        Field::new("null_count", DataType::UInt32, false),
        Field::new("page_idx", DataType::UInt32, false),
    ]));

    // Step 4: Merge pages and lookups and generate new index files
    if range_partitioned {
        merge_range_partitioned_lookups(
            &store,
            part_lookup_files,
            lookup_schema,
            metadata,
            batch_size,
            batch_readhead,
        )
        .await
    } else {
        merge_pages_and_lookups(
            &store,
            part_page_files,
            part_lookup_files,
            &page_files_map,
            lookup_schema,
            metadata,
            batch_size,
            batch_readhead,
        )
        .await
    }
}

/// Merge lookup files for a range-partitioned index.
///
/// This function assumes its inputs have been pre-validated. It streams through
/// each partition file sequentially, adjusts `page_idx` to create a contiguous
/// global index, and writes the results to a new, single lookup file.
async fn merge_range_partitioned_lookups(
    store: &Arc<dyn IndexStore>,
    part_lookup_files: &[String],
    lookup_schema: Arc<Schema>,
    mut metadata: HashMap<String, String>,
    batch_size: u64,
    batch_readhead: Option<usize>,
) -> Result<()> {
    let sorted_part_lookup_files = sort_files_by_partition_id(part_lookup_files)?;
    let mut lookup_file = store
        .new_index_file(BTREE_LOOKUP_NAME, lookup_schema)
        .await?;

    let mut rows_per_file: Vec<u32> = Vec::with_capacity(sorted_part_lookup_files.len());
    let mut num_rows_written = 0u32;

    for part_lookup_file in sorted_part_lookup_files {
        let lookup_reader = store.open_index_file(&part_lookup_file).await?;
        let reader_stream = IndexReaderStream::new(lookup_reader.clone(), batch_size).await;
        let mut stream = reader_stream.buffered(batch_readhead.unwrap_or(1)).boxed();
        while let Some(batch) = stream.next().await {
            let original_batch = batch?;
            let modified_batch = add_offset_to_page_idx(&original_batch, num_rows_written)?;
            lookup_file.write_record_batch(modified_batch).await?;
        }
        rows_per_file.push(lookup_reader.num_rows() as u32);
        num_rows_written += lookup_reader.num_rows() as u32;
    }

    metadata.insert(RANGE_PARTITIONED_META_KEY.to_string(), "true".to_string());
    metadata.insert(
        PAGE_NUM_PER_RANGE_PARTITION_META_KEY.to_string(),
        serde_json::to_string(&rows_per_file)?,
    );

    lookup_file.finish_with_metadata(metadata).await?;

    // In this mode, we only clean up lookup files, and page files are untouched.
    cleanup_partition_files(store, part_lookup_files, &[]).await;
    Ok(())
}

/// Merges partition files using a K-way sort-merge algorithm.
///
/// This function assumes its inputs have been pre-validated. It reads from all
/// partitioned page files simultaneously, merges them into a single sorted stream,
/// writes a new global page file, and generates a corresponding global lookup file.
#[allow(clippy::too_many_arguments)]
async fn merge_pages_and_lookups(
    store: &Arc<dyn IndexStore>,
    part_page_files: &[String],
    part_lookup_files: &[String],
    page_files_map: &HashMap<u64, &String>,
    lookup_schema: Arc<Schema>,
    metadata: HashMap<String, String>,
    batch_size: u64,
    batch_readhead: Option<usize>,
) -> Result<()> {
    // Create a new global page file
    let partition_id = extract_partition_id(part_lookup_files[0].as_str())?;
    let page_file = page_files_map.get(&partition_id).unwrap();
    let page_reader = store.open_index_file(page_file).await?;
    let page_schema = page_reader.schema().clone();

    let arrow_schema = Arc::new(Schema::from(&page_schema));
    let mut page_file = store
        .new_index_file(BTREE_PAGES_NAME, arrow_schema.clone())
        .await?;

    let lookup_entries = merge_pages(
        part_lookup_files,
        page_files_map,
        store,
        batch_size,
        &mut page_file,
        arrow_schema.clone(),
        batch_readhead,
    )
    .await?;
    page_file.finish().await?;

    let lookup_batch = RecordBatch::try_new(
        lookup_schema.clone(),
        vec![
            ScalarValue::iter_to_array(lookup_entries.iter().map(|(min, _, _, _)| min.clone()))?,
            ScalarValue::iter_to_array(lookup_entries.iter().map(|(_, max, _, _)| max.clone()))?,
            Arc::new(UInt32Array::from_iter_values(
                lookup_entries
                    .iter()
                    .map(|(_, _, null_count, _)| *null_count),
            )),
            Arc::new(UInt32Array::from_iter_values(
                lookup_entries.iter().map(|(_, _, _, page_idx)| *page_idx),
            )),
        ],
    )?;
    let mut lookup_file = store
        .new_index_file(BTREE_LOOKUP_NAME, lookup_schema)
        .await?;
    lookup_file.write_record_batch(lookup_batch).await?;
    lookup_file.finish_with_metadata(metadata).await?;

    // After successfully writing the merged files, delete all partition files
    // Only perform deletion after files are successfully written, ensuring debug information is not lost in case of failure
    cleanup_partition_files(store, part_lookup_files, part_page_files).await;

    Ok(())
}

fn add_offset_to_page_idx(batch: &RecordBatch, offset: u32) -> Result<RecordBatch> {
    let (page_idx_pos, _) =
        batch
            .schema()
            .column_with_name("page_idx")
            .ok_or_else(|| Error::Internal {
                message: "Column 'page_idx' not found in RecordBatch schema".to_string(),
                location: location!(),
            })?;
    let page_idx_array = batch
        .column(page_idx_pos)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| Error::Internal {
            message: "Failed to downcast 'page_idx' column to UInt32Array".to_string(),
            location: location!(),
        })?;
    let offset_array = UInt32Array::from(vec![offset; page_idx_array.len()]);
    let new_page_idx_array_ref = add(page_idx_array, &offset_array)?;
    let mut new_columns = batch.columns().to_vec();
    new_columns[page_idx_pos] = new_page_idx_array_ref;
    let new_batch = RecordBatch::try_new(batch.schema(), new_columns)?;
    Ok(new_batch)
}

/// Merge pages using Datafusion's SortPreservingMergeExec
/// which implements a K-way merge algorithm with fixed-size output batches
async fn merge_pages(
    part_lookup_files: &[String],
    page_files_map: &HashMap<u64, &String>,
    store: &Arc<dyn IndexStore>,
    batch_size: u64,
    page_file: &mut Box<dyn IndexWriter>,
    arrow_schema: Arc<Schema>,
    batch_readhead: Option<usize>,
) -> Result<Vec<(ScalarValue, ScalarValue, u32, u32)>> {
    let mut lookup_entries = Vec::new();
    let mut page_idx = 0u32;

    debug!(
        "Starting SortPreservingMerge with {} partitions",
        part_lookup_files.len()
    );

    let value_field = arrow_schema.field(0).clone().with_name(VALUE_COLUMN_NAME);
    let row_id_field = arrow_schema.field(1).clone().with_name(ROW_ID);
    let stream_schema = Arc::new(Schema::new(vec![value_field, row_id_field]));

    // Create execution plans for each stream
    let mut inputs: Vec<Arc<dyn ExecutionPlan>> = Vec::new();
    for lookup_file in part_lookup_files {
        let partition_id = extract_partition_id(lookup_file)?;
        let page_file_name =
            (*page_files_map
                .get(&partition_id)
                .ok_or_else(|| Error::Internal {
                    message: format!("Page file not found for partition ID: {}", partition_id),
                    location: location!(),
                })?)
            .clone();

        let reader = store.open_index_file(&page_file_name).await?;

        let reader_stream = IndexReaderStream::new(reader, batch_size).await;

        let stream = reader_stream
            .map(|fut| fut.map_err(DataFusionError::from))
            .buffered(batch_readhead.unwrap_or(1))
            .boxed();

        let sendable_stream =
            Box::pin(RecordBatchStreamAdapter::new(stream_schema.clone(), stream));
        inputs.push(Arc::new(OneShotExec::new(sendable_stream)));
    }

    // Create Union execution plan to combine all partitions
    let union_inputs = Arc::new(UnionExec::new(inputs));

    // Create SortPreservingMerge execution plan
    let value_column_index = stream_schema.index_of(VALUE_COLUMN_NAME)?;
    let sort_expr = PhysicalSortExpr {
        expr: Arc::new(Column::new(VALUE_COLUMN_NAME, value_column_index)),
        options: SortOptions {
            descending: false,
            nulls_first: true,
        },
    };

    let merge_exec = Arc::new(SortPreservingMergeExec::new(
        [sort_expr].into(),
        union_inputs,
    ));

    let unchunked = execute_plan(
        merge_exec,
        LanceExecutionOptions {
            use_spilling: false,
            ..Default::default()
        },
    )?;

    // Use chunk_concat_stream to ensure fixed batch sizes
    let mut chunked_stream = chunk_concat_stream(unchunked, batch_size as usize);

    // Process chunked stream
    while let Some(batch) = chunked_stream.try_next().await? {
        let writer_batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![batch.column(0).clone(), batch.column(1).clone()],
        )?;

        page_file.write_record_batch(writer_batch).await?;

        let min_val = ScalarValue::try_from_array(batch.column(0), 0)?;
        let max_val = ScalarValue::try_from_array(batch.column(0), batch.num_rows() - 1)?;
        let null_count = batch.column(0).null_count() as u32;

        lookup_entries.push((min_val, max_val, null_count, page_idx));
        page_idx += 1;
    }

    Ok(lookup_entries)
}

// Sorts file paths by partition ID extracted from file name.
fn sort_files_by_partition_id(part_files: &[String]) -> Result<Vec<String>> {
    let mut files_with_ids: Vec<(u64, &String)> = part_files
        .iter()
        .map(|file| extract_partition_id(file).map(|id| (id, file)))
        .collect::<Result<Vec<_>>>()?;

    files_with_ids.sort_unstable_by_key(|k| k.0);

    let sorted_files = files_with_ids
        .into_iter()
        .map(|(_, file)| file.clone())
        .collect();

    Ok(sorted_files)
}

/// Extract partition ID from partition file name
/// Expected format: "part_{partition_id}_{suffix}.lance"
fn extract_partition_id(filename: &str) -> Result<u64> {
    if !filename.starts_with("part_") {
        return Err(Error::Internal {
            message: format!("Invalid partition file name format: {}", filename),
            location: location!(),
        });
    }

    let parts: Vec<&str> = filename.split('_').collect();
    if parts.len() < 3 {
        return Err(Error::Internal {
            message: format!("Invalid partition file name format: {}", filename),
            location: location!(),
        });
    }

    parts[1].parse::<u64>().map_err(|_| Error::Internal {
        message: format!("Failed to parse partition ID from filename: {}", filename),
        location: location!(),
    })
}

/// Clean up partition files after successful merge
///
/// This function safely deletes partition lookup and page files after a successful merge operation.
/// File deletion failures are logged but do not affect the overall success of the merge operation.
async fn cleanup_partition_files(
    store: &Arc<dyn IndexStore>,
    part_lookup_files: &[String],
    part_page_files: &[String],
) {
    // Clean up partition lookup files
    for file_name in part_lookup_files {
        cleanup_single_file(
            store,
            file_name,
            "part_",
            "_page_lookup.lance",
            "partition lookup",
        )
        .await;
    }

    // Clean up partition page files
    for file_name in part_page_files {
        cleanup_single_file(
            store,
            file_name,
            "part_",
            "_page_data.lance",
            "partition page",
        )
        .await;
    }
}

/// Helper function to clean up a single partition file
///
/// Performs safety checks on the filename pattern before attempting deletion.
async fn cleanup_single_file(
    store: &Arc<dyn IndexStore>,
    file_name: &str,
    expected_prefix: &str,
    expected_suffix: &str,
    file_type: &str,
) {
    if file_name.starts_with(expected_prefix) && file_name.ends_with(expected_suffix) {
        match store.delete_index_file(file_name).await {
            Ok(()) => {
                debug!("Successfully deleted {} file: {}", file_type, file_name);
            }
            Err(e) => {
                warn!(
                    "Failed to delete {} file '{}': {}. \
                    This does not affect the merge operation, but may leave \
                    partition files that should be cleaned up manually.",
                    file_type, file_name, e
                );
            }
        }
    } else {
        // If the filename doesn't match the expected format, log a warning but don't attempt deletion
        warn!(
            "Skipping deletion of file '{}' as it does not match the expected \
            {} file pattern ({}*{})",
            file_name, file_type, expected_prefix, expected_suffix
        );
    }
}

pub(crate) fn part_page_data_file_path(partition_id: u64) -> String {
    format!("part_{}_{}", partition_id, BTREE_PAGES_NAME)
}

pub(crate) fn part_lookup_file_path(partition_id: u64) -> String {
    format!("part_{}_{}", partition_id, BTREE_LOOKUP_NAME)
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

/// Parameters for a btree index
#[derive(Debug, Serialize, Deserialize)]
pub struct BTreeParameters {
    /// The number of rows to include in each zone
    pub zone_size: Option<u64>,

    /// The ordinal ID of a range partition, enabling a two-step distributed index build.
    ///
    /// This parameter is key to building a **range-partitioned index**, which is composed of
    /// multiple, independently-built sub-indices that collectively cover the entire dataset.
    /// The process involves:
    ///
    /// 1.  **Global Sorting & Partitioning (Caller's Responsibility)**: First, the entire
    ///     dataset must be globally sorted and divided into  *contiguous* partitions
    ///     based on value ranges.
    ///
    /// 2.  **Independent Sub-Index Construction**: This function is called for each
    ///     partition's data separately. The `range_id` you provide (e.g., `0`, `1`, `2`, ...)
    ///     identifies which partition this sub-index represents. To ensure global
    ///     consistency, the data provided must adhere to a strict ordering guarantee:
    ///     all values used to train `range_id: N` must be **less than or equal to**
    ///     all values for `range_id: N+1`.
    ///
    /// If `None`, a single, monolithic index is built over the provided dataset.
    pub range_id: Option<u32>,
}

struct BTreeTrainingRequest {
    parameters: BTreeParameters,
    criteria: TrainingCriteria,
}

impl BTreeTrainingRequest {
    pub fn new(parameters: BTreeParameters) -> Self {
        Self {
            parameters,
            // BTree indexes need data sorted by the value column
            criteria: TrainingCriteria::new(TrainingOrdering::Values).with_row_id(),
        }
    }
}

impl TrainingRequest for BTreeTrainingRequest {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn criteria(&self) -> &TrainingCriteria {
        &self.criteria
    }
}

#[derive(Debug, Default)]
pub struct BTreeIndexPlugin;

#[async_trait]
impl ScalarIndexPlugin for BTreeIndexPlugin {
    fn new_training_request(
        &self,
        params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        if field.data_type().is_nested() {
            return Err(Error::InvalidInput {
                source: "A btree index can only be created on a non-nested field.".into(),
                location: location!(),
            });
        }

        let params = serde_json::from_str::<BTreeParameters>(params)?;
        Ok(Box::new(BTreeTrainingRequest::new(params)))
    }

    fn provides_exact_answer(&self) -> bool {
        true
    }

    fn version(&self) -> u32 {
        BTREE_INDEX_VERSION
    }

    fn new_query_parser(
        &self,
        index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        Some(Box::new(SargableQueryParser::new(index_name, false)))
    }

    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
        fragment_ids: Option<Vec<u32>>,
    ) -> Result<CreatedIndex> {
        let request = request
            .as_any()
            .downcast_ref::<BTreeTrainingRequest>()
            .unwrap();
        let value_type = data
            .schema()
            .field_with_name(VALUE_COLUMN_NAME)?
            .data_type()
            .clone();
        let flat_index_trainer = FlatIndexMetadata::new(value_type);
        train_btree_index(
            data,
            &flat_index_trainer,
            index_store,
            request
                .parameters
                .zone_size
                .unwrap_or(DEFAULT_BTREE_BATCH_SIZE),
            fragment_ids,
            request.parameters.range_id,
        )
        .await?;
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::BTreeIndexDetails::default())
                .unwrap(),
            index_version: BTREE_INDEX_VERSION,
        })
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        _index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        Ok(BTreeIndex::load(index_store, frag_reuse_index, cache).await? as Arc<dyn ScalarIndex>)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;
    use std::{collections::HashMap, sync::Arc};

    use arrow::datatypes::{Float32Type, Float64Type, Int32Type, UInt64Type};
    use arrow_array::FixedSizeListArray;
    use arrow_schema::DataType;
    use datafusion::{
        execution::{SendableRecordBatchStream, TaskContext},
        physical_plan::{sorts::sort::SortExec, stream::RecordBatchStreamAdapter, ExecutionPlan},
    };
    use datafusion_common::{DataFusionError, ScalarValue};
    use datafusion_physical_expr::{expressions::col, PhysicalSortExpr};
    use deepsize::DeepSizeOf;
    use futures::TryStreamExt;
    use lance_core::utils::tempfile::TempObjDir;
    use lance_core::{cache::LanceCache, utils::mask::RowIdTreeMap};
    use lance_datafusion::{chunker::break_stream, datagen::DatafusionDatagenExt};
    use lance_datagen::{array, gen_batch, ArrayGeneratorExt, BatchCount, RowCount};
    use lance_io::object_store::ObjectStore;

    use crate::metrics::LocalMetricsCollector;
    use crate::scalar::btree::BTREE_LOOKUP_NAME;
    use crate::{
        metrics::NoOpMetricsCollector,
        scalar::{
            btree::{BTreeIndex, BTREE_PAGES_NAME},
            flat::FlatIndexMetadata,
            lance_format::LanceIndexStore,
            IndexStore, SargableQuery, ScalarIndex, SearchResult,
        },
    };

    use super::{
        part_lookup_file_path, part_page_data_file_path, train_btree_index, OrderableScalarValue,
        DEFAULT_BTREE_BATCH_SIZE,
    };
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
        let tmpdir = TempObjDir::default();
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Generate 50,000 rows of random data with 80% nulls
        let stream = gen_batch()
            .col(
                "value",
                array::rand::<Float32Type>().with_nulls(&[true, false, false, false, false]),
            )
            .col("_rowid", array::step::<UInt64Type>())
            .into_df_stream(RowCount::from(5000), BatchCount::from(10));
        let sub_index_trainer = FlatIndexMetadata::new(DataType::Float32);

        train_btree_index(
            stream,
            &sub_index_trainer,
            test_store.as_ref(),
            5000,
            None,
            None,
        )
        .await
        .unwrap();

        let index = BTreeIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .unwrap();

        assert_eq!(index.page_lookup.null_pages.len(), 10);

        let remap_dir = TempObjDir::default();
        let remap_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            remap_dir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Remap with a no-op mapping.  The remapped index should be identical to the original
        index
            .remap(&HashMap::default(), remap_store.as_ref())
            .await
            .unwrap();

        let remap_index = BTreeIndex::load(remap_store.clone(), None, &LanceCache::no_cache())
            .await
            .unwrap();

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
        let tmpdir = TempObjDir::default();
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
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
        let data = gen_batch()
            .col("value", array::cycle::<Float64Type>(values.clone()))
            .col("_rowid", array::step::<UInt64Type>())
            .into_df_exec(RowCount::from(10), BatchCount::from(100));
        let schema = data.schema();
        let sort_expr = PhysicalSortExpr::new_default(col("value", schema.as_ref()).unwrap());
        let plan = Arc::new(SortExec::new([sort_expr].into(), data));
        let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        let stream = break_stream(stream, 64);
        let stream = stream.map_err(DataFusionError::from);
        let stream =
            Box::pin(RecordBatchStreamAdapter::new(schema, stream)) as SendableRecordBatchStream;

        let sub_index_trainer = FlatIndexMetadata::new(DataType::Float64);

        train_btree_index(
            stream,
            &sub_index_trainer,
            test_store.as_ref(),
            64,
            None,
            None,
        )
        .await
        .unwrap();

        let index = BTreeIndex::load(test_store, None, &LanceCache::no_cache())
            .await
            .unwrap();

        for (idx, value) in values.into_iter().enumerate() {
            let query = SargableQuery::Equals(ScalarValue::Float64(Some(value)));
            let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
            assert_eq!(
                result,
                SearchResult::Exact(RowIdTreeMap::from_iter(((idx as u64)..1000).step_by(7)))
            );
        }
    }

    #[tokio::test]
    async fn test_page_cache() {
        let tmpdir = TempObjDir::default();
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let data = gen_batch()
            .col("value", array::step::<Float32Type>())
            .col("_rowid", array::step::<UInt64Type>())
            .into_df_exec(RowCount::from(1000), BatchCount::from(10));
        let schema = data.schema();
        let sort_expr = PhysicalSortExpr::new_default(col("value", schema.as_ref()).unwrap());
        let plan = Arc::new(SortExec::new([sort_expr].into(), data));
        let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        let stream = break_stream(stream, 64);
        let stream = stream.map_err(DataFusionError::from);
        let stream =
            Box::pin(RecordBatchStreamAdapter::new(schema, stream)) as SendableRecordBatchStream;
        let sub_index_trainer = FlatIndexMetadata::new(DataType::Float32);

        train_btree_index(
            stream,
            &sub_index_trainer,
            test_store.as_ref(),
            64,
            None,
            None,
        )
        .await
        .unwrap();

        let cache = Arc::new(LanceCache::with_capacity(100 * 1024 * 1024));
        let index = BTreeIndex::load(test_store, None, cache.as_ref())
            .await
            .unwrap();

        let query = SargableQuery::Equals(ScalarValue::Float32(Some(0.0)));
        let metrics = LocalMetricsCollector::default();
        let query1 = index.search(&query, &metrics);
        let query2 = index.search(&query, &metrics);
        tokio::join!(query1, query2).0.unwrap();
        assert_eq!(metrics.parts_loaded.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_fragment_btree_index_consistency() {
        // Setup stores for both indexes
        let full_tmpdir = TempObjDir::default();
        let full_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            full_tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let fragment_tmpdir = TempObjDir::default();
        let fragment_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            fragment_tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let sub_index_trainer = FlatIndexMetadata::new(DataType::Int32);

        // Method 1: Build complete index directly using the same data
        // Create deterministic data for comparison - use 2 * DEFAULT_BTREE_BATCH_SIZE for testing
        let total_count = 2 * DEFAULT_BTREE_BATCH_SIZE;
        let full_data_gen = gen_batch()
            .col("value", array::step::<Int32Type>())
            .col("_rowid", array::step::<UInt64Type>())
            .into_df_stream(RowCount::from(total_count / 2), BatchCount::from(2));
        let full_data_source = Box::pin(RecordBatchStreamAdapter::new(
            full_data_gen.schema(),
            full_data_gen,
        ));

        train_btree_index(
            full_data_source,
            &sub_index_trainer,
            full_store.as_ref(),
            DEFAULT_BTREE_BATCH_SIZE,
            None,
            None,
        )
        .await
        .unwrap();

        // Method 2: Build fragment-based index using the same data split into fragments
        // Create fragment 1 index - first half of the data (0 to DEFAULT_BTREE_BATCH_SIZE-1)
        let half_count = DEFAULT_BTREE_BATCH_SIZE;
        let fragment1_gen = gen_batch()
            .col("value", array::step::<Int32Type>())
            .col("_rowid", array::step::<UInt64Type>())
            .into_df_stream(RowCount::from(half_count), BatchCount::from(1));
        let fragment1_data_source = Box::pin(RecordBatchStreamAdapter::new(
            fragment1_gen.schema(),
            fragment1_gen,
        ));

        train_btree_index(
            fragment1_data_source,
            &sub_index_trainer,
            fragment_store.as_ref(),
            DEFAULT_BTREE_BATCH_SIZE,
            Some(vec![1]), // fragment_id = 1
            None,
        )
        .await
        .unwrap();

        // Create fragment 2 index - second half of the data (DEFAULT_BTREE_BATCH_SIZE to 2*DEFAULT_BTREE_BATCH_SIZE-1)
        let start_val = DEFAULT_BTREE_BATCH_SIZE as i32;
        let end_val = (2 * DEFAULT_BTREE_BATCH_SIZE) as i32;
        let values_second_half: Vec<i32> = (start_val..end_val).collect();
        let row_ids_second_half: Vec<u64> = (start_val as u64..end_val as u64).collect();
        let fragment2_gen = gen_batch()
            .col("value", array::cycle::<Int32Type>(values_second_half))
            .col("_rowid", array::cycle::<UInt64Type>(row_ids_second_half))
            .into_df_stream(RowCount::from(half_count), BatchCount::from(1));
        let fragment2_data_source = Box::pin(RecordBatchStreamAdapter::new(
            fragment2_gen.schema(),
            fragment2_gen,
        ));

        train_btree_index(
            fragment2_data_source,
            &sub_index_trainer,
            fragment_store.as_ref(),
            DEFAULT_BTREE_BATCH_SIZE,
            Some(vec![2]), // fragment_id = 2
            None,
        )
        .await
        .unwrap();

        // Merge the fragment files
        let part_page_files = vec![
            part_page_data_file_path(1 << 32),
            part_page_data_file_path(2 << 32),
        ];

        let part_lookup_files = vec![
            part_lookup_file_path(1 << 32),
            part_lookup_file_path(2 << 32),
        ];

        super::merge_metadata_files(
            fragment_store.clone(),
            &part_page_files,
            &part_lookup_files,
            Option::from(1usize),
        )
        .await
        .unwrap();

        // Load both indexes
        let full_index = BTreeIndex::load(full_store.clone(), None, &LanceCache::no_cache())
            .await
            .unwrap();

        let merged_index = BTreeIndex::load(fragment_store.clone(), None, &LanceCache::no_cache())
            .await
            .unwrap();

        // Test queries one by one to identify the exact problem

        // Test 1: Query for value 0 (should be in first page)
        let query_0 = SargableQuery::Equals(ScalarValue::Int32(Some(0)));
        let full_result_0 = full_index
            .search(&query_0, &NoOpMetricsCollector)
            .await
            .unwrap();
        let merged_result_0 = merged_index
            .search(&query_0, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(full_result_0, merged_result_0, "Query for value 0 failed");

        // Test 2: Query for value in middle of first batch (should be in first page)
        let mid_first_batch = (DEFAULT_BTREE_BATCH_SIZE / 2) as i32;
        let query_mid_first = SargableQuery::Equals(ScalarValue::Int32(Some(mid_first_batch)));
        let full_result_mid_first = full_index
            .search(&query_mid_first, &NoOpMetricsCollector)
            .await
            .unwrap();
        let merged_result_mid_first = merged_index
            .search(&query_mid_first, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            full_result_mid_first, merged_result_mid_first,
            "Query for value {} failed",
            mid_first_batch
        );

        // Test 3: Query for first value in second batch (should be in second page)
        let first_second_batch = DEFAULT_BTREE_BATCH_SIZE as i32;
        let query_first_second =
            SargableQuery::Equals(ScalarValue::Int32(Some(first_second_batch)));
        let full_result_first_second = full_index
            .search(&query_first_second, &NoOpMetricsCollector)
            .await
            .unwrap();
        let merged_result_first_second = merged_index
            .search(&query_first_second, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            full_result_first_second, merged_result_first_second,
            "Query for value {} failed",
            first_second_batch
        );

        // Test 4: Query for value in middle of second batch (should be in second page)
        let mid_second_batch = (DEFAULT_BTREE_BATCH_SIZE + DEFAULT_BTREE_BATCH_SIZE / 2) as i32;
        let query_mid_second = SargableQuery::Equals(ScalarValue::Int32(Some(mid_second_batch)));

        let full_result_mid_second = full_index
            .search(&query_mid_second, &NoOpMetricsCollector)
            .await
            .unwrap();
        let merged_result_mid_second = merged_index
            .search(&query_mid_second, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            full_result_mid_second, merged_result_mid_second,
            "Query for value {} failed",
            mid_second_batch
        );
    }

    #[tokio::test]
    async fn test_fragment_btree_index_boundary_queries() {
        // Setup stores for both indexes
        let full_tmpdir = TempObjDir::default();
        let full_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            full_tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let fragment_tmpdir = TempObjDir::default();
        let fragment_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            fragment_tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let sub_index_trainer = FlatIndexMetadata::new(DataType::Int32);

        // Use 3 * DEFAULT_BTREE_BATCH_SIZE for more comprehensive boundary testing
        let total_count = 3 * DEFAULT_BTREE_BATCH_SIZE;

        // Method 1: Build complete index directly
        let full_data_gen = gen_batch()
            .col("value", array::step::<Int32Type>())
            .col("_rowid", array::step::<UInt64Type>())
            .into_df_stream(RowCount::from(total_count / 3), BatchCount::from(3));
        let full_data_source = Box::pin(RecordBatchStreamAdapter::new(
            full_data_gen.schema(),
            full_data_gen,
        ));

        train_btree_index(
            full_data_source,
            &sub_index_trainer,
            full_store.as_ref(),
            DEFAULT_BTREE_BATCH_SIZE,
            None,
            None,
        )
        .await
        .unwrap();

        // Method 2: Build fragment-based index using 3 fragments
        // Fragment 1: 0 to DEFAULT_BTREE_BATCH_SIZE-1
        let fragment_size = DEFAULT_BTREE_BATCH_SIZE;
        let fragment1_gen = gen_batch()
            .col("value", array::step::<Int32Type>())
            .col("_rowid", array::step::<UInt64Type>())
            .into_df_stream(RowCount::from(fragment_size), BatchCount::from(1));
        let fragment1_data_source = Box::pin(RecordBatchStreamAdapter::new(
            fragment1_gen.schema(),
            fragment1_gen,
        ));

        train_btree_index(
            fragment1_data_source,
            &sub_index_trainer,
            fragment_store.as_ref(),
            DEFAULT_BTREE_BATCH_SIZE,
            Some(vec![1]),
            None,
        )
        .await
        .unwrap();

        // Fragment 2: DEFAULT_BTREE_BATCH_SIZE to 2*DEFAULT_BTREE_BATCH_SIZE-1
        let start_val2 = DEFAULT_BTREE_BATCH_SIZE as i32;
        let end_val2 = (2 * DEFAULT_BTREE_BATCH_SIZE) as i32;
        let values_fragment2: Vec<i32> = (start_val2..end_val2).collect();
        let row_ids_fragment2: Vec<u64> = (start_val2 as u64..end_val2 as u64).collect();
        let fragment2_gen = gen_batch()
            .col("value", array::cycle::<Int32Type>(values_fragment2))
            .col("_rowid", array::cycle::<UInt64Type>(row_ids_fragment2))
            .into_df_stream(RowCount::from(fragment_size), BatchCount::from(1));
        let fragment2_data_source = Box::pin(RecordBatchStreamAdapter::new(
            fragment2_gen.schema(),
            fragment2_gen,
        ));

        train_btree_index(
            fragment2_data_source,
            &sub_index_trainer,
            fragment_store.as_ref(),
            DEFAULT_BTREE_BATCH_SIZE,
            Some(vec![2]),
            None,
        )
        .await
        .unwrap();

        // Fragment 3: 2*DEFAULT_BTREE_BATCH_SIZE to 3*DEFAULT_BTREE_BATCH_SIZE-1
        let start_val3 = (2 * DEFAULT_BTREE_BATCH_SIZE) as i32;
        let end_val3 = (3 * DEFAULT_BTREE_BATCH_SIZE) as i32;
        let values_fragment3: Vec<i32> = (start_val3..end_val3).collect();
        let row_ids_fragment3: Vec<u64> = (start_val3 as u64..end_val3 as u64).collect();
        let fragment3_gen = gen_batch()
            .col("value", array::cycle::<Int32Type>(values_fragment3))
            .col("_rowid", array::cycle::<UInt64Type>(row_ids_fragment3))
            .into_df_stream(RowCount::from(fragment_size), BatchCount::from(1));
        let fragment3_data_source = Box::pin(RecordBatchStreamAdapter::new(
            fragment3_gen.schema(),
            fragment3_gen,
        ));

        train_btree_index(
            fragment3_data_source,
            &sub_index_trainer,
            fragment_store.as_ref(),
            DEFAULT_BTREE_BATCH_SIZE,
            Some(vec![3]),
            None,
        )
        .await
        .unwrap();

        // Merge all fragment files
        let part_page_files = vec![
            part_page_data_file_path(1 << 32),
            part_page_data_file_path(2 << 32),
            part_page_data_file_path(3 << 32),
        ];

        let part_lookup_files = vec![
            part_lookup_file_path(1 << 32),
            part_lookup_file_path(2 << 32),
            part_lookup_file_path(3 << 32),
        ];

        super::merge_metadata_files(
            fragment_store.clone(),
            &part_page_files,
            &part_lookup_files,
            Option::from(1usize),
        )
        .await
        .unwrap();

        // Load both indexes
        let full_index = BTreeIndex::load(full_store.clone(), None, &LanceCache::no_cache())
            .await
            .unwrap();

        let merged_index = BTreeIndex::load(fragment_store.clone(), None, &LanceCache::no_cache())
            .await
            .unwrap();

        // === Boundary Value Tests ===

        // Test 1: Query minimum value (boundary: data start)
        let query_min = SargableQuery::Equals(ScalarValue::Int32(Some(0)));
        let full_result_min = full_index
            .search(&query_min, &NoOpMetricsCollector)
            .await
            .unwrap();
        let merged_result_min = merged_index
            .search(&query_min, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            full_result_min, merged_result_min,
            "Query for minimum value 0 failed"
        );

        // Test 2: Query maximum value (boundary: data end)
        let max_val = (3 * DEFAULT_BTREE_BATCH_SIZE - 1) as i32;
        let query_max = SargableQuery::Equals(ScalarValue::Int32(Some(max_val)));
        let full_result_max = full_index
            .search(&query_max, &NoOpMetricsCollector)
            .await
            .unwrap();
        let merged_result_max = merged_index
            .search(&query_max, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            full_result_max, merged_result_max,
            "Query for maximum value {} failed",
            max_val
        );

        // Test 3: Query fragment boundary value (last value of first fragment)
        let fragment1_last = (DEFAULT_BTREE_BATCH_SIZE - 1) as i32;
        let query_frag1_last = SargableQuery::Equals(ScalarValue::Int32(Some(fragment1_last)));
        let full_result_frag1_last = full_index
            .search(&query_frag1_last, &NoOpMetricsCollector)
            .await
            .unwrap();
        let merged_result_frag1_last = merged_index
            .search(&query_frag1_last, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            full_result_frag1_last, merged_result_frag1_last,
            "Query for fragment 1 last value {} failed",
            fragment1_last
        );

        // Test 4: Query fragment boundary value (first value of second fragment)
        let fragment2_first = DEFAULT_BTREE_BATCH_SIZE as i32;
        let query_frag2_first = SargableQuery::Equals(ScalarValue::Int32(Some(fragment2_first)));
        let full_result_frag2_first = full_index
            .search(&query_frag2_first, &NoOpMetricsCollector)
            .await
            .unwrap();
        let merged_result_frag2_first = merged_index
            .search(&query_frag2_first, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            full_result_frag2_first, merged_result_frag2_first,
            "Query for fragment 2 first value {} failed",
            fragment2_first
        );

        // Test 5: Query fragment boundary value (last value of second fragment)
        let fragment2_last = (2 * DEFAULT_BTREE_BATCH_SIZE - 1) as i32;
        let query_frag2_last = SargableQuery::Equals(ScalarValue::Int32(Some(fragment2_last)));
        let full_result_frag2_last = full_index
            .search(&query_frag2_last, &NoOpMetricsCollector)
            .await
            .unwrap();
        let merged_result_frag2_last = merged_index
            .search(&query_frag2_last, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            full_result_frag2_last, merged_result_frag2_last,
            "Query for fragment 2 last value {} failed",
            fragment2_last
        );

        // Test 6: Query fragment boundary value (first value of third fragment)
        let fragment3_first = (2 * DEFAULT_BTREE_BATCH_SIZE) as i32;
        let query_frag3_first = SargableQuery::Equals(ScalarValue::Int32(Some(fragment3_first)));
        let full_result_frag3_first = full_index
            .search(&query_frag3_first, &NoOpMetricsCollector)
            .await
            .unwrap();
        let merged_result_frag3_first = merged_index
            .search(&query_frag3_first, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            full_result_frag3_first, merged_result_frag3_first,
            "Query for fragment 3 first value {} failed",
            fragment3_first
        );

        // === Non-existent Value Tests ===

        // Test 7: Query value below minimum
        let query_below_min = SargableQuery::Equals(ScalarValue::Int32(Some(-1)));
        let full_result_below = full_index
            .search(&query_below_min, &NoOpMetricsCollector)
            .await
            .unwrap();
        let merged_result_below = merged_index
            .search(&query_below_min, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            full_result_below, merged_result_below,
            "Query for value below minimum (-1) failed"
        );

        // Test 8: Query value above maximum
        let query_above_max = SargableQuery::Equals(ScalarValue::Int32(Some(max_val + 1)));
        let full_result_above = full_index
            .search(&query_above_max, &NoOpMetricsCollector)
            .await
            .unwrap();
        let merged_result_above = merged_index
            .search(&query_above_max, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            full_result_above,
            merged_result_above,
            "Query for value above maximum ({}) failed",
            max_val + 1
        );

        // === Range Query Tests ===

        // Test 9: Cross-fragment range query (from first fragment to second fragment)
        let range_start = (DEFAULT_BTREE_BATCH_SIZE - 100) as i32;
        let range_end = (DEFAULT_BTREE_BATCH_SIZE + 100) as i32;
        let query_cross_frag = SargableQuery::Range(
            std::collections::Bound::Included(ScalarValue::Int32(Some(range_start))),
            std::collections::Bound::Excluded(ScalarValue::Int32(Some(range_end))),
        );
        let full_result_cross = full_index
            .search(&query_cross_frag, &NoOpMetricsCollector)
            .await
            .unwrap();
        let merged_result_cross = merged_index
            .search(&query_cross_frag, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            full_result_cross, merged_result_cross,
            "Cross-fragment range query [{}, {}] failed",
            range_start, range_end
        );

        // Test 10: Range query within single fragment
        let single_frag_start = 100i32;
        let single_frag_end = 200i32;
        let query_single_frag = SargableQuery::Range(
            std::collections::Bound::Included(ScalarValue::Int32(Some(single_frag_start))),
            std::collections::Bound::Excluded(ScalarValue::Int32(Some(single_frag_end))),
        );
        let full_result_single = full_index
            .search(&query_single_frag, &NoOpMetricsCollector)
            .await
            .unwrap();
        let merged_result_single = merged_index
            .search(&query_single_frag, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            full_result_single, merged_result_single,
            "Single fragment range query [{}, {}] failed",
            single_frag_start, single_frag_end
        );

        // Test 11: Large range query spanning all fragments
        let large_range_start = 100i32;
        let large_range_end = (3 * DEFAULT_BTREE_BATCH_SIZE - 100) as i32;
        let query_large_range = SargableQuery::Range(
            std::collections::Bound::Included(ScalarValue::Int32(Some(large_range_start))),
            std::collections::Bound::Excluded(ScalarValue::Int32(Some(large_range_end))),
        );
        let full_result_large = full_index
            .search(&query_large_range, &NoOpMetricsCollector)
            .await
            .unwrap();
        let merged_result_large = merged_index
            .search(&query_large_range, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            full_result_large, merged_result_large,
            "Large range query [{}, {}] failed",
            large_range_start, large_range_end
        );

        // === Range Boundary Query Tests ===

        // Test 12: Less than query (implemented using range query, from minimum to specified value)
        let lt_val = (DEFAULT_BTREE_BATCH_SIZE / 2) as i32;
        let query_lt = SargableQuery::Range(
            std::collections::Bound::Included(ScalarValue::Int32(Some(0))),
            std::collections::Bound::Excluded(ScalarValue::Int32(Some(lt_val))),
        );
        let full_result_lt = full_index
            .search(&query_lt, &NoOpMetricsCollector)
            .await
            .unwrap();
        let merged_result_lt = merged_index
            .search(&query_lt, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            full_result_lt, merged_result_lt,
            "Less than query (<{}) failed",
            lt_val
        );

        // Test 13: Greater than query (implemented using range query, from specified value to maximum)
        let gt_val = (2 * DEFAULT_BTREE_BATCH_SIZE) as i32;
        let max_range_val = (3 * DEFAULT_BTREE_BATCH_SIZE) as i32;
        let query_gt = SargableQuery::Range(
            std::collections::Bound::Excluded(ScalarValue::Int32(Some(gt_val))),
            std::collections::Bound::Excluded(ScalarValue::Int32(Some(max_range_val))),
        );
        let full_result_gt = full_index
            .search(&query_gt, &NoOpMetricsCollector)
            .await
            .unwrap();
        let merged_result_gt = merged_index
            .search(&query_gt, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            full_result_gt, merged_result_gt,
            "Greater than query (>{}) failed",
            gt_val
        );

        // Test 14: Less than or equal query (implemented using range query, including boundary value)
        let lte_val = (DEFAULT_BTREE_BATCH_SIZE - 1) as i32;
        let query_lte = SargableQuery::Range(
            std::collections::Bound::Included(ScalarValue::Int32(Some(0))),
            std::collections::Bound::Included(ScalarValue::Int32(Some(lte_val))),
        );
        let full_result_lte = full_index
            .search(&query_lte, &NoOpMetricsCollector)
            .await
            .unwrap();
        let merged_result_lte = merged_index
            .search(&query_lte, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            full_result_lte, merged_result_lte,
            "Less than or equal query (<={}) failed",
            lte_val
        );

        // Test 15: Greater than or equal query (implemented using range query, including boundary value)
        let gte_val = (2 * DEFAULT_BTREE_BATCH_SIZE) as i32;
        let query_gte = SargableQuery::Range(
            std::collections::Bound::Included(ScalarValue::Int32(Some(gte_val))),
            std::collections::Bound::Excluded(ScalarValue::Int32(Some(max_range_val))),
        );
        let full_result_gte = full_index
            .search(&query_gte, &NoOpMetricsCollector)
            .await
            .unwrap();
        let merged_result_gte = merged_index
            .search(&query_gte, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            full_result_gte, merged_result_gte,
            "Greater than or equal query (>={}) failed",
            gte_val
        );
    }

    #[test]
    fn test_extract_partition_id() {
        // Test valid partition file names
        assert_eq!(
            super::extract_partition_id("part_123_page_data.lance").unwrap(),
            123
        );
        assert_eq!(
            super::extract_partition_id("part_456_page_lookup.lance").unwrap(),
            456
        );
        assert_eq!(
            super::extract_partition_id("part_4294967296_page_data.lance").unwrap(),
            4294967296
        );

        // Test invalid file names
        assert!(super::extract_partition_id("invalid_filename.lance").is_err());
        assert!(super::extract_partition_id("part_abc_page_data.lance").is_err());
        assert!(super::extract_partition_id("part_123").is_err());
        assert!(super::extract_partition_id("part_").is_err());
    }

    #[tokio::test]
    async fn test_cleanup_partition_files() {
        // Create a test store
        let tmpdir = TempObjDir::default();
        let test_store: Arc<dyn crate::scalar::IndexStore> = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Test files with different patterns
        let lookup_files = vec![
            "part_123_page_lookup.lance".to_string(),
            "invalid_lookup_file.lance".to_string(),
            "part_456_page_lookup.lance".to_string(),
        ];

        let page_files = vec![
            "part_123_page_data.lance".to_string(),
            "invalid_page_file.lance".to_string(),
            "part_456_page_data.lance".to_string(),
        ];

        // The cleanup function should handle both valid and invalid file patterns gracefully
        // This test mainly verifies that the function doesn't panic and handles edge cases
        super::cleanup_partition_files(&test_store, &lookup_files, &page_files).await;
    }

    #[tokio::test]
    async fn test_range_btree_index_consistency() {
        // Setup stores for both indexes
        let full_tmpdir = TempObjDir::default();
        let full_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            full_tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let range_tmpdir = TempObjDir::default();
        let range_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            range_tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let sub_index_trainer = FlatIndexMetadata::new(DataType::Int32);

        // Method 1: Build complete index directly using the same data
        // Create deterministic data for comparison - use 4 * DEFAULT_BTREE_BATCH_SIZE for testing
        let total_count = 4 * DEFAULT_BTREE_BATCH_SIZE;
        let full_data_gen = gen_batch()
            .col("value", array::step::<Int32Type>())
            .col("_rowid", array::step::<UInt64Type>())
            .into_df_stream(RowCount::from(total_count / 4), BatchCount::from(4));
        let full_data_source = Box::pin(RecordBatchStreamAdapter::new(
            full_data_gen.schema(),
            full_data_gen,
        ));

        train_btree_index(
            full_data_source,
            &sub_index_trainer,
            full_store.as_ref(),
            DEFAULT_BTREE_BATCH_SIZE,
            None,
            None,
        )
        .await
        .unwrap();

        // Method 2: Build range-based index using the same data split into ranges
        // Create range 1 index - first half of the data (0 to 2 * DEFAULT_BTREE_BATCH_SIZE - 1)
        let half_count = 2 * DEFAULT_BTREE_BATCH_SIZE;
        let range1_gen = gen_batch()
            .col("value", array::step::<Int32Type>())
            .col("_rowid", array::step::<UInt64Type>())
            .into_df_stream(RowCount::from(half_count / 2), BatchCount::from(2));
        let range1_data_source = Box::pin(RecordBatchStreamAdapter::new(
            range1_gen.schema(),
            range1_gen,
        ));

        train_btree_index(
            range1_data_source,
            &sub_index_trainer,
            range_store.as_ref(),
            DEFAULT_BTREE_BATCH_SIZE,
            None,
            Option::from(1u32),
        )
        .await
        .unwrap();

        // Create range 2 index - second half of the data (2 * DEFAULT_BTREE_BATCH_SIZE to 4 * DEFAULT_BTREE_BATCH_SIZE - 1)
        let start_val = (2 * DEFAULT_BTREE_BATCH_SIZE) as i32;
        let end_val = (4 * DEFAULT_BTREE_BATCH_SIZE) as i32;
        let values_second_half: Vec<i32> = (start_val..end_val).collect();
        let row_ids_second_half: Vec<u64> = (start_val as u64..end_val as u64).collect();
        let range2_gen = gen_batch()
            .col("value", array::cycle::<Int32Type>(values_second_half))
            .col("_rowid", array::cycle::<UInt64Type>(row_ids_second_half))
            .into_df_stream(RowCount::from(half_count), BatchCount::from(1));
        let range2_data_source = Box::pin(RecordBatchStreamAdapter::new(
            range2_gen.schema(),
            range2_gen,
        ));

        train_btree_index(
            range2_data_source,
            &sub_index_trainer,
            range_store.as_ref(),
            DEFAULT_BTREE_BATCH_SIZE,
            None,
            Option::from(2u32),
        )
        .await
        .unwrap();

        // Merge the fragment files
        let part_page_files = vec![
            part_page_data_file_path(1 << 32),
            part_page_data_file_path(2 << 32),
        ];

        let part_lookup_files = vec![
            part_lookup_file_path(1 << 32),
            part_lookup_file_path(2 << 32),
        ];

        super::merge_metadata_files(
            range_store.clone(),
            &part_page_files,
            &part_lookup_files,
            Option::from(1usize),
        )
        .await
        .unwrap();

        let full_lookup_reader = full_store.open_index_file(BTREE_LOOKUP_NAME).await.unwrap();
        let range_lookup_reader = range_store
            .open_index_file(BTREE_LOOKUP_NAME)
            .await
            .unwrap();
        assert_eq!(
            full_lookup_reader.num_rows(),
            range_lookup_reader.num_rows()
        );
        let full_lookup_batch = range_lookup_reader
            .read_record_batch(0, DEFAULT_BTREE_BATCH_SIZE)
            .await
            .unwrap();
        let range_lookup_batch = range_lookup_reader
            .read_record_batch(0, DEFAULT_BTREE_BATCH_SIZE)
            .await
            .unwrap();
        assert_eq!(full_lookup_batch, range_lookup_batch);
    }
}
