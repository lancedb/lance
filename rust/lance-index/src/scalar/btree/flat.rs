// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::{ops::Bound, sync::Arc};

use arrow_array::Array;
use arrow_array::{
    cast::AsArray, types::UInt64Type, ArrayRef, BooleanArray, RecordBatch, UInt64Array,
};

use datafusion_common::DFSchema;
use datafusion_expr::execution_props::ExecutionProps;
use datafusion_physical_expr::create_physical_expr;
use deepsize::DeepSizeOf;
use lance_arrow::RecordBatchExt;
use lance_core::utils::address::RowAddress;
use lance_core::utils::mask::{NullableRowAddrSet, RowAddrTreeMap};
use lance_core::Result;
use roaring::RoaringBitmap;
use tracing::instrument;

use crate::metrics::MetricsCollector;
use crate::scalar::btree::BTREE_VALUES_COLUMN;
use crate::scalar::{AnyQuery, SargableQuery};

const VALUES_COL_IDX: usize = 0;
const IDS_COL_IDX: usize = 1;
/// A flat index is just a batch of value/row-id pairs
///
/// The batch always has two columns.  The first column "values" contains
/// the values.  The second column "row_ids" contains the row ids
///
/// Evaluating a query requires O(N) time where N is the # of rows
#[derive(Debug)]
pub struct FlatIndex {
    data: Arc<RecordBatch>,
    all_addrs_map: RowAddrTreeMap,
    null_addrs_map: RowAddrTreeMap,
    df_schema: DFSchema,
}

impl DeepSizeOf for FlatIndex {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.data.get_array_memory_size()
    }
}

impl FlatIndex {
    #[instrument(name = "FlatIndex::try_new", level = "debug", skip_all)]
    pub fn try_new(data: RecordBatch) -> Result<Self> {
        // Sort by row id to make bitmap construction more efficient
        let data = data.sort_by_column(IDS_COL_IDX, None)?;

        let has_nulls = data.column(VALUES_COL_IDX).null_count() > 0;
        let all_addrs_map = RowAddrTreeMap::from_sorted_iter(
            data.column(IDS_COL_IDX)
                .as_primitive::<UInt64Type>()
                .values()
                .iter()
                .copied(),
        )?;

        let null_addrs_map = if has_nulls {
            Self::get_null_addrs(&data)?
        } else {
            RowAddrTreeMap::default()
        };

        let df_schema = DFSchema::try_from(data.schema())?;

        Ok(Self {
            data: Arc::new(data),
            all_addrs_map,
            null_addrs_map,
            df_schema,
        })
    }

    fn ids(&self) -> &ArrayRef {
        self.data.column(IDS_COL_IDX)
    }

    pub fn all(&self) -> NullableRowAddrSet {
        // Some rows will be in both sets but that is ok, null trumps true
        NullableRowAddrSet::new(self.all_addrs_map.clone(), self.null_addrs_map.clone())
    }

    pub fn all_ignore_nulls(&self) -> NullableRowAddrSet {
        NullableRowAddrSet::new(self.all_addrs_map.clone(), Default::default())
    }

    pub fn remap_batch(
        batch: RecordBatch,
        mapping: &HashMap<u64, Option<u64>>,
    ) -> Result<RecordBatch> {
        let row_ids = batch.column(IDS_COL_IDX).as_primitive::<UInt64Type>();
        let val_idx_and_new_id = row_ids
            .values()
            .iter()
            .enumerate()
            .filter_map(|(idx, old_id)| {
                mapping
                    .get(old_id)
                    .copied()
                    .unwrap_or(Some(*old_id))
                    .map(|new_id| (idx, new_id))
            })
            .collect::<Vec<_>>();
        let new_ids = Arc::new(UInt64Array::from_iter_values(
            val_idx_and_new_id.iter().copied().map(|(_, new_id)| new_id),
        ));
        let new_val_indices = UInt64Array::from_iter_values(
            val_idx_and_new_id
                .into_iter()
                .map(|(val_idx, _)| val_idx as u64),
        );
        let new_vals =
            arrow_select::take::take(batch.column(VALUES_COL_IDX), &new_val_indices, None)?;
        Ok(RecordBatch::try_new(
            batch.schema(),
            vec![new_vals, new_ids],
        )?)
    }

    fn get_null_addrs(sorted_batch: &RecordBatch) -> Result<RowAddrTreeMap> {
        let null_mask = arrow::compute::is_null(sorted_batch.column(VALUES_COL_IDX))?;
        let null_ids = arrow_select::filter::filter(sorted_batch.column(IDS_COL_IDX), &null_mask)?;
        let null_ids = null_ids
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("Result of arrow_select::filter::filter did not match input type");
        RowAddrTreeMap::from_sorted_iter(null_ids.values().iter().copied())
    }

    pub fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<NullableRowAddrSet> {
        metrics.record_comparisons(self.data.num_rows());
        let query = query.as_any().downcast_ref::<SargableQuery>().unwrap();
        // Since we have all the values in memory we can use basic arrow-rs compute
        // functions to satisfy scalar queries.

        // Shortcuts for simple cases where we can re-use computed values
        match query {
            // x = NULL means all rows are NULL
            SargableQuery::Equals(value) => {
                if value.is_null() {
                    // if we have x = NULL then the correct SQL behavior is to return all NULLs
                    return Ok(NullableRowAddrSet::new(
                        Default::default(),
                        self.all_addrs_map.clone(),
                    ));
                }
            }
            // x IS NULL we can use pre-computed nulls
            SargableQuery::IsNull() => {
                return Ok(NullableRowAddrSet::new(
                    self.null_addrs_map.clone(),
                    Default::default(),
                ));
            }
            // x < NULL or x > NULL means all rows are NULL
            SargableQuery::Range(lower_bound, upper_bound) => match (lower_bound, upper_bound) {
                (Bound::Unbounded, Bound::Unbounded) => {
                    return Ok(NullableRowAddrSet::new(
                        self.all_addrs_map.clone(),
                        Default::default(),
                    ));
                }
                (Bound::Unbounded, Bound::Included(upper) | Bound::Excluded(upper)) => {
                    if upper.is_null() {
                        return Ok(NullableRowAddrSet::new(
                            Default::default(),
                            self.all_addrs_map.clone(),
                        ));
                    }
                }
                (Bound::Included(lower) | Bound::Excluded(lower), Bound::Unbounded) => {
                    if lower.is_null() {
                        return Ok(NullableRowAddrSet::new(
                            Default::default(),
                            self.all_addrs_map.clone(),
                        ));
                    }
                }
                _ => {}
            },
            _ => {}
        };

        // No shortcut possible, need to actually evaluate the query
        let expr = query.to_expr(BTREE_VALUES_COLUMN.to_string());
        let expr = create_physical_expr(&expr, &self.df_schema, &ExecutionProps::default())?;

        let predicate = expr.evaluate(&self.data)?;
        let predicate = predicate.into_array(self.data.num_rows())?;
        let predicate = predicate
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("Predicate should return boolean array");
        let nulls = arrow::compute::is_null(&predicate)?;

        let matching_ids = arrow_select::filter::filter(self.ids(), predicate)?;
        let matching_ids = matching_ids
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("Result of arrow_select::filter::filter did not match input type");
        let selected = RowAddrTreeMap::from_sorted_iter(matching_ids.values().iter().copied())?;

        let null_row_ids = arrow_select::filter::filter(self.ids(), &nulls)?;
        let null_row_ids = null_row_ids
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("Result of arrow_select::filter::filter did not match input type");
        let null_row_ids = RowAddrTreeMap::from_sorted_iter(null_row_ids.values().iter().copied())?;

        Ok(NullableRowAddrSet::new(selected, null_row_ids))
    }

    pub fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        let mut frag_ids = self
            .ids()
            .as_primitive::<UInt64Type>()
            .iter()
            .map(|row_id| RowAddress::from(row_id.unwrap()).fragment_id())
            .collect::<Vec<_>>();
        frag_ids.sort();
        frag_ids.dedup();
        Ok(RoaringBitmap::from_sorted_iter(frag_ids).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        metrics::NoOpMetricsCollector,
        scalar::btree::{BTREE_IDS_COLUMN, BTREE_VALUES_COLUMN},
    };

    use super::*;
    use arrow_array::{record_batch, types::Int32Type};
    use datafusion_common::ScalarValue;
    use lance_datagen::{array, gen_batch, RowCount};

    fn example_index() -> FlatIndex {
        let batch = gen_batch()
            .col(
                "values",
                array::cycle::<Int32Type>(vec![10, 100, 1000, 1234]),
            )
            .col("ids", array::cycle::<UInt64Type>(vec![5, 0, 3, 100]))
            .into_batch_rows(RowCount::from(4))
            .unwrap();

        FlatIndex::try_new(batch).unwrap()
    }

    async fn check_index(query: &SargableQuery, expected: &[u64]) {
        let index = example_index();
        let actual = index.search(query, &NoOpMetricsCollector).unwrap();
        let expected =
            NullableRowAddrSet::new(RowAddrTreeMap::from_iter(expected), Default::default());
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn test_equality() {
        check_index(&SargableQuery::Equals(ScalarValue::from(100)), &[0]).await;
        check_index(&SargableQuery::Equals(ScalarValue::from(10)), &[5]).await;
        check_index(&SargableQuery::Equals(ScalarValue::from(5)), &[]).await;
    }

    #[tokio::test]
    async fn test_range() {
        check_index(
            &SargableQuery::Range(
                Bound::Included(ScalarValue::from(100)),
                Bound::Excluded(ScalarValue::from(1234)),
            ),
            &[0, 3],
        )
        .await;
        check_index(
            &SargableQuery::Range(Bound::Unbounded, Bound::Excluded(ScalarValue::from(1000))),
            &[5, 0],
        )
        .await;
        check_index(
            &SargableQuery::Range(Bound::Included(ScalarValue::from(0)), Bound::Unbounded),
            &[5, 0, 3, 100],
        )
        .await;
        check_index(
            &SargableQuery::Range(Bound::Included(ScalarValue::from(100000)), Bound::Unbounded),
            &[],
        )
        .await;
    }

    #[tokio::test]
    async fn test_is_in() {
        check_index(
            &SargableQuery::IsIn(vec![
                ScalarValue::from(100),
                ScalarValue::from(1234),
                ScalarValue::from(3000),
            ]),
            &[0, 100],
        )
        .await;
    }

    #[tokio::test]
    async fn test_remap() {
        let index = example_index();
        // 0 -> 2000
        // 3 -> delete
        // Keep remaining as is
        let mapping = HashMap::<u64, Option<u64>>::from_iter(vec![(0, Some(2000)), (3, None)]);
        let remapped =
            FlatIndex::try_new(FlatIndex::remap_batch((*index.data).clone(), &mapping).unwrap())
                .unwrap();

        let expected = FlatIndex::try_new(
            gen_batch()
                .col("values", array::cycle::<Int32Type>(vec![10, 100, 1234]))
                .col("ids", array::cycle::<UInt64Type>(vec![5, 2000, 100]))
                .into_batch_rows(RowCount::from(3))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(remapped.data, expected.data);
    }

    // It's possible, during compaction, that an entire page of values is deleted.  We just serialize
    // it as an empty record batch.
    #[tokio::test]
    async fn test_remap_to_nothing() {
        let index = example_index();
        let mapping = HashMap::<u64, Option<u64>>::from_iter(vec![
            (5, None),
            (0, None),
            (3, None),
            (100, None),
        ]);
        let remapped = FlatIndex::remap_batch((*index.data).clone(), &mapping).unwrap();
        assert_eq!(remapped.num_rows(), 0);
    }

    #[test]
    fn test_null_handling() {
        // [null, 0, 5]
        let batch = record_batch!(
            (BTREE_VALUES_COLUMN, Int32, [None, Some(0), Some(5)]),
            (BTREE_IDS_COLUMN, UInt64, [0, 1, 2])
        )
        .unwrap();
        let index = FlatIndex::try_new(batch).unwrap();

        let check = |query: SargableQuery, true_ids: &[u64], null_ids: &[u64]| {
            let actual = index.search(&query, &NoOpMetricsCollector).unwrap();
            let expected = NullableRowAddrSet::new(
                RowAddrTreeMap::from_iter(true_ids),
                RowAddrTreeMap::from_iter(null_ids),
            );
            assert_eq!(actual, expected, "query: {:?}", query);
        };

        let null = ScalarValue::Int32(None);
        let zero = ScalarValue::Int32(Some(0));
        let three = ScalarValue::Int32(Some(3));

        check(SargableQuery::Equals(zero.clone()), &[1], &[0]);
        // x = NULL returns all rows as NULL and nothing as TRUE
        check(SargableQuery::Equals(null.clone()), &[], &[0, 1, 2]);

        check(SargableQuery::IsIn(vec![zero.clone()]), &[1], &[0]);
        // x IN (0, NULL) promotes all FALSE to NULL
        check(SargableQuery::IsIn(vec![zero, null.clone()]), &[1], &[0, 2]);

        check(SargableQuery::IsNull(), &[0], &[]);

        check(
            SargableQuery::Range(Bound::Included(three.clone()), Bound::Unbounded),
            &[2],
            &[0],
        );

        // x < NULL or x > NULL returns everything as NULL
        check(
            SargableQuery::Range(Bound::Unbounded, Bound::Included(null.clone())),
            &[],
            &[0, 1, 2],
        );

        check(
            SargableQuery::Range(Bound::Excluded(null.clone()), Bound::Unbounded),
            &[],
            &[0, 1, 2],
        );

        // x BETWEEN 3 AND NULL returns everything as NULL unless we know it is FALSE
        check(
            SargableQuery::Range(
                Bound::Included(three.clone()),
                Bound::Included(null.clone()),
            ),
            &[],
            &[0, 2],
        );
        check(
            SargableQuery::Range(Bound::Included(null.clone()), Bound::Included(three)),
            &[],
            &[0, 1],
        );
        check(
            SargableQuery::Range(Bound::Included(null.clone()), Bound::Included(null)),
            &[],
            &[0, 1, 2],
        );
    }
}
