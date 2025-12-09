// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::{ops::Bound, sync::Arc};

use arrow_array::{
    cast::AsArray, types::UInt64Type, ArrayRef, BooleanArray, RecordBatch, UInt64Array,
};

use datafusion_physical_expr::expressions::{in_list, lit, Column};
use deepsize::DeepSizeOf;
use lance_arrow::RecordBatchExt;
use lance_core::utils::address::RowAddress;
use lance_core::utils::mask::{NullableRowAddrSet, RowAddrTreeMap};
use lance_core::{Error, Result};
use roaring::RoaringBitmap;
use snafu::location;

use crate::metrics::MetricsCollector;
use crate::scalar::{AnyQuery, SargableQuery};

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
    has_nulls: bool,
}

impl DeepSizeOf for FlatIndex {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.data.get_array_memory_size()
    }
}

impl FlatIndex {
    pub fn try_new(data: RecordBatch) -> Result<Self> {
        // Sort by row id to make bitmap construction more efficient
        let data = data.sort_by_column(1, None)?;
        let has_nulls = data.column(1).null_count() > 0;
        let all_addrs_map = RowAddrTreeMap::from_sorted_iter(
            data.column(1)
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

        Ok(Self {
            data: Arc::new(data),
            all_addrs_map,
            null_addrs_map,
            has_nulls,
        })
    }

    fn values(&self) -> &ArrayRef {
        self.data.column(0)
    }

    fn ids(&self) -> &ArrayRef {
        self.data.column(1)
    }

    pub fn all(&self) -> NullableRowAddrSet {
        // Some rows will be in both sets but that is ok, null trumps true
        NullableRowAddrSet::new(self.all_addrs_map.clone(), self.null_addrs_map.clone())
    }

    pub fn remap_batch(
        batch: RecordBatch,
        mapping: &HashMap<u64, Option<u64>>,
    ) -> Result<RecordBatch> {
        let row_ids = batch.column(1).as_primitive::<UInt64Type>();
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
        let new_vals = arrow_select::take::take(batch.column(0), &new_val_indices, None)?;
        Ok(RecordBatch::try_new(
            batch.schema(),
            vec![new_vals, new_ids],
        )?)
    }

    fn get_null_addrs(sorted_batch: &RecordBatch) -> Result<RowAddrTreeMap> {
        let null_mask = arrow::compute::is_null(sorted_batch.column(0))?;
        let null_ids = arrow_select::filter::filter(sorted_batch.column(1), &null_mask)?;
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

        let mut null_is_true = false;
        let mut predicate = match query {
            SargableQuery::Equals(value) => {
                if value.is_null() {
                    // Query is x = NULL, correct SQL behavior is to return all ids as NULL
                    // We differ a little and return them all as true right now.
                    return Ok(NullableRowAddrSet::new(
                        self.null_addrs_map.clone(),
                        Default::default(),
                    ));
                } else {
                    arrow_ord::cmp::eq(self.values(), &value.to_scalar()?)?
                }
            }
            SargableQuery::IsNull() => {
                return Ok(NullableRowAddrSet::new(
                    self.null_addrs_map.clone(),
                    Default::default(),
                ));
            }
            SargableQuery::IsIn(values) => {
                let mut has_null = false;
                let choices = values
                    .iter()
                    .map(|val| {
                        has_null |= val.is_null();
                        lit(val.clone())
                    })
                    .collect::<Vec<_>>();
                let in_list_expr = in_list(
                    Arc::new(Column::new("values", 0)),
                    choices,
                    &false,
                    &self.data.schema(),
                )?;
                let result_col = in_list_expr.evaluate(&self.data)?;
                let predicate = result_col
                    .into_array(self.data.num_rows())?
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .expect("InList evaluation should return boolean array")
                    .clone();

                // If the IN query has nulls, then don't treat the nulls as null.  This is a little different
                // than SQL behavior.
                null_is_true = has_null;

                // Arrow's in_list does not handle nulls so we need to join them in here if user asked for them
                if has_null && self.has_nulls {
                    let nulls = arrow::compute::is_null(self.values())?;
                    arrow::compute::or(&predicate, &nulls)?
                } else {
                    predicate
                }
            }
            SargableQuery::Range(lower_bound, upper_bound) => match (lower_bound, upper_bound) {
                (Bound::Unbounded, Bound::Unbounded) => {
                    panic!("Scalar range query received with no upper or lower bound")
                }
                (Bound::Unbounded, Bound::Included(upper)) => {
                    arrow_ord::cmp::lt_eq(self.values(), &upper.to_scalar()?)?
                }
                (Bound::Unbounded, Bound::Excluded(upper)) => {
                    arrow_ord::cmp::lt(self.values(), &upper.to_scalar()?)?
                }
                (Bound::Included(lower), Bound::Unbounded) => {
                    arrow_ord::cmp::gt_eq(self.values(), &lower.to_scalar()?)?
                }
                (Bound::Included(lower), Bound::Included(upper)) => arrow::compute::and(
                    &arrow_ord::cmp::gt_eq(self.values(), &lower.to_scalar()?)?,
                    &arrow_ord::cmp::lt_eq(self.values(), &upper.to_scalar()?)?,
                )?,
                (Bound::Included(lower), Bound::Excluded(upper)) => arrow::compute::and(
                    &arrow_ord::cmp::gt_eq(self.values(), &lower.to_scalar()?)?,
                    &arrow_ord::cmp::lt(self.values(), &upper.to_scalar()?)?,
                )?,
                (Bound::Excluded(lower), Bound::Unbounded) => {
                    arrow_ord::cmp::gt(self.values(), &lower.to_scalar()?)?
                }
                (Bound::Excluded(lower), Bound::Included(upper)) => arrow::compute::and(
                    &arrow_ord::cmp::gt(self.values(), &lower.to_scalar()?)?,
                    &arrow_ord::cmp::lt_eq(self.values(), &upper.to_scalar()?)?,
                )?,
                (Bound::Excluded(lower), Bound::Excluded(upper)) => arrow::compute::and(
                    &arrow_ord::cmp::gt(self.values(), &lower.to_scalar()?)?,
                    &arrow_ord::cmp::lt(self.values(), &upper.to_scalar()?)?,
                )?,
            },
            SargableQuery::FullTextSearch(_) => return Err(Error::invalid_input(
                "full text search is not supported for flat index, build a inverted index for it",
                location!(),
            )),
        };
        if self.has_nulls && matches!(query, SargableQuery::Range(_, _)) {
            // Arrow's comparison kernels do not return false for nulls.  They consider nulls to
            // be less than any value.  So we need to filter out the nulls manually.
            let valid_values = arrow::compute::is_not_null(self.values())?;
            predicate = arrow::compute::and(&valid_values, &predicate)?;
        }

        // Track null row IDs for Kleene logic
        // When querying FOR nulls (IS NULL or Equals(null)), don't track them as "null results"
        // because they are the TRUE result of the query
        let null_row_ids = if null_is_true {
            self.null_addrs_map.clone()
        } else {
            Default::default()
        };

        let matching_ids = arrow_select::filter::filter(self.ids(), &predicate)?;
        let matching_ids = matching_ids
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("Result of arrow_select::filter::filter did not match input type");
        let selected = RowAddrTreeMap::from_sorted_iter(matching_ids.values().iter().copied())?;
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
    use crate::metrics::NoOpMetricsCollector;

    use super::*;
    use arrow_array::types::Int32Type;
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
}
