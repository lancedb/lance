// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::index::DatasetIndexInternalExt;
use lance_core::utils::mask::RowIdTreeMap;
use lance_core::Result;

use super::FileFragment;

use arrow_schema::Schema as ArrowSchema;
use datafusion::logical_expr::Expr;
use lance_datafusion::planner::Planner;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::expression::{IndexExprResult, PlannerIndexExt};
use std::sync::Arc;

impl FileFragment {
    /// Approximate count of rows in this fragment given a RowIdTreeMap filter.
    ///
    /// - If the filter marks this fragment as full, returns the live row count
    ///   (physical rows minus deletions).
    /// - If the filter contains a partial bitmap for this fragment, returns the exact
    ///   number of selected rows from the bitmap.
    /// - If the filter does not reference this fragment, returns 0.
    pub async fn approx_count_rows_with_map(
        &self,
        filter_row_id_map: &RowIdTreeMap,
    ) -> Result<usize> {
        let frag_id = self.id() as u32;
        if filter_row_id_map.is_fragment_full(frag_id) {
            let total_rows = self.physical_rows().await?;
            let deleted = self.count_deletions().await?;
            Ok(total_rows - deleted)
        } else if let Some(bitmap) = filter_row_id_map.get_fragment_bitmap(frag_id) {
            Ok(bitmap.len() as usize)
        } else {
            Ok(0)
        }
    }

    /// Approximate live row count with an optional filter.
    ///
    /// When `filter` is None, preserves previous behavior (returns exact count of live rows).
    /// When `filter` is provided:
    /// - If we can derive an index-based selection (RowIdMask) for this fragment, we return:
    ///   - partial selections: exact count of selected rows
    ///   - full selections: fragment live rows (physical - deletions)
    ///   - mixed selections: exact count for partial, plus full fragment approximation
    /// - If we cannot derive an index selection, we fall back to exact counting via `count_rows(filter)`.
    pub async fn approx_count_rows(&self, filter: Option<String>) -> Result<usize> {
        match filter {
            None => self.count_rows(None).await,
            Some(sql) => {
                // Parse SQL -> DF Expr using the same planner as Scanner
                let full_schema = ArrowSchema::from(self.dataset().schema());
                let planner = Planner::new(Arc::new(full_schema));
                let df_expr: Expr = planner.parse_filter(&sql)?;
                // Build filter plan using index information
                let idx_info = self.dataset().scalar_index_info().await?;
                let plan = planner.create_filter_plan(
                    df_expr.clone(),
                    &idx_info,
                    /*use_scalar_index=*/ true,
                )?;
                if let Some(index_expr) = plan.index_query {
                    // Evaluate index expression to get a RowIdMask (Exact/AtMost)
                    let metrics = NoOpMetricsCollector;
                    let res = index_expr.evaluate(self.dataset(), &metrics).await?;
                    let mut mask = match res {
                        IndexExprResult::Exact(m) => m,
                        IndexExprResult::AtMost(m) => m,
                        IndexExprResult::AtLeast(_m) => {
                            // AtLeast semantics are rare here; fall back to exact counting
                            return self.count_rows(Some(sql)).await;
                        }
                    };
                    let frag_id = self.id() as u32;
                    // Restrict mask to current fragment
                    if let Some(mut al) = mask.allow_list.take() {
                        al.retain_fragments([frag_id]);
                        mask.allow_list = Some(al);
                    }
                    if let Some(mut bl) = mask.block_list.take() {
                        bl.retain_fragments([frag_id]);
                        mask.block_list = Some(bl);
                    }
                    // If block list fully covers fragment, result is 0
                    if let Some(bl) = &mask.block_list {
                        if bl.is_fragment_full(frag_id) {
                            log::info!(
                                "approx_count_rows: fragment={} path=full_block => 0",
                                frag_id
                            );
                            return Ok(0);
                        }
                    }
                    // If allow_list exists, compute per semantics
                    if let Some(al) = &mask.allow_list {
                        if al.is_fragment_full(frag_id) {
                            // full fragment allowed: live rows (physical - deletions)
                            let total_rows = self.physical_rows().await?;
                            let deleted = self.count_deletions().await?;
                            let approx = total_rows - deleted;
                            log::info!(
                                "approx_count_rows: fragment={} path=full_allow => {}",
                                frag_id,
                                approx
                            );
                            Ok(approx)
                        } else if let Some(bitmap) = al.get_fragment_bitmap(frag_id) {
                            let exact = bitmap.len() as usize;
                            log::info!(
                                "approx_count_rows: fragment={} path=partial_allow => {}",
                                frag_id,
                                exact
                            );
                            Ok(exact)
                        } else {
                            // allow_list does not reference this fragment
                            log::info!("approx_count_rows: fragment={} path=no_ref => 0", frag_id);
                            Ok(0)
                        }
                    } else {
                        // No allow list — treat as full selection unless blocked; estimate live rows minus known partial blocks
                        let total_rows = self.physical_rows().await?;
                        let deleted = self.count_deletions().await?;
                        let mut approx = total_rows - deleted;
                        if let Some(bl) = &mask.block_list {
                            if let Some(bitmap) = bl.get_fragment_bitmap(frag_id) {
                                // subtract known blocked rows for tighter estimate
                                approx = approx.saturating_sub(bitmap.len() as usize);
                                log::info!("approx_count_rows: fragment={} path=full_allow_minus_partial_block => {}", frag_id, approx);
                            } else {
                                log::info!(
                                    "approx_count_rows: fragment={} path=full_allow_no_block => {}",
                                    frag_id,
                                    approx
                                );
                            }
                        } else {
                            log::info!(
                                "approx_count_rows: fragment={} path=full_allow_no_block => {}",
                                frag_id,
                                approx
                            );
                        }
                        Ok(approx)
                    }
                } else {
                    // No index assist path => exact fallback using count_rows(filter)
                    self.count_rows(Some(sql)).await
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::UInt64Type;
    use lance_datagen::gen_batch;
    use lance_index::{
        scalar::zonemap::ZoneMapIndexBuilderParams, scalar::BuiltinIndexType,
        scalar::ScalarIndexParams, DatasetIndexExt, IndexType,
    };

    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use crate::Dataset;

    async fn make_dataset() -> Dataset {
        // Create dataset with a single UInt64 column 'ordered', 2 fragments x 10 rows
        let tmp = tempfile::tempdir().unwrap();
        let uri = tmp.path().to_str().unwrap();
        let mut ds = gen_batch()
            .col("ordered", lance_datagen::array::step::<UInt64Type>())
            .into_dataset(uri, FragmentCount::from(2), FragmentRowCount::from(10))
            .await
            .unwrap();
        ds.create_index(
            &["ordered"],
            IndexType::BTree,
            Some("ordered_idx".to_string()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
        ds
    }

    #[tokio::test]
    async fn test_approx_no_filter_aligns_with_exact() {
        let ds = make_dataset().await;
        let frag = ds.get_fragments().into_iter().next().unwrap();
        let approx = frag.approx_count_rows(None).await.unwrap();
        let exact = frag.count_rows(None).await.unwrap();
        assert_eq!(approx, exact);
    }

    #[tokio::test]
    async fn test_approx_count_rows_without_filter() {
        let ds = make_dataset().await;
        let frag = ds.get_fragments().into_iter().next().unwrap();

        let approx_count = frag.approx_count_rows(None).await.unwrap();
        let exact_count = frag.count_rows(None).await.unwrap();

        assert_eq!(approx_count, exact_count);
        assert_eq!(approx_count, 10);
    }

    #[tokio::test]
    async fn test_approx_count_rows_with_filter() {
        // Create dataset without indices
        let tmp = tempfile::tempdir().unwrap();
        let uri = tmp.path().to_str().unwrap();
        let ds = gen_batch()
            .col("ordered", lance_datagen::array::step::<UInt64Type>())
            .into_dataset(uri, FragmentCount::from(2), FragmentRowCount::from(10))
            .await
            .unwrap();

        let frag = ds.get_fragments().into_iter().next().unwrap();

        let approx_count = frag
            .approx_count_rows(Some("ordered >= 0 AND ordered < 10".to_string()))
            .await
            .unwrap();

        assert_eq!(approx_count, 10);
    }

    #[tokio::test]
    async fn test_approx_count_rows_with_zonemap_index() {
        let tmp = tempfile::tempdir().unwrap();
        let uri = tmp.path().to_str().unwrap();
        let mut ds = gen_batch()
            .col("ordered", lance_datagen::array::step::<UInt64Type>())
            .into_dataset(uri, FragmentCount::from(2), FragmentRowCount::from(10))
            .await
            .unwrap();

        let zonemap_params = ZoneMapIndexBuilderParams::new(5);
        let index_params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap)
            .with_params(&serde_json::to_value(&zonemap_params).unwrap());

        ds.create_index(
            &["ordered"],
            IndexType::ZoneMap,
            Some("ordered_zonemap".to_string()),
            &index_params,
            true,
        )
        .await
        .unwrap();

        let frag = ds.get_fragments().into_iter().next().unwrap();

        let approx_count = frag
            .approx_count_rows(Some("ordered < 5".to_string()))
            .await
            .unwrap();

        assert_eq!(approx_count, 5);
    }

    #[tokio::test]
    async fn test_approx_count_rows_with_bloomfilter_index() {
        let tmp = tempfile::tempdir().unwrap();
        let uri = tmp.path().to_str().unwrap();
        let mut ds = gen_batch()
            .col("ordered", lance_datagen::array::step::<UInt64Type>())
            .into_dataset(uri, FragmentCount::from(2), FragmentRowCount::from(10))
            .await
            .unwrap();

        let bloomfilter_params =
            lance_index::scalar::bloomfilter::BloomFilterIndexBuilderParams::default();

        let index_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BloomFilter)
            .with_params(&serde_json::to_value(&bloomfilter_params).unwrap());

        ds.create_index(
            &["ordered"],
            IndexType::BloomFilter,
            Some("ordered_bloomfilter".to_string()),
            &index_params,
            true,
        )
        .await
        .unwrap();

        let frag = ds.get_fragments().into_iter().next().unwrap();

        let approx_count = frag
            .approx_count_rows(Some("ordered < 5".to_string()))
            .await
            .unwrap();

        assert!(approx_count <= 10);
    }
}
