// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use async_recursion::async_recursion;
use lance_index::scalar::expression::ScalarIndexExpr;
use roaring::RoaringBitmap;

use crate::{Error, Result, dataset::Dataset, index::DatasetIndexExt};

#[async_recursion]
pub(crate) async fn fragments_covered_by_scalar_index_query(
    dataset: &Dataset,
    index_expr: &ScalarIndexExpr,
) -> Result<RoaringBitmap> {
    match index_expr {
        ScalarIndexExpr::And(lhs, rhs) => Ok(fragments_covered_by_scalar_index_query(dataset, lhs)
            .await?
            & fragments_covered_by_scalar_index_query(dataset, rhs).await?),
        ScalarIndexExpr::Or(lhs, rhs) => Ok(fragments_covered_by_scalar_index_query(dataset, lhs)
            .await?
            & fragments_covered_by_scalar_index_query(dataset, rhs).await?),
        ScalarIndexExpr::Not(expr) => fragments_covered_by_scalar_index_query(dataset, expr).await,
        ScalarIndexExpr::Query(search) => {
            let indices = dataset
                .load_indices_by_name(&search.index_name)
                .await?
                .into_iter()
                .filter_map(|index| index.fragment_bitmap)
                .collect::<Vec<_>>();
            if indices.is_empty() {
                return Err(Error::internal(format!(
                    "No scalar index segments found for logical index '{}'",
                    search.index_name
                )));
            }
            Ok(indices
                .into_iter()
                .fold(RoaringBitmap::new(), |mut covered, bitmap| {
                    covered |= bitmap;
                    covered
                }))
        }
    }
}
