// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Query-time logical views over scalar index segments.

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;
use deepsize::{Context, DeepSizeOf};
use futures::future::try_join_all;
use lance_core::utils::mask::NullableRowAddrSet;
use lance_core::{Error, Result};
use lance_index::metrics::MetricsCollector;
use lance_index::scalar::{AnyQuery, CreatedIndex, ScalarIndex, SearchResult, UpdateCriteria};
use lance_index::{Index, IndexType};
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;
use serde_json::json;

use crate::dataset::Dataset;
use crate::index::scalar::fetch_index_details;
use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};

#[derive(Debug)]
pub struct LogicalScalarIndex {
    name: String,
    column: String,
    index_type: IndexType,
    segments: Vec<Arc<dyn ScalarIndex>>,
}

impl LogicalScalarIndex {
    fn try_new(name: String, column: String, segments: Vec<Arc<dyn ScalarIndex>>) -> Result<Self> {
        let Some(first) = segments.first() else {
            return Err(Error::invalid_input(format!(
                "LogicalScalarIndex '{}' on column '{}' must contain at least one segment",
                name, column
            )));
        };
        let index_type = first.index_type();
        if segments
            .iter()
            .any(|segment| segment.index_type() != index_type)
        {
            return Err(Error::invalid_input(format!(
                "LogicalScalarIndex '{}' on column '{}' mixes scalar index types",
                name, column
            )));
        }

        Ok(Self {
            name,
            column,
            index_type,
            segments,
        })
    }
}

impl DeepSizeOf for LogicalScalarIndex {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.name.deep_size_of_children(context)
            + self.column.deep_size_of_children(context)
            + self.segments.deep_size_of_children(context)
    }
}

#[async_trait]
impl Index for LogicalScalarIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn lance_index::vector::VectorIndex>> {
        Err(Error::invalid_input(format!(
            "LogicalScalarIndex '{}' is not a vector index",
            self.name
        )))
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(json!({
            "index_name": self.name,
            "column": self.column,
            "index_type": self.index_type.to_string(),
            "num_segments": self.segments.len(),
        }))
    }

    async fn prewarm(&self) -> Result<()> {
        try_join_all(self.segments.iter().map(|segment| segment.prewarm())).await?;
        Ok(())
    }

    fn index_type(&self) -> IndexType {
        self.index_type
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        let fragment_sets = try_join_all(
            self.segments
                .iter()
                .map(|segment| segment.calculate_included_frags()),
        )
        .await?;
        let mut combined = RoaringBitmap::new();
        for fragment_set in fragment_sets {
            combined |= fragment_set;
        }
        Ok(combined)
    }
}

#[async_trait]
impl ScalarIndex for LogicalScalarIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let results = try_join_all(
            self.segments
                .iter()
                .map(|segment| segment.search(query, metrics)),
        )
        .await?;
        combine_search_results(results)
    }

    fn can_remap(&self) -> bool {
        false
    }

    async fn remap(
        &self,
        _mapping: &std::collections::HashMap<u64, Option<u64>>,
        _dest_store: &dyn lance_index::scalar::IndexStore,
    ) -> Result<CreatedIndex> {
        Err(Error::invalid_input(format!(
            "LogicalScalarIndex '{}' is a query-time wrapper and does not support remap; rebuild the index to consolidate segments before remapping",
            self.name
        )))
    }

    async fn update(
        &self,
        _new_data: datafusion::physical_plan::SendableRecordBatchStream,
        _dest_store: &dyn lance_index::scalar::IndexStore,
        _old_data_filter: Option<lance_index::scalar::OldIndexDataFilter>,
    ) -> Result<CreatedIndex> {
        Err(Error::invalid_input(format!(
            "LogicalScalarIndex '{}' is a query-time wrapper and does not support update; rebuild the index to consolidate segments before updating",
            self.name
        )))
    }

    fn update_criteria(&self) -> UpdateCriteria {
        self.segments[0].update_criteria()
    }

    fn derive_index_params(&self) -> Result<lance_index::scalar::ScalarIndexParams> {
        self.segments[0].derive_index_params()
    }
}

fn combine_search_results(results: Vec<SearchResult>) -> Result<SearchResult> {
    let mut saw_at_most = false;
    let mut saw_at_least = false;
    let mut sets = Vec::with_capacity(results.len());

    for result in results {
        match result {
            SearchResult::Exact(set) => sets.push(set),
            SearchResult::AtMost(set) => {
                saw_at_most = true;
                sets.push(set);
            }
            SearchResult::AtLeast(set) => {
                saw_at_least = true;
                sets.push(set);
            }
        }
    }

    if saw_at_most && saw_at_least {
        return Err(Error::not_supported(
            "Logical scalar index cannot combine mixed AtMost and AtLeast segment results",
        ));
    }

    let combined = NullableRowAddrSet::union_all(&sets);
    Ok(if saw_at_most {
        SearchResult::AtMost(combined)
    } else if saw_at_least {
        SearchResult::AtLeast(combined)
    } else {
        SearchResult::Exact(combined)
    })
}

fn index_intersects_dataset(index: &IndexMetadata, dataset: &Dataset) -> bool {
    index
        .fragment_bitmap
        .as_ref()
        .is_some_and(|index_bitmap| index_bitmap.intersection_len(&dataset.fragment_bitmap) > 0)
}

async fn load_named_scalar_segments(
    dataset: &Dataset,
    column: &str,
    index_name: &str,
) -> Result<Vec<IndexMetadata>> {
    let usable_indices = dataset
        .load_indices_by_name(index_name)
        .await?
        .into_iter()
        .filter(|index| index_intersects_dataset(index, dataset))
        .collect::<Vec<_>>();

    let mut index_type_url = None::<String>;
    for index in &usable_indices {
        let segment_type_url = match index.index_details.as_ref() {
            Some(index_details) => index_details.type_url.clone(),
            None => {
                // Legacy manifests may omit embedded details, so fetch only the missing ones.
                fetch_index_details(dataset, column, index)
                    .await?
                    .type_url
                    .clone()
            }
        };
        match &index_type_url {
            Some(expected) if expected != &segment_type_url => {
                return Err(Error::invalid_input(format!(
                    "Scalar index '{}' on column '{}' mixes incompatible segment types",
                    index_name, column
                )));
            }
            None => index_type_url = Some(segment_type_url),
            Some(_) => {}
        }
    }

    Ok(usable_indices)
}

fn union_fragment_bitmaps(indices: &[IndexMetadata], index_name: &str) -> Result<RoaringBitmap> {
    let mut combined = RoaringBitmap::new();
    for index in indices {
        let fragment_bitmap = index.fragment_bitmap.as_ref().ok_or_else(|| {
            Error::invalid_input(format!(
                "Scalar index '{}' segment {} is missing fragment coverage",
                index_name, index.uuid
            ))
        })?;
        combined |= fragment_bitmap.clone();
    }
    Ok(combined)
}

pub async fn scalar_index_fragment_bitmap(
    dataset: &Dataset,
    column: &str,
    index_name: &str,
) -> Result<Option<RoaringBitmap>> {
    let indices = load_named_scalar_segments(dataset, column, index_name).await?;
    match indices.len() {
        0 => Ok(None),
        1 => Ok(indices
            .into_iter()
            .next()
            .and_then(|index| index.fragment_bitmap)),
        _ => union_fragment_bitmaps(&indices, index_name).map(Some),
    }
}

pub async fn open_named_scalar_index(
    dataset: &Dataset,
    column: &str,
    index_name: &str,
    metrics: &dyn MetricsCollector,
) -> Result<Arc<dyn ScalarIndex>> {
    let indices = load_named_scalar_segments(dataset, column, index_name).await?;
    match indices.len() {
        0 => Err(Error::internal(format!(
            "Scanner created plan for index query on index {} for column {} but no usable index exists with that name",
            index_name, column
        ))),
        1 => {
            let uuid = indices[0].uuid.to_string();
            dataset.open_scalar_index(column, &uuid, metrics).await
        }
        _ => {
            let segments = try_join_all(indices.iter().map(|index| {
                let uuid = index.uuid.to_string();
                async move { dataset.open_scalar_index(column, &uuid, metrics).await }
            }))
            .await?;

            Ok(Arc::new(LogicalScalarIndex::try_new(
                index_name.to_string(),
                column.to_string(),
                segments,
            )?) as Arc<dyn ScalarIndex>)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::ops::Bound;

    use arrow::datatypes::Int32Type;
    use datafusion::scalar::ScalarValue;
    use lance_core::utils::address::RowAddress;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::array;
    use lance_index::IndexType;
    use lance_index::metrics::NoOpMetricsCollector;
    use lance_index::scalar::{BuiltinIndexType, SargableQuery, ScalarIndexParams};

    use crate::index::create::CreateIndexBuilder;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};

    use super::*;

    #[tokio::test]
    async fn test_open_named_scalar_index_uses_all_zonemap_segments() {
        let dataset = lance_datagen::gen_batch()
            .col("value", array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(4), FragmentRowCount::from(16))
            .await
            .unwrap();
        let mut dataset = dataset;
        let fragments = dataset.get_fragments();
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap);
        let mut segments = Vec::new();

        for fragment in &fragments {
            let segment =
                CreateIndexBuilder::new(&mut dataset, &["value"], IndexType::ZoneMap, &params)
                    .name("value_zonemap".to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap();
            segments.push(segment);
        }

        dataset
            .commit_existing_index_segments("value_zonemap", "value", segments)
            .await
            .unwrap();

        let committed = dataset.load_indices_by_name("value_zonemap").await.unwrap();
        assert_eq!(committed.len(), fragments.len());

        let logical =
            open_named_scalar_index(&dataset, "value", "value_zonemap", &NoOpMetricsCollector)
                .await
                .unwrap();
        assert_eq!(
            logical.calculate_included_frags().await.unwrap(),
            dataset.fragment_bitmap.as_ref().clone()
        );

        let combined_bitmap = scalar_index_fragment_bitmap(&dataset, "value", "value_zonemap")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(combined_bitmap, dataset.fragment_bitmap.as_ref().clone());
    }

    #[tokio::test]
    async fn test_open_named_scalar_index_uses_all_btree_segments() {
        let test_dir = TempStrDir::default();
        let dataset = lance_datagen::gen_batch()
            .col("value", array::step::<Int32Type>())
            .into_dataset(
                test_dir.as_str(),
                FragmentCount::from(4),
                FragmentRowCount::from(16),
            )
            .await
            .unwrap();
        let mut dataset = dataset;
        let fragments = dataset.get_fragments();
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        let mut segments = Vec::new();

        for fragment in &fragments {
            let segment =
                CreateIndexBuilder::new(&mut dataset, &["value"], IndexType::BTree, &params)
                    .name("value_btree".to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap();
            segments.push(segment);
        }

        dataset
            .commit_existing_index_segments("value_btree", "value", segments)
            .await
            .unwrap();

        let committed = dataset.load_indices_by_name("value_btree").await.unwrap();
        assert_eq!(committed.len(), fragments.len());

        let logical =
            open_named_scalar_index(&dataset, "value", "value_btree", &NoOpMetricsCollector)
                .await
                .unwrap();
        assert_eq!(logical.index_type(), IndexType::BTree);
        assert_eq!(
            logical.calculate_included_frags().await.unwrap(),
            dataset.fragment_bitmap.as_ref().clone()
        );

        let combined_bitmap = scalar_index_fragment_bitmap(&dataset, "value", "value_btree")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(combined_bitmap, dataset.fragment_bitmap.as_ref().clone());
    }

    #[tokio::test]
    async fn test_btree_segment_search_is_exact_across_fragments() {
        let test_dir = TempStrDir::default();
        let dataset = lance_datagen::gen_batch()
            .col("value", array::step::<Int32Type>())
            .into_dataset(
                test_dir.as_str(),
                FragmentCount::from(4),
                FragmentRowCount::from(16),
            )
            .await
            .unwrap();
        let mut dataset = dataset;
        let fragments = dataset.get_fragments();
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        let mut segments = Vec::new();

        for fragment in &fragments {
            segments.push(
                CreateIndexBuilder::new(&mut dataset, &["value"], IndexType::BTree, &params)
                    .name("value_btree_search".to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap(),
            );
        }

        dataset
            .commit_existing_index_segments("value_btree_search", "value", segments)
            .await
            .unwrap();

        let logical = open_named_scalar_index(
            &dataset,
            "value",
            "value_btree_search",
            &NoOpMetricsCollector,
        )
        .await
        .unwrap();

        let query = SargableQuery::Range(
            Bound::Included(ScalarValue::Int32(Some(20))),
            Bound::Included(ScalarValue::Int32(Some(43))),
        );
        let result = logical.search(&query, &NoOpMetricsCollector).await.unwrap();
        let row_addrs = match result {
            SearchResult::Exact(row_addrs) => row_addrs,
            other => panic!(
                "expected exact result from segmented btree, got {:?}",
                other
            ),
        };

        let searched_fragments = row_addrs
            .true_rows()
            .row_addrs()
            .unwrap()
            .map(|row_addr| RowAddress::from(u64::from(row_addr)).fragment_id())
            .collect::<Vec<_>>();
        assert_eq!(searched_fragments.len(), 24);
        assert_eq!(
            searched_fragments.into_iter().collect::<BTreeSet<_>>(),
            BTreeSet::from([1, 2])
        );
    }

    #[tokio::test]
    async fn test_zonemap_segment_search_keeps_fragment_ids() {
        let dataset = lance_datagen::gen_batch()
            .col("value", array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(4), FragmentRowCount::from(16))
            .await
            .unwrap();
        let mut dataset = dataset;
        let target_fragment = dataset.get_fragments()[2].id() as u32;
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap);

        let segment =
            CreateIndexBuilder::new(&mut dataset, &["value"], IndexType::ZoneMap, &params)
                .name("value_zonemap_single_fragment".to_string())
                .fragments(vec![target_fragment])
                .execute_uncommitted()
                .await
                .unwrap();

        dataset
            .commit_existing_index_segments("value_zonemap_single_fragment", "value", vec![segment])
            .await
            .unwrap();

        let logical = open_named_scalar_index(
            &dataset,
            "value",
            "value_zonemap_single_fragment",
            &NoOpMetricsCollector,
        )
        .await
        .unwrap();

        assert_eq!(
            logical
                .calculate_included_frags()
                .await
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![target_fragment]
        );

        let query = SargableQuery::Range(
            Bound::Included(ScalarValue::Int32(Some(0))),
            Bound::Included(ScalarValue::Int32(Some(10_000))),
        );
        let result = logical.search(&query, &NoOpMetricsCollector).await.unwrap();
        let searched_fragments = result
            .row_addrs()
            .true_rows()
            .row_addrs()
            .unwrap()
            .map(|row_addr| RowAddress::from(u64::from(row_addr)).fragment_id())
            .collect::<Vec<_>>();
        assert!(!searched_fragments.is_empty());
        assert!(
            searched_fragments
                .iter()
                .all(|fragment_id| *fragment_id == target_fragment)
        );
    }

    #[tokio::test]
    async fn test_commit_existing_zonemap_segments_replaces_overlapping_segments() {
        let dataset = lance_datagen::gen_batch()
            .col("value", array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(16))
            .await
            .unwrap();
        let mut dataset = dataset;
        let fragments = dataset.get_fragments();
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap);

        let mut first_segments = Vec::new();
        for fragment in &fragments {
            first_segments.push(
                CreateIndexBuilder::new(&mut dataset, &["value"], IndexType::ZoneMap, &params)
                    .name("value_zonemap_replace".to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap(),
            );
        }

        dataset
            .commit_existing_index_segments("value_zonemap_replace", "value", first_segments)
            .await
            .unwrap();

        let mut replacement_segments = Vec::new();
        for fragment in &fragments {
            replacement_segments.push(
                CreateIndexBuilder::new(&mut dataset, &["value"], IndexType::ZoneMap, &params)
                    .name("value_zonemap_replace".to_string())
                    .replace(true)
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap(),
            );
        }
        let replacement_uuids = replacement_segments
            .iter()
            .map(|segment| segment.uuid)
            .collect::<Vec<_>>();

        dataset
            .commit_existing_index_segments("value_zonemap_replace", "value", replacement_segments)
            .await
            .unwrap();

        let committed = dataset
            .load_indices_by_name("value_zonemap_replace")
            .await
            .unwrap();
        assert_eq!(committed.len(), fragments.len());
        assert_eq!(
            committed
                .iter()
                .map(|segment| segment.uuid)
                .collect::<Vec<_>>(),
            replacement_uuids
        );
        assert_eq!(
            scalar_index_fragment_bitmap(&dataset, "value", "value_zonemap_replace")
                .await
                .unwrap()
                .unwrap(),
            dataset.fragment_bitmap.as_ref().clone()
        );
    }
}
