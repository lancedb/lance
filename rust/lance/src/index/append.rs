// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use futures::FutureExt;
use lance_core::{Error, Result};
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::CreatedIndex;
use lance_index::VECTOR_INDEX_VERSION;
use lance_table::format::{Fragment, IndexMetadata};
use roaring::RoaringBitmap;
use snafu::location;
use uuid::Uuid;

use super::vector::ivf::optimize_vector_indices;
use super::DatasetIndexInternalExt;
use crate::dataset::index::LanceIndexStoreExt;
use crate::dataset::Dataset;
use crate::index::scalar::load_training_data;
use crate::index::vector_index_details;

#[derive(Debug, Clone)]
pub struct IndexMergeResults<'a> {
    pub new_uuid: Uuid,
    pub removed_indices: Vec<&'a IndexMetadata>,
    pub new_fragment_bitmap: RoaringBitmap,
    pub new_index_version: i32,
    pub new_index_details: prost_types::Any,
}

/// Merge in-inflight unindexed data, with a specific number of previous indices
/// into a new index, to improve the query performance.
///
/// The merge behavior is controlled by [`OptimizeOptions::num_indices_to_merge].
///
/// Returns
/// -------
/// - the UUID of the new index
/// - merged indices,
/// - Bitmap of the fragments that covered in the newly created index.
pub async fn merge_indices<'a>(
    dataset: Arc<Dataset>,
    old_indices: &[&'a IndexMetadata],
    options: &OptimizeOptions,
) -> Result<Option<IndexMergeResults<'a>>> {
    if old_indices.is_empty() {
        return Err(Error::Index {
            message: "Append index: no previous index found".to_string(),
            location: location!(),
        });
    };

    let unindexed = dataset.unindexed_fragments(&old_indices[0].name).await?;
    merge_indices_with_unindexed_frags(dataset, old_indices, &unindexed, options).await
}

/// Merge a list of provided unindexed data, with a specific number of previous indices
/// into a new index, to improve the query performance.
pub async fn merge_indices_with_unindexed_frags<'a>(
    dataset: Arc<Dataset>,
    old_indices: &[&'a IndexMetadata],
    unindexed: &[Fragment],
    options: &OptimizeOptions,
) -> Result<Option<IndexMergeResults<'a>>> {
    if old_indices.is_empty() {
        return Err(Error::Index {
            message: "Append index: no previous index found".to_string(),
            location: location!(),
        });
    };

    let column = dataset
        .schema()
        .field_by_id(old_indices[0].fields[0])
        .ok_or(Error::Index {
            message: format!(
                "Append index: column {} does not exist",
                old_indices[0].fields[0]
            ),
            location: location!(),
        })?;

    let field_path = dataset.schema().field_path(old_indices[0].fields[0])?;
    let mut indices = Vec::with_capacity(old_indices.len());
    for idx in old_indices {
        let index = dataset
            .open_generic_index(&field_path, &idx.uuid.to_string(), &NoOpMetricsCollector)
            .await?;
        indices.push(index);
    }

    if indices
        .windows(2)
        .any(|w| w[0].index_type() != w[1].index_type())
    {
        return Err(Error::Index {
            message: format!("Append index: invalid index deltas: {:?}", old_indices),
            location: location!(),
        });
    }

    let mut frag_bitmap = RoaringBitmap::new();
    unindexed.iter().for_each(|frag| {
        frag_bitmap.insert(frag.id as u32);
    });

    let index_type = indices[0].index_type();
    let (new_uuid, indices_merged, created_index) = match index_type {
        it if it.is_scalar() => {
            // There are no delta indices for scalar, so adding all indexed
            // fragments to the new index.
            old_indices.iter().for_each(|idx| {
                frag_bitmap.extend(idx.fragment_bitmap.as_ref().unwrap().iter());
            });

            let index = dataset
                .open_scalar_index(
                    &field_path,
                    &old_indices[0].uuid.to_string(),
                    &NoOpMetricsCollector,
                )
                .await?;

            let update_criteria = index.update_criteria();

            let fragments = if update_criteria.requires_old_data {
                None
            } else {
                Some(unindexed.to_vec())
            };
            let new_data_stream = load_training_data(
                dataset.as_ref(),
                &field_path,
                &update_criteria.data_criteria,
                fragments,
                true,
                None,
            )
            .await?;

            let new_uuid = Uuid::new_v4();

            let new_store = LanceIndexStore::from_dataset_for_new(&dataset, &new_uuid.to_string())?;
            let created_index = index.update(new_data_stream, &new_store).await?;

            // TODO: don't hard-code index version
            Ok((new_uuid, 1, created_index))
        }
        it if it.is_vector() => {
            let new_data_stream = if unindexed.is_empty() {
                None
            } else {
                let mut scanner = dataset.scan();
                scanner
                    .with_fragments(unindexed.to_vec())
                    .with_row_id()
                    .project(&[&field_path])?;
                if column.nullable {
                    let column_expr =
                        lance_datafusion::logical_expr::field_path_to_expr(&field_path)?;
                    scanner.filter_expr(column_expr.is_not_null());
                }
                Some(scanner.try_into_stream().await?)
            };

            let (new_uuid, indices_merged) = optimize_vector_indices(
                dataset.as_ref().clone(),
                new_data_stream,
                &field_path,
                &indices,
                options,
            )
            .boxed()
            .await?;

            old_indices[old_indices.len() - indices_merged..]
                .iter()
                .for_each(|idx| {
                    frag_bitmap.extend(idx.fragment_bitmap.as_ref().unwrap().iter());
                });

            Ok((
                new_uuid,
                indices_merged,
                CreatedIndex {
                    index_details: vector_index_details(),
                    index_version: VECTOR_INDEX_VERSION,
                },
            ))
        }
        _ => Err(Error::Index {
            message: format!(
                "Append index: invalid index type: {:?}",
                indices[0].index_type()
            ),
            location: location!(),
        }),
    }?;

    let removed_indices = old_indices[old_indices.len() - indices_merged..].to_vec();
    for removed in removed_indices.iter() {
        frag_bitmap |= removed.fragment_bitmap.as_ref().unwrap();
    }

    Ok(Some(IndexMergeResults {
        new_uuid,
        removed_indices,
        new_fragment_bitmap: frag_bitmap,
        new_index_version: created_index.index_version as i32,
        new_index_details: created_index.index_details,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::datatypes::{Float32Type, UInt32Type};
    use arrow_array::cast::AsArray;
    use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchIterator, UInt32Array};
    use arrow_schema::{DataType, Field, Schema};
    use futures::TryStreamExt;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datafusion::utils::reader_to_stream;
    use lance_datagen::{array, Dimension, RowCount};
    use lance_index::vector::hnsw::builder::HnswBuildParams;
    use lance_index::vector::sq::builder::SQBuildParams;
    use lance_index::{
        vector::{ivf::IvfBuildParams, pq::PQBuildParams},
        DatasetIndexExt, IndexType,
    };
    use lance_linalg::distance::MetricType;
    use lance_testing::datagen::generate_random_array;
    use rstest::rstest;

    use crate::dataset::builder::DatasetBuilder;
    use crate::dataset::{MergeInsertBuilder, WhenMatched, WhenNotMatched, WriteParams};
    use crate::index::vector::VectorIndexParams;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};

    #[tokio::test]
    async fn test_append_index() {
        const DIM: usize = 64;
        const IVF_PARTITIONS: usize = 2;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let vectors = generate_random_array(1000 * DIM);

        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            true,
        )]));
        let array = Arc::new(FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap());
        let batch = RecordBatch::try_new(schema.clone(), vec![array.clone()]).unwrap();

        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

        let ivf_params = IvfBuildParams::new(IVF_PARTITIONS);
        let pq_params = PQBuildParams {
            num_sub_vectors: 2,
            ..Default::default()
        };
        let params = VectorIndexParams::with_ivf_pq_params(MetricType::L2, ivf_params, pq_params);

        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        let vectors = generate_random_array(1000 * DIM);
        let array = Arc::new(FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap());
        let batch = RecordBatch::try_new(schema.clone(), vec![array.clone()]).unwrap();

        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        dataset.append(batches, None).await.unwrap();

        let index = &dataset.load_indices().await.unwrap()[0];
        assert!(!dataset
            .unindexed_fragments(&index.name)
            .await
            .unwrap()
            .is_empty());

        let q = array.value(5);
        let mut scanner = dataset.scan();
        scanner
            .nearest("vector", q.as_primitive::<Float32Type>(), 10)
            .unwrap();
        let results = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(results[0].num_rows(), 10); // Flat search.

        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        let dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();
        let indices = dataset.load_indices().await.unwrap();

        assert!(dataset
            .unindexed_fragments(&index.name)
            .await
            .unwrap()
            .is_empty());

        // There should be two indices directories existed.
        let object_store = dataset.object_store();
        let index_dirs = object_store.read_dir(dataset.indices_dir()).await.unwrap();
        assert_eq!(index_dirs.len(), 2);

        let mut scanner = dataset.scan();
        scanner
            .nearest("vector", q.as_primitive::<Float32Type>(), 10)
            .unwrap();
        let results = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let vectors = &results[0]["vector"];
        // Second batch of vectors should be in the index.
        let contained = vectors.as_fixed_size_list().iter().any(|v| {
            let vec = v.as_ref().unwrap();
            array.iter().any(|a| a.as_ref().unwrap() == vec)
        });
        assert!(contained);

        // Check that the index has all 2000 rows.
        let mut num_rows = 0;
        for index in indices.iter() {
            let index = dataset
                .open_vector_index(
                    "vector",
                    index.uuid.to_string().as_str(),
                    &NoOpMetricsCollector,
                )
                .await
                .unwrap();
            num_rows += index.num_rows();
        }
        assert_eq!(num_rows, 2000);
    }

    #[rstest]
    #[tokio::test]
    async fn test_query_delta_indices(
        #[values(
            VectorIndexParams::ivf_pq(2, 8, 4, MetricType::L2, 2),
            VectorIndexParams::with_ivf_hnsw_sq_params(
                MetricType::L2,
                IvfBuildParams::new(2),
                HnswBuildParams::default(),
                SQBuildParams::default()
            )
        )]
        index_params: VectorIndexParams,
    ) {
        const DIM: usize = 64;
        const TOTAL: usize = 1000;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let vectors = generate_random_array(TOTAL * DIM);

        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    DIM as i32,
                ),
                true,
            ),
            Field::new("id", DataType::UInt32, false),
        ]));
        let array = Arc::new(FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                array.clone(),
                Arc::new(UInt32Array::from_iter_values(0..TOTAL as u32)),
            ],
        )
        .unwrap();

        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &index_params, true)
            .await
            .unwrap();
        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vector_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_indices"], 1);
        assert_eq!(stats["num_indexed_fragments"], 1);
        assert_eq!(stats["num_unindexed_fragments"], 0);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                array.clone(),
                Arc::new(UInt32Array::from_iter_values(
                    TOTAL as u32..(TOTAL * 2) as u32,
                )),
            ],
        )
        .unwrap();

        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        dataset.append(batches, None).await.unwrap();
        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vector_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_indices"], 1);
        assert_eq!(stats["num_indexed_fragments"], 1);
        assert_eq!(stats["num_unindexed_fragments"], 1);

        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        let dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();
        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vector_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_indices"], 2);
        assert_eq!(stats["num_indexed_fragments"], 2);
        assert_eq!(stats["num_unindexed_fragments"], 0);

        let results = dataset
            .scan()
            .project(&["id"])
            .unwrap()
            .nearest("vector", array.value(0).as_primitive::<Float32Type>(), 2)
            .unwrap()
            .refine(1)
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 2);
        let mut id_arr = results["id"].as_primitive::<UInt32Type>().values().to_vec();
        id_arr.sort();
        assert_eq!(id_arr, vec![0, 1000]);
    }

    #[tokio::test]
    async fn test_merge_indices_after_merge_insert() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        // Create initial dataset using lance_datagen
        let mut dataset = lance_datagen::gen_batch()
            .col("id", array::step::<UInt32Type>())
            .col("value", array::cycle_utf8_literals(&["a", "b", "c"]))
            .col(
                "vector",
                array::rand_vec::<Float32Type>(Dimension::from(64)),
            )
            .into_dataset_with_params(
                test_uri,
                FragmentCount(1),
                FragmentRowCount(1000),
                Some(WriteParams {
                    max_rows_per_file: 1000,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

        // Create initial index
        let ivf_params = IvfBuildParams::new(2);
        let pq_params = PQBuildParams {
            num_sub_vectors: 2,
            ..Default::default()
        };
        let params = VectorIndexParams::with_ivf_pq_params(MetricType::L2, ivf_params, pq_params);

        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        // Load initial index metadata
        let initial_indices = dataset.load_indices().await.unwrap();
        assert_eq!(initial_indices.len(), 1);
        let index_name = initial_indices[0].name.clone();

        // Prepare new data for merge insert (updates to existing rows)
        let new_batch = lance_datagen::gen_batch()
            .col("id", array::step_custom::<UInt32Type>(500, 1)) // IDs 500-999
            .col("value", array::cycle_utf8_literals(&["d", "e", "f"])) // Different values
            .col(
                "vector",
                array::rand_vec::<Float32Type>(Dimension::from(64)),
            )
            .into_batch_rows(RowCount::from(500))
            .unwrap();

        // Record the maximum fragment ID before merge insert
        let max_fragment_id_before = dataset.manifest.max_fragment_id().unwrap_or(0);

        // Execute merge insert operation
        let merge_job =
            MergeInsertBuilder::try_new(Arc::new(dataset.clone()), vec!["id".to_string()])
                .unwrap()
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::InsertAll)
                .try_build()
                .unwrap();

        let schema = new_batch.schema();
        let new_reader = Box::new(RecordBatchIterator::new([Ok(new_batch)], schema.clone()));
        let new_stream = reader_to_stream(new_reader);
        let (updated_dataset, merge_stats) = merge_job.execute(new_stream).await.unwrap();

        // Check merge stats
        assert_eq!(merge_stats.num_updated_rows, 500); // Updates for rows 500-999
        assert_eq!(merge_stats.num_inserted_rows, 0); // No new inserts in this case

        // Get the newly added fragments by comparing fragment IDs
        let unindexed_fragments: Vec<Fragment> = updated_dataset
            .get_fragments()
            .into_iter()
            .filter(|f| f.id() as u64 > max_fragment_id_before)
            .map(|f| f.metadata().clone())
            .collect();

        // Now run merge with known unindexed fragments
        let old_indices = updated_dataset
            .load_indices_by_name(&index_name)
            .await
            .unwrap();
        let old_indices_refs: Vec<&IndexMetadata> = old_indices.iter().collect();

        let merge_result = merge_indices_with_unindexed_frags(
            updated_dataset.clone(),
            &old_indices_refs,
            &unindexed_fragments,
            &OptimizeOptions::merge(old_indices.len()),
        )
        .await
        .unwrap();

        assert!(merge_result.is_some());
        let merge_result = merge_result.unwrap();

        // Verify that the new index covers all fragments
        let new_fragment_bitmap = &merge_result.new_fragment_bitmap;

        // Check that unindexed fragments are now included
        for fragment in &unindexed_fragments {
            assert!(new_fragment_bitmap.contains(fragment.id as u32));
        }

        // Check that old indexed fragments are still included
        // All fragments with ID <= max_fragment_id_before should be included
        for frag_id in 0..=max_fragment_id_before as u32 {
            assert!(new_fragment_bitmap.contains(frag_id));
        }

        // Verify the index can be used for search
        let dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();
        let indices = dataset.load_indices().await.unwrap();

        // There should still be indices (old one might be kept plus new one)
        assert!(!indices.is_empty());

        // Test that search works by querying for nearest neighbors
        let query_batch = lance_datagen::gen_batch()
            .col("query", array::rand_vec::<Float32Type>(Dimension::from(64)))
            .into_batch_rows(RowCount::from(1))
            .unwrap();

        let q = query_batch.column(0).as_fixed_size_list();
        let mut scanner = dataset.scan();
        scanner
            .nearest("vector", q.value(0).as_primitive::<Float32Type>(), 10)
            .unwrap();
        let results = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(results[0].num_rows(), 10);
    }
}
