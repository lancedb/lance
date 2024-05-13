// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use futures::future::try_join_all;
use lance_core::{Error, Result};
use lance_index::optimize::{IndexHandling, NewDataHandling, OptimizeOptions};
use lance_index::{Index, IndexType};
use lance_table::format::{Fragment, Index as IndexMetadata};
use roaring::RoaringBitmap;
use snafu::{location, Location};
use std::borrow::Cow;
use std::sync::Arc;
use uuid::Uuid;

use super::scalar::optimize_scalar_index;
use super::vector::ivf::optimize_vector_indices;
use super::{unindexed_fragments, DatasetIndexInternalExt};
use crate::dataset::Dataset;

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
    dataset: &Dataset,
    fragments: Option<&[Fragment]>,
    old_indices: &[&'a IndexMetadata],
    options: &OptimizeOptions,
) -> Result<Option<(Uuid, Vec<&'a IndexMetadata>, RoaringBitmap)>> {
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

    let merge_indices_meta = indices_to_merge(old_indices, &options.index_handling);

    let merge_indices = open_indices(dataset, &column.name, merge_indices_meta.as_ref()).await?;

    // We need one index to get the parameters.
    let reference_index = if let Some(idx) = merge_indices.first() {
        idx.clone()
    } else {
        let meta = old_indices.first().ok_or(Error::Index {
            message: "Append index: no previous index found".to_string(),
            location: location!(),
        })?;
        dataset
            .open_generic_index(&column.name, &meta.uuid.to_string())
            .await?
    };

    let frags_to_index = fragments_to_index(
        fragments.unwrap_or_else(|| dataset.fragments().as_slice()),
        merge_indices_meta.as_ref(),
        &options.new_data_handling,
    )?;

    if frags_to_index.is_empty()
        && (options.index_handling == IndexHandling::NewDelta || merge_indices_meta.len() == 1)
    {
        // No new data to index or no indices to merge.
        return Ok(None);
    }

    let mut frag_bitmap = RoaringBitmap::new();
    merge_indices_meta.iter().for_each(|idx| {
        frag_bitmap.extend(idx.fragment_bitmap.as_ref().unwrap().iter());
    });
    frags_to_index.iter().for_each(|frag| {
        frag_bitmap.insert(frag.id as u32);
    });

    let new_uuid = match reference_index.index_type() {
        IndexType::Scalar => {
            optimize_scalar_index(dataset, column, merge_indices_meta.as_ref(), frags_to_index)
                .await
        }
        IndexType::Vector => {
            let new_data_stream = if frags_to_index.is_empty() {
                None
            } else {
                let mut scanner = dataset.scan();
                scanner
                    .with_fragments(frags_to_index)
                    .with_row_id()
                    .project(&[&column.name])?;
                Some(scanner.try_into_stream().await?)
            };

            optimize_vector_indices(
                dataset,
                new_data_stream,
                &column.name,
                reference_index.as_ref(),
                &merge_indices,
            )
            .await
        }
    }?;

    Ok(Some((new_uuid, merge_indices_meta.to_vec(), frag_bitmap)))
}

fn indices_to_merge<'a, 'b>(
    old_indices: &'b [&'a IndexMetadata],
    index_handling: &IndexHandling,
) -> Cow<'b, [&'a IndexMetadata]> {
    match index_handling {
        IndexHandling::NewDelta => Cow::Borrowed(&[]),
        IndexHandling::MergeLatestN(n) => {
            let start_pos = if *n > old_indices.len() {
                0
            } else {
                old_indices.len() - n
            };
            Cow::Borrowed(&old_indices[start_pos..])
        }
        IndexHandling::MergeAll => Cow::Borrowed(old_indices),
        IndexHandling::MergeIndices(target_uuids) => Cow::Owned(
            old_indices
                .iter()
                .filter(|&idx| target_uuids.contains(&idx.uuid))
                .cloned()
                .collect::<Vec<_>>(),
        ),
    }
}

async fn open_indices(
    dataset: &Dataset,
    col_name: &str,
    indices: &[&IndexMetadata],
) -> Result<Vec<Arc<dyn Index>>> {
    let opening = indices.iter().map(|idx| async move {
        let uuid_str = idx.uuid.to_string();
        dataset.open_generic_index(col_name, &uuid_str).await
    });
    let opened = try_join_all(opening).await?;

    if opened
        .windows(2)
        .any(|w| w[0].index_type() != w[1].index_type())
    {
        return Err(Error::Index {
            message: format!("Append index: invalid index deltas: {:?}", indices),
            location: location!(),
        });
    }

    Ok(opened)
}

fn fragments_to_index(
    fragments: &[Fragment],
    existing_indices: &[&IndexMetadata],
    new_data_handling: &NewDataHandling,
) -> Result<Vec<Fragment>> {
    match new_data_handling {
        NewDataHandling::IndexAll => unindexed_fragments(existing_indices, fragments),
        NewDataHandling::Fragments(ref target_frags) => {
            let mut fragments = unindexed_fragments(existing_indices, fragments)?;
            fragments.retain(|frag| target_frags.contains(&(frag.id as u32)));
            Ok(fragments)
        }
        NewDataHandling::Ignore => Ok(vec![]),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    use arrow_array::cast::AsArray;
    use arrow_array::types::UInt32Type;
    use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchIterator, UInt32Array};
    use arrow_schema::{DataType, Field, Schema};
    use futures::{stream, StreamExt, TryStreamExt};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::optimize::IndexHandling;
    use lance_index::{
        vector::{ivf::IvfBuildParams, pq::PQBuildParams},
        DatasetIndexExt,
    };
    use lance_linalg::distance::MetricType;
    use lance_testing::datagen::generate_random_array;
    use tempfile::tempdir;

    use crate::dataset::builder::DatasetBuilder;
    use crate::index::vector::ivf::IVFIndex;
    use crate::index::vector::{pq::PQIndex, VectorIndexParams};

    #[tokio::test]
    async fn test_append_index() {
        const DIM: usize = 64;
        const IVF_PARTITIONS: usize = 2;

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

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
        scanner.nearest("vector", q.as_primitive(), 10).unwrap();
        let results = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(results[0].num_rows(), 10); // Flat search.

        dataset.optimize_indices(&Default::default()).await.unwrap();
        let dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();
        let index = &dataset.load_indices().await.unwrap()[0];

        assert!(dataset
            .unindexed_fragments(&index.name)
            .await
            .unwrap()
            .is_empty());

        // There should be two indices directories existed.
        let object_store = dataset.object_store();
        let index_dirs = object_store
            .read_dir_all(&dataset.indices_dir(), None)
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(index_dirs.len(), 2);

        let mut scanner = dataset.scan();
        scanner.nearest("vector", q.as_primitive(), 10).unwrap();
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
        let binding = dataset
            .open_vector_index("vector", index.uuid.to_string().as_str())
            .await
            .unwrap();
        let ivf_index = binding.as_any().downcast_ref::<IVFIndex>().unwrap();
        let row_in_index = stream::iter(0..IVF_PARTITIONS)
            .map(|part_id| async move {
                let part = ivf_index.load_partition(part_id, true).await.unwrap();
                let pq_idx = part.as_any().downcast_ref::<PQIndex>().unwrap();
                pq_idx.row_ids.as_ref().unwrap().len()
            })
            .buffered(2)
            .collect::<Vec<usize>>()
            .await
            .iter()
            .sum::<usize>();
        assert_eq!(row_in_index, 2000);
    }

    #[tokio::test]
    async fn test_query_delta_indices() {
        const DIM: usize = 64;
        const IVF_PARTITIONS: usize = 2;
        const TOTAL: usize = 1000;

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

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
            .create_index(
                &["vector"],
                IndexType::Vector,
                None,
                &VectorIndexParams::with_ivf_pq_params(
                    MetricType::L2,
                    IvfBuildParams::new(IVF_PARTITIONS),
                    PQBuildParams {
                        num_sub_vectors: 2,
                        ..Default::default()
                    },
                ),
                true,
            )
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
            .optimize_indices(&OptimizeOptions {
                index_handling: IndexHandling::NewDelta,
                ..Default::default()
            })
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
            .nearest("vector", array.value(0).as_primitive(), 2)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 2);
        let mut id_arr = results["id"].as_primitive::<UInt32Type>().values().to_vec();
        id_arr.sort();
        assert_eq!(id_arr, vec![0, 1000]);
    }
}
