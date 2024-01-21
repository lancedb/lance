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

use std::sync::Arc;

use lance_core::{Error, Result};
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::IndexType;
use lance_table::format::Index as IndexMetadata;
use roaring::RoaringBitmap;
use snafu::{location, Location};
use uuid::Uuid;

use super::vector::ivf::optimize_vector_indices;
use super::DatasetIndexInternalExt;
use crate::dataset::scanner::ColumnOrdering;
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
    dataset: Arc<Dataset>,
    old_indices: &[&'a IndexMetadata],
    options: &OptimizeOptions,
) -> Result<Option<(Uuid, Vec<&'a IndexMetadata>, RoaringBitmap)>> {
    if old_indices.is_empty() {
        return Err(Error::Index {
            message: "Append index: no prevoius index found".to_string(),
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

    let mut indices = Vec::with_capacity(old_indices.len());
    for idx in old_indices {
        let index = dataset
            .open_generic_index(&column.name, &idx.uuid.to_string())
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
    let unindexed = dataset.unindexed_fragments(&old_indices[0].name).await?;

    let mut frag_bitmap = RoaringBitmap::new();
    old_indices.iter().for_each(|idx| {
        frag_bitmap.extend(idx.fragment_bitmap.as_ref().unwrap().iter());
    });
    unindexed.iter().for_each(|frag| {
        frag_bitmap.push(frag.id as u32);
    });

    let (new_uuid, indices_merged) = match indices[0].index_type() {
        IndexType::Scalar => {
            let index = dataset
                .open_scalar_index(&column.name, &old_indices[0].uuid.to_string())
                .await?;

            let mut scanner = dataset.scan();
            scanner
                .with_fragments(unindexed)
                .with_row_id()
                .order_by(Some(vec![ColumnOrdering::asc_nulls_first(
                    column.name.clone(),
                )]))?
                .project(&[&column.name])?;
            let new_data_stream = scanner.try_into_stream().await?;

            let new_uuid = Uuid::new_v4();

            let index_dir = dataset.indices_dir().child(new_uuid.to_string());
            let new_store = LanceIndexStore::new((*dataset.object_store).clone(), index_dir);

            index.update(new_data_stream.into(), &new_store).await?;

            Ok((new_uuid, 1))
        }
        IndexType::Vector => {
            let new_data_stream = if unindexed.is_empty() {
                None
            } else {
                let mut scanner = dataset.scan();
                scanner
                    .with_fragments(unindexed)
                    .with_row_id()
                    .project(&[&column.name])?;
                Some(scanner.try_into_stream().await?)
            };

            optimize_vector_indices(
                &dataset.object_store,
                &dataset.indices_dir(),
                dataset.version().version,
                new_data_stream,
                &column.name,
                &indices,
                options,
            )
            .await
        }
    }?;

    Ok(Some((
        new_uuid,
        old_indices[old_indices.len() - indices_merged..].to_vec(),
        frag_bitmap,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::cast::AsArray;
    use arrow_array::types::UInt32Type;
    use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchIterator, UInt32Array};
    use arrow_schema::{DataType, Field, Schema};
    use futures::{stream, StreamExt, TryStreamExt};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::optimize::IndexDeltaOption;
    use lance_index::{
        vector::{ivf::IvfBuildParams, pq::PQBuildParams},
        DatasetIndexExt, IndexType,
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
                index_delta_option: IndexDeltaOption::NewDelta, // Just create index for delta
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
