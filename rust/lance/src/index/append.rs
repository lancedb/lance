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

use lance_core::{format::Index as IndexMetadata, Error, Result};
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::IndexType;
use log::info;
use roaring::RoaringBitmap;
use snafu::{location, Location};
use uuid::Uuid;

use crate::dataset::index::unindexed_fragments;
use crate::dataset::scanner::ColumnOrdering;
use crate::dataset::Dataset;
use crate::index::vector::ivf::IVFIndex;

use super::DatasetIndexInternalExt;

/// Append new data to the index, without re-train.
///
/// Returns the UUID of the new index along with a vector of newly indexed fragment ids
pub async fn append_index(
    dataset: Arc<Dataset>,
    old_index: &IndexMetadata,
) -> Result<Option<(Uuid, Option<RoaringBitmap>)>> {
    let unindexed = unindexed_fragments(old_index, dataset.as_ref()).await?;
    if unindexed.is_empty() {
        return Ok(None);
    };

    let frag_bitmap = old_index.fragment_bitmap.as_ref().map(|bitmap| {
        let mut bitmap = bitmap.clone();
        bitmap.extend(unindexed.iter().map(|frag| frag.id as u32));
        bitmap
    });

    let column = dataset
        .schema()
        .field_by_id(old_index.fields[0])
        .ok_or(Error::Index {
            message: format!(
                "Append index: column {} does not exist",
                old_index.fields[0]
            ),
            location: location!(),
        })?;

    let index = dataset
        .open_generic_index(&column.name, &old_index.uuid.to_string())
        .await?;

    match index.index_type() {
        IndexType::Scalar => {
            let index = dataset
                .open_scalar_index(&column.name, &old_index.uuid.to_string())
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

            Ok(Some((new_uuid, frag_bitmap)))
        }
        IndexType::Vector => {
            let mut scanner = dataset.scan();
            scanner.with_fragments(unindexed);
            scanner.with_row_id();
            scanner.project(&[&column.name])?;
            let new_data_stream = scanner.try_into_stream().await?;

            let index = dataset
                .open_vector_index(&column.name, old_index.uuid.to_string().as_str())
                .await?;

            let Some(ivf_idx) = index.as_any().downcast_ref::<IVFIndex>() else {
                info!("Index type: {:?} does not support append", index);
                return Ok(None);
            };

            let new_index = ivf_idx
                .append(dataset.as_ref(), new_data_stream, old_index, &column.name)
                .await?;

            Ok(Some((new_index, frag_bitmap)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::cast::AsArray;
    use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use futures::{stream, StreamExt, TryStreamExt};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::{
        vector::{ivf::IvfBuildParams, pq::PQBuildParams},
        IndexType,
    };
    use lance_linalg::distance::MetricType;
    use lance_testing::datagen::generate_random_array;
    use tempfile::tempdir;

    use crate::index::vector::{pq::PQIndex, VectorIndexParams};
    use crate::index::DatasetIndexExt;

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
        assert!(!unindexed_fragments(index, &dataset)
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

        dataset.optimize_indices(None).await.unwrap();
        let index = &dataset.load_indices().await.unwrap()[0];
        assert!(unindexed_fragments(index, &dataset)
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
        assert_eq!(dataset.index_cache_entry_count(), 6)
    }
}
