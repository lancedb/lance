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
use log::info;
use uuid::Uuid;

use crate::dataset::Dataset;
use crate::format::Index as IndexMetadata;
use crate::index::vector::{ivf::IVFIndex, open_index};

/// Append new data to the index, without re-train.
pub async fn append_index(
    dataset: Arc<Dataset>,
    old_index: &IndexMetadata,
) -> Result<Option<Uuid>> {
    let unindexed = old_index.unindexed_fragments(dataset.as_ref()).await?;
    if unindexed.is_empty() {
        return Ok(None);
    };

    let column = dataset
        .schema()
        .field_by_id(old_index.fields[0])
        .ok_or(Error::Index {
            message: format!(
                "Append index: column {} does not exist",
                old_index.fields[0]
            ),
        })?;

    let index = open_index(
        dataset.clone(),
        &column.name,
        old_index.uuid.to_string().as_str(),
    )
    .await?;

    let Some(ivf_idx) = index.as_any().downcast_ref::<IVFIndex>() else {
        info!("Index type: {:?} does not support append", index);
        return Ok(None);
    };

    let mut scanner = dataset.scan();
    scanner.with_fragments(unindexed);
    scanner.with_row_id();
    let stream = scanner.try_into_stream().await?;

    let new_index = ivf_idx
        .append(dataset.as_ref(), stream, old_index, &column.name)
        .await?;
    Ok(Some(new_index))
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::cast::AsArray;
    use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use futures::{stream, StreamExt, TryStreamExt};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::vector::pq::PQBuildParams;
    use lance_linalg::distance::MetricType;
    use lance_testing::datagen::generate_random_array;
    use tempfile::tempdir;

    use crate::index::vector::{ivf::IvfBuildParams, pq::PQIndex, VectorIndexParams};
    use crate::index::{DatasetIndexExt, IndexType};

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
        let mut pq_params = PQBuildParams::default();
        pq_params.num_sub_vectors = 2;
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
        assert!(index.unindexed_fragments(&dataset).await.unwrap().len() > 0);

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

        dataset.optimize_indices().await.unwrap();
        let index = &dataset.load_indices().await.unwrap()[0];
        assert!(index
            .unindexed_fragments(&dataset)
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
        let binding = open_index(Arc::new(dataset), "vector", index.uuid.to_string().as_str())
            .await
            .unwrap();
        let ivf_index = binding.as_any().downcast_ref::<IVFIndex>().unwrap();
        let row_in_index = stream::iter(0..IVF_PARTITIONS)
            .map(|part_id| async move {
                let part = ivf_index.load_partition(part_id).await.unwrap();
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
}
