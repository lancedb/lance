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

use futures::{stream, StreamExt, TryStreamExt};
use lance_core::{
    format::Index as IndexMetadata,
    io::{object_store::ObjectStore, RecordBatchStream},
    Error, Result,
};
use lance_index::{optimize::OptimizeOptions, IndexType, INDEX_FILE_NAME};
use lance_index::{scalar::lance_format::LanceIndexStore, Index};
use log::info;
use object_store::path::Path;
use roaring::RoaringBitmap;
use snafu::{location, Location};
use uuid::Uuid;

use super::DatasetIndexInternalExt;
use crate::dataset::scanner::ColumnOrdering;
use crate::dataset::Dataset;
use crate::index::vector::ivf::IVFIndex;

// TODO: move to `lance-index` crate.
async fn optimize_vector_indices(
    object_store: &ObjectStore,
    index_dir: &Path,
    unindexed: impl RecordBatchStream + Unpin + 'static,
    existing_indices: &[Arc<dyn Index>],
    options: OptimizeOptions,
) -> Result<Uuid> {
    // Senity check the indices
    if existing_indices.is_empty() {
        return Err(Error::Index {
            message: "optimizing vector index: no existing index found".to_string(),
            location: location!(),
        });
    }

    let new_uuid = Uuid::new_v4();
    let index_file = index_dir.child(new_uuid.to_string()).child(INDEX_FILE_NAME);
    let mut writer = object_store.create(&index_file).await?;

    let first_idx = existing_indices[0]
        .as_any()
        .downcast_ref::<IVFIndex>()
        .ok_or(Error::Index {
            message: "optimizing vector index: first index is not IVF".to_string(),
            location: location!(),
        })?;

    // TODO: merge two IVF implementations.
    let ivf = lance_index::vector::ivf::new_ivf_with_pq(
        first_idx.ivf.centroids.values(),
        first_idx.ivf.dimension(),
        first_idx.metric_type,
        column,
        pq_index.pq.clone(),
        None,
        None,
    )?;

    let shuffled = shuffle_dataset(
        data,
        column,
        ivf,
        self.ivf.num_partitions() as u32,
        pq_index.pq.num_sub_vectors(),
        10000,
        2,
    )
    .await?;

    let mut ivf_mut = Ivf::new(self.ivf.centroids.clone());
    write_index_partitions(&mut writer, &mut ivf_mut, shuffled, Some(&[self])).await?;
    let metadata = IvfPQIndexMetadata {
        name: metadata.name.clone(),
        column: column.to_string(),
        dimension: self.ivf.dimension() as u32,
        dataset_version: dataset.version().version,
        metric_type: self.metric_type,
        ivf: ivf_mut,
        pq: pq_index.pq.clone(),
        transforms: vec![],
    };

    let metadata = pb::Index::try_from(&metadata)?;
    let pos = writer.write_protobuf(&metadata).await?;
    writer.write_magics(pos).await?;
    writer.shutdown().await?;

    Ok(new_uuid)
}

/// Append new data to the index, without re-train.
///
/// Returns the UUID of the new index along with a vector of newly indexed fragment ids
///
/// TODO: move this function to `lance-index`
pub async fn append_index<'a>(
    dataset: Arc<Dataset>,
    old_indices: &[&'a IndexMetadata],
    options: OptimizeOptions,
) -> Result<Option<(Uuid, Vec<&'a IndexMetadata>, Option<RoaringBitmap>)>> {
    let mut frag_bitmap = RoaringBitmap::new();
    old_indices.iter().for_each(|idx| {
        frag_bitmap.extend(idx.fragment_bitmap.as_ref().unwrap().iter());
    });

    let unindexed = dataset
        .fragments()
        .iter()
        .filter(|f| !frag_bitmap.contains(f.id as u32))
        .map(|f| f.clone())
        .collect::<Vec<_>>();

    let latest_idx = old_indices.last().ok_or(Error::Index {
        message: "Append index: no index found".to_string(),
        location: location!(),
    })?;
    let column = dataset
        .schema()
        .field_by_id(latest_idx.fields[0])
        .ok_or(Error::Index {
            message: format!(
                "Append index: column {} does not exist",
                latest_idx.fields[0]
            ),
            location: location!(),
        })?;

    // Open all indices.
    let indices = stream::iter(old_indices.iter())
        .map(|idx| async move {
            dataset
                .open_generic_index(&column.name, &idx.uuid.to_string())
                .await
        })
        .buffered(4)
        .try_collect::<Vec<_>>()
        .await?;

    // Sanity check.
    if !indices
        .windows(2)
        .all(|w| w[0].index_type() == w[1].index_type())
    {
        return Err(Error::Index {
            message: "Append index: indices have different types".to_string(),
            location: location!(),
        });
    }

    match indices[0].index_type() {
        IndexType::Scalar => {
            if indices.len() > 1 {
                return Err(Error::Index {
                    message: "Append index: scalar index does not support more than one index yet"
                        .to_string(),
                    location: location!(),
                });
            }
            if options.num_indices_to_merge != 1 {
                return Err(Error::Index {
                    message: "Append index: scalar index does not support merge".to_string(),
                    location: location!(),
                });
            }

            let old_index = &old_indices[0];
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

            Ok(Some((new_uuid, vec![old_index], frag_bitmap)))
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

        dataset.optimize_indices().await.unwrap();
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
