// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! IVF - Inverted File index.

use core::fmt;
use std::{
    any::Any,
    collections::HashMap,
    sync::{Arc, Weak},
};

use arrow::{
    array::as_struct_array,
    compute::{concat_batches, sort_to_indices, take},
};
use arrow_arith::numeric::sub;
use arrow_array::{RecordBatch, StructArray, UInt32Array};
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use futures::prelude::stream::{self, StreamExt, TryStreamExt};
use lance_core::{cache::DEFAULT_INDEX_CACHE_SIZE, Error, Result};
use lance_file::v2::reader::FileReader;
use lance_index::{
    pb,
    vector::{
        ivf::storage::IVF_METADATA_KEY,
        quantizer::Quantization,
        v3::{storage::IvfQuantizationStorage, subindex::IvfSubIndex, DISTANCE_TYPE_KEY},
        Query, DIST_COL,
    },
    Index, IndexType, INDEX_AUXILIARY_FILE_NAME, INDEX_FILE_NAME,
};
use lance_io::{
    object_store::ObjectStore, scheduler::ScanScheduler, traits::Reader, ReadBatchParams,
};
use lance_linalg::{distance::DistanceType, kernels::normalize_arrow};
use moka::sync::Cache;
use object_store::path::Path;
use prost::Message;
use roaring::RoaringBitmap;
use snafu::{location, Location};
use tracing::instrument;

use crate::{
    index::{vector::VectorIndex, PreFilter},
    session::Session,
};

use super::{centroids_to_vectors, Ivf, IvfIndexPartitionStatistics, IvfIndexStatistics};
/// IVF Index.
#[derive(Debug)]
pub struct IVFIndex<I: IvfSubIndex + 'static, Q: Quantization> {
    uuid: String,

    /// Ivf model
    ivf: Ivf,

    reader: FileReader,
    storage: IvfQuantizationStorage<Q>,

    /// Index in each partition.
    sub_index_cache: Cache<String, Arc<I>>,

    distance_type: DistanceType,

    // The session cache holds an Arc to this object so we need to
    // hold a weak pointer to avoid cycles
    /// The session cache, used when fetching pages
    #[allow(dead_code)]
    session: Weak<Session>,
}

impl<I: IvfSubIndex, Q: Quantization> DeepSizeOf for IVFIndex<I, Q> {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.uuid.deep_size_of_children(context) + self.storage.deep_size_of_children(context)
        // Skipping session since it is a weak ref
    }
}

impl<I: IvfSubIndex + 'static, Q: Quantization> IVFIndex<I, Q> {
    /// Create a new IVF index.
    pub(crate) async fn try_new(
        object_store: Arc<ObjectStore>,
        index_dir: Path,
        uuid: String,
        session: Weak<Session>,
    ) -> Result<Self> {
        let scheduler = ScanScheduler::new(object_store, 16);

        let index_reader = FileReader::try_open(
            scheduler
                .open_file(&index_dir.child(uuid.as_str()).child(INDEX_FILE_NAME))
                .await?,
            None,
        )
        .await?;
        let distance_type = DistanceType::try_from(
            index_reader
                .schema()
                .metadata
                .get(DISTANCE_TYPE_KEY)
                .ok_or(Error::Index {
                    message: format!("{} not found", DISTANCE_TYPE_KEY),
                    location: location!(),
                })?
                .as_str(),
        )?;

        let ivf_pb_bytes =
            hex::decode(index_reader.schema().metadata.get(IVF_METADATA_KEY).ok_or(
                Error::Index {
                    message: format!("{} not found", IVF_METADATA_KEY),
                    location: location!(),
                },
            )?)
            .map_err(|e| Error::Index {
                message: format!("Failed to decode IVF metadata: {}", e),
                location: location!(),
            })?;
        let ivf = Ivf::try_from(&pb::Ivf::decode(ivf_pb_bytes.as_ref())?)?;

        let storage_reader = FileReader::try_open(
            scheduler
                .open_file(
                    &index_dir
                        .child(uuid.as_str())
                        .child(INDEX_AUXILIARY_FILE_NAME),
                )
                .await?,
            None,
        )
        .await?;
        let storage = IvfQuantizationStorage::open(storage_reader).await?;

        Ok(Self {
            uuid,
            ivf,
            reader: index_reader,
            storage,
            sub_index_cache: Cache::new(DEFAULT_INDEX_CACHE_SIZE as u64),
            distance_type,
            session,
        })
    }

    #[instrument(level = "debug", skip(self))]
    pub async fn load_partition(&self, partition_id: usize, write_cache: bool) -> Result<Arc<I>> {
        let cache_key = format!("{}-ivf-{}", self.uuid, partition_id);
        let part_index = if let Some(part_idx) = self.sub_index_cache.get(&cache_key) {
            part_idx
        } else {
            if partition_id >= self.ivf.lengths.len() {
                return Err(Error::Index {
                    message: format!(
                        "partition id {} is out of range of {} partitions",
                        partition_id,
                        self.ivf.lengths.len()
                    ),
                    location: location!(),
                });
            }

            let offset = self.ivf.offsets[partition_id];
            let length = self.ivf.lengths[partition_id] as usize;
            let batches = self
                .reader
                .read_stream(ReadBatchParams::Range(offset..offset + length), 4096, 16)?
                .peekable()
                .try_collect::<Vec<_>>()
                .await?;
            let schema = Arc::new(self.reader.schema().as_ref().into());
            let batch = concat_batches(&schema, batches.iter())?;
            let idx = Arc::new(I::load(batch)?);
            if write_cache {
                self.sub_index_cache.insert(cache_key.clone(), idx.clone());
            }
            idx
        };
        Ok(part_index)
    }

    async fn search_in_partition(
        &self,
        partition_id: usize,
        query: &Query,
        pre_filter: Arc<PreFilter>,
    ) -> Result<RecordBatch> {
        let part_index = self.load_partition(partition_id, true).await?;

        let query = self.preprocess_query(partition_id, query)?;
        let storage = self.storage.load_partition(partition_id).await?;
        let param = (&query).into();
        pre_filter.wait_for_ready().await?;
        part_index.search(query.key, query.k, param, &storage, pre_filter)
    }

    /// preprocess the query vector given the partition id.
    ///
    /// Internal API with no stability guarantees.
    pub fn preprocess_query(&self, partition_id: usize, query: &Query) -> Result<Query> {
        if I::use_residual() {
            let partition_centroids = self.ivf.centroids.value(partition_id);
            let residual_key = sub(&query.key, &partition_centroids)?;
            let mut part_query = query.clone();
            part_query.key = residual_key;
            Ok(part_query)
        } else {
            Ok(query.clone())
        }
    }

    pub fn find_partitions(&self, query: &Query) -> Result<UInt32Array> {
        let dt = if self.distance_type == DistanceType::Cosine {
            DistanceType::L2
        } else {
            self.distance_type
        };

        self.ivf.find_partitions(&query.key, query.nprobes, dt)
    }
}

#[async_trait]
impl<I: IvfSubIndex + 'static, Q: Quantization + 'static> Index for IVFIndex<I, Q> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn index_type(&self) -> IndexType {
        IndexType::Vector
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let partitions_statistics = self
            .ivf
            .lengths
            .iter()
            .map(|&len| IvfIndexPartitionStatistics { size: len })
            .collect::<Vec<_>>();

        let centroid_vecs = centroids_to_vectors(&self.ivf.centroids)?;

        Ok(serde_json::to_value(IvfIndexStatistics {
            index_type: "IVF".to_string(),
            uuid: self.uuid.clone(),
            uri: self.uuid.clone(),
            metric_type: self.distance_type.to_string(),
            num_partitions: self.ivf.num_partitions(),
            sub_index: Default::default(),
            partitions: partitions_statistics,
            centroids: centroid_vecs,
        })?)
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        unimplemented!(
            "this method is only needed for migrating older manifests, not for this new index"
        )
    }
}

#[async_trait]
impl<I: IvfSubIndex + fmt::Debug + 'static, Q: Quantization + fmt::Debug + 'static> VectorIndex
    for IVFIndex<I, Q>
{
    async fn search(&self, query: &Query, pre_filter: Arc<PreFilter>) -> Result<RecordBatch> {
        let mut query = query.clone();
        if self.distance_type == DistanceType::Cosine {
            let key = normalize_arrow(&query.key)?;
            query.key = key;
        };

        let partition_ids = self.find_partitions(&query)?;
        assert!(partition_ids.len() <= query.nprobes);
        let part_ids = partition_ids.values().to_vec();
        let batches = stream::iter(part_ids)
            .map(|part_id| self.search_in_partition(part_id as usize, &query, pre_filter.clone()))
            .buffer_unordered(num_cpus::get())
            .try_collect::<Vec<_>>()
            .await?;
        let batch = concat_batches(&batches[0].schema(), &batches)?;

        let dist_col = batch.column_by_name(DIST_COL).ok_or_else(|| {
            Error::io(
                format!(
                    "_distance column does not exist in batch: {}",
                    batch.schema()
                ),
                location!(),
            )
        })?;

        // TODO: Use a heap sort to get the top-k.
        let limit = query.k * query.refine_factor.unwrap_or(1) as usize;
        let selection = sort_to_indices(dist_col, None, Some(limit))?;
        let struct_arr = StructArray::from(batch);
        let taken_distances = take(&struct_arr, &selection, None)?;
        Ok(as_struct_array(&taken_distances).into())
    }

    fn is_loadable(&self) -> bool {
        false
    }

    fn use_residual(&self) -> bool {
        false
    }

    fn check_can_remap(&self) -> Result<()> {
        Ok(())
    }

    async fn load(
        &self,
        _reader: Arc<dyn Reader>,
        _offset: usize,
        _length: usize,
    ) -> Result<Box<dyn VectorIndex>> {
        Err(Error::Index {
            message: "Flat index does not support load".to_string(),
            location: location!(),
        })
    }

    fn row_ids(&self) -> &[u64] {
        todo!("this method is for only IVF_HNSW_* index");
    }

    fn remap(&mut self, _mapping: &HashMap<u64, Option<u64>>) -> Result<()> {
        // This will be needed if we want to clean up IVF to allow more than just
        // one layer (e.g. IVF -> IVF -> PQ).  We need to pass on the call to
        // remap to the lower layers.

        // Currently, remapping for IVF is implemented in remap_index_file which
        // mirrors some of the other IVF routines like build_ivf_pq_index
        Err(Error::Index {
            message: "Remapping IVF in this way not supported".to_string(),
            location: location!(),
        })
    }

    fn metric_type(&self) -> DistanceType {
        self.distance_type
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        ops::Range,
        sync::Arc,
    };

    use arrow::{
        array::AsArray,
        datatypes::{Float32Type, UInt64Type},
    };
    use arrow_array::{Array, FixedSizeListArray, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::ROW_ID;
    use lance_index::DatasetIndexExt;
    use lance_linalg::distance::DistanceType;
    use lance_testing::datagen::generate_random_array_with_range;
    use tempfile::tempdir;

    use crate::{index::vector::VectorIndexParams, Dataset};

    const DIM: usize = 32;

    async fn generate_test_dataset(
        test_uri: &str,
        range: Range<f32>,
    ) -> (Dataset, Arc<FixedSizeListArray>) {
        let vectors = generate_random_array_with_range(1000 * DIM, range);
        let metadata: HashMap<String, String> = vec![("test".to_string(), "ivf_pq".to_string())]
            .into_iter()
            .collect();

        let schema: Arc<_> = Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            true,
        )])
        .with_metadata(metadata)
        .into();
        let array = Arc::new(FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap());
        let batch = RecordBatch::try_new(schema.clone(), vec![array.clone()]).unwrap();

        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let dataset = Dataset::write(batches, test_uri, None).await.unwrap();
        (dataset, array)
    }

    fn ground_truth(
        vectors: &FixedSizeListArray,
        query: &[f32],
        k: usize,
        distance_type: DistanceType,
    ) -> Vec<(f32, u64)> {
        let mut dists = vec![];
        for i in 0..vectors.len() {
            let dist = distance_type.func()(
                query,
                vectors.value(i).as_primitive::<Float32Type>().values(),
            );
            dists.push((dist, i as u64));
        }
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dists.truncate(k);
        dists
    }

    #[tokio::test]
    async fn test_build_ivf_flat() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let (mut dataset, vectors) = generate_test_dataset(test_uri, 0.0..1.0).await;

        let nlist = 16;
        let params = VectorIndexParams::ivf_flat(nlist, DistanceType::L2);
        dataset
            .create_index(
                &["vector"],
                lance_index::IndexType::Vector,
                None,
                &params,
                true,
            )
            .await
            .unwrap();

        let query = vectors.value(0);
        let k = 100;
        let result = dataset
            .scan()
            .nearest("vector", query.as_primitive::<Float32Type>(), k)
            .unwrap()
            .nprobs(nlist)
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();

        let row_ids = result
            .column_by_name(ROW_ID)
            .unwrap()
            .as_primitive::<UInt64Type>()
            .values()
            .to_vec();
        let dists = result
            .column_by_name("_distance")
            .unwrap()
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        let results = dists
            .into_iter()
            .zip(row_ids.into_iter())
            .collect::<Vec<_>>();
        let row_ids = results.iter().map(|(_, id)| *id).collect::<HashSet<_>>();

        let gt = ground_truth(
            &vectors,
            query.as_primitive::<Float32Type>().values(),
            k,
            DistanceType::L2,
        );
        let gt_set = gt.iter().map(|r| r.1).collect::<HashSet<_>>();

        let recall = row_ids.intersection(&gt_set).count() as f32 / k as f32;
        assert!(
            recall >= 1.0,
            "recall: {}\n results: {:?}\n\ngt: {:?}",
            recall,
            results,
            gt,
        );
    }
}
