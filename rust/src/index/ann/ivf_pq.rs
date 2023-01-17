// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::Arc;
use std::{collections::BTreeMap, vec};

use arrow_arith::{
    aggregate::{max, min},
    arithmetic::subtract_dyn,
};
use arrow_array::cast::as_struct_array;
use arrow_array::{cast::as_primitive_array, ArrayRef, UInt8Array};
use arrow_array::{
    Array, BooleanArray, FixedSizeListArray, Float32Array, RecordBatch, StructArray, UInt32Array,
};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use arrow_select::{
    concat::{concat, concat_batches},
    filter::filter_record_batch,
    take::take,
};
use async_trait::async_trait;
use futures::{stream, StreamExt, TryStreamExt};

use super::distance::euclidean_distance;
use super::pq::ProductQuantizer;
use super::SearchParams;
use crate::index::{pb, IndexBuilder, IndexType};
use crate::io::object_reader::ObjectReader;
use crate::{arrow::*, io::read_metadata_offset};
use crate::{
    dataset::{Dataset, Scanner, ROW_ID},
    io::read_message,
};
use crate::{Error, Result};

const SPEC_VERSION: u32 = 1;

/// IVF PQ Index.
pub struct IvfPQIndex<'a> {
    name: String,

    reader: ObjectReader<'a>,

    /// Vector column to search for.
    column: String,

    dimension: u32,

    ivf: IvfModel,

    num_bits: u32,
    num_sub_vectors: u32,
}

impl<'a> IvfPQIndex<'a> {
    /// Open index with the index `name` for the dataset.
    pub async fn open(dataset: &'a Dataset, name: &str) -> Result<IvfPQIndex<'a>> {
        // Prefetch at most 1024 of pages.
        let index_file = dataset.index_dir().child(format!("{}.idx", name));
        let mut reader = dataset.object_store.open(&index_file).await?;

        let file_size = reader.size().await?;
        let prefetch_size = dataset.object_store.prefetch_size();
        let begin = if file_size < prefetch_size {
            0
        } else {
            file_size - prefetch_size
        };
        let tail_bytes = reader.get_range(begin..file_size).await?;
        let metadata_pos = read_metadata_offset(&tail_bytes)?;
        let proto: pb::Index = if metadata_pos < file_size - tail_bytes.len() {
            // We have not read the metadata bytes yet.
            reader.read_message(metadata_pos).await?
        } else {
            let offset = tail_bytes.len() - (file_size - metadata_pos);
            read_message(&tail_bytes.slice(offset..))?
        };
        let index_metadata = IvfPQIndexMetadata::try_from(&proto)?;

        Ok(Self {
            name: name.to_string(),
            reader,
            column: index_metadata.column.clone(),
            dimension: index_metadata.dimension,
            ivf: index_metadata.ivf,
            num_bits: index_metadata.num_bits,
            num_sub_vectors: index_metadata.num_sub_vectors,
        })
    }

    pub async fn search(&self, params: &SearchParams) -> Result<RecordBatch> {
        let partition_ids = self.ivf.locate_partitions(&params.key, params.nprob)?;
        let key = &params.key;
        let candidates = stream::iter(partition_ids.iter().map(|p| p.unwrap()))
            .then(|part_id| async move {
                self.search_in_partition(part_id as usize, key, params.k)
                    .await
            })
            .collect::<Vec<_>>()
            .await;
        let mut batches = vec![];
        for b in candidates {
            batches.push(b?);
        }
        let batch = concat_batches(&batches[0].schema(), &batches)?;

        let score_col = batch.column_with_name("score").unwrap();
        let refined_index = sort_to_indices(score_col, None, Some(params.k))?;

        let struct_arr = StructArray::from(batch);
        let taken_scores = take(&struct_arr, &refined_index, None)?;
        Ok(as_struct_array(&taken_scores).into())
    }

    async fn search_in_partition(
        &self,
        partition_id: usize,
        key: &Float32Array,
        k: usize,
    ) -> Result<RecordBatch> {
        let offset = self.ivf.offsets[partition_id];
        let length = self.ivf.lengths[partition_id] as usize;
        let partition_centroids = self.ivf.centroids.value(partition_id);
        let resi_key = subtract_dyn(key, &partition_centroids)?;
        let residual_key: &Float32Array = as_primitive_array(&resi_key);

        // TODO: read code book, PQ code and row_ids in parallel.
        let code_book_length = ProductQuantizer::codebook_length(self.num_bits, self.dimension);
        let codebook = self
            .reader
            .read_fixed_stride_array(&DataType::Float32, offset, code_book_length as usize)
            .await?;

        let pq_code_offset = offset + code_book_length as usize * 4;
        let pq_code_length = self.num_sub_vectors as usize * length;
        let pq_code = self
            .reader
            .read_fixed_stride_array(&DataType::UInt8, pq_code_offset, pq_code_length)
            .await?;

        let row_id_offset = pq_code_offset + pq_code_length /* *1 */;
        let row_ids = self
            .reader
            .read_fixed_stride_array(&DataType::UInt64, row_id_offset, length)
            .await?;

        let centroids: &Float32Array = as_primitive_array(codebook.as_ref());
        let pq =
            ProductQuantizer::new_with_centroids(self.num_bits, self.num_sub_vectors, centroids);
        let pq_code_arr: &UInt8Array = as_primitive_array(&pq_code);
        let distances = pq.search(
            pq_code_arr.data_ref().buffers()[0].typed_data(),
            residual_key,
        ) as ArrayRef;

        let top_k_indices = sort_to_indices(&distances, None, Some(k))?;

        let scores = take(&distances, &top_k_indices, None)?;
        let best_row_ids = take(&row_ids, &top_k_indices, None)?;
        Ok(RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                ArrowField::new("score", DataType::Float32, false),
                ArrowField::new(ROW_ID, DataType::UInt64, false),
            ])),
            vec![scores, best_row_ids],
        )?)
    }
}

/// Ivf PQ index metadata.
///
#[derive(Debug)]
pub struct IvfPQIndexMetadata {
    name: String,

    dataset_version: u64,

    /// Vector column to search for.
    column: String,

    dimension: u32,

    // Ivf related
    ivf: IvfModel,

    // PQ configurations.
    num_bits: u32,
    num_sub_vectors: u32,
}

impl IvfPQIndexMetadata {
    fn new(
        dataset: &Dataset,
        name: &str,
        column: &str,
        ivf: IvfModel,
        pq_num_bits: u32,
        pq_num_sub_vectors: u32,
    ) -> Self {
        let field = dataset.schema().field(name).unwrap();
        let dimension = match field.data_type() {
            DataType::FixedSizeList(_, d) => d,
            _ => panic!("only support fixed size list"),
        };
        Self {
            name: name.to_string(),
            column: column.to_string(),
            dataset_version: dataset.version().version,
            dimension: dimension as u32,
            ivf,
            num_bits: pq_num_bits,
            num_sub_vectors: pq_num_sub_vectors,
        }
    }
}

/// Convert a IvfPQIndex to protobuf payload
impl TryFrom<&IvfPQIndexMetadata> for pb::Index {
    type Error = Error;

    fn try_from(idx: &IvfPQIndexMetadata) -> std::result::Result<Self, Self::Error> {
        Ok(Self {
            name: idx.name.clone(),
            columns: vec![idx.column.clone()],
            dataset_version: idx.dataset_version,
            index_type: pb::IndexType::VectorIvfPq.into(),

            implementation: Some(pb::index::Implementation::VectorIndex(pb::VectorIndex {
                spec_version: SPEC_VERSION,
                dimensions: idx.dimension as u32,
                pq: Some(pb::ProductQuantilizationInfo {
                    num_bits: idx.num_bits,
                    num_subvectors: idx.num_sub_vectors,
                }),
                ivf: Some(pb::Ivf::try_from(&idx.ivf)?),
            })),
        })
    }
}

impl TryFrom<&pb::Index> for IvfPQIndexMetadata {
    type Error = Error;

    fn try_from(idx: &pb::Index) -> Result<Self> {
        assert_eq!(idx.columns.len(), 1);
        assert_eq!(idx.index_type, pb::IndexType::VectorIvfPq as i32);

        let metadata = if let Some(idx_impl) = idx.implementation.as_ref() {
            match idx_impl {
                pb::index::Implementation::VectorIndex(vidx) => Self {
                    name: idx.name.clone(),
                    column: idx.columns[0].clone(),
                    dataset_version: idx.dataset_version,
                    ivf: vidx
                        .ivf
                        .as_ref()
                        .map(|ivf| IvfModel::try_from(ivf).unwrap())
                        .unwrap(),
                    dimension: vidx.dimensions,
                    num_bits: vidx.pq.as_ref().map(|pq| pq.num_bits).unwrap_or(8),
                    num_sub_vectors: vidx.pq.as_ref().map(|pq| pq.num_subvectors).unwrap(),
                },
            }
        } else {
            return Err(Error::IO("Invalid protobuf".to_string()));
        };
        Ok(metadata)
    }
}

/// Partition one [`RecordBatch`] based on the partition column(s).
///
/// Currently, it only supports partitioning over one column.
fn partition(batch: &RecordBatch, partition_col: &str) -> Result<BTreeMap<u32, RecordBatch>> {
    let part_col = batch.column_with_name(partition_col).unwrap();
    let partition_ids: &UInt32Array = as_primitive_array(part_col);
    let min_id = min(partition_ids).unwrap_or(0);
    let max_id = max(partition_ids).unwrap_or(1024 * 1024);

    let mut partitions = BTreeMap::<u32, RecordBatch>::new();
    // Reconstruct the results.
    for part_id in min_id..max_id + 1 {
        let predicates = BooleanArray::from_unary(partition_ids, |x| x == part_id);
        let parted_batch = filter_record_batch(batch, &predicates)?;
        if parted_batch.num_rows() > 0 {
            partitions.insert(part_id, parted_batch);
        }
    }
    Ok(partitions)
}

/// Computer the residual array from the centroids.
///
/// TODO: not optimized. only occured in write path.
fn compute_residual(
    vectors: &FixedSizeListArray,
    centroids: &Float32Array,
) -> Result<Arc<FixedSizeListArray>> {
    assert_eq!(vectors.value_length() as usize, centroids.len());
    let mut residual_vectors = vec![];
    for idx in 0..vectors.len() {
        let val = vectors.value(idx);
        let residual = subtract_dyn(val.as_ref(), centroids)?;
        residual_vectors.push(residual);
    }
    let residual_refs: Vec<&dyn Array> = residual_vectors.iter().map(|r| r.as_ref()).collect();

    let values = concat(&residual_refs)?;
    let f32_values: &Float32Array = as_primitive_array(values.as_ref());
    Ok(Arc::new(FixedSizeListArray::try_new(
        f32_values,
        centroids.len() as i32,
    )?))
}

/// IvfModel
#[derive(Debug)]
struct IvfModel {
    // A 2-D (num_partitions * dimension) array.
    centroids: FixedSizeListArray,
    // Offset of each partition in the file.
    offsets: Vec<usize>,
    // Number of vector in each partition.
    lengths: Vec<u32>,
}

impl IvfModel {
    fn try_new(centroids: &[f32], dimension: u32) -> Result<Self> {
        let centorids =
            FixedSizeListArray::try_new(Float32Array::from(centroids.to_vec()), dimension as i32)?;
        Ok(Self {
            centroids: centorids,
            offsets: vec![],
            lengths: vec![],
        })
    }

    fn add_partition(&mut self, offset: usize, len: u32) {
        self.offsets.push(offset);
        self.lengths.push(len);
    }

    /// Scan the dataset, and partition the batches based on the IVF partitions / Voronois.
    ///
    /// Currently, it will keep all batches in memory.
    async fn partition(&self, scanner: &Scanner<'_>) -> Result<BTreeMap<u32, RecordBatch>> {
        const PARTITION_ID_COLUMN: &str = "__ivf_part_id";

        let schema = scanner.schema();
        let column_name = schema.field(0).name();
        let partitions_with_id = scanner
            .into_stream()
            .and_then(|b| async move {
                let now = std::time::Instant::now();
                let arr = b.column_with_name(column_name).ok_or_else(|| {
                    Error::IO(format!("Dataset does not have column {}", column_name))
                })?;
                let vectors: &FixedSizeListArray = arr.as_any().downcast_ref().unwrap();
                let vec = vectors.clone();
                let centroids = self.centroids.clone();
                let partition_ids = tokio::task::spawn_blocking(move || {
                    (0..vec.len())
                        .map(|idx| {
                            let arr = vec.value(idx);
                            let f: &Float32Array = as_primitive_array(&arr);
                            Ok(argmin(euclidean_distance(f, &centroids)?.as_ref()).unwrap())
                        })
                        .collect::<Result<Vec<u32>>>()
                })
                .await??;
                let partition_column = Arc::new(UInt32Array::from(partition_ids));
                let batch_with_part_id = b.try_with_column(
                    ArrowField::new(PARTITION_ID_COLUMN, DataType::UInt32, false),
                    partition_column,
                )?;
                let partitioned = partition(&batch_with_part_id, PARTITION_ID_COLUMN)?;
                Ok(partitioned)
            })
            .try_fold(
                BTreeMap::<u32, Vec<RecordBatch>>::new(),
                |mut builder, partitions| async move {
                    for (id, batch) in partitions {
                        builder.entry(id).or_insert(vec![]);
                        if let Some(batches) = builder.get_mut(&id) {
                            batches.push(batch);
                        }
                    }
                    Ok(builder)
                },
            )
            .await?;
        let mut partitions = BTreeMap::<u32, RecordBatch>::new();
        for (key, value) in partitions_with_id.iter() {
            let batch = concat_batches(&value[0].schema(), value)?;
            let arr = self.centroids.value(key.clone() as usize);
            let centroids: &Float32Array = as_primitive_array(&arr);
            let column = batch.column_with_name(column_name).unwrap();
            let vectors = as_fixed_size_list_array(column);
            let residual = compute_residual(vectors, centroids)?;
            let residual_schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("_vector", residual.data_type().clone(), false),
                ArrowField::new(ROW_ID, DataType::UInt64, false),
            ]));
            let batch = RecordBatch::try_new(
                residual_schema,
                vec![residual, batch.column_with_name(ROW_ID).unwrap().clone()],
            )?;

            partitions.insert(*key, batch);
        }
        Ok(partitions)
    }

    /// Use the query vector to find `nprobes` closes partition.
    fn locate_partitions(&self, query: &Float32Array, nprobes: usize) -> Result<UInt32Array> {
        let distances = euclidean_distance(query, &self.centroids)? as ArrayRef;
        let top_k_partitions = sort_to_indices(&distances, None, Some(nprobes))?;
        Ok(top_k_partitions)
    }
}

/// Convert IvfModel to protobuf.
impl TryFrom<&IvfModel> for pb::Ivf {
    type Error = Error;

    fn try_from(ivf: &IvfModel) -> Result<Self> {
        if ivf.offsets.len() != ivf.centroids.len() {
            return Err(Error::IO("Ivf model has not been populated".to_string()));
        }
        let centroids_arr = ivf.centroids.values();
        let f32_centroids: &Float32Array = as_primitive_array(&centroids_arr);
        Ok(Self {
            centroids: f32_centroids.iter().map(|v| v.unwrap()).collect(),
            offsets: ivf.offsets.iter().map(|o| *o as u64).collect(),
            lengths: ivf.lengths.clone(),
        })
    }
}

/// Convert IvfModel to protobuf.
impl TryFrom<&pb::Ivf> for IvfModel {
    type Error = Error;

    fn try_from(proto: &pb::Ivf) -> Result<Self> {
        // let centroids_arr = proto.centroids.values();
        let f32_centroids: Float32Array = Float32Array::from(proto.centroids.clone());
        let dimension = f32_centroids.len() / proto.offsets.len();
        let centroids = FixedSizeListArray::try_new(f32_centroids, dimension as i32)?;
        Ok(Self {
            centroids,
            offsets: proto.offsets.iter().map(|o| *o as usize).collect(),
            lengths: proto.lengths.clone(),
        })
    }
}

async fn train_kmean_model(
    scanner: &Scanner<'_>,
    dimension: usize,
    nclusters: u32,
    max_iterations: u32,
) -> Result<Vec<f32>> {
    let schema = scanner.schema();
    assert_eq!(schema.fields.len(), 1);
    // Copy to memory for now, optimize later.
    let batches = scanner.into_stream().collect::<Vec<_>>().await;
    let mut arr_list = vec![];
    for batch in batches {
        let b = batch.unwrap();
        let arr = b.column_with_name("vector").unwrap();
        let list_arr = as_fixed_size_list_array(&arr);
        arr_list.push(list_arr.values().clone());
    }

    let arrays = arr_list.iter().map(|l| l.as_ref()).collect::<Vec<_>>();

    let all_vectors = concat(&arrays)?;
    let values: &Float32Array = as_primitive_array(&all_vectors);
    let now = std::time::Instant::now();
    let centroids =
        super::kmeans::train_kmean_model_from_array(values, dimension, nclusters, max_iterations)
            .unwrap();
    println!("Finish kmean in {} seconds", now.elapsed().as_secs_f32());

    Ok(centroids)
}

struct IvfPqIndexBuilder<'a> {
    dataset: &'a Dataset,

    /// Vector column to search for.
    column: String,

    dimension: usize,

    /// Number of IVF partitions.
    num_partitions: u32,

    // PQ parameters
    nbits: u32,

    num_sub_vectors: u32,

    /// Max iterations to train k-mean models.
    kmean_max_iters: u32,
}

impl<'a> IvfPqIndexBuilder<'a> {
    pub fn try_new(
        dataset: &'a Dataset,
        column: &str,
        num_partitions: u32,
        num_sub_vectors: u32,
    ) -> Result<Self> {
        let field = dataset.schema().field(column).ok_or(Error::IO(format!(
            "Column {} does not exist in the dataset",
            column
        )))?;
        let dimension = match field.data_type() {
            DataType::FixedSizeList(_, d) => d,
            _ => {
                return Err(Error::IO(format!("Column {} is not a vector type", column)));
            }
        };
        Ok(Self {
            dataset,
            column: column.to_string(),
            dimension: dimension as usize,
            num_partitions,
            num_sub_vectors,
            nbits: 8,
            kmean_max_iters: 100,
        })
    }
}

#[async_trait]
impl IndexBuilder for IvfPqIndexBuilder<'_> {
    fn index_type() -> IndexType {
        IndexType::VectorIvfPQ
    }

    /// Build the IVF_PQ index
    async fn build(&self) -> Result<()> {
        // Step 1. Sanity check
        let schema = self.dataset.schema().field(&self.column);
        if schema.is_none() {
            return Err(Error::IO(format!(
                "Building index: column {} does not exist in dataset: {:?}",
                self.column, self.dataset
            )));
        }

        let object_store = self.dataset.object_store.as_ref();

        // object_store.
        // Just use column.idx for POC
        let path = self
            .dataset
            .index_dir()
            .child(format!("{}.idx", self.column));
        let mut writer = object_store.create(&path).await?;

        let mut scanner = self.dataset.scan();
        scanner.project(&[&self.column])?;

        let now = std::time::Instant::now();
        let mut ivf_model = IvfModel::try_new(
            &train_kmean_model(
                &scanner,
                self.dimension,
                self.num_partitions,
                self.kmean_max_iters,
            )
            .await?,
            self.dimension as u32,
        )?;
        println!("Finish IVF training in: {}", now.elapsed().as_secs_f32());

        scanner = self.dataset.scan();
        scanner.project(&[&self.column])?;
        scanner.with_row_id();
        let partitioned_batches = ivf_model.partition(&scanner).await?;
        println!("Finish partitioning in {}", now.elapsed().as_secs_f32());

        for (key, batch) in partitioned_batches.iter() {
            let arr = batch.column_with_name("_vector").unwrap();
            let data = arr.as_any().downcast_ref::<FixedSizeListArray>().unwrap();

            let now = std::time::Instant::now();
            let mut pq =
                ProductQuantizer::new(self.num_sub_vectors, self.nbits, data.value_length() as u32);
            let code = pq.fit_transform(data)?;
            println!(
                "Product qualitization on partition: {}, size={}: time={}, {:?}",
                key,
                arr.len(),
                now.elapsed().as_secs_f32(),
                code.get_array_memory_size(),
            );
            let row_id_column = batch.column_with_name(ROW_ID).unwrap();
            ivf_model.add_partition(writer.tell(), data.len() as u32);
            writer
                .write_plain_encoded_array(pq.codebook.as_ref().unwrap())
                .await?;
            writer
                .write_plain_encoded_array(code.values().as_ref())
                .await?;
            writer
                .write_plain_encoded_array(row_id_column.as_ref())
                .await?;
        }

        let metadata = IvfPQIndexMetadata::new(
            self.dataset,
            &self.column,
            &self.column,
            ivf_model,
            self.nbits,
            self.num_sub_vectors,
        );

        let metadata = pb::Index::try_from(&metadata)?;
        let pos = writer.write_protobuf(&metadata).await?;
        writer.write_magic(pos).await?;
        writer.shutdown().await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::utils::generate_random_array;
    use std::env::current_dir;

    #[tokio::test(flavor = "multi_thread", worker_threads = 32)]
    async fn test_build_index() {
        let dataset_uri = current_dir().unwrap().join("vec_data");
        let dataset = Dataset::open(dataset_uri.as_path().to_str().unwrap())
            .await
            .unwrap();
        let idx = IvfPqIndexBuilder::try_new(&dataset, "vector", 256, 16).unwrap();
        idx.build().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_read_index() {
        let dataset_uri = current_dir().unwrap().join("vec_data");
        let dataset = Dataset::open(dataset_uri.as_path().to_str().unwrap())
            .await
            .unwrap();
        let idx = IvfPQIndex::open(&dataset, "vec").await.unwrap();
        let params = &SearchParams {
            key: generate_random_array(1024),
            k: 10,
            nprob: 10,
        };
        idx.search(params).await.unwrap();
    }
}
