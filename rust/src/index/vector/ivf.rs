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

//! IVF - Inverted File index.

use std::sync::Arc;

use arrow_arith::arithmetic::subtract_dyn;
use arrow_array::{
    cast::{as_primitive_array, as_struct_array},
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, StructArray, UInt32Array,
};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::DataType;
use arrow_select::{
    concat::{concat, concat_batches},
    take::take,
};
use async_trait::async_trait;
use futures::{stream::{self, StreamExt}, TryStreamExt};

use super::{
    pq::{PQIndex, ProductQuantizer},
    Query, VectorIndex,
};
use crate::{index::pb, dataset::scanner::Scanner};
use crate::io::{object_reader::ObjectReader, read_message, read_metadata_offset};
use crate::utils::distance::l2_distance;
use crate::{arrow::*, index::pb::vector_index_stage::Stage};
use crate::{
    dataset::{Dataset, ROW_ID},
    index::{IndexBuilder, IndexType},
};
use crate::{Error, Result};

const INDEX_FILE_NAME: &str = "index.idx";

/// IVF PQ Index.
#[derive(Debug)]
pub struct IvfPQIndex<'a> {
    reader: ObjectReader<'a>,

    /// Index name.
    name: String,

    /// The column to build the indices.
    column: String,

    /// Vector dimension.
    dimension: usize,

    /// Ivf file.
    ivf: Ivf,

    /// Number of bits used for product quantization centroids.
    pq: ProductQuantizer,
}

impl<'a> IvfPQIndex<'a> {
    /// Open the IvfPQ index on dataset, specified by the index `name`.
    async fn new(dataset: &'a Dataset, name: &str) -> Result<IvfPQIndex<'a>> {
        let index_dir = dataset.indices_dir().child(name);
        let index_file = index_dir.child(INDEX_FILE_NAME);

        let object_store = dataset.object_store();
        let mut reader = object_store.open(&index_file).await?;

        let file_size = reader.size().await?;
        let prefetch_size = object_store.prefetch_size();
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
            reader,
            name: name.to_string(),
            column: index_metadata.column.clone(),
            dimension: index_metadata.dimension as usize,
            ivf: index_metadata.ivf,
            pq: index_metadata.pq,
        })
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
        let residual_key = subtract_dyn(key, &partition_centroids)?;

        // TODO: Keep PQ index in LRU
        let pq_index = PQIndex::load(&self.reader, &self.pq, offset, length).await?;
        pq_index.search(as_primitive_array(&residual_key), k)
    }
}

#[async_trait]
impl VectorIndex for IvfPQIndex<'_> {
    async fn search(&self, query: &Query) -> Result<RecordBatch> {
        let partition_ids = self.ivf.find_partitions(&query.key, query.nprobs)?;
        let candidates = stream::iter(partition_ids.values())
            .then(|part_id| async move {
                self.search_in_partition(*part_id as usize, &query.key, query.k)
                    .await
            })
            .collect::<Vec<_>>()
            .await;
        let mut batches = vec![];
        for b in candidates {
            batches.push(b?);
        }
        let batch = concat_batches(&batches[0].schema(), &batches)?;

        let score_col = batch.column_by_name("score").ok_or_else(|| {
            Error::IO(format!(
                "score column does not exist in batch: {}",
                batch.schema()
            ))
        })?;
        let refined_index = sort_to_indices(score_col, None, Some(query.k))?;

        let struct_arr = StructArray::from(batch);
        let taken_scores = take(&struct_arr, &refined_index, None)?;
        Ok(as_struct_array(&taken_scores).into())
    }
}

/// Ivf PQ index metadata.
///
/// It contains the on-disk data for a IVF PQ index.
#[derive(Debug)]
pub struct IvfPQIndexMetadata {
    /// Index name
    name: String,

    /// The column to build the index for.
    column: String,

    /// Vector dimension.
    dimension: u32,

    /// The version of dataset where this index was built.
    dataset_version: u64,

    // Ivf related
    ivf: Ivf,

    // PQ configurations.
    pq: ProductQuantizer,
}

/// Convert a IvfPQIndex to protobuf payload
impl TryFrom<&IvfPQIndexMetadata> for pb::Index {
    type Error = Error;

    fn try_from(idx: &IvfPQIndexMetadata) -> std::result::Result<Self, Self::Error> {
        Ok(Self {
            name: idx.name.clone(),
            columns: vec![idx.column.clone()],
            dataset_version: idx.dataset_version,
            index_type: pb::IndexType::Vector.into(),
            implementation: Some(pb::index::Implementation::VectorIndex(pb::VectorIndex {
                spec_version: 1,
                dimension: idx.dimension,
                stages: vec![
                    pb::VectorIndexStage {
                        stage: Some(pb::vector_index_stage::Stage::Ivf(pb::Ivf::try_from(
                            &idx.ivf,
                        )?)),
                    },
                    pb::VectorIndexStage {
                        stage: Some(pb::vector_index_stage::Stage::Pq(pb::Pq {
                            num_bits: idx.pq.nbits,
                            num_sub_vectors: idx.pq.num_sub_vectors as u32,
                        })),
                    },
                ],
            })),
        })
    }
}

impl TryFrom<&pb::Index> for IvfPQIndexMetadata {
    type Error = Error;

    fn try_from(idx: &pb::Index) -> Result<Self> {
        if idx.columns.len() != 1 {
            return Err(Error::Schema("IVF_PQ only supports 1 column".to_string()));
        }
        assert_eq!(idx.index_type, pb::IndexType::Vector as i32);

        let metadata =
            if let Some(idx_impl) = idx.implementation.as_ref() {
                match idx_impl {
                    pb::index::Implementation::VectorIndex(vidx) => {
                        if vidx.stages.len() != 2 {
                            return Err(Error::IO("Only support IVF_PQ now".to_string()));
                        };
                        let stage0 = vidx.stages[0].stage.as_ref().ok_or_else(|| {
                            Error::IO("VectorIndex stage 0 is missing".to_string())
                        })?;
                        let ivf = match stage0 {
                            Stage::Ivf(ivf_pb) => Ok(Ivf::try_from(ivf_pb)?),
                            _ => Err(Error::IO("Stage 0 only supports IVF".to_string())),
                        }?;
                        let stage1 = vidx.stages[1].stage.as_ref().ok_or_else(|| {
                            Error::IO("VectorIndex stage 0 is missing".to_string())
                        })?;
                        let pq = match stage1 {
                            Stage::Pq(pq_proto) => Ok(ProductQuantizer::new(
                                pq_proto.num_sub_vectors as usize,
                                pq_proto.num_bits,
                                vidx.dimension as usize,
                            )),
                            _ => Err(Error::IO("Stage 1 only supports PQ".to_string())),
                        }?;

                        Ok::<Self, Error>(Self {
                            name: idx.name.clone(),
                            column: idx.columns[0].to_string(),
                            dimension: vidx.dimension,
                            dataset_version: idx.dataset_version,
                            ivf,
                            pq,
                        })
                    }
                }?
            } else {
                return Err(Error::IO("Invalid protobuf".to_string()));
            };
        Ok(metadata)
    }
}

/// Ivf Model
#[derive(Debug)]
struct Ivf {
    /// Centroids of each partition.
    ///
    /// It is a 2-D `(num_partitions * dimension)` of float32 array, 64-bit aligned via Arrow
    /// memory allocator.
    centroids: FixedSizeListArray,

    /// Offset of each partition in the file.
    offsets: Vec<usize>,
 
    /// Number of vectors in each partition.
    lengths: Vec<u32>,
}

impl Ivf {
    fn new(centroids: FixedSizeListArray) -> Self {
        Self {
            centroids,
            offsets: vec![],
            lengths: vec![],
        }
    }

    /// Ivf model dimension.
    fn dimension(&self) -> usize {
        self.centroids.value_length() as usize
    }

    /// Use the query vector to find `nprobes` closest partitions.
    fn find_partitions(&self, query: &Float32Array, nprobes: usize) -> Result<UInt32Array> {
        if query.len() != self.dimension() {
            return Err(Error::IO(format!(
                "Ivf::find_partition: dimension mismatch: {} != {}",
                query.len(),
                self.dimension()
            )));
        }
        let distances = l2_distance(query, &self.centroids)? as ArrayRef;
        let top_k_partitions = sort_to_indices(&distances, None, Some(nprobes))?;
        Ok(top_k_partitions)
    }
}

/// Convert IvfModel to protobuf.
impl TryFrom<&Ivf> for pb::Ivf {
    type Error = Error;

    fn try_from(ivf: &Ivf) -> Result<Self> {
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
impl TryFrom<&pb::Ivf> for Ivf {
    type Error = Error;

    fn try_from(proto: &pb::Ivf) -> Result<Self> {
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

/// Computer the residual array from the centroids.
///
/// TODO: not optimized. only occurred in write path.
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

pub struct IvfPqIndexBuilder<'a> {
    dataset: &'a Dataset,

    /// Index name
    name: String,

    /// Vector column to search for.
    column: String,

    dimension: usize,

    /// Number of IVF partitions.
    num_partitions: u32,

    // PQ parameters
    nbits: u32,

    num_sub_vectors: u32,

    /// Max iterations to train a k-mean model.
    kmean_max_iters: u32,
}

impl<'a> IvfPqIndexBuilder<'a> {
    pub fn try_new(
        dataset: &'a Dataset,
        name: &str,
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
            name: name.to_string(),
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
        IndexType::Vector
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

        let object_store = self.dataset.object_store();

        // object_store.
        // Just use column.idx for POC
        let path = self
            .dataset
            .indices_dir()
            .child(format!("{}.idx", self.column));
        let mut writer = object_store.create(&path).await?;

        let mut scanner = self.dataset.scan();
        scanner.project(&[&self.column])?;

        let mut ivf_model = Ivf::new(
            &train_kmean_model(
                &scanner,
                self.dimension,
                self.num_partitions,
                self.kmean_max_iters,
            )
            .await?,
            self.dimension,
        )?;

        scanner = self.dataset.scan();
        scanner.project(&[&self.column])?;
        scanner.with_row_id();
        let partitioned_batches = ivf_model.partition(&scanner).await?;

        for (key, batch) in partitioned_batches.iter() {
            let arr = batch.column_with_name("_vector").unwrap();
            let data = arr.as_any().downcast_ref::<FixedSizeListArray>().unwrap();

            let now = std::time::Instant::now();
            let mut pq =
                ProductQuantizer::new(self.num_sub_vectors, self.nbits, data.value_length() as u32);
            let code = pq.fit_transform(data)?;
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
        writer.write_magics(pos).await?;
        writer.shutdown().await?;

        Ok(())
    }
}

async fn train_kmean_model(
    scanner: &Scanner,
    dimension: usize,
    nclusters: u32,
    max_iterations: u32,
) -> Result<FixedSizeListArray> {
    let schema = scanner.schema();
    assert_eq!(schema.fields.len(), 1);
    // Copy all to memory for now, optimize later.
    let batches = scanner.into_stream().try_collect::<Vec<_>>().await?;
    let mut arr_list = vec![];
    for batch in batches {
        let arr = batch.column_by_name("vector").unwrap();
        let list_arr = as_fixed_size_list_array(&arr);
        arr_list.push(list_arr.values().clone());
    }

    let arrays = arr_list.iter().map(|l| l.as_ref()).collect::<Vec<_>>();

    let all_vectors = concat(&arrays)?;
    let values: &Float32Array = as_primitive_array(&all_vectors);
    let centroids =
        super::kmeans::train_kmeans(values, dimension, nclusters, max_iterations)
            .unwrap();

    Ok(centroids)
}