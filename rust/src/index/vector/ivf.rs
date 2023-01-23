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
    UInt8Array,
};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use arrow_select::{concat::concat_batches, take::take};
use async_trait::async_trait;
use futures::stream::{self, StreamExt};

use super::{pq::ProductQuantizer, Query, VectorIndex};
use crate::arrow::*;
use crate::dataset::Dataset;
use crate::index::pb;
use crate::io::{object_reader::ObjectReader, read_message, read_metadata_offset};
use crate::utils::distance::l2_distance;
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
    dimension: u32,

    /// Ivf file.
    ivf: Ivf,

    /// Number of bits used for product quantlization centroids.
    num_bits: u32,
    num_sub_vectors: u32,
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
            dimension: index_metadata.dimension,
            ivf: index_metadata.ivf,
            num_bits: index_metadata.num_bits,
            num_sub_vectors: index_metadata.num_sub_vectors,
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
        let resi_key = subtract_dyn(key, &partition_centroids)?;
        let residual_key: &Float32Array = as_primitive_array(&resi_key);

        // TODO: read code book, PQ code and row_ids in parallel.
        let code_book_length = ProductQuantizer::codebook_length(self.num_bits, self.dimension);
        let codebook = self
            .reader
            .read_fixed_stride_array(&DataType::Float32, offset, code_book_length as usize, ..)
            .await?;

        let pq_code_offset = offset + code_book_length as usize * 4;
        let pq_code_length = self.num_sub_vectors as usize * length;
        let pq_code = self
            .reader
            .read_fixed_stride_array(&DataType::UInt8, pq_code_offset, pq_code_length, ..)
            .await?;

        let row_id_offset = pq_code_offset + pq_code_length /* *1 */;
        let row_ids = self
            .reader
            .read_fixed_stride_array(&DataType::UInt64, row_id_offset, length, ..)
            .await?;

        let centroids: &Float32Array = as_primitive_array(codebook.as_ref());
        let pq = ProductQuantizer::new_with_centroids(
            self.num_bits,
            self.num_sub_vectors,
            Arc::new(centroids.clone()),
        );
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
                ArrowField::new("_rowid", DataType::UInt64, false),
            ])),
            vec![scores, best_row_ids],
        )?)
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
    num_bits: u32,
    num_sub_vectors: u32,
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
                            num_bits: idx.num_bits,
                            num_sub_vectors: idx.num_sub_vectors,
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

        let metadata = if let Some(idx_impl) = idx.implementation.as_ref() {
            match idx_impl {
                pb::index::Implementation::VectorIndex(vidx) => Self {
                    name: idx.name.clone(),
                    column: idx.columns[0].to_string(),
                    dimension: vidx.dimension,
                    dataset_version: idx.dataset_version,
                    ivf: vidx
                        .ivf
                        .as_ref()
                        .map(|ivf| Ivf::try_from(ivf).unwrap())
                        .ok_or_else(|| Error::IO("Could not read IVF metadata".to_string()))?,
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
