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

use arrow_array::{
    cast::as_primitive_array, Array, ArrayRef, FixedSizeListArray, Float32Array, UInt32Array,
};
use arrow_ord::sort::sort_to_indices;

use crate::arrow::*;
use crate::dataset::Dataset;
use crate::index::pb;
use crate::io::{read_message, read_metadata_offset, ObjectStore};
use crate::utils::distance::l2_distance;
use crate::{Error, Result};

const INDEX_FILE_NAME: &str = "index.idx";

/// IVF PQ Index.
#[derive(Debug)]
pub struct IvfPQIndex<'a> {
    /// Object store to read the indices.
    object_store: &'a ObjectStore,

    /// Index name.
    name: String,

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
        todo!()
    }
}

/// Ivf PQ index metadata.
///
#[derive(Debug)]
pub struct IvfPQIndexMetadata {
    /// Index name
    name: String,

    /// The column to build the index for.
    column: String,

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
            index_type: pb::IndexType::VectorIvfPq.into(),

            implementation: Some(pb::index::Implementation::VectorIndex(pb::VectorIndex {
                spec_version: 1,
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
        if idx.columns.len() != 1 {
            return Err(Error::Schema("IVF_PQ only supports 1 column".to_string()));
        }
        assert_eq!(idx.index_type, pb::IndexType::VectorIvfPq as i32);

        let metadata = if let Some(idx_impl) = idx.implementation.as_ref() {
            match idx_impl {
                pb::index::Implementation::VectorIndex(vidx) => Self {
                    name: idx.name.clone(),
                    column: idx.columns[0].to_string(),
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
    /// Create an IVF model.
    fn try_new(centroids: &[f32], dimension: u32) -> Result<Self> {
        let centorids =
            FixedSizeListArray::try_new(Float32Array::from(centroids.to_vec()), dimension as i32)?;
        Ok(Self {
            centroids: centorids,
            offsets: vec![],
            lengths: vec![],
        })
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
