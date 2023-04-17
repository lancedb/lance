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

///! DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node
///
/// Modified from diskann paper. The vector store is backed by the `lance` dataset.
mod builder;
mod search;

use arrow_array::UInt8Array;
use byteorder::{ByteOrder, LE};

use super::{
    graph::{Vertex, VertexSerDe},
    MetricType,
};
use crate::index::vector::pq::PQBuildParams;
use crate::Result;

pub struct DiskANNParams {
    /// out-degree bound (R)
    pub r: usize,

    /// Distance threshold
    pub alpha: f32,

    /// Search list size
    pub l: usize,

    /// Parameters to build PQ index.
    pub pq_params: PQBuildParams,

    /// Metric type.
    pub metric_type: MetricType,
}

// Default values from DiskANN paper.
impl Default for DiskANNParams {
    fn default() -> Self {
        Self {
            r: 90,
            alpha: 1.2,
            l: 100,
            pq_params: PQBuildParams::default(),
            metric_type: MetricType::L2,
        }
    }
}

impl DiskANNParams {
    pub fn new(r: usize, alpha: f32, l: usize) -> Self {
        Self {
            r,
            alpha,
            l,
            pq_params: PQBuildParams::default(),
            metric_type: MetricType::L2,
        }
    }

    pub fn r(&mut self, r: usize) -> &mut Self {
        self.r = r;
        self
    }

    pub fn alpha(&mut self, alpha: f32) -> &mut Self {
        self.alpha = alpha;
        self
    }

    pub fn l(&mut self, l: usize) -> &mut Self {
        self.l = l;
        self
    }

    /// Set the number of bits for PQ.
    pub fn pq_num_bits(&mut self, nbits: usize) -> &mut Self {
        self.pq_params.num_bits = nbits;
        self
    }

    pub fn pq_num_sub_vectors(&mut self, m: usize) -> &mut Self {
        self.pq_params.num_sub_vectors = m;
        self
    }

    pub fn use_opq(&mut self, use_opq: bool) -> &mut Self {
        self.pq_params.use_opq = use_opq;
        self
    }

    pub fn metric_type(&mut self, metric_type: MetricType) -> &mut Self {
        self.metric_type = metric_type;
        self
    }
}

/// DiskANN Vertex with PQ as persisted data.
#[derive(Clone, Debug)]
pub(crate) struct PQVertex {
    row_id: u64,
    /// PQ code for the vector.
    pq: UInt8Array,
}

impl Vertex for PQVertex {}

struct PQVertexSerDe {
    num_sub_vectors: usize,

    /// Number of bits for each PQ code.
    num_bits: usize,
}

impl PQVertexSerDe {
    pub(crate) fn new(num_sub_vectors: usize, num_bits: usize) -> Self {
        Self {
            num_sub_vectors,
            num_bits,
        }
    }
}

impl VertexSerDe<PQVertex> for PQVertexSerDe {
    #[inline]
    fn size(&self) -> usize {
        8 /* row_id*/ + 8 /* Length of pq */ + self.num_sub_vectors * 1 /* only 8 bits now */
    }

    fn deserialize(&self, data: &[u8]) -> Result<PQVertex> {
        let row_id = LE::read_u64(&data[0..8]);
        let pq_size = LE::read_u64(&data[8..16]) as usize;
        let pq = UInt8Array::from_iter_values(data[16..16 + pq_size].iter().copied());
        Ok(PQVertex { row_id, pq })
    }

    fn serialize(&self, vertex: &PQVertex) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.size());
        bytes.extend_from_slice(&vertex.row_id.to_le_bytes());
        bytes.extend_from_slice(&vertex.pq.len().to_le_bytes());
        bytes.extend_from_slice(&vertex.pq.values());
        bytes
    }
}
