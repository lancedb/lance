// Copyright 2024 Lance Developers.
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

use arrow_array::{Array, FixedSizeListArray};
use lance_linalg::distance::MetricType;

use super::HNSW;
use crate::vector::graph::storage::VectorStorage;

/// Build a HNSW graph.
///
/// Currently, the HNSW graph is fully built in memory.
pub struct HNSWBuilder<V: VectorStorage<f32>> {
    /// max level of
    max_level: u16,

    /// max number of connections ifor each element per layers.
    m_max: usize,

    /// Size of the dynamic list for the candidates
    ef_construction: usize,

    metric_type: MetricType,

    vectors: V,
}

impl<V: VectorStorage<f32>> HNSWBuilder<V> {
    pub fn new(vectors: V) -> Self {
        Self {
            max_level: 8,
            m_max: 16,
            ef_construction: 100,
            metric_type: MetricType::L2,
            vectors,
        }
    }

    /// Metric type
    pub fn metric_type(mut self, metric_type: MetricType) -> Self {
        self.metric_type = metric_type;
        self
    }

    /// The maximum level of the graph.
    pub fn max_level(mut self, max_level: u16) -> Self {
        self.max_level = max_level;
        self
    }

    /// The maximum number of connections for each node per layer.
    pub fn max_num_edges(mut self, m_max: usize) -> Self {
        self.m_max = m_max;
        self
    }

    /// Number of candidates to be considered when searching for the nearest neighbors
    /// during the construction of the graph.
    pub fn ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    pub fn build(self) -> HNSW {
        
        unimplemented!()
    }
}
