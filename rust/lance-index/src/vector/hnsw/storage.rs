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

use std::sync::Arc;

use arrow_array::UInt32Array;
use lance_linalg::distance::MetricType;

use crate::vector::graph::storage::{DistCalculator, VectorStorage};

/// Remapping level id to vector id.
///
/// Each node in one level of HNSW has an vector id.
/// Which is not the same as the node id in this level of graph
pub(super) struct HnswRemappingStorage {
    raw_vectors: Arc<dyn VectorStorage>,

    vector_ids: Arc<UInt32Array>,
}

impl HnswRemappingStorage {
    pub fn new(raw_vectors: Arc<dyn VectorStorage>, vector_ids: Arc<UInt32Array>) -> Self {
        Self {
            raw_vectors,
            vector_ids,
        }
    }
}

impl VectorStorage for HnswRemappingStorage {
    fn len(&self) -> usize {
        self.raw_vectors.len()
    }

    fn metric_type(&self) -> MetricType {
        self.raw_vectors.metric_type()
    }

    fn dist_calculator(&self, query: &[f32]) -> Box<dyn DistCalculator> {
        let calc = self.raw_vectors.dist_calculator(query);
        Box::new(HnswDistCalculator {
            raw_calculator: calc,
            vector_ids: self.vector_ids.clone(),
        })
    }
}

struct HnswDistCalculator {
    raw_calculator: Box<dyn DistCalculator>,
    vector_ids: Arc<UInt32Array>,
}

impl DistCalculator for HnswDistCalculator {
    fn distance(&self, vector_ids: &[u32]) -> Vec<f32> {
        let vector_ids = vector_ids
            .iter()
            .map(|&i| self.vector_ids.value(i as usize))
            .collect::<Vec<_>>();
        self.raw_calculator.distance(&vector_ids)
    }
}
