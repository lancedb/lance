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

//! In-memory graph representations.

use std::sync::Arc;

use super::storage::{DistCalculator, VectorStorage};
use arrow_array::types::Float32Type;
use lance_linalg::{distance::MetricType, MatrixView};

/// All data are stored in memory
pub struct InMemoryVectorStorage {
    vectors: Arc<MatrixView<Float32Type>>,
    metric_type: MetricType,
}

impl InMemoryVectorStorage {
    pub fn new(vectors: Arc<MatrixView<Float32Type>>, metric_type: MetricType) -> Self {
        Self {
            vectors,
            metric_type,
        }
    }

    pub fn vector(&self, id: u32) -> &[f32] {
        self.vectors.row(id as usize).unwrap()
    }

    /// Distance between two vectors.
    pub fn distance_between(&self, a: u32, b: u32) -> f32 {
        let vector1 = self.vectors.row(a as usize).unwrap();
        let vector2 = self.vectors.row(b as usize).unwrap();
        self.metric_type.func()(vector1, vector2)
    }
}

impl VectorStorage for InMemoryVectorStorage {
    fn len(&self) -> usize {
        self.vectors.num_rows()
    }

    fn metric_type(&self) -> MetricType {
        self.metric_type
    }

    fn dist_calculator(&self, query: &[f32]) -> Box<dyn DistCalculator> {
        Box::new(InMemoryDistanceCal {
            vectors: self.vectors.clone(),
            query: query.to_vec(),
            metric_type: self.metric_type,
        })
    }
}

struct InMemoryDistanceCal {
    vectors: Arc<MatrixView<Float32Type>>,
    query: Vec<f32>,
    metric_type: MetricType,
}

impl DistCalculator for InMemoryDistanceCal {
    fn distance(&self, ids: &[u32]) -> Vec<f32> {
        ids.iter()
            .map(|id| {
                let vector = self.vectors.row(*id as usize).unwrap();
                self.metric_type.func()(&self.query, vector)
            })
            .collect()
    }
}
