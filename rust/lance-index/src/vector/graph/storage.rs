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

use lance_arrow::ArrowFloatType;
use num_traits::{Float, ToPrimitive};
use std::sync::Arc;

use lance_linalg::{
    distance::{Cosine, Dot, MetricType, L2},
    MatrixView,
};

pub trait DistCalculator {
    fn distance(&self, ids: &[u32]) -> Vec<f32>;
}

/// Vector Storage is the abstraction to store the vectors.
///
/// It can be in-memory raw vectors or on disk PQ code.
///
/// It abstracts away the logic to compute the distance between vectors.
///
/// TODO: should we rename this to "VectorDistance"?;
pub trait VectorStorage<T: Float> {
    fn len(&self) -> usize;

    /// Create a [DistCalculator] to compute the distance between the query.
    ///
    /// Using dist calcualtor can be more efficient as it can pre-compute some
    /// values.
    fn dist_calculator(&self, query: &[f32], metric_type: MetricType) -> Box<dyn DistCalculator>;
}

struct InMemoryDistanceCal<T: ArrowFloatType + L2 + Cosine + Dot> {
    vectors: Arc<MatrixView<T>>,
    query: Vec<f32>,
    metric_type: MetricType,
}

impl<T: ArrowFloatType + L2 + Cosine + Dot> DistCalculator for InMemoryDistanceCal<T> {
    fn distance(&self, ids: &[u32]) -> Vec<f32> {
        ids.iter()
            .map(|id| {
                let vector = self
                    .vectors
                    .row(*id as usize)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_f32().unwrap())
                    .collect::<Vec<_>>();
                self.metric_type.func()(&self.query, &vector)
            })
            .collect()
    }
}

impl<T: ArrowFloatType + L2 + Cosine + Dot + 'static> VectorStorage<T::Native> for MatrixView<T> {
    fn len(&self) -> usize {
        self.num_rows()
    }

    fn dist_calculator(&self, query: &[f32], metric_type: MetricType) -> Box<dyn DistCalculator> {
        Box::new(InMemoryDistanceCal {
            vectors: Arc::new(self.clone()),
            query: query.to_vec(),
            metric_type,
        })
    }
}
