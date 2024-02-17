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

use lance_linalg::{
    distance::{Cosine, Dot, MetricType, L2},
    MatrixView,
};

pub trait DistCalculator {
    fn distance(&self, ids: &[u32]) -> Vec<f32>;
}

pub trait VectorStorage<T: Float> {
    fn len(&self) -> usize;

    /// Create a distance calculator from query.
    ///
    fn dist_calculator(&self, query: &[f32], metric_type: MetricType) -> Box<dyn DistCalculator>;
}

struct InMemoryDistanceCal<'a, T: ArrowFloatType> {
    vectors: &'a MatrixView<T>,
    query: &'a [f32],
    metric_type: MetricType,
}

impl<'a, T: ArrowFloatType + L2 + Cosine + Dot> DistCalculator for InMemoryDistanceCal<'_, T> {
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
                self.metric_type.func()(self.query, &vector)
            })
            .collect()
    }
}

impl<T: ArrowFloatType + L2 + Cosine + Dot> VectorStorage<T::Native> for MatrixView<T> {
    fn len(&self) -> usize {
        self.num_rows()
    }

    fn dist_calculator(&self, query: &[f32], metric_type: MetricType) -> Box<dyn DistCalculator> {
        Box::new(InMemoryDistanceCal {
            vectors: self,
            query,
            metric_type,
        })
    }
}
