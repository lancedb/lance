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

use lance_linalg::distance::MetricType;

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
pub trait VectorStorage: Send + Sync {
    fn len(&self) -> usize;

    /// Returns true if this graph is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the metric type of the vectors.
    fn metric_type(&self) -> MetricType;

    /// Create a [DistCalculator] to compute the distance between the query.
    ///
    /// Using dist calcualtor can be more efficient as it can pre-compute some
    /// values.
    fn dist_calculator(&self, query: &[f32]) -> Box<dyn DistCalculator>;
}
