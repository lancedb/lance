// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::any::Any;

use arrow_array::ArrayRef;
use lance_linalg::distance::MetricType;

pub trait DistCalculator {
    fn distance(&self, id: u32) -> f32;
}

/// Vector Storage is the abstraction to store the vectors.
///
/// It can be in-memory raw vectors or on disk PQ code.
///
/// It abstracts away the logic to compute the distance between vectors.
///
/// TODO: should we rename this to "VectorDistance"?;
pub trait VectorStorage: Send + Sync {
    fn as_any(&self) -> &dyn Any;

    fn len(&self) -> usize;

    /// Returns true if this graph is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn row_ids(&self) -> &[u64];

    /// Return the metric type of the vectors.
    fn metric_type(&self) -> MetricType;

    /// Create a [DistCalculator] to compute the distance between the query.
    ///
    /// Using dist calcualtor can be more efficient as it can pre-compute some
    /// values.
    fn dist_calculator(&self, query: ArrayRef) -> Box<dyn DistCalculator>;
}
