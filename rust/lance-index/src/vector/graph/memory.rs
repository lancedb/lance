// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! In-memory graph representations.

use std::sync::Arc;

use super::storage::{DistCalculator, VectorStorage};
use arrow_array::types::Float32Type;
use lance_linalg::{distance::MetricType, MatrixView};

/// All data are stored in memory
pub struct InMemoryVectorStorage {
    row_ids: Vec<u64>,
    vectors: Arc<MatrixView<Float32Type>>,
    metric_type: MetricType,
}

impl InMemoryVectorStorage {
    pub fn new(vectors: Arc<MatrixView<Float32Type>>, metric_type: MetricType) -> Self {
        let row_ids = (0..vectors.num_rows() as u64).collect();
        Self {
            row_ids,
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
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn len(&self) -> usize {
        self.vectors.num_rows()
    }

    fn row_ids(&self) -> &[u64] {
        &self.row_ids
    }

    fn metric_type(&self) -> MetricType {
        self.metric_type
    }

    fn dist_calculator(&self, query: &[f32]) -> Box<dyn DistCalculator> {
        Box::new(InMemoryDistanceCal {
            vectors: self.vectors.clone(),
            query: query.as_ptr(),
            metric_type: self.metric_type,
        })
    }
}

struct InMemoryDistanceCal {
    vectors: Arc<MatrixView<Float32Type>>,
    query: *const f32,
    metric_type: MetricType,
}

impl DistCalculator for InMemoryDistanceCal {
    #[inline]
    fn distance(&self, id: u32) -> f32 {
        let vector = self.vectors.row(id as usize).unwrap();
        self.metric_type.func()(
            unsafe { std::slice::from_raw_parts(self.query, vector.len()) },
            vector,
        )
    }
}
