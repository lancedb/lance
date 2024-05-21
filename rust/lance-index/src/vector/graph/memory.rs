// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! In-memory graph representations.

use std::sync::Arc;

use super::storage::{DistCalculator, VectorStorage};
use arrow::array::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::ArrayRef;
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

    pub fn vector(&self, id: u32) -> ArrayRef {
        self.vectors.row(id as usize).unwrap()
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

    fn dist_calculator(&self, query: ArrayRef) -> Box<dyn DistCalculator> {
        Box::new(InMemoryDistanceCal {
            vectors: self.vectors.clone(),
            query,
            metric_type: self.metric_type,
        })
    }

    fn dist_calculator_from_id(&self, id: u32) -> Box<dyn DistCalculator> {
        Box::new(InMemoryDistanceCal {
            vectors: self.vectors.clone(),
            query: self.vectors.row(id as usize).unwrap(),
            metric_type: self.metric_type,
        })
    }

    /// Distance between two vectors.
    fn distance_between(&self, a: u32, b: u32) -> f32 {
        let vector1 = self.vectors.row_ref(a as usize).unwrap();
        let vector2 = self.vectors.row_ref(b as usize).unwrap();
        self.metric_type.func()(vector1, vector2)
    }
}

struct InMemoryDistanceCal {
    vectors: Arc<MatrixView<Float32Type>>,
    query: ArrayRef,
    metric_type: MetricType,
}

impl DistCalculator<'_> for InMemoryDistanceCal {
    #[inline]
    fn distance(&self, id: u32) -> f32 {
        let vector = self.vectors.row_ref(id as usize).unwrap();
        self.metric_type.func()(self.query.as_primitive::<Float32Type>().values(), vector)
    }
    fn prefetch(&self, id: u32) {
        // TODO use rust intrinsics instead of x86 intrinsics
        // TODO finish this
        unsafe {
            let vector = self.vectors.row_ref(id as usize).unwrap();
            let ptr = vector.as_ptr() as *const i8;
            let end_ptr = vector.as_ptr().add(vector.len()) as *const i8;

            let mut current_ptr = ptr;
            while current_ptr < end_ptr {
                const CACHE_LINE_SIZE: usize = 64;
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                    _mm_prefetch(current_ptr, _MM_HINT_T0);
                }
                current_ptr = current_ptr.add(CACHE_LINE_SIZE);
            }
        }
    }
}
