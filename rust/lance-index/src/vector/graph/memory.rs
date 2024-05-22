// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! In-memory graph representations.

use super::storage::{DistCalculator, VectorStore};
use arrow::array::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::ArrayRef;
use lance_linalg::{distance::DistanceType, MatrixView};

/// All data are stored in memory
pub struct InMemoryVectorStorage {
    row_ids: Vec<u64>,
    vectors: MatrixView<Float32Type>,
    distance_type: DistanceType,
}

impl InMemoryVectorStorage {
    pub fn new(vectors: MatrixView<Float32Type>, distance_type: DistanceType) -> Self {
        let row_ids = (0..vectors.num_rows() as u64).collect();
        Self {
            row_ids,
            vectors,
            distance_type,
        }
    }

    pub fn vector(&self, id: u32) -> ArrayRef {
        self.vectors.row(id as usize).unwrap()
    }
}

impl VectorStore for InMemoryVectorStorage {
    type DistanceCalculator<'a> = InMemoryDistanceCal;
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn len(&self) -> usize {
        self.vectors.num_rows()
    }

    fn row_ids(&self) -> &[u64] {
        &self.row_ids
    }

    fn metric_type(&self) -> DistanceType {
        self.distance_type
    }

    fn dist_calculator(&self, query: ArrayRef) -> Self::DistanceCalculator<'_> {
        InMemoryDistanceCal {
            vectors: self.vectors.clone(),
            query,
            distance_type: self.distance_type,
        }
    }

    fn dist_calculator_from_id(&self, id: u32) -> Self::DistanceCalculator<'_> {
        InMemoryDistanceCal {
            vectors: self.vectors.clone(),
            query: self.vectors.row(id as usize).unwrap(),
            distance_type: self.distance_type,
        }
    }

    /// Distance between two vectors.
    fn distance_between(&self, a: u32, b: u32) -> f32 {
        let vector1 = self.vectors.row_ref(a as usize).unwrap();
        let vector2 = self.vectors.row_ref(b as usize).unwrap();
        self.distance_type.func()(vector1, vector2)
    }
}

pub struct InMemoryDistanceCal {
    vectors: MatrixView<Float32Type>,
    query: ArrayRef,
    distance_type: DistanceType,
}

impl DistCalculator for InMemoryDistanceCal {
    #[inline]
    fn distance(&self, id: u32) -> f32 {
        let vector = self.vectors.row_ref(id as usize).unwrap();
        self.distance_type.func()(self.query.as_primitive::<Float32Type>().values(), vector)
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
