//! Approximate Nearest Neighbour Index.

use arrow_array::ArrowNumericType;
use arrow_array::FixedSizeListArray;
use arrow_array::Float32Array;
use arrow_array::PrimitiveArray;
use crate::Index;

pub mod flat;

pub enum AnnIndexType {
    /// Flat Index
    Flat,
    /// Invert index with product quantiazation
    IvfPQ,
}

pub trait AnnIndex {
    fn ann_index_type() -> AnnIndexType;
    fn dim(&self) -> usize;
}
