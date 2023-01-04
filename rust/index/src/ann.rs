//! Approximate Nearest Neighbour Index.

use async_trait::async_trait;

pub mod flat;

pub enum AnnIndexType {
    /// Flat Index
    Flat,
    /// Invert index with product quantization.
    IvfPQ,
}

#[async_trait]
pub trait AnnIndex {
    fn ann_index_type() -> AnnIndexType;
    fn dim(&self) -> usize;
}
