//! Approximate Nearest Neighbour Index.

pub mod flat;

pub enum AnnIndexType {
    /// Flat Index
    Flat,
    /// Invert index with product quantization.
    IvfPQ,
}

pub trait AnnIndex {
    fn ann_index_type() -> AnnIndexType;
    fn dim(&self) -> usize;
}
