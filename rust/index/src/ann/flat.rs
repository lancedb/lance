//! Flat index
use std::io::Result;
use crate::ann::{AnnIndex, AnnIndexType};
use crate::Index;

/// Flat index
pub struct FlatIndex {
    columns: Vec<String>,
}

impl FlatIndex {
    /// Create an empty index.
    pub fn new() -> Result<Self> {
        Ok(Self {
            columns: vec![],
        })
    }

    /// Build index
    pub fn build() -> Result<()> {
        Ok(())
    }
}

impl Index for FlatIndex {
    fn prefetch(&mut self) -> Result<()> {
        Ok(())
    }

    fn columns(&self) -> &[String] {
        self.columns.as_slice()
    }
}

impl AnnIndex for FlatIndex {
    fn ann_index_type() -> AnnIndexType {
        todo!()
    }

    fn dim(&self) -> usize {
        todo!()
    }
}