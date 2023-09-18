//! Data encodings
//!

use arrow_array::{Array, ArrayRef, UInt32Array};
use async_trait::async_trait;

pub mod binary;
pub mod dictionary;
pub mod plain;
pub mod rle;

use crate::error::Result;
use crate::format::pb;
use crate::io::ReadBatchParams;

/// Encoding enum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Encoding {
    /// Plain encoding.
    Plain,
    /// Binary encoding.
    VarBinary,
    /// Dictionary encoding.
    Dictionary,
    /// RLE encoding.
    RLE,
}

impl From<Encoding> for pb::Encoding {
    fn from(e: Encoding) -> Self {
        match e {
            Encoding::Plain => Self::Plain,
            Encoding::VarBinary => Self::VarBinary,
            Encoding::Dictionary => Self::Dictionary,
            Encoding::RLE => Self::Rle,
        }
    }
}

/// Encoder - Write an arrow array to the file.
#[async_trait]
pub trait Encoder {
    /// Write an slice of Arrays, and returns the file offset of the beginning of the batch.
    async fn encode(&mut self, array: &[&dyn Array]) -> Result<usize>;
}

/// Decoder - Read Arrow Data.
#[async_trait]
pub(crate) trait Decoder: Send + AsyncIndex<ReadBatchParams> {
    async fn decode(&self) -> Result<ArrayRef>;

    /// Take by indices.
    async fn take(&self, indices: &UInt32Array) -> Result<ArrayRef>;
}

#[async_trait]
pub trait AsyncIndex<IndexType> {
    type Output: Send + Sync;

    async fn get(&self, index: IndexType) -> Self::Output;
}
