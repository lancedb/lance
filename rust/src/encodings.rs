use std::io::Result;

use arrow_array::Array;

pub mod binary;
pub mod plain;
pub mod rle;
use crate::format::pb;

#[derive(Debug)]
pub enum Encoding {
    Plain,
    VarBinary,
    Dictionary,
    RLE,
}

impl From<Encoding> for pb::Encoding {
    fn from(e: Encoding) -> Self {
        match e {
            Encoding::Plain => pb::Encoding::Plain,
            Encoding::VarBinary => pb::Encoding::VarBinary,
            Encoding::Dictionary => pb::Encoding::Dictionary,
            Encoding::RLE => pb::Encoding::Rle,
        }
    }
}

/// Encoder - Write an arrow array to the file.
pub trait Encoder {
    /// Write an array, and returns the file offset of the beginning of the batch.
    fn write(&mut self, array: &dyn Array) -> Result<i64>;
}
