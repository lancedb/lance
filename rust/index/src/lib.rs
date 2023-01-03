use arrow_array::RecordBatchReader;
use std::io::Result;

pub mod ann;

pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/lance_index.pb.rs"));
}

#[cfg(test)]
pub mod tests;

/// Generic index traits
pub trait Index {
    /// Indexed columns
    fn columns(&self) -> &[String];

    /// Build index from a batch reader
    fn build(&self, batch_reader: &impl RecordBatchReader) -> Result<()>;
}
