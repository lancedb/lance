use arrow_array::RecordBatch;
use async_trait::async_trait;
use std::io::Result;

pub mod ann;
pub mod io;

#[allow(clippy::all)]
pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/lance_index.pb.rs"));
}

#[cfg(test)]
pub mod tests;

/// Generic index traits
#[async_trait]
pub trait Index {
    /// Indexed columns
    fn columns(&self) -> &[String];

    /// Build index from a batch reader
    async fn build(&self, _reader: &RecordBatch) -> Result<()>;
}
