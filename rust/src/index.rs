use std::io::Result;

use arrow_array::RecordBatch;
use async_trait::async_trait;

pub mod ann;

#[allow(clippy::all)]
pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/lance_index.pb.rs"));
}

/// Generic index traits
#[async_trait]
pub trait Index {
    /// Indexed columns
    fn columns(&self) -> &[String];

    /// Build index from a batch reader
    async fn build(&self, _reader: &RecordBatch) -> Result<()>;
}
