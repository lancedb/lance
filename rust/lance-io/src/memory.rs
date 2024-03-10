use std::ops::Range;

use async_trait::async_trait;
use bytes::Bytes;
use lance_core::Result;
use object_store::path::Path;

use crate::traits::Reader;

/// A memory buffer reader
///
pub struct MemoryBufReader {
    buf: Bytes,
    block_size: usize,
    path: Path,
}

impl MemoryBufReader {
    /// Create a new memory buffer reader
    pub fn new(buf: Bytes, block_size: usize, path: Path) -> Self {
        Self {
            buf,
            block_size,
            path,
        }
    }
}

#[async_trait]
impl Reader for MemoryBufReader {
    fn path(&self) -> &Path {
        &self.path
    }

    /// Suggest optimal I/O size per storage device.
    fn block_size(&self) -> usize {
        self.block_size
    }

    /// Object/File Size.
    async fn size(&self) -> Result<usize> {
        Ok(self.buf.len())
    }

    /// Read a range of bytes from the object.
    ///
    /// TODO: change to read_at()?
    async fn get_range(&self, range: Range<usize>) -> Result<Bytes> {
        Ok(self.buf.slice(range))
    }
}
