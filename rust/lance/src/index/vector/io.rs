use std::sync::Arc;

use object_store::path::Path;
use prost::Message;

use crate::{
    io::{
        object_reader::{read_message, ObjectReader},
        read_message_from_buf, read_metadata_offset,
    },
    session::Session,
    Result,
};

use super::{IndexShardLoader, VectorIndex};

/// A helper with some common I/O operations that indexes need to perform
pub struct IndexReader {
    reader: Arc<dyn ObjectReader>,
    session: Option<Arc<Session>>,
    block_size: usize,
}

impl IndexReader {
    pub fn new(reader: Arc<dyn ObjectReader>, session: Arc<Session>, block_size: usize) -> Self {
        Self {
            reader,
            session: Some(session),
            block_size,
        }
    }

    pub fn path(&self) -> &Path {
        self.reader.path()
    }

    /// Loads a sub index by first checking the session cache
    pub async fn load_sub_index(
        &self,
        loader: &dyn IndexShardLoader,
        key: &str,
        shard_index: usize,
    ) -> Result<Arc<dyn VectorIndex>> {
        if let Some(session) = &self.session {
            if let Some(index) = session.index_cache.get(key) {
                return Ok(index);
            }
        }
        let index: Arc<dyn VectorIndex> =
            loader.load(self.reader.as_ref(), shard_index).await?.into();
        if let Some(session) = &self.session {
            session.index_cache.insert(key, index.clone());
        }
        Ok(index)
    }

    /// Reads a protobuf message from the end of a file
    pub async fn read_tail_proto<M: Message + Default>(&self) -> Result<M> {
        let file_size = self.reader.size().await?;
        let begin = if file_size < self.block_size {
            0
        } else {
            file_size - self.block_size
        };
        let tail_bytes = self.reader.get_range(begin..file_size).await?;
        let metadata_pos = read_metadata_offset(&tail_bytes)?;
        if metadata_pos < file_size - tail_bytes.len() {
            // We have not read the metadata bytes yet.
            read_message(self.reader.as_ref(), metadata_pos).await
        } else {
            let offset = tail_bytes.len() - (file_size - metadata_pos);
            Ok(read_message_from_buf(&tail_bytes.slice(offset..))?)
        }
    }
}
