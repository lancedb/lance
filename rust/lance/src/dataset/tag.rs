use crate::{Error, Result};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use serde::{Deserialize, Serialize};
use std::ops::Range;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagContents {
    pub version: u64,
    pub manifest_size: usize,
}

impl TagContents {
    pub async fn from_path(path: &Path, object_store: &ObjectStore) -> Result<Self> {
        let tag_reader = object_store.open(path).await?;
        let tag_bytes = tag_reader
            .get_range(Range {
                start: 0,
                end: tag_reader.size().await?,
            })
            .await?;
        Ok(serde_json::from_str(
            String::from_utf8(tag_bytes.to_vec()).unwrap().as_str(),
        )?)
    }
}
