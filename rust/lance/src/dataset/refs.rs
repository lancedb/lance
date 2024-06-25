use std::ops::Range;

use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use serde::{Deserialize, Serialize};

use crate::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TagContents {
    pub version: u64,
    pub manifest_size: usize,
}

pub fn base_tags_path(base_path: &Path) -> Path {
    base_path.child("_refs").child("tags")
}

pub fn tag_path(base_path: &Path, tag: &str) -> Path {
    base_tags_path(base_path).child(format!("{}.json", tag))
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
