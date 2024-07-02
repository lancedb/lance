use std::ops::Range;

use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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

pub fn check_valid_ref(s: &str) -> Result<()> {
    if s.is_empty() {
        return Err(Error::InvalidRef {
            message: "Ref cannot be empty".to_string(),
        });
    }

    if !s
        .chars()
        .all(|c| c.is_alphanumeric() || c == '.' || c == '-' || c == '_')
    {
        return Err(Error::InvalidRef {
            message: "Ref characters must be either alphanumeric, '.', '-' or '_'".to_string(),
        });
    }

    if s.starts_with('.') {
        return Err(Error::InvalidRef {
            message: "Ref cannot begin with a dot".to_string(),
        });
    }

    if s.ends_with('.') {
        return Err(Error::InvalidRef {
            message: "Ref cannot end with a dot".to_string(),
        });
    }

    if s.ends_with(".lock") {
        return Err(Error::InvalidRef {
            message: "Ref cannot end with .lock".to_string(),
        });
    }

    if s.contains("..") {
        return Err(Error::InvalidRef {
            message: "Ref cannot have two consecutive dots".to_string(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::common::assert_contains;

    use rstest::rstest;

    #[rstest]
    fn test_ok_ref(
        #[values(
            "ref",
            "ref-with-dashes",
            "ref.extension",
            "ref_with_underscores",
            "v1.2.3-rc4"
        )]
        r: &str,
    ) {
        check_valid_ref(r).unwrap();
    }

    #[rstest]
    fn test_err_ref(
        #[values(
            "",
            "../ref",
            ".ref",
            "/ref",
            "@",
            "deeply/nested/ref",
            "nested//ref",
            "nested/ref",
            "nested\\ref",
            "ref*",
            "ref.lock",
            "ref/",
            "ref?",
            "ref@{ref",
            "ref[",
            "ref^",
            "~/ref"
        )]
        r: &str,
    ) {
        assert_contains!(
            check_valid_ref(r).err().unwrap().to_string(),
            "Ref is invalid: Ref"
        );
    }
}
