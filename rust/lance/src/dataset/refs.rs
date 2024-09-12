// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;

use futures::future;
use futures::stream::{StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_io::object_store::ObjectStore;
use lance_table::io::commit::CommitHandler;
use object_store::path::Path;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{Error, Result};
use std::collections::HashMap;

/// Lance Ref
#[derive(Debug, Clone)]
pub enum Ref {
    Version(u64),
    Tag(String),
}

impl From<u64> for Ref {
    fn from(ref_: u64) -> Self {
        Self::Version(ref_)
    }
}

impl From<&str> for Ref {
    fn from(ref_: &str) -> Self {
        Self::Tag(ref_.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct Tags {
    object_store: Arc<ObjectStore>,
    commit_handler: Arc<dyn CommitHandler>,
    base: Path,
}

impl Tags {
    pub fn new(
        object_store: Arc<ObjectStore>,
        commit_handler: Arc<dyn CommitHandler>,
        base: Path,
    ) -> Self {
        Self {
            object_store,
            commit_handler,
            base,
        }
    }

    /// Get all tags.
    pub async fn list(&self) -> Result<HashMap<String, TagContents>> {
        let mut tags = HashMap::<String, TagContents>::new();

        let tag_files = self
            .object_store()
            .read_dir(base_tags_path(&self.base))
            .await?;

        let tag_names: Vec<String> = tag_files
            .iter()
            .filter_map(|name| name.strip_suffix(".json"))
            .map(|name| name.to_string())
            .collect_vec();

        futures::stream::iter(tag_names)
            .map(|tag_name| {
                let tag_file = tag_path(&self.base, &tag_name);
                async move {
                    let contents = TagContents::from_path(&tag_file, self.object_store()).await?;
                    Ok((tag_name, contents))
                }
            })
            .buffer_unordered(10)
            .try_for_each(|result| {
                let (tag_name, contents) = result;
                tags.insert(tag_name, contents);
                future::ready(Ok::<(), Error>(()))
            })
            .await?;

        Ok(tags)
    }

    pub async fn get_version(&self, tag: &str) -> Result<u64> {
        check_valid_ref(tag)?;

        let tag_file = tag_path(&self.base, tag);

        if !self.object_store().exists(&tag_file).await? {
            return Err(Error::RefNotFound {
                message: format!("tag {} does not exist", tag),
            });
        }

        let tag_contents = TagContents::from_path(&tag_file, self.object_store()).await?;

        Ok(tag_contents.version)
    }

    pub async fn create(&mut self, tag: &str, version: u64) -> Result<()> {
        check_valid_ref(tag)?;

        let tag_file = tag_path(&self.base, tag);

        if self.object_store().exists(&tag_file).await? {
            return Err(Error::RefConflict {
                message: format!("tag {} already exists", tag),
            });
        }

        let manifest_file = self
            .commit_handler
            .resolve_version(&self.base, version, &self.object_store.inner)
            .await?;

        if !self.object_store().exists(&manifest_file).await? {
            return Err(Error::VersionNotFound {
                message: format!("version {} does not exist", version),
            });
        }

        let tag_contents = TagContents {
            version,
            manifest_size: self.object_store().size(&manifest_file).await?,
        };

        self.object_store()
            .put(
                &tag_file,
                serde_json::to_string_pretty(&tag_contents)?.as_bytes(),
            )
            .await
    }

    pub async fn delete(&mut self, tag: &str) -> Result<()> {
        check_valid_ref(tag)?;

        let tag_file = tag_path(&self.base, tag);

        if !self.object_store().exists(&tag_file).await? {
            return Err(Error::RefNotFound {
                message: format!("tag {} does not exist", tag),
            });
        }

        self.object_store().delete(&tag_file).await
    }

    pub async fn update(&mut self, tag: &str, version: u64) -> Result<()> {
        check_valid_ref(tag)?;

        let tag_file = tag_path(&self.base, tag);

        if !self.object_store().exists(&tag_file).await? {
            return Err(Error::RefNotFound {
                message: format!("tag {} does not exist", tag),
            });
        }

        let manifest_file = self
            .commit_handler
            .resolve_version(&self.base, version, &self.object_store.inner)
            .await?;

        if !self.object_store().exists(&manifest_file).await? {
            return Err(Error::VersionNotFound {
                message: format!("version {} does not exist", version),
            });
        }

        let tag_contents = TagContents {
            version,
            manifest_size: self.object_store().size(&manifest_file).await?,
        };

        self.object_store()
            .put(
                &tag_file,
                serde_json::to_string_pretty(&tag_contents)?.as_bytes(),
            )
            .await
    }

    pub(crate) fn object_store(&self) -> &ObjectStore {
        &self.object_store
    }
}

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
            "~/ref",
            "ref.",
            "ref..ref"
        )]
        r: &str,
    ) {
        assert_contains!(
            check_valid_ref(r).err().unwrap().to_string(),
            "Ref is invalid: Ref"
        );
    }
}
