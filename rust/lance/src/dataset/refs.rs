// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;

use futures::stream::{StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_io::object_store::ObjectStore;
use lance_table::io::commit::CommitHandler;
use object_store::path::Path;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::dataset::refs::Ref::{Branch, Tag, Version};
use crate::{Error, Result};
use serde::de::DeserializeOwned;
use snafu::location;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Formatter;

/// Lance Ref
#[derive(Debug, Clone)]
pub enum Ref {
    // This is a version number of Main branch
    Version(u64),
    // 0 is the optional branch name, if null it is the main branch
    // 1 is the tag name
    Tag(String),
    // 0 is the branch name
    // 1 is the version number
    Branch(String, u64),
}

impl From<u64> for Ref {
    fn from(ref_: u64) -> Self {
        Version(ref_)
    }
}

impl From<&str> for Ref {
    fn from(ref_: &str) -> Self {
        Tag(ref_.to_string())
    }
}

impl From<(&str, u64)> for Ref {
    fn from(_ref: (&str, u64)) -> Self {
        Branch(_ref.0.to_string(), _ref.1)
    }
}

impl fmt::Display for Ref {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Version(version_number) => write!(f, "Main:{}", version_number),
            Tag(tag_name) => write!(f, "{}", tag_name),
            Branch(branch, ref_) => write!(f, "{}:{}", branch, ref_),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Refs {
    object_store: Arc<ObjectStore>,
    commit_handler: Arc<dyn CommitHandler>,
    base: Path,
}

impl Refs {
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

    async fn fetch_tags(&self) -> Result<Vec<(String, TagContents)>> {
        let base_path = base_tags_path(&self.base);
        let tag_files = self.object_store().read_dir(base_path).await?;

        let tag_names: Vec<String> = tag_files
            .iter()
            .filter_map(|name| name.strip_suffix(".json"))
            .map(|name| name.to_string())
            .collect_vec();

        futures::stream::iter(tag_names)
            .map(|tag_name| async move {
                let contents =
                    TagContents::from_path(&tag_path(&self.base, &tag_name), self.object_store())
                        .await?;
                Ok((tag_name, contents))
            })
            .buffer_unordered(10)
            .try_collect()
            .await
    }

    async fn fetch_branches(&self) -> Result<Vec<(String, BranchContents)>> {
        let base_path = base_branches_path(&self.base);
        let branch_files = self.object_store().read_dir(base_path).await?;

        let branch_names: Vec<String> = branch_files
            .iter()
            .filter_map(|name| name.strip_suffix(".json"))
            .map(|name| name.to_string())
            .collect_vec();

        futures::stream::iter(branch_names)
            .map(|tag_name| async move {
                let contents =
                    BranchContents::from_path(&tag_path(&self.base, &tag_name), self.object_store())
                        .await?;
                Ok((tag_name, contents))
            })
            .buffer_unordered(10)
            .try_collect()
            .await
    }

    pub(crate) fn sort_tags(tags: &mut [(String, TagContents)], order: Option<std::cmp::Ordering>) {
        tags.sort_by(|a, b| {
            let desired_ordering = order.unwrap_or(Ordering::Greater);
            let version_ordering = a.1.version.cmp(&b.1.version);
            let version_result = match desired_ordering {
                Ordering::Less => version_ordering,
                _ => version_ordering.reverse(),
            };
            version_result.then_with(|| a.0.cmp(&b.0))
        });
    }

    pub async fn list_tags(&self) -> Result<HashMap<String, TagContents>> {
        self.fetch_tags()
            .await
            .map(|tags| tags.into_iter().collect())
    }

    pub async fn list_tags_ordered(
        &self,
        order: Option<std::cmp::Ordering>,
    ) -> Result<Vec<(String, TagContents)>> {
        let mut tags = self.fetch_tags().await?;
        Self::sort_tags(&mut tags, order);
        Ok(tags)
    }

    pub async fn get_tag(&self, tag: &str) -> Result<TagContents> {
        check_valid_ref(tag)?;

        let tag_file = tag_path(&self.base, tag);

        if !self.object_store().exists(&tag_file).await? {
            return Err(Error::RefNotFound {
                message: format!("tag {} does not exist", tag),
            });
        }

        let tag_contents = TagContents::from_path(&tag_file, self.object_store()).await?;

        Ok(tag_contents)
    }

    pub async fn create_tag(&mut self, tag: &str, version: u64, source_branch: Option<String>) -> Result<()> {
        check_valid_ref(tag)?;

        let tag_file = tag_path(&self.base, tag);

        if self.object_store().exists(&tag_file).await? {
            return Err(Error::RefConflict {
                message: format!("tag {} already exists", tag),
            });
        }

        let manifest_file = self
            .commit_handler
            .resolve_version_location(&self.base, version, &self.object_store.inner)
            .await?;

        if !self.object_store().exists(&manifest_file.path).await? {
            return Err(Error::VersionNotFound {
                message: format!("version {} does not exist", version),
            });
        }

        let manifest_size = if let Some(size) = manifest_file.size {
            size as usize
        } else {
            self.object_store().size(&manifest_file.path).await? as usize
        };

        let tag_contents = TagContents {
            branch: source_branch,
            version,
            manifest_size,
        };

        self.object_store()
            .put(
                &tag_file,
                serde_json::to_string_pretty(&tag_contents)?.as_bytes(),
            )
            .await
            .map(|_| ())
    }

    pub async fn delete_tag(&mut self, tag: &str) -> Result<()> {
        check_valid_ref(tag)?;

        let tag_file = tag_path(&self.base, tag);

        if !self.object_store().exists(&tag_file).await? {
            return Err(Error::RefNotFound {
                message: format!("tag {} does not exist", tag),
            });
        }

        self.object_store().delete(&tag_file).await
    }

    pub async fn update_tag(&mut self, tag: &str, version: u64, source_branch: Option<String>) -> Result<()> {
        check_valid_ref(tag)?;

        let tag_file = tag_path(&self.base, tag);

        if !self.object_store().exists(&tag_file).await? {
            return Err(Error::RefNotFound {
                message: format!("tag {} does not exist", tag),
            });
        }

        let manifest_file = self
            .commit_handler
            .resolve_version_location(&self.base, version, &self.object_store.inner)
            .await?;

        if !self.object_store().exists(&manifest_file.path).await? {
            return Err(Error::VersionNotFound {
                message: format!("version {} does not exist", version),
            });
        }

        let manifest_size = if let Some(size) = manifest_file.size {
            size as usize
        } else {
            self.object_store().size(&manifest_file.path).await? as usize
        };

        let tag_contents = TagContents {
            branch: source_branch,
            version,
            manifest_size,
        };

        self.object_store()
            .put(
                &tag_file,
                serde_json::to_string_pretty(&tag_contents)?.as_bytes(),
            )
            .await
            .map(|_| ())
    }

    pub async fn list_branches(&self) -> Result<Vec<BranchContents>> {
        self.fetch_branches()
            .await
            .map(|branches| branches.into_iter().map(|(_, branch)| branch).collect())
    }

    pub async fn create_branch(&mut self, branch_name: &str, version: impl Into<Ref>) -> Result<()> {
        check_valid_ref(branch_name)?;

        let ref_: Ref = version.into();
        let (parent_branch, version_number) = match ref_ {
            Version(version_number) => (None, version_number),
            Tag(tag_name) => {
                let tag = self.get_tag(tag_name.as_str()).await?;
                (tag.branch, tag.version)
            },
            Branch(parent_branch, version_number) => (Some(parent_branch), version_number),
        };

        let branch_file = branch_path(&self.base, branch_name);

        if self.object_store().exists(&branch_file).await? {
            return Err(Error::RefConflict {
                message: format!("branch {} already exists", branch_name),
            });
        }

        let base_path = if let Some(branch) = &parent_branch {
            branch_path(&self.base, &branch.as_str())
        } else {
            self.base.clone()
        };
        // Verify the source version exists
        let manifest_file = self
            .commit_handler
            .resolve_version_location(&base_path, version_number, &self.object_store.inner)
            .await?;

        if !self.object_store().exists(&manifest_file.path).await? {
            return Err(Error::VersionNotFound {
                message: format!("Manifest file {} does not exist", &manifest_file.path),
            });
        };

        let branch_contents = BranchContents {
            parent_branch,
            parent_version: version_number,
            create_at: chrono::Utc::now().timestamp() as u64,
            manifest_size: if let Some(size) = manifest_file.size {
                size as usize
            } else {
                self.object_store().size(&manifest_file.path).await? as usize
            },
        };

        self.object_store()
            .put(
                &branch_file,
                serde_json::to_string_pretty(&branch_contents)?.as_bytes(),
            )
            .await
            .map(|_| ())
    }

    pub async fn delete_branch(&mut self, branch: &str) -> Result<()> {
        check_valid_ref(branch)?;

        let branch_file = branch_path(&self.base, branch);

        if !self.object_store().exists(&branch_file).await? {
            return Err(Error::RefNotFound {
                message: format!("branch {} does not exist", branch),
            });
        }

        self.object_store().delete(&branch_file).await
    }

    pub async fn get_branch(&self, branch: &str) -> Result<BranchContents> {
        check_valid_ref(branch)?;

        let branch_file = branch_path(&self.base, branch);

        if !self.object_store().exists(&branch_file).await? {
            return Err(Error::RefNotFound {
                message: format!("branch {} does not exist", branch),
            });
        }

        let branch_contents = BranchContents::from_path(&branch_file, self.object_store()).await?;

        Ok(branch_contents)
    }

    pub(crate) fn object_store(&self) -> &ObjectStore {
        &self.object_store
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TagContents {
    pub branch: Option<String>,
    pub version: u64,
    pub manifest_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchContents {
    pub parent_branch: Option<String>,
    pub parent_version: u64,
    pub create_at: u64, // unix timestamp
    pub manifest_size: usize,
}

pub fn base_tags_path(base_path: &Path) -> Path {
    base_path.child("_refs").child("tags")
}

pub fn base_branches_path(base_path: &Path) -> Path {
    base_path.child("_refs").child("branches")
}

pub fn tag_path(base_path: &Path, branch: &str) -> Path {
    base_tags_path(base_path).child(format!("{}.json", branch))
}

pub fn branch_path(base_path: &Path, branch: &str) -> Path {
    base_branches_path(base_path).child(format!("{}.json", branch))
}

async fn from_path<T>(path: &Path, object_store: &ObjectStore) -> Result<T>
where
    T: DeserializeOwned,
{
    let tag_reader = object_store.open(path).await?;
    let tag_bytes = tag_reader
        .get_range(Range {
            start: 0,
            end: tag_reader.size().await?,
        })
        .await?;
    let json_str = String::from_utf8(tag_bytes.to_vec()).map_err(|e| {
        Error::corrupt_file(path.clone(), e.to_string(), location!())
    })?;
    Ok(serde_json::from_str(&json_str)?)
}

impl TagContents {
    pub async fn from_path(path: &Path, object_store: &ObjectStore) -> Result<Self> {
        from_path(path, object_store).await
    }
}

impl BranchContents {
    pub async fn from_path(path: &Path, object_store: &ObjectStore) -> Result<Self> {
        from_path(path, object_store).await
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
