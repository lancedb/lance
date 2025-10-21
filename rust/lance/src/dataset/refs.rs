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

use crate::dataset::branch_location::BranchLocation;
use crate::dataset::refs::Ref::{Tag, Version};
use crate::{Error, Result};
use serde::de::DeserializeOwned;
use snafu::location;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Formatter;
use std::io::ErrorKind;

/// Lance Ref
#[derive(Debug, Clone)]
pub enum Ref {
    // This is a global version identifier present as (branch_name, version_number)
    // if branch_name is None, it points to the main branch
    // if version_number is None, it points to the latest version
    Version(Option<String>, Option<u64>),
    // Tag name points to the global version identifier, could be considered as an alias of specific global version
    Tag(String),
}

impl From<u64> for Ref {
    fn from(ref_: u64) -> Self {
        Version(None, Some(ref_))
    }
}

impl From<&str> for Ref {
    fn from(ref_: &str) -> Self {
        Tag(ref_.to_string())
    }
}

impl From<(&str, u64)> for Ref {
    fn from(_ref: (&str, u64)) -> Self {
        Version(Some(_ref.0.to_string()), Some(_ref.1))
    }
}

impl From<(Option<String>, Option<u64>)> for Ref {
    fn from(_ref: (Option<String>, Option<u64>)) -> Self {
        Version(_ref.0, _ref.1)
    }
}

impl From<(&str, Option<u64>)> for Ref {
    fn from(_ref: (&str, Option<u64>)) -> Self {
        Version(Some(_ref.0.to_string()), _ref.1)
    }
}

impl fmt::Display for Ref {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Version(branch, version_number) => {
                let branch_name = branch.as_deref().unwrap_or("main");
                let version_str = version_number
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "latest".to_string());
                write!(f, "{}:{}", branch_name, version_str)
            }
            Tag(tag_name) => write!(f, "{}", tag_name),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Refs {
    pub(crate) object_store: Arc<ObjectStore>,
    pub(crate) commit_handler: Arc<dyn CommitHandler>,
    pub(crate) base_location: BranchLocation,
}

impl Refs {
    pub fn new(
        object_store: Arc<ObjectStore>,
        commit_handler: Arc<dyn CommitHandler>,
        base_location: BranchLocation,
    ) -> Self {
        Self {
            object_store,
            commit_handler,
            base_location,
        }
    }

    pub fn tags(&self) -> Tags<'_> {
        Tags { refs: self }
    }

    pub fn branches(&self) -> Branches<'_> {
        Branches { refs: self }
    }

    pub fn base(&self) -> &Path {
        &self.base_location.path
    }

    pub fn root(&self) -> Result<BranchLocation> {
        self.base_location.find_main()
    }
}

/// Tags operation
#[derive(Debug, Clone)]
pub struct Tags<'a> {
    refs: &'a Refs,
}

/// Branches operation
#[derive(Debug, Clone)]
pub struct Branches<'a> {
    refs: &'a Refs,
}

impl Tags<'_> {
    fn object_store(&self) -> &ObjectStore {
        &self.refs.object_store
    }
}

impl Branches<'_> {
    fn object_store(&self) -> &ObjectStore {
        &self.refs.object_store
    }
}

impl Tags<'_> {
    pub async fn fetch_tags(&self) -> Result<Vec<(String, TagContents)>> {
        let root_location = self.refs.root()?;
        let base_path = base_tags_path(&root_location.path);
        let tag_files = self.object_store().read_dir(base_path).await?;

        let tag_names: Vec<String> = tag_files
            .iter()
            .filter_map(|name| name.strip_suffix(".json"))
            .map(|name| name.to_string())
            .collect_vec();

        let root_path = &root_location.path;
        futures::stream::iter(tag_names)
            .map(|tag_name| async move {
                let contents =
                    TagContents::from_path(&tag_path(root_path, &tag_name), self.object_store())
                        .await?;
                Ok((tag_name, contents))
            })
            .buffer_unordered(10)
            .try_collect()
            .await
    }

    pub async fn list(&self) -> Result<HashMap<String, TagContents>> {
        self.fetch_tags()
            .await
            .map(|tags| tags.into_iter().collect())
    }

    pub async fn list_tags_ordered(
        &self,
        order: Option<Ordering>,
    ) -> Result<Vec<(String, TagContents)>> {
        let mut tags = self.fetch_tags().await?;
        tags.sort_by(|a, b| {
            let desired_ordering = order.unwrap_or(Ordering::Greater);
            let version_ordering = a.1.version.cmp(&b.1.version);
            let version_result = match desired_ordering {
                Ordering::Less => version_ordering,
                _ => version_ordering.reverse(),
            };
            version_result.then_with(|| a.0.cmp(&b.0))
        });
        Ok(tags)
    }

    pub async fn get_version(&self, tag: &str) -> Result<u64> {
        self.get(tag).await.map(|tag| tag.version)
    }

    pub async fn get(&self, tag: &str) -> Result<TagContents> {
        check_valid_tag(tag)?;

        let root_location = self.refs.root()?;
        let tag_file = tag_path(&root_location.path, tag);

        if !self.object_store().exists(&tag_file).await? {
            return Err(Error::RefNotFound {
                message: format!("tag {} does not exist", tag),
            });
        }

        let tag_contents = TagContents::from_path(&tag_file, self.object_store()).await?;

        Ok(tag_contents)
    }

    pub async fn create(&mut self, tag: &str, version: u64) -> Result<()> {
        self.create_on_branch(tag, version, None).await
    }

    pub async fn create_on_branch(
        &mut self,
        tag: &str,
        version_number: u64,
        branch: Option<&str>,
    ) -> Result<()> {
        check_valid_tag(tag)?;

        let root_location = self.refs.root()?;
        let branch = branch.map(String::from);
        let tag_file = tag_path(&root_location.path, tag);

        if self.object_store().exists(&tag_file).await? {
            return Err(Error::RefConflict {
                message: format!("tag {} already exists", tag),
            });
        }

        let branch_location = self.refs.base_location.find_branch(branch.clone())?;
        let manifest_file = self
            .refs
            .commit_handler
            .resolve_version_location(
                &branch_location.path,
                version_number,
                &self.refs.object_store.inner,
            )
            .await?;

        if !self.object_store().exists(&manifest_file.path).await? {
            return Err(Error::VersionNotFound {
                message: format!(
                    "version {}::{} does not exist",
                    branch.unwrap_or("Main".to_string()),
                    version_number
                ),
            });
        }

        let manifest_size = if let Some(size) = manifest_file.size {
            size as usize
        } else {
            self.object_store().size(&manifest_file.path).await? as usize
        };

        let tag_contents = TagContents {
            branch,
            version: version_number,
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

    pub async fn delete(&mut self, tag: &str) -> Result<()> {
        check_valid_tag(tag)?;

        let root_location = self.refs.root()?;
        let tag_file = tag_path(&root_location.path, tag);

        if !self.object_store().exists(&tag_file).await? {
            return Err(Error::RefNotFound {
                message: format!("tag {} does not exist", tag),
            });
        }

        self.object_store().delete(&tag_file).await
    }

    pub async fn update(&mut self, tag: &str, version: u64) -> Result<()> {
        self.update_on_branch(tag, version, None).await
    }

    /// Update a tag to a branch::version
    pub async fn update_on_branch(
        &mut self,
        tag: &str,
        version_number: u64,
        branch: Option<&str>,
    ) -> Result<()> {
        check_valid_tag(tag)?;

        let branch = branch.map(String::from);
        let root_location = self.refs.root()?;
        let tag_file = tag_path(&root_location.path, tag);

        if !self.object_store().exists(&tag_file).await? {
            return Err(Error::RefNotFound {
                message: format!("tag {} does not exist", tag),
            });
        }

        let target_branch_location = self.refs.base_location.find_branch(branch.clone())?;
        let manifest_file = self
            .refs
            .commit_handler
            .resolve_version_location(
                &target_branch_location.path,
                version_number,
                &self.refs.object_store.inner,
            )
            .await?;

        if !self.object_store().exists(&manifest_file.path).await? {
            return Err(Error::VersionNotFound {
                message: format!("version {} does not exist", version_number),
            });
        }

        let manifest_size = if let Some(size) = manifest_file.size {
            size as usize
        } else {
            self.object_store().size(&manifest_file.path).await? as usize
        };

        let tag_contents = TagContents {
            branch,
            version: version_number,
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
}

impl Branches<'_> {
    pub async fn fetch(&self) -> Result<Vec<(String, BranchContents)>> {
        let root_location = self.refs.root()?;
        let base_path = base_branches_contents_path(&root_location.path);
        let branch_files = self.object_store().read_dir(base_path).await?;

        let branch_names: Vec<String> = branch_files
            .iter()
            .filter_map(|name| name.strip_suffix(".json"))
            .map(|str| {
                Path::from_url_path(str)
                    .map_err(|e| Error::InvalidRef {
                        message: format!(
                            "Failed to decode branch name: {} due to exception {}",
                            str, e
                        ),
                    })
                    .map(|path| path.to_string())
            })
            .collect::<Result<Vec<_>>>()?;

        let branch_path = &root_location.path;
        futures::stream::iter(branch_names)
            .map(|name| async move {
                let contents = BranchContents::from_path(
                    &branch_contents_path(branch_path, &name),
                    self.object_store(),
                )
                .await?;
                Ok((name, contents))
            })
            .buffer_unordered(10)
            .try_collect()
            .await
    }

    pub async fn list(&self) -> Result<HashMap<String, BranchContents>> {
        self.fetch()
            .await
            .map(|branches| branches.into_iter().collect())
    }

    pub async fn get(&self, branch: &str) -> Result<BranchContents> {
        check_valid_branch(branch)?;

        let root_location = self.refs.root()?;
        let branch_file = branch_contents_path(&root_location.path, branch);

        if !self.object_store().exists(&branch_file).await? {
            return Err(Error::RefNotFound {
                message: format!("branch {} does not exist", branch),
            });
        }

        let branch_contents = BranchContents::from_path(&branch_file, self.object_store()).await?;

        Ok(branch_contents)
    }

    pub async fn create(
        &mut self,
        branch_name: &str,
        version_number: u64,
        source_branch: Option<&str>,
    ) -> Result<()> {
        check_valid_branch(branch_name)?;

        let source_branch = source_branch.map(String::from);
        let root_location = self.refs.root()?;
        let branch_file = branch_contents_path(&root_location.path, branch_name);
        if self.object_store().exists(&branch_file).await? {
            return Err(Error::RefConflict {
                message: format!("branch {} already exists", branch_name),
            });
        }

        let branch_location = self.refs.base_location.find_branch(source_branch.clone())?;
        // Verify the source version exists
        let manifest_file = self
            .refs
            .commit_handler
            .resolve_version_location(
                &branch_location.path,
                version_number,
                &self.refs.object_store.inner,
            )
            .await?;

        if !self.object_store().exists(&manifest_file.path).await? {
            return Err(Error::VersionNotFound {
                message: format!("Manifest file {} does not exist", &manifest_file.path),
            });
        };

        let branch_contents = BranchContents {
            parent_branch: source_branch,
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

    /// Delete a branch
    ///
    /// If the `BranchContents` does not exist, it will return an error directly unless `force` is true.
    /// If `force` is true, it will try to delete the branch directories no matter `BranchContents` exists or not.
    pub async fn delete(&mut self, branch: &str, force: bool) -> Result<()> {
        check_valid_branch(branch)?;

        let root_location = self.refs.root()?;
        let branch_file = branch_contents_path(&root_location.path, branch);
        if self.object_store().exists(&branch_file).await? {
            self.object_store().delete(&branch_file).await?;
        } else if force {
            log::warn!("BranchContents of {} does not exist", branch);
        } else {
            return Err(Error::RefNotFound {
                message: format!("Branch {} does not exist", branch),
            });
        }

        // Clean up branch directories
        self.cleanup_branch_directories(branch).await
    }

    pub async fn list_ordered(
        &self,
        order: Option<Ordering>,
    ) -> Result<Vec<(String, BranchContents)>> {
        let mut branches = self.fetch().await?;
        branches.sort_by(|a, b| {
            let desired_ordering = order.unwrap_or(Ordering::Greater);
            let version_ordering = a.1.parent_version.cmp(&b.1.parent_version);
            let version_result = match desired_ordering {
                Ordering::Less => version_ordering,
                _ => version_ordering.reverse(),
            };
            version_result.then_with(|| a.0.cmp(&b.0))
        });
        Ok(branches)
    }

    /// Clean up empty parent directories
    async fn cleanup_branch_directories(&self, branch: &str) -> Result<()> {
        let branches = self.list().await?;
        let remaining_branches: Vec<&str> = branches.keys().map(|k| k.as_str()).collect();

        if let Some(delete_path) =
            Self::get_cleanup_path(branch, &remaining_branches, &self.refs.base_location)?
        {
            if let Err(e) = self.refs.object_store.remove_dir_all(delete_path).await {
                match &e {
                    Error::IO { source, .. } => {
                        if let Some(io_err) = source.downcast_ref::<std::io::Error>() {
                            if io_err.kind() == ErrorKind::NotFound {
                                log::debug!("Branch directory already deleted: {}", io_err);
                            } else {
                                return Err(e);
                            }
                        } else {
                            return Err(e);
                        }
                    }
                    _ => return Err(e),
                }
            }
        }
        Ok(())
    }

    fn get_cleanup_path(
        branch: &str,
        remaining_branches: &[&str],
        base_location: &BranchLocation,
    ) -> Result<Option<Path>> {
        let mut longest_used_length = 0;
        for &candidate in remaining_branches {
            let common_len = branch
                .chars()
                .zip(candidate.chars())
                .take_while(|(a, b)| a == b)
                .count();

            if common_len > longest_used_length {
                longest_used_length = common_len;
            }
        }
        // Means this branch path is used as a prefix of other branches
        if longest_used_length == branch.len() {
            return Ok(None);
        }

        let mut used_relative_path = &branch[..longest_used_length];
        if let Some(last_slash_index) = used_relative_path.rfind('/') {
            used_relative_path = &used_relative_path[..last_slash_index];
        }
        let unused_dir = &branch[used_relative_path.len()..].trim_start_matches('/');
        if let Some(sub_dir) = unused_dir.split('/').next() {
            let relative_dir = format!("{}/{}", used_relative_path, sub_dir);
            // Use base_location to generate the cleanup path
            let absolute_dir = base_location.find_branch(Some(relative_dir))?;
            Ok(Some(absolute_dir.path))
        } else {
            Ok(None)
        }
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

pub fn base_branches_contents_path(base_path: &Path) -> Path {
    base_path.child("_refs").child("branches")
}

pub fn tag_path(base_path: &Path, branch: &str) -> Path {
    base_tags_path(base_path).child(format!("{}.json", branch))
}

// Note: child will encode '/' to '%2F'
pub fn branch_contents_path(base_path: &Path, branch: &str) -> Path {
    base_branches_contents_path(base_path).child(format!("{}.json", branch))
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
    let json_str = String::from_utf8(tag_bytes.to_vec())
        .map_err(|e| Error::corrupt_file(path.clone(), e.to_string(), location!()))?;
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

pub fn check_valid_branch(branch_name: &str) -> Result<()> {
    if branch_name.is_empty() {
        return Err(Error::InvalidRef {
            message: "Branch name cannot be empty".to_string(),
        });
    }

    // Validate if the branch name starts or ends with a '/'
    if branch_name.starts_with('/') || branch_name.ends_with('/') {
        return Err(Error::InvalidRef {
            message: "Branch name cannot start or end with a '/'".to_string(),
        });
    }

    // Validate if there are any consecutive '/' in the branch name
    if branch_name.contains("//") {
        return Err(Error::InvalidRef {
            message: "Branch name cannot contain consecutive '/'".to_string(),
        });
    }

    // Validate if there are any dangerous characters in the branch name
    if branch_name.contains("..") || branch_name.contains('\\') {
        return Err(Error::InvalidRef {
            message: "Branch name cannot contain '..' or '\\'".to_string(),
        });
    }

    for segment in branch_name.split('/') {
        if segment.is_empty() {
            return Err(Error::InvalidRef {
                message: "Branch name cannot have empty segments between '/'".to_string(),
            });
        }
        if !segment
            .chars()
            .all(|c| c.is_alphanumeric() || c == '.' || c == '-' || c == '_')
        {
            return Err(Error::InvalidRef {
                message: format!("Branch segment '{}' contains invalid characters. Only alphanumeric, '.', '-', '_' are allowed.", segment),
            });
        }
    }

    if branch_name.ends_with(".lock") {
        return Err(Error::InvalidRef {
            message: "Branch name cannot end with '.lock'".to_string(),
        });
    }

    if branch_name.eq("main") {
        return Err(Error::InvalidRef {
            message: "Branch name cannot be 'main'".to_string(),
        });
    }
    Ok(())
}

pub fn check_valid_tag(s: &str) -> Result<()> {
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
        check_valid_tag(r).unwrap();
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
            check_valid_tag(r).err().unwrap().to_string(),
            "Ref is invalid: Ref"
        );
    }

    #[rstest]
    fn test_valid_branch_names(
        #[values(
            "feature/login",
            "bugfix/issue-123",
            "release/v1.2.3",
            "user/someone/my-feature",
            "normal",
            "with-dash",
            "with_underscore",
            "with.dot"
        )]
        branch_name: &str,
    ) {
        assert!(
            check_valid_branch(branch_name).is_ok(),
            "Branch name '{}' should be valid",
            branch_name
        );
    }

    #[rstest]
    fn test_invalid_branch_names(
        #[values(
            "",
            "/start-with-slash",
            "end-with-slash/",
            "have//consecutive-slash",
            "have..dot-dot",
            "have\\backslash",
            "segment/",
            "/segment",
            "segment//empty",
            "name.lock",
            "bad@character",
            "bad segment"
        )]
        branch_name: &str,
    ) {
        assert!(
            check_valid_branch(branch_name).is_err(),
            "Branch name '{}' should be invalid",
            branch_name
        );
    }

    #[test]
    fn test_path_functions() {
        let base_path = Path::from("dataset");

        // Test base_tags_path
        let tags_path = base_tags_path(&base_path);
        assert_eq!(tags_path, Path::from("dataset/_refs/tags"));

        // Test base_branches_path
        let branches_path = base_branches_contents_path(&base_path);
        assert_eq!(branches_path, Path::from("dataset/_refs/branches"));

        // Test tag_path
        let tag_file_path = tag_path(&base_path, "v1.0.0");
        assert_eq!(tag_file_path, Path::from("dataset/_refs/tags/v1.0.0.json"));

        // Test branch_path
        let branch_file_path = branch_contents_path(&base_path, "feature");
        assert_eq!(
            branch_file_path,
            Path::from("dataset/_refs/branches/feature.json")
        );
    }

    #[tokio::test]
    async fn test_refs_from_traits() {
        // Test From<u64> for Ref
        let version_ref: Ref = 42u64.into();
        match version_ref {
            Version(branch, v) => {
                assert_eq!(v, Some(42));
                assert_eq!(branch, None)
            }
            _ => panic!("Expected Version variant"),
        }

        // Test From<&str> for Ref
        let tag_ref: Ref = "test_tag".into();
        match tag_ref {
            Tag(name) => assert_eq!(name, "test_tag"),
            _ => panic!("Expected Tag variant"),
        }

        // Test From<(&str, u64)> for Ref
        let branch_ref: Ref = ("test_branch", 10u64).into();
        match branch_ref {
            Version(name, version) => {
                assert_eq!(name.unwrap(), "test_branch");
                assert_eq!(version, Some(10));
            }
            _ => panic!("Expected Branch variant"),
        }
    }

    #[tokio::test]
    async fn test_branch_contents_serialization() {
        let branch_contents = BranchContents {
            parent_branch: Some("main".to_string()),
            parent_version: 42,
            create_at: 1234567890,
            manifest_size: 1024,
        };

        // Test serialization
        let json = serde_json::to_string(&branch_contents).unwrap();
        assert!(json.contains("parentBranch"));
        assert!(json.contains("parentVersion"));
        assert!(json.contains("createAt"));
        assert!(json.contains("manifestSize"));

        // Test deserialization
        let deserialized: BranchContents = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.parent_branch, branch_contents.parent_branch);
        assert_eq!(deserialized.parent_version, branch_contents.parent_version);
        assert_eq!(deserialized.create_at, branch_contents.create_at);
        assert_eq!(deserialized.manifest_size, branch_contents.manifest_size);
    }

    #[tokio::test]
    async fn test_tag_contents_serialization() {
        let tag_contents = TagContents {
            branch: Some("feature".to_string()),
            version: 10,
            manifest_size: 2048,
        };

        // Test serialization
        let json = serde_json::to_string(&tag_contents).unwrap();
        assert!(json.contains("branch"));
        assert!(json.contains("version"));
        assert!(json.contains("manifestSize"));

        // Test deserialization
        let deserialized: TagContents = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.branch, tag_contents.branch);
        assert_eq!(deserialized.version, tag_contents.version);
        assert_eq!(deserialized.manifest_size, tag_contents.manifest_size);
    }

    #[rstest]
    #[case("feature/auth", &["feature/login", "feature/signup"], Some("feature/auth"))]
    #[case("feature/auth/module", &["feature/other"], Some("feature/auth"))]
    #[case("a/b/c", &["a/b/d", "a/e"], Some("a/b/c"))]
    #[case("feature/auth", &["feature/auth/sub"], None)]
    #[case("feature", &["feature/sub1", "feature/sub2"], None)]
    #[case("a/b", &["a/b/c", "a/b/d"], None)]
    #[case("main", &[], Some("main"))]
    #[case("a", &["a"], None)]
    #[case("single", &["other"], Some("single"))]
    #[case("feature/auth/login/oauth", &["feature/auth/login/basic", "feature/auth/signup"], Some("feature/auth/login/oauth"))]
    #[case("feature/user-auth", &["feature/user-signup"], Some("feature/user-auth"))]
    #[case("release/2024.01", &["release/2024.02"], Some("release/2024.01"))]
    #[case("very/long/common/prefix/branch1", &["very/long/common/prefix/branch2"], Some("very/long/common/prefix/branch1"))]
    #[case("feature", &["bugfix", "hotfix"], Some("feature"))]
    #[case("feature/sub", &["feature", "other"], Some("feature/sub"))]
    fn test_get_cleanup_path(
        #[case] branch_to_delete: &str,
        #[case] remaining_branches: &[&str],
        #[case] expected_relative_cleanup_path: Option<&str>,
    ) {
        let dataset_root_dir = "file:///var/balabala/dataset1".to_string();
        let base_location = BranchLocation {
            path: Path::from(format!("{}/tree/random_branch", dataset_root_dir.as_str())),
            uri: format!("{}/tree/random_branch", dataset_root_dir.as_str()),
            branch: Some("random_branch".to_string()),
        };

        let result =
            Branches::get_cleanup_path(branch_to_delete, remaining_branches, &base_location)
                .unwrap();

        match expected_relative_cleanup_path {
            Some(expected_relative) => {
                assert!(
                    result.is_some(),
                    "Expected cleanup path but got None for branch: {}",
                    branch_to_delete
                );
                let expected_full_path = base_location
                    .find_branch(Some(expected_relative.to_string()))
                    .unwrap()
                    .path;
                assert_eq!(result.unwrap().as_ref(), expected_full_path.as_ref());
            }
            None => {
                assert!(
                    result.is_none(),
                    "Expected no cleanup but got: {:?} for branch: {}",
                    result,
                    branch_to_delete
                );
            }
        }
    }
}
