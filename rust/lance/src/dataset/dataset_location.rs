// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::dataset::refs::check_valid_branch;
use crate::dataset::BLOB_DIR;
use lance_core::{Error, Result};
use object_store::path::Path;
use snafu::location;

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct LanceLocation {
    pub path: Path,
    pub uri: String,
}

impl LanceLocation {
    /// Get the parent location of the current location
    pub(crate) fn parent(&self) -> Result<Self> {
        Ok(Self {
            path: Path::parse(Self::parent_str(self.path.as_ref())?)?,
            uri: Self::parent_str(self.uri.as_str())?.to_string(),
        })
    }

    /// Joins a path segment to the current path
    pub(crate) fn join(&self, segment: &str) -> Result<Self> {
        let (new_path, new_uri) = {
            // Handle empty segment
            if segment.is_empty() {
                (self.path.clone(), self.uri.clone())
            } else {
                let segments = segment.split('/');
                let mut new_path_str = self.path.as_ref().to_string();
                let mut new_uri = self.uri.clone();
                for segment in segments {
                    new_path_str = Self::join_str(new_path_str.as_str(), segment)?;
                    new_uri = Self::join_str(new_uri.as_str(), segment)?;
                }
                (Path::parse(new_path_str)?, new_uri)
            }
        };
        Ok(Self {
            path: new_path,
            uri: new_uri,
        })
    }

    fn parent_str(path_str: &str) -> Result<&str> {
        let trimmed = path_str.trim_end_matches('/');
        match trimmed.rfind('/') {
            Some(0) => Ok("/"),
            Some(pos) => Ok(&trimmed[..pos]),
            _ => Err(Error::invalid_input(
                format!("Can not construct the parent path of {}", path_str),
                location!(),
            )),
        }
    }

    fn join_str(base: &str, segment: &str) -> Result<String> {
        let normalized_segment = segment.trim_start_matches('/');
        let is_base_dir = base.ends_with("/");
        if is_base_dir {
            Ok(format!("{}{}", base, normalized_segment))
        } else {
            Ok(format!("{}/{}", base, normalized_segment))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct DatasetLocation {
    pub root: LanceLocation,
    pub base: LanceLocation,
    pub branch: Option<String>,
}

impl DatasetLocation {
    /// the uri and base is the target dataset location
    pub fn new(uri: String, base_path: Path, branch: Option<String>) -> Result<Self> {
        let base = LanceLocation {
            path: base_path,
            uri,
        };
        let root = if let Some(branch_name) = branch.as_ref() {
            check_valid_branch(branch_name)?;
            let segment_layer_count = branch_name.split('/').count();
            let mut root_location = base.parent()?;
            for _ in 0..segment_layer_count {
                root_location = root_location.parent()?;
            }
            root_location
        } else {
            base.clone()
        };
        let location = Self { root, base, branch };
        Ok(location)
    }

    pub fn switch_branch(&self, branch: Option<String>) -> Result<Self> {
        let root = self.root.clone();
        let base = if let Some(branch_name) = branch.as_ref() {
            let mut base_location = root.join("tree")?;
            for segment in branch_name.split('/') {
                base_location = base_location.join(segment)?;
            }
            base_location
        } else {
            self.root.clone()
        };
        Ok(Self { root, base, branch })
    }

    pub fn blobs_location(&self) -> Result<Self> {
        Ok(Self {
            root: self.root.clone(),
            base: self.base.join(BLOB_DIR)?,
            branch: self.branch.clone(),
        })
    }

    pub fn base_path(&self) -> &Path {
        &self.base.path
    }

    pub fn base_uri(&self) -> &String {
        &self.base.uri
    }

    pub fn root_path(&self) -> &Path {
        &self.root.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::path::Path;

    #[test]
    fn test_lance_location_parent() {
        // Test normal path
        let location = LanceLocation {
            path: Path::from("a/b/c"),
            uri: "file:///a/b/c".to_string(),
        };
        let parent = location.parent().unwrap();
        assert_eq!(parent.path, Path::from("a/b"));
        assert_eq!(parent.uri, "file:///a/b");

        // Test root path
        let root_location = LanceLocation {
            path: Path::from("/"),
            uri: "file:///".to_string(),
        };
        assert!(root_location.parent().is_err());

        // Test single level path
        let single_location = LanceLocation {
            path: Path::from("a"),
            uri: "file:///a".to_string(),
        };
        assert!(single_location.parent().is_err());
    }

    #[test]
    fn test_lance_location_join() {
        let location = LanceLocation {
            path: Path::from("a/b"),
            uri: "file:///a/b".to_string(),
        };

        // Test normal join operation
        let joined = location.join("c").unwrap();
        assert_eq!(joined.path, Path::from("a/b/c"));
        assert_eq!(joined.uri, "file:///a/b/c");

        // Test join with leading slash (should be normalized)
        let slash_joined = location.join("/d").unwrap();
        assert_eq!(slash_joined.path, Path::from("a/b/d"));
        assert_eq!(slash_joined.uri, "file:///a/b/d");

        // Test join with empty string (should return original location)
        let joined_empty = location.join("").unwrap();
        assert_eq!(joined_empty.path, Path::from("a/b"));
        assert_eq!(joined_empty.uri, "file:///a/b");

        // Test multi-segment path joining (core test case)
        let multi_joined = location.join("level1/level2/level3").unwrap();
        assert_eq!(multi_joined.path, Path::from("a/b/level1/level2/level3"));
        assert_eq!(multi_joined.uri, "file:///a/b/level1/level2/level3");

        // Test join with special characters
        let special_joined = location.join("file.parquet").unwrap();
        assert_eq!(special_joined.path, Path::from("a/b/file.parquet"));
        assert_eq!(special_joined.uri, "file:///a/b/file.parquet");
    }

    #[test]
    fn test_lance_location_join_slash() {
        // Test with trailing slashes in base path
        let location = LanceLocation {
            path: Path::from("a/b/"),
            uri: "file:///a/b/".to_string(),
        };
        let parent = location.parent().unwrap();
        assert_eq!(parent.path, Path::from("a"));
        assert_eq!(parent.uri, "file:///a");

        // Test join operation with multiple segments
        let joined = location.join("c/d").unwrap();
        assert_eq!(joined.path, Path::from("a/b/c/d"));
        assert_eq!(joined.uri, "file:///a/b/c/d");
    }

    #[test]
    fn test_dataset_location_with_branch() {
        // Test without branch (main branch)
        let location =
            DatasetLocation::new("file:///data".to_string(), Path::from("data"), None).unwrap();
        assert_eq!(location.root.uri, "file:///data");
        assert_eq!(location.base.uri, "file:///data");
        assert_eq!(location.branch, None);

        // Test with branch
        let branch_location = DatasetLocation::new(
            "file:///data/tree/branch1".to_string(),
            Path::from("data/tree/branch1"),
            Some("branch1".to_string()),
        )
        .unwrap();
        assert_eq!(branch_location.root.uri, "file:///data");
        assert_eq!(branch_location.base.uri, "file:///data/tree/branch1");
        assert_eq!(branch_location.branch, Some("branch1".to_string()));
    }

    #[test]
    fn test_dataset_location_switch_branch() {
        let location =
            DatasetLocation::new("file:///data".to_string(), Path::from("data"), None).unwrap();

        // Test switching to single-level branch
        let branch_location = location
            .switch_branch(Some("test_branch".to_string()))
            .unwrap();
        assert_eq!(branch_location.branch, Some("test_branch".to_string()));
        assert_eq!(branch_location.base.uri, "file:///data/tree/test_branch");

        // Test switching to multi-segment branch (core test case)
        let multi_branch_location = location
            .switch_branch(Some("feature/auth/module".to_string()))
            .unwrap();
        assert_eq!(
            multi_branch_location.branch,
            Some("feature/auth/module".to_string())
        );
        assert_eq!(
            multi_branch_location.base.uri,
            "file:///data/tree/feature/auth/module"
        );

        // Test switching back to main branch
        let main_location = branch_location.switch_branch(None).unwrap();
        assert_eq!(main_location.branch, None);
        assert_eq!(main_location.base.uri, "file:///data");

        // Test switching from one multi-segment branch to another
        let another_multi_branch = multi_branch_location
            .switch_branch(Some("bugfix/login/issue".to_string()))
            .unwrap();
        assert_eq!(
            another_multi_branch.branch,
            Some("bugfix/login/issue".to_string())
        );
        assert_eq!(
            another_multi_branch.base.uri,
            "file:///data/tree/bugfix/login/issue"
        );
    }

    #[test]
    fn test_dataset_location_blobs_location() {
        let location =
            DatasetLocation::new("file:///data".to_string(), Path::from("data"), None).unwrap();

        let blobs_location = location.blobs_location().unwrap();
        assert_eq!(
            blobs_location.base.uri,
            format!("file:///data/{}", BLOB_DIR)
        );
        assert_eq!(
            blobs_location.base.path.as_ref(),
            format!("data/{}", BLOB_DIR)
        );
        assert_eq!(blobs_location.branch, None);

        let branch_blob_location = location
            .switch_branch(Some("test_branch".to_string()))
            .unwrap()
            .blobs_location()
            .unwrap();
        assert_eq!(
            branch_blob_location.base.uri,
            format!("file:///data/tree/test_branch/{}", BLOB_DIR)
        );
        assert_eq!(
            branch_blob_location.base.path.as_ref(),
            format!("data/tree/test_branch/{}", BLOB_DIR)
        );
        assert_eq!(branch_blob_location.branch, Some("test_branch".to_string()));
    }

    #[test]
    fn test_dataset_location_blobs_with_branch() {
        // Test blobs location with branch
        let branch_location = DatasetLocation::new(
            "file:///data/tree/mybranch".to_string(),
            Path::from("data/tree/mybranch"),
            Some("mybranch".to_string()),
        )
        .unwrap();

        let blobs_location = branch_location.blobs_location().unwrap();
        assert_eq!(
            blobs_location.base.uri,
            format!("file:///data/tree/mybranch/{}", BLOB_DIR)
        );
        assert_eq!(blobs_location.branch, Some("mybranch".to_string()));
        assert_eq!(blobs_location.root.uri, "file:///data");
    }

    #[test]
    fn test_lance_location_special_characters() {
        let location = LanceLocation {
            path: Path::from("data/test_dataset"),
            uri: "file:///data/test_dataset".to_string(),
        };

        // Test join with underscores and numbers
        let joined = location.join("version_123").unwrap();
        assert_eq!(joined.path, Path::from("data/test_dataset/version_123"));
        assert_eq!(joined.uri, "file:///data/test_dataset/version_123");

        // Test join with dots (file extensions)
        let dotted = location.join("file.parquet").unwrap();
        assert_eq!(dotted.path, Path::from("data/test_dataset/file.parquet"));
        assert_eq!(dotted.uri, "file:///data/test_dataset/file.parquet");
    }
}
