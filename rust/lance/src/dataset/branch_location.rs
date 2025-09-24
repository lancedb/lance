// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::{Error, Result};
use object_store::path::Path;
use snafu::location;

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct BranchLocation {
    pub path: Path,
    pub uri: String,
    pub branch: Option<String>,
}

impl BranchLocation {
    /// Find the root location
    pub fn find_root(&self) -> Result<Self> {
        if let Some(branch_name) = self.branch.as_ref() {
            let mut path = self.path.clone();
            let mut uri = self.uri.clone();
            let segment_count = branch_name.split('/').count();
            for _ in 0..segment_count + 1 {
                path = Path::parse(Self::parent_str(path.as_ref())?)?;
                uri = Self::parent_str(uri.as_str())?.to_string();
            }
            Ok(Self {
                path,
                uri,
                branch: None,
            })
        } else {
            Ok(self.clone())
        }
    }

    /// Find the target branch location
    pub fn find_branch(&self, branch_name: Option<String>) -> Result<Self> {
        if branch_name == self.branch {
            return Ok(self.clone());
        }

        let root_location = self.find_root()?;
        if let Some(target_branch) = branch_name.as_ref() {
            let (new_path, new_uri) = {
                // Handle empty segment
                if target_branch.is_empty() {
                    (self.path.clone(), self.uri.clone())
                } else {
                    let segments = target_branch.split('/');
                    let mut new_path_str = Self::join_str(root_location.path.as_ref(), "tree")?;
                    let mut new_uri = Self::join_str(root_location.uri.as_str(), "tree")?;
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
                branch: Some(target_branch.clone()),
            })
        } else {
            Ok(root_location)
        }
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

#[cfg(test)]
mod tests {
    use crate::dataset::branch_location::BranchLocation;
    use object_store::path::Path;

    // Create a BranchLocation instance for testing
    fn create_test_location() -> BranchLocation {
        BranchLocation {
            path: Path::parse("/repo/root/tree/feature/new").unwrap(),
            uri: "https://example.com/repo/root/tree/feature/new".to_string(),
            branch: Some("feature/new".to_string()),
        }
    }

    #[test]
    fn test_find_root_from_branch() {
        let location = create_test_location();
        let root_location = location.find_root().unwrap();

        assert_eq!(root_location.path.as_ref(), "repo/root");
        assert_eq!(root_location.uri, "https://example.com/repo/root");
        assert_eq!(root_location.branch, None);
    }

    #[test]
    fn test_find_root_from_root() {
        let mut location = create_test_location();
        // Change current branch to Main
        location.branch = None;
        let root_location = location.find_root().unwrap();

        assert_eq!(root_location.path, location.path);
        assert_eq!(root_location.uri, location.uri);
        assert_eq!(root_location.branch, None);
    }

    #[test]
    fn test_find_branch_from_same_branch() {
        let location = create_test_location();
        let target_branch = location.branch.clone();
        let new_location = location.find_branch(target_branch).unwrap();

        assert_eq!(new_location.path, location.path);
        assert_eq!(new_location.uri, location.uri);
        assert_eq!(new_location.branch, location.branch);
    }

    #[test]
    fn test_find_main_branch() {
        let location = create_test_location();
        let new_location = location.find_branch(None).unwrap();

        let expected_root = location.find_root().unwrap();
        assert_eq!(new_location.path, expected_root.path);
        assert_eq!(new_location.uri, expected_root.uri);
        assert_eq!(new_location.branch, None);
    }

    #[test]
    fn test_find_simple_branch() {
        let location = create_test_location();
        let new_branch = Some("featureA".to_string());
        let new_location = location.find_branch(new_branch.clone()).unwrap();

        assert_eq!(new_location.path.as_ref(), "repo/root/tree/featureA");
        assert_eq!(
            new_location.uri,
            "https://example.com/repo/root/tree/featureA"
        );
        assert_eq!(new_location.branch, new_branch);
    }

    #[test]
    fn test_find_complex_branch() {
        let location = create_test_location();
        let new_branch = Some("bugfix/issue-123".to_string());
        let new_location = location.find_branch(new_branch.clone()).unwrap();

        assert_eq!(
            new_location.path.as_ref(),
            "repo/root/tree/bugfix/issue-123"
        );
        assert_eq!(
            new_location.uri,
            "https://example.com/repo/root/tree/bugfix/issue-123"
        );
        assert_eq!(new_location.branch, new_branch);
    }

    #[test]
    fn test_find_empty_branch() {
        let location = create_test_location();
        let new_branch = Some("".to_string());
        let new_location = location.find_branch(new_branch.clone()).unwrap();

        assert_eq!(new_location.path, location.path);
        assert_eq!(new_location.uri, location.uri);
        assert_eq!(new_location.branch, new_branch);
    }
}
