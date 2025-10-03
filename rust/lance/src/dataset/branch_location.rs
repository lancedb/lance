// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::{Error, Result};
use object_store::path::Path;
use snafu::location;

pub const BRANCH_DIR: &str = "tree";

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct BranchLocation {
    pub path: Path,
    pub uri: String,
    pub branch: Option<String>,
}

impl BranchLocation {
    /// Find the root location
    pub fn find_main(&self) -> Result<Self> {
        if let Some(branch_name) = self.branch.as_ref() {
            let root_path_str = Self::get_root_path(self.path.as_ref(), branch_name)?;
            let root_uri = Self::get_root_path(self.uri.as_str(), branch_name)?;
            Ok(Self {
                path: Path::parse(root_path_str)?,
                uri: root_uri,
                branch: None,
            })
        } else {
            Ok(self.clone())
        }
    }

    fn get_root_path(path_str: &str, branch_name: &str) -> Result<String> {
        let branch_suffix = format!("{}/{}", BRANCH_DIR, branch_name);
        let branch_suffix = branch_suffix.as_str();
        let root_path_str = path_str
            .strip_suffix(branch_suffix)
            .or_else(|| {
                if cfg!(windows) {
                    let windows_suffix = branch_suffix.replace('/', "\\");
                    path_str.strip_suffix(&windows_suffix)
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                Error::invalid_input(
                    format!(
                        "Can not find the root location of branch {} by uri {}",
                        branch_name, path_str,
                    ),
                    location!(),
                )
            })?;
        let root_path_str = if root_path_str.ends_with('/') {
            root_path_str.trim_end_matches('/').to_string()
        } else if cfg!(windows) {
            root_path_str.trim_end_matches('\\').to_string()
        } else {
            return Err(Error::invalid_input(
                format!(
                    "Invalid dataset root uri {} for branch {}",
                    root_path_str, path_str,
                ),
                location!(),
            ));
        };
        Ok(root_path_str)
    }

    /// Find the target branch location
    pub fn find_branch(&self, branch_name: Option<String>) -> Result<Self> {
        if branch_name == self.branch {
            return Ok(self.clone());
        }

        let root_location = self.find_main()?;
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
    use lance_core::utils::tempfile::TempStdDir;
    use object_store::path::Path;
    use std::fs;
    use std::path::PathBuf;

    // Create a BranchLocation instance for testing
    fn create_branch_location(root_path: PathBuf) -> BranchLocation {
        let branch_dir = root_path.join("tree/feature/new");
        let test_uri = branch_dir.to_str().unwrap().to_string();
        BranchLocation {
            path: Path::parse(&test_uri).unwrap(),
            uri: test_uri,
            branch: Some("feature/new".to_string()),
        }
    }

    #[test]
    fn test_find_main_from_branch() {
        let root_path = TempStdDir::default().to_owned();
        let location = create_branch_location(root_path.clone());
        let main_location = location.find_main().unwrap();

        assert_eq!(
            main_location.path.as_ref(),
            Path::parse(root_path.to_str().unwrap()).unwrap().as_ref()
        );
        assert_eq!(main_location.uri, root_path.to_str().unwrap().to_string());
        assert_eq!(main_location.branch, None);
        assert!(fs::create_dir(std::path::Path::new(main_location.uri.as_str())).is_ok());
    }

    #[test]
    fn test_find_main_from_root() {
        let root_path = TempStdDir::default().to_owned();
        let mut location = create_branch_location(root_path);
        // Change current branch to Main
        location.branch = None;
        let root_location = location.find_main().unwrap();

        assert_eq!(root_location.path, location.path);
        assert_eq!(root_location.uri, location.uri);
        assert_eq!(root_location.branch, None);
        assert!(fs::create_dir_all(std::path::Path::new(root_location.uri.as_str())).is_ok());
    }

    #[test]
    fn test_find_branch_from_same_branch() {
        let root_path = TempStdDir::default().to_owned();
        let location = create_branch_location(root_path);
        let target_branch = location.branch.clone();
        let new_location = location.find_branch(target_branch).unwrap();

        assert_eq!(new_location.path, location.path);
        assert_eq!(new_location.uri, location.uri);
        assert_eq!(new_location.branch, location.branch);
        assert!(fs::create_dir_all(std::path::Path::new(new_location.uri.as_str())).is_ok());
    }

    #[test]
    fn test_find_main_branch() {
        let root_path = TempStdDir::default().to_owned();
        let location = create_branch_location(root_path);
        let main_location = location.find_branch(None).unwrap();

        let expected_root = location.find_main().unwrap();
        assert_eq!(main_location.path, expected_root.path);
        assert_eq!(main_location.uri, expected_root.uri);
        assert_eq!(main_location.branch, None);
        assert!(fs::create_dir_all(std::path::Path::new(main_location.uri.as_str())).is_ok());
    }

    #[test]
    fn test_find_simple_branch() {
        let root_path = TempStdDir::default().to_owned();
        let location = create_branch_location(root_path);
        let new_branch = Some("featureA".to_string());
        let main_location = location.find_main().unwrap();
        let new_location = location.find_branch(new_branch.clone()).unwrap();

        assert_eq!(
            new_location.path.as_ref(),
            format!("{}/tree/featureA", main_location.path.as_ref())
        );
        assert_eq!(
            new_location.uri,
            format!("{}/tree/featureA", main_location.uri)
        );
        assert_eq!(new_location.branch, new_branch);
        assert!(fs::create_dir_all(std::path::Path::new(new_location.uri.as_str())).is_ok());
    }

    #[test]
    fn test_find_complex_branch() {
        let root_path = TempStdDir::default().to_owned();
        let location = create_branch_location(root_path);
        let new_branch = Some("bugfix/issue-123".to_string());
        let main_location = location.find_main().unwrap();
        let new_location = location.find_branch(new_branch).unwrap();

        assert_eq!(
            new_location.path.as_ref(),
            format!("{}/tree/bugfix/issue-123", main_location.path.as_ref())
        );
        assert_eq!(
            new_location.uri,
            format!("{}/tree/bugfix/issue-123", main_location.uri)
        );
        assert!(fs::create_dir_all(std::path::Path::new(new_location.uri.as_str())).is_ok());
    }

    #[test]
    fn test_find_empty_branch() {
        let root_path = TempStdDir::default().to_owned();
        let location = create_branch_location(root_path);
        let new_branch = Some("".to_string());
        let new_location = location.find_branch(new_branch.clone()).unwrap();

        assert_eq!(new_location.path, location.path);
        assert_eq!(new_location.uri, location.uri);
        assert_eq!(new_location.branch, new_branch);
    }

    #[test]
    #[cfg(windows)]
    fn test_branch_location_on_windows() {
        let branch_location = BranchLocation {
            path: Path::parse("C:\\Users\\Username\\Documents\\dataset\\tree\\feature\\new")
                .unwrap(),
            uri: "C:\\Users\\Username\\Documents\\dataset\\tree\\feature\\new".to_string(),
            branch: Some("feature/new".to_string()),
        };

        let main_location = branch_location.find_main().unwrap();
        assert_eq!(main_location.uri, "C:\\Users\\Username\\Documents\\dataset");
        assert_eq!(
            main_location.path.as_ref(),
            Path::parse("C:\\Users\\Username\\Documents\\dataset")
                .unwrap()
                .as_ref()
        );
        assert_eq!(main_location.branch, None);

        let new_branch = branch_location
            .find_branch(Some("feature/nathan/A".to_string()))
            .unwrap();
        assert_eq!(
            new_branch.uri,
            "C:\\Users\\Username\\Documents\\dataset/tree/feature/nathan/A"
        );
        assert_eq!(
            new_branch.path.as_ref(),
            Path::parse("C:\\Users\\Username\\Documents\\dataset/tree/feature/nathan/A")
                .unwrap()
                .as_ref()
        );
        assert_eq!(new_branch.branch, Some("feature/nathan/A".to_string()));
    }
}
