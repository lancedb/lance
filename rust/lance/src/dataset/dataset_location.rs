// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::Error;
use lance_core::Result;
use object_store::path::Path;
use snafu::location;
use crate::dataset::BLOB_DIR;

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct LanceLocation {
    pub path: Path,
    pub uri: String,
}

impl LanceLocation {

    /// Get the parent location of the current location
    pub(crate) fn parent(&self) -> Result<LanceLocation> {
        Ok(LanceLocation {
            path: Path::from(Self::parent_str(self.path.as_ref())?),
            uri: LanceLocation::parent_str(self.uri.as_str())?.to_string(),
        })
    }

    /// Joins a path segment to the current path
    pub(crate) fn join(&self, segment: &str) -> Result<LanceLocation> {
        let new_path = {
            // Handle empty segment
            if segment.is_empty() {
                self.path.clone()
            } else {
                Path::from(LanceLocation::join_str(self.path.as_ref(), segment)?)
            }
        };
        Ok(LanceLocation {
            path: new_path,
            uri: Self::join_str(self.uri.as_str(), segment)?,
        })
    }

    fn parent_str(path_str: &str) -> Result<&str> {
        let trimmed = path_str.trim_end_matches('/');
        match trimmed.rfind('/') {
            Some(0) => Ok("/"),
            Some(pos) => Ok(&trimmed[..pos]),
            _ => Err(Error::invalid_input(format!("Can not construct the parent path of {}", path_str), location!())),
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
    pub fn new(uri: String, base: Path, branch: Option<String>) -> Result<Self> {
        let base = LanceLocation {
            path: base,
            uri,
        };
        let root = if let Some(_) = &branch {
            base.parent().and_then(|p| p.parent())?
        } else {
            base.clone()
        };
        Ok(Self {
            root,
            base,
            branch,
        })
    }

    pub fn switch_branch(&self, branch: Option<String>) -> Result<DatasetLocation> {
        let root = self.root.clone();
        let base = if let Some(branch_name) = branch.as_ref() {
            root.join("tree").and_then(|p| p.join(branch_name.as_str()))?
        } else {
            self.base.clone()
        };
        Ok(Self {
            root,
            base,
            branch,
        })
    }

    pub fn blobs_location(&self) -> Result<DatasetLocation> {
        Ok(DatasetLocation {
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
}