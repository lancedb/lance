// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// An absolute POSIX image path, excluding traversal and `/run/lance`.
///
/// Redundant separators and `.` are normalized before path comparisons.
/// Symlinks require a separate [`ImageRoot::resolve`] check.
///
/// ```
/// use lance_function::ContainerPath;
/// let path = ContainerPath::new("/opt/function/./input.arrow")?;
/// assert_eq!(path.as_str(), "/opt/function/input.arrow");
/// # Ok::<(), lance_function::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ContainerPath(String);

impl ContainerPath {
    /// Validate a descriptor path without accessing the filesystem.
    pub fn new(path: impl AsRef<str>) -> Result<Self> {
        let path = path.as_ref();
        if !path.starts_with('/') || path.contains('\0') || path.split('/').any(|part| part == "..")
        {
            return Err(Error::incompatible(format!(
                "image path {path:?} must be absolute POSIX without NUL or '..'"
            )));
        }
        let parts: Vec<_> = path
            .split('/')
            .filter(|part| !part.is_empty() && *part != ".")
            .collect();
        let normalized = format!("/{}", parts.join("/"));
        if reserved(&normalized) {
            return Err(Error::incompatible(format!(
                "image path {path:?} uses reserved /run/lance"
            )));
        }
        Ok(Self(normalized))
    }

    /// The normalized container path, never a host cache path.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ContainerPath {
    type Error = Error;

    fn try_from(value: String) -> Result<Self> {
        Self::new(value)
    }
}

impl From<ContainerPath> for String {
    fn from(value: ContainerPath) -> Self {
        value.0
    }
}

fn reserved(path: &str) -> bool {
    path == "/run/lance" || path.starts_with("/run/lance/")
}

/// Expected kind after all image symlinks have been resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathKind {
    /// A regular file (schemas and the interpreter).
    File,
    /// A directory (declared Python import paths).
    Directory,
}

/// A caller-owned, immutable, unpacked image root.
///
/// The caller must prevent concurrent filesystem mutation throughout validation
/// and execution. This resolver is for an immutable cache, not a sandbox against
/// a process racing filesystem writes. Mounts are checked independently by the
/// launcher before execution.
#[derive(Debug, Clone)]
pub struct ImageRoot {
    root: PathBuf,
}

impl ImageRoot {
    /// Select a host rootfs directory; only this trusted host path is canonicalized.
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        if !cfg!(unix) {
            return Err(Error::incompatible(
                "image root validation requires a POSIX host filesystem",
            ));
        }
        let root = root.as_ref().canonicalize().map_err(|error| {
            Error::incompatible(format!("image root {}: {error}", root.as_ref().display()))
        })?;
        if !root.is_dir() {
            return Err(Error::incompatible(format!(
                "image root {} is not a directory",
                root.display()
            )));
        }
        Ok(Self { root })
    }

    /// Resolve a path using container-root symlink semantics and check its kind.
    ///
    /// Absolute symlink targets restart at the image root. Relative `..` in a
    /// symlink target walks the image tree and stops at its root, as on Linux.
    /// Links into `/run/lance` and loops are rejected before opening any content.
    pub fn resolve(&self, path: &ContainerPath, kind: PathKind) -> Result<PathBuf> {
        // Linux limits one path lookup to 40 symlink traversals. Match it so
        // validation cannot accept a path that the image interpreter cannot open.
        const MAX_SYMLINKS: usize = 40;
        let mut remaining: VecDeque<String> = path.as_str().split('/').map(str::to_owned).collect();
        let mut resolved = Vec::<String>::new();
        let mut links = 0;
        while let Some(part) = remaining.pop_front() {
            match part.as_str() {
                "" | "." => continue,
                ".." => {
                    resolved.pop();
                    continue;
                }
                _ => resolved.push(part),
            }
            let container = format!("/{}", resolved.join("/"));
            if reserved(&container) {
                return Err(Error::incompatible(format!(
                    "image path {} resolves through reserved {container}",
                    path.as_str()
                )));
            }
            let host = resolved
                .iter()
                .fold(self.root.clone(), |path, part| path.join(part));
            let metadata = fs::symlink_metadata(&host).map_err(|error| {
                Error::incompatible(format!(
                    "image path {} at {container}: {error}",
                    path.as_str()
                ))
            })?;
            if metadata.file_type().is_symlink() {
                links += 1;
                if links > MAX_SYMLINKS {
                    return Err(Error::incompatible(format!(
                        "image path {} exceeds {MAX_SYMLINKS} symlink traversals",
                        path.as_str()
                    )));
                }
                let target = fs::read_link(&host).map_err(|error| {
                    Error::incompatible(format!("image link {container}: {error}"))
                })?;
                let target = target.to_str().ok_or_else(|| {
                    Error::incompatible(format!("image link {container} target is not UTF-8"))
                })?;
                resolved.pop();
                if target.starts_with('/') {
                    resolved.clear();
                }
                for part in target.split('/').rev() {
                    remaining.push_front(part.to_owned());
                }
            } else if !remaining.is_empty() && !metadata.is_dir() {
                return Err(Error::incompatible(format!(
                    "image path {container} is not a directory"
                )));
            }
        }
        let host = resolved
            .iter()
            .fold(self.root.clone(), |path, part| path.join(part));
        let metadata = fs::symlink_metadata(&host).map_err(|error| {
            Error::incompatible(format!("image path {}: {error}", path.as_str()))
        })?;
        let matches = match kind {
            PathKind::File => metadata.is_file(),
            PathKind::Directory => metadata.is_dir(),
        };
        if !matches {
            return Err(Error::incompatible(format!(
                "image path {} is not {kind:?}",
                path.as_str()
            )));
        }
        Ok(host)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorCode;
    use rstest::rstest;

    #[rstest]
    #[case::empty("")]
    #[case::relative("opt/code")]
    #[case::traversal("/opt/../code")]
    #[case::reserved("/run//./lance/schema")]
    #[case::nul("/opt/\0code")]
    fn rejects_invalid_paths(#[case] path: &str) {
        let error = ContainerPath::new(path).unwrap_err();
        assert_eq!(error.code, ErrorCode::Incompatible);
        assert!(error.message.contains("image path"));
    }

    #[test]
    fn normalizes_posix_paths() {
        assert_eq!(
            ContainerPath::new("//opt/./code/").unwrap().as_str(),
            "/opt/code"
        );
        assert!(ContainerPath::new("/run/lance-other/code").is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_stay_in_image_root() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        fs::create_dir_all(root.path().join("opt/code")).unwrap();
        fs::create_dir(root.path().join("bin")).unwrap();
        fs::write(root.path().join("opt/code/schema.arrow"), b"image bytes").unwrap();
        symlink("/opt/code", root.path().join("bin/absolute")).unwrap();
        symlink("../../opt/code", root.path().join("bin/relative")).unwrap();
        symlink("/run/lance/schema", root.path().join("reserved")).unwrap();
        symlink("loop", root.path().join("loop")).unwrap();
        symlink("/etc/passwd", root.path().join("host-file")).unwrap();
        let image = ImageRoot::new(root.path()).unwrap();
        for path in ["/bin/absolute/schema.arrow", "/bin/relative/schema.arrow"] {
            let file = image
                .resolve(&ContainerPath::new(path).unwrap(), PathKind::File)
                .unwrap();
            assert_eq!(fs::read(file).unwrap(), b"image bytes");
        }
        for (path, reason) in [
            ("/reserved", "reserved"),
            ("/loop", "symlink"),
            ("/host-file", "/etc"),
        ] {
            let error = image
                .resolve(&ContainerPath::new(path).unwrap(), PathKind::File)
                .unwrap_err();
            assert_eq!(error.code, ErrorCode::Incompatible);
            assert!(error.message.contains(reason), "{error}");
        }
    }
}
