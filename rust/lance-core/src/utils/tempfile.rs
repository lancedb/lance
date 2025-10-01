// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use object_store::path::Path as ObjPath;
use std::{
    ops::Deref,
    path::{Path as StdPath, PathBuf},
};
use tempfile::NamedTempFile;

use crate::Result;

#[derive(Debug)]
pub struct TempDir {
    tempdir: tempfile::TempDir,
}

impl TempDir {
    fn new() -> Self {
        let tempdir = tempfile::tempdir().unwrap();
        Self { tempdir }
    }

    pub fn try_new() -> Result<Self> {
        let tempdir = tempfile::tempdir()?;
        Ok(Self { tempdir })
    }

    pub fn path_str(&self) -> String {
        if cfg!(windows) {
            self.tempdir.path().to_str().unwrap().replace("\\", "/")
        } else {
            self.tempdir.path().to_str().unwrap().to_owned()
        }
    }

    pub fn std_path(&self) -> &StdPath {
        self.tempdir.path()
    }

    pub fn obj_path(&self) -> ObjPath {
        ObjPath::parse(self.path_str()).unwrap()
    }
}

impl Default for TempDir {
    fn default() -> Self {
        Self::new()
    }
}

pub struct TempObjDir {
    _tempdir: TempDir,
    path: ObjPath,
}

impl Deref for TempObjDir {
    type Target = ObjPath;

    fn deref(&self) -> &Self::Target {
        &self.path
    }
}

impl AsRef<ObjPath> for TempObjDir {
    fn as_ref(&self) -> &ObjPath {
        &self.path
    }
}

impl Default for TempObjDir {
    fn default() -> Self {
        let tempdir = TempDir::default();
        let path = tempdir.obj_path();
        Self {
            _tempdir: tempdir,
            path,
        }
    }
}

pub struct TempStrDir {
    _tempdir: TempDir,
    string: String,
}

impl std::fmt::Display for TempStrDir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.string.fmt(f)
    }
}

impl TempStrDir {
    pub fn as_into_string(&self) -> impl Into<String> {
        self.string.clone()
    }
}

impl Default for TempStrDir {
    fn default() -> Self {
        let tempdir = TempDir::default();
        let string = tempdir.path_str();
        Self {
            _tempdir: tempdir,
            string,
        }
    }
}

impl Deref for TempStrDir {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.string
    }
}

impl AsRef<str> for TempStrDir {
    fn as_ref(&self) -> &str {
        self.string.as_ref()
    }
}

#[derive(Default)]
pub struct TempStdDir {
    _tempdir: TempDir,
}

impl AsRef<StdPath> for TempStdDir {
    fn as_ref(&self) -> &StdPath {
        self._tempdir.std_path()
    }
}

impl Deref for TempStdDir {
    type Target = StdPath;

    fn deref(&self) -> &Self::Target {
        self._tempdir.std_path()
    }
}

pub struct TempFile {
    temppath: NamedTempFile,
}

impl TempFile {
    fn new() -> Self {
        let temppath = tempfile::NamedTempFile::new().unwrap();
        Self { temppath }
    }

    fn path_str(&self) -> String {
        if cfg!(windows) {
            self.temppath.path().to_str().unwrap().replace("\\", "/")
        } else {
            self.temppath.path().to_str().unwrap().to_owned()
        }
    }

    pub fn std_path(&self) -> &StdPath {
        self.temppath.path()
    }

    pub fn obj_path(&self) -> ObjPath {
        ObjPath::parse(self.path_str()).unwrap()
    }
}

impl Default for TempFile {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Default)]
pub struct TempStdFile {
    _tempfile: TempFile,
}

impl AsRef<StdPath> for TempStdFile {
    fn as_ref(&self) -> &StdPath {
        self._tempfile.std_path()
    }
}

impl Deref for TempStdFile {
    type Target = StdPath;

    fn deref(&self) -> &Self::Target {
        self._tempfile.std_path()
    }
}

pub struct TempObjFile {
    _tempfile: TempFile,
    path: ObjPath,
}

impl AsRef<ObjPath> for TempObjFile {
    fn as_ref(&self) -> &ObjPath {
        &self.path
    }
}

impl std::ops::Deref for TempObjFile {
    type Target = ObjPath;

    fn deref(&self) -> &Self::Target {
        &self.path
    }
}

impl Default for TempObjFile {
    fn default() -> Self {
        let tempfile = TempFile::default();
        let path = tempfile.obj_path();
        Self {
            _tempfile: tempfile,
            path,
        }
    }
}

pub struct TempStdPath {
    _tempdir: TempDir,
    path: PathBuf,
}

impl Default for TempStdPath {
    fn default() -> Self {
        let tempdir = TempDir::default();
        let path = format!("{}/some_file", tempdir.path_str());
        let path = PathBuf::from(path);
        Self {
            _tempdir: tempdir,
            path,
        }
    }
}

impl Deref for TempStdPath {
    type Target = PathBuf;

    fn deref(&self) -> &Self::Target {
        &self.path
    }
}

impl AsRef<StdPath> for TempStdPath {
    fn as_ref(&self) -> &StdPath {
        self.path.as_path()
    }
}
