// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{ops::Deref, path::Path as StdPath};
use object_store::path::Path as ObjPath;
use tempfile::NamedTempFile;

pub struct TempDir {
    tempdir: tempfile::TempDir,
}

impl TempDir {
    fn new() -> Self {
        let tempdir = tempfile::tempdir().unwrap();
        Self {
            tempdir
        }
    }

    fn norm_path(&self) -> String {
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
        ObjPath::parse(&self.norm_path()).unwrap()
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

impl AsRef<ObjPath> for TempObjDir {
    fn as_ref(&self) -> &ObjPath {
        &self.path
    }
}

impl Default for TempObjDir {
    fn default() -> Self {
        let tempdir = TempDir::default();
        let path = tempdir.obj_path();
        Self { _tempdir: tempdir, path }
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
        let string = tempdir.norm_path();
        Self {
            _tempdir: tempdir,
            string
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

pub struct TempStdDir {
    _tempdir: TempDir,
}

impl Default for TempStdDir {
    fn default() -> Self {
        Self {
            _tempdir: TempDir::default()
        }
    }
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
        Self {
            temppath
        }
    }

    fn norm_path(&self) -> String {
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
        ObjPath::parse(&self.norm_path()).unwrap()
    }
}

impl Default for TempFile {
    fn default() -> Self {
        Self::new()
    }
}

pub struct TempStdFile {
    _tempfile: TempFile,
}

impl Default for TempStdFile {
    fn default() -> Self {
        Self {
            _tempfile: TempFile::default()
        }
    }
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
        Self { _tempfile: tempfile, path }
    }
}

