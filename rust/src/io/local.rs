// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Optimized local I/Os

use std::fs::File;
use std::io::ErrorKind;
use std::ops::Range;
use std::sync::Arc;

// TODO: Clean up windows/unix stuff
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use object_store::path::Path;

use super::object_reader::ObjectReader;
use crate::{Error, Result};

/// Convert an [`object_store::path::Path`] to a [`std::path::Path`].
pub(crate) fn to_local_path(path: &Path) -> String {
    if cfg!(windows) {
        path.to_string()
    } else {
        format!("/{path}")
    }
}

/// Recursively remove a directory, specified by [`object_store::path::Path`].
pub(super) fn remove_dir_all(path: &Path) -> Result<()> {
    let local_path = to_local_path(path);
    std::fs::remove_dir_all(local_path)?;
    Ok(())
}

/// [ObjectReader] for local file system.
pub struct LocalObjectReader {
    /// File handler.
    file: Arc<File>,

    /// Fie path.
    path: Path,

    /// Block size, in bytes.
    block_size: usize,
}

impl LocalObjectReader {
    /// Open a local object reader, with default prefetch size.
    pub fn open(path: &Path, block_size: usize) -> Result<Box<dyn ObjectReader>> {
        let local_path = to_local_path(path);
        let file = File::open(local_path).map_err(|e| match e.kind() {
            ErrorKind::NotFound => Error::NotFound {
                uri: path.to_string(),
            },
            _ => Error::IO {
                message: e.to_string(),
            },
        })?;
        Ok(Box::new(Self {
            file: Arc::new(file),
            block_size,
            path: path.clone(),
        }))
    }
}

#[async_trait]
impl ObjectReader for LocalObjectReader {
    fn path(&self) -> &Path {
        &self.path
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    /// Returns the file size.
    async fn size(&self) -> Result<usize> {
        Ok(self.file.metadata()?.len() as usize)
    }

    /// Reads a range of data.
    async fn get_range(&self, range: Range<usize>) -> Result<Bytes> {
        let file = self.file.clone();
        tokio::task::spawn_blocking(move || {
            let mut buf = BytesMut::with_capacity(range.len());
            // Safety: `buf` is set with appropriate capacity above. It is
            // written to below and we check all data is initialized at that point.
            unsafe { buf.set_len(range.len()) };
            #[cfg(unix)]
            let bytes_read = file.read_at(buf.as_mut(), range.start as u64)?;
            #[cfg(windows)]
            let bytes_read = file.seek_read(buf.as_mut(), range.start as u64)?;
            if bytes_read != range.len() {
                return Err(Error::IO {
                    message: format!(
                        "failed to read all bytes from file: expected {}, got {}",
                        range.len(),
                        bytes_read
                    ),
                });
            }
            Ok(buf.freeze())
        })
        .await?
    }
}
