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
use snafu::{location, Location};
use tracing::instrument;

use crate::io::Reader;
use crate::{Error, Result};

/// Convert an [`object_store::path::Path`] to a [`std::path::Path`].
pub fn to_local_path(path: &Path) -> String {
    if cfg!(windows) {
        path.to_string()
    } else {
        format!("/{path}")
    }
}

/// Recursively remove a directory, specified by [`object_store::path::Path`].
pub fn remove_dir_all(path: &Path) -> Result<()> {
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
    #[instrument(level = "debug")]
    pub fn open(path: &Path, block_size: usize) -> Result<Box<dyn Reader>> {
        let local_path = to_local_path(path);
        let file = File::open(local_path).map_err(|e| match e.kind() {
            ErrorKind::NotFound => Error::NotFound {
                uri: path.to_string(),
                location: location!(),
            },
            _ => Error::IO {
                message: e.to_string(),
                location: location!(),
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
impl Reader for LocalObjectReader {
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
    #[instrument(level = "debug", skip(self))]
    async fn get_range(&self, range: Range<usize>) -> Result<Bytes> {
        let file = self.file.clone();
        tokio::task::spawn_blocking(move || {
            let mut buf = BytesMut::with_capacity(range.len());
            // Safety: `buf` is set with appropriate capacity above. It is
            // written to below and we check all data is initialized at that point.
            unsafe { buf.set_len(range.len()) };
            #[cfg(unix)]
            file.read_exact_at(buf.as_mut(), range.start as u64)?;
            #[cfg(windows)]
            read_exact_at(file, buf.as_mut(), range.start as u64)?;

            Ok(buf.freeze())
        })
        .await?
    }
}

#[cfg(windows)]
fn read_exact_at(file: Arc<File>, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    let expected_len = buf.len();
    while !buf.is_empty() {
        match file.seek_read(buf, offset) {
            Ok(0) => break,
            Ok(n) => {
                let tmp = buf;
                buf = &mut tmp[n..];
                offset += n as u64;
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    if !buf.is_empty() {
        Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!(
                "failed to fill whole buffer. Expected {} bytes, got {}",
                expected_len, offset
            ),
        ))
    } else {
        Ok(())
    }
}
