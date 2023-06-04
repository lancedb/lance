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
use std::ops::Range;
use std::path::Path as StdPath;
use std::sync::Arc;

// TODO: Clean up windows/unix stuff
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use object_store::path::Path;
use shellexpand::tilde;

use super::object_reader::ObjectReader;
use crate::{Error, Result};

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
    pub fn open(path: &str, block_size: usize) -> Result<Box<dyn ObjectReader>> {
        let expanded = tilde(path).to_string();
        let expanded_path = StdPath::new(&expanded);

        let local_path = format!("/{}", expanded_path.to_str().unwrap());
        let file = File::open(&local_path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => Error::NotFound {
                uri: local_path.clone(),
            },
            _ => Error::IO {
                message: format!("Failed to open file {}: {}", local_path, e),
            },
        })?;
        Ok(Box::new(Self {
            path: local_path.clone().into(),
            file: file.into(),
            block_size,
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
    ///
    /// TODO: return [arrow_buffer::Buffer] to avoid one memory copy from Bytes to Buffer.
    async fn get_range(&self, range: Range<usize>) -> Result<Bytes> {
        let file = self.file.clone();
        tokio::task::spawn_blocking(move || {
            let mut buf = BytesMut::zeroed(range.len());
            #[cfg(unix)]
            file.read_at(buf.as_mut(), range.start as u64)?;
            #[cfg(windows)]
            file.seek_read(buf.as_mut(), range.start as u64)?;
            Ok(buf.freeze())
        })
        .await?
    }
}
