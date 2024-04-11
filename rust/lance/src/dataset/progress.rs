// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use async_trait::async_trait;
use lance_table::format::Fragment;

use crate::Result;

/// Progress of writing a [Fragment].
///
/// When start writing a [`Fragment`], WriteProgress::begin() will be called before
/// writing any data.
///
/// When stop writing a [`Fragment`], WriteProgress::complete() will be called after.
///
/// This might be called concurrently when writing multiple [`Fragment`]s. Therefore,
/// the methods require non-exclusive access to `self`.
///
/// This is an experimental API and may change in the future.
#[async_trait]
pub trait WriteFragmentProgress: std::fmt::Debug + Sync + Send {
    /// Indicate the beginning of writing a [Fragment], with the in-flight multipart ID.
    async fn begin(&self, fragment: &Fragment, multipart_id: &str) -> Result<()>;

    /// Complete writing a [Fragment].
    async fn complete(&self, fragment: &Fragment) -> Result<()>;
}

/// By default, Progress tracker is Noop.
#[derive(Debug, Clone, Default)]
pub struct NoopFragmentWriteProgress {}

impl NoopFragmentWriteProgress {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl WriteFragmentProgress for NoopFragmentWriteProgress {
    #[inline]
    async fn begin(&self, _fragment: &Fragment, _multipart_id: &str) -> Result<()> {
        Ok(())
    }

    #[inline]
    async fn complete(&self, _fragment: &Fragment) -> Result<()> {
        Ok(())
    }
}
