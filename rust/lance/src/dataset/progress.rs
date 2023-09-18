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

use async_trait::async_trait;

use crate::format::Fragment;
use crate::Result;

/// Progress of writing a [Fragment].
///
/// When start writing a [`Fragment`], WriteProgress::begin() will be called before
/// writing any data.
///
/// When stop writing a [`Fragment`], WriteProgress::complete() will be called after.
///
/// This is an experimental API and may change in the future.
#[async_trait]
pub trait WriteFragmentProgress: std::fmt::Debug + Sync + Send {
    /// Indicate the beginning of writing a [Fragment], with the in-flight multipart ID.
    async fn begin(&mut self, fragment: &Fragment, multipart_id: &str) -> Result<()>;

    /// Complete writing a [Fragment].
    async fn complete(&mut self, fragment: &Fragment) -> Result<()>;
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
    async fn begin(&mut self, _fragment: &Fragment, _multipart_id: &str) -> Result<()> {
        Ok(())
    }

    #[inline]
    async fn complete(&mut self, _fragment: &Fragment) -> Result<()> {
        Ok(())
    }
}
