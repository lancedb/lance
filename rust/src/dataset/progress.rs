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

use crate::format::Fragment;
use crate::Result;

/// Progress of writing a [Fragment].
pub trait WriteFragmentProgress: std::fmt::Debug + Sync + Send {
    /// Indicate the beginning of writing a [Fragment].
    fn begin(&mut self, fragment: &Fragment) -> Result<()>;

    /// Complete writing a [Fragment].
    fn complete(&mut self, fragment: &Fragment) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct NoopFragmentWriteProgress {}

impl NoopFragmentWriteProgress {
    pub fn new() -> Self {
        Self {}
    }
}

impl WriteFragmentProgress for NoopFragmentWriteProgress {
    fn begin(&mut self, _fragment: &Fragment) -> Result<()> {
        Ok(())
    }

    fn complete(&mut self, _fragment: &Fragment) -> Result<()> {
        Ok(())
    }
}
