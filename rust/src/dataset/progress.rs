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
use object_store::path::Path;

use crate::format::Fragment;
use crate::io::ObjectStore;
use crate::Result;

/// Progress of writing a [Fragment].
#[async_trait]
pub trait WriteFragmentProgress: std::fmt::Debug + Sync + Send {
    /// Indicate the beginning of writing a [Fragment].
    async fn begin(&mut self, fragment: &Fragment) -> Result<()>;

    /// Complete writing a [Fragment].
    async fn complete(&mut self, fragment: &Fragment) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct NoopFragmentWriteProgress {}

impl NoopFragmentWriteProgress {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl WriteFragmentProgress for NoopFragmentWriteProgress {
    #[inline]
    async fn begin(&mut self, _fragment: &Fragment) -> Result<()> {
        Ok(())
    }

    #[inline]
    async fn complete(&mut self, _fragment: &Fragment) -> Result<()> {
        Ok(())
    }
}

/// Keep track of the progress of writing a [Fragment] to object store.
#[derive(Debug, Clone)]
pub struct FSFragmentWriteProgress {
    pub base_path: Path,
    pub object_store: ObjectStore,
}

impl FSFragmentWriteProgress {
    pub async fn try_new(uri: &str) -> Result<Self> {
        let (object_store, base_path) = ObjectStore::from_uri(uri).await?;
        Ok(Self {
            object_store,
            base_path,
        })
    }

    fn in_progress_file(&self, fragment: &Fragment) -> Path {
        self.base_path
            .child(format!("fragment_{}.inprogress", fragment.id))
    }

    fn fragment_file(&self, fragment: &Fragment) -> Path {
        self.base_path
            .child(format!("fragment_{}.json", fragment.id))
    }
}

#[async_trait]
impl WriteFragmentProgress for FSFragmentWriteProgress {
    async fn begin(&mut self, fragment: &Fragment) -> Result<()> {
        let in_progress_path = self.in_progress_file(fragment);
        self.object_store.put(&in_progress_path, &vec![]).await?;
        let fragment_file = self.fragment_file(fragment);
        self.object_store
            .put(&fragment_file, serde_json::to_string(fragment)?.as_bytes())
            .await?;
        Ok(())
    }

    async fn complete(&mut self, fragment: &Fragment) -> Result<()> {
        let in_progress_path = self.in_progress_file(fragment);
        self.object_store.delete(&in_progress_path).await?;
        Ok(())
    }
}
