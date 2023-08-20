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
use crate::io::ObjectStore;
use crate::Result;

use object_store::path::Path;

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
}

impl WriteFragmentProgress for FSFragmentWriteProgress {
    fn begin(&mut self, fragment: &Fragment) -> Result<()> {
        let marker_path = self.base_path.child(format!("{}.inprogress", fragment.id));
        self.object_store.put(marker, vec![])?;
        let path = format!("{}/{}.json", self.base_dir, fragment.id);
        std::fs::write(path, serde_json::to_string(fragment)?)?;
        Ok(())
    }

    fn complete(&mut self, fragment: &Fragment) -> Result<()> {
        let path = format!("{}/{}.json", self.base_dir, fragment.id);
        std::fs::remove_file(path)?;
        Ok(())
    }
}
