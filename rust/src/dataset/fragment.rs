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

//! Wraps a Fragment of the dataset.

use std::sync::Arc;

use arrow_array::RecordBatch;

use crate::dataset::Dataset;
use crate::datatypes::Schema;
use crate::format::Fragment;
use crate::io::FileReader;
use crate::Result;

use super::scanner::Scanner;

/// A Fragment of a Lance [`Dataset`].
///
/// The interface is similar to `pyarrow.dataset.Fragment`.
pub struct FileFragment {
    dataset: Arc<Dataset>,

    metadata: Fragment,
}

impl FileFragment {
    /// Creates a new FileFragment.
    pub fn new(dataset: Arc<Dataset>, metadata: Fragment) -> Self {
        Self { dataset, metadata }
    }

    /// Returns the fragment's metadata.
    pub fn metadata(&self) -> &Fragment {
        &self.metadata
    }

    /// The id of this [`FileFragment`].
    pub fn id(&self) -> usize {
        self.metadata.id as usize
    }

    async fn do_open(&self, paths: &[&str]) -> Result<FileReader> {
        // TODO: support open multiple data failes.
        let path = self.dataset.data_dir().child(paths[0]);
        let reader = FileReader::try_new_with_fragment(
            &self.dataset.object_store,
            &path,
            self.id() as u64,
            None,
        )
        .await?;
        Ok(reader)
    }

    /// Count the rows in this fragment.
    ///
    pub async fn count_rows(&self) -> Result<usize> {
        let reader = self
            .do_open(&[self.metadata.files[0].path.as_str()])
            .await?;
        Ok(reader.len())
    }

    /// Take rows from this fragment.
    pub async fn take(&self, indices: &[u32], projection: &Schema) -> Result<RecordBatch> {
        let reader = self
            .do_open(&[self.metadata.files[0].path.as_str()])
            .await?;
        reader.take(indices, projection).await
    }

    /// Scan this [`FileFragment`].
    ///
    /// See [`Dataset::scan`].
    pub fn scan(&self) -> Scanner {
        Scanner::from_fragment(self.dataset.clone(), self.metadata.clone())
    }
}

#[cfg(test)]
mod tests {

    #[tokio::test]
    async fn test_scan() {}

    #[tokio::test]
    async fn test_fragment_take() {}

    #[tokio::test]
    async fn test_fragment_count() {}
}
