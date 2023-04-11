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

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use arrow_schema::{Schema as ArrowSchema, SchemaRef};
use futures::Stream;

use crate::dataset::Dataset;
use crate::datatypes::Schema;
use crate::format::{Fragment, Manifest};
use crate::io::{FileReader, RecordBatchStream};
use crate::Result;

/// A Fragment of a Lance [`Dataset`].
///
/// The interface is similar to `pyarrow.dataset.Fragment`.
pub struct FileFragment<'a> {
    dataset: &'a Dataset,

    metadata: Fragment,

    manifest: Option<&'a Manifest>,
}

impl<'a> FileFragment<'a> {
    /// Creates a new FileFragment.
    pub fn new(dataset: &'a Dataset, metadata: Fragment, manifest: Option<&'a Manifest>) -> Self {
        Self {
            dataset,
            metadata,
            manifest,
        }
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

    pub async fn scan(&self, projection: &Schema) -> Result<FragmentRecordBatchStream> {
        let schema = Arc::new(ArrowSchema::from(projection));
        Ok(FragmentRecordBatchStream::new(schema))
    }
}

/// Scanning over a Fragment.
///
/// A Stream contains one or more [`FileReader`]s, each of which is responsible for separated columns
/// in the fragment.
pub struct FragmentRecordBatchStream {
    schema: SchemaRef,
    readers: Vec<FileReader>,
}

impl FragmentRecordBatchStream {
    pub fn new(schema: SchemaRef) -> Self {
        Self {
            schema,
            readers: vec![],
        }
    }
}

impl RecordBatchStream for FragmentRecordBatchStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Stream for FragmentRecordBatchStream {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        todo!()
    }
}

#[cfg(test)]
mod tests {

    #[tokio::test]
    async fn test_scan() {}
}
