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
use uuid::Uuid;

use crate::dataset::{Dataset, DATA_DIR};
use crate::datatypes::Schema;
use crate::format::{DataFile, Fragment};
use crate::io::{FileReader, FileWriter};
use crate::{Error, Result};

use super::scanner::Scanner;

/// A Fragment of a Lance [`Dataset`].
///
/// The interface is similar to `pyarrow.dataset.Fragment`.
#[derive(Debug, Clone)]
pub struct FileFragment {
    dataset: Arc<Dataset>,

    metadata: Fragment,
}

impl FileFragment {
    /// Creates a new FileFragment.
    pub fn new(dataset: Arc<Dataset>, metadata: Fragment) -> Self {
        Self { dataset, metadata }
    }

    pub fn dataset(&self) -> &Dataset {
        self.dataset.as_ref()
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
        // TODO: support open multiple data files.
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

    /// Create a new Writer for new columns.
    ///
    /// After it is called, this Fragment contains the metadata of the new DataFile,
    /// containing the columns, even the data has not written yet.
    ///
    /// Internal use only.
    pub async fn new_writer<'a>(&mut self, schema: &'a Schema) -> Result<FileWriter<'a>> {
        // Sanity check.
        //
        // To keep it simple, new schema must have no intersection with the existing schema.
        let existing_schema = self.dataset.schema();
        for field in schema.fields.iter() {
            // Just check the first level names.
            if existing_schema.field(&field.name).is_some() {
                return Err(Error::IO(format!(
                    "Append column: duplicated column {} already exists",
                    field.name
                )));
            }
        }

        let object_store = self.dataset.object_store.as_ref();
        let file_path = format!("{}.lance", Uuid::new_v4());
        self.metadata.add_file(&file_path, schema);

        let full_path = object_store
            .base_path()
            .child(DATA_DIR)
            .child(file_path.as_str());
        FileWriter::try_new(object_store, &full_path, schema).await
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

    use crate::{arrow::RecordBatchBuffer, dataset::WriteParams};

    use super::*;

    use arrow_array::{Int32Array, RecordBatchReader, StringArray};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use tempfile::tempdir;

    async fn create_dataset(test_uri: &str) -> Dataset {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, true),
            ArrowField::new("s", DataType::Utf8, true),
        ]));

        let batches = RecordBatchBuffer::new(
            (0..10)
                .map(|i| {
                    RecordBatch::try_new(
                        schema.clone(),
                        vec![
                            Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20)),
                            Arc::new(StringArray::from_iter_values(
                                (i * 20..(i + 1) * 20).map(|v| format!("s-{}", v)),
                            )),
                        ],
                    )
                    .unwrap()
                })
                .collect(),
        );

        let mut write_params = WriteParams::default();
        write_params.max_rows_per_file = 40;
        write_params.max_rows_per_group = 2;
        let mut batches: Box<dyn RecordBatchReader> = Box::new(batches);

        Dataset::write(&mut batches, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();

        dataset
    }

    #[tokio::test]
    async fn test_fragment_scan() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = create_dataset(test_uri).await;
        let fragment = &dataset.get_fragments()[2];
        let mut scanner = fragment.scan();
        let batches = scanner
            .with_row_id()
            .filter(" i  < 110")
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(batches.len(), 2);

        assert_eq!(
            batches[0].column_by_name("i").unwrap().as_ref(),
            &Int32Array::from_iter_values(80..100)
        );
        assert_eq!(
            batches[1].column_by_name("i").unwrap().as_ref(),
            &Int32Array::from_iter_values(100..110)
        );
    }

    #[tokio::test]
    async fn test_fragment_take() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = create_dataset(test_uri).await;
        let fragment = &dataset.get_fragments()[3];

        let batch = fragment
            .take(&[1, 2, 4, 5, 8], dataset.schema())
            .await
            .unwrap();
        assert_eq!(
            batch.column_by_name("i").unwrap().as_ref(),
            &Int32Array::from_iter_values(vec![121, 122, 124, 125, 128])
        );
    }

    #[tokio::test]
    async fn test_fragment_count() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = create_dataset(test_uri).await;
        let fragment = &dataset.get_fragments()[3];

        assert_eq!(fragment.count_rows().await.unwrap(), 40);
    }

    #[tokio::test]
    async fn test_append_columns() {}
}
