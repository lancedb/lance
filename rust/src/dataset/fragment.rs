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

use std::ops::Range;
use std::sync::Arc;

use arrow_array::RecordBatch;
use uuid::Uuid;

use crate::arrow::*;
use crate::dataset::{Dataset, DATA_DIR};
use crate::datatypes::Schema;
use crate::format::Fragment;
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

    /// Open all the data files as part of the projection schema.
    ///
    /// Parameters
    /// - projection: The projection schema.
    pub async fn open(&self, projection: &Schema) -> Result<FragmentReader> {
        let full_schema = self.dataset.schema();

        let mut opened_files = vec![];
        for data_file in self.metadata.files.iter() {
            let data_file_schema = data_file.schema(&full_schema);
            let schema_per_file = data_file_schema.intersection(projection)?;
            if !schema_per_file.fields.is_empty() {
                let path = self.dataset.data_dir().child(data_file.path.as_str());
                let reader = FileReader::try_new_with_fragment(
                    &self.dataset.object_store,
                    &path,
                    self.id() as u64,
                    None,
                )
                .await?;
                opened_files.push((reader, schema_per_file));
            }
        }

        if opened_files.is_empty() {
            return Err(Error::IO(format!(
                "Does not find any data file for schema: {}\nfragment_id={}",
                projection,
                self.id()
            )));
        }

        FragmentReader::try_new(
            self.id(),
            opened_files,
            projection.clone(),
        )
    }

    /// Count the rows in this fragment.
    ///
    pub async fn count_rows(&self) -> Result<usize> {
        if self.metadata.files.is_empty() {
            return Err(Error::IO(format!(
                "Fragment {} does not contain any data",
                self.id()
            )));
        };

        // Just open any file. All of them should have same size.
        let path = self
            .dataset
            .data_dir()
            .child(self.metadata.files[0].path.as_str());
        let reader = FileReader::try_new_with_fragment(
            &self.dataset.object_store,
            &path,
            self.id() as u64,
            None,
        )
        .await?;

        Ok(reader.len())
    }

    /// Create a new Writer for new columns.
    ///
    /// After it is called, this Fragment contains the metadata of the new DataFile,
    /// containing the columns, even the data has not written yet.
    ///
    /// It is the caller's responsibility to close the [`FileWriter`].
    ///
    /// Internal use only.
    pub(super) async fn new_writer<'a>(&mut self, schema: &'a Schema) -> Result<FileWriter<'a>> {
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
        let reader = self.open(projection).await?;
        reader.take(indices).await
    }

    /// Scan this [`FileFragment`].
    ///
    /// See [`Dataset::scan`].
    pub fn scan(&self) -> Scanner {
        Scanner::from_fragment(self.dataset.clone(), self.metadata.clone())
    }
}

/// [`FragmentReader`] is an abstract reader for a [`FileFragment`].
///
/// It opens the data files that contains the columns of the projection schema, and
/// reconstruct the RecordBatch from columns read from each data file.
pub struct FragmentReader {
    /// Readers and schema of each opened data file.
    readers: Vec<(FileReader, Schema)>,

    /// Projection Schema
    schema: Schema,

    /// ID of the fragment
    fragment_id: usize,
}

impl std::fmt::Display for FragmentReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FragmentReader(id={})", self.fragment_id)
    }
}

fn merge_batches(batches: &[RecordBatch]) -> Result<RecordBatch> {
    if batches.is_empty() {
        return Err(Error::IO("Cannot merge empty batches".to_string()));
    }

    let mut merged = batches[0].clone();
    for i in 1..batches.len() {
        merged = merged.merge(&batches[i])?;
    }
    Ok(merged)
}

impl FragmentReader {
    fn try_new(
        fragment_id: usize,
        readers: Vec<(FileReader, Schema)>,
        projection: Schema,
    ) -> Result<Self> {
        if readers.is_empty() {
            return Err(Error::IO(
                "Cannot create FragmentReader with zero readers".to_string(),
            ));
        }

        let num_batches = readers[0].0.num_batches();
        if !readers.iter().all(|r| r.0.num_batches() == num_batches) {
            return Err(Error::IO(
                "Cannot create FragmentReader from data files with different number of batches".to_string(),
            ));
        }

        Ok(Self {
            readers,
            schema: projection,
            fragment_id,
        })
    }

    pub(crate) fn schema(&self) -> &Schema {
        &self.schema
    }

    pub(crate) fn with_row_id(&mut self) -> &mut Self {
        self.readers[0].0.with_row_id(true);
        self
    }

    pub(crate) fn len(&self) -> usize {
        self.readers[0].0.len()
    }

    pub async fn read_batch(&self, batch_id: usize) -> Result<RecordBatch> {
        // TODO: use tokio::async buffer to make parallel reads.
        let mut batches = vec![];
        for (reader, schema) in self.readers.iter() {
            let batch = reader.read_batch(batch_id as i32, .., schema).await?;
            batches.push(batch);
        }
        merge_batches(&batches)
    }

    pub async fn read_range(&self, range: Range<usize>) -> Result<RecordBatch> {
        // TODO: Putting this loop in async blocks cause lifetime issues.
        // We need to fix
        let mut batches = vec![];
        for (reader, schema) in self.readers.iter() {
            let batch = reader.read_range(range.start..range.end, schema).await?;
            batches.push(batch);
        }

        merge_batches(&batches)
    }

    /// Take rows from this fragment.
    pub async fn take(&self, indices: &[u32]) -> Result<RecordBatch> {
        let mut batches = vec![];

        // TODO: Putting this loop in async blocks cause lifetime issues.
        // We need to fix
        for (reader, schema) in self.readers.iter() {
            let batch = reader.take(indices, &schema).await?;
            batches.push(batch);
        }

        merge_batches(&batches)
    }
}

#[cfg(test)]
mod tests {

    use arrow_arith::arithmetic::multiply_scalar;
    use arrow_array::{cast::AsArray, ArrayRef, Int32Array, RecordBatchReader, StringArray};
    use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use tempfile::tempdir;

    use super::*;
    use crate::{arrow::RecordBatchBuffer, dataset::WriteParams};

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
    async fn test_append_new_columns() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = create_dataset(test_uri).await;
        println!("Test URI: {:?}", test_uri);

        let fragment = &mut dataset.get_fragments()[0];
        let new_arrow_schema = Arc::new(ArrowSchema::new(Fields::from(vec![ArrowField::new(
            "double_i",
            DataType::Int32,
            true,
        )])));
        let schema: Schema = new_arrow_schema.as_ref().try_into().unwrap();
        let mut writer = fragment.new_writer(&schema).await.unwrap();
        let mut scanner = fragment
            .scan()
            .batch_size(50)
            .project(&["i"])
            .unwrap()
            .try_into_stream()
            .await
            .unwrap();
        while let Some(batch) = scanner.try_next().await.unwrap() {
            let input_col = batch.column_by_name("i").unwrap();
            let result_col: Int32Array = multiply_scalar(&input_col.as_primitive(), 2).unwrap();
            let batch = RecordBatch::try_new(
                new_arrow_schema.clone(),
                vec![Arc::new(result_col) as ArrayRef],
            )
            .unwrap();
            writer.write(&[&batch]).await.unwrap();
        }

        writer.finish().await.unwrap();

        assert_eq!(fragment.metadata.files.len(), 2);

        // Scan again
        let full_schema = dataset.schema().merge(&schema);
        let new_project = full_schema.project(&["i", "double_i"]).unwrap();
        let reader = fragment.open(&new_project).await.unwrap();
        let batch = reader.read_range(12..28).await.unwrap();

        assert_eq!(batch.schema().as_ref(), &(&new_project).into());
        println!("Batches: {:?}", batch);
    }
}
