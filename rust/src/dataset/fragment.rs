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

use arrow_array::cast::as_primitive_array;
use arrow_array::{RecordBatch, RecordBatchReader, UInt64Array};
use futures::future::try_join_all;
use futures::{join, StreamExt, TryFutureExt, TryStreamExt};
use object_store::path::Path;
use uuid::Uuid;

use super::hash_joiner::HashJoiner;
use super::scanner::Scanner;
use super::updater::Updater;
use crate::arrow::*;
use crate::dataset::{Dataset, DATA_DIR};
use crate::datatypes::Schema;
use crate::format::Fragment;
use crate::io::deletion::{deletion_file_path, read_deletion_file, write_deletion_file};
use crate::io::{FileReader, FileWriter, ObjectStore, ReadBatchParams};
use crate::{Error, Result};

use super::WriteParams;

/// A Fragment of a Lance [`Dataset`].
///
/// The interface is modeled after `pyarrow.dataset.Fragment`.
#[derive(Debug, Clone)]
pub struct FileFragment {
    dataset: Arc<Dataset>,

    pub(super) metadata: Fragment,
}

impl FileFragment {
    /// Creates a new FileFragment.
    pub fn new(dataset: Arc<Dataset>, metadata: Fragment) -> Self {
        Self { dataset, metadata }
    }

    /// Create a new [`FileFragment`] from a [`RecordBatchReader`].
    ///
    /// This method can be used before a `Dataset` is created. For example,
    /// Fragments can be created distributed first, before a central machine to
    /// commit the dataset with these fragments.
    ///
    pub async fn create(
        dataset_uri: &str,
        id: usize,
        reader: &mut dyn RecordBatchReader,
        params: Option<WriteParams>,
    ) -> Result<Fragment> {
        let params = params.unwrap_or_default();

        let schema = Schema::try_from(reader.schema().as_ref())?;
        let (object_store, base_path) = ObjectStore::from_uri(dataset_uri).await?;
        let filename = format!("{}.lance", Uuid::new_v4());
        let fragment = Fragment::with_file(id as u64, &filename, &schema);

        let full_path = base_path.child(DATA_DIR).child(filename.clone());

        let mut writer = FileWriter::try_new(&object_store, &full_path, schema.clone()).await?;
        let mut buffer = RecordBatchBuffer::empty();

        for rst in reader {
            let batch = rst?; // TODO: close writer on Error?
            buffer.batches.push(batch);
            if buffer.num_rows() >= params.max_rows_per_group {
                let batches = buffer.finish()?;
                writer.write(&batches).await?;
                buffer = RecordBatchBuffer::empty();
            }
        }

        if buffer.num_rows() > 0 {
            let batches = buffer.finish()?;
            writer.write(&batches).await?;
        };

        // Params.max_rows_per_file is ignored in this case.
        writer.finish().await?;

        Ok(fragment)
    }

    pub fn dataset(&self) -> &Dataset {
        self.dataset.as_ref()
    }

    pub fn schema(&self) -> &Schema {
        self.dataset.schema()
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
            let data_file_schema = data_file.schema(full_schema);
            let schema_per_file = data_file_schema.intersection(projection)?;
            if !schema_per_file.fields.is_empty() {
                let path = self.dataset.data_dir().child(data_file.path.as_str());
                let reader = FileReader::try_new_with_fragment(
                    &self.dataset.object_store,
                    &path,
                    self.id() as u64,
                    Some(self.dataset.manifest.as_ref()),
                )
                .await?;
                let initialized_schema = reader.schema().project_by_schema(&schema_per_file)?;
                opened_files.push((reader, initialized_schema));
            }
        }

        if opened_files.is_empty() {
            return Err(Error::IO {
                message: format!(
                    "Does not find any data file for schema: {}\nfragment_id={}",
                    projection,
                    self.id()
                ),
            });
        }

        FragmentReader::try_new(self.id(), opened_files)
    }

    /// Count the rows in this fragment.
    pub async fn count_rows(&self) -> Result<usize> {
        let total_rows = self.fragment_length();

        let deletion_count = read_deletion_file(
            &self.dataset.base,
            &self.metadata,
            self.dataset.object_store(),
        )
        .map_ok(|v| v.map(|v| v.len()).unwrap_or_default());

        let (total_rows, deletion_count) =
            futures::future::try_join(total_rows, deletion_count).await?;

        Ok(total_rows - deletion_count)
    }

    /// Get the number of physical rows in the fragment. This includes deleted rows.
    ///
    /// If there are no deleted rows, this is equal to the number of rows in the
    /// fragment.
    pub async fn fragment_length(&self) -> Result<usize> {
        if self.metadata.files.is_empty() {
            return Err(Error::IO {
                message: format!("Fragment {} does not contain any data", self.id()),
            });
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

    /// Validate the fragment
    ///
    /// Verifies:
    /// * All data files exist and have the same length
    /// * Deletion file exists and has rowids in the correct range
    pub async fn validate(&self) -> Result<()> {
        let data_file_paths: Vec<Path> = self
            .metadata
            .files
            .iter()
            .map(|data_file| self.dataset.data_dir().child(data_file.path.as_str()))
            .collect::<Vec<_>>();
        let get_lengths = data_file_paths.iter().map(|path| {
            let reader = FileReader::try_new_with_fragment(
                &self.dataset.object_store,
                path,
                self.id() as u64,
                None,
            );
            reader.map_ok(|r| r.len())
        });
        let get_lengths = try_join_all(get_lengths);

        let deletion_vector = read_deletion_file(
            &self.dataset.base,
            &self.metadata,
            self.dataset.object_store(),
        );

        let (get_lengths, deletion_vector) = join!(get_lengths, deletion_vector);

        let get_lengths = get_lengths?;
        let expected_length = get_lengths.first().unwrap_or(&0);
        for (length, path) in get_lengths.iter().zip(data_file_paths.into_iter()) {
            if length != expected_length {
                return Err(Error::corrupt_file(path, "data file has incorrect length"));
            }
        }

        if let Some(deletion_vector) = deletion_vector? {
            for row_id in deletion_vector {
                if row_id >= *expected_length as u32 {
                    let deletion_file_meta = self.metadata.deletion_file.clone().unwrap();
                    return Err(Error::corrupt_file(
                        deletion_file_path(
                            &self.dataset.base,
                            self.metadata.id,
                            &deletion_file_meta,
                        ),
                        "deletion vector contains row id that is out of range",
                    ));
                }
            }
        }

        Ok(())
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

    /// Create an [`Updater`] to append new columns.
    pub async fn updater<T: AsRef<str>>(&self, columns: Option<&[T]>) -> Result<Updater> {
        let mut schema = self.dataset.schema().clone();
        if let Some(columns) = columns {
            schema = schema.project(columns)?;
        }
        let reader = self.open(&schema);
        let deletion_vector = read_deletion_file(
            &self.dataset.base,
            &self.metadata,
            self.dataset.object_store(),
        );
        let (reader, deletion_vector) = join!(reader, deletion_vector);
        let reader = reader?;
        let deletion_vector = deletion_vector?.unwrap_or_default();

        Ok(Updater::new(self.clone(), reader, deletion_vector))
    }

    pub(crate) async fn merge(
        mut self,
        join_column: &str,
        joiner: &HashJoiner,
        full_schema: &Schema,
    ) -> Result<Self> {
        let mut scanner = self.scan();
        scanner.project(&[join_column])?;

        let mut batch_stream = scanner
            .try_into_stream()
            .await?
            .and_then(|batch| joiner.collect(batch.column(0).clone()))
            .boxed();

        let file_schema = full_schema.project_by_schema(joiner.out_schema().as_ref())?;

        let filename = format!("{}.lance", Uuid::new_v4());
        let full_path = self.dataset.data_dir().child(filename.clone());
        let mut writer =
            FileWriter::try_new(&self.dataset.object_store, &full_path, file_schema.clone())
                .await?;

        while let Some(batch) = batch_stream.try_next().await? {
            writer.write(&[batch]).await?;
        }

        writer.finish().await?;

        self.metadata.add_file(&filename, &file_schema);

        Ok(self)
    }

    /// Delete rows from the fragment.
    ///
    /// If all rows are deleted, returns `Ok(None)`. Otherwise, returns a new
    /// fragment with the updated deletion vector. This must be persisted to
    /// the manifest.
    pub(crate) async fn delete(mut self, predicate: &str) -> Result<Option<Self>> {
        // Load existing deletion vector
        let mut deletion_vector = read_deletion_file(
            &self.dataset.base,
            &self.metadata,
            self.dataset.object_store(),
        )
        .await?
        .unwrap_or_default();

        // scan with predicate and row ids
        let mut scanner = self.scan();
        scanner
            .with_row_id()
            .filter(predicate)?
            .project::<&str>(&[])?;

        // As we get row ids, add them into our deletion vector
        scanner
            .try_into_stream()
            .await?
            .try_for_each(|batch| {
                let array = batch["_rowid"].clone();
                let int_array: &UInt64Array = as_primitive_array(array.as_ref());

                // _row_id is global, not within fragment level. The high bits
                // are the fragment_id, the low bits are the row_id within the
                // fragment.
                let local_row_ids = int_array.iter().map(|v| v.unwrap() as u32);

                deletion_vector.extend(local_row_ids);
                futures::future::ready(Ok(()))
            })
            .await?;

        // TODO: could we keep the number of rows in memory when we first get
        // the fragment metadata?
        let fragment_length = self.fragment_length().await?;
        if deletion_vector.len() == fragment_length
            && deletion_vector.contains_range(0..fragment_length as u32)
        {
            return Ok(None);
        } else if deletion_vector.len() >= fragment_length {
            let dv_len = deletion_vector.len();
            let examples: Vec<u32> = deletion_vector
                .into_iter()
                .filter(|x| *x >= fragment_length as u32)
                .take(5)
                .collect();
            return Err(Error::Internal {
                message: format!(
                    "Deletion vector includes rows that aren't in the fragment. \
                Fragment length: {}; Deletion vector length: {}; \
                Examples: {:?}",
                    fragment_length, dv_len, examples
                ),
            });
        }

        self.metadata.deletion_file = write_deletion_file(
            &self.dataset.base,
            self.metadata.id,
            self.dataset.version().version,
            &deletion_vector,
            self.dataset.object_store(),
        )
        .await?;

        Ok(Some(self))
    }
}

impl From<FileFragment> for Fragment {
    fn from(fragment: FileFragment) -> Self {
        fragment.metadata
    }
}

/// [`FragmentReader`] is an abstract reader for a [`FileFragment`].
///
/// It opens the data files that contains the columns of the projection schema, and
/// reconstruct the RecordBatch from columns read from each data file.
pub struct FragmentReader {
    /// Readers and schema of each opened data file.
    readers: Vec<(FileReader, Schema)>,

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
        return Err(Error::IO {
            message: "Cannot merge empty batches".to_string(),
        });
    }

    let mut merged = batches[0].clone();
    for batch in batches.iter() {
        merged = merged.merge(batch)?;
    }
    Ok(merged)
}

impl FragmentReader {
    fn try_new(fragment_id: usize, readers: Vec<(FileReader, Schema)>) -> Result<Self> {
        if readers.is_empty() {
            return Err(Error::IO {
                message: "Cannot create FragmentReader with zero readers".to_string(),
            });
        }

        let num_batches = readers[0].0.num_batches();
        if !readers.iter().all(|r| r.0.num_batches() == num_batches) {
            return Err(Error::IO {
                message:
                    "Cannot create FragmentReader from data files with different number of batches"
                        .to_string(),
            });
        }
        Ok(Self {
            readers,
            fragment_id,
        })
    }

    pub(crate) fn with_row_id(&mut self) -> &mut Self {
        self.readers[0].0.with_row_id(true);
        self
    }

    pub(crate) fn num_batches(&self) -> usize {
        self.readers[0].0.num_batches()
    }

    pub(crate) fn num_rows_in_batch(&self, batch_id: usize) -> usize {
        self.readers[0].0.num_rows_in_batch(batch_id as i32)
    }

    pub(crate) async fn read_batch(
        &self,
        batch_id: usize,
        params: impl Into<ReadBatchParams> + Clone,
    ) -> Result<RecordBatch> {
        // TODO: use tokio::async buffer to make parallel reads.
        let mut batches = vec![];
        for (reader, schema) in self.readers.iter() {
            let batch = reader
                .read_batch(batch_id as i32, params.clone(), schema)
                .await?;
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
            let batch = reader.take(indices, schema).await?;
            batches.push(batch);
        }

        merge_batches(&batches)
    }
}

#[cfg(test)]
mod tests {

    use arrow_arith::arithmetic::multiply_scalar;
    use arrow_array::{cast::AsArray, ArrayRef, Int32Array, RecordBatchReader, StringArray};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
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

        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 2,
            ..Default::default()
        };
        let mut batches: Box<dyn RecordBatchReader> = Box::new(batches);

        Dataset::write(&mut batches, test_uri, Some(write_params))
            .await
            .unwrap();

        Dataset::open(test_uri).await.unwrap()
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
        let mut dataset = create_dataset(test_uri).await;
        let fragment = dataset
            .get_fragments()
            .into_iter()
            .find(|f| f.id() == 3)
            .unwrap();

        let batch = fragment
            .take(&[1, 2, 4, 5, 8], dataset.schema())
            .await
            .unwrap();
        assert_eq!(
            batch.column_by_name("i").unwrap().as_ref(),
            &Int32Array::from(vec![121, 122, 124, 125, 128])
        );

        dataset.delete("i in (122, 124)").await.unwrap();
        dataset.validate().await.unwrap();

        // Cannot get rows 2 and 4 anymore
        let fragment = dataset
            .get_fragments()
            .into_iter()
            .find(|f| f.id() == 3)
            .unwrap();
        assert!(fragment.metadata().deletion_file.is_some());
        let batch = fragment
            .take(&[1, 2, 4, 5, 8], dataset.schema())
            .await
            .unwrap();
        assert_eq!(
            batch.column_by_name("i").unwrap().as_ref(),
            &Int32Array::from(vec![121, 125, 128])
        );
    }

    #[tokio::test]
    async fn test_fragment_count() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = create_dataset(test_uri).await;
        let fragment = dataset.get_fragments().pop().unwrap();

        assert_eq!(fragment.count_rows().await.unwrap(), 40);
        assert_eq!(fragment.fragment_length().await.unwrap(), 40);

        let fragment = fragment
            .delete("i >= 160 and i <= 172")
            .await
            .unwrap()
            .unwrap();

        fragment.validate().await.unwrap();

        assert_eq!(fragment.count_rows().await.unwrap(), 27);
        assert_eq!(fragment.fragment_length().await.unwrap(), 40);
    }

    #[tokio::test]
    async fn test_append_new_columns() {
        for with_delete in [true, false] {
            let test_dir = tempdir().unwrap();
            let test_uri = test_dir.path().to_str().unwrap();
            let mut dataset = create_dataset(test_uri).await;
            dataset.validate().await.unwrap();
            assert_eq!(dataset.count_rows().await.unwrap(), 200);

            if with_delete {
                dataset.delete("i >= 15 and i < 20").await.unwrap();
                dataset.validate().await.unwrap();
                assert_eq!(dataset.count_rows().await.unwrap(), 195);
            }

            let fragment = &mut dataset.get_fragment(0).unwrap();
            let mut updater = fragment.updater(Some(&["i"])).await.unwrap();
            let new_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "double_i",
                DataType::Int32,
                true,
            )]));
            while let Some(batch) = updater.next().await.unwrap() {
                let input_col = batch.column_by_name("i").unwrap();
                let result_col: Int32Array = multiply_scalar(input_col.as_primitive(), 2).unwrap();
                let batch = RecordBatch::try_new(
                    new_schema.clone(),
                    vec![Arc::new(result_col) as ArrayRef],
                )
                .unwrap();
                updater.update(batch).await.unwrap();
            }
            let new_fragment = updater.finish().await.unwrap();

            assert_eq!(new_fragment.files.len(), 2);

            // Scan again
            let full_schema = dataset.schema().merge(new_schema.as_ref()).unwrap();
            let before_version = dataset.version().version;
            let dataset = Dataset::commit(
                test_uri,
                &full_schema,
                &[new_fragment],
                crate::dataset::WriteMode::Create,
            )
            .await
            .unwrap();

            // We only kept the first fragment of 40 rows
            assert_eq!(
                dataset.count_rows().await.unwrap(),
                if with_delete { 35 } else { 40 }
            );
            assert_eq!(dataset.version().version, before_version + 1);
            dataset.validate().await.unwrap();
            let new_projection = full_schema.project(&["i", "double_i"]).unwrap();

            let stream = dataset
                .scan()
                .project(&["i", "double_i"])
                .unwrap()
                .try_into_stream()
                .await
                .unwrap();
            let batches = stream.try_collect::<Vec<_>>().await.unwrap();

            assert_eq!(batches[0].schema().as_ref(), &(&new_projection).into());
            let max_value_in_batch = if with_delete { 15 } else { 20 };
            let expected_batch = RecordBatch::try_new(
                Arc::new(ArrowSchema::new(vec![
                    ArrowField::new("i", DataType::Int32, true),
                    ArrowField::new("double_i", DataType::Int32, true),
                ])),
                vec![
                    Arc::new(Int32Array::from_iter_values(0..max_value_in_batch)),
                    Arc::new(Int32Array::from_iter_values(
                        (0..(2 * max_value_in_batch)).step_by(2),
                    )),
                ],
            )
            .unwrap();
            assert_eq!(batches[0], expected_batch);
        }
    }
}
