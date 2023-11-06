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

use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::cast::as_primitive_array;
use arrow_array::{RecordBatch, RecordBatchReader, UInt64Array};
use futures::future::try_join_all;
use futures::stream::BoxStream;
use futures::{join, StreamExt, TryFutureExt, TryStreamExt};
use lance_core::{
    datatypes::Schema,
    io::{
        deletion::{deletion_file_path, read_deletion_file, write_deletion_file, DeletionVector},
        object_store::ObjectStore,
        FileReader, FileWriter, ReadBatchParams,
    },
    Error, Result, ROW_ID,
};
use object_store::path::Path;
use snafu::{location, Location};
use uuid::Uuid;

use super::chunker::chunk_stream;
use super::hash_joiner::HashJoiner;
use super::scanner::Scanner;
use super::updater::Updater;
use super::write::reader_to_stream;
use super::WriteParams;
use crate::arrow::*;
use crate::dataset::{Dataset, DATA_DIR};
use crate::format::Fragment;

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
        reader: impl RecordBatchReader + Send + 'static,
        params: Option<WriteParams>,
    ) -> Result<Fragment> {
        let params = params.unwrap_or_default();
        let progress = params.progress.as_ref();

        let reader = Box::new(reader);
        let (stream, schema) = reader_to_stream(reader)?;

        if schema.fields.is_empty() {
            return Err(Error::invalid_input(
                "Cannot write with an empty schema.",
                location!(),
            ));
        }

        let (object_store, base_path) = ObjectStore::from_uri(dataset_uri).await?;
        let filename = format!("{}.lance", Uuid::new_v4());
        let mut fragment = Fragment::with_file(id as u64, &filename, &schema, 0);
        let full_path = base_path.child(DATA_DIR).child(filename.clone());
        let mut writer = FileWriter::try_new(
            &object_store,
            &full_path,
            schema.clone(),
            &Default::default(),
        )
        .await?;

        progress.begin(&fragment, writer.multipart_id()).await?;

        let mut buffered_reader = chunk_stream(stream, params.max_rows_per_group);
        while let Some(batched_chunk) = buffered_reader.next().await {
            let batch = batched_chunk?;
            writer.write(&batch).await?;
        }

        fragment.physical_rows = writer.finish().await?;

        progress.complete(&fragment).await?;

        Ok(fragment)
    }

    pub async fn create_from_file(
        filename: &str,
        schema: &Schema,
        fragment_id: usize,
        physical_rows: Option<usize>,
    ) -> Result<Fragment> {
        let fragment = Fragment::with_file(
            fragment_id as u64,
            filename,
            schema,
            physical_rows.unwrap_or_default(),
        );
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
                    Some(&self.dataset.session.file_metadata_cache),
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
                location: location!(),
            });
        }

        FragmentReader::try_new(self.id(), opened_files)
    }

    /// Count the rows in this fragment.
    pub async fn count_rows(&self) -> Result<usize> {
        let total_rows = self.physical_rows();

        let deletion_count = self.count_deletions();

        let (total_rows, deletion_count) =
            futures::future::try_join(total_rows, deletion_count).await?;

        Ok(total_rows - deletion_count)
    }

    /// Get the number of rows that have been deleted in this fragment.
    pub async fn count_deletions(&self) -> Result<usize> {
        match &self.metadata().deletion_file {
            Some(f) if f.num_deleted_rows > 0 => Ok(f.num_deleted_rows),
            _ => {
                read_deletion_file(
                    &self.dataset.base,
                    &self.metadata,
                    self.dataset.object_store(),
                )
                .map_ok(|v| v.map(|v| v.len()).unwrap_or_default())
                .await
            }
        }
    }

    /// Get the number of physical rows in the fragment. This includes deleted rows.
    ///
    /// If there are no deleted rows, this is equal to the number of rows in the
    /// fragment.
    pub async fn physical_rows(&self) -> Result<usize> {
        if self.metadata.files.is_empty() {
            return Err(Error::IO {
                message: format!("Fragment {} does not contain any data", self.id()),
                location: location!(),
            });
        };

        if self.metadata.physical_rows > 0 {
            return Ok(self.metadata.physical_rows);
        }

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
            Some(&self.dataset.session.file_metadata_cache),
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
                Some(self.dataset.manifest.as_ref()),
                Some(&self.dataset.session.file_metadata_cache),
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
                return Err(Error::corrupt_file(
                    path,
                    format!(
                        "data file has incorrect length. Expected: {} Got: {}",
                        expected_length, length
                    ),
                    location!(),
                ));
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
                        format!("deletion vector contains row id that is out of range. Row id: {} Fragment length: {}", row_id, expected_length),
                        location!(),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Take rows from this fragment based on the offset in the file.
    ///
    /// This will always return the same number of rows as the input indices.
    /// If indices are out-of-bounds, this will return an error.
    pub async fn take(&self, indices: &[u32], projection: &Schema) -> Result<RecordBatch> {
        // Re-map the indices to row ids using the deletion vector
        let deletion_vector = self.get_deletion_vector().await?;
        let row_ids = if let Some(deletion_vector) = deletion_vector {
            // Naive case is O(N*M), where N = indices.len() and M = deletion_vector.len()
            // We can do better by sorting the deletion vector and using binary search
            // This is O(N * log M + M log M).
            let mut sorted_deleted_ids = deletion_vector
                .as_ref()
                .clone()
                .into_iter()
                .collect::<Vec<_>>();
            sorted_deleted_ids.sort();

            let mut row_ids = indices.to_vec();
            for row_id in row_ids.iter_mut() {
                // We find the number of deleted rows that are less than each row
                // index, and that becomes the initial offset. We increment the
                // index by that amount, plus the number of deleted row ids we
                // encounter along the way. So for example, if deleted rows are
                // [2, 3, 5] and we want row 4, we need to advanced by 2 (since
                // 2 and 3 are less than 4). That puts us at row 6, but since
                // we passed row 5, we need to advance by 1 more, giving a final
                // row id of 7.
                let mut new_row_id = *row_id;
                let offset = sorted_deleted_ids.partition_point(|v| *v <= new_row_id);

                let mut deletion_i = offset;
                let mut i = 0;
                while i < offset {
                    // Advance the row id
                    new_row_id += 1;
                    while deletion_i < sorted_deleted_ids.len()
                        && sorted_deleted_ids[deletion_i] == new_row_id
                    {
                        // If we encounter a deleted row, we need to advance
                        // again.
                        deletion_i += 1;
                        new_row_id += 1;
                    }
                    i += 1;
                }

                *row_id = new_row_id;
            }

            Cow::Owned(row_ids)
        } else {
            Cow::Borrowed(indices)
        };

        // Then call take rows
        self.take_rows(&row_ids, projection, false).await
    }

    /// Get the deletion vector for this fragment, using the cache if available.
    pub(crate) async fn get_deletion_vector(&self) -> Result<Option<Arc<DeletionVector>>> {
        let Some(deletion_file) = self.metadata.deletion_file.as_ref() else {
            return Ok(None);
        };

        let cache = &self.dataset.session.file_metadata_cache;
        let path = deletion_file_path(&self.dataset.base, self.metadata.id, deletion_file);
        if let Some(deletion_vector) = cache.get::<DeletionVector>(&path) {
            Ok(Some(deletion_vector))
        } else {
            let deletion_vector = read_deletion_file(
                &self.dataset.base,
                &self.metadata,
                self.dataset.object_store(),
            )
            .await?;
            match deletion_vector {
                Some(deletion_vector) => {
                    let deletion_vector = Arc::new(deletion_vector);
                    cache.insert(path, deletion_vector.clone());
                    Ok(Some(deletion_vector))
                }
                None => Ok(None),
            }
        }
    }

    /// Take rows based on internal local row ids
    ///
    /// If the row ids are out-of-bounds, this will return an error. But if the
    /// row id is marked deleted, it will be ignored. Thus, the number of rows
    /// returned may be less than the number of row ids provided.
    ///
    /// To recover the original row ids from the returned RecordBatch, set the
    /// `with_row_id` parameter to true. This will add a column named `_row_id`
    /// to the RecordBatch at the end.
    pub(crate) async fn take_rows(
        &self,
        row_ids: &[u32],
        projection: &Schema,
        with_row_id: bool,
    ) -> Result<RecordBatch> {
        let mut reader = self.open(projection).await?;
        if with_row_id {
            reader.with_row_id();
        }
        if row_ids.len() > 1 && Self::row_ids_contiguous(row_ids) {
            let range = (row_ids[0] as usize)..(row_ids[row_ids.len() - 1] as usize + 1);
            reader.read_range(range).await
        } else {
            reader.take(row_ids).await
        }
    }

    fn row_ids_contiguous(row_ids: &[u32]) -> bool {
        if row_ids.is_empty() {
            return false;
        }

        let mut last_id = row_ids[0];

        for id in row_ids.iter().skip(1) {
            if *id != last_id + 1 {
                return false;
            }
            last_id = *id;
        }

        true
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

    pub(crate) async fn merge(mut self, join_column: &str, joiner: &HashJoiner) -> Result<Self> {
        let mut updater = self.updater(Some(&[join_column])).await?;

        while let Some(batch) = updater.next().await? {
            let batch = joiner.collect(batch[join_column].clone()).await?;
            updater.update(batch).await?;
        }

        self.metadata = updater.finish().await?;

        Ok(self)
    }

    /// Delete rows from the fragment.
    ///
    /// If all rows are deleted, returns `Ok(None)`. Otherwise, returns a new
    /// fragment with the updated deletion vector. This must be persisted to
    /// the manifest.
    pub async fn delete(mut self, predicate: &str) -> Result<Option<Self>> {
        // Load existing deletion vector
        let mut deletion_vector = read_deletion_file(
            &self.dataset.base,
            &self.metadata,
            self.dataset.object_store(),
        )
        .await?
        .unwrap_or_default();

        let starting_length = deletion_vector.len();

        // scan with predicate and row ids
        let mut scanner = self.scan();

        // if predicate is `true`, delete the whole fragment
        // else if predicate is `false`, filter the predicate
        let predicate_lower = predicate.trim().to_lowercase();
        if predicate_lower == "true" {
            return Ok(None);
        } else if predicate_lower == "false" {
            return Ok(Some(self));
        }

        scanner
            .with_row_id()
            .filter(predicate)?
            .project::<&str>(&[])?;

        // As we get row ids, add them into our deletion vector
        scanner
            .try_into_stream()
            .await?
            .try_for_each(|batch| {
                let array = batch[ROW_ID].clone();
                let int_array: &UInt64Array = as_primitive_array(array.as_ref());

                // _row_id is global, not within fragment level. The high bits
                // are the fragment_id, the low bits are the row_id within the
                // fragment.
                let local_row_ids = int_array.iter().map(|v| v.unwrap() as u32);

                deletion_vector.extend(local_row_ids);
                futures::future::ready(Ok(()))
            })
            .await?;

        // If we haven't deleted any additional rows, we can return the fragment as-is.
        if deletion_vector.len() == starting_length {
            return Ok(Some(self));
        }

        // TODO: could we keep the number of rows in memory when we first get
        // the fragment metadata?
        let physical_rows = self.physical_rows().await?;
        if deletion_vector.len() == physical_rows
            && deletion_vector.contains_range(0..physical_rows as u32)
        {
            return Ok(None);
        } else if deletion_vector.len() >= physical_rows {
            let dv_len = deletion_vector.len();
            let examples: Vec<u32> = deletion_vector
                .into_iter()
                .filter(|x| *x >= physical_rows as u32)
                .take(5)
                .collect();
            return Err(Error::Internal {
                message: format!(
                    "Deletion vector includes rows that aren't in the fragment. \
                Num physical rows {}; Deletion vector length: {}; \
                Examples: {:?}",
                    physical_rows, dv_len, examples
                ),
                location: location!(),
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
            location: location!(),
        });
    }

    let mut merged = batches[0].clone();
    for batch in batches.iter().skip(1) {
        merged = merged.merge(batch)?;
    }
    Ok(merged)
}

impl FragmentReader {
    fn try_new(fragment_id: usize, readers: Vec<(FileReader, Schema)>) -> Result<Self> {
        if readers.is_empty() {
            return Err(Error::IO {
                message: "Cannot create FragmentReader with zero readers".to_string(),
                location: location!(),
            });
        }

        let num_batches = readers[0].0.num_batches();
        if !readers.iter().all(|r| r.0.num_batches() == num_batches) {
            return Err(Error::IO {
                message:
                    "Cannot create FragmentReader from data files with different number of batches"
                        .to_string(),
                location: location!(),
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

    pub(crate) fn with_make_deletions_null(&mut self) -> &mut Self {
        for (reader, _) in self.readers.iter_mut() {
            reader.with_make_deletions_null(true);
        }
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
        // Boxed to avoid lifetime issue.
        let stream: BoxStream<_> = futures::stream::iter(&self.readers)
            .map(|(reader, schema)| reader.take(indices, schema))
            .buffered(num_cpus::get())
            .boxed();
        let batches: Vec<RecordBatch> = stream.try_collect::<Vec<_>>().await?;

        merge_batches(&batches)
    }
}

#[cfg(test)]
mod tests {

    use arrow_arith::numeric::mul;
    use arrow_array::{ArrayRef, Int32Array, RecordBatchIterator, StringArray};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use arrow_select::concat::concat_batches;
    use futures::TryStreamExt;
    use tempfile::tempdir;

    use super::*;
    use crate::dataset::transaction::Operation;
    use crate::dataset::{WriteParams, ROW_ID};

    async fn create_dataset(test_uri: &str) -> Dataset {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, true),
            ArrowField::new("s", DataType::Utf8, true),
        ]));

        let batches: Vec<RecordBatch> = (0..10)
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
            .collect();

        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, test_uri, Some(write_params))
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
            .filter(" i  < 105")
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(batches.len(), 3);

        assert_eq!(
            batches[0].column_by_name("i").unwrap().as_ref(),
            &Int32Array::from_iter_values(80..90)
        );
        assert_eq!(
            batches[1].column_by_name("i").unwrap().as_ref(),
            &Int32Array::from_iter_values(90..100)
        );
        assert_eq!(
            batches[2].column_by_name("i").unwrap().as_ref(),
            &Int32Array::from_iter_values(100..105)
        );
    }

    #[tokio::test]
    async fn test_fragment_scan_deletions() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = create_dataset(test_uri).await;
        dataset.delete("i >= 0 and i < 15").await.unwrap();

        let fragment = &dataset.get_fragments()[0];
        let mut reader = fragment.open(dataset.schema()).await.unwrap();
        reader.with_make_deletions_null();
        reader.with_row_id();

        // Since the first batch is all deleted, it will return an empty batch.
        let batch1 = reader.read_batch(0, ..).await.unwrap();
        assert_eq!(batch1.num_rows(), 0);

        // The second batch is partially deleted, so the deleted rows will be
        // marked null with null row ids.
        let batch2 = reader.read_batch(1, ..).await.unwrap();
        assert_eq!(
            batch2.column_by_name(ROW_ID).unwrap().as_ref(),
            &UInt64Array::from_iter((10..20).map(|v| if v < 15 { None } else { Some(v) }))
        );

        // The final batch is not deleted, so it will be returned as-is.
        let batch3 = reader.read_batch(2, ..).await.unwrap();
        assert_eq!(
            batch3.column_by_name(ROW_ID).unwrap().as_ref(),
            &UInt64Array::from_iter_values(20..30)
        );
    }

    #[tokio::test]
    async fn test_fragment_take_indices() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = create_dataset(test_uri).await;
        let fragment = dataset
            .get_fragments()
            .into_iter()
            .find(|f| f.id() == 3)
            .unwrap();

        // Repeated indices are repeated in result.
        let batch = fragment
            .take(&[1, 2, 4, 5, 5, 8], dataset.schema())
            .await
            .unwrap();
        assert_eq!(
            batch.column_by_name("i").unwrap().as_ref(),
            &Int32Array::from(vec![121, 122, 124, 125, 125, 128])
        );

        dataset.delete("i in (122, 123, 125)").await.unwrap();
        dataset.validate().await.unwrap();

        // Deleted rows are skipped
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
            &Int32Array::from(vec![121, 124, 127, 128, 131])
        );

        // Empty indices gives empty result
        let batch = fragment.take(&[], dataset.schema()).await.unwrap();
        assert_eq!(
            batch.column_by_name("i").unwrap().as_ref(),
            &Int32Array::from(Vec::<i32>::new())
        );
    }

    #[tokio::test]
    async fn test_fragment_take_rows() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = create_dataset(test_uri).await;
        let fragment = dataset
            .get_fragments()
            .into_iter()
            .find(|f| f.id() == 3)
            .unwrap();

        // Repeated indices are repeated in result.
        let batch = fragment
            .take_rows(&[1, 2, 4, 5, 5, 8], dataset.schema(), false)
            .await
            .unwrap();
        assert_eq!(
            batch.column_by_name("i").unwrap().as_ref(),
            &Int32Array::from(vec![121, 122, 124, 125, 125, 128])
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
            .take_rows(&[1, 2, 4, 5, 8], dataset.schema(), false)
            .await
            .unwrap();
        assert_eq!(
            batch.column_by_name("i").unwrap().as_ref(),
            &Int32Array::from(vec![121, 125, 128])
        );

        // Empty indices gives empty result
        let batch = fragment
            .take_rows(&[], dataset.schema(), false)
            .await
            .unwrap();
        assert_eq!(
            batch.column_by_name("i").unwrap().as_ref(),
            &Int32Array::from(Vec::<i32>::new())
        );

        // Can get row ids
        let batch = fragment
            .take_rows(&[1, 2, 4, 5, 8], dataset.schema(), true)
            .await
            .unwrap();
        assert_eq!(
            batch.column_by_name("i").unwrap().as_ref(),
            &Int32Array::from(vec![121, 125, 128])
        );
        assert_eq!(
            batch.column_by_name(ROW_ID).unwrap().as_ref(),
            &UInt64Array::from(vec![(3 << 32) + 1, (3 << 32) + 5, (3 << 32) + 8])
        );
    }

    #[tokio::test]
    async fn test_recommit_from_file() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = create_dataset(test_uri).await;
        let schema = dataset.schema();
        let dataset_rows = dataset.count_rows().await.unwrap();

        let mut paths: Vec<String> = Vec::new();
        for f in dataset.get_fragments() {
            for file in Fragment::from(f.clone()).files {
                let p = file.path.clone();
                paths.push(p);
            }
        }

        let mut fragments: Vec<Fragment> = Vec::new();
        for (idx, path) in paths.iter().enumerate() {
            let f = FileFragment::create_from_file(path, schema, idx, None)
                .await
                .unwrap();
            fragments.push(f)
        }

        let op = Operation::Overwrite {
            schema: schema.clone(),
            fragments,
        };

        let new_dataset = Dataset::commit(test_uri, op, None, None).await.unwrap();

        assert_eq!(new_dataset.count_rows().await.unwrap(), dataset_rows);

        // Fragments will have number of rows recorded in metadata, even though
        // we passed `None` when constructing the `FileFragment`.
        let fragments = new_dataset.get_fragments();
        assert_eq!(fragments.len(), 5);
        for f in fragments {
            assert_eq!(f.metadata.num_rows(), Some(40));
            assert_eq!(f.count_rows().await.unwrap(), 40);
            assert_eq!(f.metadata().deletion_file, None);
        }
    }

    #[tokio::test]
    async fn test_fragment_count() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = create_dataset(test_uri).await;
        let fragment = dataset.get_fragments().pop().unwrap();

        assert_eq!(fragment.count_rows().await.unwrap(), 40);
        assert_eq!(fragment.physical_rows().await.unwrap(), 40);
        assert!(fragment.metadata.deletion_file.is_none());

        let fragment = fragment
            .delete("i >= 160 and i <= 172")
            .await
            .unwrap()
            .unwrap();

        fragment.validate().await.unwrap();

        assert_eq!(fragment.count_rows().await.unwrap(), 27);
        assert_eq!(fragment.physical_rows().await.unwrap(), 40);
        assert!(fragment.metadata.deletion_file.is_some());
        assert_eq!(
            fragment.metadata.deletion_file.unwrap().num_deleted_rows,
            13
        );
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
                let result_col = mul(input_col, &Int32Array::new_scalar(2)).unwrap();
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

            let op = Operation::Overwrite {
                fragments: vec![new_fragment],
                schema: full_schema.clone(),
            };

            let dataset = Dataset::commit(test_uri, op, None, None).await.unwrap();

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

            assert_eq!(batches[1].schema().as_ref(), &(&new_projection).into());
            let max_value_in_batch = if with_delete { 15 } else { 20 };
            let expected_batch = RecordBatch::try_new(
                Arc::new(ArrowSchema::new(vec![
                    ArrowField::new("i", DataType::Int32, true),
                    ArrowField::new("double_i", DataType::Int32, true),
                ])),
                vec![
                    Arc::new(Int32Array::from_iter_values(10..max_value_in_batch)),
                    Arc::new(Int32Array::from_iter_values(
                        (20..(2 * max_value_in_batch)).step_by(2),
                    )),
                ],
            )
            .unwrap();
            assert_eq!(batches[1], expected_batch);
        }
    }

    #[tokio::test]
    async fn test_merge_fragment() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = create_dataset(test_uri).await;
        dataset.validate().await.unwrap();
        assert_eq!(dataset.count_rows().await.unwrap(), 200);

        let deleted_range = 15..20;
        dataset.delete("i >= 15 and i < 20").await.unwrap();
        dataset.validate().await.unwrap();
        assert_eq!(dataset.count_rows().await.unwrap(), 195);

        // Create data to merge: merge in double the data
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, true),
            ArrowField::new("double_i", DataType::Int32, true),
        ]));
        let to_merge = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..200)),
                Arc::new(Int32Array::from_iter_values((0..400).step_by(2))),
            ],
        )
        .unwrap();

        let stream = RecordBatchIterator::new(vec![Ok(to_merge)], schema.clone());
        dataset.merge(stream, "i", "i").await.unwrap();
        dataset.validate().await.unwrap();

        // Validate the resulting data
        let batches = dataset
            .scan()
            .project(&["i", "double_i"])
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let batch = concat_batches(&schema, &batches).unwrap();

        let mut row_id: i32 = 0;
        let mut i: usize = 0;
        let array_i: &Int32Array = as_primitive_array(&batch["i"]);
        let array_double_i: &Int32Array = as_primitive_array(&batch["double_i"]);
        while row_id < 200 {
            if deleted_range.contains(&row_id) {
                row_id += 1;
                continue;
            }
            assert_eq!(array_i.value(i), row_id);
            assert_eq!(array_double_i.value(i), 2 * row_id);
            row_id += 1;
            i += 1;
        }
    }

    #[tokio::test]
    async fn test_write_batch_size() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            true,
        )]));

        let in_memory_batch = 1024;
        let batches: Vec<RecordBatch> = (0..10)
            .map(|i| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(Int32Array::from_iter_values(
                        i * in_memory_batch..(i + 1) * in_memory_batch,
                    ))],
                )
                .unwrap()
            })
            .collect();

        let batch_iter = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());

        let fragment = FileFragment::create(
            test_uri,
            10,
            batch_iter,
            Some(WriteParams {
                max_rows_per_group: 100,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let (object_store, base_path) = ObjectStore::from_uri(test_uri).await.unwrap();
        let file_reader = FileReader::try_new_with_fragment(
            &object_store,
            &base_path
                .child("data")
                .child(fragment.files[0].path.as_str()),
            10,
            None,
            None,
        )
        .await
        .unwrap();

        for i in 0..file_reader.num_batches() - 1 {
            assert_eq!(file_reader.num_rows_in_batch(i as i32), 100);
        }
        assert_eq!(
            file_reader.num_rows_in_batch(file_reader.num_batches() as i32 - 1) as i32,
            in_memory_batch * 10 % 100
        );
    }
}
