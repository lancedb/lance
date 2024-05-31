// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Wraps a Fragment of the dataset.

pub mod write;

use std::borrow::Cow;
use std::collections::{BTreeMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow_array::cast::as_primitive_array;
use arrow_array::{RecordBatch, RecordBatchReader, UInt32Array, UInt64Array};
use arrow_schema::Schema as ArrowSchema;
use datafusion::logical_expr::Expr;
use datafusion::scalar::ScalarValue;
use futures::future::try_join_all;
use futures::{join, stream, FutureExt, StreamExt, TryFutureExt, TryStreamExt};
use lance_core::utils::address::RowAddress;
use lance_core::utils::deletion::DeletionVector;
use lance_core::ROW_ID_FIELD;
use lance_core::{datatypes::Schema, Error, Result, ROW_ID};
use lance_file::reader::{read_batch, FileReader};
use lance_file::v2;
use lance_file::v2::reader::ReaderProjection;
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::ScanScheduler;
use lance_io::ReadBatchParams;
use lance_table::format::{DataFile, DeletionFile, Fragment};
use lance_table::io::deletion::{deletion_file_path, read_deletion_file, write_deletion_file};
use lance_table::rowids::RowIdSequence;
use lance_table::utils::stream::{
    wrap_with_row_id_and_delete, ReadBatchFutStream, ReadBatchTask, ReadBatchTaskStream,
    RowIdAndDeletesConfig,
};
use snafu::{location, Location};

use self::write::FragmentCreateBuilder;

use super::hash_joiner::HashJoiner;
use super::rowids::load_row_id_sequence;
use super::scanner::Scanner;
use super::updater::Updater;
use super::WriteParams;
use crate::arrow::*;
use crate::dataset::Dataset;

/// A Fragment of a Lance [`Dataset`].
///
/// The interface is modeled after `pyarrow.dataset.Fragment`.
#[derive(Debug, Clone)]
pub struct FileFragment {
    dataset: Arc<Dataset>,

    pub(super) metadata: Fragment,
}

const DEFAULT_BATCH_READ_SIZE: u32 = 1024;

/// A trait for file readers to be implemented by both the v1 and v2 readers
#[async_trait::async_trait]
#[allow(clippy::len_without_is_empty)]
pub trait GenericFileReader: std::fmt::Debug + Send + Sync {
    /// Reads the requested range of rows from the file, returning as a stream
    /// of tasks.
    fn read_range_tasks(
        &self,
        range: Range<u64>,
        batch_size: u32,
        projection: Arc<lance_core::datatypes::Schema>,
    ) -> Result<ReadBatchTaskStream>;
    /// Reads all rows from the file, returning as a stream of tasks
    fn read_all_tasks(
        &self,
        batch_size: u32,
        projection: Arc<lance_core::datatypes::Schema>,
    ) -> Result<ReadBatchTaskStream>;
    /// Take specific rows from the file, returning as a stream of tasks
    fn take_all_tasks(
        &self,
        indices: &[u32],
        batch_size: u32,
        projection: Arc<lance_core::datatypes::Schema>,
    ) -> Result<ReadBatchTaskStream>;

    /// Return the number of rows in the file
    fn len(&self) -> u32;

    // Helper functions to fallback to the legacy implementation while we
    // slowly migrate functionality over to the generic reader

    // Clone the reader, this is needed because Box<dyn Foo: Clone> doesn't
    // implement Clone
    fn clone_box(&self) -> Box<dyn GenericFileReader>;
    // Return true if the reader is a v1 reader
    fn is_legacy(&self) -> bool;
    // Return a reference to the legacy reader, panics if called on a v2
    // file.
    fn as_legacy(&self) -> &FileReader {
        self.as_legacy_opt()
            .expect("legacy function called on v2 file")
    }
    // Return a reference to the legacy reader if this is a v1 reader and
    // return None otherwise
    fn as_legacy_opt(&self) -> Option<&FileReader>;
    // Return a mutable reference to the legacy reader if this is a v1 reader
    // and return None otherwise
    fn as_legacy_opt_mut(&mut self) -> Option<&mut FileReader>;
}

fn ranges_to_tasks(
    reader: &FileReader,
    ranges: Vec<(i32, Range<usize>)>,
    projection: Arc<Schema>,
) -> ReadBatchTaskStream {
    let reader = reader.clone();
    stream::iter(ranges)
        .map(move |(batch_idx, range)| {
            let num_rows = range.end - range.start;
            let range = range.clone();
            let reader = reader.clone();
            let projection = projection.clone();
            let task = tokio::task::spawn(async move {
                read_batch(
                    &reader,
                    &ReadBatchParams::Range(range.clone()),
                    &projection,
                    batch_idx,
                    false,
                    None,
                )
                .await
            })
            .map(|task_out| task_out.unwrap())
            .boxed();
            ReadBatchTask {
                task,
                num_rows: num_rows as u32,
            }
        })
        .boxed()
}

#[async_trait::async_trait]
impl GenericFileReader for FileReader {
    /// Reads the requested range of rows from the file, returning as a stream
    fn read_range_tasks(
        &self,
        range: Range<u64>,
        batch_size: u32,
        projection: Arc<Schema>,
    ) -> Result<ReadBatchTaskStream> {
        let mut to_skip = range.start as u32;
        let mut remaining = range.end as u32 - to_skip;
        let mut ranges = Vec::new();
        let mut batch_idx = 0;
        while remaining > 0 {
            let next_batch_len = self.num_rows_in_batch(batch_idx) as u32;
            let next_batch_idx = batch_idx;
            batch_idx += 1;
            if to_skip >= next_batch_len {
                to_skip -= next_batch_len;
                continue;
            }
            let batch_start = to_skip;
            to_skip = 0;
            let batch_end = next_batch_len.min(batch_start + remaining);
            remaining -= batch_end - batch_start;
            for chunk_start in (batch_start..batch_end).step_by(batch_size as usize) {
                let chunk_end = (chunk_start + batch_size).min(batch_end);
                ranges.push((next_batch_idx, (chunk_start as usize..chunk_end as usize)));
            }
        }
        Ok(ranges_to_tasks(self, ranges, projection))
    }

    fn read_all_tasks(
        &self,
        batch_size: u32,
        projection: Arc<Schema>,
    ) -> Result<ReadBatchTaskStream> {
        let ranges = (0..self.num_batches())
            .flat_map(move |batch_idx| {
                let rows_in_batch = self.num_rows_in_batch(batch_idx as i32);
                (0..rows_in_batch)
                    .step_by(batch_size as usize)
                    .map(move |start| {
                        let end = (start + batch_size as usize).min(rows_in_batch);
                        (batch_idx as i32, start..end)
                    })
            })
            .collect::<Vec<_>>();
        Ok(ranges_to_tasks(self, ranges, projection))
    }

    fn take_all_tasks(
        &self,
        indices: &[u32],
        _batch_size: u32,
        projection: Arc<Schema>,
    ) -> Result<ReadBatchTaskStream> {
        let indices_vec = indices.to_vec();
        let mut reader = self.clone();
        // In the new path the row id is added by the fragment and not the file
        reader.with_row_id(false);
        let task_fut =
            async move { reader.take(&indices_vec, projection.as_ref(), None).await }.boxed();
        let task = std::future::ready(ReadBatchTask {
            task: task_fut,
            num_rows: indices.len() as u32,
        })
        .boxed();
        Ok(futures::stream::once(task).boxed())
    }

    /// Return the number of rows in the file
    fn len(&self) -> u32 {
        self.len() as u32
    }

    fn clone_box(&self) -> Box<dyn GenericFileReader> {
        Box::new(self.clone())
    }

    fn is_legacy(&self) -> bool {
        true
    }

    fn as_legacy_opt(&self) -> Option<&Self> {
        Some(self)
    }

    fn as_legacy_opt_mut(&mut self) -> Option<&mut Self> {
        Some(self)
    }
}

mod v2_adapter {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct Reader {
        reader: Arc<v2::reader::FileReader>,
        field_id_to_column_idx: Arc<BTreeMap<i32, u32>>,
    }

    impl Reader {
        pub fn new(
            reader: Arc<v2::reader::FileReader>,
            field_id_to_column_idx: Arc<BTreeMap<i32, u32>>,
        ) -> Self {
            Self {
                reader,
                field_id_to_column_idx,
            }
        }

        pub fn projection_from_lance(&self, schema: &Schema) -> ReaderProjection {
            let arrow_schema = Arc::new(ArrowSchema::from(schema));
            let column_indices = schema
                .fields
                .iter()
                .map(|f| {
                    *self.field_id_to_column_idx.get(&f.id).unwrap_or_else(|| {
                        panic!(
                            "attempt to project field with id {} which did not exist in the data file",
                            f.id
                        )
                    })
                })
                .collect::<Vec<_>>();
            ReaderProjection {
                schema: arrow_schema,
                column_indices,
            }
        }
    }

    #[async_trait::async_trait]
    impl GenericFileReader for Reader {
        /// Reads the requested range of rows from the file, returning as a stream
        fn read_range_tasks(
            &self,
            range: Range<u64>,
            batch_size: u32,
            projection: Arc<Schema>,
        ) -> Result<ReadBatchTaskStream> {
            let projection = self.projection_from_lance(projection.as_ref());
            Ok(self
                .reader
                .read_tasks(
                    ReadBatchParams::Range(range.start as usize..range.end as usize),
                    batch_size,
                    &projection,
                )?
                .map(|v2_task| ReadBatchTask {
                    task: v2_task.task.map_err(Error::from).boxed(),
                    num_rows: v2_task.num_rows,
                })
                .boxed())
        }

        fn read_all_tasks(
            &self,
            batch_size: u32,
            projection: Arc<Schema>,
        ) -> Result<ReadBatchTaskStream> {
            let projection = self.projection_from_lance(projection.as_ref());
            Ok(self
                .reader
                .read_tasks(ReadBatchParams::RangeFull, batch_size, &projection)?
                .map(|v2_task| ReadBatchTask {
                    task: v2_task.task.map_err(Error::from).boxed(),
                    num_rows: v2_task.num_rows,
                })
                .boxed())
        }

        fn take_all_tasks(
            &self,
            indices: &[u32],
            batch_size: u32,
            projection: Arc<Schema>,
        ) -> Result<ReadBatchTaskStream> {
            let indices = UInt32Array::from(indices.to_vec());
            let projection = self.projection_from_lance(projection.as_ref());
            Ok(self
                .reader
                .read_tasks(ReadBatchParams::Indices(indices), batch_size, &projection)?
                .map(|v2_task| ReadBatchTask {
                    task: v2_task.task.map_err(Error::from).boxed(),
                    num_rows: v2_task.num_rows,
                })
                .boxed())
        }

        /// Return the number of rows in the file
        fn len(&self) -> u32 {
            self.reader.metadata().num_rows as u32
        }

        fn clone_box(&self) -> Box<dyn GenericFileReader> {
            Box::new(self.clone())
        }

        fn is_legacy(&self) -> bool {
            false
        }

        fn as_legacy_opt(&self) -> Option<&FileReader> {
            None
        }

        fn as_legacy_opt_mut(&mut self) -> Option<&mut FileReader> {
            None
        }
    }
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
        let mut builder = FragmentCreateBuilder::new(dataset_uri);

        if let Some(params) = params.as_ref() {
            builder = builder.write_params(params);
        }

        builder.write(reader, Some(id as u64)).await
    }

    pub async fn create_from_file(
        filename: &str,
        schema: &Schema,
        fragment_id: usize,
        physical_rows: Option<usize>,
    ) -> Result<Fragment> {
        let fragment =
            Fragment::with_file_legacy(fragment_id as u64, filename, schema, physical_rows);
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

    /// Open a FileFragment with a given default projection.
    ///
    /// All read operations (other than `read_projected`) will use the supplied
    /// default projection. For `read_projected`, the projection must be a subset
    /// of the default projection.
    ///
    /// Parameters
    /// - `projection`: The projection schema.
    /// - `with_row_id`: If true, the row id will be included in the output.
    ///
    /// `projection` may be an empty schema only if `with_row_id` is true. In that
    /// case, the reader will only be generating row ids.
    pub async fn open(&self, projection: &Schema, with_row_id: bool) -> Result<FragmentReader> {
        let open_files = self.open_readers(projection, with_row_id);
        let deletion_vec_load =
            self.load_deletion_vector(&self.dataset.object_store, &self.metadata);

        let row_id_load = if self.dataset.manifest.uses_move_stable_row_ids() {
            futures::future::Either::Left(
                load_row_id_sequence(&self.dataset, &self.metadata).map_ok(Some),
            )
        } else {
            futures::future::Either::Right(futures::future::ready(Ok(None)))
        };

        let (opened_files, deletion_vec, row_id_sequence) =
            join!(open_files, deletion_vec_load, row_id_load);
        let opened_files = opened_files?;
        let deletion_vec = deletion_vec?;
        let row_id_sequence = row_id_sequence?;

        if opened_files.is_empty() {
            return Err(Error::io(
                format!(
                    "Does not find any data file for schema: {}\nfragment_id={}",
                    projection,
                    self.id()
                ),
                location!(),
            ));
        }

        let mut reader = FragmentReader::try_new(
            self.id(),
            deletion_vec,
            row_id_sequence,
            opened_files,
            ArrowSchema::from(projection),
            self.count_rows().await?,
        )?;

        if with_row_id {
            reader.with_row_id();
        }
        Ok(reader)
    }

    fn get_field_id_offset(data_file: &DataFile) -> u32 {
        data_file.fields.first().copied().unwrap_or(0) as u32
    }

    async fn open_reader(
        &self,
        data_file: &DataFile,
        projection: Option<&Schema>,
        with_row_id: bool,
    ) -> Result<Option<(Box<dyn GenericFileReader>, Arc<Schema>)>> {
        let full_schema = self.dataset.schema();
        // The data file may contain fields that are not part of the dataset any longer, remove those
        let data_file_schema = data_file.schema(full_schema);
        let projection = projection.unwrap_or(full_schema);
        // Also remove any fields that are not part of the user's provided projection
        let schema_per_file = Arc::new(data_file_schema.intersection(projection)?);

        if data_file.is_legacy_file() {
            let max_field_id = data_file.fields.iter().max().unwrap();
            if with_row_id || !schema_per_file.fields.is_empty() {
                let path = self.dataset.data_dir().child(data_file.path.as_str());
                let field_id_offset = Self::get_field_id_offset(data_file);
                let mut reader = FileReader::try_new_with_fragment_id(
                    &self.dataset.object_store,
                    &path,
                    self.schema().clone(),
                    self.id() as u32,
                    field_id_offset as i32,
                    *max_field_id,
                    Some(&self.dataset.session.file_metadata_cache),
                )
                .await?;
                reader.with_row_id(with_row_id);
                let initialized_schema = reader
                    .schema()
                    .project_by_schema(schema_per_file.as_ref())?;
                Ok(Some((Box::new(reader), Arc::new(initialized_schema))))
            } else {
                Ok(None)
            }
        } else if schema_per_file.fields.is_empty() {
            Ok(None)
        } else {
            let path = self.dataset.data_dir().child(data_file.path.as_str());
            let store_scheduler = ScanScheduler::new(self.dataset.object_store.clone(), 16);
            let file_scheduler = store_scheduler.open_file(&path).await?;
            let reader = Arc::new(v2::reader::FileReader::try_open(file_scheduler, None).await?);
            let field_id_to_column_idx = Arc::new(BTreeMap::from_iter(
                data_file
                    .fields
                    .iter()
                    .copied()
                    .zip(data_file.column_indices.iter().copied())
                    .filter_map(|(field_id, column_index)| {
                        if column_index < 0 {
                            None
                        } else {
                            Some((field_id, column_index as u32))
                        }
                    }),
            ));
            let reader = v2_adapter::Reader::new(reader, field_id_to_column_idx);
            Ok(Some((Box::new(reader), schema_per_file)))
        }
    }

    async fn open_readers(
        &self,
        projection: &Schema,
        with_row_id: bool,
    ) -> Result<Vec<(Box<dyn GenericFileReader>, Arc<Schema>)>> {
        let mut opened_files = vec![];
        for (i, data_file) in self.metadata.files.iter().enumerate() {
            let with_row_id = with_row_id && i == 0;
            if let Some((reader, schema)) = self
                .open_reader(data_file, Some(projection), with_row_id)
                .await?
            {
                opened_files.push((reader, schema));
            }
        }

        Ok(opened_files)
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
            Some(DeletionFile {
                num_deleted_rows: Some(num_deleted),
                ..
            }) => Ok(*num_deleted),
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

    async fn load_deletion_vector(
        &self,
        object_store: &ObjectStore,
        fragment: &Fragment,
    ) -> Result<Option<Arc<DeletionVector>>> {
        if let Some(deletion_file) = &fragment.deletion_file {
            let path = deletion_file_path(&self.dataset.base, fragment.id, deletion_file);

            let deletion_vector = self
                .dataset
                .session
                .file_metadata_cache
                .get_or_insert(&path, |_| async {
                    read_deletion_file(&self.dataset.base, fragment, object_store)
                        .await?
                        .ok_or(Error::io(
                            format!(
                                "Deletion file {:?} not found in fragment {}",
                                deletion_file, fragment.id
                            ),
                            location!(),
                        ))
                })
                .await?;
            Ok(Some(deletion_vector))
        } else {
            Ok(None)
        }
    }

    /// Get the number of physical rows in the fragment. This includes deleted rows.
    ///
    /// If there are no deleted rows, this is equal to the number of rows in the
    /// fragment.
    pub async fn physical_rows(&self) -> Result<usize> {
        if self.metadata.files.is_empty() {
            return Err(Error::io(
                format!("Fragment {} does not contain any data", self.id()),
                location!(),
            ));
        };

        // Early versions that did not write the writer version also could write
        // incorrect `physical_row` values. So if we don't have a writer version,
        // we should not used the cached value. On write, we update the values
        // in the manifest, fixing the issue for future reads.
        // See: https://github.com/lancedb/lance/issues/1531
        if self.dataset.manifest.writer_version.is_some() && self.metadata.physical_rows.is_some() {
            return Ok(self.metadata.physical_rows.unwrap());
        }

        // Just open any file. All of them should have same size.
        let some_file = &self.metadata.files[0];
        let (reader, _) = self
            .open_reader(some_file, None, false)
            .await?
            .ok_or_else(|| Error::Internal {
                message: format!(
                    "The data file {} did not have any fields contained in the dataset schema",
                    some_file.path
                ),
                location: location!(),
            })?;

        Ok(reader.len() as usize)
    }

    /// Validate the fragment
    ///
    /// Verifies:
    /// * All field ids in the fragment are distinct
    /// * Within each data file, field ids are in increasing order
    /// * All fields in the schema have a corresponding field in one of the data
    ///  files
    /// * All data files exist and have the same length
    /// * Field ids are distinct between data files.
    /// * Deletion file exists and has rowids in the correct range
    /// * `Fragment.physical_rows` matches length of file
    /// * `DeletionFile.num_deleted_rows` matches length of deletion vector
    pub async fn validate(&self) -> Result<()> {
        let mut seen_fields = HashSet::new();
        for data_file in &self.metadata.files {
            let last = -1;
            for field_id in &data_file.fields {
                if *field_id <= last {
                    return Err(Error::corrupt_file(
                        self.dataset
                            .data_dir()
                            .child(self.metadata.files[0].path.as_str()),
                        format!(
                            "Field id {} is not in increasing order in fragment {:#?}",
                            field_id, self
                        ),
                        location!(),
                    ));
                }

                if !seen_fields.insert(field_id) {
                    return Err(Error::corrupt_file(
                        self.dataset
                            .data_dir()
                            .child(self.metadata.files[0].path.as_str()),
                        format!(
                            "Field id {} is duplicated in fragment {:#?}",
                            field_id, self
                        ),
                        location!(),
                    ));
                }
            }
        }

        if self.metadata.files.iter().any(|f| f.is_legacy_file())
            != self.metadata.files.iter().all(|f| f.is_legacy_file())
        {
            return Err(Error::corrupt_file(
                self.dataset
                    .data_dir()
                    .child(self.metadata.files[0].path.as_str()),
                "Fragment contains a mix of v1 and v2 data files".to_string(),
                location!(),
            ));
        }

        for field in self.schema().fields_pre_order() {
            if !seen_fields.contains(&field.id) {
                return Err(Error::corrupt_file(
                    self.dataset
                        .data_dir()
                        .child(self.metadata.files[0].path.as_str()),
                    format!(
                        "Field {} is missing in fragment {}\nField: {:#?}\nFragment: {:#?}",
                        field.id,
                        self.id(),
                        field,
                        self.metadata()
                    ),
                    location!(),
                ));
            }
        }

        for data_file in &self.metadata.files {
            data_file.validate(&self.dataset.data_dir())?;
        }

        let get_lengths = self.metadata.files.iter().map(|data_file| async move {
            let (reader, _) = self
                .open_reader(data_file, None, false)
                .await?
                .ok_or_else(|| {
                    Error::corrupt_file(
                        self.dataset.data_dir().child(data_file.path.clone()),
                        "did not have any fields in common with the dataset schema",
                        location!(),
                    )
                })?;
            Result::Ok(reader.len() as usize)
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
        for (length, data_file) in get_lengths.iter().zip(self.metadata.files.iter()) {
            if length != expected_length {
                let path = self.dataset.data_dir().child(data_file.path.as_str());
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
        if let Some(physical_rows) = self.metadata.physical_rows {
            if physical_rows != *expected_length {
                return Err(Error::corrupt_file(
                    self.dataset
                        .data_dir()
                        .child(self.metadata.files[0].path.as_str()),
                    format!(
                        "Fragment metadata has incorrect physical_rows. Actual: {} Metadata: {}",
                        expected_length, physical_rows
                    ),
                    location!(),
                ));
            }
        }

        if let Some(deletion_vector) = deletion_vector? {
            if let Some(num_deletions) = self
                .metadata
                .deletion_file
                .as_ref()
                .unwrap()
                .num_deleted_rows
            {
                if num_deletions != deletion_vector.len() {
                    return Err(Error::corrupt_file(
                        deletion_file_path(
                            &self.dataset.base,
                            self.metadata.id,
                            self.metadata.deletion_file.as_ref().unwrap(),
                        ),
                        format!(
                            "deletion vector length does not match metadata. Metadata: {} Deletion vector: {}",
                            num_deletions, deletion_vector.len()
                        ),
                        location!(),
                    ));
                }
            }

            for row_id in deletion_vector {
                if row_id >= *expected_length as u32 {
                    let deletion_file_meta = self.metadata.deletion_file.as_ref().unwrap();
                    return Err(Error::corrupt_file(
                        deletion_file_path(
                            &self.dataset.base,
                            self.metadata.id,
                            deletion_file_meta,
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

    /// Take rows based on local offsets
    ///
    /// If the offsets are out-of-bounds, this will return an error. But if the
    /// offset is marked deleted, it will be ignored. Thus, the number of rows
    /// returned may be less than the number of offsets provided.
    ///
    /// To recover the original row ids from the returned RecordBatch, set the
    /// `with_row_id` parameter to true. This will add a column named `_row_id`
    /// to the RecordBatch at the end.
    pub(crate) async fn take_rows(
        &self,
        offsets: &[u32],
        projection: &Schema,
        with_row_id: bool,
    ) -> Result<RecordBatch> {
        let reader = self.open(projection, with_row_id).await?;

        if offsets.len() > 1 && Self::row_ids_contiguous(offsets) {
            let range = (offsets[0] as usize)..(offsets[offsets.len() - 1] as usize + 1);
            reader.legacy_read_range_as_batch(range).await
        } else {
            // FIXME, change this method to streams
            reader.take_as_batch(offsets).await
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
    ///
    /// The `columns` parameter is a list of existing columns to be read from
    /// the fragment. They can be used to derive new columns. This is allowed to
    /// be empty.
    ///
    /// The `schemas` parameter is a tuple of the write schema (just the new fields)
    /// and the full schema (the target schema after the update). If the write
    /// schema is None, it is inferred from the first batch of results. The full
    /// schema is inferred by appending the write schema to the existing schema.
    pub async fn updater<T: AsRef<str>>(
        &self,
        columns: Option<&[T]>,
        schemas: Option<(Schema, Schema)>,
    ) -> Result<Updater> {
        let mut schema = self.dataset.schema().clone();
        if let Some(columns) = columns {
            schema = schema.project(columns)?;
        }
        // If there is no projection, we are least need to read the row id
        let with_row_id = schema.fields.is_empty();
        let reader = self.open(&schema, with_row_id);
        let deletion_vector = read_deletion_file(
            &self.dataset.base,
            &self.metadata,
            self.dataset.object_store(),
        );
        let (reader, deletion_vector) = join!(reader, deletion_vector);
        let reader = reader?;
        let deletion_vector = deletion_vector?.unwrap_or_default();

        Updater::try_new(self.clone(), reader, deletion_vector, schemas)
    }

    pub(crate) async fn merge(mut self, join_column: &str, joiner: &HashJoiner) -> Result<Self> {
        let mut updater = self.updater(Some(&[join_column]), None).await?;

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
    pub async fn delete(self, predicate: &str) -> Result<Option<Self>> {
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

        // if predicate is `true`, delete the whole fragment
        // else if predicate is `false`, filter the predicate
        // We do this on the expression level after expression optimization has
        // occurred so we also catch expressions that are equivalent to `true`
        if let Some(predicate) = &scanner.filter {
            if matches!(predicate, Expr::Literal(ScalarValue::Boolean(Some(false)))) {
                return Ok(Some(self));
            }
            if matches!(predicate, Expr::Literal(ScalarValue::Boolean(Some(true)))) {
                return Ok(None);
            }
        }

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

        self.write_deletions(deletion_vector).await
    }

    pub(crate) async fn extend_deletions(
        self,
        new_deletions: impl IntoIterator<Item = u32>,
    ) -> Result<Option<Self>> {
        let mut deletion_vector = read_deletion_file(
            &self.dataset.base,
            &self.metadata,
            self.dataset.object_store(),
        )
        .await?
        .unwrap_or_default();

        deletion_vector.extend(new_deletions);

        self.write_deletions(deletion_vector).await
    }

    async fn write_deletions(mut self, deletion_vector: DeletionVector) -> Result<Option<Self>> {
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
#[derive(Debug)]
pub struct FragmentReader {
    /// Readers and schema of each opened data file.
    readers: Vec<(Box<dyn GenericFileReader>, Arc<Schema>)>,

    /// The output schema. The defines the order in which the columns are returned.
    output_schema: ArrowSchema,

    /// The deleted row IDs
    deletion_vec: Option<Arc<DeletionVector>>,

    /// The row id sequence.
    ///
    /// Only populated if the move-stable row id feature is enabled.
    row_id_sequence: Option<Arc<RowIdSequence>>,

    /// ID of the fragment
    fragment_id: usize,

    /// True if we should generate a row id for the output
    with_row_id: bool,

    /// If true, deleted rows will be set to null, which is fast
    /// If false, deleted rows will be removed from the batch, requiring a copy
    make_deletions_null: bool,

    // total number of rows in the fragment
    num_rows: usize,
}

// Custom clone impl needed because it is not easy to clone Box<dyn GenericFileReader>
//
// We currently need FragmentReader to be Clone because the pushdown scan clones it
// to reuse the fragment reader for both "scan with row id" and "scan without row id"
impl Clone for FragmentReader {
    fn clone(&self) -> Self {
        Self {
            readers: self
                .readers
                .iter()
                .map(|(reader, schema)| (reader.clone_box(), schema.clone()))
                .collect::<Vec<_>>(),
            output_schema: self.output_schema.clone(),
            deletion_vec: self.deletion_vec.clone(),
            row_id_sequence: self.row_id_sequence.clone(),
            fragment_id: self.fragment_id,
            with_row_id: self.with_row_id,
            make_deletions_null: self.make_deletions_null,
            num_rows: self.num_rows,
        }
    }
}

impl std::fmt::Display for FragmentReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FragmentReader(id={})", self.fragment_id)
    }
}

fn merge_batches(batches: &[RecordBatch]) -> Result<RecordBatch> {
    if batches.is_empty() {
        return Err(Error::io(
            "Cannot merge empty batches".to_string(),
            location!(),
        ));
    }

    let mut merged = batches[0].clone();
    for batch in batches.iter().skip(1) {
        merged = merged.merge(batch)?;
    }
    Ok(merged)
}

impl FragmentReader {
    fn try_new(
        fragment_id: usize,
        deletion_vec: Option<Arc<DeletionVector>>,
        row_id_sequence: Option<Arc<RowIdSequence>>,
        readers: Vec<(Box<dyn GenericFileReader>, Arc<Schema>)>,
        output_schema: ArrowSchema,
        num_rows: usize,
    ) -> Result<Self> {
        if readers.is_empty() {
            return Err(Error::io(
                "Cannot create FragmentReader with zero readers".to_string(),
                location!(),
            ));
        }

        if let Some(legacy_reader) = readers[0].0.as_legacy_opt() {
            let num_batches = legacy_reader.num_batches();
            for reader in readers.iter().skip(1) {
                if let Some(other_legacy) = reader.0.as_legacy_opt() {
                    if other_legacy.num_batches() != num_batches {
                        return Err(Error::io(
                                "Cannot create FragmentReader from data files with different number of batches"
                                    .to_string(),
                            location!(),
                        ));
                    }
                } else {
                    return Err(Error::io(
                        "Cannot mix legacy and non-legacy readers".to_string(),
                        location!(),
                    ));
                }
            }
        }
        Ok(Self {
            readers,
            output_schema,
            deletion_vec,
            row_id_sequence,
            fragment_id,
            with_row_id: false,
            make_deletions_null: false,
            num_rows,
        })
    }

    pub(crate) fn with_row_id(&mut self) -> &mut Self {
        self.with_row_id = true;
        if let Some(legacy_reader) = self.readers[0].0.as_legacy_opt_mut() {
            legacy_reader.with_row_id(true);
        }
        self.output_schema = self
            .output_schema
            .try_with_column(ROW_ID_FIELD.clone())
            .expect("Table already has a column named _rowid");
        self
    }

    pub(crate) fn with_make_deletions_null(&mut self) -> &mut Self {
        self.make_deletions_null = true;
        for (reader, _) in self.readers.iter_mut() {
            if let Some(legacy_reader) = reader.as_legacy_opt_mut() {
                legacy_reader.with_make_deletions_null(true);
            }
        }
        self
    }

    /// TODO: This method is relied upon by the v1 pushdown mechanism and will need to stay
    /// in place until v1 is removed.  v2 uses a different mechanism for pushdown and so there
    /// is little benefit in updating the v1 pushdown node.
    pub(crate) fn legacy_num_batches(&self) -> usize {
        let legacy_reader = self.readers[0].0.as_legacy();
        let num_batches = legacy_reader.num_batches();
        assert!(
            self.readers
                .iter()
                .all(|r| r.0.as_legacy().num_batches() == num_batches),
            "Data files have varying number of batches, which is not yet supported."
        );
        num_batches
    }

    /// TODO: This method is relied upon by the v1 pushdown mechanism and will need to stay
    /// in place until v1 is removed.  v2 uses a different mechanism for pushdown and so there
    /// is little benefit in updating the v1 pushdown node.
    ///
    /// This method is also used by the updater.  Even though the updater has been updated to
    /// use streams, the updater still needs to know the batch size in v1 so that it can create
    /// files with the same batch size.
    pub(crate) fn legacy_num_rows_in_batch(&self, batch_id: u32) -> Option<u32> {
        if let Some(legacy_reader) = self.readers[0].0.as_legacy_opt() {
            if batch_id < legacy_reader.num_batches() as u32 {
                Some(legacy_reader.num_rows_in_batch(batch_id as i32) as u32)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Read the page statistics of the fragment for the specified fields.
    ///
    /// TODO: This method is relied upon by the v1 pushdown mechanism and will need to stay
    /// in place until v1 is removed.  v2 uses a different mechanism for pushdown and so there
    /// is little benefit in updating the v1 pushdown node.
    pub(crate) async fn legacy_read_page_stats(
        &self,
        projection: Option<&Schema>,
    ) -> Result<Option<RecordBatch>> {
        let mut stats_batches = vec![];
        for (reader, schema) in self.readers.iter() {
            let schema = match projection {
                Some(projection) => Arc::new(schema.intersection(projection)?),
                None => schema.clone(),
            };
            let reader = reader.as_legacy();
            if let Some(stats_batch) = reader.read_page_stats(&schema.field_ids()).await? {
                stats_batches.push(stats_batch);
            }
        }

        if stats_batches.is_empty() {
            Ok(None)
        } else {
            Ok(Some(merge_batches(&stats_batches)?))
        }
    }

    #[cfg(test)]
    async fn read_impl<'a, Fut>(
        &'a self,
        read_fn: impl Fn(&'a dyn GenericFileReader, &'a Schema) -> Fut,
    ) -> Result<RecordBatch>
    where
        Fut: std::future::Future<Output = Result<RecordBatch>> + 'a,
    {
        let futures = self
            .readers
            .iter()
            .map(|(reader, schema)| read_fn(reader.as_ref(), schema))
            .collect::<Vec<_>>();
        let batches = try_join_all(futures).await?;
        Ok(merge_batches(&batches)?.project_by_schema(&self.output_schema)?)
    }

    #[cfg(test)]
    pub(crate) async fn legacy_read_batch(
        &self,
        batch_id: usize,
        params: impl Into<ReadBatchParams> + Clone,
    ) -> Result<RecordBatch> {
        self.read_impl(move |reader, schema| {
            reader.as_legacy().read_batch(
                batch_id as i32,
                params.clone(),
                schema,
                self.deletion_vec.as_ref().map(|dv| dv.as_ref()),
            )
        })
        .await
    }

    /// Read a batch of rows from the fragment, with a subset of columns.
    ///
    /// Note: the projection must be a subset of the schema the reader was created with.
    /// Otherwise incorrect data will be returned.
    ///
    /// TODO: This method is relied upon by the v1 pushdown mechanism and will need to stay
    /// in place until v1 is removed.  v2 uses a different mechanism for pushdown and so there
    /// is little benefit in updating the v1 pushdown node.
    pub(crate) async fn legacy_read_batch_projected(
        &self,
        batch_id: usize,
        params: impl Into<ReadBatchParams> + Clone,
        projection: &Schema,
    ) -> Result<RecordBatch> {
        let read_tasks = self
            .readers
            .iter()
            .enumerate()
            .map(|(reader_idx, (reader, schema))| {
                let projection = schema.intersection(projection);
                let params = params.clone();

                let reader = reader.as_legacy();

                async move {
                    // Apply ? inside the task to keep read_tasks a simple iter of futures
                    // for try_join_all
                    let projection = projection?;
                    // We always get the row_id from the first reader and so we need that even
                    // if the projection is empty
                    let need_for_row_id = self.with_row_id && reader_idx == 0;
                    if projection.fields.is_empty() && !need_for_row_id {
                        // The projection caused one of the data files to become
                        // irrelevant and so we can skip it
                        Result::Ok(None)
                    } else {
                        Ok(Some(
                            reader
                                .read_batch(
                                    batch_id as i32,
                                    params,
                                    &projection,
                                    self.deletion_vec.as_ref().map(|dv| dv.as_ref()),
                                )
                                .await?,
                        ))
                    }
                }
            });
        let batches = try_join_all(read_tasks).await?;
        let batches = batches.into_iter().flatten().collect::<Vec<_>>();

        let output_schema = {
            let mut output_schema = ArrowSchema::from(projection);
            if self.with_row_id {
                output_schema = output_schema.try_with_column(ROW_ID_FIELD.clone())?;
            }
            output_schema
        };

        let result = merge_batches(&batches)?.project_by_schema(&output_schema)?;

        Ok(result)
    }

    fn new_read_impl(
        &self,
        params: ReadBatchParams,
        batch_size: u32,
        read_fn: impl Fn(&dyn GenericFileReader, &Arc<Schema>) -> Result<ReadBatchTaskStream>,
    ) -> Result<ReadBatchFutStream> {
        let total_num_rows = self.readers[0].0.len();
        // Note that the fragment length might be considerably smaller if there are deleted rows.
        // E.g. if a fragment has 100 rows but rows 0..10 are deleted we still need to make
        // sure it is valid to read / take 0..100
        if !params.valid_given_len(total_num_rows as usize) {
            return Err(Error::invalid_input(
                format!(
                    "Invalid read params {} for fragment with {} addressible rows",
                    params, total_num_rows
                ),
                location!(),
            ));
        }
        // If just the row id there is no need to actually read any data
        // and we don't need to involve the readers at all.
        //
        // TODO: This is somewhat redundant at the moment.  The `wrap_with_row_id_and_delete`
        // function can handle empty (zero column) batches.  However, the v1 reader will
        // not emit such batches and so we need this path.
        //
        // We could potentially delete the support for no-columns in the wrap function or
        // we can delete this path once we migrate away from any support of v1.
        if self.with_row_id && self.output_schema.fields.len() == 1 {
            let mut offsets = params
                .slice(0, total_num_rows as usize)
                .unwrap()
                .to_offsets()
                .unwrap();
            if let Some(deletion_vector) = self.deletion_vec.as_ref() {
                // TODO: More efficient set subtraction
                offsets = UInt32Array::from_iter_values(
                    offsets
                        .values()
                        .iter()
                        .copied()
                        .filter(|row_offset| !deletion_vector.contains(*row_offset)),
                );
            }
            let row_ids: Vec<u64> = offsets
                .values()
                .iter()
                .map(|row_id| {
                    u64::from(RowAddress::new_from_parts(self.fragment_id as u32, *row_id))
                })
                .collect();
            let num_intact_rows = row_ids.len() as u32;
            let row_ids_array = UInt64Array::from(row_ids);
            let row_id_schema = Arc::new(self.output_schema.clone());
            let tasks = (0..num_intact_rows)
                .step_by(batch_size as usize)
                .map(move |offset| {
                    let length = batch_size.min(num_intact_rows - offset);
                    let array = Arc::new(row_ids_array.slice(offset as usize, length as usize));
                    let batch = RecordBatch::try_new(row_id_schema.clone(), vec![array]);
                    std::future::ready(batch.map_err(Error::from)).boxed()
                });
            return Ok(stream::iter(tasks).boxed());
        }
        // Read each data file, these reads should produce streams of equal sized
        // tasks.  In other words, if we get 3 tasks of 20 rows and then a task
        // of 10 rows from one data file we should get the same from the other.
        let read_streams = self
            .readers
            .iter()
            .filter_map(|(reader, schema)| {
                // Normally we filter out empty readers in the open_readers method
                // However, we will keep the first empty reader to use for row id
                // purposes on some legacy paths and so we need to filter that out
                // here.
                if schema.fields.is_empty() {
                    None
                } else {
                    Some(read_fn(reader.as_ref(), schema))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        // Merge the streams, this merges the generated batches
        let merged = lance_table::utils::stream::merge_streams(read_streams);

        // Add the row id column (if needed) and delete rows (if a deletion
        // vector is present).
        let config = RowIdAndDeletesConfig {
            deletion_vector: self.deletion_vec.clone(),
            make_deletions_null: self.make_deletions_null,
            with_row_id: self.with_row_id,
            params,
            total_num_rows,
        };
        let output_schema = Arc::new(self.output_schema.clone());

        Ok(wrap_with_row_id_and_delete(
            merged,
            self.fragment_id as u32,
            self.row_id_sequence.clone(),
            config,
        )
        // Finally, reorder the columns to match the order specified in the projection
        .map(move |batch_fut| {
            let output_schema = output_schema.clone();
            batch_fut
                .map(move |batch| {
                    batch?
                        .project_by_schema(&output_schema)
                        .map_err(Error::from)
                })
                .boxed()
        })
        .boxed())
    }

    pub fn read_range(&self, range: Range<u32>, batch_size: u32) -> Result<ReadBatchFutStream> {
        self.new_read_impl(
            ReadBatchParams::Range(range.start as usize..range.end as usize),
            batch_size,
            move |reader, schema| {
                reader.read_range_tasks(
                    range.start as u64..range.end as u64,
                    batch_size,
                    schema.clone(),
                )
            },
        )
    }

    pub fn read_all(&self, batch_size: u32) -> Result<ReadBatchFutStream> {
        self.new_read_impl(
            ReadBatchParams::RangeFull,
            batch_size,
            move |reader, schema| reader.read_all_tasks(batch_size, schema.clone()),
        )
    }

    // Legacy function that reads a range of data and concatenates the results
    // into a single batch
    //
    // TODO: Move away from this by changing callers to support consuming a stream
    pub async fn legacy_read_range_as_batch(&self, range: Range<usize>) -> Result<RecordBatch> {
        let batches = self
            .read_range(
                range.start as u32..range.end as u32,
                DEFAULT_BATCH_READ_SIZE,
            )?
            .buffered(num_cpus::get())
            .try_collect::<Vec<_>>()
            .await?;
        concat_batches(&Arc::new(self.output_schema.clone()), batches.iter()).map_err(Error::from)
    }

    /// Take rows from this fragment.
    pub async fn take(&self, indices: &[u32], batch_size: u32) -> Result<ReadBatchFutStream> {
        let indices_arr = UInt32Array::from(indices.to_vec());
        self.new_read_impl(
            ReadBatchParams::Indices(indices_arr),
            batch_size,
            move |reader, schema| reader.take_all_tasks(indices, batch_size, schema.clone()),
        )
    }

    /// Take rows from this fragment, will perform a copy if the underlying reader returns multiple
    /// batches.  May return an error if the taken rows do not fit into a single batch.
    pub async fn take_as_batch(&self, indices: &[u32]) -> Result<RecordBatch> {
        let batches = self
            .take(indices, u32::MAX)
            .await?
            .buffered(num_cpus::get())
            .try_collect::<Vec<_>>()
            .await?;
        concat_batches(&Arc::new(self.output_schema.clone()), batches.iter()).map_err(Error::from)
    }
}

#[cfg(test)]
mod tests {

    use arrow_arith::numeric::mul;
    use arrow_array::{ArrayRef, Int32Array, RecordBatchIterator, StringArray};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use pretty_assertions::assert_eq;
    use rstest::rstest;
    use tempfile::tempdir;

    use super::*;
    use crate::dataset::transaction::Operation;

    async fn create_dataset(test_uri: &str, use_experimental_writer: bool) -> Dataset {
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
            use_experimental_writer,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        Dataset::open(test_uri).await.unwrap()
    }

    async fn create_dataset_v2(test_uri: &str) -> Dataset {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            true,
        )]));

        let batches: Vec<RecordBatch> = (0..10)
            .map(|i| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20))],
                )
                .unwrap()
            })
            .collect();

        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            use_experimental_writer: true,
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
        let dataset = create_dataset(test_uri, false).await;
        let fragment = &dataset.get_fragments()[2];
        let mut scanner = fragment.scan();
        let batches = scanner
            .with_row_id()
            .filter(" i < 105")
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
    async fn test_fragment_scan_v2() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = create_dataset_v2(test_uri).await;
        let fragment = &dataset.get_fragments()[2];
        let mut scanner = fragment.scan();
        let batches = scanner
            .with_row_id()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(batches.len(), 1);

        assert_eq!(
            batches[0].column_by_name("i").unwrap().as_ref(),
            &Int32Array::from_iter_values(80..120)
        );

        let mut scanner = fragment.scan();
        let batches = scanner
            .with_row_id()
            .batch_size(20)
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
            &Int32Array::from_iter_values(100..120)
        );
    }

    #[tokio::test]
    async fn test_out_of_range() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        // Creates 400 rows in 10 fragments
        let mut dataset = create_dataset(test_uri, false).await;
        // Delete last 20 rows in first fragment
        dataset.delete("i >= 20").await.unwrap();
        // Last fragment has 20 rows but 40 addressible rows
        let fragment = &dataset.get_fragments()[0];
        assert_eq!(fragment.metadata.num_rows().unwrap(), 20);

        for with_row_id in [false, true] {
            let reader = fragment.open(fragment.schema(), with_row_id).await.unwrap();
            for valid_range in [0..40, 20..40] {
                reader
                    .read_range(valid_range, 100)
                    .unwrap()
                    .buffered(1)
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap();
            }
            for invalid_range in [0..41, 41..42] {
                assert!(reader.read_range(invalid_range, 100).is_err());
            }
        }
    }

    #[tokio::test]
    async fn test_fragment_scan_deletions() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = create_dataset(test_uri, false).await;
        dataset.delete("i >= 0 and i < 15").await.unwrap();

        let fragment = &dataset.get_fragments()[0];
        let mut reader = fragment.open(dataset.schema(), true).await.unwrap();
        reader.with_make_deletions_null();

        // Since the first batch is all deleted, it will return an empty batch.
        let batch1 = reader.legacy_read_batch(0, ..).await.unwrap();
        assert_eq!(batch1.num_rows(), 0);

        // The second batch is partially deleted, so the deleted rows will be
        // marked null with null row ids.
        let batch2 = reader.legacy_read_batch(1, ..).await.unwrap();
        assert_eq!(
            batch2.column_by_name(ROW_ID).unwrap().as_ref(),
            &UInt64Array::from_iter((10..20).map(|v| if v < 15 { None } else { Some(v) }))
        );

        // The final batch is not deleted, so it will be returned as-is.
        let batch3 = reader.legacy_read_batch(2, ..).await.unwrap();
        assert_eq!(
            batch3.column_by_name(ROW_ID).unwrap().as_ref(),
            &UInt64Array::from_iter_values(20..30)
        );
    }

    #[tokio::test]
    async fn test_fragment_take_indices() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = create_dataset(test_uri, false).await;
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
        let mut dataset = create_dataset(test_uri, false).await;
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
        let dataset = create_dataset(test_uri, false).await;
        let schema = dataset.schema();
        let dataset_rows = dataset.count_rows(None).await.unwrap();

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

        let new_dataset = Dataset::commit(test_uri, op, None, None, None)
            .await
            .unwrap();

        assert_eq!(new_dataset.count_rows(None).await.unwrap(), dataset_rows);

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

    #[rstest]
    #[tokio::test]
    async fn test_fragment_count(#[values(false, true)] use_experimental_writer: bool) {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = create_dataset(test_uri, use_experimental_writer).await;
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
            Some(13)
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_append_new_columns(#[values(false, true)] use_experimental_writer: bool) {
        for with_delete in [true, false] {
            let test_dir = tempdir().unwrap();
            let test_uri = test_dir.path().to_str().unwrap();
            let mut dataset = create_dataset(test_uri, use_experimental_writer).await;
            dataset.validate().await.unwrap();
            assert_eq!(dataset.count_rows(None).await.unwrap(), 200);

            if with_delete {
                dataset.delete("i >= 15 and i < 20").await.unwrap();
                dataset.validate().await.unwrap();
                assert_eq!(dataset.count_rows(None).await.unwrap(), 195);
            }

            let fragment = &mut dataset.get_fragment(0).unwrap();
            let mut updater = fragment.updater(Some(&["i"]), None).await.unwrap();
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
            let mut full_schema = dataset.schema().merge(new_schema.as_ref()).unwrap();
            full_schema.set_field_id(None);
            let before_version = dataset.version().version;

            let op = Operation::Overwrite {
                fragments: vec![new_fragment],
                schema: full_schema.clone(),
            };

            let dataset = Dataset::commit(test_uri, op, None, None, None)
                .await
                .unwrap();

            // We only kept the first fragment of 40 rows
            assert_eq!(
                dataset.count_rows(None).await.unwrap(),
                if with_delete { 35 } else { 40 }
            );
            assert_eq!(dataset.version().version, before_version + 1);
            dataset.validate().await.unwrap();
            let new_projection = full_schema.project(&["i", "double_i"]).unwrap();

            let stream = dataset
                .scan()
                .batch_size(10)
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

    #[rstest]
    #[tokio::test]
    async fn test_merge_fragment(#[values(false, true)] use_experimental_writer: bool) {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = create_dataset(test_uri, use_experimental_writer).await;
        dataset.validate().await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 200);

        let deleted_range = 15..20;
        dataset.delete("i >= 15 and i < 20").await.unwrap();
        dataset.validate().await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 195);

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
        let file_reader = FileReader::try_new_with_fragment_id(
            &object_store,
            &base_path
                .child("data")
                .child(fragment.files[0].path.as_str()),
            schema.as_ref().try_into().unwrap(),
            10,
            0,
            1,
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

    #[tokio::test]
    async fn test_shuffled_columns() -> Result<()> {
        // Validates we can handle datasets where the order of columns is not
        // aligned with the order of the data files. This can happen when replacing
        // columns in a dataset.
        let batch_i = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "i",
                DataType::Int32,
                true,
            )])),
            vec![Arc::new(Int32Array::from_iter_values(0..20))],
        )?;

        let batch_s = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "s",
                DataType::Utf8,
                true,
            )])),
            vec![Arc::new(StringArray::from_iter_values(
                (0..20).map(|v| format!("s-{}", v)),
            ))],
        )?;

        // Write batch_i as a fragment
        let test_dir = tempdir()?;
        let test_uri = test_dir.path().to_str().unwrap();

        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch_i.clone())], batch_i.schema().clone()),
            test_uri,
            None,
        )
        .await?;
        let fragment = dataset.get_fragments().pop().unwrap();

        // Write batch_s using add_columns
        let mut updater = fragment.updater(Some(&["i"]), None).await?;
        updater.next().await?;
        updater.update(batch_s.clone()).await?;
        let frag = updater.finish().await?;

        // Rearrange schema so it's `s` then `i`.
        let schema = updater.schema().unwrap().clone().project(&["s", "i"])?;

        let dataset = Dataset::commit(
            test_uri,
            Operation::Merge {
                schema,
                fragments: vec![frag],
            },
            Some(dataset.manifest.version),
            None,
            None,
        )
        .await?;

        let expected_data = batch_s.merge(&batch_i)?;
        let actual_data = dataset.scan().try_into_batch().await?;
        assert_eq!(expected_data, actual_data);

        // Also take, read_range, and read_batch_projected
        let reader = dataset
            .get_fragments()
            .first()
            .unwrap()
            .open(dataset.schema(), false)
            .await?;
        let actual_data = reader.take_as_batch(&[0, 1, 2]).await?;
        assert_eq!(expected_data.slice(0, 3), actual_data);

        let actual_data = reader.legacy_read_range_as_batch(0..3).await?;
        assert_eq!(expected_data.slice(0, 3), actual_data);

        let actual_data = reader
            .legacy_read_batch_projected(0, .., &dataset.schema().project(&["s", "i"]).unwrap())
            .await?;
        assert_eq!(expected_data, actual_data);

        // Also check case of row_id.
        let expected_data = expected_data.try_with_column(
            ROW_ID_FIELD.clone(),
            Arc::new(UInt64Array::from_iter_values(0..20)),
        )?;
        let actual_data = dataset.scan().with_row_id().try_into_batch().await?;
        assert_eq!(expected_data, actual_data);

        Ok(())
    }

    #[tokio::test]
    async fn test_row_id_reader() -> Result<()> {
        // Make sure we can create a fragment reader that only captures the row_id.
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "i",
                DataType::Int32,
                true,
            )])),
            vec![Arc::new(Int32Array::from_iter_values(0..20))],
        )?;

        let test_dir = tempdir()?;
        let test_uri = test_dir.path().to_str().unwrap();

        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema().clone()),
            test_uri,
            None,
        )
        .await?;

        let fragment = dataset.get_fragments().pop().unwrap();

        let reader = fragment
            .open(&dataset.schema().project::<&str>(&[])?, true)
            .await?;
        let batch = reader.legacy_read_range_as_batch(0..20).await?;

        let expected_data = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ROW_ID_FIELD.clone()])),
            vec![Arc::new(UInt64Array::from_iter_values(0..20))],
        )?;
        assert_eq!(expected_data, batch);

        // We should get error if we pass empty schema and with_row_id false
        let res = fragment
            .open(&dataset.schema().project::<&str>(&[])?, false)
            .await;
        assert!(matches!(res, Err(Error::IO { .. })));

        Ok(())
    }
}
