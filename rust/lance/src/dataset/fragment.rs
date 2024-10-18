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
use arrow_array::{RecordBatch, RecordBatchReader, StructArray, UInt32Array, UInt64Array};
use arrow_schema::Schema as ArrowSchema;
use datafusion::logical_expr::Expr;
use datafusion::scalar::ScalarValue;
use futures::future::try_join_all;
use futures::{join, stream, FutureExt, StreamExt, TryFutureExt, TryStreamExt};
use lance_core::datatypes::SchemaCompareOptions;
use lance_core::utils::deletion::DeletionVector;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::{datatypes::Schema, Error, Result};
use lance_core::{ROW_ADDR, ROW_ADDR_FIELD, ROW_ID_FIELD};
use lance_encoding::decoder::DecoderPlugins;
use lance_file::reader::{read_batch, FileReader};
use lance_file::v2::reader::{CachedFileMetadata, FileReaderOptions, ReaderProjection};
use lance_file::version::LanceFileVersion;
use lance_file::{determine_file_version, v2};
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{FileScheduler, ScanScheduler, SchedulerConfig};
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

    /// Schema of the reader
    fn projection(&self) -> &Arc<Schema>;

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

#[derive(Clone, Debug)]
struct V1Reader {
    reader: FileReader,
    projection: Arc<Schema>,
}

impl V1Reader {
    fn new(reader: FileReader, projection: Arc<Schema>) -> Self {
        Self { reader, projection }
    }
}

#[async_trait::async_trait]
impl GenericFileReader for V1Reader {
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
            let next_batch_len = self.reader.num_rows_in_batch(batch_idx) as u32;
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
        Ok(ranges_to_tasks(&self.reader, ranges, projection))
    }

    fn read_all_tasks(
        &self,
        batch_size: u32,
        projection: Arc<Schema>,
    ) -> Result<ReadBatchTaskStream> {
        let ranges = (0..self.reader.num_batches())
            .flat_map(move |batch_idx| {
                let rows_in_batch = self.reader.num_rows_in_batch(batch_idx as i32);
                (0..rows_in_batch)
                    .step_by(batch_size as usize)
                    .map(move |start| {
                        let end = (start + batch_size as usize).min(rows_in_batch);
                        (batch_idx as i32, start..end)
                    })
            })
            .collect::<Vec<_>>();
        Ok(ranges_to_tasks(&self.reader, ranges, projection))
    }

    fn take_all_tasks(
        &self,
        indices: &[u32],
        _batch_size: u32,
        projection: Arc<Schema>,
    ) -> Result<ReadBatchTaskStream> {
        let indices_vec = indices.to_vec();
        let reader = self.reader.clone();
        // In the new path the row id is added by the fragment and not the file
        let task_fut = async move { reader.take(&indices_vec, projection.as_ref()).await }.boxed();
        let task = std::future::ready(ReadBatchTask {
            task: task_fut,
            num_rows: indices.len() as u32,
        })
        .boxed();
        Ok(futures::stream::once(task).boxed())
    }

    fn projection(&self) -> &Arc<Schema> {
        &self.projection
    }

    /// Return the number of rows in the file
    fn len(&self) -> u32 {
        self.reader.len() as u32
    }

    fn clone_box(&self) -> Box<dyn GenericFileReader> {
        Box::new(self.clone())
    }

    fn is_legacy(&self) -> bool {
        true
    }

    fn as_legacy_opt(&self) -> Option<&FileReader> {
        Some(&self.reader)
    }

    fn as_legacy_opt_mut(&mut self) -> Option<&mut FileReader> {
        Some(&mut self.reader)
    }
}

mod v2_adapter {
    use lance_encoding::decoder::FilterExpression;

    use super::*;

    #[derive(Debug, Clone)]
    pub struct Reader {
        reader: Arc<v2::reader::FileReader>,
        projection: Arc<Schema>,
        field_id_to_column_idx: Arc<BTreeMap<u32, u32>>,
    }

    impl Reader {
        pub fn new(
            reader: Arc<v2::reader::FileReader>,
            projection: Arc<Schema>,
            field_id_to_column_idx: Arc<BTreeMap<u32, u32>>,
        ) -> Self {
            Self {
                reader,
                projection,
                field_id_to_column_idx,
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
            let projection = ReaderProjection::from_field_ids(
                self.reader.as_ref(),
                projection.as_ref(),
                self.field_id_to_column_idx.as_ref(),
            )?;
            Ok(self
                .reader
                .read_tasks(
                    ReadBatchParams::Range(range.start as usize..range.end as usize),
                    batch_size,
                    Some(projection),
                    FilterExpression::no_filter(),
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
            let projection = ReaderProjection::from_field_ids(
                self.reader.as_ref(),
                projection.as_ref(),
                self.field_id_to_column_idx.as_ref(),
            )?;
            Ok(self
                .reader
                .read_tasks(
                    ReadBatchParams::RangeFull,
                    batch_size,
                    Some(projection),
                    FilterExpression::no_filter(),
                )?
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
            let projection = ReaderProjection::from_field_ids(
                self.reader.as_ref(),
                projection.as_ref(),
                self.field_id_to_column_idx.as_ref(),
            )?;
            Ok(self
                .reader
                .read_tasks(
                    ReadBatchParams::Indices(indices),
                    batch_size,
                    Some(projection),
                    FilterExpression::no_filter(),
                )?
                .map(|v2_task| ReadBatchTask {
                    task: v2_task.task.map_err(Error::from).boxed(),
                    num_rows: v2_task.num_rows,
                })
                .boxed())
        }

        fn projection(&self) -> &Arc<Schema> {
            &self.projection
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

#[derive(Debug, Default)]
pub struct FragReadConfig {
    // Add the row id column
    pub with_row_id: bool,
    // Add the row address column
    pub with_row_address: bool,
}

impl FragReadConfig {
    pub fn with_row_id(mut self, value: bool) -> Self {
        self.with_row_id = value;
        self
    }

    pub fn with_row_address(mut self, value: bool) -> Self {
        self.with_row_address = value;
        self
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
        dataset: &Dataset,
        fragment_id: usize,
        physical_rows: Option<usize>,
    ) -> Result<Fragment> {
        let filepath = dataset.data_dir().child(filename);
        let file_version =
            determine_file_version(dataset.object_store.as_ref(), &filepath, None).await?;

        if file_version != dataset.manifest.data_storage_format.lance_file_version()? {
            return Err(Error::io(
                format!(
                    "File version mismatch. Dataset verison: {:?} Fragment version: {:?}",
                    dataset.manifest.data_storage_format.lance_file_version()?,
                    file_version
                ),
                location!(),
            ));
        }

        if file_version == LanceFileVersion::Legacy {
            let fragment = Fragment::with_file_legacy(
                fragment_id as u64,
                filename,
                dataset.schema(),
                physical_rows,
            );
            Ok(fragment)
        } else {
            // Load the file metadata, confirm the schema is compatible, and
            // determine the column offsets
            let mut frag = Fragment::new(fragment_id as u64);
            let scheduler = ScanScheduler::new(
                dataset.object_store.clone(),
                SchedulerConfig::max_bandwidth(&dataset.object_store),
            );
            let file_scheduler = scheduler.open_file(&filepath).await?;
            let reader = v2::reader::FileReader::try_open(
                file_scheduler,
                None,
                Arc::<DecoderPlugins>::default(),
                &dataset.session.file_metadata_cache,
                FileReaderOptions::default(),
            )
            .await?;
            // If the schemas are not compatible we can't calculate field id offsets
            reader
                .schema()
                .check_compatible(dataset.schema(), &SchemaCompareOptions::default())?;
            let projection = v2::reader::ReaderProjection::from_whole_schema(
                dataset.schema(),
                reader.metadata().version(),
            );
            let physical_rows = reader.metadata().num_rows as usize;
            frag.physical_rows = Some(physical_rows);
            frag.id = fragment_id as u64;

            let column_indices = projection
                .column_indices
                .into_iter()
                .map(|c| c as i32)
                .collect();

            frag.add_file(
                filename,
                dataset.schema().field_ids(),
                column_indices,
                &file_version,
            );
            Ok(frag)
        }
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

    /// The number of data files in this fragment.
    pub fn num_data_files(&self) -> usize {
        self.metadata.files.len()
    }

    /// Gets the data file for a given field
    pub fn data_file_for_field(&self, field_id: u32) -> Option<&DataFile> {
        self.metadata
            .files
            .iter()
            .find(|f| f.fields.contains(&(field_id as i32)))
    }

    /// Open a FileFragment with a given default projection.
    ///
    /// All read operations (other than `read_projected`) will use the supplied
    /// default projection. For `read_projected`, the projection must be a subset
    /// of the default projection.
    ///
    /// Parameters
    /// - `projection`: The projection schema.
    /// - `read_config`: Controls what columns are included in the output.
    /// - `scan_scheduler`: The scheduler to use for reading data files.  If not supplied
    ///                     and the data is v2 data then a new scheduler will be created
    ///
    /// `projection` may be an empty schema only if `with_row_id` is true. In that
    /// case, the reader will only be generating row ids.
    pub async fn open(
        &self,
        projection: &Schema,
        read_config: FragReadConfig,
        scan_scheduler: Option<(Arc<ScanScheduler>, u64)>,
    ) -> Result<FragmentReader> {
        let open_files = self.open_readers(projection, scan_scheduler);
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

        if opened_files.is_empty() && !read_config.with_row_id && !read_config.with_row_address {
            return Err(Error::io(
                format!(
                    "Did not find any data files for schema: {}\nfragment_id={}",
                    projection,
                    self.id()
                ),
                location!(),
            ));
        }

        let num_physical_rows = self.physical_rows().await?;

        let mut reader = FragmentReader::try_new(
            self.id(),
            deletion_vec,
            row_id_sequence,
            opened_files,
            ArrowSchema::from(projection),
            self.count_rows().await?,
            num_physical_rows,
        )?;

        if read_config.with_row_id {
            reader.with_row_id();
        }
        if read_config.with_row_address {
            reader.with_row_address();
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
        scan_scheduler: Option<(Arc<ScanScheduler>, u64)>,
    ) -> Result<Option<Box<dyn GenericFileReader>>> {
        let full_schema = self.dataset.schema();
        // The data file may contain fields that are not part of the dataset any longer, remove those
        let data_file_schema = data_file.schema(full_schema);
        let projection = projection.unwrap_or(full_schema);
        // Also remove any fields that are not part of the user's provided projection
        let schema_per_file = Arc::new(projection.intersection_ignore_types(&data_file_schema)?);

        if data_file.is_legacy_file() {
            let max_field_id = data_file.fields.iter().max().unwrap();
            if !schema_per_file.fields.is_empty() {
                let path = self.dataset.data_dir().child(data_file.path.as_str());
                let field_id_offset = Self::get_field_id_offset(data_file);
                let reader = FileReader::try_new_with_fragment_id(
                    &self.dataset.object_store,
                    &path,
                    self.schema().clone(),
                    self.id() as u32,
                    field_id_offset as i32,
                    *max_field_id,
                    Some(&self.dataset.session.file_metadata_cache),
                )
                .await?;
                let initialized_schema = reader
                    .schema()
                    .project_by_schema(schema_per_file.as_ref())?;
                let reader = V1Reader::new(reader, Arc::new(initialized_schema));
                Ok(Some(Box::new(reader)))
            } else {
                Ok(None)
            }
        } else if schema_per_file.fields.is_empty() {
            Ok(None)
        } else {
            let path = self.dataset.data_dir().child(data_file.path.as_str());
            let (store_scheduler, priority_offset) = scan_scheduler.unwrap_or_else(|| {
                (
                    ScanScheduler::new(
                        self.dataset.object_store.clone(),
                        SchedulerConfig::max_bandwidth(&self.dataset.object_store),
                    ),
                    0,
                )
            });
            let file_scheduler = store_scheduler
                .open_file_with_priority(&path, priority_offset)
                .await?;
            let file_metadata = self.get_file_metadata(&file_scheduler).await?;
            let reader = Arc::new(
                v2::reader::FileReader::try_open_with_file_metadata(
                    file_scheduler,
                    None,
                    Arc::<DecoderPlugins>::default(),
                    file_metadata,
                    &self.dataset.session.file_metadata_cache,
                    FileReaderOptions::default(),
                )
                .await?,
            );
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
                            Some((field_id as u32, column_index as u32))
                        }
                    }),
            ));
            let reader = v2_adapter::Reader::new(reader, schema_per_file, field_id_to_column_idx);
            Ok(Some(Box::new(reader)))
        }
    }

    async fn open_readers(
        &self,
        projection: &Schema,
        scan_scheduler: Option<(Arc<ScanScheduler>, u64)>,
    ) -> Result<Vec<Box<dyn GenericFileReader>>> {
        let mut opened_files = vec![];
        for data_file in &self.metadata.files {
            if let Some(reader) = self
                .open_reader(data_file, Some(projection), scan_scheduler.clone())
                .await?
            {
                opened_files.push(reader);
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
        let reader = self
            .open_reader(some_file, None, None)
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
    ///   files
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
            let reader = self
                .open_reader(data_file, None, None)
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

            for offset in deletion_vector {
                if offset >= *expected_length as u32 {
                    let deletion_file_meta = self.metadata.deletion_file.as_ref().unwrap();
                    return Err(Error::corrupt_file(
                        deletion_file_path(
                            &self.dataset.base,
                            self.metadata.id,
                            deletion_file_meta,
                        ),
                        format!("deletion vector contains an offset that is out of range. Offset: {} Fragment length: {}", offset, expected_length),
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

    /// Get the file metadata for this fragment, using the cache if available.
    async fn get_file_metadata(
        &self,
        file_scheduler: &FileScheduler,
    ) -> Result<Arc<CachedFileMetadata>> {
        let cache = &self.dataset.session.file_metadata_cache;
        let path = file_scheduler.reader().path();

        let file_metadata = cache
            .get_or_insert(path, |_path| async {
                let file_metadata: CachedFileMetadata =
                    v2::reader::FileReader::read_all_metadata(file_scheduler).await?;
                Ok(file_metadata)
            })
            .await?;
        Ok(file_metadata)
    }

    /// Take rows based on internal local row offsets
    ///
    /// If the row offsets are out-of-bounds, this will return an error. But if the
    /// row offset is marked deleted, it will be ignored. Thus, the number of rows
    /// returned may be less than the number of row offsets provided.
    ///
    /// To recover the original row addresses from the returned RecordBatch, set the
    /// `with_row_address` parameter to true. This will add a column named `_rowaddr`
    /// to the RecordBatch at the end.
    pub(crate) async fn take_rows(
        &self,
        row_offsets: &[u32],
        projection: &Schema,
        with_row_address: bool,
    ) -> Result<RecordBatch> {
        let reader = self
            .open(
                projection,
                FragReadConfig::default().with_row_address(with_row_address),
                None,
            )
            .await?;

        if row_offsets.len() > 1 && Self::row_ids_contiguous(row_offsets) {
            let range =
                (row_offsets[0] as usize)..(row_offsets[row_offsets.len() - 1] as usize + 1);
            reader.legacy_read_range_as_batch(range).await
        } else {
            // FIXME, change this method to streams
            reader.take_as_batch(row_offsets).await
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
        batch_size: Option<u32>,
    ) -> Result<Updater> {
        let mut schema = self.dataset.schema().clone();

        let mut with_row_addr = false;
        if let Some(columns) = columns {
            let mut projection = Vec::new();
            for column in columns {
                if column.as_ref() == ROW_ADDR {
                    with_row_addr = true;
                } else {
                    projection.push(column.as_ref());
                }
            }
            schema = schema.project(&projection)?;
        }
        // If there is no projection, we at least need to read the row addresses
        with_row_addr |= schema.fields.is_empty();

        let reader = self.open(
            &schema,
            FragReadConfig::default().with_row_address(with_row_addr),
            None,
        );
        let deletion_vector = read_deletion_file(
            &self.dataset.base,
            &self.metadata,
            self.dataset.object_store(),
        );
        let (reader, deletion_vector) = join!(reader, deletion_vector);
        let reader = reader?;
        let deletion_vector = deletion_vector?.unwrap_or_default();

        Updater::try_new(self.clone(), reader, deletion_vector, schemas, batch_size)
    }

    pub(crate) async fn merge(mut self, join_column: &str, joiner: &HashJoiner) -> Result<Self> {
        let mut updater = self.updater(Some(&[join_column]), None, None).await?;

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

        // scan with predicate and row addresses
        let mut scanner = self.scan();

        let predicate_lower = predicate.trim().to_lowercase();
        if predicate_lower == "true" {
            return Ok(None);
        } else if predicate_lower == "false" {
            return Ok(Some(self));
        }

        scanner
            .with_row_address()
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

        // As we get row addrs, add them into our deletion vector
        scanner
            .try_into_stream()
            .await?
            .try_for_each(|batch| {
                let array = batch[ROW_ADDR].clone();
                let int_array: &UInt64Array = as_primitive_array(array.as_ref());

                // _rowaddr is global, not within fragment level. The high bits
                // are the fragment_id, the low bits are the row_id within the
                // fragment.
                let local_row_ids = int_array.values().iter().map(|v| *v as u32);

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
    readers: Vec<Box<dyn GenericFileReader>>,

    /// The output schema. The defines the order in which the columns are returned.
    output_schema: ArrowSchema,

    /// The deleted row IDs
    deletion_vec: Option<Arc<DeletionVector>>,

    /// The row id sequence
    ///
    /// Only populated if the move-stable row id feature is enabled.
    row_id_sequence: Option<Arc<RowIdSequence>>,

    /// ID of the fragment
    fragment_id: usize,

    /// True if we should generate a row id for the output
    with_row_id: bool,

    /// True if we should generate a row address column in output
    with_row_addr: bool,

    /// If true, deleted rows will be set to null, which is fast
    /// If false, deleted rows will be removed from the batch, requiring a copy
    make_deletions_null: bool,

    // total number of real rows in the fragment (num_physical_rows - num_deleted_rows)
    num_rows: usize,

    // total number of physical rows in the fragment (all rows, ignoring deletions)
    num_physical_rows: usize,
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
                .map(|reader| reader.clone_box())
                .collect::<Vec<_>>(),
            output_schema: self.output_schema.clone(),
            deletion_vec: self.deletion_vec.clone(),
            row_id_sequence: self.row_id_sequence.clone(),
            fragment_id: self.fragment_id,
            with_row_id: self.with_row_id,
            with_row_addr: self.with_row_addr,
            make_deletions_null: self.make_deletions_null,
            num_rows: self.num_rows,
            num_physical_rows: self.num_physical_rows,
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
        readers: Vec<Box<dyn GenericFileReader>>,
        output_schema: ArrowSchema,
        num_rows: usize,
        num_physical_rows: usize,
    ) -> Result<Self> {
        if let Some(legacy_reader) = readers.first().and_then(|reader| reader.as_legacy_opt()) {
            let num_batches = legacy_reader.num_batches();
            for reader in readers.iter().skip(1) {
                if let Some(other_legacy) = reader.as_legacy_opt() {
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
            with_row_addr: false,
            make_deletions_null: false,
            num_rows,
            num_physical_rows,
        })
    }

    pub(crate) fn with_row_id(&mut self) -> &mut Self {
        self.with_row_id = true;
        self.output_schema = self
            .output_schema
            .try_with_column(ROW_ID_FIELD.clone())
            .expect("Table already has a column named _rowid");
        self
    }

    pub(crate) fn with_row_address(&mut self) -> &mut Self {
        self.with_row_addr = true;
        self.output_schema = self
            .output_schema
            .try_with_column(ROW_ADDR_FIELD.clone())
            .expect("Table already has a column named _rowaddr");
        self
    }

    pub(crate) fn with_make_deletions_null(&mut self) -> &mut Self {
        self.make_deletions_null = true;
        self
    }

    /// TODO: This method is relied upon by the v1 pushdown mechanism and will need to stay
    /// in place until v1 is removed.  v2 uses a different mechanism for pushdown and so there
    /// is little benefit in updating the v1 pushdown node.
    pub(crate) fn legacy_num_batches(&self) -> usize {
        let legacy_reader = self.readers[0].as_legacy();
        let num_batches = legacy_reader.num_batches();
        assert!(
            self.readers
                .iter()
                .all(|r| r.as_legacy().num_batches() == num_batches),
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
        if let Some(legacy_reader) = self.readers.first().and_then(|r| r.as_legacy_opt()) {
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
        for reader in self.readers.iter() {
            let schema = match projection {
                Some(projection) => Arc::new(reader.projection().intersection(projection)?),
                None => reader.projection().clone(),
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
        let first_reader = self.readers[0].as_legacy();
        // All batches have the same size in v1, except for the last one.
        let batch_offset = batch_id * first_reader.num_rows_in_batch(0);
        let rows_in_batch = first_reader.num_rows_in_batch(batch_id as i32);

        let batches = if !projection.fields.is_empty() {
            let read_tasks = self.readers.iter().map(|reader| {
                let projection = reader.projection().intersection(projection);
                let params = params.clone();

                let reader = reader.as_legacy();

                async move {
                    // Apply ? inside the task to keep read_tasks a simple iter of futures
                    // for try_join_all
                    let projection = projection?;
                    if projection.fields.is_empty() {
                        // The projection caused one of the data files to become
                        // irrelevant and so we can skip it
                        Result::Ok(None)
                    } else {
                        Ok(Some(
                            reader
                                .read_batch(batch_id as i32, params, &projection)
                                .await?,
                        ))
                    }
                }
            });
            let results = try_join_all(read_tasks).await?;
            results.into_iter().flatten().collect::<Vec<RecordBatch>>()
        } else {
            // If we are selecting no columns, we can assume we are just getting
            // the row ids. If this is the case, we need to generate an empty
            // batch with the correct number of rows.
            let expected_rows = params
                .clone()
                .into()
                .slice(0, rows_in_batch)
                .unwrap()
                .to_offsets()?
                .len();
            vec![RecordBatch::from(StructArray::new_empty_fields(
                expected_rows,
                None,
            ))]
        };

        let params = params.into();
        let result = merge_batches(&batches)?;

        // Need to apply deletions and row ids.
        // In order to apply deletions we need to change the parameters to be
        // relative to the file, not the batch.
        let file_params = match params {
            ReadBatchParams::Indices(indices) => ReadBatchParams::Indices(
                indices
                    .values()
                    .iter()
                    .map(|i| *i + batch_offset as u32)
                    .collect(),
            ),
            ReadBatchParams::RangeFull => {
                ReadBatchParams::Range(batch_offset..(batch_offset + rows_in_batch))
            }
            ReadBatchParams::RangeFrom(start) => {
                ReadBatchParams::Range((start.start + batch_offset)..(batch_offset + rows_in_batch))
            }
            ReadBatchParams::RangeTo(end) => {
                ReadBatchParams::Range(batch_offset..(end.end + batch_offset))
            }
            ReadBatchParams::Range(range) => {
                ReadBatchParams::Range((range.start + batch_offset)..(range.end + batch_offset))
            }
        };
        let result = lance_table::utils::stream::apply_row_id_and_deletes(
            result,
            0,
            self.fragment_id as u32,
            &RowIdAndDeletesConfig {
                params: file_params,
                deletion_vector: self.deletion_vec.clone(),
                row_id_sequence: self.row_id_sequence.clone(),
                with_row_id: self.with_row_id,
                with_row_addr: self.with_row_addr,
                make_deletions_null: self.make_deletions_null,
                total_num_rows: first_reader.len() as u32,
            },
        )?;

        let output_schema = {
            let mut output_schema = ArrowSchema::from(projection);
            if self.with_row_id {
                output_schema = output_schema.try_with_column(ROW_ID_FIELD.clone())?;
            }
            if self.with_row_addr {
                output_schema = output_schema.try_with_column(ROW_ADDR_FIELD.clone())?;
            }
            output_schema
        };

        Ok(result.project_by_schema(&output_schema)?)
    }

    fn new_read_impl(
        &self,
        params: ReadBatchParams,
        batch_size: u32,
        read_fn: impl Fn(&dyn GenericFileReader) -> Result<ReadBatchTaskStream>,
    ) -> Result<ReadBatchFutStream> {
        let total_num_rows = self.num_physical_rows as u32;
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
        // If just the row id or address there is no need to actually read any data
        // and we don't need to involve the readers at all.
        //
        // The v1 reader does not support reading batches with zero columns, so
        // we need this as a separate code path.
        // In these cases, we can just emit batches with zero columns and rely
        // on `wrap_with_row_id_and_delete` to add the row id or address column.
        //
        // We could potentially delete the support for no-columns in the wrap function or
        // we can delete this path once we migrate away from any support of v1.
        let merged = if self.with_row_addr as usize + self.with_row_id as usize
            == self.output_schema.fields.len()
        {
            let selected_rows = params
                .slice(0, total_num_rows as usize)
                .unwrap()
                .to_offsets()
                .unwrap()
                .len();
            let tasks = (0..selected_rows)
                .step_by(batch_size as usize)
                .map(move |offset| {
                    let num_rows = (batch_size as usize).min(selected_rows - offset);
                    let batch = RecordBatch::from(StructArray::new_empty_fields(num_rows, None));
                    ReadBatchTask {
                        task: std::future::ready(Ok(batch)).boxed(),
                        num_rows: num_rows as u32,
                    }
                });
            stream::iter(tasks).boxed()
        } else {
            // Read each data file, these reads should produce streams of equal sized
            // tasks.  In other words, if we get 3 tasks of 20 rows and then a task
            // of 10 rows from one data file we should get the same from the other.
            let read_streams = self
                .readers
                .iter()
                .filter_map(|reader| {
                    // Normally we filter out empty readers in the open_readers method
                    // However, we will keep the first empty reader to use for row id
                    // purposes on some legacy paths and so we need to filter that out
                    // here.
                    if reader.projection().fields.is_empty() {
                        None
                    } else {
                        Some(read_fn(reader.as_ref()))
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            // Merge the streams, this merges the generated batches
            lance_table::utils::stream::merge_streams(read_streams)
        };

        // Add the row id column (if needed) and delete rows (if a deletion
        // vector is present).
        let config = RowIdAndDeletesConfig {
            deletion_vector: self.deletion_vec.clone(),
            row_id_sequence: self.row_id_sequence.clone(),
            make_deletions_null: self.make_deletions_null,
            with_row_id: self.with_row_id,
            with_row_addr: self.with_row_addr,
            params,
            total_num_rows,
        };
        let output_schema = Arc::new(self.output_schema.clone());
        Ok(
            wrap_with_row_id_and_delete(merged, self.fragment_id as u32, config)
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
                .boxed(),
        )
    }

    fn patch_range_for_deletions(&self, range: Range<u32>, dv: &DeletionVector) -> Range<u32> {
        let mut start = range.start;
        let mut end = range.end;
        for val in dv.to_sorted_iter() {
            if val <= start {
                start += 1;
                end += 1;
            } else if val < end {
                end += 1;
            } else {
                break;
            }
        }
        start..end
    }

    fn do_read_range(
        &self,
        mut range: Range<u32>,
        batch_size: u32,
        skip_deleted_rows: bool,
    ) -> Result<ReadBatchFutStream> {
        if skip_deleted_rows {
            if let Some(deletion_vector) = self.deletion_vec.as_ref() {
                range = self.patch_range_for_deletions(range, deletion_vector.as_ref());
            }
        }
        self.new_read_impl(
            ReadBatchParams::Range(range.start as usize..range.end as usize),
            batch_size,
            move |reader| {
                reader.read_range_tasks(
                    range.start as u64..range.end as u64,
                    batch_size,
                    reader.projection().clone(),
                )
            },
        )
    }

    /// Reads a range of rows from the fragment
    ///
    /// This function interprets the request as the Xth to the Nth row of the fragment (after deletions)
    /// and will always return range.len().min(self.num_rows()) rows.
    pub fn read_range(&self, range: Range<u32>, batch_size: u32) -> Result<ReadBatchFutStream> {
        self.do_read_range(range, batch_size, true)
    }

    /// Takes a range of rows from the fragment
    ///
    /// Unlike [`Self::read_range`], this function will NOT skip deleted rows.  If rows are deleted they will
    /// be filtered or set to null.  This function may return less than range.len() rows as a result.
    pub fn take_range(&self, range: Range<u32>, batch_size: u32) -> Result<ReadBatchFutStream> {
        self.do_read_range(range, batch_size, false)
    }

    pub fn read_all(&self, batch_size: u32) -> Result<ReadBatchFutStream> {
        self.new_read_impl(ReadBatchParams::RangeFull, batch_size, move |reader| {
            reader.read_all_tasks(batch_size, reader.projection().clone())
        })
    }

    // Legacy function that reads a range of data and concatenates the results
    // into a single batch
    //
    // TODO: Move away from this by changing callers to support consuming a stream
    pub async fn legacy_read_range_as_batch(&self, range: Range<usize>) -> Result<RecordBatch> {
        let batches = self
            .take_range(
                range.start as u32..range.end as u32,
                DEFAULT_BATCH_READ_SIZE,
            )?
            .buffered(get_num_compute_intensive_cpus())
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
            move |reader| reader.take_all_tasks(indices, batch_size, reader.projection().clone()),
        )
    }

    /// Take rows from this fragment, will perform a copy if the underlying reader returns multiple
    /// batches.  May return an error if the taken rows do not fit into a single batch.
    pub async fn take_as_batch(&self, indices: &[u32]) -> Result<RecordBatch> {
        let batches = self
            .take(indices, u32::MAX)
            .await?
            .buffered(get_num_compute_intensive_cpus())
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
    use lance_core::ROW_ID;
    use lance_datagen::{array, gen, RowCount};
    use lance_file::version::LanceFileVersion;
    use lance_io::object_store::ObjectStoreRegistry;
    use pretty_assertions::assert_eq;
    use rstest::rstest;
    use tempfile::tempdir;
    use v2::writer::FileWriterOptions;

    use super::*;
    use crate::{dataset::transaction::Operation, utils::test::TestDatasetGenerator};

    async fn create_dataset(test_uri: &str, data_storage_version: LanceFileVersion) -> Dataset {
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
            data_storage_version: Some(data_storage_version),
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
            data_storage_version: Some(LanceFileVersion::Stable),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        Dataset::open(test_uri).await.unwrap()
    }

    #[rstest]
    #[tokio::test]
    async fn test_fragment_scan(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = create_dataset(test_uri, data_storage_version).await;
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

        if data_storage_version == LanceFileVersion::Legacy {
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
        } else {
            assert_eq!(batches.len(), 1);

            assert_eq!(
                batches[0].column_by_name("i").unwrap().as_ref(),
                &Int32Array::from_iter_values(80..105)
            )
        }
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
        let mut dataset = create_dataset(test_uri, LanceFileVersion::Legacy).await;
        // Delete last 20 rows in first fragment
        dataset.delete("i >= 20").await.unwrap();
        // Last fragment has 20 rows but 40 addressible rows
        let fragment = &dataset.get_fragments()[0];
        assert_eq!(fragment.metadata.num_rows().unwrap(), 20);

        // Test with take_range (all rows addressible)
        for with_row_id in [false, true] {
            let reader = fragment
                .open(
                    fragment.schema(),
                    FragReadConfig::default().with_row_id(with_row_id),
                    None,
                )
                .await
                .unwrap();
            for valid_range in [0..40, 20..40] {
                reader
                    .take_range(valid_range, 100)
                    .unwrap()
                    .buffered(1)
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap();
            }
            for invalid_range in [0..41, 41..42] {
                assert!(reader.take_range(invalid_range, 100).is_err());
            }
        }

        // Test with read_range (only non-deleted rows addressible)
        for with_row_id in [false, true] {
            let reader = fragment
                .open(
                    fragment.schema(),
                    FragReadConfig::default().with_row_id(with_row_id),
                    None,
                )
                .await
                .unwrap();
            for valid_range in [0..20, 0..10, 10..20] {
                reader
                    .read_range(valid_range, 100)
                    .unwrap()
                    .buffered(1)
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap();
            }
            for invalid_range in [0..21, 21..22] {
                assert!(reader.read_range(invalid_range, 100).is_err());
            }
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_fragment_take_range_deletions(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = create_dataset(test_uri, data_storage_version).await;
        dataset.delete("i >= 0 and i < 15").await.unwrap();

        let fragment = &dataset.get_fragments()[0];
        let mut reader = fragment
            .open(
                dataset.schema(),
                FragReadConfig::default().with_row_id(true),
                None,
            )
            .await
            .unwrap();
        reader.with_make_deletions_null();

        if data_storage_version == LanceFileVersion::Legacy {
            // Since the first batch is all deleted, it will return an empty batch.
            let batch1 = reader
                .legacy_read_batch_projected(0, .., dataset.schema())
                .await
                .unwrap();
            assert_eq!(batch1.num_rows(), 0);

            // The second batch is partially deleted, so the deleted rows will be
            // marked null with null row ids.
            let batch2 = reader
                .legacy_read_batch_projected(1, .., dataset.schema())
                .await
                .unwrap();
            assert_eq!(
                batch2.column_by_name(ROW_ID).unwrap().as_ref(),
                &UInt64Array::from_iter((10..20).map(|v| if v < 15 { None } else { Some(v) }))
            );

            // The final batch is not deleted, so it will be returned as-is.
            let batch3 = reader
                .legacy_read_batch_projected(2, .., dataset.schema())
                .await
                .unwrap();
            assert_eq!(
                batch3.column_by_name(ROW_ID).unwrap().as_ref(),
                &UInt64Array::from_iter_values(20..30)
            );
        } else {
            let to_batches = |range: Range<u32>| {
                let batch_size = range.len() as u32;
                reader
                    .take_range(range, batch_size)
                    .unwrap()
                    .buffered(1)
                    .try_collect::<Vec<_>>()
            };
            // Since the first batch is all deleted, it will return an empty batch.
            let batches = to_batches(0..10).await.unwrap();
            assert_eq!(batches.len(), 1);
            let batch = batches.into_iter().next().unwrap();
            assert_eq!(batch.num_rows(), 0);

            let batches = to_batches(10..20).await.unwrap();
            assert_eq!(batches.len(), 1);
            let batch = batches.into_iter().next().unwrap();
            // The second batch is partially deleted, so the deleted rows will be
            // marked null with null row ids.
            assert_eq!(
                batch.column_by_name(ROW_ID).unwrap().as_ref(),
                &UInt64Array::from_iter((10..20).map(|v| if v < 15 { None } else { Some(v) }))
            );

            // The final batch is not deleted, so it will be returned as-is.
            let batches = to_batches(20..30).await.unwrap();
            assert_eq!(batches.len(), 1);
            let batch = batches.into_iter().next().unwrap();
            assert_eq!(
                batch.column_by_name(ROW_ID).unwrap().as_ref(),
                &UInt64Array::from_iter_values(20..30)
            );
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_range_scan_deletions(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = create_dataset(test_uri, data_storage_version).await;

        let version = dataset.version().version;

        let check = |cond: &'static str, range: Range<u32>, expected: Vec<i32>| async {
            let mut dataset = dataset.checkout_version(version).await.unwrap();
            dataset.restore().await.unwrap();
            dataset.delete(cond).await.unwrap();

            let fragment = &dataset.get_fragments()[0];
            let reader = fragment
                .open(
                    dataset.schema(),
                    FragReadConfig::default().with_row_id(true),
                    None,
                )
                .await
                .unwrap();

            // Using batch_size=20 here.  If we use batch_size=range.len() we get
            // multiple batches because we might have to read from a larger range
            // to satisfy the request
            let mut stream = reader.read_range(range, 20).unwrap();
            let mut batches = Vec::new();
            while let Some(next) = stream.next().await {
                batches.push(next.await.unwrap());
            }
            let schema = Arc::new(dataset.schema().into());
            let batch = arrow_select::concat::concat_batches(&schema, batches.iter()).unwrap();

            assert_eq!(batch.num_rows(), expected.len());
            assert_eq!(
                batch.column_by_name("i").unwrap().as_ref(),
                &Int32Array::from(expected)
            );
        };
        // Deleting from the start
        check("i < 5", 0..2, vec![5, 6]).await;
        check("i < 5", 0..15, (5..20).collect()).await;
        // Deleting from the middle
        check("i >= 5 and i < 15", 7..9, vec![17, 18]).await;
        check("i >= 5 and i < 15", 3..5, vec![3, 4]).await;
        check("i >= 5 and i < 15", 3..6, vec![3, 4, 15]).await;
        check("i >= 5 and i < 15", 5..6, vec![15]).await;
        check("i >= 5 and i < 15", 5..10, vec![15, 16, 17, 18, 19]).await;
        check(
            "i >= 5 and i < 15",
            0..10,
            vec![0, 1, 2, 3, 4, 15, 16, 17, 18, 19],
        )
        .await;
        // Deleting from the end
        check("i >= 15", 10..15, vec![10, 11, 12, 13, 14]).await;
        check("i >= 15", 0..15, (0..15).collect()).await;
    }

    #[rstest]
    #[tokio::test]
    async fn test_fragment_take_indices(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = create_dataset(test_uri, data_storage_version).await;
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

    #[rstest]
    #[tokio::test]
    async fn test_fragment_take_rows(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = create_dataset(test_uri, data_storage_version).await;
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
            batch.column_by_name(ROW_ADDR).unwrap().as_ref(),
            &UInt64Array::from(vec![(3 << 32) + 1, (3 << 32) + 5, (3 << 32) + 8])
        );
    }

    #[tokio::test]
    async fn test_recommit_from_file() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = create_dataset(test_uri, LanceFileVersion::Legacy).await;
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
            let f = FileFragment::create_from_file(path, &dataset, idx, None)
                .await
                .unwrap();
            fragments.push(f)
        }

        let op = Operation::Overwrite {
            schema: schema.clone(),
            fragments,
            config_upsert_values: None,
        };

        let registry = Arc::new(ObjectStoreRegistry::default());
        let new_dataset = Dataset::commit(test_uri, op, None, None, None, registry, false)
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
    async fn test_fragment_count(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = create_dataset(test_uri, data_storage_version).await;
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
    async fn test_append_new_columns(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        for with_delete in [true, false] {
            let test_dir = tempdir().unwrap();
            let test_uri = test_dir.path().to_str().unwrap();
            let mut dataset = create_dataset(test_uri, data_storage_version).await;
            dataset.validate().await.unwrap();
            assert_eq!(dataset.count_rows(None).await.unwrap(), 200);

            if with_delete {
                dataset.delete("i >= 15 and i < 20").await.unwrap();
                dataset.validate().await.unwrap();
                assert_eq!(dataset.count_rows(None).await.unwrap(), 195);
            }

            let fragment = &mut dataset.get_fragment(0).unwrap();
            let mut updater = fragment.updater(Some(&["i"]), None, None).await.unwrap();
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
                config_upsert_values: None,
            };

            let registry = Arc::new(ObjectStoreRegistry::default());
            let dataset = Dataset::commit(test_uri, op, None, None, None, registry, false)
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
    async fn test_merge_fragment(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = create_dataset(test_uri, data_storage_version).await;
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
        // V1 ONLY
        //
        // This test is only for the legacy version of the file format.
        // It ensures that the `max_rows_per_group` property is respected
        // and this property does not exist in V2.
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
                data_storage_version: Some(LanceFileVersion::Legacy),
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
        let mut updater = fragment.updater(Some(&["i"]), None, None).await?;
        updater.next().await?;
        updater.update(batch_s.clone()).await?;
        let frag = updater.finish().await?;

        // Rearrange schema so it's `s` then `i`.
        let schema = updater.schema().unwrap().clone().project(&["s", "i"])?;
        let registry = Arc::new(ObjectStoreRegistry::default());

        let dataset = Dataset::commit(
            test_uri,
            Operation::Merge {
                schema,
                fragments: vec![frag],
            },
            Some(dataset.manifest.version),
            None,
            None,
            registry,
            false,
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
            .open(dataset.schema(), FragReadConfig::default(), None)
            .await?;
        let actual_data = reader.take_as_batch(&[0, 1, 2]).await?;
        assert_eq!(expected_data.slice(0, 3), actual_data);

        let actual_data = reader
            .read_range(0..3, 3)
            .unwrap()
            .next()
            .await
            .unwrap()
            .await
            .unwrap();
        assert_eq!(expected_data.slice(0, 3), actual_data);

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
            .open(
                &dataset.schema().project::<&str>(&[])?,
                FragReadConfig::default().with_row_id(true),
                None,
            )
            .await?;
        let batch = reader.legacy_read_range_as_batch(0..20).await?;

        let expected_data = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ROW_ID_FIELD.clone()])),
            vec![Arc::new(UInt64Array::from_iter_values(0..20))],
        )?;
        assert_eq!(expected_data, batch);

        // We should get error if we pass empty schema and with_row_id false
        let res = fragment
            .open(
                &dataset.schema().project::<&str>(&[])?,
                FragReadConfig::default(),
                None,
            )
            .await;
        assert!(matches!(res, Err(Error::IO { .. })));

        Ok(())
    }

    #[tokio::test]
    async fn create_from_file_v2() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let make_gen = || {
            gen()
                .col("str", array::rand_type(&DataType::Utf8))
                .col("int", array::rand_type(&DataType::Int32))
        };

        let batch = make_gen().into_batch_rows(RowCount::from(128)).unwrap();
        let dataset = TestDatasetGenerator::new(vec![batch], LanceFileVersion::Stable)
            .make_hostile(test_uri)
            .await;

        let new_data = make_gen().into_batch_rows(RowCount::from(128)).unwrap();
        let store = ObjectStore::local();
        let file_path = dataset.data_dir().child("some_file.lance");
        let object_writer = store.create(&file_path).await.unwrap();
        let mut file_writer =
            v2::writer::FileWriter::new_lazy(object_writer, FileWriterOptions::default());
        file_writer.write_batch(&new_data).await.unwrap();
        file_writer.finish().await.unwrap();

        let frag = FileFragment::create_from_file("some_file.lance", &dataset, 0, Some(128))
            .await
            .unwrap();

        assert_eq!(
            Fragment::try_infer_version(&[frag.clone()])
                .unwrap()
                .unwrap(),
            LanceFileVersion::Stable.resolve()
        );

        let op = Operation::Append {
            fragments: vec![frag],
        };
        let object_store_registry = Arc::new(ObjectStoreRegistry::default());
        let dataset = Dataset::commit(
            &dataset.uri,
            op,
            Some(dataset.version().version),
            None,
            None,
            object_store_registry,
            false,
        )
        .await
        .unwrap();

        assert_eq!(
            dataset
                .count_rows(Some("int IS NOT NULL".to_string()))
                .await
                .unwrap(),
            256
        );
    }
}
