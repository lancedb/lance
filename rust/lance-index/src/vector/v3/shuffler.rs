// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shuffler is a component that takes a stream of record batches and shuffles them into
//! the corresponding IVF partitions.

use std::sync::Arc;

use arrow::{array::AsArray, compute::sort_to_indices};
use arrow_array::{RecordBatch, UInt32Array};
use arrow_schema::Schema;
use futures::{future::try_join_all, prelude::*};
use lance_arrow::{RecordBatchExt, SchemaExt};
use lance_core::{
    cache::LanceCache,
    utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu},
    Error, Result,
};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_file::writer::FileWriter;
use lance_io::{
    object_store::ObjectStore,
    scheduler::{ScanScheduler, SchedulerConfig},
    stream::{RecordBatchStream, RecordBatchStreamAdapter},
    utils::CachedFileSize,
};
use object_store::path::Path;

use crate::vector::{LOSS_METADATA_KEY, PART_ID_COLUMN};

#[async_trait::async_trait]
/// A reader that can read the shuffled partitions.
pub trait ShuffleReader: Send + Sync {
    /// Read a partition by partition_id
    /// will return Ok(None) if partition_size is 0
    /// check reader.partition_size(partition_id) before calling this function
    async fn read_partition(
        &self,
        partition_id: usize,
    ) -> Result<Option<Box<dyn RecordBatchStream + Unpin + 'static>>>;

    /// Get the size of the partition by partition_id
    fn partition_size(&self, partition_id: usize) -> Result<usize>;

    /// Get the total loss,
    /// if the loss is not available, return None,
    /// in such case, the caller should sum up the losses from each batch's metadata.
    /// Must be called after all partitions are read.
    fn total_loss(&self) -> Option<f64>;
}

#[async_trait::async_trait]
/// A shuffler that can shuffle the incoming stream of record batches into IVF partitions.
/// Returns a IvfShuffleReader that can be used to read the shuffled partitions.
pub trait Shuffler: Send + Sync {
    /// Shuffle the incoming stream of record batches into IVF partitions.
    /// Returns a IvfShuffleReader that can be used to read the shuffled partitions.
    async fn shuffle(
        &self,
        data: Box<dyn RecordBatchStream + Unpin + 'static>,
    ) -> Result<Box<dyn ShuffleReader>>;
}

pub struct IvfShuffler {
    object_store: Arc<ObjectStore>,
    output_dir: Path,
    num_partitions: usize,

    // options
    precomputed_shuffle_buffers: Option<Vec<String>>,
    progress: Arc<dyn crate::progress::IndexBuildProgress>,
}

impl IvfShuffler {
    pub fn new(output_dir: Path, num_partitions: usize) -> Self {
        Self {
            object_store: Arc::new(ObjectStore::local()),
            output_dir,
            num_partitions,
            precomputed_shuffle_buffers: None,
            progress: crate::progress::noop_progress(),
        }
    }

    pub fn with_progress(mut self, progress: Arc<dyn crate::progress::IndexBuildProgress>) -> Self {
        self.progress = progress;
        self
    }

    pub fn with_precomputed_shuffle_buffers(
        mut self,
        precomputed_shuffle_buffers: Option<Vec<String>>,
    ) -> Self {
        self.precomputed_shuffle_buffers = precomputed_shuffle_buffers;
        self
    }
}

#[async_trait::async_trait]
impl Shuffler for IvfShuffler {
    async fn shuffle(
        &self,
        data: Box<dyn RecordBatchStream + Unpin + 'static>,
    ) -> Result<Box<dyn ShuffleReader>> {
        let num_partitions = self.num_partitions;
        let mut partition_sizes = vec![0; num_partitions];
        let schema = data.schema().without_column(PART_ID_COLUMN);
        let mut writers = stream::iter(0..num_partitions)
            .map(|partition_id| {
                let part_path = self.output_dir.child(format!("ivf_{}.lance", partition_id));
                let spill_path = self.output_dir.child(format!("ivf_{}.spill", partition_id));
                let object_store = self.object_store.clone();
                let schema = schema.clone();
                async move {
                    let writer = object_store.create(&part_path).await?;
                    let file_writer = FileWriter::try_new(
                        writer,
                        lance_core::datatypes::Schema::try_from(&schema)?,
                        Default::default(),
                    )?
                    .with_page_metadata_spill(object_store.clone(), spill_path);
                    Result::Ok(file_writer)
                }
            })
            .buffered(self.object_store.io_parallelism())
            .try_collect::<Vec<_>>()
            .await?;
        let mut parallel_sort_stream = data
            .map(|batch| {
                spawn_cpu(move || {
                    let batch = batch?;

                    let loss = batch
                        .metadata()
                        .get(LOSS_METADATA_KEY)
                        .map(|s| s.parse::<f64>().unwrap_or_default())
                        .unwrap_or_default();

                    let part_ids: &UInt32Array = batch[PART_ID_COLUMN].as_primitive();

                    let indices = sort_to_indices(&part_ids, None, None)?;
                    let batch = batch.take(&indices)?;

                    let part_ids: &UInt32Array = batch[PART_ID_COLUMN].as_primitive();
                    let batch = batch.drop_column(PART_ID_COLUMN)?;

                    let mut partition_buffers = vec![Vec::new(); num_partitions];

                    let mut start = 0;
                    while start < batch.num_rows() {
                        let part_id: u32 = part_ids.value(start);
                        let mut end = start + 1;
                        while end < batch.num_rows() && part_ids.value(end) == part_id {
                            end += 1;
                        }

                        let part_batches = &mut partition_buffers[part_id as usize];
                        part_batches.push(batch.slice(start, end - start));
                        start = end;
                    }

                    Ok::<(Vec<Vec<RecordBatch>>, f64), Error>((partition_buffers, loss))
                })
            })
            .buffered(get_num_compute_intensive_cpus());

        let mut total_loss = 0.0;
        let mut counter: u64 = 0;
        while let Some(shuffled) = parallel_sort_stream.next().await {
            let (shuffled, loss) = shuffled?;
            total_loss += loss;

            let mut futs = Vec::new();
            for (part_id, (writer, batches)) in writers.iter_mut().zip(shuffled.iter()).enumerate()
            {
                if !batches.is_empty() {
                    partition_sizes[part_id] += batches.iter().map(|b| b.num_rows()).sum::<usize>();
                    futs.push(writer.write_batches(batches.iter()));
                }
            }
            try_join_all(futs).await?;

            counter += 1;
            self.progress.stage_progress("shuffle", counter).await?;
        }

        // finish all writers
        for writer in writers.iter_mut() {
            writer.finish().await?;
        }

        Ok(Box::new(IvfShufflerReader::new(
            self.object_store.clone(),
            self.output_dir.clone(),
            partition_sizes,
            total_loss,
        )))
    }
}

pub struct IvfShufflerReader {
    scheduler: Arc<ScanScheduler>,
    output_dir: Path,
    partition_sizes: Vec<usize>,
    loss: f64,
}

impl IvfShufflerReader {
    pub fn new(
        object_store: Arc<ObjectStore>,
        output_dir: Path,
        partition_sizes: Vec<usize>,
        loss: f64,
    ) -> Self {
        let scheduler_config = SchedulerConfig::max_bandwidth(&object_store);
        let scheduler = ScanScheduler::new(object_store, scheduler_config);
        Self {
            scheduler,
            output_dir,
            partition_sizes,
            loss,
        }
    }
}

#[async_trait::async_trait]
impl ShuffleReader for IvfShufflerReader {
    async fn read_partition(
        &self,
        partition_id: usize,
    ) -> Result<Option<Box<dyn RecordBatchStream + Unpin + 'static>>> {
        if partition_id >= self.partition_sizes.len() {
            return Ok(None);
        }

        let partition_path = self.output_dir.child(format!("ivf_{}.lance", partition_id));

        let reader = FileReader::try_open(
            self.scheduler
                .open_file(&partition_path, &CachedFileSize::unknown())
                .await?,
            None,
            Arc::<DecoderPlugins>::default(),
            &LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await?;
        let schema: Schema = reader.schema().as_ref().into();
        Ok(Some(Box::new(RecordBatchStreamAdapter::new(
            Arc::new(schema),
            reader.read_stream(
                lance_io::ReadBatchParams::RangeFull,
                u32::MAX,
                16,
                FilterExpression::no_filter(),
            )?,
        ))))
    }

    fn partition_size(&self, partition_id: usize) -> Result<usize> {
        Ok(self.partition_sizes.get(partition_id).copied().unwrap_or(0))
    }

    fn total_loss(&self) -> Option<f64> {
        Some(self.loss)
    }
}

pub struct EmptyReader;

#[async_trait::async_trait]
impl ShuffleReader for EmptyReader {
    async fn read_partition(
        &self,
        _partition_id: usize,
    ) -> Result<Option<Box<dyn RecordBatchStream + Unpin + 'static>>> {
        Ok(None)
    }

    fn partition_size(&self, _partition_id: usize) -> Result<usize> {
        Ok(0)
    }

    fn total_loss(&self) -> Option<f64> {
        None
    }
}
