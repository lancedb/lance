// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shuffler is a component that takes a stream of record batches and shuffles them into
//! the corresponding IVF partitions.

use std::sync::Arc;

use arrow::{array::AsArray, compute::sort_to_indices};
use arrow_array::{RecordBatch, UInt32Array};
use future::join_all;
use futures::prelude::*;
use lance_arrow::RecordBatchExt;
use lance_core::{
    cache::FileMetadataCache,
    utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu},
    Error, Result,
};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::v2::{
    reader::{FileReader, FileReaderOptions},
    writer::FileWriter,
};
use lance_io::{
    object_store::ObjectStore,
    scheduler::{ScanScheduler, SchedulerConfig},
    stream::{RecordBatchStream, RecordBatchStreamAdapter},
};
use object_store::path::Path;

use crate::vector::PART_ID_COLUMN;

#[async_trait::async_trait]
/// A reader that can read the shuffled partitions.
pub trait ShuffleReader: Send + Sync {
    /// Read a partition by partition_id
    /// will return Ok(None) if partition_size is 0
    /// check reader.partiton_size(partition_id) before calling this function
    async fn read_partition(
        &self,
        partition_id: usize,
    ) -> Result<Option<Box<dyn RecordBatchStream + Unpin + 'static>>>;

    /// Get the size of the partition by partition_id
    fn partition_size(&self, partition_id: usize) -> Result<usize>;
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
    buffer_size: usize,
}

impl IvfShuffler {
    pub fn new(output_dir: Path, num_partitions: usize) -> Self {
        Self {
            object_store: Arc::new(ObjectStore::local()),
            output_dir,
            num_partitions,
            buffer_size: 4096,
        }
    }

    pub fn with_buffer_size(mut self, buffer_size: usize) -> Self {
        self.buffer_size = buffer_size;
        self
    }
}

#[async_trait::async_trait]
impl Shuffler for IvfShuffler {
    async fn shuffle(
        &self,
        data: Box<dyn RecordBatchStream + Unpin + 'static>,
    ) -> Result<Box<dyn ShuffleReader>> {
        let mut writers: Vec<FileWriter> = vec![];
        let mut partition_sizes = vec![0; self.num_partitions];
        let mut first_pass = true;

        let num_partitions = self.num_partitions;
        let mut parallel_sort_stream = data
            .map(|batch| {
                spawn_cpu(move || {
                    let batch = batch?;

                    let part_ids: &UInt32Array = batch
                        .column_by_name(PART_ID_COLUMN)
                        .expect("Partition ID column not found")
                        .as_primitive();

                    let indices = sort_to_indices(&part_ids, None, None)?;
                    let batch = batch.take(&indices)?;

                    let part_ids: &UInt32Array = batch
                        .column_by_name(PART_ID_COLUMN)
                        .expect("Partition ID column not found")
                        .as_primitive();

                    let mut partition_buffers =
                        (0..num_partitions).map(|_| Vec::new()).collect::<Vec<_>>();

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

                    Ok::<Vec<Vec<RecordBatch>>, Error>(partition_buffers)
                })
            })
            .buffered(get_num_compute_intensive_cpus());

        // part_id:           |       0        |       1        |       3        |
        // partition_buffers: |[batch,batch,..]|[batch,batch,..]|[batch,batch,..]|
        let mut partition_buffers = (0..self.num_partitions)
            .map(|_| Vec::new())
            .collect::<Vec<_>>();

        let mut counter = 0;
        while let Some(shuffled) = parallel_sort_stream.next().await {
            let shuffled = shuffled?;

            for (part_id, batches) in shuffled.into_iter().enumerate() {
                let part_batches = &mut partition_buffers[part_id];
                part_batches.extend(batches);
            }

            counter += 1;

            if first_pass {
                let schema = partition_buffers
                    .iter()
                    .flatten()
                    .find(|_| true)
                    .map(|batch| batch.schema())
                    .expect("there should be at least one batch");
                writers = stream::iter(0..self.num_partitions)
                    .map(|partition_id| {
                        let part_path =
                            self.output_dir.child(format!("ivf_{}.lance", partition_id));
                        let object_store = self.object_store.clone();
                        let schema = schema.clone();
                        async move {
                            let writer = object_store.create(&part_path).await?;
                            FileWriter::try_new(
                                writer,
                                lance_core::datatypes::Schema::try_from(schema.as_ref())?,
                                Default::default(),
                            )
                        }
                    })
                    .buffered(10)
                    .try_collect::<Vec<_>>()
                    .await?;

                first_pass = false;
            }

            // do flush
            if counter % self.buffer_size == 0 {
                log::info!("shuffle {} batches, flushing", counter);
                let mut futs = vec![];
                for (part_id, writer) in writers.iter_mut().enumerate() {
                    let batches = &partition_buffers[part_id];
                    partition_sizes[part_id] += batches.iter().map(|b| b.num_rows()).sum::<usize>();
                    futs.push(writer.write_batches(batches.iter()));
                }
                join_all(futs)
                    .await
                    .into_iter()
                    .collect::<Result<Vec<_>>>()?;

                partition_buffers.iter_mut().for_each(|b| b.clear());
            }
        }

        // final flush
        for (part_id, batches) in partition_buffers.into_iter().enumerate() {
            let writer = &mut writers[part_id];
            partition_sizes[part_id] += batches.iter().map(|b| b.num_rows()).sum::<usize>();
            for batch in batches.iter() {
                writer.write_batch(batch).await?;
            }
        }

        // finish all writers
        for writer in writers.iter_mut() {
            writer.finish().await?;
        }

        Ok(Box::new(IvfShufflerReader::new(
            self.object_store.clone(),
            self.output_dir.clone(),
            partition_sizes,
        )))
    }
}

pub struct IvfShufflerReader {
    scheduler: Arc<ScanScheduler>,
    output_dir: Path,
    partition_sizes: Vec<usize>,
}

impl IvfShufflerReader {
    pub fn new(
        object_store: Arc<ObjectStore>,
        output_dir: Path,
        partition_sizes: Vec<usize>,
    ) -> Self {
        let scheduler_config = SchedulerConfig::max_bandwidth(&object_store);
        let scheduler = ScanScheduler::new(object_store, scheduler_config);
        Self {
            scheduler,
            output_dir,
            partition_sizes,
        }
    }
}

#[async_trait::async_trait]
impl ShuffleReader for IvfShufflerReader {
    async fn read_partition(
        &self,
        partition_id: usize,
    ) -> Result<Option<Box<dyn RecordBatchStream + Unpin + 'static>>> {
        let partition_path = self.output_dir.child(format!("ivf_{}.lance", partition_id));

        let reader = FileReader::try_open(
            self.scheduler.open_file(&partition_path).await?,
            None,
            Arc::<DecoderPlugins>::default(),
            &FileMetadataCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await?;
        let schema = reader.schema().as_ref().into();

        Ok(Some(Box::new(RecordBatchStreamAdapter::new(
            Arc::new(schema),
            reader.read_stream(
                lance_io::ReadBatchParams::RangeFull,
                4096,
                16,
                FilterExpression::no_filter(),
            )?,
        ))))
    }

    fn partition_size(&self, partition_id: usize) -> Result<usize> {
        Ok(self.partition_sizes[partition_id])
    }
}
