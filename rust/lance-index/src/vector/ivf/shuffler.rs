// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Disk-based shuffle a stream of [RecordBatch] into each IVF partition.
//!
//! 1. write the entire stream to a file
//! 2. count the number of rows in each partition
//! 3. read the data back into memory and shuffle into grouped vectors
//!
//! Problems for the future:
//! 1. while groupby column will stay the same, we may want to include extra data columns in the future
//! 2. shuffling into memory is fast but we should add disk buffer to support bigger datasets

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{
    ArrayBuilder, FixedSizeListBuilder, StructBuilder, UInt32Builder, UInt64Builder, UInt8Builder,
};
use arrow::buffer::{OffsetBuffer, ScalarBuffer};
use arrow::compute::sort_to_indices;
use arrow::datatypes::UInt32Type;
use arrow_array::{cast::AsArray, types::UInt64Type, Array, RecordBatch, UInt32Array};
use arrow_array::{FixedSizeListArray, ListArray, StructArray, UInt64Array, UInt8Array};
use arrow_schema::{DataType, Field, Fields, SchemaRef};
use futures::stream::repeat_with;
use futures::{stream, FutureExt, Stream, StreamExt, TryStreamExt};
use lance_arrow::RecordBatchExt;
use lance_core::utils::address::RowAddress;
use lance_core::{datatypes::Schema, Error, Result, ROW_ID};
use lance_encoding::decoder::{DecoderMiddlewareChain, FilterExpression};
use lance_file::reader::FileReader;
use lance_file::v2::writer::FileWriterOptions;
use lance_file::writer::FileWriter;
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::ScanScheduler;
use lance_io::stream::RecordBatchStream;
use lance_io::ReadBatchParams;
use lance_table::format::SelfDescribingFileReader;
use lance_table::io::manifest::ManifestDescribing;
use log::info;
use object_store::path::Path;
use snafu::{location, Location};
use tempfile::TempDir;

use crate::vector::ivf::Ivf;
use crate::vector::transform::{KeepFiniteVectors, Transformer};
use crate::vector::PART_ID_COLUMN;

const UNSORTED_BUFFER: &str = "unsorted.lance";

fn get_temp_dir() -> Result<Path> {
    let dir = TempDir::new()?;
    let tmp_dir_path = Path::from_filesystem_path(dir.path()).map_err(|e| Error::IO {
        source: Box::new(e),
        location: location!(),
    })?;
    Ok(tmp_dir_path)
}

/// A builder for a partition of data
///
/// After we sort a batch of data into partitions we append those slices into this builder.
///
/// The builder is pre-allocated and so this extend operation should only be a memcpy
#[derive(Debug)]
struct PartitionBuilder {
    builder: StructBuilder,
}

// Fork of arrow_array::builder::make_builder that handles FixedSizeList >_<
//
// Not really suitable for upstreaming because FixedSizeListBuilder<Box<dyn ArrayBuilder>> is
// awkward and the entire make_builder function needs some overhaul (dyn ArrayBuilder should have
// an extend(array: &dyn Array) method).
fn make_builder(datatype: &DataType, capacity: usize) -> Box<dyn ArrayBuilder> {
    if let DataType::FixedSizeList(inner, dim) = datatype {
        let inner_builder =
            arrow_array::builder::make_builder(inner.data_type(), capacity * (*dim) as usize);
        Box::new(FixedSizeListBuilder::new(inner_builder, *dim))
    } else {
        arrow_array::builder::make_builder(datatype, capacity)
    }
}

// Fork of StructBuilder::from_fields that handles FixedSizeList >_<
fn from_fields(fields: impl Into<Fields>, capacity: usize) -> StructBuilder {
    let fields = fields.into();
    let mut builders = Vec::with_capacity(fields.len());
    for field in &fields {
        builders.push(make_builder(field.data_type(), capacity));
    }
    StructBuilder::new(fields, builders)
}

impl PartitionBuilder {
    fn new(schema: &arrow_schema::Schema, initial_capacity: usize) -> Self {
        let builder = from_fields(schema.fields.clone(), initial_capacity);
        Self { builder }
    }

    fn extend(&mut self, batch: &RecordBatch) {
        for _ in 0..batch.num_rows() {
            self.builder.append(true);
        }
        let schema = batch.schema_ref();
        for (field_idx, (col, field)) in batch.columns().iter().zip(schema.fields()).enumerate() {
            match field.data_type() {
                DataType::UInt32 => {
                    let col = col.as_any().downcast_ref::<UInt32Array>().unwrap();
                    self.builder
                        .field_builder::<UInt32Builder>(field_idx)
                        .unwrap()
                        .append_slice(col.values());
                }
                DataType::UInt64 => {
                    let col = col.as_any().downcast_ref::<UInt64Array>().unwrap();
                    self.builder
                        .field_builder::<UInt64Builder>(field_idx)
                        .unwrap()
                        .append_slice(col.values());
                }
                DataType::FixedSizeList(inner, _) => {
                    let col = col.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
                    match inner.data_type() {
                        DataType::UInt8 => {
                            let values =
                                col.values().as_any().downcast_ref::<UInt8Array>().unwrap();
                            let fsl_builder = self
                                .builder
                                .field_builder::<FixedSizeListBuilder<Box<dyn ArrayBuilder>>>(
                                    field_idx,
                                )
                                .unwrap();
                            // TODO: Upstream an append_many to FSL builder
                            for _ in 0..col.len() {
                                fsl_builder.append(true);
                            }
                            fsl_builder
                                .values()
                                .as_any_mut()
                                .downcast_mut::<UInt8Builder>()
                                .unwrap()
                                .append_slice(values.values());
                        }
                        _ => panic!("Unexpected fixed size list item type in shuffled file"),
                    }
                }
                _ => panic!("Unexpected column type in shuffled file"),
            }
        }
    }

    // Convert the partition builder into a list array with 1 row
    fn finish(mut self) -> Result<ListArray> {
        let struct_array = Arc::new(self.builder.finish());

        let item_field = Arc::new(Field::new("item", struct_array.data_type().clone(), true));

        Ok(ListArray::try_new(
            item_field,
            OffsetBuffer::new(ScalarBuffer::<i32>::from(vec![
                0,
                struct_array.len() as i32,
            ])),
            struct_array,
            None,
        )?)
    }
}

struct PartitionListBuilder {
    partitions: Vec<Option<PartitionBuilder>>,
}

impl PartitionListBuilder {
    fn new(schema: &arrow_schema::Schema, partition_sizes: Vec<u64>) -> Self {
        Self {
            partitions: Vec::from_iter(partition_sizes.into_iter().map(|part_size| {
                if part_size == 0 {
                    None
                } else {
                    Some(PartitionBuilder::new(schema, part_size as usize))
                }
            })),
        }
    }

    fn extend(&mut self, batch: &RecordBatch) {
        if batch.num_rows() == 0 {
            return;
        }

        let part_ids = batch
            .column_by_name(PART_ID_COLUMN)
            .unwrap()
            .as_primitive::<UInt32Type>();

        let part_id = part_ids.value(0) as usize;

        let builder = &mut self.partitions[part_id];
        builder
            .as_mut()
            .expect("partition size was zero but received data for partition")
            .extend(batch);
    }

    fn finish(self) -> Result<Vec<ListArray>> {
        self.partitions
            .into_iter()
            .filter_map(|builder| builder.map(|builder| builder.finish()))
            .collect()
    }
}

/// Disk-based shuffle for a stream of [RecordBatch] into each IVF partition.
/// Sub-quantizer will be applied if provided.
///
/// Parameters
/// ----------
///   *data*: input data stream.
///   *column*: column name of the vector column.
///   *ivf*: IVF model.
///   *num_partitions*: number of IVF partitions.
///   *num_sub_vectors*: number of PQ sub-vectors.
///
/// Returns
/// -------
///   Result<Vec<impl Stream<Item = Result<RecordBatch>>>>: a vector of streams
///   of shuffled partitioned data. Each stream corresponds to a partition and
///   is sorted within the stream. Consumer of these streams is expected to merge
///   the streams into a single stream by k-list merge algo.
///
#[allow(clippy::too_many_arguments)]
pub async fn shuffle_dataset(
    data: impl RecordBatchStream + Unpin + 'static,
    column: &str,
    ivf: Arc<Ivf>,
    precomputed_partitions: Option<Vec<Vec<u32>>>,
    num_partitions: u32,
    shuffle_partition_batches: usize,
    shuffle_partition_concurrency: usize,
    precomputed_shuffle_buffers: Option<(Path, Vec<String>)>,
) -> Result<Vec<impl Stream<Item = Result<RecordBatch>>>> {
    // step 1: either use precomputed shuffle files or write shuffle data to a file
    let shuffler = if let Some((path, buffers)) = precomputed_shuffle_buffers {
        info!("Precomputed shuffle files provided, skip calculation of IVF partition.");
        let mut shuffler = IvfShuffler::try_new(num_partitions, Some(path))?;
        unsafe {
            shuffler.set_unsorted_buffers(&buffers);
        }

        shuffler
    } else {
        info!(
            "Calculating IVF partitions for vectors (num_partitions={})",
            num_partitions
        );
        let mut shuffler = IvfShuffler::try_new(num_partitions, None)?;

        let column = column.to_owned();
        let precomputed_partitions = precomputed_partitions.map(Arc::new);
        let stream =
            data.zip(repeat_with(move || ivf.clone()))
                .map(move |(b, ivf)| {
                    // If precomputed_partitions map is provided, use it
                    // for fast partitions.
                    let partition_map = precomputed_partitions
                        .as_ref()
                        .cloned()
                        .unwrap_or(Arc::new(Vec::new()));
                    let nan_filter = KeepFiniteVectors::new(&column);

                    tokio::task::spawn(async move {
                        let mut batch = b?;

                        if !partition_map.is_empty() {
                            let row_ids = batch.column_by_name(ROW_ID).ok_or(Error::Index {
                                message: "column does not exist".to_string(),
                                location: location!(),
                            })?;
                            let part_ids = UInt32Array::from_iter(
                                row_ids.as_primitive::<UInt64Type>().values().iter().map(
                                    |row_id| {
                                        let row_addr = RowAddress::new_from_id(*row_id);
                                        partition_map[row_addr.fragment_id() as usize]
                                            [row_addr.row_id() as usize]
                                    },
                                ),
                            );
                            let part_ids = UInt32Array::from(part_ids);
                            batch = batch
                                .try_with_column(
                                    Field::new(PART_ID_COLUMN, part_ids.data_type().clone(), true),
                                    Arc::new(part_ids.clone()),
                                )
                                .expect("failed to add part id column");

                            if part_ids.null_count() > 0 {
                                info!(
                                    "Filter out rows without valid partition IDs: null_count={}",
                                    part_ids.null_count()
                                );
                                let indices = UInt32Array::from_iter(
                                    part_ids
                                        .iter()
                                        .enumerate()
                                        .filter_map(|(idx, v)| v.map(|_| idx as u32)),
                                );
                                assert_eq!(indices.len(), batch.num_rows() - part_ids.null_count());
                                batch = batch.take(&indices)?;
                            }
                        }

                        // Filter out NaNs/Infs
                        let minibatch_size = std::env::var("LANCE_SHUFFLE_BATCH_SIZE")
                            .unwrap_or("64".to_string())
                            .parse::<usize>()
                            .unwrap_or(64);
                        let num_out_batches = batch.num_rows().div_ceil(minibatch_size);
                        let mut out_batches = Vec::with_capacity(num_out_batches);
                        for offset in (0..batch.num_rows()).step_by(minibatch_size) {
                            let mut batch = batch.slice(
                                offset,
                                std::cmp::min(minibatch_size, batch.num_rows() - offset),
                            );
                            batch = nan_filter.transform(&batch)?;
                            out_batches.push(Result::Ok(ivf.transform(&batch))?);
                        }
                        Result::Ok(futures::stream::iter(out_batches))
                    })
                })
                .buffer_unordered(num_cpus::get())
                .map(|res| match res {
                    Ok(Ok(batch)) => Ok(batch),
                    Ok(Err(err)) => Err(Error::io(err.to_string(), location!())),
                    Err(err) => Err(Error::io(err.to_string(), location!())),
                })
                .try_flatten()
                .boxed();

        let start = std::time::Instant::now();
        shuffler.write_unsorted_stream(stream).await?;
        info!(
            "wrote partition assignment to unsorted tmp file in {:?}",
            start.elapsed()
        );

        shuffler
    };

    // step 2: stream in the shuffle data in chunks and write sorted chuncks out
    let start = std::time::Instant::now();
    let partition_files = shuffler
        .write_partitioned_shuffles(shuffle_partition_batches, shuffle_partition_concurrency)
        .await?;
    info!("created sorted chunks in {:?}", start.elapsed());

    // step 3: load the sorted chunks, consumers are expect to be responsible for merging the streams
    let start = std::time::Instant::now();
    let stream = shuffler.load_partitioned_shuffles(partition_files).await?;
    info!("merged partitioned shuffles in {:?}", start.elapsed());

    Ok(stream)
}

#[derive(Clone)]
pub struct IvfShuffler {
    unsorted_buffers: Vec<String>,

    num_partitions: u32,

    // The schema of the partition files.  Only initialized after the unsorted parts are written
    schema: Option<SchemaRef>,

    output_dir: Path,
}

/// Represents a range of batches in a file that should be shuffled
struct ShuffleInput {
    // the idx of the file in IvfShuffler::unsorted_buffers
    file_idx: usize,
    // the start index of the batch in the file
    start: usize,
    // the end index of the batch in the file
    end: usize,
}

impl IvfShuffler {
    pub fn try_new(num_partitions: u32, output_dir: Option<Path>) -> Result<Self> {
        let output_dir = match output_dir {
            Some(output_dir) => output_dir,
            None => get_temp_dir()?,
        };

        Ok(Self {
            num_partitions,
            output_dir,
            unsorted_buffers: vec![],
            schema: None,
        })
    }

    /// Set the unsorted buffers to be shuffled.
    ///
    /// # Safety
    ///
    /// user must ensure the buffers are valid.
    pub unsafe fn set_unsorted_buffers(&mut self, unsorted_buffers: &[impl ToString]) {
        self.unsorted_buffers = unsorted_buffers.iter().map(|x| x.to_string()).collect();
    }

    pub async fn write_unsorted_stream(
        &mut self,
        data: impl Stream<Item = Result<RecordBatch>>,
    ) -> Result<()> {
        let object_store = ObjectStore::local();
        let path = self.output_dir.child(UNSORTED_BUFFER);
        let writer = object_store.create(&path).await?;

        let mut data = Box::pin(data.peekable());
        let schema = match data.as_mut().peek().await {
            Some(Ok(batch)) => batch.schema(),
            Some(Err(err)) => {
                return Err(Error::io(err.to_string(), location!()));
            }
            None => {
                return Err(Error::io("empty stream".to_string(), location!()));
            }
        };
        self.schema = Some(schema.clone());

        // validate the schema,
        // we need to have row ID and partition ID column
        schema
            .column_with_name(ROW_ID)
            .ok_or(Error::io("row ID column not found".to_owned(), location!()))?;
        schema.column_with_name(PART_ID_COLUMN).ok_or(Error::io(
            "partition ID column not found".to_owned(),
            location!(),
        ))?;

        let mut file_writer = FileWriter::<ManifestDescribing>::with_object_writer(
            writer,
            Schema::try_from(schema.as_ref())?,
            &Default::default(),
        )?;

        let mut batches_processed = 0;
        while let Some(batch) = data.next().await {
            if batches_processed % 1000 == 0 {
                info!("Partition assignment progress {}/?", batches_processed);
            }
            batches_processed += 1;
            file_writer.write(&[batch?]).await?;
        }

        file_writer.finish().await?;

        unsafe {
            self.set_unsorted_buffers(&[UNSORTED_BUFFER]);
        }

        Ok(())
    }

    async fn total_batches(&self) -> Result<Vec<usize>> {
        let mut total_batches = vec![];
        for buffer in &self.unsorted_buffers {
            let object_store = ObjectStore::local();
            let path = self.output_dir.child(buffer.as_str());
            let reader = FileReader::try_new_self_described(&object_store, &path, None).await?;
            total_batches.push(reader.num_batches());
        }
        Ok(total_batches)
    }

    async fn count_partition_size(&self, inputs: &[ShuffleInput]) -> Result<Vec<u64>> {
        let object_store = ObjectStore::local();
        let mut partition_sizes = vec![0; self.num_partitions as usize];

        for &ShuffleInput {
            file_idx,
            start,
            end,
        } in inputs
        {
            let file_name = &self.unsorted_buffers[file_idx];
            let path = self.output_dir.child(file_name.as_str());
            let reader = FileReader::try_new_self_described(&object_store, &path, None).await?;
            let lance_schema = reader
                .schema()
                .project(&[PART_ID_COLUMN])
                .expect("part id should exist");

            let mut stream = stream::iter(start..end)
                .map(|i| reader.read_batch(i as i32, .., &lance_schema))
                .buffer_unordered(16);

            while let Some(batch) = stream.next().await {
                let batch = batch?;
                let part_ids: &UInt32Array = batch.column(0).as_primitive();
                part_ids.values().iter().for_each(|part_id| {
                    partition_sizes[*part_id as usize] += 1;
                });
            }
        }

        Ok(partition_sizes)
    }

    async fn shuffle_to_partitions(
        &self,
        schema: &arrow_schema::Schema,
        inputs: &[ShuffleInput],
        partition_size: Vec<u64>,
        num_batches_to_sort: usize,
    ) -> Result<Vec<ListArray>> {
        info!("Shuffling into memory");

        let mut num_processed = 0;
        let mut partitions_builder = PartitionListBuilder::new(schema, partition_size);

        for &ShuffleInput {
            file_idx,
            start,
            end,
        } in inputs
        {
            let object_store = ObjectStore::local();
            let file_name = &self.unsorted_buffers[file_idx];
            let path = self.output_dir.child(file_name.as_str());
            let reader = FileReader::try_new_self_described(&object_store, &path, None).await?;

            let mut stream = stream::iter(start..end)
                .map(|i| reader.read_batch(i as i32, ReadBatchParams::RangeFull, reader.schema()))
                .buffered(16);

            while let Some(batch) = stream.next().await {
                if num_processed % 100 == 0 {
                    info!("Shuffle Progress {}/{}", num_processed, num_batches_to_sort);
                }
                num_processed += 1;

                let batch = batch?;

                // skip empty batches
                if batch.num_rows() == 0 {
                    continue;
                }

                let part_ids: &UInt32Array = batch
                    .column_by_name(PART_ID_COLUMN)
                    .expect("Partition ID column not found")
                    .as_primitive();
                let indices = sort_to_indices(&part_ids, None, None)?;
                let batch = batch.take(&indices)?;

                let mut start = 0;
                let mut prev_id = part_ids.value(0);
                for (idx, part_id) in batch[PART_ID_COLUMN]
                    .as_primitive::<UInt32Type>()
                    .values()
                    .iter()
                    .enumerate()
                {
                    if *part_id != prev_id {
                        partitions_builder.extend(&batch.slice(start, idx - start));
                        start = idx;
                        prev_id = *part_id;
                    }
                }
                partitions_builder.extend(&batch.slice(start, part_ids.len() - start));
            }
        }

        partitions_builder.finish()
    }

    pub async fn write_partitioned_shuffles(
        &self,
        batches_per_partition: usize,
        concurrent_jobs: usize,
    ) -> Result<Vec<String>> {
        let num_batches = self.total_batches().await?;
        let total_batches = num_batches.iter().sum();
        let schema = self
            .schema
            .as_ref()
            .expect("unsorted data not written yet so schema not initialized")
            .clone();

        info!(
            "Sorting unsorted data into sorted chunks (batches_per_chunk={} concurrent_jobs={})",
            batches_per_partition, concurrent_jobs
        );
        stream::iter((0..total_batches).step_by(batches_per_partition))
            .zip(stream::repeat(num_batches))
            .map(|(i, num_batches)| {
                let schema = schema.clone();
                let this = self.clone();
                tokio::spawn(async move {
                    let schema = schema.clone();
                    // first, calculate which files and ranges needs to be processed
                    let start = i;
                    let end = std::cmp::min(i + batches_per_partition, total_batches);
                    let num_batches_to_sort = end - start;
                    let mut input = vec![];

                    let mut cumulative_size = 0;
                    for (file_idx, partition_size) in num_batches.iter().enumerate() {
                        let cur_start = cumulative_size;
                        let cur_end = cumulative_size + partition_size;

                        cumulative_size += partition_size;

                        let should_include_file = start < cur_end && end > cur_start;

                        if !should_include_file {
                            continue;
                        }

                        // the currnet part doesn't overlap with the current batch
                        if start >= cur_end {
                            continue;
                        }

                        let local_start = if start < cur_start {
                            0
                        } else {
                            start - cur_start
                        };
                        let local_end = std::cmp::min(end - cur_start, *partition_size);

                        input.push(ShuffleInput {
                            file_idx,
                            start: local_start,
                            end: local_end,
                        });
                    }

                    // second, count the number of rows in each partition
                    let size_counts = this.count_partition_size(&input).await?;

                    // third, shuffle the data into each partition
                    let shuffled = this
                        .shuffle_to_partitions(
                            schema.as_ref(),
                            &input,
                            size_counts,
                            num_batches_to_sort,
                        )
                        .await?;

                    // finally, write the shuffled data to disk
                    let object_store = ObjectStore::local();
                    let output_file = format!("sorted_{}.lance", i);
                    let path = this.output_dir.child(output_file.clone());
                    let writer = object_store.create(&path).await?;

                    info!(
                        "Chunk loaded into memory and sorted, writing to disk at {}",
                        path
                    );

                    let sorted_file_schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
                        "partitions",
                        shuffled.first().unwrap().data_type().clone(),
                        true,
                    )]));
                    let lance_schema = Schema::try_from(sorted_file_schema.as_ref())?;
                    let mut file_writer = lance_file::v2::writer::FileWriter::try_new(
                        writer,
                        path.to_string(),
                        lance_schema,
                        FileWriterOptions::default(),
                    )?;

                    for partition_and_idx in shuffled.into_iter().enumerate() {
                        let (idx, partition) = partition_and_idx;
                        if idx % 1000 == 0 {
                            info!("Writing partition {}/{}", idx, this.num_partitions);
                        }
                        let batch = RecordBatch::try_new(
                            sorted_file_schema.clone(),
                            vec![Arc::new(partition)],
                        )?;
                        file_writer.write_batch(&batch).await?;
                    }

                    file_writer.finish().await?;

                    Ok(output_file) as Result<String>
                })
                .map(|join_res| join_res.unwrap())
            })
            .buffered(concurrent_jobs)
            .try_collect()
            .await
    }

    pub async fn load_partitioned_shuffles(
        &self,
        files: Vec<String>,
    ) -> Result<Vec<impl Stream<Item = Result<RecordBatch>>>> {
        // impl RecordBatchStream
        let mut streams = vec![];

        for file in files {
            let object_store = Arc::new(ObjectStore::local());
            let path = self.output_dir.child(file);
            let scan_scheduler = ScanScheduler::new(object_store, 16);
            let file_scheduler = scan_scheduler.open_file(&path).await?;
            let reader = lance_file::v2::reader::FileReader::try_open(
                file_scheduler,
                None,
                DecoderMiddlewareChain::default(),
            )
            .await?;
            let stream = reader
                .read_stream(
                    ReadBatchParams::RangeFull,
                    1,
                    32,
                    FilterExpression::no_filter(),
                )?
                .and_then(|batch| {
                    let list_array = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .expect("ListArray expected");
                    let struct_array = list_array
                        .values()
                        .as_any()
                        .downcast_ref::<StructArray>()
                        .expect("StructArray expected")
                        .clone();
                    let batch: RecordBatch = struct_array.into();
                    std::future::ready(Ok(batch))
                });

            streams.push(stream);
        }

        Ok(streams)
    }
}

#[cfg(test)]
mod test {
    use arrow_array::{
        types::{UInt32Type, UInt8Type},
        FixedSizeListArray, UInt64Array, UInt8Array,
    };
    use arrow_schema::DataType;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::ROW_ID_FIELD;
    use lance_io::stream::RecordBatchStreamAdapter;
    use rand::RngCore;

    use crate::vector::PQ_CODE_COLUMN;

    use super::*;

    fn make_schema(pq_dim: u32) -> Arc<arrow_schema::Schema> {
        Arc::new(arrow_schema::Schema::new(vec![
            ROW_ID_FIELD.clone(),
            arrow_schema::Field::new(PART_ID_COLUMN, DataType::UInt32, true),
            arrow_schema::Field::new(
                PQ_CODE_COLUMN,
                DataType::FixedSizeList(
                    Arc::new(arrow_schema::Field::new("item", DataType::UInt8, true)),
                    pq_dim as i32,
                ),
                false,
            ),
        ]))
    }

    fn make_stream_and_shuffler(
        include_empty_batches: bool,
    ) -> (impl RecordBatchStream, IvfShuffler) {
        let schema = make_schema(32);

        let schema2 = schema.clone();

        let stream =
            stream::iter(0..if include_empty_batches { 101 } else { 100 }).map(move |idx| {
                if include_empty_batches && idx == 100 {
                    return Ok(RecordBatch::try_new(
                        schema2.clone(),
                        vec![
                            Arc::new(UInt64Array::from_iter_values([])),
                            Arc::new(UInt32Array::from_iter_values([])),
                            Arc::new(
                                FixedSizeListArray::try_new_from_values(
                                    Arc::new(UInt8Array::from_iter_values([])) as Arc<dyn Array>,
                                    32,
                                )
                                .unwrap(),
                            ),
                        ],
                    )
                    .unwrap());
                }
                let row_ids = Arc::new(UInt64Array::from_iter(idx * 1024..(idx + 1) * 1024));

                let part_id = Arc::new(UInt32Array::from_iter(
                    (idx * 1024..(idx + 1) * 1024).map(|_| idx as u32),
                ));

                let values = Arc::new(UInt8Array::from_iter((0..32 * 1024).map(|_| idx as u8)));
                let pq_codes = Arc::new(
                    FixedSizeListArray::try_new_from_values(values as Arc<dyn Array>, 32).unwrap(),
                );

                Ok(
                    RecordBatch::try_new(schema2.clone(), vec![row_ids, part_id, pq_codes])
                        .unwrap(),
                )
            });

        let stream = RecordBatchStreamAdapter::new(schema.clone(), stream);

        let shuffler = IvfShuffler::try_new(100, None).unwrap();

        (stream, shuffler)
    }

    fn check_batch(batch: RecordBatch, idx: usize, num_rows: usize) {
        let row_ids = batch
            .column_by_name(ROW_ID)
            .unwrap()
            .as_primitive::<UInt64Type>();
        let part_ids = batch
            .column_by_name(PART_ID_COLUMN)
            .unwrap()
            .as_primitive::<UInt32Type>();
        let pq_codes = batch
            .column_by_name(PQ_CODE_COLUMN)
            .unwrap()
            .as_fixed_size_list()
            .values()
            .as_primitive::<UInt8Type>();

        assert_eq!(row_ids.len(), num_rows);
        assert_eq!(part_ids.len(), num_rows);
        assert_eq!(pq_codes.len(), num_rows * 32);

        for i in 0..num_rows {
            assert_eq!(part_ids.value(i), idx as u32);
        }

        for v in pq_codes.values() {
            assert_eq!(*v, idx as u8);
        }
    }

    #[tokio::test]
    async fn test_shuffler_single_partition() {
        let (stream, mut shuffler) = make_stream_and_shuffler(false);

        shuffler.write_unsorted_stream(stream).await.unwrap();
        let partition_files = shuffler.write_partitioned_shuffles(100, 1).await.unwrap();

        assert_eq!(partition_files.len(), 1);

        let mut result_stream = shuffler
            .load_partitioned_shuffles(partition_files)
            .await
            .unwrap();

        let mut num_batches = 0;
        let mut stream = result_stream.pop().unwrap();

        while let Some(item) = stream.next().await {
            check_batch(item.unwrap(), num_batches, 1024);
            num_batches += 1;
        }

        assert_eq!(num_batches, 100);
    }

    #[tokio::test]
    async fn test_shuffler_single_partition_with_empty_batch() {
        let (stream, mut shuffler) = make_stream_and_shuffler(true);

        shuffler.write_unsorted_stream(stream).await.unwrap();
        let partition_files = shuffler.write_partitioned_shuffles(101, 1).await.unwrap();

        assert_eq!(partition_files.len(), 1);

        let mut result_stream = shuffler
            .load_partitioned_shuffles(partition_files)
            .await
            .unwrap();

        let mut num_batches = 0;
        let mut stream = result_stream.pop().unwrap();

        while let Some(item) = stream.next().await {
            check_batch(item.unwrap(), num_batches, 1024);
            num_batches += 1;
        }

        assert_eq!(num_batches, 100);
    }

    #[tokio::test]
    async fn test_shuffler_multiple_partition() {
        let (stream, mut shuffler) = make_stream_and_shuffler(false);

        shuffler.write_unsorted_stream(stream).await.unwrap();
        let partition_files = shuffler.write_partitioned_shuffles(1, 100).await.unwrap();

        assert_eq!(partition_files.len(), 100);

        let mut result_stream = shuffler
            .load_partitioned_shuffles(partition_files)
            .await
            .unwrap();

        let mut num_batches = 0;
        result_stream.reverse();

        while let Some(mut stream) = result_stream.pop() {
            while let Some(item) = stream.next().await {
                check_batch(item.unwrap(), num_batches, 1024);
                num_batches += 1
            }
        }

        assert_eq!(num_batches, 100);
    }

    #[tokio::test]
    async fn test_shuffler_multi_buffer_single_partition() {
        let (stream, mut shuffler) = make_stream_and_shuffler(false);
        shuffler.write_unsorted_stream(stream).await.unwrap();

        // set the same buffer twice we should get double the data
        unsafe { shuffler.set_unsorted_buffers(&[UNSORTED_BUFFER, UNSORTED_BUFFER]) }

        let partition_files = shuffler.write_partitioned_shuffles(200, 1).await.unwrap();

        assert_eq!(partition_files.len(), 1);

        let mut result_stream = shuffler
            .load_partitioned_shuffles(partition_files)
            .await
            .unwrap();

        let mut num_batches = 0;
        result_stream.reverse();

        while let Some(mut stream) = result_stream.pop() {
            while let Some(item) = stream.next().await {
                check_batch(item.unwrap(), num_batches, 2048);
                num_batches += 1
            }
        }

        assert_eq!(num_batches, 100);
    }

    #[tokio::test]
    async fn test_shuffler_multi_buffer_multi_partition() {
        let (stream, mut shuffler) = make_stream_and_shuffler(false);
        shuffler.write_unsorted_stream(stream).await.unwrap();

        // set the same buffer twice we should get double the data
        unsafe { shuffler.set_unsorted_buffers(&[UNSORTED_BUFFER, UNSORTED_BUFFER]) }

        let partition_files = shuffler.write_partitioned_shuffles(1, 32).await.unwrap();
        assert_eq!(partition_files.len(), 200);

        let mut result_stream = shuffler
            .load_partitioned_shuffles(partition_files)
            .await
            .unwrap();

        let mut num_batches = 0;
        result_stream.reverse();

        while let Some(mut stream) = result_stream.pop() {
            while let Some(item) = stream.next().await {
                check_batch(item.unwrap(), num_batches % 100, 1024);
                num_batches += 1
            }
        }

        assert_eq!(num_batches, 200);
    }

    fn make_big_stream_and_shuffler(
        num_batches: u32,
        num_partitions: u32,
        pq_dim: u32,
    ) -> (impl RecordBatchStream, IvfShuffler) {
        let schema = make_schema(pq_dim);

        let schema2 = schema.clone();

        let stream = stream::iter(0..num_batches).map(move |idx| {
            let mut rng = rand::thread_rng();
            let row_ids = Arc::new(UInt64Array::from_iter(
                (idx * 1024..(idx + 1) * 1024).map(u64::from),
            ));

            let part_id = Arc::new(UInt32Array::from_iter(
                (idx * 1024..(idx + 1) * 1024).map(|_| rng.next_u32() % num_partitions),
            ));

            let values = Arc::new(UInt8Array::from_iter((0..pq_dim * 1024).map(|_| idx as u8)));
            let pq_codes = Arc::new(
                FixedSizeListArray::try_new_from_values(values as Arc<dyn Array>, pq_dim as i32)
                    .unwrap(),
            );

            Ok(RecordBatch::try_new(schema2.clone(), vec![row_ids, part_id, pq_codes]).unwrap())
        });

        let stream = RecordBatchStreamAdapter::new(schema.clone(), stream);

        let shuffler = IvfShuffler::try_new(num_partitions, None).unwrap();

        (stream, shuffler)
    }

    // Change NUM_BATCHES = 1000 * 1024 and NUM_PARTITIONS to 35000 to test 1B shuffle
    const NUM_BATCHES: u32 = 100;
    const NUM_PARTITIONS: u32 = 1000;
    const PQ_DIM: u32 = 48;
    const BATCHES_PER_PARTITION: u32 = 10200;
    const NUM_CONCURRENT_JOBS: u32 = 16;

    #[test_log::test(tokio::test(flavor = "multi_thread"))]
    async fn test_big_shuffle() {
        let (stream, mut shuffler) =
            make_big_stream_and_shuffler(NUM_BATCHES, NUM_PARTITIONS, PQ_DIM);

        shuffler.write_unsorted_stream(stream).await.unwrap();
        let partition_files = shuffler
            .write_partitioned_shuffles(
                BATCHES_PER_PARTITION as usize,
                NUM_CONCURRENT_JOBS as usize,
            )
            .await
            .unwrap();

        let expected_num_part_files = NUM_BATCHES.div_ceil(BATCHES_PER_PARTITION);

        assert_eq!(partition_files.len(), expected_num_part_files as usize);

        let mut result_stream = shuffler
            .load_partitioned_shuffles(partition_files)
            .await
            .unwrap();

        let mut num_batches = 0;
        result_stream.reverse();

        while let Some(mut stream) = result_stream.pop() {
            while stream.next().await.is_some() {
                num_batches += 1
            }
        }

        assert_eq!(num_batches, NUM_PARTITIONS * expected_num_part_files);
    }
}
