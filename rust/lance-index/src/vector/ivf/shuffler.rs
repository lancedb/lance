// Copyright 2024 Lance Developers.
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

use arrow_array::types::UInt64Type;
use arrow_array::{
    cast::AsArray, FixedSizeListArray, RecordBatch, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_schema::{DataType, Field as ArrowField, Field, Schema as ArrowSchema};
use futures::stream::repeat_with;
use futures::{stream, Stream, StreamExt, TryStreamExt};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::datatypes::Schema;
use lance_file::reader::FileReader;
use lance_file::writer::FileWriter;
use lance_io::object_store::ObjectStore;
use lance_io::stream::RecordBatchStream;
use lance_io::ReadBatchParams;
use lance_table::format::SelfDescribingFileReader;
use lance_table::io::manifest::ManifestDescribing;

use crate::vector::{PART_ID_COLUMN, PQ_CODE_COLUMN};
use lance_core::{Error, Result, ROW_ID, ROW_ID_FIELD};
use log::info;
use object_store::path::Path;
use snafu::{location, Location};
use tempfile::TempDir;

const UNSORTED_BUFFER: &str = "unsorted.lance";

fn get_temp_dir() -> Result<Path> {
    let dir = TempDir::new()?;
    let tmp_dir_path = Path::from_filesystem_path(dir.path()).map_err(|e| Error::IO {
        message: format!("failed to get buffer path in shuffler: {}", e),
        location: location!(),
    })?;
    Ok(tmp_dir_path)
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
    ivf: Arc<dyn crate::vector::ivf::Ivf>,
    precomputed_partitions: Option<HashMap<u64, u32>>,
    num_partitions: u32,
    num_sub_vectors: usize,
    shuffle_partition_batches: usize,
    shuffle_partition_concurrency: usize,
    precomputed_shuffle_buffers: Option<(Path, Vec<String>)>,
) -> Result<Vec<impl Stream<Item = Result<RecordBatch>>>> {
    let column: Arc<str> = column.into();

    // TODO: dynamically detect schema from the transforms.
    let schema = Arc::new(arrow_schema::Schema::new(vec![
        ROW_ID_FIELD.clone(),
        Field::new(PART_ID_COLUMN, DataType::UInt32, false),
        Field::new(
            PQ_CODE_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, true)),
                num_sub_vectors as i32,
            ),
            false,
        ),
    ]));

    // step 1: either use precomputed shuffle files or write shuffle data to a file
    let shuffler = if let Some((path, buffers)) = precomputed_shuffle_buffers {
        let mut shuffler = IvfShuffler::try_new(
            num_partitions,
            num_sub_vectors,
            Some(path),
            Schema::try_from(schema.as_ref())?,
        )?;
        unsafe {
            shuffler.set_unsorted_buffers(&buffers);
        }

        shuffler
    } else {
        let mut shuffler = IvfShuffler::try_new(
            num_partitions,
            num_sub_vectors,
            None,
            Schema::try_from(schema.as_ref())?,
        )?;

        let precomputed_partitions = precomputed_partitions.map(Arc::new);
        let stream = data
            .zip(repeat_with(move || ivf.clone()))
            .map(move |(b, ivf)| {
                let col_ref = column.clone();

                // If precomputed_partitions map is provided, use it
                // for fast partitions.
                let partition_map = precomputed_partitions
                    .as_ref()
                    .cloned()
                    .unwrap_or(Arc::new(HashMap::new()));

                tokio::task::spawn(async move {
                    let batch = b?;

                    let part_ids = if !partition_map.is_empty() {
                        let row_ids = batch.column_by_name(ROW_ID).ok_or(Error::Index {
                            message: "column does not exist".to_string(),
                            location: location!(),
                        })?;
                        let part_ids = row_ids
                            .as_primitive::<UInt64Type>()
                            .values()
                            .iter()
                            .filter_map(|row_id| partition_map.get(row_id).copied())
                            .collect::<Vec<_>>();
                        Some(UInt32Array::from(part_ids))
                    } else {
                        None
                    };

                    ivf.partition_transform(&batch, col_ref.as_ref(), part_ids)
                        .await
                })
            })
            .buffer_unordered(num_cpus::get())
            .map(|res| match res {
                Ok(Ok(batch)) => Ok(batch),
                Ok(Err(err)) => Err(Error::IO {
                    message: err.to_string(),
                    location: location!(),
                }),
                Err(err) => Err(Error::IO {
                    message: err.to_string(),
                    location: location!(),
                }),
            })
            .boxed();

        let stream = lance_io::stream::RecordBatchStreamAdapter::new(schema.clone(), stream);

        let start = std::time::Instant::now();
        shuffler.write_unsorted_stream(stream).await?;
        info!("wrote unstored stream in {:?} seconds", start.elapsed());

        shuffler
    };

    // step 2: stream in the shuffle data in chunks and write sorted chuncks out
    let start = std::time::Instant::now();
    let partition_files = shuffler
        .write_partitioned_shuffles(shuffle_partition_batches, shuffle_partition_concurrency)
        .await?;
    info!("counted partition sizes in {:?} seconds", start.elapsed());

    // step 3: load the sorted chuncks, consumers are expect to be responsible for merging the streams
    let start = std::time::Instant::now();
    let stream = shuffler.load_partitioned_shuffles(partition_files).await?;
    info!(
        "merged partitioned shuffles in {:?} seconds",
        start.elapsed()
    );

    Ok(stream)
}

pub struct IvfShuffler {
    unsorted_buffers: Vec<String>,

    num_partitions: u32,

    pq_width: usize,

    output_dir: Path,

    schema: Schema,
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
    pub fn try_new(
        num_partitions: u32,
        pq_width: usize,
        output_dir: Option<Path>,
        schema: Schema,
    ) -> Result<Self> {
        let output_dir = match output_dir {
            Some(output_dir) => output_dir,
            None => get_temp_dir()?,
        };

        Ok(Self {
            num_partitions,
            pq_width,
            output_dir,
            schema,
            unsorted_buffers: vec![],
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
        data: impl RecordBatchStream + Unpin + 'static,
    ) -> Result<()> {
        let object_store = ObjectStore::local();
        let path = self.output_dir.child(UNSORTED_BUFFER);
        let writer = object_store.create(&path).await?;

        let mut file_writer = FileWriter::<ManifestDescribing>::with_object_writer(
            writer,
            self.schema.clone(),
            &Default::default(),
        )?;

        let mut data = Box::pin(data);

        while let Some(batch) = data.next().await {
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
                .map(|i| reader.read_batch(i as i32, .., &lance_schema, None))
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
        inputs: &[ShuffleInput],
        partition_size: Vec<u64>,
    ) -> Result<(Vec<Vec<u64>>, Vec<Vec<u8>>)> {
        let mut row_id_buffers = partition_size
            .iter()
            .map(|s| Vec::with_capacity(*s as usize))
            .collect::<Vec<_>>();
        let mut pq_code_buffers = partition_size
            .iter()
            .map(|s| Vec::with_capacity((*s as usize) * self.pq_width))
            .collect::<Vec<_>>();

        info!("Shuffling into memory");

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
            let total_batch = reader.num_batches();

            let mut stream = stream::iter(start..end)
                .map(|i| {
                    reader.read_batch(i as i32, ReadBatchParams::RangeFull, reader.schema(), None)
                })
                .buffered(16)
                .enumerate();

            while let Some((idx, batch)) = stream.next().await {
                if idx % 100 == 0 {
                    info!("Shuffle Progress {}/{}", idx, total_batch);
                }

                let batch = batch?;

                // skip empty batches
                if batch.num_rows() == 0 {
                    continue;
                }

                let row_ids: &UInt64Array = batch
                    .column_by_name(ROW_ID)
                    .expect("Row ID column not found")
                    .as_primitive();

                let part_ids: &UInt32Array = batch
                    .column_by_name(PART_ID_COLUMN)
                    .expect("Partition ID column not found")
                    .as_primitive();

                let pq_codes: &UInt8Array = batch
                    .column_by_name(PQ_CODE_COLUMN)
                    .expect("PQ Code column not found")
                    .as_fixed_size_list()
                    .values()
                    .as_primitive();

                let num_sub_vectors = pq_codes.len() / row_ids.len();

                row_ids
                    .values()
                    .iter()
                    .zip(part_ids.values().iter())
                    .enumerate()
                    .for_each(|(i, (row_id, part_id))| {
                        row_id_buffers[*part_id as usize].push(*row_id);
                        pq_code_buffers[*part_id as usize].extend(
                            &pq_codes.values()[i * num_sub_vectors..(i + 1) * num_sub_vectors],
                        );
                    });
            }
        }

        Ok((row_id_buffers, pq_code_buffers))
    }

    pub async fn write_partitioned_shuffles(
        &self,
        batches_per_partition: usize,
        concurrent_jobs: usize,
    ) -> Result<Vec<String>> {
        let num_batches = self.total_batches().await?;
        let total_batches = num_batches.iter().sum();

        stream::iter((0..total_batches).step_by(batches_per_partition))
            .zip(stream::repeat(num_batches))
            .map(|(i, num_batches)| async move {
                // first, calculate which files and ranges needs to be processed
                let start = i;
                let end = std::cmp::min(i + batches_per_partition, total_batches);
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
                let size_counts = self.count_partition_size(&input).await?;

                // third, shuffle the data into each partition
                let (row_id_buffers, pq_code_buffers) =
                    self.shuffle_to_partitions(&input, size_counts).await?;

                // finally, write the shuffled data to disk
                let object_store = ObjectStore::local();
                let output_file = format!("sorted_{}.lance", i);
                let path = self.output_dir.child(output_file.clone());
                let writer = object_store.create(&path).await?;

                // TODO: dynamically detect schema from the transforms.
                let schema = Arc::new(ArrowSchema::new(vec![
                    ROW_ID_FIELD.clone(),
                    ArrowField::new(PART_ID_COLUMN, DataType::UInt32, false),
                    ArrowField::new(
                        PQ_CODE_COLUMN,
                        DataType::FixedSizeList(
                            Arc::new(ArrowField::new("item", DataType::UInt8, true)),
                            self.pq_width as i32,
                        ),
                        false,
                    ),
                ]));

                let mut file_writer = FileWriter::<ManifestDescribing>::with_object_writer(
                    writer,
                    self.schema.clone(),
                    &Default::default(),
                )?;

                let shuffled = row_id_buffers
                    .into_iter()
                    .zip(pq_code_buffers.into_iter())
                    .enumerate()
                    .filter(|(_, (row_ids, _))| !row_ids.is_empty())
                    .map(|(part_id, (row_ids, pq_codes))| {
                        let length = row_ids.len();
                        let batch = RecordBatch::try_new(
                            schema.clone(),
                            vec![
                                Arc::new(UInt64Array::from(row_ids)),
                                Arc::new(UInt32Array::from_iter_values(
                                    std::iter::repeat(part_id as u32).take(length),
                                )),
                                Arc::new(FixedSizeListArray::try_new_from_values(
                                    UInt8Array::from(pq_codes),
                                    self.pq_width as i32,
                                )?),
                            ],
                        )?;

                        Ok(batch) as Result<_>
                    });

                for batch in shuffled {
                    file_writer.write(&[batch?]).await?;
                }

                file_writer.finish().await?;

                Ok(output_file) as Result<String>
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
            let object_store = ObjectStore::local();
            let path = self.output_dir.child(file);
            let reader = FileReader::try_new_self_described(&object_store, &path, None).await?;
            let reader = Arc::new(reader);

            let stream = stream::iter(0..reader.num_batches())
                .zip(stream::repeat(reader))
                .map(|(i, reader)| async move {
                    reader
                        .read_batch(i as i32, ReadBatchParams::RangeFull, reader.schema(), None)
                        .await
                })
                .buffered(4);
            streams.push(stream);
        }

        Ok(streams)
    }
}

#[cfg(test)]
mod test {
    use arrow_array::{
        types::{UInt32Type, UInt64Type, UInt8Type},
        Array,
    };
    use lance_io::stream::RecordBatchStreamAdapter;

    use super::*;

    fn make_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            ROW_ID_FIELD.clone(),
            ArrowField::new(PART_ID_COLUMN, DataType::UInt32, false),
            ArrowField::new(
                PQ_CODE_COLUMN,
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::UInt8, true)),
                    32,
                ),
                false,
            ),
        ]))
    }

    fn make_stream_and_shuffler(
        include_empty_batches: bool,
    ) -> (impl RecordBatchStream, IvfShuffler) {
        let schema = make_schema();

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

        let shuffler =
            IvfShuffler::try_new(100, 32, None, Schema::try_from(schema.as_ref()).unwrap())
                .unwrap();

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
}
