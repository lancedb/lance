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

//! Disk-based shuffle a stream of [RecordBatch] into each IVF partition.
//!
//! 1. write the entire stream to a file
//! 2. count the number of rows in each partition
//! 3. read the data back into memory and shuffle into grouped vectors
//!
//! Problems for the future:
//! 1. while groupby column will stay the same, we may want to include extra data columns in the future
//! 2. shuffling into memory is fast but we should add disk buffer to support bigger datasets

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{FixedSizeListArray, RecordBatch, UInt32Array, UInt64Array, UInt8Array};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use futures::{stream, Stream, StreamExt, TryStreamExt};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::datatypes::Schema;
use lance_core::io::{FileReader, FileWriter, ReadBatchParams, RecordBatchStream};

use crate::vector::{PART_ID_COLUMN, PQ_CODE_COLUMN};
use lance_core::io::object_store::ObjectStore;
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

pub struct IvfShuffler {
    num_partitions: u32,

    pq_width: usize,

    output_dir: Path,

    schema: Schema,
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
        })
    }

    pub async fn write_unsorted_stream(
        &self,
        data: impl RecordBatchStream + Unpin + 'static,
    ) -> Result<()> {
        let object_store = ObjectStore::local();
        let path = self.output_dir.child(UNSORTED_BUFFER);
        let writer = object_store.create(&path).await?;

        let mut file_writer =
            FileWriter::with_object_writer(writer, self.schema.clone(), &Default::default())?;

        let mut data = Box::pin(data);

        while let Some(batch) = data.next().await {
            file_writer.write(&[batch?]).await?;
        }

        file_writer.finish().await?;

        Ok(())
    }

    pub async fn total_batches(&self) -> Result<usize> {
        let object_store = ObjectStore::local();
        let path = self.output_dir.child(UNSORTED_BUFFER);
        let reader = FileReader::try_new(&object_store, &path).await?;
        Ok(reader.num_batches())
    }

    pub async fn count_partition_size(&self, start: usize, end: usize) -> Result<Vec<u64>> {
        let object_store = ObjectStore::local();
        let path = self.output_dir.child(UNSORTED_BUFFER);
        let reader = FileReader::try_new(&object_store, &path).await?;

        let mut partition_sizes = vec![0; self.num_partitions as usize];

        let lance_schema = reader
            .schema()
            .project(&[PART_ID_COLUMN])
            .expect("part id should exist");

        let mut stream = stream::iter(start..end)
            .map(|i| reader.read_batch(i as i32, ReadBatchParams::RangeFull, &lance_schema))
            .buffered(64);

        while let Some(batch) = stream.next().await {
            let batch = batch?;
            let part_ids: &UInt32Array = batch.column(0).as_primitive();
            for part_id in part_ids.values() {
                partition_sizes[*part_id as usize] += 1;
            }
        }

        Ok(partition_sizes)
    }

    pub async fn shuffle_to_partitions(
        &self,
        partition_size: Vec<u64>,
        start: usize,
        end: usize,
    ) -> Result<(Vec<Vec<u64>>, Vec<Vec<u8>>)> {
        let mut row_id_buffers = partition_size
            .iter()
            .map(|s| Vec::with_capacity(*s as usize))
            .collect::<Vec<_>>();
        let mut pq_code_buffers = partition_size
            .iter()
            .map(|s| Vec::with_capacity((*s as usize) * self.pq_width))
            .collect::<Vec<_>>();

        let object_store = ObjectStore::local();
        let path = self.output_dir.child(UNSORTED_BUFFER);
        let reader = FileReader::try_new(&object_store, &path).await?;
        let total_batch = reader.num_batches();

        info!("Shuffling into memory");

        let mut stream = stream::iter(start..end)
            .map(|i| reader.read_batch(i as i32, ReadBatchParams::RangeFull, reader.schema()))
            .buffered(16)
            .enumerate();

        while let Some((idx, batch)) = stream.next().await {
            if idx % 100 == 0 {
                info!("Shuffle Progress {}/{}", idx, total_batch);
            }

            let batch = batch?;

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
                        pq_codes.values()[i * num_sub_vectors..(i + 1) * num_sub_vectors].iter(),
                    );
                });
        }

        Ok((row_id_buffers, pq_code_buffers))
    }

    pub async fn write_partitoned_shuffles(
        &self,
        batches_per_partition: usize,
        concurrent_jobs: usize,
    ) -> Result<Vec<String>> {
        let total_batches = self.total_batches().await?;

        stream::iter((0..total_batches).step_by(batches_per_partition))
            .map(|i| async move {
                let start = i;
                let end = std::cmp::min(i + batches_per_partition, total_batches);

                let size_counts = self.count_partition_size(start, end).await?;

                let (row_id_buffers, pq_code_buffers) =
                    self.shuffle_to_partitions(size_counts, start, end).await?;

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

                let mut file_writer = FileWriter::with_object_writer(
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
                                Arc::new(UInt64Array::from_iter_values(row_ids.into_iter())),
                                Arc::new(UInt32Array::from_iter_values(
                                    std::iter::repeat(part_id as u32).take(length),
                                )),
                                Arc::new(FixedSizeListArray::try_new_from_values(
                                    UInt8Array::from_iter_values(pq_codes.into_iter()),
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
            let reader = FileReader::try_new(&object_store, &path).await?;
            let reader = Arc::new(reader);

            let stream = stream::iter(0..reader.num_batches())
                .zip(stream::repeat(reader))
                .map(|(i, reader)| async move {
                    reader
                        .read_batch(i as i32, ReadBatchParams::RangeFull, reader.schema())
                        .await
                })
                .buffered(16);
            streams.push(stream);
        }

        Ok(streams)
    }
}
