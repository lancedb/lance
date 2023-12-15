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

use arrow_array::cast::AsArray;
use arrow_array::{UInt32Array, UInt64Array, UInt8Array};
use futures::{stream, StreamExt};
use lance_core::datatypes::Schema;
use lance_core::io::{FileReader, FileWriter, ReadBatchParams, RecordBatchStream};

use crate::vector::{PART_ID_COLUMN, PQ_CODE_COLUMN};
use lance_core::io::object_store::ObjectStore;
use lance_core::{Error, Result, ROW_ID};
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

    pub async fn count_partition_size(&self) -> Result<Vec<u64>> {
        let object_store = ObjectStore::local();
        let path = self.output_dir.child(UNSORTED_BUFFER);
        let reader = FileReader::try_new(&object_store, &path).await?;

        let mut partition_sizes = vec![0; self.num_partitions as usize];

        let lance_schema = reader
            .schema()
            .project(&[PART_ID_COLUMN])
            .expect("part id should exist");

        let mut stream = stream::iter(0..reader.num_batches())
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

        let mut stream = stream::iter(0..reader.num_batches())
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
}
