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

use std::collections::BTreeMap;
use std::pin::Pin;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use futures::Stream;
use object_store::path::Path;
use tempfile::TempDir;

use crate::datatypes::Schema;
use crate::io::FileReader;
use crate::{
    error::Result,
    io::{FileWriter, ObjectStore, RecordBatchStream},
};

/// Shuffle [RecordBatch] based on their IVF partition.
///
///
pub struct Shuffler {
    buffer: BTreeMap<u32, Vec<RecordBatch>>,

    /// The size, as number of rows, of each partition in memory before flushing to disk.
    flush_size: usize,

    /// Partition ID to file-group ID mapping, in memory.
    /// No external dependency is required, because we don't need to guarantee the
    /// persistence of this mapping, as well as the temp files.
    parted_groups: BTreeMap<u32, Vec<u32>>,

    /// We need to keep the temp_dir with Shuffler because ObjectStore crate does not
    /// work with a NamedTempFile.
    temp_dir: TempDir,

    writer: FileWriter,
}

impl Shuffler {
    pub async fn try_new(schema: Schema) -> Result<Self> {
        // TODO: create a `ObjectWriter::tempfile()` method.
        let temp_dir = tempfile::tempdir()?;
        let (object_store, _) = ObjectStore::from_uri(temp_dir.path().to_str().unwrap()).await?;
        let path = Self::lance_buffer_path(&temp_dir)?;
        let writer = object_store.create(&path).await?;

        Ok(Self {
            buffer: BTreeMap::new(),
            flush_size: 1024, // TODO: change to parameterized value later.
            temp_dir,
            parted_groups: BTreeMap::new(),
            writer: FileWriter::with_object_writer(writer, schema)?,
        })
    }

    fn lance_buffer_path(dir: &TempDir) -> Result<Path> {
        let file_path = dir.path().join("temp_part.lance");
        let path = Path::from_filesystem_path(file_path)?;
        Ok(path)
    }

    /// Insert a [RecordBatch] with the same key (Partition ID).
    pub async fn insert(&mut self, key: u32, batch: RecordBatch) -> Result<()> {
        let batches = self.buffer.entry(key).or_default();
        batches.push(batch);
        let total = batches.iter().map(|b| b.num_rows()).sum::<usize>();
        /// If there are more than `flush_size` rows in the buffer, flush them to disk
        /// as one group.
        if total > self.flush_size {
            self.parted_groups
                .entry(key)
                .or_default()
                .push(self.writer.batch_id as u32);
            self.writer.write(&batches).await?;
            batches.clear();
        };
        Ok(())
    }

    /// Iterate over the shuffled [RecordBatch]s for a given partition key.
    pub async fn key_iter(&self, key: u32) -> Result<impl RecordBatchStream + '_> {
        let file_path = Self::lance_buffer_path(&self.temp_dir)?;
        let (object_store, path) = ObjectStore::from_uri(&file_path.to_string()).await?;
        let reader = FileReader::try_new(&object_store, &path).await?;
        Ok(PartitionStream::new(reader, &self.parted_groups, key))
    }
}

pub(crate) struct PartitionStream<'a> {
    reader: FileReader,
    partition_id: u32,
    parted_groups: &'a BTreeMap<u32, Vec<u32>>,
}

impl<'a> PartitionStream<'a> {
    fn new(reader: FileReader, parted_groups: &'a BTreeMap<u32, Vec<u32>>, part_id: u32) -> Self {
        Self {
            reader,
            partition_id: part_id,
            parted_groups,
        }
    }
}

impl Stream for PartitionStream<'_> {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        todo!()
    }
}

impl<'a> RecordBatchStream<'a> for PartitionStream<'a> {
    fn schema(&self) -> SchemaRef {
        todo!()
    }
}

#[cfg(test)]
mod tests {}
