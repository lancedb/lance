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

use std::collections::{BTreeMap, HashSet};

use arrow_array::RecordBatch;
use arrow_schema::Schema as ArrowSchema;
use object_store::path::Path;
use snafu::{location, Location};
use tempfile::TempDir;

use crate::datatypes::Schema;
use crate::io::reader::batches_stream;
use crate::io::FileReader;
use crate::{
    error::{Error, Result},
    io::{FileWriter, ObjectStore, RecordBatchStream},
};

const BUFFER_FILE_NAME: &str = "buffer.lance";

/// Shuffle [RecordBatch] based on their IVF partition.
///
/// Internally, we shuffle several partitions of [RecordBatch]s into a single LanceFile,
/// and keep tracks of the partition ID to file-group ID mapping in memory.
///
/// The IVF partition / group id mapping then be passed to [Shuffler] to retrieve the
/// all the [RecordBatch]s for a given IVF partition.
///
/// It enables distributed indexing as well:
///  1. Main thread: train IVF model
///  2. Distributed IVF model cross workers.
///  3. Each worker takes parts of the IVF partitions, i.e., `IVF / num_workers` of partitions.
///  4. Each worker shuffle the [RecordBatch]s of the assigned partitions into a single LanceFile,
///     and later aggregated to create the final index file.
#[allow(dead_code)]
pub struct ShufflerBuilder {
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

fn lance_buffer_path(dir: &TempDir) -> Path {
    let file_path = dir.path().join(BUFFER_FILE_NAME);
    Path::from(file_path.to_str().unwrap())
}

impl ShufflerBuilder {
    #[allow(dead_code)]
    pub async fn try_new(schema: &ArrowSchema, flush_threshold: usize) -> Result<Self> {
        // TODO: create a `ObjectWriter::tempfile()` method.
        let temp_dir = tempfile::tempdir()?;

        let object_store = ObjectStore::local();
        let path = lance_buffer_path(&temp_dir);
        let writer = object_store.create(&path).await?;
        let lance_schema = Schema::try_from(schema)?;
        Ok(Self {
            buffer: BTreeMap::new(),
            flush_size: flush_threshold, // TODO: change to parameterized value later.
            temp_dir,
            parted_groups: BTreeMap::new(),
            writer: FileWriter::with_object_writer(writer, lance_schema)?,
        })
    }

    /// Insert a [RecordBatch] with the same key (Partition ID).
    #[allow(dead_code)]
    pub async fn insert(&mut self, key: u32, batch: RecordBatch) -> Result<()> {
        let batches = self.buffer.entry(key).or_default();
        batches.push(batch);
        let total = batches.iter().map(|b| b.num_rows()).sum::<usize>();
        // If there are more than `flush_size` rows in the buffer, flush them to disk
        // as one group.
        if total >= self.flush_size {
            self.parted_groups
                .entry(key)
                .or_default()
                .push(self.writer.next_batch_id() as u32);
            self.writer.write(batches).await?;
            batches.clear();
        };
        Ok(())
    }

    #[allow(dead_code)]
    pub async fn finish(&mut self) -> Result<Shuffler> {
        for (key, batches) in self.buffer.iter() {
            if !batches.is_empty() {
                self.parted_groups
                    .entry(*key)
                    .or_default()
                    .push(self.writer.next_batch_id() as u32);
                self.writer.write(batches.as_slice()).await?;
            }
        }
        self.writer.finish().await?;
        Ok(Shuffler::new(&self.parted_groups, &self.temp_dir))
    }
}

pub struct Shuffler<'a> {
    /// Partition ID to file-group ID mapping, in memory.
    /// No external dependency is required, because we don't need to guarantee the
    /// persistence of this mapping, as well as the temp files.
    parted_groups: BTreeMap<u32, Vec<u32>>,

    /// We need to keep the temp_dir with Shuffler because ObjectStore crate does not
    /// work with a NamedTempFile.
    temp_dir: &'a TempDir,
}

impl<'a> Shuffler<'a> {
    fn new(parted_groups: &BTreeMap<u32, Vec<u32>>, temp_dir: &'a TempDir) -> Self {
        Self {
            parted_groups: parted_groups.clone(),
            temp_dir,
        }
    }

    /// Iterate over the shuffled [RecordBatch]s for a given partition key.
    pub async fn key_iter(&self, key: u32) -> Result<Option<impl RecordBatchStream + '_>> {
        if !self.parted_groups.contains_key(&key) {
            return Ok(None);
        }

        let object_store = ObjectStore::local();
        let path = lance_buffer_path(self.temp_dir);
        let reader = FileReader::try_new(&object_store, &path)
            .await
            .map_err(|e| Error::IO {
                message: format!("failed to open shuffler buffer file: {}, {}", path, e),
                location: location!(),
            })?;
        let schema = reader.schema().clone();

        let group_ids = self
            .parted_groups
            .get(&key)
            .unwrap() // Checked existence already.
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let stream = batches_stream(reader, schema, move |id| group_ids.contains(&(*id as u32)));
        Ok(Some(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::UInt32Array;
    use arrow_schema::{DataType, Field, Schema};
    use futures::TryStreamExt;

    #[tokio::test]
    async fn test_shuffler() {
        let schema = Schema::new(vec![Field::new("a", DataType::UInt32, false)]);
        let mut shuffler = ShufflerBuilder::try_new(&schema, 4).await.unwrap();
        for i in 0..20 {
            shuffler
                .insert(
                    i % 3,
                    RecordBatch::try_new(
                        Arc::new(schema.clone()),
                        vec![Arc::new(UInt32Array::from(vec![i]))],
                    )
                    .unwrap(),
                )
                .await
                .unwrap();
        }
        let reader = shuffler.finish().await.unwrap();
        assert_eq!(reader.keys(), vec![0, 1, 2]);
        for i in 0..3 {
            let stream = reader.key_iter(i).await.unwrap().expect("key exists");
            let batches = stream.try_collect::<Vec<_>>().await.unwrap();
            assert_eq!(batches.len(), 2, "key {} has {} batches", i, batches.len());
        }

        assert!(reader.key_iter(5).await.unwrap().is_none())
    }
}
