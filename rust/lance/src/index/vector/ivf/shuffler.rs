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
use std::sync::Mutex;

use arrow_array::RecordBatch;
use tempfile::TempDir;

use crate::datatypes::Schema;
use crate::{
    error::Result,
    io::{FileWriter, ObjectStore, RecordBatchStream},
};

/// Shuffle [RecordBatch] based on their IVF partition.
///
///
pub struct Shuffler {
    buffer: Mutex<BTreeMap<u32, Mutex<Vec<RecordBatch>>>>,
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
        let temp_dir = tempfile::tempdir()?;
        let (object_store, _) = ObjectStore::from_uri(temp_dir.path().to_str().unwrap()).await?;
        let file_path = temp_dir.path().join("temp_part.lance");
        let writer = object_store.create(&file_path).await?;
        dbg!(println!("temp file: {:?}", file_path));
        Ok(Self {
            buffer: Mutex::new(BTreeMap::new()),
            flush_size: 1024, // TODO: change to parameterized value later.
            temp_dir,
            parted_groups: BTreeMap::new(),
            writer: FileWriter::with_object_writer(writer, schema)?,
        })
    }

    /// Insert a [RecordBatch] with the same key (Partition ID).
    pub fn insert(&self, key: u32, batch: RecordBatch) -> Result<Self> {
        {
            let mut buffer = self.buffer.lock().unwrap();
            let entry = buffer.entry(key).or_default();
            entry.push(batch);
            let batches = entry.get_mut().unwrap();
            let total = batches.iter().map(|b| b.num_rows()).sum::<usize>();
            if total > self.flush_size {
                let mut batches = Vec::new();
                std::mem::swap(&mut batches, batches);
                self.writer.write_batches(batches)?;
            }
        }
        todo!()
    }

    /// Iterate over the shuffled [RecordBatch] for a given partition.
    pub fn key_iter(&self, key: u32) -> Result<impl RecordBatchStream> {
        todo!()
    }
}

#[cfg(test)]
mod tests {}
