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

use arrow_array::RecordBatch;

use super::fragment::FragmentReader;
use crate::dataset::FileFragment;
use crate::{io::FileWriter, Error, Result};

/// Update or insert a new column.
pub struct Updater {
    fragment: FileFragment,

    /// The reader over the [`Fragment`]
    reader: FragmentReader,

    last_input: Option<RecordBatch>,

    writer: Option<FileWriter>,

    batch_id: usize,
}

impl Updater {
    /// Create a new updater with source reader, and destination writer.
    pub(super) fn new(fragment: FileFragment, reader: FragmentReader) -> Self {
        Self {
            fragment,
            reader,
            last_input: None,
            writer: None,
            batch_id: 0,
        }
    }

    /// Returns the next [`RecordBatch`] as input for updater.
    pub async fn next(&mut self) -> Result<Option<&RecordBatch>> {
        let batch = self.reader.read_batch(self.batch_id, ..).await?;
        self.batch_id += 1;

        self.last_input = Some(batch);
        Ok(self.last_input.as_ref())
    }

    /// Update one batch.
    pub async fn update(&mut self, batch: &RecordBatch) -> Result<()> {
        let Some(last) = self.last_input.as_ref() else {
            return Err(Error::IO("Fragment Updater: no input data is available before update".to_string()));
        };

        if last.num_rows() != batch.num_rows() {
            return Err(Error::IO(format!(
                "Fragment Updater: new batch has different size with the source batch: {} != {}",
                last.num_rows(),
                batch.num_rows()
            )));
        };

        if self.writer.is_none() {
            let output_schema = batch.schema();
            // Need to assign field id correctly here.
            let merged = self.fragment.schema().merge(output_schema.as_ref())?;
            // Get the schema with correct field id.
            let schema = merged.project_by_schema(output_schema.as_ref())?;

            self.writer = Some(self.fragment.new_writer(schema).await?);
        }

        let writer = self.writer.as_mut().unwrap();
        writer.write(&[batch]).await?;

        Ok(())
    }
}
