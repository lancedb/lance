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

use std::ops::Range;

use arrow_array::{RecordBatch, UInt32Array};
use uuid::Uuid;

use super::fragment::FragmentReader;
use super::Dataset;
use crate::dataset::FileFragment;
use crate::datatypes::Schema;
use crate::format::Fragment;
use crate::io::deletion::DeletionVector;
use crate::{io::FileWriter, Error, Result};

/// Update or insert a new column.
///
/// To use, call [`Updater::next`] to get the next [`RecordBatch`] as input,
/// then call [`Updater::update`] to update the batch. Repeat until
/// [`Updater::next`] returns `None`.
pub struct Updater {
    fragment: FileFragment,

    /// The reader over the [`Fragment`]
    reader: FragmentReader,

    last_input: Option<RecordBatch>,

    writer: Option<FileWriter>,

    output_schema: Option<Schema>,

    batch_id: usize,

    start_row_id: u32,

    deletion_vector: DeletionVector,
}

impl Updater {
    /// Create a new updater with source reader, and destination writer.
    pub(super) fn new(
        fragment: FileFragment,
        reader: FragmentReader,
        deletion_vector: DeletionVector,
    ) -> Self {
        Self {
            fragment,
            reader,
            last_input: None,
            writer: None,
            output_schema: None,
            batch_id: 0,
            start_row_id: 0,
            deletion_vector,
        }
    }

    pub fn fragment(&self) -> &FileFragment {
        &self.fragment
    }

    pub fn dataset(&self) -> &Dataset {
        self.fragment.dataset()
    }

    /// Returns the next [`RecordBatch`] as input for updater.
    pub async fn next(&mut self) -> Result<Option<&RecordBatch>> {
        if self.batch_id >= self.reader.num_batches() {
            return Ok(None);
        }
        let batch = self.reader.read_batch(self.batch_id, ..).await?;
        self.batch_id += 1;

        self.last_input = Some(batch);
        Ok(self.last_input.as_ref())
    }

    /// Create a new Writer for new columns.
    ///
    /// After it is called, this Fragment contains the metadata of the new DataFile,
    /// containing the columns, even the data has not written yet.
    ///
    /// It is the caller's responsibility to close the [`FileWriter`].
    ///
    /// Internal use only.
    async fn new_writer(&mut self, schema: Schema) -> Result<FileWriter> {
        // Sanity check.
        //
        // To keep it simple, new schema must have no intersection with the existing schema.
        let existing_schema = self.fragment.dataset().schema();
        for field in schema.fields.iter() {
            // Just check the first level names.
            if existing_schema.field(&field.name).is_some() {
                return Err(Error::IO {
                    message: format!(
                        "Append column: duplicated column {} already exists",
                        field.name
                    ),
                });
            }
        }

        let file_name = format!("{}.lance", Uuid::new_v4());
        self.fragment.metadata.add_file(&file_name, &schema);

        let full_path = self.fragment.dataset().data_dir().child(file_name.as_str());

        FileWriter::try_new(
            self.fragment.dataset().object_store.as_ref(),
            &full_path,
            schema,
        )
        .await
    }

    /// Update one batch.
    pub async fn update(&mut self, batch: RecordBatch) -> Result<()> {
        let Some(last) = self.last_input.as_ref() else {
            return Err(Error::IO {
                message: "Fragment Updater: no input data is available before update".to_string(),
            });
        };

        if last.num_rows() != batch.num_rows() {
            return Err(Error::IO {
                message: format!(
                "Fragment Updater: new batch has different size with the source batch: {} != {}",
                last.num_rows(),
                batch.num_rows()
            ),
            });
        };

        if self.writer.is_none() {
            let output_schema = batch.schema();
            // Need to assign field id correctly here.
            let merged = self.fragment.schema().merge(output_schema.as_ref())?;
            // Get the schema with correct field id.
            let schema = merged.project_by_schema(output_schema.as_ref())?;
            self.output_schema = Some(merged);

            self.writer = Some(self.new_writer(schema).await?);
        }

        let writer = self.writer.as_mut().unwrap();
        // Because of deleted rows, the number of row ids in the batch might not
        // match the length.
        let row_id_stride = self.reader.num_rows_in_batch(self.batch_id - 1) as u32; // Subtract since we incremented in next()
        let batch = add_blanks(
            batch,
            self.start_row_id..(self.start_row_id + row_id_stride),
            &self.deletion_vector,
        )?;
        // validation just in case
        if batch.num_rows() != row_id_stride as usize {
            return Err(Error::Internal {
                message: format!(
                    "Fragment Updater: batch size mismatch: {} != {}",
                    batch.num_rows(),
                    row_id_stride
                ),
            });
        }

        writer.write(&[batch]).await?;

        self.start_row_id += row_id_stride;

        Ok(())
    }

    /// Finish updating this fragment, and returns the updated [`Fragment`].
    pub async fn finish(&mut self) -> Result<Fragment> {
        if let Some(writer) = self.writer.as_mut() {
            writer.finish().await?;
        }

        Ok(self.fragment.metadata().clone())
    }

    pub fn schema(&self) -> Option<&Schema> {
        self.output_schema.as_ref()
    }
}

/// Add blank rows where there are deleted rows
pub(crate) fn add_blanks(
    batch: RecordBatch,
    row_id_range: Range<u32>,
    deletion_vector: &DeletionVector,
) -> Result<RecordBatch> {
    // Fast early return
    if !row_id_range
        .clone()
        .any(|row_id| deletion_vector.contains(row_id))
    {
        return Ok(batch);
    }

    if batch.num_rows() == 0 {
        // TODO: implement adding blanks for an empty batch.
        // This is difficult because we need to create a batch for arbitrary schemas.
        return Err(Error::NotSupported {
            source: "Missing many rows in merge".into(),
        });
    }

    let mut array_i = 0;
    let selection_vector: Vec<u32> = row_id_range
        .map(move |row_id| {
            if deletion_vector.contains(row_id) {
                // For simplicity, we just use the first value for deleted rows
                // TODO: optimize this to use small value for each column.
                0
            } else {
                array_i += 1;
                array_i - 1
            }
        })
        .collect();
    let selection_vector = UInt32Array::from(selection_vector);

    let arrays = batch
        .columns()
        .iter()
        .map(|array| {
            arrow::compute::take(array.as_ref(), &selection_vector, None).map_err(|e| {
                Error::Arrow {
                    message: format!("Failed to add blanks: {}", e),
                }
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let batch = RecordBatch::try_new(batch.schema(), arrays)?;

    Ok(batch)
}
