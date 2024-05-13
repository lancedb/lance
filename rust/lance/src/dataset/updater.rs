// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;

use arrow_array::{RecordBatch, UInt32Array};
use lance_core::utils::deletion::DeletionVector;
use lance_core::{datatypes::Schema, Error, Result};
use lance_file::writer::FileWriter;
use lance_table::format::Fragment;
use lance_table::io::manifest::ManifestDescribing;
use snafu::{location, Location};
use uuid::Uuid;

use super::fragment::FragmentReader;
use super::Dataset;
use crate::dataset::FileFragment;
/// Update or insert a new column.
///
/// To use, call [`Updater::next`] to get the next [`RecordBatch`] as input,
/// then call [`Updater::update`] to update the batch. Repeat until
/// [`Updater::next`] returns `None`.
///
/// `write_schema` dictates the schema of the new file, while `final_schema` is
/// the schema of the full fragment after the update. These are optional and if
/// not specified, the updater will infer the write schema from the first batch
/// of results and will append them to the current schema to get the final schema.
pub struct Updater {
    fragment: FileFragment,

    /// The reader over the [`Fragment`]
    reader: FragmentReader,

    last_input: Option<RecordBatch>,

    writer: Option<FileWriter<ManifestDescribing>>,

    /// The final schema of the fragment after the update.
    final_schema: Option<Schema>,

    /// The schema the new files will be written in. This only contains new columns.
    write_schema: Option<Schema>,

    batch_id: usize,

    start_row_id: u32,

    deletion_vector: DeletionVector,
}

impl Updater {
    /// Create a new updater with source reader, and destination writer.
    ///
    /// The `schemas` parameter is a tuple of the write schema (just the new fields)
    /// and the final schema (all the fields).
    ///
    /// If the schemas are not known, they can be None and will be inferred from
    /// the first batch of results.
    pub(super) fn new(
        fragment: FileFragment,
        reader: FragmentReader,
        deletion_vector: DeletionVector,
        schemas: Option<(Schema, Schema)>,
    ) -> Self {
        let (write_schema, final_schema) = if let Some((write_schema, final_schema)) = schemas {
            (Some(write_schema), Some(final_schema))
        } else {
            (None, None)
        };

        Self {
            fragment,
            reader,
            last_input: None,
            writer: None,
            write_schema,
            final_schema,
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
        if self.batch_id >= self.reader.legacy_num_batches() {
            return Ok(None);
        }
        let batch = self.reader.legacy_read_batch(self.batch_id, ..).await?;
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
    async fn new_writer(&mut self, schema: Schema) -> Result<FileWriter<ManifestDescribing>> {
        let file_name = format!("{}.lance", Uuid::new_v4());
        self.fragment.metadata.add_file_legacy(&file_name, &schema);

        let full_path = self.fragment.dataset().data_dir().child(file_name.as_str());

        FileWriter::try_new(
            self.fragment.dataset().object_store.as_ref(),
            &full_path,
            schema,
            &Default::default(),
        )
        .await
    }

    /// Update one batch.
    pub async fn update(&mut self, batch: RecordBatch) -> Result<()> {
        let Some(last) = self.last_input.as_ref() else {
            return Err(Error::io(
                // TODO: Define more granular Error and wrap it in here.
                "Fragment Updater: no input data is available before update".to_string(),
                location!(),
            ));
        };

        if last.num_rows() != batch.num_rows() {
            return Err(Error::io(       // TODO: Define more granular Error and wrap it in here.
                format!(
                    "Fragment Updater: new batch has different size with the source batch: {} != {}",
                    last.num_rows(),
                    batch.num_rows()
                ),
                location!(),
            ));
        };

        if self.writer.is_none() {
            if self.write_schema.is_none() {
                // Need to infer the schema.
                let output_schema = batch.schema();
                let mut final_schema = self.fragment.schema().merge(output_schema.as_ref())?;
                final_schema.set_field_id(Some(self.fragment.dataset().manifest.max_field_id()));
                self.final_schema = Some(final_schema);
                self.final_schema.as_ref().unwrap().validate()?;
                self.write_schema = Some(
                    self.final_schema
                        .as_ref()
                        .unwrap()
                        .project_by_schema(output_schema.as_ref())?,
                );
            }

            self.writer = Some(
                self.new_writer(self.write_schema.as_ref().unwrap().clone())
                    .await?,
            );
        }

        let writer = self.writer.as_mut().unwrap();
        // Because of deleted rows, the number of row ids in the batch might not
        // match the length.
        let row_id_stride = self.reader.legacy_num_rows_in_batch(self.batch_id - 1) as u32; // Subtract since we incremented in next()
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
                location: location!(),
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

    /// Get the final schema of the fragment after the update.
    ///
    /// This may be None if the schema is not known. This can happen if it was
    /// not specified up front and the first batch of results has not yet been
    /// processed.
    pub fn schema(&self) -> Option<&Schema> {
        self.final_schema.as_ref()
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
            location: location!(),
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
                    location: location!(),
                }
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let batch = RecordBatch::try_new(batch.schema(), arrays)?;

    Ok(batch)
}
