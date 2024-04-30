// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::borrow::Cow;

use arrow_array::RecordBatchReader;
use arrow_schema::Schema as ArrowSchema;
use datafusion::execution::SendableRecordBatchStream;
use futures::StreamExt;
use lance_core::datatypes::Schema;
use lance_core::Error;
use lance_datafusion::chunker::chunk_stream;
use lance_datafusion::utils::{peek_reader_schema, reader_to_stream};
use lance_file::writer::FileWriter;
use lance_io::object_store::ObjectStore;
use lance_table::format::Fragment;
use lance_table::io::manifest::ManifestDescribing;
use snafu::{location, Location};
use uuid::Uuid;

use crate::dataset::builder::DatasetBuilder;
use crate::dataset::{WriteMode, WriteParams, DATA_DIR};
use crate::Result;

/// Builder for writing a new fragment.
///
/// This builder can be re-used to write multiple fragments.
pub struct FragmentCreateBuilder<'a> {
    dataset_uri: &'a str,
    schema: Option<&'a Schema>,
    write_params: Option<&'a WriteParams>,
}

impl<'a> FragmentCreateBuilder<'a> {
    pub fn new(dataset_uri: &'a str) -> Self {
        Self {
            dataset_uri,
            schema: None,
            write_params: None,
        }
    }

    /// Set the schema of the fragment. If it is not known, it will be inferred.
    ///
    /// If the schema isn't provided, but the `write_mode` is `WriteMode::Append`,
    /// the schema will be inferred from the existing dataset.
    ///
    /// If that fails, the schema will be inferred from the first batch.
    pub fn schema(mut self, schema: &'a Schema) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Set the write parameters.
    pub fn write_params(mut self, params: &'a WriteParams) -> Self {
        self.write_params = Some(params);
        self
    }

    /// Write a fragment.
    pub async fn write(
        &self,
        reader: impl RecordBatchReader + Send + 'static,
        id: Option<u64>,
    ) -> Result<Fragment> {
        let (stream, schema) = self.get_stream_and_schema(Box::new(reader)).await?;
        self.write_impl(stream, schema, id).await
    }

    async fn write_impl(
        &self,
        stream: SendableRecordBatchStream,
        schema: Schema,
        id: Option<u64>,
    ) -> Result<Fragment> {
        let id = id.unwrap_or_default();

        let params = self.write_params.map(Cow::Borrowed).unwrap_or_default();
        let progress = params.progress.as_ref();

        Self::validate_schema(&schema, stream.schema().as_ref())?;

        let (object_store, base_path) = ObjectStore::from_uri(self.dataset_uri).await?;
        let filename = format!("{}.lance", Uuid::new_v4());
        let mut fragment = Fragment::with_file_legacy(id, &filename, &schema, None);
        let full_path = base_path.child(DATA_DIR).child(filename.clone());
        let mut writer = FileWriter::<ManifestDescribing>::try_new(
            &object_store,
            &full_path,
            schema,
            &Default::default(),
        )
        .await?;

        progress.begin(&fragment, writer.multipart_id()).await?;

        let mut buffered_reader = chunk_stream(stream, params.max_rows_per_group);
        while let Some(batched_chunk) = buffered_reader.next().await {
            let batch = batched_chunk?;
            writer.write(&batch).await?;
        }

        if writer.is_empty() {
            return Err(Error::invalid_input("Input data was empty.", location!()));
        }

        fragment.physical_rows = Some(writer.finish().await?);

        progress.complete(&fragment).await?;

        Ok(fragment)
    }

    async fn get_stream_and_schema(
        &self,
        reader: Box<dyn RecordBatchReader + Send>,
    ) -> Result<(SendableRecordBatchStream, Schema)> {
        if let Some(schema) = self.schema {
            // Just wrap the stream and use as usual.
            let stream = reader_to_stream(reader);

            return Ok((stream, schema.clone()));
        } else if matches!(self.write_params.map(|p| p.mode), Some(WriteMode::Append)) {
            if let Some(schema) = self.existing_dataset_schema().await? {
                return Ok((reader_to_stream(reader), schema));
            }
        }
        // Infer the schema from the first batch.
        let (reader, schema) = peek_reader_schema(reader).await?;
        let stream = reader_to_stream(reader);
        Ok((stream, schema))
    }

    async fn existing_dataset_schema(&self) -> Result<Option<Schema>> {
        match DatasetBuilder::from_uri(self.dataset_uri).load().await {
            Ok(dataset) => {
                // Use the schema from the dataset, because it has the correct
                // field ids.
                Ok(Some(dataset.schema().clone()))
            }
            Err(Error::DatasetNotFound { .. }) => {
                // If the dataset does not exist, we can use the schema from
                // the reader.
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

    fn validate_schema(expected: &Schema, actual: &ArrowSchema) -> Result<()> {
        if actual.fields().is_empty() {
            return Err(Error::invalid_input(
                "Cannot write with an empty schema.",
                location!(),
            ));
        }
        let actual_lance = Schema::try_from(actual)?;
        actual_lance.check_compatible(expected, &Default::default())?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator, StringArray};
    use arrow_schema::{DataType, Field as ArrowField};
    use lance_arrow::SchemaExt;

    use super::*;

    fn test_data() -> Box<dyn RecordBatchReader + Send> {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int64, false),
            ArrowField::new("b", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        );
        Box::new(RecordBatchIterator::new(vec![batch], schema))
    }

    #[tokio::test]
    async fn test_fragment_write_validation() {
        // Writing with empty schema produces an error
        let empty_schema = Arc::new(ArrowSchema::empty());
        let empty_reader = Box::new(RecordBatchIterator::new(vec![], empty_schema));
        let tmp_dir = tempfile::tempdir().unwrap();
        let result = FragmentCreateBuilder::new(tmp_dir.path().to_str().unwrap())
            .write(empty_reader, None)
            .await;
        assert!(result.is_err());
        assert!(
            matches!(result.as_ref().unwrap_err(), Error::InvalidInput { source, .. }
            if source.to_string().contains("Cannot write with an empty schema.")),
            "{:?}",
            &result
        );

        // Writing empty reader produces an error
        let arrow_schema = test_data().schema();
        let empty_reader = Box::new(RecordBatchIterator::new(vec![], arrow_schema.clone()));
        let result = FragmentCreateBuilder::new(tmp_dir.path().to_str().unwrap())
            .write(empty_reader, None)
            .await;
        assert!(result.is_err());
        assert!(
            matches!(result.as_ref().unwrap_err(), Error::InvalidInput { source, .. }
            if source.to_string().contains("Input data was empty.")),
            "{:?}",
            &result
        );

        // Writing with incorrect schema produces an error.
        let wrong_schema = arrow_schema
            .as_ref()
            .try_with_column(ArrowField::new("c", DataType::Utf8, false))
            .unwrap();
        let wrong_schema = Schema::try_from(&wrong_schema).unwrap();
        let result = FragmentCreateBuilder::new(tmp_dir.path().to_str().unwrap())
            .schema(&wrong_schema)
            .write(test_data(), None)
            .await;
        assert!(result.is_err());
        assert!(
            matches!(result.as_ref().unwrap_err(), Error::SchemaMismatch { difference, .. }
            if difference.contains("fields did not match")),
            "{:?}",
            &result
        );
    }

    #[tokio::test]
    async fn test_fragment_write_default_schema() {
        // Infers schema and uses 0 as default field id
        let data = test_data();
        let tmp_dir = tempfile::tempdir().unwrap();
        let fragment = FragmentCreateBuilder::new(tmp_dir.path().to_str().unwrap())
            .write(data, None)
            .await
            .unwrap();

        // If unspecified, the fragment id should be 0.
        assert_eq!(fragment.id, 0);
        assert_eq!(fragment.deletion_file, None);
        assert_eq!(fragment.files.len(), 1);
        assert_eq!(fragment.files[0].fields, vec![0, 1]);
    }

    #[tokio::test]
    async fn test_fragment_write_with_schema() {
        // Uses provided schema. Field ids are correct in fragment metadata.
        let data = test_data();

        let arrow_schema = data.schema();
        let mut custom_schema = Schema::try_from(arrow_schema.as_ref()).unwrap();
        custom_schema.mut_field_by_id(0).unwrap().id = 3;
        custom_schema.mut_field_by_id(1).unwrap().id = 1;

        let tmp_dir = tempfile::tempdir().unwrap();
        let fragment = FragmentCreateBuilder::new(tmp_dir.path().to_str().unwrap())
            .schema(&custom_schema)
            .write(data, Some(42))
            .await
            .unwrap();

        assert_eq!(fragment.id, 42);
        assert_eq!(fragment.deletion_file, None);
        assert_eq!(fragment.files.len(), 1);
        assert_eq!(fragment.files[0].fields, vec![1, 3]);
    }
}
