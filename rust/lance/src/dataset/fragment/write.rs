// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_schema::Schema as ArrowSchema;
use datafusion::execution::SendableRecordBatchStream;
use futures::{StreamExt, TryStreamExt};
use lance_core::datatypes::Schema;
use lance_core::Error;
use lance_datafusion::chunker::{break_stream, chunk_stream};
use lance_datafusion::utils::StreamingWriteSource;
use lance_file::v2::writer::FileWriterOptions;
use lance_file::version::LanceFileVersion;
use lance_file::writer::FileWriter;
use lance_io::object_store::ObjectStore;
use lance_table::format::{DataFile, Fragment};
use lance_table::io::manifest::ManifestDescribing;
use snafu::location;
use std::borrow::Cow;
use uuid::Uuid;

use crate::dataset::builder::DatasetBuilder;
use crate::dataset::write::do_write_fragments;
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
        source: impl StreamingWriteSource,
        id: Option<u64>,
    ) -> Result<Fragment> {
        let (stream, schema) = self.get_stream_and_schema(Box::new(source)).await?;
        self.write_impl(stream, schema, id).await
    }

    /// Write multi fragment which separated by max_rows_per_file.
    pub async fn write_fragments(
        &self,
        source: impl StreamingWriteSource,
    ) -> Result<Vec<Fragment>> {
        let (stream, schema) = self.get_stream_and_schema(Box::new(source)).await?;
        self.write_fragments_v2_impl(stream, schema).await
    }

    async fn write_v2_impl(
        &self,
        stream: SendableRecordBatchStream,
        schema: Schema,
        id: u64,
    ) -> Result<Fragment> {
        let params = self.write_params.map(Cow::Borrowed).unwrap_or_default();
        let progress = params.progress.as_ref();

        Self::validate_schema(&schema, stream.schema().as_ref())?;

        let (object_store, base_path) = ObjectStore::from_uri_and_params(
            params.store_registry(),
            self.dataset_uri,
            &params.store_params.clone().unwrap_or_default(),
        )
        .await?;
        let filename = format!("{}.lance", Uuid::new_v4());
        let mut fragment = Fragment::new(id);
        let full_path = base_path.child(DATA_DIR).child(filename.clone());
        let obj_writer = object_store.create(&full_path).await?;
        let mut writer = lance_file::v2::writer::FileWriter::try_new(
            obj_writer,
            schema,
            FileWriterOptions {
                format_version: params.data_storage_version,
                ..Default::default()
            },
        )?;

        let (major, minor) = writer.version().to_numbers();

        let data_file = DataFile::new_unstarted(filename, major, minor);
        fragment.files.push(data_file);

        progress.begin(&fragment).await?;

        let break_limit = (128 * 1024).min(params.max_rows_per_file);

        let mut broken_stream = break_stream(stream, break_limit)
            .map_ok(|batch| vec![batch])
            .boxed();
        while let Some(batched_chunk) = broken_stream.next().await {
            let batch_chunk = batched_chunk?;
            writer.write_batches(batch_chunk.iter()).await?;
        }

        fragment.physical_rows = Some(writer.finish().await? as usize);

        if matches!(fragment.physical_rows, Some(0)) {
            return Err(Error::invalid_input("Input data was empty.", location!()));
        }

        let field_ids = writer
            .field_id_to_column_indices()
            .iter()
            .map(|(field_id, _)| *field_id as i32)
            .collect::<Vec<_>>();
        let column_indices = writer
            .field_id_to_column_indices()
            .iter()
            .map(|(_, column_index)| *column_index as i32)
            .collect::<Vec<_>>();

        fragment.files[0].fields = field_ids;
        fragment.files[0].column_indices = column_indices;

        progress.complete(&fragment).await?;

        Ok(fragment)
    }
    async fn write_fragments_v2_impl(
        &self,
        stream: SendableRecordBatchStream,
        schema: Schema,
    ) -> Result<Vec<Fragment>> {
        let params = self.write_params.map(Cow::Borrowed).unwrap_or_default();

        Self::validate_schema(&schema, stream.schema().as_ref())?;

        let (object_store, base_path) = ObjectStore::from_uri_and_params(
            params.store_registry(),
            self.dataset_uri,
            &params.store_params.clone().unwrap_or_default(),
        )
        .await?;
        do_write_fragments(
            object_store,
            &base_path,
            &schema,
            stream,
            params.into_owned(),
            LanceFileVersion::Stable,
        )
        .await
    }

    async fn write_impl(
        &self,
        stream: SendableRecordBatchStream,
        schema: Schema,
        id: Option<u64>,
    ) -> Result<Fragment> {
        let id = id.unwrap_or_default();

        let params = self.write_params.map(Cow::Borrowed).unwrap_or_default();

        let storage_version = params.storage_version_or_default();

        if storage_version != LanceFileVersion::Legacy {
            return self.write_v2_impl(stream, schema, id).await;
        }
        let progress = params.progress.as_ref();

        Self::validate_schema(&schema, stream.schema().as_ref())?;

        let (object_store, base_path) = ObjectStore::from_uri_and_params(
            params.store_registry(),
            self.dataset_uri,
            &params.store_params.clone().unwrap_or_default(),
        )
        .await?;
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

        progress.begin(&fragment).await?;

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
        source: impl StreamingWriteSource,
    ) -> Result<(SendableRecordBatchStream, Schema)> {
        if let Some(schema) = self.schema {
            return Ok((source.into_stream(), schema.clone()));
        } else if matches!(self.write_params.map(|p| p.mode), Some(WriteMode::Append)) {
            if let Some(schema) = self.existing_dataset_schema().await? {
                return Ok((source.into_stream(), schema));
            }
        }
        source.into_stream_and_schema().await
    }

    async fn existing_dataset_schema(&self) -> Result<Option<Schema>> {
        let mut builder = DatasetBuilder::from_uri(self.dataset_uri);
        let storage_options = self
            .write_params
            .and_then(|p| p.store_params.as_ref())
            .and_then(|p| p.storage_options.clone());
        if let Some(storage_options) = storage_options {
            builder = builder.with_storage_options(storage_options);
        }
        match builder.load().await {
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

    use arrow_array::{
        Int64Array, RecordBatch, RecordBatchIterator, RecordBatchReader, StringArray,
    };
    use arrow_schema::{DataType, Field as ArrowField};
    use lance_arrow::SchemaExt;
    use rstest::rstest;

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
        assert_eq!(fragment.files[0].fields, vec![3, 1]);
        assert_eq!(fragment.files[0].column_indices, vec![0, 1]);
    }

    #[tokio::test]
    async fn test_write_fragments_validation() {
        // Writing with empty schema produces an error
        let empty_schema = Arc::new(ArrowSchema::empty());
        let empty_reader = Box::new(RecordBatchIterator::new(vec![], empty_schema));
        let tmp_dir = tempfile::tempdir().unwrap();
        let result = FragmentCreateBuilder::new(tmp_dir.path().to_str().unwrap())
            .write_fragments(empty_reader)
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
            .write_fragments(empty_reader)
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);

        // Writing with incorrect schema produces an error.
        let wrong_schema = arrow_schema
            .as_ref()
            .try_with_column(ArrowField::new("c", DataType::Utf8, false))
            .unwrap();
        let wrong_schema = Schema::try_from(&wrong_schema).unwrap();
        let result = FragmentCreateBuilder::new(tmp_dir.path().to_str().unwrap())
            .schema(&wrong_schema)
            .write_fragments(test_data())
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
    async fn test_write_fragments_default_schema() {
        // Infers schema and uses 0 as default field id
        let data = test_data();
        let tmp_dir = tempfile::tempdir().unwrap();
        let fragments = FragmentCreateBuilder::new(tmp_dir.path().to_str().unwrap())
            .write_fragments(data)
            .await
            .unwrap();

        // If unspecified, the fragment id should be 0.
        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0].deletion_file, None);
        assert_eq!(fragments[0].files.len(), 1);
        assert_eq!(fragments[0].files[0].fields, vec![0, 1]);
    }

    #[tokio::test]
    async fn test_write_fragments_with_options() {
        // Uses provided schema. Field ids are correct in fragment metadata.
        let data = test_data();
        let tmp_dir = tempfile::tempdir().unwrap();
        let writer_params = WriteParams {
            max_rows_per_file: 1,
            ..Default::default()
        };
        let fragments = FragmentCreateBuilder::new(tmp_dir.path().to_str().unwrap())
            .write_params(&writer_params)
            .write_fragments(data)
            .await
            .unwrap();

        assert_eq!(fragments.len(), 3);
        assert_eq!(fragments[0].deletion_file, None);
        assert_eq!(fragments[0].files.len(), 1);
        assert_eq!(fragments[0].files[0].column_indices, vec![0, 1]);
        assert_eq!(fragments[1].deletion_file, None);
        assert_eq!(fragments[1].files.len(), 1);
        assert_eq!(fragments[1].files[0].column_indices, vec![0, 1]);
        assert_eq!(fragments[2].deletion_file, None);
        assert_eq!(fragments[2].files.len(), 1);
        assert_eq!(fragments[2].files[0].column_indices, vec![0, 1]);
    }

    #[rstest]
    #[tokio::test]
    async fn test_write_fragments_with_format_version(
        #[values(
            LanceFileVersion::V2_0,
            LanceFileVersion::V2_1,
            LanceFileVersion::Legacy,
            LanceFileVersion::Stable
        )]
        file_version: LanceFileVersion,
    ) {
        let data = test_data();
        let tmp_dir = tempfile::tempdir().unwrap();
        let writer_params = WriteParams {
            data_storage_version: Some(file_version),
            ..Default::default()
        };
        let fragment = FragmentCreateBuilder::new(tmp_dir.path().to_str().unwrap())
            .write_params(&writer_params)
            .write(data, None)
            .await
            .unwrap();

        assert!(!fragment.files.is_empty());
        fragment.files.iter().for_each(|f| {
            let (major_version, minor_version) = file_version.to_numbers();
            assert_eq!(f.file_major_version, major_version);
            assert_eq!(f.file_minor_version, minor_version);
        })
    }
}
