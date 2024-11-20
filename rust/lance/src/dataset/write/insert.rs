// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::RecordBatchIterator;
use arrow_array::RecordBatchReader;
use lance_core::datatypes::NullabilityComparison;
use lance_core::datatypes::Schema;
use lance_core::datatypes::SchemaCompareOptions;
use lance_datafusion::utils::peek_reader_schema;
use lance_datafusion::utils::reader_to_stream;
use lance_file::version::LanceFileVersion;
use lance_io::object_store::ObjectStore;
use lance_table::feature_flags::can_write_dataset;
use lance_table::io::commit::CommitHandler;
use object_store::path::Path;
use snafu::{location, Location};

use crate::dataset::builder::DatasetBuilder;
use crate::dataset::transaction::Operation;
use crate::dataset::transaction::Transaction;
use crate::dataset::write::write_fragments_internal;
use crate::dataset::ReadParams;
use crate::Dataset;
use crate::{Error, Result};

use super::commit::CommitBuilder;
use super::resolve_commit_handler;
use super::WriteDestination;
use super::WriteMode;
use super::WriteParams;
use super::WrittenFragments;

/// Insert or create a new dataset.
///
/// There are different variants of `execute()` methods. Those with the `_stream`
/// suffix take an iterator of data so that larger than memory data can be written
/// out. However, this eliminates optimizations that can be made when the full
/// data is known up-front.
///
/// Those with the `_uncommitted` suffix write the data files but do not commit
/// the transactions. These changes to the dataset will not be visible until
/// they are passed to the [`CommitBuilder`].
#[derive(Debug, Clone)]
pub struct InsertBuilder<'a> {
    dest: WriteDestination<'a>,
    // TODO: make these parameters a part of the builder, and add specific methods.
    params: Option<&'a WriteParams>,
}

impl<'a> InsertBuilder<'a> {
    pub fn new(dest: impl Into<WriteDestination<'a>>) -> Self {
        Self {
            dest: dest.into(),
            params: None,
        }
    }

    pub fn with_params(mut self, params: &'a WriteParams) -> Self {
        self.params = Some(params);
        self
    }

    /// Execute the insert operation with the given data.
    ///
    /// This writes the data fragments and commits them into the dataset.
    pub async fn execute(&self, data: Vec<RecordBatch>) -> Result<Dataset> {
        let (transaction, context) = self.write_uncommitted_impl(data).await?;
        Self::do_commit(&context, transaction).await
    }

    /// Execute the insert operation with the given stream.
    ///
    /// This writes the data fragments and commits them into the dataset.
    pub async fn execute_stream(
        &self,
        stream: impl RecordBatchReader + Send + 'static,
    ) -> Result<Dataset> {
        // Box it so we don't monomorphize for every one. We take the generic
        // parameter for API ergonomics.
        let stream = Box::new(stream);
        self.execute_stream_impl(stream).await
    }

    async fn execute_stream_impl(
        &self,
        stream: Box<dyn RecordBatchReader + Send + 'static>,
    ) -> Result<Dataset> {
        let (transaction, context) = self.write_uncommitted_stream_impl(stream).await?;
        Self::do_commit(&context, transaction).await
    }

    /// Write data files, but don't commit the transaction yet.
    ///
    /// Use [`CommitBuilder`] to commit the transaction.
    ///
    /// # Example: Append data to a dataset
    ///
    /// ```rust
    /// use lance::dataset::{CommitBuilder, InsertBuilder, WriteMode, WriteParams};
    ///
    /// # use std::sync::Arc;
    /// # use arrow_array::RecordBatch;
    /// # use lance::Result;
    /// # use lance::dataset::Dataset;
    /// # async fn example(dataset: Arc<Dataset>, data: Vec<RecordBatch>) -> Result<()> {
    /// let transaction = InsertBuilder::new(dataset.clone())
    ///     .with_params(&WriteParams { mode: WriteMode::Append, ..Default::default() })
    ///     .execute_uncommitted(data)
    ///     .await?;
    /// CommitBuilder::new(dataset)
    ///     .execute(transaction)
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn execute_uncommitted(&self, data: Vec<RecordBatch>) -> Result<Transaction> {
        self.write_uncommitted_impl(data).await.map(|(t, _)| t)
    }

    async fn do_commit(context: &WriteContext<'_>, transaction: Transaction) -> Result<Dataset> {
        let mut commit_builder = CommitBuilder::new(context.dest.clone())
            .use_move_stable_row_ids(context.params.enable_move_stable_row_ids)
            .with_storage_format(context.storage_version)
            .with_object_store_registry(context.params.object_store_registry.clone())
            .enable_v2_manifest_paths(context.params.enable_v2_manifest_paths)
            .with_commit_handler(context.commit_handler.clone())
            .with_object_store(context.object_store.clone());

        if let Some(params) = context.params.store_params.as_ref() {
            commit_builder = commit_builder.with_store_params(params.clone());
        }

        if let Some(session) = context.params.session.as_ref() {
            commit_builder = commit_builder.with_session(session.clone());
        }

        commit_builder.execute(transaction).await
    }

    async fn write_uncommitted_impl(
        &self,
        data: Vec<RecordBatch>,
    ) -> Result<(Transaction, WriteContext<'_>)> {
        // TODO: This should be able to split the data up based on max_rows_per_file
        // and write in parallel. https://github.com/lancedb/lance/issues/1980
        if data.is_empty() {
            return Err(Error::InvalidInput {
                source: "No data to write".into(),
                location: location!(),
            });
        }
        let schema = data[0].schema();
        for batch in data.iter().skip(1) {
            if batch.schema() != schema {
                return Err(Error::InvalidInput {
                    source: "All record batches must have the same schema".into(),
                    location: location!(),
                });
            }
        }
        let reader = RecordBatchIterator::new(data.into_iter().map(Ok), schema);
        self.write_uncommitted_stream_impl(Box::new(reader)).await
    }

    /// Write data files, but don't commit the transaction yet.
    ///
    /// Use [`CommitBuilder`] to commit the transaction.
    pub async fn execute_uncommitted_stream(
        &self,
        stream: Box<dyn RecordBatchReader + Send + 'static>,
    ) -> Result<Transaction> {
        let (transaction, _) = self.write_uncommitted_stream_impl(stream).await?;
        Ok(transaction)
    }

    async fn write_uncommitted_stream_impl(
        &self,
        stream: Box<dyn RecordBatchReader + Send + 'static>,
    ) -> Result<(Transaction, WriteContext<'_>)> {
        let mut context = self.resolve_context().await?;

        let (batches, schema) = peek_reader_schema(stream).await?;
        let stream = reader_to_stream(batches);

        self.validate_write(&mut context, &schema)?;

        let written_frags = write_fragments_internal(
            context.dest.dataset(),
            context.object_store.clone(),
            &context.base_path,
            schema.clone(),
            stream,
            context.params.clone(),
        )
        .await?;

        let transaction = Self::build_transaction(schema, written_frags, &context)?;

        Ok((transaction, context))
    }

    fn build_transaction(
        schema: Schema,
        written_frags: WrittenFragments,
        context: &WriteContext<'_>,
    ) -> Result<Transaction> {
        let operation = match context.params.mode {
            WriteMode::Create | WriteMode::Overwrite => Operation::Overwrite {
                // Use the full schema, not the written schema
                schema,
                fragments: written_frags.default.0,
                config_upsert_values: None,
            },
            WriteMode::Append => Operation::Append {
                fragments: written_frags.default.0,
            },
        };

        let blobs_op = written_frags.blob.map(|blob| match context.params.mode {
            WriteMode::Create | WriteMode::Overwrite => Operation::Overwrite {
                schema: blob.1,
                fragments: blob.0,
                config_upsert_values: None,
            },
            WriteMode::Append => Operation::Append { fragments: blob.0 },
        });

        Ok(Transaction::new(
            context
                .dest
                .dataset()
                .map(|ds| ds.manifest.version)
                .unwrap_or(0),
            operation,
            blobs_op,
            None,
        ))
    }

    fn validate_write(&self, context: &mut WriteContext, data_schema: &Schema) -> Result<()> {
        // Write mode
        match (&context.params.mode, &context.dest) {
            (WriteMode::Create, WriteDestination::Dataset(ds)) => {
                return Err(Error::DatasetAlreadyExists {
                    uri: ds.uri.clone(),
                    location: location!(),
                });
            }
            (WriteMode::Append | WriteMode::Overwrite, WriteDestination::Uri(uri)) => {
                log::warn!("No existing dataset at {uri}, it will be created");
                context.params.mode = WriteMode::Create;
            }
            _ => {}
        }

        // Validate schema
        if matches!(context.params.mode, WriteMode::Append) {
            if let WriteDestination::Dataset(dataset) = &context.dest {
                // If the dataset is already using (or not using) move stable row ids, we need to match
                // and ignore whatever the user provided as input
                if context.params.enable_move_stable_row_ids
                    != dataset.manifest.uses_move_stable_row_ids()
                {
                    log::info!(
                        "Ignoring user provided move stable row ids setting of {}, dataset already has it set to {}",
                        context.params.enable_move_stable_row_ids,
                        dataset.manifest.uses_move_stable_row_ids()
                    );
                    context.params.enable_move_stable_row_ids =
                        dataset.manifest.uses_move_stable_row_ids();
                }
                let m = dataset.manifest.as_ref();
                let mut schema_cmp_opts = SchemaCompareOptions {
                    compare_dictionary: true,
                    // array nullability is checked later, using actual data instead
                    // of the schema
                    compare_nullability: NullabilityComparison::Ignore,
                    ..Default::default()
                };
                if m.blob_dataset_version.is_none() {
                    // Balanced datasets don't yet support schema evolution
                    schema_cmp_opts.ignore_field_order = true;
                    schema_cmp_opts.allow_missing_if_nullable = true;
                }

                data_schema.check_compatible(&m.schema, &schema_cmp_opts)?;
            }
        }

        // If we are writing a dataset with non-default storage, we need to enable move stable row ids
        if context.dest.dataset().is_none()
            && !context.params.enable_move_stable_row_ids
            && data_schema.fields.iter().any(|f| !f.is_default_storage())
        {
            log::info!("Enabling move stable row ids because non-default storage is used");
            context.params.enable_move_stable_row_ids = true;
        }

        // Feature flags
        if let WriteDestination::Dataset(dataset) = &context.dest {
            if !can_write_dataset(dataset.manifest.writer_feature_flags) {
                let message = format!(
                    "This dataset cannot be written by this version of Lance. \
                Please upgrade Lance to write to this dataset.\n Flags: {}",
                    dataset.manifest.writer_feature_flags
                );
                return Err(Error::NotSupported {
                    source: message.into(),
                    location: location!(),
                });
            }
        }

        Ok(())
    }

    async fn resolve_context(&self) -> Result<WriteContext<'a>> {
        let params = self.params.cloned().unwrap_or_default();
        let (object_store, base_path, commit_handler) = match &self.dest {
            WriteDestination::Dataset(dataset) => (
                dataset.object_store.clone(),
                dataset.base.clone(),
                dataset.commit_handler.clone(),
            ),
            WriteDestination::Uri(uri) => {
                let (object_store, base_path) = ObjectStore::from_uri_and_params(
                    params.object_store_registry.clone(),
                    uri,
                    &params.store_params.clone().unwrap_or_default(),
                )
                .await?;
                let commit_handler = resolve_commit_handler(
                    uri,
                    params.commit_handler.clone(),
                    &params.store_params,
                )
                .await?;
                (Arc::new(object_store), base_path, commit_handler)
            }
        };
        let dest = match &self.dest {
            WriteDestination::Dataset(dataset) => WriteDestination::Dataset(dataset.clone()),
            WriteDestination::Uri(uri) => {
                // Check if it already exists.
                let builder = DatasetBuilder::from_uri(uri).with_read_params(ReadParams {
                    store_options: params.store_params.clone(),
                    commit_handler: params.commit_handler.clone(),
                    object_store_registry: params.object_store_registry.clone(),
                    ..Default::default()
                });

                match builder.load().await {
                    Ok(dataset) => WriteDestination::Dataset(Arc::new(dataset)),
                    Err(Error::DatasetNotFound { .. } | Error::NotFound { .. }) => {
                        WriteDestination::Uri(uri)
                    }
                    Err(e) => return Err(e),
                }
            }
        };

        let storage_version = match (&params.mode, &dest) {
            (WriteMode::Overwrite, WriteDestination::Dataset(dataset)) => {
                // If overwriting an existing dataset, allow the user to specify but use
                // the existing version if they don't
                params.data_storage_version.map(Ok).unwrap_or_else(|| {
                    let m = dataset.manifest.as_ref();
                    m.data_storage_format.lance_file_version()
                })?
            }
            (_, WriteDestination::Dataset(dataset)) => {
                // If appending to an existing dataset, always use the dataset version
                let m = dataset.manifest.as_ref();
                m.data_storage_format.lance_file_version()?
            }
            // Otherwise (no existing dataset) fallback to the default if the user didn't specify
            (_, WriteDestination::Uri(_)) => params.storage_version_or_default(),
        };

        Ok(WriteContext {
            params,
            dest,
            object_store,
            base_path,
            commit_handler,
            storage_version,
        })
    }
}

#[derive(Debug)]
struct WriteContext<'a> {
    params: WriteParams,
    dest: WriteDestination<'a>,
    object_store: Arc<ObjectStore>,
    base_path: Path,
    commit_handler: Arc<dyn CommitHandler>,
    storage_version: LanceFileVersion,
}
