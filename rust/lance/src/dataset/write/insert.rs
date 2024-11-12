// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::RecordBatchReader;
use lance_core::datatypes::NullabilityComparison;
use lance_core::datatypes::Schema;
use lance_core::datatypes::SchemaCompareOptions;
use lance_datafusion::utils::peek_reader_schema;
use lance_datafusion::utils::reader_to_stream;
use lance_file::version::LanceFileVersion;
use lance_io::object_store::ObjectStore;
use lance_io::object_store::ObjectStoreParams;
use lance_io::object_store::ObjectStoreRegistry;
use lance_table::feature_flags::can_write_dataset;
use lance_table::io::commit::commit_handler_from_url;
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
use super::WriteMode;
use super::WriteParams;

#[derive(Debug, Clone)]
pub enum InsertDestination<'a> {
    Dataset(Arc<Dataset>),
    Uri(&'a str),
}

impl InsertDestination<'_> {
    pub fn dataset(&self) -> Option<&Dataset> {
        match self {
            InsertDestination::Dataset(dataset) => Some(dataset.as_ref()),
            InsertDestination::Uri(_) => None,
        }
    }
}

impl From<Arc<Dataset>> for InsertDestination<'_> {
    fn from(dataset: Arc<Dataset>) -> Self {
        InsertDestination::Dataset(dataset)
    }
}

impl<'a> From<&'a str> for InsertDestination<'a> {
    fn from(uri: &'a str) -> Self {
        InsertDestination::Uri(uri)
    }
}

impl<'a> From<&'a String> for InsertDestination<'a> {
    fn from(uri: &'a String) -> Self {
        InsertDestination::Uri(uri.as_str())
    }
}

impl<'a> From<&'a Path> for InsertDestination<'a> {
    fn from(path: &'a Path) -> Self {
        InsertDestination::Uri(path.as_ref())
    }
}

#[derive(Debug, Clone)]
pub struct InsertBuilder<'a> {
    dest: InsertDestination<'a>,
    // TODO: make these parameters a part of the builder, and add specific methods.
    params: Option<&'a WriteParams>,
    // TODO: num_jobs
}

impl<'a> InsertBuilder<'a> {
    pub fn new(dest: impl Into<InsertDestination<'a>>) -> Self {
        Self {
            dest: dest.into(),
            params: None,
        }
    }

    pub fn with_params(mut self, params: &'a WriteParams) -> Self {
        self.params = Some(params);
        self
    }

    pub async fn execute(&self, data: &[RecordBatch]) -> Result<Dataset> {
        // TODO: validate schema is the same for all batches
        todo!()
    }

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

    /// Write data files, but don't commit the transaction yet.
    ///
    /// Use CommitBuilder to commit the transaction.
    pub async fn write_uncommitted(&self, data: &[RecordBatch]) -> Result<Transaction> {
        todo!()
    }

    /// Write but don't commit the transaction yet.
    pub async fn write_uncommitted_stream(
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

        let transaction = Transaction::new(
            context
                .dest
                .dataset()
                .map(|ds| ds.manifest.version)
                .unwrap_or(0),
            operation,
            blobs_op,
            None,
        );

        Ok((transaction, context))
    }

    fn validate_write(&self, context: &mut WriteContext, data_schema: &Schema) -> Result<()> {
        // Write mode
        match (&context.params.mode, &context.dest) {
            (WriteMode::Create, InsertDestination::Dataset(_)) => {
                return Err(Error::InvalidInput {
                    source: "Dataset already exists".into(),
                    location: location!(),
                });
            }
            (WriteMode::Append | WriteMode::Overwrite, InsertDestination::Uri(uri)) => {
                log::warn!("No existing dataset at {uri}, it will be created");
                context.params.mode = WriteMode::Create;
            }
            _ => {}
        }

        // Validate schema
        if matches!(context.params.mode, WriteMode::Append) {
            if let InsertDestination::Dataset(dataset) = &context.dest {
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

        // Feature flags
        if let InsertDestination::Dataset(dataset) = &context.dest {
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
            InsertDestination::Dataset(dataset) => (
                dataset.object_store.clone(),
                dataset.base.clone(),
                dataset.commit_handler.clone(),
            ),
            InsertDestination::Uri(uri) => {
                Self::params_from_uri(
                    uri,
                    &params.commit_handler,
                    &params.store_params,
                    params.object_store_registry.clone(),
                )
                .await?
            }
        };
        let dest = match &self.dest {
            InsertDestination::Dataset(dataset) => InsertDestination::Dataset(dataset.clone()),
            InsertDestination::Uri(uri) => {
                // Check if it already exists.
                let builder = DatasetBuilder::from_uri(uri).with_read_params(ReadParams {
                    store_options: params.store_params.clone(),
                    commit_handler: params.commit_handler.clone(),
                    object_store_registry: params.object_store_registry.clone(),
                    ..Default::default()
                });

                match builder.load().await {
                    Ok(dataset) => InsertDestination::Dataset(Arc::new(dataset)),
                    Err(Error::DatasetNotFound { .. } | Error::NotFound { .. }) => {
                        InsertDestination::Uri(uri)
                    }
                    Err(e) => return Err(e),
                }
            }
        };

        let storage_version = match (&params.mode, &dest) {
            (WriteMode::Overwrite, InsertDestination::Dataset(dataset)) => {
                // If overwriting an existing dataset, allow the user to specify but use
                // the existing version if they don't
                params.data_storage_version.map(Ok).unwrap_or_else(|| {
                    let m = dataset.manifest.as_ref();
                    m.data_storage_format.lance_file_version()
                })?
            }
            (_, InsertDestination::Dataset(dataset)) => {
                // If appending to an existing dataset, always use the dataset version
                let m = dataset.manifest.as_ref();
                m.data_storage_format.lance_file_version()?
            }
            // Otherwise (no existing dataset) fallback to the default if the user didn't specify
            (_, InsertDestination::Uri(_)) => params.storage_version_or_default(),
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

    async fn params_from_uri(
        uri: &str,
        commit_handler: &Option<Arc<dyn CommitHandler>>,
        store_options: &Option<ObjectStoreParams>,
        object_store_registry: Arc<ObjectStoreRegistry>,
    ) -> Result<(Arc<ObjectStore>, Path, Arc<dyn CommitHandler>)> {
        let (mut object_store, base_path) = match store_options.as_ref() {
            Some(store_options) => {
                ObjectStore::from_uri_and_params(object_store_registry, uri, store_options).await?
            }
            None => ObjectStore::from_uri(uri).await?,
        };

        if let Some(block_size) = store_options.as_ref().and_then(|opts| opts.block_size) {
            object_store.set_block_size(block_size);
        }

        let commit_handler = match &commit_handler {
            None => {
                if store_options.is_some() && store_options.as_ref().unwrap().object_store.is_some()
                {
                    return Err(Error::InvalidInput { source: "when creating a dataset with a custom object store the commit_handler must also be specified".into(), location: location!() });
                }
                commit_handler_from_url(uri, store_options).await?
            }
            Some(commit_handler) => {
                if uri.starts_with("s3+ddb") {
                    return Err(Error::InvalidInput {
                        source:
                            "`s3+ddb://` scheme and custom commit handler are mutually exclusive"
                                .into(),
                        location: location!(),
                    });
                } else {
                    commit_handler.clone()
                }
            }
        };

        Ok((Arc::new(object_store), base_path, commit_handler))
    }
}

struct WriteContext<'a> {
    params: WriteParams,
    dest: InsertDestination<'a>,
    object_store: Arc<ObjectStore>,
    base_path: Path,
    commit_handler: Arc<dyn CommitHandler>,
    storage_version: LanceFileVersion,
}
