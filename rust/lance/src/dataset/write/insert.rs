// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchIterator};
use datafusion::execution::SendableRecordBatchStream;
use humantime::format_duration;
use lance_core::datatypes::{NullabilityComparison, Schema, SchemaCompareOptions};
use lance_core::utils::tracing::{DATASET_WRITING_EVENT, TRACE_DATASET_EVENTS};
use lance_core::{ROW_ADDR, ROW_ID, ROW_OFFSET};
use lance_datafusion::utils::StreamingWriteSource;
use lance_file::version::LanceFileVersion;
use lance_io::object_store::ObjectStore;
use lance_table::feature_flags::can_write_dataset;
use lance_table::format::Fragment;
use lance_table::io::commit::CommitHandler;
use object_store::path::Path;
use snafu::location;

use crate::dataset::builder::DatasetBuilder;
use crate::dataset::transaction::{Operation, Transaction, TransactionBuilder};
use crate::dataset::write::{validate_and_resolve_target_bases, write_fragments_internal};
use crate::dataset::ReadParams;
use crate::Dataset;
use crate::{Error, Result};
use tracing::info;

use super::commit::CommitBuilder;
use super::resolve_commit_handler;
use super::WriteDestination;
use super::WriteMode;
use super::WriteParams;
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
    pub async fn execute_stream(&self, source: impl StreamingWriteSource) -> Result<Dataset> {
        let (stream, schema) = source.into_stream_and_schema().await?;
        self.execute_stream_impl(stream, schema).await
    }

    async fn execute_stream_impl(
        &self,
        stream: SendableRecordBatchStream,
        schema: Schema,
    ) -> Result<Dataset> {
        let (transaction, context) = self.write_uncommitted_stream_impl(stream, schema).await?;
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
            .use_stable_row_ids(context.params.enable_stable_row_ids)
            .with_storage_format(context.storage_version)
            .enable_v2_manifest_paths(context.params.enable_v2_manifest_paths)
            .with_commit_handler(context.commit_handler.clone())
            .with_object_store(context.object_store.clone())
            .with_skip_auto_cleanup(context.params.skip_auto_cleanup);

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
        let (stream, schema) = reader.into_stream_and_schema().await?;
        self.write_uncommitted_stream_impl(stream, schema).await
    }

    /// Write data files, but don't commit the transaction yet.
    ///
    /// Use [`CommitBuilder`] to commit the transaction.
    pub async fn execute_uncommitted_stream(
        &self,
        source: impl StreamingWriteSource,
    ) -> Result<Transaction> {
        let (stream, schema) = source.into_stream_and_schema().await?;
        let (transaction, _) = self.write_uncommitted_stream_impl(stream, schema).await?;
        Ok(transaction)
    }

    async fn write_uncommitted_stream_impl(
        &self,
        stream: SendableRecordBatchStream,
        schema: Schema,
    ) -> Result<(Transaction, WriteContext<'_>)> {
        let mut context = self.resolve_context().await?;

        info!(
            target: TRACE_DATASET_EVENTS,
            event=DATASET_WRITING_EVENT,
            uri=context.dest.uri(),
            mode=?context.params.mode
        );

        self.validate_write(&mut context, &schema)?;

        let existing_base_paths = context.dest.dataset().map(|ds| &ds.manifest.base_paths);
        let target_base_info =
            validate_and_resolve_target_bases(&mut context.params, existing_base_paths).await?;

        let (written_fragments, _) = write_fragments_internal(
            context.dest.dataset(),
            context.object_store.clone(),
            &context.base_path,
            schema.clone(),
            stream,
            context.params.clone(),
            target_base_info,
        )
        .await?;

        let transaction = Self::build_transaction(schema, written_fragments, &context)?;

        Ok((transaction, context))
    }

    fn build_transaction(
        schema: Schema,
        fragments: Vec<Fragment>,
        context: &WriteContext<'_>,
    ) -> Result<Transaction> {
        let operation = match context.params.mode {
            WriteMode::Create => {
                // Fetch auto_cleanup params from context
                let config_upsert_values = match context.params.auto_cleanup.as_ref() {
                    Some(auto_cleanup_params) => {
                        let mut upsert_values = HashMap::new();

                        upsert_values.insert(
                            String::from("lance.auto_cleanup.interval"),
                            auto_cleanup_params.interval.to_string(),
                        );

                        match auto_cleanup_params.older_than.to_std() {
                            Ok(d) => {
                                upsert_values.insert(
                                    String::from("lance.auto_cleanup.older_than"),
                                    format_duration(d).to_string(),
                                );
                            }
                            Err(e) => {
                                return Err(Error::InvalidInput {
                                    source: e.into(),
                                    location: location!(),
                                })
                            }
                        };

                        Some(upsert_values)
                    }
                    None => None,
                };

                Operation::Overwrite {
                    // Use the full schema, not the written schema
                    schema,
                    fragments,
                    config_upsert_values,
                    initial_bases: context.params.initial_bases.clone(),
                }
            }
            WriteMode::Overwrite => {
                Operation::Overwrite {
                    // Use the full schema, not the written schema
                    schema,
                    fragments,
                    config_upsert_values: None,
                    initial_bases: context.params.initial_bases.clone(),
                }
            }
            WriteMode::Append => Operation::Append { fragments },
        };

        let transaction = TransactionBuilder::new(
            context
                .dest
                .dataset()
                .map(|ds| ds.manifest.version)
                .unwrap_or(0),
            operation,
        )
        .transaction_properties(context.params.transaction_properties.clone())
        .build();

        Ok(transaction)
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
                // If the dataset is already using (or not using) stable row ids, we need to match
                // and ignore whatever the user provided as input
                if context.params.enable_stable_row_ids != dataset.manifest.uses_stable_row_ids() {
                    log::info!(
                        "Ignoring user provided stable row ids setting of {}, dataset already has it set to {}",
                        context.params.enable_stable_row_ids,
                        dataset.manifest.uses_stable_row_ids()
                    );
                    context.params.enable_stable_row_ids = dataset.manifest.uses_stable_row_ids();
                }

                let schema_cmp_opts = SchemaCompareOptions {
                    compare_dictionary: dataset.manifest.should_use_legacy_format(),
                    compare_nullability: NullabilityComparison::Ignore,
                    allow_missing_if_nullable: true,
                    ignore_field_order: true,
                    ..Default::default()
                };

                data_schema.check_compatible(dataset.schema(), &schema_cmp_opts)?;
            }
        }

        // Make sure we aren't using any reserved column names
        for field in data_schema.fields.iter() {
            if field.name == ROW_ID || field.name == ROW_ADDR || field.name == ROW_OFFSET {
                return Err(Error::InvalidInput {
                    source: format!(
                        "The column {} is a reserved name and cannot be used in a Lance dataset",
                        field.name
                    )
                    .into(),
                    location: location!(),
                });
            }
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
                let registry = params
                    .session
                    .as_ref()
                    .map(|s| s.store_registry())
                    .unwrap_or_else(|| Arc::new(Default::default()));
                let (object_store, base_path) = ObjectStore::from_uri_and_params(
                    registry,
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
                (object_store, base_path, commit_handler)
            }
        };
        let dest = match &self.dest {
            WriteDestination::Dataset(dataset) => WriteDestination::Dataset(dataset.clone()),
            WriteDestination::Uri(uri) => {
                // Check if it already exists.
                let builder = DatasetBuilder::from_uri(uri).with_read_params(ReadParams {
                    store_options: params.store_params.clone(),
                    commit_handler: params.commit_handler.clone(),
                    session: params.session.clone(),
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

#[cfg(test)]
mod test {
    use arrow_array::StructArray;
    use arrow_schema::{DataType, Field, Schema};

    use crate::session::Session;

    use super::*;

    #[tokio::test]
    async fn test_pass_session() {
        let session = Arc::new(Session::new(0, 0, Default::default()));
        let dataset = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                session: Some(session.clone()),
                ..Default::default()
            })
            .execute_stream(RecordBatchIterator::new(
                vec![],
                Arc::new(Schema::new(vec![Field::new("col", DataType::Int32, false)])),
            ))
            .await
            .unwrap();

        assert_eq!(Arc::as_ptr(&dataset.session()), Arc::as_ptr(&session));
    }

    #[tokio::test]
    async fn test_write_empty_struct() {
        // Regresses a 2.1 issue where empty structs did not get assigned any columns
        // in the file because we only look at leaf columns.
        let schema = Arc::new(Schema::new(vec![Field::new(
            "empties",
            DataType::Struct(Vec::<Field>::new().into()),
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StructArray::new_empty_fields(1, None))],
        )
        .unwrap();
        let dataset = InsertBuilder::new("memory://")
            .execute_stream(RecordBatchIterator::new(vec![Ok(batch)], schema.clone()))
            .await
            .unwrap();

        assert_eq!(
            dataset
                .count_rows(Some("empties IS NOT NULL".to_string()))
                .await
                .unwrap(),
            1
        );
    }
}
