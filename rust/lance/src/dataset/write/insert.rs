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
use lance_table::format::{
    Fragment, RowAddressLayoutDelta, RowAddressPlacementDelta, RowAddressPlacementKind,
    RowAddressTargetFragment, RowAddressTargetRange,
};
use lance_table::io::commit::CommitHandler;
use object_store::path::Path;
use uuid::Uuid;

use crate::Dataset;
use crate::blob::normalize_prepared_blob_schema;
use crate::dataset::ReadParams;
use crate::dataset::builder::DatasetBuilder;
use crate::dataset::transaction::{Operation, Transaction, TransactionBuilder};
use crate::dataset::write::{
    validate_and_resolve_target_bases_with_primary, validate_max_rows_per_file,
    write_fragments_internal_with_fragment_limit,
};
use crate::{Error, Result};
use tracing::info;

use super::WriteDestination;
use super::WriteMode;
use super::WriteParams;
use super::commit::CommitBuilder;
use super::resolve_commit_handler;
use crate::dataset::progress::{WriteProgressFn, WriteStats};

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
    write_progress: Option<WriteProgressFn>,
}

impl<'a> InsertBuilder<'a> {
    pub fn new(dest: impl Into<WriteDestination<'a>>) -> Self {
        Self {
            dest: dest.into(),
            params: None,
            write_progress: None,
        }
    }

    pub fn with_params(mut self, params: &'a WriteParams) -> Self {
        self.params = Some(params);
        self
    }

    /// Register a callback that is invoked after each batch of rows is written.
    ///
    /// The callback receives cumulative [`WriteStats`] and can be used to drive
    /// a progress bar or compute throughput. It must be cheap and non-blocking;
    /// spawn a task if you need async work.
    ///
    /// This overrides any `write_progress` set in [`WriteParams`].
    pub fn progress(mut self, callback: impl Fn(WriteStats) + Send + Sync + 'static) -> Self {
        self.write_progress = Some(WriteProgressFn::new(callback));
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
        // and write in parallel. https://github.com/lance-format/lance/issues/1980
        if data.is_empty() {
            return Err(Error::invalid_input_source("No data to write".into()));
        }
        let schema = data[0].schema();
        for batch in data.iter().skip(1) {
            if batch.schema() != schema {
                return Err(Error::invalid_input_source(
                    "All record batches must have the same schema".into(),
                ));
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
        let max_output_fragments = Self::v2_3_output_fragment_limit(&context)?;

        let existing_base_paths = context.dest.dataset().map(|ds| &ds.manifest.base_paths);
        let target_base_info = validate_and_resolve_target_bases_with_primary(
            &mut context.params,
            existing_base_paths,
            &context.object_store,
            &context.base_path,
            &context.dest.uri(),
        )
        .await?;

        let (written_fragments, written_schema) = write_fragments_internal_with_fragment_limit(
            context.dest.dataset(),
            context.object_store.clone(),
            &context.base_path,
            schema.clone(),
            stream,
            context.params.clone(),
            target_base_info,
            max_output_fragments,
        )
        .await?;

        let transaction = Self::build_transaction(written_schema, written_fragments, &context)?;

        Ok((transaction, context))
    }

    fn build_transaction(
        schema: Schema,
        fragments: Vec<Fragment>,
        context: &WriteContext<'_>,
    ) -> Result<Transaction> {
        let operation = match context.params.mode {
            WriteMode::Create => {
                let mut upsert_values = HashMap::new();
                if let Some(auto_cleanup_params) = context.params.auto_cleanup.as_ref() {
                    upsert_values.insert(
                        String::from("lance.auto_cleanup.interval"),
                        auto_cleanup_params.interval.to_string(),
                    );

                    let duration = auto_cleanup_params
                        .older_than
                        .to_std()
                        .map_err(|e| Error::invalid_input_source(e.into()))?;
                    upsert_values.insert(
                        String::from("lance.auto_cleanup.older_than"),
                        format_duration(duration).to_string(),
                    );
                }
                let config_upsert_values = if upsert_values.is_empty() {
                    None
                } else {
                    Some(upsert_values)
                };
                Operation::Overwrite {
                    // Use the full schema, not the written schema
                    schema,
                    fragments,
                    config_upsert_values,
                    initial_bases: context.params.initial_bases.clone(),
                }
            }
            WriteMode::Overwrite => Operation::Overwrite {
                schema,
                fragments,
                config_upsert_values: None,
                initial_bases: context.params.initial_bases.clone(),
            },
            WriteMode::Append => Operation::Append { fragments },
        };

        let row_address_layout_delta = if context.storage_version.resolve()
            == LanceFileVersion::V2_3
        {
            let mut delta = if matches!(context.params.mode, WriteMode::Create) {
                RowAddressLayoutDelta::for_create(Uuid::new_v4())
            } else {
                let layout = context
                    .dest
                    .dataset()
                    .and_then(|dataset| dataset.manifest.row_address_layout.as_ref())
                    .ok_or_else(|| {
                        Error::invalid_input(
                            "storage-version-2.3 successor transaction requires a row-address layout",
                        )
                    })?;
                RowAddressLayoutDelta {
                    expected_layout_fingerprint: layout.fingerprint.clone(),
                    ..Default::default()
                }
            };
            let (Operation::Append {
                fragments: new_fragments,
            }
            | Operation::Overwrite {
                fragments: new_fragments,
                ..
            }) = &operation
            else {
                unreachable!("insert builder only produces append or overwrite operations")
            };
            delta.placements = new_fragments
                .iter()
                .enumerate()
                .map(|(ordinal, fragment)| {
                    let ordinal = checked_v2_3_output_fragment_ordinal(ordinal)?;
                    let physical_rows = fragment.physical_rows.ok_or_else(|| {
                        Error::invalid_input(
                            "storage-version-2.3 output fragment is missing physical_rows",
                        )
                    })?;
                    let physical_rows = checked_v2_3_output_fragment_rows(physical_rows)?;
                    if physical_rows == 0 {
                        return Err(Error::invalid_input(
                            "storage-version-2.3 output fragments must not be empty",
                        ));
                    }
                    Ok(RowAddressPlacementDelta {
                        source_selections: Vec::new(),
                        target: RowAddressTargetRange {
                            fragment: RowAddressTargetFragment::NewFragmentOrdinal(ordinal),
                            start_offset: 0,
                            end_offset: physical_rows,
                        },
                        placement_kind: RowAddressPlacementKind::Direct,
                        output_cardinality: physical_rows as u64,
                        output_row_sequence_fingerprint: Vec::new(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Some(delta)
        } else {
            None
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
        .row_address_layout_delta(row_address_layout_delta)
        .build();

        Ok(transaction)
    }

    fn v2_3_output_fragment_limit(context: &WriteContext<'_>) -> Result<Option<u64>> {
        if context.storage_version.resolve() != LanceFileVersion::V2_3 {
            return Ok(None);
        }

        let Some(dataset) = context.dest.dataset() else {
            // A new namespace can allocate every id except the reserved sentinel.
            return Ok(Some(u32::MAX as u64));
        };
        let physical_remaining =
            remaining_v2_3_fragment_ids(dataset.manifest.max_fragment_id(), "physical fragment")?;
        let logical_remaining = remaining_v2_3_fragment_ids(
            dataset.manifest.max_logical_fragment_id.map(u64::from),
            "logical fragment",
        )?;
        Ok(Some(physical_remaining.min(logical_remaining)))
    }

    fn validate_write(&self, context: &mut WriteContext, data_schema: &Schema) -> Result<()> {
        // Write mode
        match (&context.params.mode, &context.dest) {
            (WriteMode::Create, WriteDestination::Dataset(ds)) => {
                return Err(Error::dataset_already_exists(ds.uri.clone()));
            }
            (WriteMode::Append | WriteMode::Overwrite, WriteDestination::Uri(uri)) => {
                log::warn!("No existing dataset at {uri}, it will be created");
                context.params.mode = WriteMode::Create;
            }
            _ => {}
        }

        validate_max_rows_per_file(context.params.max_rows_per_file)?;
        if context.storage_version.resolve() == LanceFileVersion::V2_3
            && context.params.max_rows_per_file > u32::MAX as usize
        {
            return Err(Error::format_capacity_exceeded(format!(
                "max_rows_per_file {} exceeds the storage-version-2.3 per-fragment slot limit {}",
                context.params.max_rows_per_file,
                u32::MAX
            )));
        }

        // Validate schema
        if matches!(context.params.mode, WriteMode::Append)
            && let WriteDestination::Dataset(dataset) = &context.dest
        {
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

            let normalized_data_schema = normalize_prepared_blob_schema(data_schema)?;
            normalized_data_schema.check_compatible(dataset.schema(), &schema_cmp_opts)?;
        }

        // Make sure we aren't using any reserved column names
        for field in data_schema.fields.iter() {
            if field.name == ROW_ID || field.name == ROW_ADDR || field.name == ROW_OFFSET {
                return Err(Error::invalid_input_source(
                    format!(
                        "The column {} is a reserved name and cannot be used in a Lance dataset",
                        field.name
                    )
                    .into(),
                ));
            }
        }

        // Feature flags
        if let WriteDestination::Dataset(dataset) = &context.dest
            && !can_write_dataset(dataset.manifest.writer_feature_flags)
        {
            let message = format!(
                "This dataset cannot be written by this version of Lance. \
                Please upgrade Lance to write to this dataset.\n Flags: {}",
                dataset.manifest.writer_feature_flags
            );
            return Err(Error::not_supported_source(message.into()));
        }

        Ok(())
    }

    async fn resolve_context(&self) -> Result<WriteContext<'a>> {
        let mut params = self.params.cloned().unwrap_or_default();
        if let Some(cb) = self.write_progress.clone() {
            params.write_progress = Some(cb);
        }
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

        // Stable logical row identity is intrinsic to storage version 2.3.
        // The legacy toggle controls row-id sidecars and must not make two
        // different 2.3 formats (or leave legacy metadata behind when callers
        // keep passing an old option).
        if storage_version.resolve() == LanceFileVersion::V2_3 {
            params.enable_stable_row_ids = false;
        }

        if let WriteDestination::Dataset(dataset) = &dest {
            let current_version = dataset
                .manifest
                .data_storage_format
                .lance_file_version()?
                .resolve();
            let requested_version = storage_version.resolve();
            if current_version != requested_version
                && (current_version == LanceFileVersion::V2_3
                    || requested_version == LanceFileVersion::V2_3)
            {
                return Err(Error::not_supported_source(
                    format!(
                        "Cannot change an existing dataset between storage version {} and {}; storage version 2.3 is only supported for newly created datasets",
                        current_version, requested_version
                    )
                    .into(),
                ));
            }
        }

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

fn remaining_v2_3_fragment_ids(maximum: Option<u64>, allocator: &str) -> Result<u64> {
    let maximum_valid_id = u32::MAX as u64 - 1;
    match maximum {
        None => Ok(u32::MAX as u64),
        Some(maximum) => maximum_valid_id.checked_sub(maximum).ok_or_else(|| {
            Error::format_capacity_exceeded(format!(
                "storage-version-2.3 {allocator} high-water mark {maximum} reaches the reserved id {}",
                u32::MAX
            ))
        }),
    }
}

fn checked_v2_3_output_fragment_ordinal(ordinal: usize) -> Result<u32> {
    u32::try_from(ordinal).map_err(|_| {
        Error::format_capacity_exceeded(format!(
            "storage-version-2.3 output fragment ordinal {ordinal} exceeds u32 capacity"
        ))
    })
}

fn checked_v2_3_output_fragment_rows(physical_rows: usize) -> Result<u32> {
    u32::try_from(physical_rows).map_err(|_| {
        Error::format_capacity_exceeded(format!(
            "storage-version-2.3 output fragment row count {physical_rows} exceeds u32 capacity"
        ))
    })
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
    use std::collections::HashMap;

    use arrow_array::{ArrayRef, BinaryArray, Int32Array, RecordBatchReader, StructArray};
    use arrow_schema::{ArrowError, DataType, Field, Schema};
    use futures::TryStreamExt;
    use lance_arrow::BLOB_META_KEY;
    use lance_core::error::FormatCapacityExceeded;
    use lance_io::utils::tracking_store::{IOTracker, IoOperation};

    use crate::session::Session;

    use super::*;

    fn assert_format_capacity(error: &Error) {
        let Error::InvalidInput { source, .. } = error else {
            panic!("expected InvalidInput with FormatCapacityExceeded, got {error:?}")
        };
        assert!(
            source.downcast_ref::<FormatCapacityExceeded>().is_some(),
            "expected FormatCapacityExceeded, got {source:?}"
        );
    }

    fn file_count(path: &std::path::Path) -> usize {
        std::fs::read_dir(path)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| {
                if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
                    file_count(&entry.path())
                } else {
                    1
                }
            })
            .sum()
    }

    async fn data_object_count(dataset: &Dataset) -> usize {
        dataset
            .object_store
            .read_dir_all(&dataset.data_dir(), None)
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
            .len()
    }

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

    #[tokio::test]
    async fn v2_3_ignores_the_legacy_stable_row_id_toggle() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let dataset = InsertBuilder::new("memory://v23-legacy-row-id-toggle")
            .with_params(&WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_3),
                enable_stable_row_ids: true,
                ..Default::default()
            })
            .execute_stream(RecordBatchIterator::new(vec![Ok(batch)], schema))
            .await
            .unwrap();

        assert!(dataset.manifest.has_stable_row_identity());
        assert!(dataset.manifest.uses_stable_logical_row_addresses());
        assert!(
            dataset
                .manifest
                .fragments
                .iter()
                .all(|fragment| fragment.row_id_meta.is_none())
        );
    }

    #[tokio::test]
    async fn v2_3_append_rejects_exhausted_allocators_before_data_writes() {
        let test_dir = lance_core::utils::tempfile::TempStrDir::default();
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1]))])
            .unwrap();
        let mut dataset = InsertBuilder::new(test_dir.as_str())
            .with_params(&WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            })
            .execute_stream(RecordBatchIterator::new(
                vec![Ok(batch.clone())],
                schema.clone(),
            ))
            .await
            .unwrap();
        let manifest = Arc::make_mut(&mut dataset.manifest);
        manifest.max_fragment_id = Some(u32::MAX - 1);
        manifest.max_logical_fragment_id = Some(u32::MAX - 1);

        let before = file_count(std::path::Path::new(test_dir.as_str()));
        let params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };
        let error = InsertBuilder::new(Arc::new(dataset))
            .with_params(&params)
            .execute_stream(RecordBatchIterator::new(vec![Ok(batch)], schema))
            .await
            .unwrap_err();
        assert_format_capacity(&error);
        assert_eq!(
            file_count(std::path::Path::new(test_dir.as_str())),
            before,
            "allocator exhaustion must be rejected before speculative data writes"
        );
    }

    #[tokio::test]
    async fn v2_3_append_bounds_speculation_at_remaining_allocator_capacity() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let initial =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1]))])
                .unwrap();
        let mut dataset = InsertBuilder::new("memory://v23-bounded-speculation")
            .with_params(&WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            })
            .execute_stream(RecordBatchIterator::new(vec![Ok(initial)], schema.clone()))
            .await
            .unwrap();
        let manifest = Arc::make_mut(&mut dataset.manifest);
        manifest.max_fragment_id = Some(u32::MAX - 2);
        manifest.max_logical_fragment_id = Some(u32::MAX - 2);

        let tracker = Arc::new(IOTracker::default());
        dataset = dataset.with_object_store_wrappers([
            tracker.clone() as Arc<dyn lance_io::object_store::WrappingObjectStore>
        ]);
        let original_version = dataset.version().version;
        let dataset = Arc::new(dataset);
        let data_objects_before = data_object_count(dataset.as_ref()).await;
        tracker.incremental_stats();

        let appended =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![2, 3]))])
                .unwrap();
        let error = InsertBuilder::new(dataset.clone())
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                max_rows_per_file: 1,
                ..Default::default()
            })
            .execute_stream(RecordBatchIterator::new(vec![Ok(appended)], schema))
            .await
            .unwrap_err();
        assert_format_capacity(&error);

        let requests = tracker.incremental_stats().requests;
        assert!(
            requests.iter().any(|request| {
                request.operation == IoOperation::Put && request.path.to_string().contains("data/")
            }),
            "the first fragment is allowed to be written speculatively: {requests:?}"
        );
        assert!(
            requests
                .iter()
                .any(|request| request.operation == IoOperation::Delete),
            "the bounded speculative fragment must trigger cleanup: {requests:?}"
        );
        assert_eq!(
            data_object_count(dataset.as_ref()).await,
            data_objects_before
        );
        assert!(
            requests.iter().all(|request| {
                request.operation != IoOperation::Put
                    || (!request.path.to_string().contains("_transactions/")
                        && !request.path.to_string().contains("_versions/"))
            }),
            "allocator rejection must precede transaction and manifest publication: {requests:?}"
        );
        let mut latest = dataset.as_ref().clone();
        latest.checkout_latest().await.unwrap();
        assert_eq!(latest.version().version, original_version);
    }

    #[tokio::test]
    async fn rejects_zero_max_rows_before_data_writes_for_all_v2_versions() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let initial =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1]))])
                .unwrap();
        for (storage_version, uri) in [
            (LanceFileVersion::V2_2, "memory://v22-zero-max-rows"),
            (LanceFileVersion::V2_3, "memory://v23-zero-max-rows"),
        ] {
            let mut dataset = InsertBuilder::new(uri)
                .with_params(&WriteParams {
                    data_storage_version: Some(storage_version),
                    ..Default::default()
                })
                .execute_stream(RecordBatchIterator::new(
                    vec![Ok(initial.clone())],
                    schema.clone(),
                ))
                .await
                .unwrap();
            let tracker = Arc::new(IOTracker::default());
            dataset = dataset.with_object_store_wrappers([
                tracker.clone() as Arc<dyn lance_io::object_store::WrappingObjectStore>
            ]);
            tracker.incremental_stats();

            let error = InsertBuilder::new(Arc::new(dataset))
                .with_params(&WriteParams {
                    mode: WriteMode::Append,
                    max_rows_per_file: 0,
                    ..Default::default()
                })
                .execute_stream(RecordBatchIterator::new(
                    vec![Ok(initial.clone())],
                    schema.clone(),
                ))
                .await
                .unwrap_err();
            assert!(matches!(&error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("max_rows_per_file"));
            assert!(
                tracker.incremental_stats().requests.iter().all(|request| {
                    request.operation != IoOperation::Put
                        || !request.path.to_string().starts_with("data/")
                }),
                "zero max_rows_per_file must fail before data PUT for {storage_version}"
            );
        }
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn v2_3_output_shape_overflow_is_typed() {
        let overflow = u32::MAX as usize + 1;
        assert_format_capacity(&checked_v2_3_output_fragment_ordinal(overflow).unwrap_err());
        assert_format_capacity(&checked_v2_3_output_fragment_rows(overflow).unwrap_err());
    }

    #[tokio::test]
    async fn allow_overwrite_to_v2_2_without_blob_upgrade() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1]))])
            .unwrap();

        let dataset = InsertBuilder::new("memory://blob-version-guard")
            .execute_stream(RecordBatchIterator::new(
                vec![Ok(batch.clone())],
                schema.clone(),
            ))
            .await
            .unwrap();

        let dataset = Arc::new(dataset);
        let params = WriteParams {
            mode: WriteMode::Overwrite,
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        };

        let result = InsertBuilder::new(dataset.clone())
            .with_params(&params)
            .execute_stream(RecordBatchIterator::new(vec![Ok(batch)], schema.clone()))
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn create_v2_2_dataset_rejects_legacy_blob_schema() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("blob", DataType::Binary, false).with_metadata(HashMap::from([(
                BLOB_META_KEY.to_string(),
                "true".to_string(),
            )])),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(BinaryArray::from(vec![Some(b"abc".as_slice())]))],
        )
        .unwrap();

        let dataset = InsertBuilder::new("memory://forced-blob-v2")
            .with_params(&WriteParams {
                mode: WriteMode::Create,
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            })
            .execute_stream(RecordBatchIterator::new(vec![Ok(batch)], schema.clone()))
            .await;

        let err = dataset.unwrap_err();
        match err {
            Error::InvalidInput { source, .. } => {
                let message = source.to_string();
                assert!(message.contains("Legacy blob columns"));
                assert!(message.contains("lance.blob.v2"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn create_v2_2_dataset_rejects_nested_legacy_blob_schema() {
        let image_field = Field::new("image_bytes", DataType::Binary, true).with_metadata(
            HashMap::from([(BLOB_META_KEY.to_string(), "true".to_string())]),
        );
        let schema = Arc::new(Schema::new(vec![Field::new(
            "summary_image_nested",
            DataType::Struct(vec![image_field.clone()].into()),
            true,
        )]));
        let image_values: ArrayRef = Arc::new(BinaryArray::from(vec![Some(b"abc".as_slice())]));
        let nested_values = StructArray::from(vec![(Arc::new(image_field), image_values)]);
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(nested_values)]).unwrap();

        let dataset = InsertBuilder::new("memory://forced-nested-blob-v2")
            .with_params(&WriteParams {
                mode: WriteMode::Create,
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            })
            .execute_stream(RecordBatchIterator::new(vec![Ok(batch)], schema.clone()))
            .await;

        let err = dataset.unwrap_err();
        match err {
            Error::InvalidInput { source, .. } => {
                let message = source.to_string();
                assert!(message.contains("Legacy blob columns"));
                assert!(message.contains("summary_image_nested.image_bytes"));
                assert!(message.contains("lance.blob.v2"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    mod external_error {
        use super::*;
        use std::fmt;

        #[derive(Debug)]
        struct MyTestError {
            code: i32,
            details: String,
        }

        impl fmt::Display for MyTestError {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "MyTestError({}): {}", self.code, self.details)
            }
        }

        impl std::error::Error for MyTestError {}

        fn create_failing_iterator(
            schema: Arc<Schema>,
            fail_at_batch: usize,
            error_code: i32,
        ) -> impl Iterator<Item = std::result::Result<RecordBatch, ArrowError>> {
            let mut batch_count = 0;
            std::iter::from_fn(move || {
                if batch_count >= 5 {
                    return None;
                }
                batch_count += 1;
                if batch_count == fail_at_batch {
                    Some(Err(ArrowError::ExternalError(Box::new(MyTestError {
                        code: error_code,
                        details: format!("Failed at batch {}", batch_count),
                    }))))
                } else {
                    let batch = RecordBatch::try_new(
                        schema.clone(),
                        vec![Arc::new(Int32Array::from(vec![batch_count as i32; 10]))],
                    )
                    .unwrap();
                    Some(Ok(batch))
                }
            })
        }

        #[tokio::test]
        async fn test_insert_builder_preserves_external_error() {
            let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

            let error_code = 42;
            let iter = create_failing_iterator(schema.clone(), 3, error_code);
            let reader = RecordBatchIterator::new(iter, schema);

            let result = InsertBuilder::new("memory://test_external_error")
                .execute_stream(Box::new(reader) as Box<dyn RecordBatchReader + Send>)
                .await;

            match result {
                Err(Error::External { source }) => {
                    let original = source
                        .downcast_ref::<MyTestError>()
                        .expect("Should be able to downcast to MyTestError");
                    assert_eq!(original.code, error_code);
                    assert!(original.details.contains("batch 3"));
                }
                Err(other) => panic!("Expected Error::External variant, got: {:?}", other),
                Ok(_) => panic!("Expected error, got success"),
            }
        }

        #[tokio::test]
        async fn test_insert_builder_first_batch_error() {
            let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

            let error_code = 999;
            let iter = std::iter::once(Err(ArrowError::ExternalError(Box::new(MyTestError {
                code: error_code,
                details: "immediate failure".to_string(),
            }))));
            let reader = RecordBatchIterator::new(iter, schema);

            let result = InsertBuilder::new("memory://test_first_batch_error")
                .execute_stream(Box::new(reader) as Box<dyn RecordBatchReader + Send>)
                .await;

            match result {
                Err(Error::External { source }) => {
                    let original = source.downcast_ref::<MyTestError>().unwrap();
                    assert_eq!(original.code, error_code);
                }
                Err(other) => panic!("Expected External, got: {:?}", other),
                Ok(_) => panic!("Expected error"),
            }
        }
    }

    #[tokio::test]
    async fn test_write_progress_callback() {
        use std::sync::Mutex;
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        // Three batches of 100 rows each.
        let batches: Vec<_> = (0..3)
            .map(|_| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(Int32Array::from(vec![0i32; 100]))],
                )
                .unwrap()
            })
            .collect();

        let stats_log: Arc<Mutex<Vec<crate::dataset::WriteStats>>> =
            Arc::new(Mutex::new(Vec::new()));
        let log_clone = stats_log.clone();

        InsertBuilder::new("memory://test_write_progress")
            .progress(move |stats| {
                log_clone.lock().unwrap().push(stats);
            })
            .execute_stream(RecordBatchIterator::new(
                batches.into_iter().map(Ok),
                schema,
            ))
            .await
            .unwrap();

        let log = stats_log.lock().unwrap();
        assert!(
            !log.is_empty(),
            "progress callback must be called at least once"
        );
        // bytes_written and rows_written must be monotonically non-decreasing.
        for window in log.windows(2) {
            assert!(
                window[1].bytes_written >= window[0].bytes_written,
                "bytes_written must not decrease: {:?} -> {:?}",
                window[0].bytes_written,
                window[1].bytes_written,
            );
            assert!(
                window[1].rows_written >= window[0].rows_written,
                "rows_written must not decrease",
            );
        }
        let last = log.last().unwrap();
        assert!(last.bytes_written > 0, "final bytes_written must be > 0");
        assert_eq!(last.rows_written, 300, "all 300 rows must be reported");
        assert_eq!(last.files_written, 1, "a single file should be written");
    }
}
