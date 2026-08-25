// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use lance_file::version::{ConcreteFileVersion, LanceFileVersion};
use lance_io::object_store::{ObjectStore, ObjectStoreParams};
use lance_select::RowAddrTreeMap;
use lance_table::{
    format::{DataStorageFormat, is_detached_version},
    io::commit::{CommitConfig, CommitHandler, ManifestNamingScheme},
};

use crate::io::commit::DEFAULT_COMMIT_RETRY_TIMEOUT;
use crate::{
    Dataset, Error, Result,
    dataset::{
        ManifestWriteConfig, ReadParams,
        builder::DatasetBuilder,
        commit_detached_transaction, commit_new_dataset, commit_transaction,
        refs::Refs,
        transaction::{Operation, Transaction},
    },
    session::Session,
};

use super::{WriteDestination, resolve_commit_handler};
use crate::dataset::branch_location::BranchLocation;
use crate::dataset::transaction::validate_operation;
use lance_core::utils::tracing::{DATASET_COMMITTED_EVENT, TRACE_DATASET_EVENTS};
use tracing::info;

/// Create a new commit from a [`Transaction`].
///
/// Transactions can be created using a write method like [`super::InsertBuilder::execute_uncommitted`].
#[derive(Debug, Clone)]
pub struct CommitBuilder<'a> {
    dest: WriteDestination<'a>,
    use_stable_row_ids: Option<bool>,
    enable_v2_manifest_paths: bool,
    storage_format: Option<ConcreteFileVersion>,
    commit_handler: Option<Arc<dyn CommitHandler>>,
    store_params: Option<ObjectStoreParams>,
    object_store: Option<Arc<ObjectStore>>,
    source_store: Option<Arc<ObjectStore>>,
    session: Option<Arc<Session>>,
    detached: bool,
    commit_config: CommitConfig,
    retry_timeout: Duration,
    affected_rows: Option<RowAddrTreeMap>,
    transaction_properties: Option<Arc<HashMap<String, String>>>,
    timeout: Option<Duration>,
    /// When `Some`, this commit is the second step of `migrate_to_stable_row_ids`.
    migration_next_row_id: Option<u64>,
}

/// Default timeout applied to [`CommitBuilder::execute`] when none is set.
pub const DEFAULT_COMMIT_TIMEOUT: Duration = Duration::from_secs(1800);

impl<'a> CommitBuilder<'a> {
    pub fn new(dest: impl Into<WriteDestination<'a>>) -> Self {
        Self {
            dest: dest.into(),
            use_stable_row_ids: None,
            enable_v2_manifest_paths: true,
            storage_format: None,
            commit_handler: None,
            store_params: None,
            object_store: None,
            source_store: None,
            session: None,
            detached: false,
            commit_config: Default::default(),
            retry_timeout: DEFAULT_COMMIT_RETRY_TIMEOUT,
            affected_rows: None,
            transaction_properties: None,
            timeout: Some(DEFAULT_COMMIT_TIMEOUT),
            migration_next_row_id: None,
        }
    }

    /// Whether to use stable row ids. This makes the `_rowid` column stable
    /// after compaction, but not updates.
    ///
    /// This is only used for new datasets. Existing datasets will use their
    /// existing setting.
    ///
    /// **Default is false.**
    pub fn use_stable_row_ids(mut self, use_stable_row_ids: bool) -> Self {
        self.use_stable_row_ids = Some(use_stable_row_ids);
        self
    }

    /// Pass the storage format to use for the dataset.
    ///
    /// This is only needed when creating a new empty table. If any data files are
    /// passed, the storage format will be inferred from the data files.
    ///
    /// All data files must use the same storage format as the existing dataset.
    /// If a different format is passed, an error will be returned.
    pub fn with_storage_format(mut self, storage_format: LanceFileVersion) -> Self {
        self.storage_format = Some(storage_format.resolve());

        self
    }

    pub(crate) fn with_exact_storage_format(mut self, storage_format: ConcreteFileVersion) -> Self {
        self.storage_format = Some(storage_format);
        self
    }

    /// Pass an object store to use.
    pub fn with_object_store(mut self, object_store: Arc<ObjectStore>) -> Self {
        self.object_store = Some(object_store);
        self
    }

    /// Pass the object store of the dataset being cloned from.
    ///
    /// Only used by `Operation::Clone`: the source manifest is read through this store
    /// while the new dataset is written through the destination store. This lets a clone
    /// cross object stores/accounts (e.g. between two Azure accounts), where the source
    /// is not reachable with the destination's credentials. Defaults to the destination
    /// store when not set, preserving same-store behavior.
    pub fn with_source_store(mut self, source_store: Arc<ObjectStore>) -> Self {
        self.source_store = Some(source_store);
        self
    }

    /// Pass a commit handler to use for the dataset.
    ///
    /// Takes precedence over the destination dataset's own handler. If not
    /// set, a `Dataset` destination commits through its own handler and a
    /// `Uri` destination resolves one from the uri.
    pub fn with_commit_handler(mut self, commit_handler: Arc<dyn CommitHandler>) -> Self {
        self.commit_handler = Some(commit_handler);
        self
    }

    /// Pass store parameters to use for the dataset.
    ///
    /// If an object store is passed, these parameters will be ignored.
    pub fn with_store_params(mut self, store_params: ObjectStoreParams) -> Self {
        self.store_params = Some(store_params);
        self
    }

    /// Pass a session to use for the dataset.
    ///
    /// If a session is not passed, but a dataset is used as the destination,
    /// then the dataset's session will be used.
    ///
    /// By passing a session or re-using a dataset, you can re-use the
    /// file metadata and index caches, which can significantly improve
    /// performance.
    pub fn with_session(mut self, session: Arc<Session>) -> Self {
        self.session = Some(session);
        self
    }

    ///  If set to true, and this is a new dataset, uses the new v2 manifest
    ///  paths. These allow constant-time lookups for the latest manifest on object storage.
    ///  This parameter has no effect on existing datasets. To migrate an existing
    ///  dataset, use the [`Dataset::migrate_manifest_paths_v2`] method. **Default is True.**
    ///
    /// <div class="warning">
    ///  WARNING: turning this on will make the dataset unreadable for older
    ///  versions of Lance (prior to 0.17.0).
    /// </div>
    pub fn enable_v2_manifest_paths(mut self, enable: bool) -> Self {
        self.enable_v2_manifest_paths = enable;
        self
    }

    /// Commit a version that is not part of the mainline history.
    ///
    /// This commit will never show up in the dataset's history.
    ///
    /// This can be used to stage changes or to handle "secondary" datasets
    /// whose lineage is tracked elsewhere.
    pub fn with_detached(mut self, detached: bool) -> Self {
        self.detached = detached;
        self
    }

    /// Set the maximum number of retries for commit operations.
    ///
    /// If a commit operation fails, it will be retried up to `max_retries` times.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.commit_config.num_retries = max_retries;
        self
    }

    /// Set the wall-clock budget used by commit conflict backoff.
    ///
    /// The first commit attempt is always allowed to complete. If it conflicts,
    /// each backoff sleep is bounded by the time remaining in this budget. The
    /// default is 30 seconds.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    /// use lance::dataset::CommitBuilder;
    ///
    /// let _builder = CommitBuilder::new("memory://dataset")
    ///     .with_retry_timeout(Duration::from_secs(10));
    /// ```
    pub fn with_retry_timeout(mut self, retry_timeout: Duration) -> Self {
        self.retry_timeout = retry_timeout;
        self
    }

    /// Require the latest manifest to carry `value` under schema-metadata
    /// `key`, and make that part of the commit: on every attempt the latest
    /// manifest is judged, and publication is conditioned atomically on that
    /// same manifest still being the predecessor (`CommitHandler::commit_after`),
    /// so a dataset recreated at the same path at any point is refused with
    /// [`Error::PrerequisiteFailed`]. Only commit handlers whose store can
    /// make that decision atomically accept the option (the external
    /// manifest store); others fail with [`Error::NotSupported`]. Not
    /// available on detached commits.
    pub fn with_required_schema_metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.commit_config
            .required_schema_metadata
            .insert(key.into(), value.into());
        self
    }

    pub fn with_skip_auto_cleanup(mut self, skip_auto_cleanup: bool) -> Self {
        self.commit_config.skip_auto_cleanup = skip_auto_cleanup;
        self
    }

    /// Provide the set of row addresses that were deleted or updated. This is
    /// used to perform fast conflict resolution.
    pub fn with_affected_rows(mut self, affected_rows: RowAddrTreeMap) -> Self {
        self.affected_rows = Some(affected_rows);
        self
    }

    /// Set a timeout for the commit operation.
    ///
    /// The timeout bounds the *entire* [`Self::execute`] / [`Self::execute_batch`]
    /// call, including all conflict retries — it is not applied per attempt.
    /// Pass `None` to disable the timeout entirely.
    ///
    /// The default is 30 minutes (see [`DEFAULT_COMMIT_TIMEOUT`]).
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidInput`] if `timeout` is `Some(Duration::ZERO)` (raised
    ///   when [`Self::execute`] is called, not here).
    /// - [`Error::Timeout`] if the operation does not complete within the
    ///   timeout.
    pub fn with_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
        self
    }

    /// provide Configuration key-value pairs associated with this transaction.
    /// This is used to store metadata about the transaction, such as commit messages, engine information, etc.
    /// this properties map will be persisted as a part of the transaction object
    pub fn with_transaction_properties(
        mut self,
        transaction_properties: HashMap<String, String>,
    ) -> Self {
        self.transaction_properties = Some(Arc::new(transaction_properties));
        self
    }

    /// Configure this commit as the second step of a stable row ID migration.
    ///
    /// Sets `use_stable_row_ids = true` and supplies the `next_row_id` that was
    /// computed during the first migration commit. This bypasses the normal
    /// "cannot enable stable row IDs on an existing dataset" check so that the
    /// flag can be activated without creating the dataset from scratch.
    pub(crate) fn with_stable_row_id_migration_activation(mut self, next_row_id: u64) -> Self {
        self.migration_next_row_id = Some(next_row_id);
        self
    }

    pub async fn execute(self, transaction: Transaction) -> Result<Dataset> {
        let timeout = self.timeout;
        if let Some(t) = timeout
            && t.is_zero()
        {
            return Err(Error::invalid_input(
                "CommitBuilder timeout must be non-zero; pass `None` to disable",
            ));
        }
        // Box the inner future so wrapping it in `tokio::time::Timeout` does
        // not deepen the future type — downstream `async fn`s that await
        // `execute` otherwise hit the compiler's layout-query depth limit.
        let fut = Box::pin(self.execute_inner(transaction));
        match timeout {
            Some(t) => match tokio::time::timeout(t, fut).await {
                Ok(res) => res,
                Err(_) => Err(Error::timeout(format!(
                    "Commit timed out after {:?}. Increase the timeout via \
                     CommitBuilder::with_timeout or pass `None` to disable.",
                    t
                ))),
            },
            None => fut.await,
        }
    }

    async fn execute_inner(self, transaction: Transaction) -> Result<Dataset> {
        let session = self
            .session
            .or_else(|| self.dest.dataset().map(|ds| ds.session.clone()))
            .unwrap_or_default();

        // Store used to read the source manifest for a clone (see with_source_store).
        let source_store = self.source_store.clone();

        let (object_store, base_path, commit_handler) = match &self.dest {
            WriteDestination::Dataset(dataset) => (
                dataset.object_store.clone(),
                dataset.base.clone(),
                self.commit_handler
                    .clone()
                    .unwrap_or_else(|| dataset.commit_handler.clone()),
            ),
            WriteDestination::Uri(uri) => {
                let commit_handler = if let (Some(_), Some(commit_handler)) =
                    (&self.object_store, &self.commit_handler)
                {
                    commit_handler.clone()
                } else {
                    resolve_commit_handler(uri, self.commit_handler.clone(), &self.store_params)
                        .await?
                };
                let (object_store, base_path) = if let Some(passed_store) = self.object_store {
                    (
                        passed_store,
                        ObjectStore::extract_path_from_uri(session.store_registry(), uri)?,
                    )
                } else {
                    ObjectStore::from_uri_and_params(
                        session.store_registry(),
                        uri,
                        &self.store_params.clone().unwrap_or_default(),
                    )
                    .await?
                };
                (object_store, base_path, commit_handler)
            }
        };

        let dest = match &self.dest {
            WriteDestination::Dataset(dataset) => WriteDestination::Dataset(dataset.clone()),
            WriteDestination::Uri(uri) => {
                // Check if it already exists.
                let mut builder = DatasetBuilder::from_uri(uri)
                    .with_read_params(ReadParams {
                        store_options: self.store_params.clone(),
                        commit_handler: self.commit_handler.clone(),
                        ..Default::default()
                    })
                    .with_session(session.clone());

                // If we are using a detached version, we need to load the dataset.
                // Otherwise, we are writing to the main history, and need to check
                // out the latest version.
                if is_detached_version(transaction.read_version) {
                    builder = builder.with_version(transaction.read_version)
                }

                match builder.load().await {
                    Ok(dataset) => WriteDestination::Dataset(Arc::new(dataset)),
                    Err(Error::DatasetNotFound { .. } | Error::NotFound { .. }) => {
                        WriteDestination::Uri(uri)
                    }
                    Err(e) => return Err(e),
                }
            }
        };

        if dest.dataset().is_none()
            && !matches!(
                transaction.operation,
                Operation::Overwrite { .. } | Operation::Clone { .. }
            )
        {
            return Err(Error::dataset_not_found(
                base_path.to_string(),
                "The dataset must already exist unless the operation is Overwrite".into(),
            ));
        }

        // Validate the operation before proceeding with the commit
        // This ensures that operations like Merge have proper validation for data integrity
        if let Some(dataset) = dest.dataset() {
            validate_operation(Some(&dataset.manifest), &transaction.operation)?;
        } else {
            validate_operation(None, &transaction.operation)?;
        }

        let (metadata_cache, index_cache) = match &dest {
            WriteDestination::Dataset(ds) => (ds.metadata_cache.clone(), ds.index_cache.clone()),
            WriteDestination::Uri(uri) => (
                Arc::new(session.metadata_cache.for_dataset(uri)),
                Arc::new(session.index_cache.for_dataset(uri)),
            ),
        };

        let manifest_naming_scheme = if let Some(ds) = dest.dataset() {
            ds.manifest_location.naming_scheme
        } else if self.enable_v2_manifest_paths {
            ManifestNamingScheme::V2
        } else {
            ManifestNamingScheme::V1
        };

        let use_stable_row_ids = if self.migration_next_row_id.is_some() {
            // Migration activation always enables stable row IDs regardless of
            // the current dataset state.
            true
        } else if let Some(ds) = dest.dataset() {
            ds.manifest.uses_stable_row_ids()
        } else {
            self.use_stable_row_ids.unwrap_or(false)
        };
        // Validate storage format matches existing dataset
        if let Some(ds) = dest.dataset()
            && let Some(storage_format) = self.storage_format
        {
            let passed_storage_format = DataStorageFormat::new(storage_format);
            if ds.manifest.data_storage_format != passed_storage_format
                && !matches!(transaction.operation, Operation::Overwrite { .. })
            {
                return Err(Error::invalid_input_source(format!(
                    "Storage format mismatch. Existing dataset uses {:?}, but new data uses {:?}",
                    ds.manifest.data_storage_format,
                    passed_storage_format
                ).into()));
            }
        }

        let manifest_config = ManifestWriteConfig {
            use_stable_row_ids,
            storage_format: self.storage_format.map(DataStorageFormat::new),
            migration_next_row_id: self.migration_next_row_id,
            ..Default::default()
        };

        if !self.commit_config.required_schema_metadata.is_empty() {
            if self.detached {
                return Err(Error::invalid_input(
                    "required schema metadata cannot be enforced on a detached commit",
                ));
            }
            // Creation has no predecessor to judge, so the option would be
            // meaningless there rather than enforced.
            if dest.dataset().is_none() {
                return Err(Error::invalid_input(
                    "required schema metadata cannot apply to creating a dataset",
                ));
            }
        }
        let (manifest, manifest_location) = if let Some(dataset) = dest.dataset() {
            if self.detached {
                if matches!(manifest_naming_scheme, ManifestNamingScheme::V1) {
                    return Err(Error::not_supported_source(
                        "detached commits cannot be used with v1 manifest paths".into(),
                    ));
                }
                commit_detached_transaction(
                    dataset,
                    object_store.as_ref(),
                    commit_handler.as_ref(),
                    &transaction,
                    &manifest_config,
                    &self.commit_config,
                    self.retry_timeout,
                )
                .await?
            } else {
                commit_transaction(
                    dataset,
                    object_store.as_ref(),
                    commit_handler.as_ref(),
                    &transaction,
                    &manifest_config,
                    &self.commit_config,
                    self.retry_timeout,
                    manifest_naming_scheme,
                    self.affected_rows.as_ref(),
                )
                .await?
            }
        } else if self.detached {
            // I think we may eventually want this, and we can probably handle it, but leaving a TODO for now
            return Err(Error::not_supported_source(
                "detached commits cannot currently be used to create new datasets".into(),
            ));
        } else {
            commit_new_dataset(
                object_store.as_ref(),
                source_store.as_deref(),
                commit_handler.as_ref(),
                &base_path,
                &transaction,
                &manifest_config,
                manifest_naming_scheme,
                metadata_cache.as_ref(),
                session.store_registry(),
            )
            .await?
        };

        info!(
            target: TRACE_DATASET_EVENTS,
            event=DATASET_COMMITTED_EVENT,
            uri=dest.uri(),
            read_version=transaction.read_version,
            committed_version=manifest.version,
            detached=self.detached,
            operation=&transaction.operation.name()
        );

        let fragment_bitmap = Arc::new(manifest.fragments.iter().map(|f| f.id as u32).collect());

        match &self.dest {
            WriteDestination::Dataset(dataset) => {
                let base_object_stores = if manifest.base_paths == dataset.manifest.base_paths {
                    dataset.base_object_stores.clone()
                } else {
                    Default::default()
                };
                Ok(Dataset {
                    manifest: Arc::new(manifest),
                    manifest_location,
                    session,
                    fragment_bitmap,
                    base_object_stores,
                    ..dataset.as_ref().clone()
                })
            }
            WriteDestination::Uri(uri) => {
                let refs = Refs::new(
                    object_store.clone(),
                    commit_handler.clone(),
                    BranchLocation {
                        path: base_path.clone(),
                        uri: uri.to_string(),
                        branch: manifest.branch.clone(),
                    },
                );

                Ok(Dataset {
                    object_store,
                    base: base_path,
                    uri: uri.to_string(),
                    manifest: Arc::new(manifest),
                    manifest_location,
                    session,
                    commit_handler,
                    refs,
                    index_cache,
                    fragment_bitmap,
                    metadata_cache,
                    file_reader_options: None,
                    store_params: self.store_params.clone().map(Box::new),
                    base_store_params: None,
                    base_object_stores: Default::default(),
                })
            }
        }
    }

    /// Commit a set of transactions as a single new version.
    ///
    /// <div class="warning">
    ///   Only works for append transactions right now. Other kinds of transactions
    ///   will be supported in the future.
    /// </div>
    pub async fn execute_batch(self, transactions: Vec<Transaction>) -> Result<BatchCommitResult> {
        if transactions.is_empty() {
            return Err(Error::invalid_input_source(
                "No transactions to commit".into(),
            ));
        }
        if transactions
            .iter()
            .any(|t| !matches!(t.operation, Operation::Append { .. }))
        {
            return Err(Error::not_supported_source(
                "Only append transactions are supported in batch commits".into(),
            ));
        }

        let read_version = transactions.iter().map(|t| t.read_version).min().unwrap();

        let merged = Transaction {
            uuid: uuid::Uuid::new_v4().hyphenated().to_string(),
            operation: Operation::Append {
                fragments: transactions
                    .iter()
                    .flat_map(|t| match &t.operation {
                        Operation::Append { fragments } => fragments.clone(),
                        _ => unreachable!(),
                    })
                    .collect(),
            },
            read_version,
            tag: None,
            transaction_properties: None,
        };
        let dataset = self.execute(merged.clone()).await?;
        Ok(BatchCommitResult { dataset, merged })
    }
}

pub struct BatchCommitResult {
    pub dataset: Dataset,
    /// The final transaction that was committed.
    pub merged: Transaction,
    // TODO: Reject conflicts that need to be retried.
    // /// Transactions that were rejected due to conflicts.
    // pub rejected: Vec<Transaction>,
}

#[cfg(test)]
mod tests {
    use arrow::array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};

    use lance_io::utils::CachedFileSize;
    use lance_io::{assert_io_eq, assert_io_gt};
    use lance_table::format::{
        DataFile, Fragment, IndexMetadata, Manifest, Transaction as TableTransaction,
    };
    use lance_table::io::commit::{CommitError, ManifestLocation, ManifestWriter};
    use std::time::Duration;

    use object_store::path::Path;
    use object_store::throttle::ThrottleConfig;

    use crate::utils::test::ThrottledStoreWrapper;

    use crate::dataset::{InsertBuilder, WriteMode, WriteParams};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_table::io::commit::external_manifest::{
        ExternalManifestCommitHandler, ExternalManifestStore, Reservation,
    };
    use lance_table::io::commit::{CANDIDATES_DIR, PredecessorIdentity};

    use super::*;

    fn sample_fragment() -> Fragment {
        let (major_version, minor_version) =
            LanceFileVersion::Stable.resolve().to_data_file_numbers();

        Fragment {
            id: 0,
            files: vec![DataFile {
                path: "file.lance".to_string(),
                fields: Arc::from([0]),
                column_indices: Arc::from([0]),
                file_major_version: major_version,
                file_minor_version: minor_version,
                file_size_bytes: CachedFileSize::new(100),
                base_id: None,
            }],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: None,
            physical_rows: Some(10),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        }
    }

    fn sample_transaction(read_version: u64) -> Transaction {
        Transaction {
            uuid: uuid::Uuid::new_v4().hyphenated().to_string(),
            operation: Operation::Append {
                fragments: vec![sample_fragment()],
            },
            read_version,
            tag: None,
            transaction_properties: None,
        }
    }

    #[derive(Debug)]
    struct SlowConflictingCommitHandler;

    #[async_trait::async_trait]
    impl CommitHandler for SlowConflictingCommitHandler {
        fn is_version_not_found_definitive(&self) -> bool {
            true
        }

        async fn commit(
            &self,
            _manifest: &mut Manifest,
            _indices: Option<Vec<IndexMetadata>>,
            _base_path: &object_store::path::Path,
            _object_store: &ObjectStore,
            _manifest_writer: ManifestWriter,
            _naming_scheme: ManifestNamingScheme,
            _transaction: Option<TableTransaction>,
        ) -> std::result::Result<ManifestLocation, CommitError> {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Err(CommitError::CommitConflict)
        }
    }

    /// An external manifest store that identifies each row and can reserve a
    /// version conditioned on its predecessor. `hold_next_reservation` parks
    /// the next conditioned reservation so a test can move the dataset
    /// underneath it.
    /// `(path, size, identity)` per `(base_uri, version)`.
    type StoredRow = (String, u64, String);
    type StoredRows = HashMap<(String, u64), StoredRow>;

    #[derive(Debug, Default)]
    struct IdentifiedStore {
        rows: std::sync::Mutex<StoredRows>,
        generation: std::sync::Mutex<Option<String>>,
        next_identity: std::sync::atomic::AtomicU64,
        hold_next_reservation: std::sync::atomic::AtomicBool,
        reservation_held: tokio::sync::Notify,
        release_reservation: tokio::sync::Notify,
        /// Apply the next reservation but report an error for it.
        lose_next_reservation_response: std::sync::atomic::AtomicBool,
        /// Park the next reservation read before it observes the row.
        hold_next_reservation_read: std::sync::atomic::AtomicBool,
        reservation_read_held: tokio::sync::Notify,
        release_reservation_read: tokio::sync::Notify,
        /// Apply the next reservation but report a conflict for it, as a
        /// store whose internal retry saw its own write does.
        conflict_after_applying_next_reservation: std::sync::atomic::AtomicBool,
        /// Apply the next reservation but report `PrerequisiteFailed` for it.
        refuse_after_applying_next_reservation: std::sync::atomic::AtomicBool,
        /// Fail the n-th `forget_version` call (1-based; 0 = never) before
        /// it touches a row.
        fail_forget_call: std::sync::atomic::AtomicUsize,
        forget_calls: std::sync::atomic::AtomicUsize,
        /// Behave as a store that keeps no generation.
        hide_generation: std::sync::atomic::AtomicBool,
        /// Behave as a store that cannot enumerate its records.
        hide_records: std::sync::atomic::AtomicBool,
        /// The highest forgotten version per dataset; never reserved again.
        reuse_floor: std::sync::Mutex<HashMap<String, u64>>,
        /// Report an error for the next reservation without applying it,
        /// holding it for [`Self::apply_pending_reservation`].
        delay_next_reservation: std::sync::atomic::AtomicBool,
        pending_reservation: std::sync::Mutex<Option<((String, u64), StoredRow)>>,
    }

    impl IdentifiedStore {
        /// The remote write of a delayed reservation finally lands.
        fn apply_pending_reservation(&self) {
            let (key, row) = self.pending_reservation.lock().unwrap().take().unwrap();
            self.rows.lock().unwrap().insert(key, row);
        }

        fn mint(&self) -> String {
            format!(
                "identity-{}",
                self.next_identity
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            )
        }

        fn handler(self: &Arc<Self>) -> Arc<dyn CommitHandler> {
            Arc::new(ExternalManifestCommitHandler {
                external_manifest_store: self.clone(),
            })
        }

        fn below_floor(&self, base_uri: &str, version: u64) -> bool {
            self.reuse_floor
                .lock()
                .unwrap()
                .get(base_uri)
                .is_some_and(|floor| version <= *floor)
        }

        fn versions(&self) -> Vec<u64> {
            let mut versions: Vec<u64> = self.rows.lock().unwrap().keys().map(|k| k.1).collect();
            versions.sort();
            versions
        }
    }

    #[async_trait::async_trait]
    impl ExternalManifestStore for IdentifiedStore {
        async fn get(&self, base_uri: &str, version: u64) -> Result<String> {
            self.rows
                .lock()
                .unwrap()
                .get(&(base_uri.to_string(), version))
                .map(|row| row.0.clone())
                .ok_or_else(|| Error::not_found(format!("{base_uri}@{version}")))
        }

        async fn get_latest_version(&self, base_uri: &str) -> Result<Option<(u64, String)>> {
            Ok(self
                .rows
                .lock()
                .unwrap()
                .iter()
                .filter(|(key, _)| key.0 == base_uri)
                .max_by_key(|(key, _)| key.1)
                .map(|(key, row)| (key.1, row.0.clone())))
        }

        async fn put_if_not_exists(
            &self,
            base_uri: &str,
            version: u64,
            path: &str,
            size: u64,
            _e_tag: Option<String>,
        ) -> Result<()> {
            self.generation
                .lock()
                .unwrap()
                .get_or_insert_with(|| self.mint());
            let identity = self.mint();
            let mut rows = self.rows.lock().unwrap();
            let key = (base_uri.to_string(), version);
            if rows.contains_key(&key) {
                return Err(Error::commit_conflict_source(
                    version,
                    "manifest already exists".into(),
                ));
            }
            if self.below_floor(base_uri, version) {
                return Err(lance_core::error::PrerequisiteFailedSnafu {
                    message: format!("version {version} was forgotten and is never reused"),
                }
                .build());
            }
            rows.insert(key, (path.to_string(), size, identity));
            Ok(())
        }

        async fn put_if_exists(
            &self,
            base_uri: &str,
            version: u64,
            path: &str,
            size: u64,
            _e_tag: Option<String>,
        ) -> Result<()> {
            let mut rows = self.rows.lock().unwrap();
            let row = rows
                .get_mut(&(base_uri.to_string(), version))
                .ok_or_else(|| Error::not_found(format!("{base_uri}@{version}")))?;
            row.0 = path.to_string();
            row.1 = size;
            Ok(())
        }

        async fn delete(&self, base_uri: &str) -> Result<()> {
            *self.generation.lock().unwrap() = None;
            self.reuse_floor.lock().unwrap().remove(base_uri);
            self.rows.lock().unwrap().retain(|key, _| key.0 != base_uri);
            Ok(())
        }

        async fn reuse_floor(&self, base_uri: &str) -> Result<Option<u64>> {
            Ok(self.reuse_floor.lock().unwrap().get(base_uri).copied())
        }

        async fn generation(&self, _base_uri: &str) -> Result<Option<String>> {
            if self
                .hide_generation
                .load(std::sync::atomic::Ordering::SeqCst)
            {
                return Ok(None);
            }
            Ok(self.generation.lock().unwrap().clone())
        }

        async fn forget_version(
            &self,
            base_uri: &str,
            version: u64,
            path: &str,
            generation: Option<&str>,
        ) -> Result<()> {
            let call = self
                .forget_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                + 1;
            if self
                .fail_forget_call
                .compare_exchange(
                    call,
                    0,
                    std::sync::atomic::Ordering::SeqCst,
                    std::sync::atomic::Ordering::SeqCst,
                )
                .is_ok()
            {
                return Err(Error::io("simulated coordinator outage"));
            }
            if self.generation.lock().unwrap().as_deref() != generation {
                return Ok(());
            }
            let mut rows = self.rows.lock().unwrap();
            let key = (base_uri.to_string(), version);
            if rows.get(&key).is_some_and(|row| row.0 == path) {
                rows.remove(&key);
                let mut floors = self.reuse_floor.lock().unwrap();
                let floor = floors.entry(base_uri.to_string()).or_insert(version);
                *floor = (*floor).max(version);
            }
            Ok(())
        }

        fn supports_predecessor_condition(&self) -> bool {
            true
        }

        async fn get_identity(&self, base_uri: &str, version: u64) -> Result<Option<String>> {
            Ok(self
                .rows
                .lock()
                .unwrap()
                .get(&(base_uri.to_string(), version))
                .map(|row| row.2.clone()))
        }

        async fn list_versions(
            &self,
            base_uri: &str,
        ) -> Result<Option<Vec<(u64, String, Option<u64>)>>> {
            if self.hide_records.load(std::sync::atomic::Ordering::SeqCst) {
                return Ok(None);
            }
            Ok(Some(
                self.rows
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|(key, _)| key.0 == base_uri)
                    .map(|(key, row)| (key.1, row.0.clone(), Some(row.1)))
                    .collect(),
            ))
        }

        async fn get_reservation(
            &self,
            base_uri: &str,
            version: u64,
        ) -> Result<Option<(String, String)>> {
            if self
                .hold_next_reservation_read
                .swap(false, std::sync::atomic::Ordering::SeqCst)
            {
                self.reservation_read_held.notify_one();
                self.release_reservation_read.notified().await;
            }
            Ok(self
                .rows
                .lock()
                .unwrap()
                .get(&(base_uri.to_string(), version))
                .map(|row| (row.0.clone(), row.2.clone())))
        }

        async fn put_if_predecessor(
            &self,
            base_uri: &str,
            version: u64,
            path: &str,
            size: u64,
            predecessor: &PredecessorIdentity,
        ) -> Result<Reservation> {
            if self
                .hold_next_reservation
                .swap(false, std::sync::atomic::Ordering::SeqCst)
            {
                self.reservation_held.notify_one();
                self.release_reservation.notified().await;
            }
            let identity = self.mint();
            let visible_generation = if self
                .hide_generation
                .load(std::sync::atomic::Ordering::SeqCst)
            {
                None
            } else {
                self.generation.lock().unwrap().clone()
            };
            let mut rows = self.rows.lock().unwrap();
            let current = rows
                .get(&(base_uri.to_string(), predecessor.version))
                .map(|row| row.2.clone());
            if visible_generation != predecessor.generation
                || current.as_deref() != Some(predecessor.identity.as_str())
            {
                return Ok(Reservation::Refused {
                    reason: format!("manifest {} changed", predecessor.version),
                });
            }
            if self.below_floor(base_uri, version) {
                return Ok(Reservation::Refused {
                    reason: format!("version {version} was forgotten and is never reused"),
                });
            }
            let key = (base_uri.to_string(), version);
            if rows.contains_key(&key) {
                return Ok(Reservation::Taken);
            }
            if self
                .delay_next_reservation
                .swap(false, std::sync::atomic::Ordering::SeqCst)
            {
                *self.pending_reservation.lock().unwrap() =
                    Some((key, (path.to_string(), size, identity)));
                return Err(Error::io("simulated reservation timeout"));
            }
            rows.insert(key, (path.to_string(), size, identity.clone()));
            if self
                .conflict_after_applying_next_reservation
                .swap(false, std::sync::atomic::Ordering::SeqCst)
            {
                return Err(Error::commit_conflict_source(
                    version,
                    "simulated retry observed the applied reservation".into(),
                ));
            }
            if self
                .refuse_after_applying_next_reservation
                .swap(false, std::sync::atomic::Ordering::SeqCst)
            {
                return Err(lance_core::error::PrerequisiteFailedSnafu {
                    message: "simulated retry saw the predecessor superseded".to_string(),
                }
                .build());
            }
            if self
                .lose_next_reservation_response
                .swap(false, std::sync::atomic::Ordering::SeqCst)
            {
                return Err(Error::io("simulated lost reservation response"));
            }
            Ok(Reservation::Reserved { identity })
        }
    }

    fn gen_schema(generation: &str) -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new_with_metadata(
            vec![ArrowField::new("i", DataType::Int32, false)],
            HashMap::from([("gen".to_string(), generation.to_string())]),
        ))
    }

    fn gen_batch(generation: &str) -> RecordBatch {
        RecordBatch::try_new(
            gen_schema(generation),
            vec![Arc::new(Int32Array::from_iter_values(0..3_i32))],
        )
        .unwrap()
    }

    async fn create_with(uri: &str, generation: &str, handler: Arc<dyn CommitHandler>) -> Dataset {
        InsertBuilder::new(uri)
            .with_params(&WriteParams {
                commit_handler: Some(handler),
                ..Default::default()
            })
            .execute(vec![gen_batch(generation)])
            .await
            .unwrap()
    }

    async fn staged_append(dataset: &Dataset, generation: &str) -> Transaction {
        InsertBuilder::new(WriteDestination::Dataset(Arc::new(dataset.clone())))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .execute_uncommitted(vec![gen_batch(generation)])
            .await
            .unwrap()
    }

    /// Drop the dataset at `uri` from storage and from the store, then
    /// declare a new one there with a different generation.
    async fn recreate(uri: &str, store: &Arc<IdentifiedStore>, generation: &str) -> Dataset {
        std::fs::remove_dir_all(uri).unwrap();
        store.delete(uri.trim_start_matches('/')).await.unwrap();
        create_with(uri, generation, store.handler()).await
    }

    /// Stores that cannot decide the predecessor condition atomically refuse
    /// the option outright rather than checking it on the side.
    #[tokio::test]
    async fn test_required_schema_metadata_is_refused_where_publication_cannot_be_conditioned() {
        let dataset = InsertBuilder::new("memory://required-unsupported")
            .execute(vec![gen_batch("a")])
            .await
            .unwrap();
        let txn = staged_append(&dataset, "a").await;
        let err = CommitBuilder::new(WriteDestination::Dataset(Arc::new(dataset)))
            .with_required_schema_metadata("gen", "a")
            .execute(txn)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::NotSupported { .. }), "{err}");
    }

    /// Matching metadata lands; metadata changed before the commit fails it
    /// even though the stale handle still shows the old value.
    #[tokio::test]
    async fn test_required_schema_metadata_is_judged_on_the_latest_manifest() {
        let store = Arc::new(IdentifiedStore::default());
        let uri = TempStrDir::default();
        let dataset = create_with(uri.as_str(), "a", store.handler()).await;

        let txn = staged_append(&dataset, "a").await;
        let committed = CommitBuilder::new(WriteDestination::Dataset(Arc::new(dataset.clone())))
            .with_required_schema_metadata("gen", "a")
            .execute(txn)
            .await
            .unwrap();
        assert_eq!(committed.version().version, 2);

        let stale = committed.clone();
        let txn = staged_append(&stale, "a").await;
        let mut moved = committed.clone();
        moved
            .update_schema_metadata([("gen".to_string(), Some("b".to_string()))])
            .await
            .unwrap();
        let err = CommitBuilder::new(WriteDestination::Dataset(Arc::new(stale)))
            .with_required_schema_metadata("gen", "a")
            .execute(txn)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::PrerequisiteFailed { .. }), "{err}");
        assert_eq!(store.versions(), vec![1, 2, 3]);
    }

    /// A dataset recreated at the same path restarts its versions, so the
    /// stale handle's conflict scan sees nothing newer; the requirement is
    /// judged on the store's latest entry and refuses the writer.
    #[tokio::test]
    async fn test_required_schema_metadata_refuses_a_same_version_recreation() {
        let store = Arc::new(IdentifiedStore::default());
        let uri = TempStrDir::default();
        let stale = create_with(uri.as_str(), "a", store.handler()).await;
        let mut recreated = recreate(uri.as_str(), &store, "b").await;
        assert_eq!(recreated.version().version, stale.version().version);

        let txn = staged_append(&stale, "a").await;
        let err = CommitBuilder::new(WriteDestination::Dataset(Arc::new(stale)))
            .with_required_schema_metadata("gen", "a")
            .execute(txn)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::PrerequisiteFailed { .. }), "{err}");
        recreated.checkout_latest().await.unwrap();
        assert_eq!(recreated.version().version, 1);
        assert_eq!(store.versions(), vec![1]);
    }

    /// A recreation landing after the judgement but before the reservation
    /// is refused by the reservation itself: nothing is published, and the
    /// recreated dataset is untouched.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_a_recreation_between_judgement_and_reservation_is_refused() {
        let store = Arc::new(IdentifiedStore::default());
        let uri = TempStrDir::default();
        let stale = create_with(uri.as_str(), "a", store.handler()).await;
        let txn = staged_append(&stale, "a").await;

        store
            .hold_next_reservation
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let committing = tokio::spawn(async move {
            CommitBuilder::new(WriteDestination::Dataset(Arc::new(stale)))
                .with_required_schema_metadata("gen", "a")
                .execute(txn)
                .await
        });
        tokio::time::timeout(Duration::from_secs(30), store.reservation_held.notified())
            .await
            .expect("the commit never reached its reservation");

        let mut recreated = recreate(uri.as_str(), &store, "b").await;
        store.release_reservation.notify_one();

        let err = committing.await.unwrap().unwrap_err();
        assert!(matches!(err, Error::PrerequisiteFailed { .. }), "{err}");
        recreated.checkout_latest().await.unwrap();
        assert_eq!(recreated.version().version, 1);
        assert_eq!(store.versions(), vec![1]);
        assert!(
            recreated.schema().metadata.get("gen").map(String::as_str) == Some("b"),
            "the recreated dataset kept its own manifest"
        );
    }

    /// A reservation the store applied but did not acknowledge -- whether
    /// it reports an error or a conflict its own retry saw -- is read back
    /// rather than abandoned: the row names this commit's staging object,
    /// so the commit proceeds and lands.
    #[rstest::rstest]
    #[case::lost_response("lost")]
    #[case::conflict_after_apply("conflict")]
    #[case::refused_after_apply("refused")]
    #[tokio::test]
    async fn test_a_lost_reservation_response_is_read_back(#[case] reported_as: &str) {
        let store = Arc::new(IdentifiedStore::default());
        let uri = TempStrDir::default();
        let dataset = create_with(uri.as_str(), "a", store.handler()).await;
        let txn = staged_append(&dataset, "a").await;
        let hook = match reported_as {
            "conflict" => &store.conflict_after_applying_next_reservation,
            "refused" => &store.refuse_after_applying_next_reservation,
            _ => &store.lose_next_reservation_response,
        };
        hook.store(true, std::sync::atomic::Ordering::SeqCst);
        let committed = CommitBuilder::new(WriteDestination::Dataset(Arc::new(dataset)))
            .with_required_schema_metadata("gen", "a")
            .execute(txn)
            .await
            .unwrap();
        assert_eq!(committed.version().version, 2);
        assert_eq!(store.versions(), vec![1, 2]);
    }

    /// Drop and recreate at `uri` while the stale writer is parked, and give
    /// the recreated dataset its own version 2. Returns it and its version-2
    /// store row.
    async fn recreate_and_publish_v2(
        uri: &str,
        store: &Arc<IdentifiedStore>,
    ) -> (Dataset, (String, u64, String)) {
        let recreated = recreate(uri, store, "b").await;
        let own = staged_append(&recreated, "b").await;
        let recreated = CommitBuilder::new(WriteDestination::Dataset(Arc::new(recreated)))
            .execute(own)
            .await
            .unwrap();
        let row = store
            .rows
            .lock()
            .unwrap()
            .iter()
            .find(|(key, _)| key.1 == 2)
            .map(|(_, row)| row.clone())
            .unwrap();
        (recreated, row)
    }

    fn v2_row(store: &IdentifiedStore) -> (String, u64, String) {
        store
            .rows
            .lock()
            .unwrap()
            .iter()
            .find(|(key, _)| key.1 == 2)
            .map(|(_, row)| row.clone())
            .unwrap()
    }

    /// A lost reservation response is read back as one `(path, identity)`
    /// observation. When a recreation has replaced the entry by then, the
    /// observation names the replacement's manifest, not this commit's, so
    /// nothing is adopted and the replacement is untouched.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_a_lost_reservation_readback_does_not_adopt_a_replacement() {
        let store = Arc::new(IdentifiedStore::default());
        let uri = TempStrDir::default();
        let stale = create_with(uri.as_str(), "a", store.handler()).await;
        let txn = staged_append(&stale, "a").await;

        store
            .lose_next_reservation_response
            .store(true, std::sync::atomic::Ordering::SeqCst);
        store
            .hold_next_reservation_read
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let committing = tokio::spawn(async move {
            CommitBuilder::new(WriteDestination::Dataset(Arc::new(stale)))
                .with_required_schema_metadata("gen", "a")
                .execute(txn)
                .await
        });
        tokio::time::timeout(
            Duration::from_secs(30),
            store.reservation_read_held.notified(),
        )
        .await
        .expect("the commit never read its reservation back");

        let (mut recreated, own_row) = recreate_and_publish_v2(uri.as_str(), &store).await;
        store.release_reservation_read.notify_one();

        let err = committing.await.unwrap().unwrap_err();
        assert!(matches!(err, Error::PrerequisiteFailed { .. }), "{err}");
        assert_eq!(v2_row(&store), own_row);
        recreated.checkout_latest().await.unwrap();
        assert_eq!(recreated.version().version, 2);
        assert_eq!(
            recreated.schema().metadata.get("gen").map(String::as_str),
            Some("b")
        );
    }

    /// A conditioned commit publishes its manifest at a path of its own and
    /// never at the canonical one, so a recreated dataset's canonical
    /// manifest for the same version can never be touched by it.
    #[tokio::test]
    async fn test_conditioned_publication_never_uses_the_canonical_path() {
        let store = Arc::new(IdentifiedStore::default());
        let uri = TempStrDir::default();
        let dataset = create_with(uri.as_str(), "a", store.handler()).await;
        let txn = staged_append(&dataset, "a").await;
        let committed = CommitBuilder::new(WriteDestination::Dataset(Arc::new(dataset)))
            .with_required_schema_metadata("gen", "a")
            .execute(txn)
            .await
            .unwrap();
        let canonical = committed
            .manifest_location()
            .naming_scheme
            .manifest_path(&committed.base, 2);
        let published = committed.manifest_location().path.clone();
        assert_ne!(published, canonical);
        assert_eq!(published.filename(), canonical.filename());
        assert_eq!(v2_row(&store).0, published.to_string());
        let object_store = committed.object_store.clone();
        assert!(object_store.exists(&published).await.unwrap());
        assert!(!object_store.exists(&canonical).await.unwrap());

        // The published manifest is what a fresh open reads.
        let reopened = DatasetBuilder::from_uri(uri.as_str())
            .with_commit_handler(store.handler())
            .load()
            .await
            .unwrap();
        assert_eq!(reopened.version().version, 2);
        assert_eq!(reopened.manifest_location().path, published);
    }

    /// A conditioned version is history like any other: a handle that never
    /// saw it discovers it through the store and rebases onto it instead of
    /// conflicting forever, and the dataset's version list includes it.
    #[tokio::test]
    async fn test_conditioned_versions_are_discovered_as_history() {
        let store = Arc::new(IdentifiedStore::default());
        let uri = TempStrDir::default();
        let dataset = create_with(uri.as_str(), "a", store.handler()).await;
        let stale = dataset.clone();
        let txn = staged_append(&dataset, "a").await;
        CommitBuilder::new(WriteDestination::Dataset(Arc::new(dataset)))
            .with_required_schema_metadata("gen", "a")
            .execute(txn)
            .await
            .unwrap();

        let txn = staged_append(&stale, "a").await;
        let committed = CommitBuilder::new(WriteDestination::Dataset(Arc::new(stale)))
            .execute(txn)
            .await
            .unwrap();
        assert_eq!(committed.version().version, 3);
        assert_eq!(store.versions(), vec![1, 2, 3]);
        let versions: Vec<u64> = committed
            .versions()
            .await
            .unwrap()
            .iter()
            .map(|v| v.version)
            .collect();
        assert_eq!(versions, vec![1, 2, 3]);
    }

    /// A drop ends the generation before anything else: a reservation that
    /// reaches the store after the drop began is refused even where the
    /// recreation happens to carry the predecessor's identity again.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_a_drop_fences_a_conditioned_reservation() {
        let store = Arc::new(IdentifiedStore::default());
        let uri = TempStrDir::default();
        let stale = create_with(uri.as_str(), "a", store.handler()).await;
        let txn = staged_append(&stale, "a").await;
        let predecessor_identity = v_row(&store, 1).2;

        store
            .hold_next_reservation
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let committing = tokio::spawn(async move {
            CommitBuilder::new(WriteDestination::Dataset(Arc::new(stale)))
                .with_required_schema_metadata("gen", "a")
                .execute(txn)
                .await
        });
        tokio::time::timeout(Duration::from_secs(30), store.reservation_held.notified())
            .await
            .expect("the commit never reached its reservation");

        // Drop, then recreate with the *same* metadata and force the old
        // identity back onto version 1: only the generation tells them apart.
        let mut recreated = recreate(uri.as_str(), &store, "a").await;
        store
            .rows
            .lock()
            .unwrap()
            .get_mut(&(uri.as_str().trim_start_matches('/').to_string(), 1))
            .unwrap()
            .2 = predecessor_identity;
        store.release_reservation.notify_one();

        let err = committing.await.unwrap().unwrap_err();
        assert!(matches!(err, Error::PrerequisiteFailed { .. }), "{err}");
        recreated.checkout_latest().await.unwrap();
        assert_eq!(recreated.version().version, 1);
        assert_eq!(store.versions(), vec![1]);
    }

    /// Cleanup forgets the record of a manifest it deleted, so history stays
    /// what exists.
    #[tokio::test]
    async fn test_cleanup_forgets_deleted_conditioned_versions() {
        let store = Arc::new(IdentifiedStore::default());
        let uri = TempStrDir::default();
        let mut dataset = create_with(uri.as_str(), "a", store.handler()).await;
        for _ in 0..2 {
            let txn = staged_append(&dataset, "a").await;
            dataset = CommitBuilder::new(WriteDestination::Dataset(Arc::new(dataset)))
                .with_required_schema_metadata("gen", "a")
                .execute(txn)
                .await
                .unwrap();
        }
        assert_eq!(store.versions(), vec![1, 2, 3]);
        crate::dataset::cleanup::cleanup_old_versions(
            &dataset,
            crate::dataset::cleanup::CleanupPolicy {
                before_version: Some(3),
                delete_unverified: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(store.versions(), vec![3]);
    }

    /// A forget carries the path and generation cleanup listed, so a
    /// recreation that reused the version since keeps its record.
    #[tokio::test]
    async fn test_a_delayed_forget_leaves_a_recreated_row() {
        let store = Arc::new(IdentifiedStore::default());
        let uri = TempStrDir::default();
        let old = create_with(uri.as_str(), "a", store.handler()).await;
        let base = old.base.clone();
        let handler = store.handler();
        let txn = staged_append(&old, "a").await;
        let old = CommitBuilder::new(WriteDestination::Dataset(Arc::new(old)))
            .with_required_schema_metadata("gen", "a")
            .execute(txn)
            .await
            .unwrap();
        // What cleanup carries: the snapshot's generation and the listed path.
        let generation = old.manifest_location.generation.clone();
        assert!(generation.is_some());
        let old_path = Path::parse(v_row(&store, 2).0).unwrap();

        let replacement = recreate(uri.as_str(), &store, "b").await;
        let txn = staged_append(&replacement, "b").await;
        let replacement = CommitBuilder::new(WriteDestination::Dataset(Arc::new(replacement)))
            .with_required_schema_metadata("gen", "b")
            .execute(txn)
            .await
            .unwrap();
        let replacement_row = v_row(&store, 2);

        handler
            .forget_version(&base, 2, &old_path, generation.as_deref())
            .await
            .unwrap();
        assert_eq!(store.versions(), vec![1, 2]);
        assert_eq!(v_row(&store, 2), replacement_row);

        // The current generation's own record still forgets.
        let generation = replacement.manifest_location.generation.clone();
        handler
            .forget_version(
                &base,
                2,
                &Path::parse(replacement_row.0).unwrap(),
                generation.as_deref(),
            )
            .await
            .unwrap();
        assert_eq!(store.versions(), vec![1]);
    }

    /// A reservation that errors without a readable outcome may still land:
    /// its manifest is retained and the commit reports an unknown status.
    #[tokio::test]
    async fn test_an_absent_readback_retains_the_candidate() {
        let store = Arc::new(IdentifiedStore::default());
        let uri = TempStrDir::default();
        let dataset = create_with(uri.as_str(), "a", store.handler()).await;
        let object_store = dataset.object_store(None).await.unwrap();
        let txn = staged_append(&dataset, "a").await;
        store
            .delay_next_reservation
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let err = CommitBuilder::new(WriteDestination::Dataset(Arc::new(dataset.clone())))
            .with_max_retries(1)
            .with_required_schema_metadata("gen", "a")
            .execute(txn)
            .await
            .unwrap_err();
        assert!(err.is_commit_status_unknown(), "{err}");
        assert_eq!(store.versions(), vec![1]);

        store.apply_pending_reservation();
        assert!(
            object_store
                .exists(&Path::parse(v2_row(&store).0).unwrap())
                .await
                .unwrap()
        );
        let mut dataset = dataset;
        dataset.checkout_latest().await.unwrap();
        assert_eq!(dataset.version().version, 2);
        assert_eq!(dataset.count_rows(None).await.unwrap(), 6);
    }

    /// Commit `n` conditioned appends and return the last snapshot.
    async fn append_n(mut dataset: Dataset, generation: &str, n: usize) -> Dataset {
        for _ in 0..n {
            let txn = staged_append(&dataset, generation).await;
            dataset = CommitBuilder::new(WriteDestination::Dataset(Arc::new(dataset)))
                .with_required_schema_metadata("gen", generation)
                .execute(txn)
                .await
                .unwrap();
        }
        dataset
    }

    /// A handle over a dataset since dropped and recreated cannot clean it
    /// up: the listing carries the recreation's generation, not the
    /// snapshot's, and nothing is touched.
    #[tokio::test]
    async fn test_stale_cleanup_is_refused_after_a_recreation() {
        let store = Arc::new(IdentifiedStore::default());
        let uri = TempStrDir::default();
        let stale = append_n(
            create_with(uri.as_str(), "a", store.handler()).await,
            "a",
            2,
        )
        .await;
        assert_eq!(stale.version().version, 3);

        let mut recreated = append_n(recreate(uri.as_str(), &store, "b").await, "b", 1).await;
        let rows_before = store.rows.lock().unwrap().clone();

        let err = crate::dataset::cleanup::cleanup_old_versions(
            &stale,
            crate::dataset::cleanup::CleanupPolicy {
                before_version: Some(3),
                delete_unverified: true,
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("cleanup refused"), "{err}");
        assert_eq!(*store.rows.lock().unwrap(), rows_before);
        recreated.checkout_latest().await.unwrap();
        assert_eq!(recreated.version().version, 2);
        assert_eq!(recreated.count_rows(None).await.unwrap(), 6);
    }

    /// A forget that fails part-way leaves records whose manifests are gone;
    /// the retry lists them, finds the objects missing and forgets them, so
    /// cleanup converges with nothing dangling and nothing orphaned.
    #[tokio::test]
    async fn test_cleanup_retry_converges_after_a_forget_failure() {
        let store = Arc::new(IdentifiedStore::default());
        let uri = TempStrDir::default();
        let dataset = append_n(
            create_with(uri.as_str(), "a", store.handler()).await,
            "a",
            2,
        )
        .await;
        let old_paths: Vec<Path> = [1, 2]
            .iter()
            .map(|v| Path::parse(v_row(&store, *v).0).unwrap())
            .collect();
        let policy = || crate::dataset::cleanup::CleanupPolicy {
            before_version: Some(3),
            delete_unverified: true,
            ..Default::default()
        };
        store
            .fail_forget_call
            .store(2, std::sync::atomic::Ordering::SeqCst);
        crate::dataset::cleanup::cleanup_old_versions(&dataset, policy())
            .await
            .unwrap_err();
        let after_failure = store.versions();
        assert!(
            after_failure.len() == 2 && after_failure.contains(&3),
            "the failed forget kept its record: {after_failure:?}"
        );

        crate::dataset::cleanup::cleanup_old_versions(&dataset, policy())
            .await
            .unwrap();
        assert_eq!(store.versions(), vec![3]);
        for path in old_paths {
            assert!(
                !dataset.object_store.exists(&path).await.unwrap(),
                "orphaned candidate manifest {path}"
            );
        }
    }

    /// A store that owns publication but keeps no generation can fence
    /// nothing: commits beyond creation and cleanup through it are refused
    /// rather than trusted.
    #[tokio::test]
    async fn test_an_owning_store_without_a_generation_is_refused() {
        let store = Arc::new(IdentifiedStore::default());
        store
            .hide_generation
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let uri = TempStrDir::default();
        let created = create_with(uri.as_str(), "a", store.handler()).await;
        let txn = staged_append(&created, "a").await;
        let err = CommitBuilder::new(WriteDestination::Dataset(Arc::new(created.clone())))
            .execute(txn)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::PrerequisiteFailed { .. }), "{err}");
        assert!(err.to_string().contains("generation"), "{err}");
        let err = crate::dataset::cleanup::cleanup_old_versions(
            &created,
            crate::dataset::cleanup::CleanupPolicy {
                before_version: Some(1),
                delete_unverified: true,
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("no generation"), "{err}");
        assert_eq!(store.versions(), vec![1]);
    }

    /// A store that owns publication holds the only path to its manifests;
    /// when it cannot enumerate them nothing may stand in, least of all a
    /// cleanup that would take an empty listing for an empty history.
    #[tokio::test]
    async fn test_cleanup_refuses_an_owning_store_that_cannot_enumerate() {
        let store = Arc::new(IdentifiedStore::default());
        let uri = TempStrDir::default();
        let dataset = create_with(uri.as_str(), "a", store.handler()).await;
        store
            .hide_records
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let err = crate::dataset::cleanup::cleanup_old_versions(
            &dataset,
            crate::dataset::cleanup::CleanupPolicy {
                before_version: Some(1),
                delete_unverified: true,
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("cannot enumerate"), "{err}");
        assert!(
            dataset
                .object_store
                .exists(&dataset.manifest_location.path)
                .await
                .unwrap()
        );
    }

    /// A stray candidate manifest is reclaimed only once its version is at
    /// or below the no-reuse floor cleanup itself raised; one above it may
    /// still be reserved and is left alone.
    #[tokio::test]
    async fn test_cleanup_reclaims_only_candidates_below_the_reuse_floor() {
        let store = Arc::new(IdentifiedStore::default());
        let uri = TempStrDir::default();
        let dataset = append_n(
            create_with(uri.as_str(), "a", store.handler()).await,
            "a",
            2,
        )
        .await;
        let current = dataset.manifest_location.path.clone();
        let stray = |version: u64| {
            let canonical = dataset
                .manifest_location
                .naming_scheme
                .manifest_path(&dataset.base, version);
            dataset
                .base
                .clone()
                .join(CANDIDATES_DIR)
                .join("stray")
                .join(canonical.filename().unwrap())
        };
        let forgotten = stray(2);
        let pending = stray(99);
        for path in [&forgotten, &pending] {
            dataset.object_store.put(path, b"stray").await.unwrap();
        }
        let policy = || crate::dataset::cleanup::CleanupPolicy {
            before_version: Some(3),
            delete_unverified: true,
            ..Default::default()
        };

        // The first pass forgets 1 and 2 -- the floor rises after the sweep.
        crate::dataset::cleanup::cleanup_old_versions(&dataset, policy())
            .await
            .unwrap();
        assert_eq!(store.reuse_floor("dummy").await.unwrap(), None);
        assert!(dataset.object_store.exists(&forgotten).await.unwrap());
        // The second pass reclaims what is below the floor and nothing else.
        crate::dataset::cleanup::cleanup_old_versions(&dataset, policy())
            .await
            .unwrap();
        assert!(!dataset.object_store.exists(&forgotten).await.unwrap());
        assert!(dataset.object_store.exists(&pending).await.unwrap());
        assert!(dataset.object_store.exists(&current).await.unwrap());
    }

    /// A reservation delayed past the cleanup that forgot its version is
    /// refused by the floor, even though its predecessor is retained.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_a_forgotten_version_is_never_reserved_again() {
        let store = Arc::new(IdentifiedStore::default());
        let uri = TempStrDir::default();
        let stale = create_with(uri.as_str(), "a", store.handler()).await;
        stale.tags().create("keep", 1).await.unwrap();
        let txn = staged_append(&stale, "a").await;
        store
            .hold_next_reservation
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let delayed = tokio::spawn(async move {
            CommitBuilder::new(WriteDestination::Dataset(Arc::new(stale)))
                .with_required_schema_metadata("gen", "a")
                .execute(txn)
                .await
        });
        tokio::time::timeout(Duration::from_secs(30), store.reservation_held.notified())
            .await
            .expect("the commit never reached its reservation");

        // Versions 2 and 3 land meanwhile; cleanup then forgets 2 (1 is tagged).
        let other = crate::dataset::builder::DatasetBuilder::from_uri(uri.as_str())
            .with_commit_handler(store.handler())
            .load()
            .await
            .unwrap();
        let other = append_n(other, "a", 2).await;
        crate::dataset::cleanup::cleanup_old_versions(
            &other,
            crate::dataset::cleanup::CleanupPolicy {
                before_version: Some(3),
                delete_unverified: true,
                error_if_tagged_old_versions: false,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(store.versions(), vec![1, 3]);

        store.release_reservation.notify_one();
        let err = delayed.await.unwrap().unwrap_err();
        assert!(matches!(err, Error::PrerequisiteFailed { .. }), "{err}");
        assert_eq!(store.versions(), vec![1, 3]);
    }

    /// A version retained by a tag may sit below the no-reuse floor; its
    /// manifest is still recorded and is never reclaimed.
    #[tokio::test]
    async fn test_cleanup_preserves_a_tagged_manifest_below_the_reuse_floor() {
        let store = Arc::new(IdentifiedStore::default());
        let uri = TempStrDir::default();
        let dataset = append_n(
            create_with(uri.as_str(), "a", store.handler()).await,
            "a",
            2,
        )
        .await;
        dataset.tags().create("keep", 1).await.unwrap();
        let tagged = Path::parse(v_row(&store, 1).0).unwrap();
        let policy = || crate::dataset::cleanup::CleanupPolicy {
            before_version: Some(3),
            delete_unverified: true,
            error_if_tagged_old_versions: false,
            ..Default::default()
        };
        for _ in 0..2 {
            crate::dataset::cleanup::cleanup_old_versions(&dataset, policy())
                .await
                .unwrap();
        }
        assert_eq!(store.versions(), vec![1, 3]);
        assert!(dataset.object_store.exists(&tagged).await.unwrap());
    }

    /// The floor ends with the generation: a dataset dropped and recreated
    /// at the same URI starts at version 1 again, and a stale handle's
    /// reclaim decision is refused.
    #[tokio::test]
    async fn test_the_reuse_floor_ends_with_the_generation() {
        let store = Arc::new(IdentifiedStore::default());
        let uri = TempStrDir::default();
        let old = append_n(
            create_with(uri.as_str(), "a", store.handler()).await,
            "a",
            2,
        )
        .await;
        crate::dataset::cleanup::cleanup_old_versions(
            &old,
            crate::dataset::cleanup::CleanupPolicy {
                before_version: Some(3),
                delete_unverified: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(store.reuse_floor(old.base.as_ref()).await.unwrap(), Some(2));
        let old_generation = old.manifest_location.generation.clone();

        let recreated = recreate(uri.as_str(), &store, "b").await;
        assert_eq!(recreated.version().version, 1);
        assert_eq!(store.versions(), vec![1]);
        let live = Path::parse(v_row(&store, 1).0).unwrap();
        let handler = store.handler();
        assert!(
            !handler
                .may_reclaim(&recreated.base, 1, &live, old_generation.as_deref())
                .await
                .unwrap(),
            "a stale generation may not reclaim"
        );
    }

    /// An unconditioned commit through an owning store is bound to the
    /// generation it began in: paused across a drop, it is refused rather
    /// than recorded in the recreation.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_a_delayed_unconditioned_reservation_cannot_cross_a_recreation() {
        let store = Arc::new(IdentifiedStore::default());
        let uri = TempStrDir::default();
        let stale = create_with(uri.as_str(), "a", store.handler()).await;
        let txn = staged_append(&stale, "a").await;
        store
            .hold_next_reservation
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let delayed = tokio::spawn(async move {
            CommitBuilder::new(WriteDestination::Dataset(Arc::new(stale)))
                .execute(txn)
                .await
        });
        tokio::time::timeout(Duration::from_secs(30), store.reservation_held.notified())
            .await
            .expect("the commit never reached its reservation");

        let mut recreated = recreate(uri.as_str(), &store, "b").await;
        store.release_reservation.notify_one();
        let err = delayed.await.unwrap().unwrap_err();
        assert!(matches!(err, Error::PrerequisiteFailed { .. }), "{err}");
        assert_eq!(store.versions(), vec![1]);
        recreated.checkout_latest().await.unwrap();
        assert_eq!(recreated.version().version, 1);
        assert_eq!(
            recreated.schema().metadata.get("gen").map(String::as_str),
            Some("b")
        );
    }

    /// A handle over a dataset since dropped and recreated -- even at the
    /// same version, so nothing rebases -- cannot commit into the
    /// recreation: the transaction is bound to the generation the handle
    /// was loaded under.
    #[tokio::test]
    async fn test_a_stale_handle_cannot_commit_into_a_recreation_at_the_same_version() {
        let store = Arc::new(IdentifiedStore::default());
        let uri = TempStrDir::default();
        let stale = plain_append_n(
            create_with(uri.as_str(), "a", store.handler()).await,
            "a",
            1,
        )
        .await;
        assert_eq!(stale.version().version, 2);
        let mut recreated = plain_append_n(recreate(uri.as_str(), &store, "b").await, "b", 1).await;
        assert_eq!(recreated.version().version, 2);
        let rows_before = store.rows.lock().unwrap().clone();

        let txn = staged_append(&stale, "a").await;
        let err = CommitBuilder::new(WriteDestination::Dataset(Arc::new(stale)))
            .execute(txn)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::PrerequisiteFailed { .. }), "{err}");
        assert_eq!(*store.rows.lock().unwrap(), rows_before);
        recreated.checkout_latest().await.unwrap();
        assert_eq!(recreated.version().version, 2);
    }

    /// The handle that creates a dataset already carries the generation its
    /// first record minted, so it can clean up without reloading.
    #[tokio::test]
    async fn test_a_creation_handle_carries_its_generation() {
        let store = Arc::new(IdentifiedStore::default());
        let uri = TempStrDir::default();
        let created = create_with(uri.as_str(), "a", store.handler()).await;
        let minted = store.generation.lock().unwrap().clone();
        assert!(minted.is_some());
        assert_eq!(created.manifest_location.generation, minted);
        crate::dataset::cleanup::cleanup_old_versions(
            &created,
            crate::dataset::cleanup::CleanupPolicy {
                before_version: Some(1),
                delete_unverified: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    }

    /// Commit `n` unconditioned appends and return the last snapshot.
    async fn plain_append_n(mut dataset: Dataset, generation: &str, n: usize) -> Dataset {
        for _ in 0..n {
            let txn = staged_append(&dataset, generation).await;
            dataset = CommitBuilder::new(WriteDestination::Dataset(Arc::new(dataset)))
                .execute(txn)
                .await
                .unwrap();
        }
        dataset
    }

    fn v_row(store: &IdentifiedStore, version: u64) -> (String, u64, String) {
        store
            .rows
            .lock()
            .unwrap()
            .iter()
            .find(|(key, _)| key.1 == version)
            .map(|(_, row)| row.clone())
            .unwrap()
    }

    /// Creating a dataset has no predecessor to judge; the option is refused
    /// rather than dropped.
    #[tokio::test]
    async fn test_required_schema_metadata_is_refused_for_new_datasets() {
        let uri = TempStrDir::default();
        let txn = InsertBuilder::new(uri.as_str())
            .execute_uncommitted(vec![gen_batch("a")])
            .await
            .unwrap();
        let err = CommitBuilder::new(uri.as_str())
            .with_required_schema_metadata("gen", "a")
            .execute(txn)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
    }

    /// Detached commits never chain onto the judged manifest, so the option
    /// is refused rather than silently ignored.
    #[tokio::test]
    async fn test_required_schema_metadata_is_refused_on_detached_commits() {
        let dataset = InsertBuilder::new("memory://required-detached")
            .execute(vec![gen_batch("a")])
            .await
            .unwrap();
        let txn = staged_append(&dataset, "a").await;
        let err = CommitBuilder::new(WriteDestination::Dataset(Arc::new(dataset)))
            .with_detached(true)
            .with_required_schema_metadata("gen", "a")
            .execute(txn)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
    }

    #[tokio::test]
    async fn test_reuse_session() {
        // Need to use in-memory for accurate IOPS tracking.
        let session = Arc::new(Session::default());
        // Create new dataset
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..10_i32))],
        )
        .unwrap();
        let dataset = InsertBuilder::new("memory://test")
            .with_params(&WriteParams {
                session: Some(session.clone()),
                enable_v2_manifest_paths: true,
                ..Default::default()
            })
            .execute(vec![batch])
            .await
            .unwrap();
        let dataset = Arc::new(dataset);

        let io_stats = dataset.object_store.as_ref().io_stats_incremental();
        assert_io_gt!(io_stats, read_iops, 0);
        assert_io_gt!(io_stats, write_iops, 0);

        // Commit transaction 5 times
        for i in 0..5 {
            let new_ds = CommitBuilder::new(dataset.clone())
                .execute(sample_transaction(1))
                .await
                .unwrap();
            assert_eq!(new_ds.manifest.version, i + 2);

            // Because we are writing transactions sequentially, and caching them,
            // we shouldn't need to read anything from disk. Except we do need
            // to check for the latest version to see if we need to do conflict
            // resolution.
            let io_stats = dataset.object_store.as_ref().io_stats_incremental();
            assert_io_eq!(io_stats, read_iops, 1, "check latest version, i = {} ", i);
            // Should see 2 IOPs:
            // 1. Write the transaction files
            // 2. Write (conditional put) the manifest
            // (the version hint is only written on non-lexically-ordered stores)
            assert_io_eq!(io_stats, write_iops, 2, "write txn + manifest, i = {}", i);
        }

        // Commit transaction with URI and session
        let new_ds = CommitBuilder::new("memory://test")
            .with_session(dataset.session.clone())
            .execute(sample_transaction(1))
            .await
            .unwrap();
        assert_eq!(new_ds.manifest().version, 7);
        // Session should still be re-used
        // However, the dataset needs to be loaded and the read version checked out.
        // The read version's manifest body is served from the session cache (it
        // was cached when v1 was first created), so the checkout only pays the
        // version-resolution head, not a manifest read.
        let io_stats = dataset.object_store.as_ref().io_stats_incremental();
        assert_io_eq!(io_stats, read_iops, 3, "load dataset + check version");
        assert_io_eq!(io_stats, write_iops, 2, "write txn + manifest");

        // Commit transaction with URI and new session. Re-use the store
        // registry so we see the same store.
        let new_session = Arc::new(Session::new(0, 0, session.store_registry()));
        let new_ds = CommitBuilder::new("memory://test")
            .with_session(new_session)
            .execute(sample_transaction(1))
            .await
            .unwrap();
        assert_eq!(new_ds.manifest().version, 8);
        // Now we have to load all previous transactions.

        let io_stats = dataset.object_store.as_ref().io_stats_incremental();
        assert_io_gt!(io_stats, read_iops, 10);
        assert_io_eq!(io_stats, write_iops, 2, "write txn + manifest");
    }

    #[tokio::test]
    async fn test_commit_iops() {
        // If there's no conflicts, we should be able to commit in 2 io requests:
        // * write txn file (this could be optional one day)
        // * write manifest
        let session = Arc::new(Session::default());
        let write_params = WriteParams {
            session: Some(session.clone()),
            ..Default::default()
        };
        let data = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "a",
                DataType::Int32,
                false,
            )])),
            vec![Arc::new(Int32Array::from(vec![0; 5]))],
        )
        .unwrap();
        let dataset = InsertBuilder::new("memory://")
            .with_params(&write_params)
            .execute(vec![data])
            .await
            .unwrap();

        dataset.object_store.as_ref().io_stats_incremental(); // Reset the stats
        let read_version = dataset.manifest().version;
        let new_ds = CommitBuilder::new(Arc::new(dataset))
            .execute(sample_transaction(read_version))
            .await
            .unwrap();

        // Assert io requests
        let io_stats = new_ds.object_store.as_ref().io_stats_incremental();
        // This could be zero, if we decided to be optimistic. However, that
        // would mean wasted write requests (txn + manifest) if there was
        // a conflict. We choose to be pessimistic for more consistent performance.
        assert_io_eq!(io_stats, read_iops, 1);
        assert_io_eq!(io_stats, write_iops, 2);
        // We can't write them in parallel. The transaction file must exist before
        // we can write the manifest.
        assert_io_eq!(io_stats, num_stages, 3);
    }

    #[tokio::test]
    #[rstest::rstest]
    async fn test_commit_conflict_iops(#[values(true, false)] use_cache: bool) {
        let cache_size = if use_cache { 1_000_000 } else { 0 };
        let session = Arc::new(Session::new(0, cache_size, Default::default()));
        // We need throttled to correctly count num hops. Otherwise, memory store
        // returns synchronously, and each request is 1 hop.
        let throttled = Arc::new(ThrottledStoreWrapper {
            config: ThrottleConfig {
                wait_list_per_call: Duration::from_millis(5),
                wait_get_per_call: Duration::from_millis(5),
                wait_put_per_call: Duration::from_millis(5),
                ..Default::default()
            },
        });
        let write_params = WriteParams {
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(throttled),
                ..Default::default()
            }),
            session: Some(session.clone()),
            ..Default::default()
        };
        let data = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "a",
                DataType::Int32,
                false,
            )])),
            vec![Arc::new(Int32Array::from(vec![0; 5]))],
        )
        .unwrap();
        let mut dataset = InsertBuilder::new("memory://")
            .with_params(&write_params)
            .execute(vec![data])
            .await
            .unwrap();
        let original_dataset = Arc::new(dataset.clone());

        // Create 3 other transactions that happen concurrently.
        let num_other_txns = 3;
        for _ in 0..num_other_txns {
            dataset = CommitBuilder::new(original_dataset.clone())
                .execute(sample_transaction(dataset.manifest().version))
                .await
                .unwrap();
        }
        dataset.object_store.as_ref().io_stats_incremental();

        let new_ds = CommitBuilder::new(original_dataset.clone())
            .execute(sample_transaction(original_dataset.manifest().version))
            .await
            .unwrap();

        let io_stats = new_ds.object_store.as_ref().io_stats_incremental();

        // If there is a conflict with two transaction, the retry should require io requests:
        // * 1 list version
        // * num_other_txns read manifests (cache-able)
        // * num_other_txns read txn files (cache-able)
        // * 1 write txn file
        // * 1 write manifest
        // For total of 3 + 2 * num_other_txns io requests. If we have caching enabled, we can skip 2 * num_other_txns
        // of those. We should be able to read in 5 hops.
        if use_cache {
            assert_io_eq!(io_stats, read_iops, 1); // Just list versions
            assert_io_eq!(io_stats, num_stages, 3);
        } else {
            // We need to read the other manifests and transactions.

            use lance_io::assert_io_lt;
            assert_io_eq!(io_stats, read_iops, 1 + num_other_txns * 2);
            // It's possible to read the txns for some versions before we
            // finish reading later versions and so the entire "read versions
            // and txs" may appear as 1 hop instead of 2.
            assert_io_lt!(io_stats, num_stages, 6);
        }
        assert_io_eq!(io_stats, write_iops, 2); // txn + manifest
    }

    #[test]
    fn test_commit_timeout_default_is_thirty_minutes() {
        let builder = CommitBuilder::new("memory://default-timeout");
        assert_eq!(builder.timeout, Some(DEFAULT_COMMIT_TIMEOUT));
        assert_eq!(DEFAULT_COMMIT_TIMEOUT, Duration::from_secs(1800));
    }

    #[test]
    fn test_commit_retry_timeout_default_is_thirty_seconds() {
        let builder = CommitBuilder::new("memory://default-retry-timeout");
        assert_eq!(builder.retry_timeout, DEFAULT_COMMIT_RETRY_TIMEOUT);
        assert_eq!(DEFAULT_COMMIT_RETRY_TIMEOUT, Duration::from_secs(30));
    }

    #[tokio::test]
    async fn test_commit_timeout_zero_rejected() {
        let dataset = Arc::new(
            InsertBuilder::new("memory://test")
                .execute(vec![
                    RecordBatch::try_new(
                        Arc::new(ArrowSchema::new(vec![ArrowField::new(
                            "i",
                            DataType::Int32,
                            false,
                        )])),
                        vec![Arc::new(Int32Array::from_iter_values(0..10_i32))],
                    )
                    .unwrap(),
                ])
                .await
                .unwrap(),
        );
        let res = CommitBuilder::new(dataset.clone())
            .with_timeout(Some(Duration::ZERO))
            .execute(sample_transaction(1))
            .await;
        assert!(
            matches!(res, Err(Error::InvalidInput { .. })),
            "got {res:?}"
        );
    }

    #[tokio::test]
    async fn test_commit_timeout_triggers() {
        let throttled = Arc::new(ThrottledStoreWrapper {
            config: ThrottleConfig {
                wait_put_per_call: Duration::from_secs(5),
                ..Default::default()
            },
        });
        let write_params = WriteParams {
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(throttled),
                ..Default::default()
            }),
            ..Default::default()
        };
        let dataset = InsertBuilder::new("memory://timeout")
            .with_params(&write_params)
            .execute(vec![
                RecordBatch::try_new(
                    Arc::new(ArrowSchema::new(vec![ArrowField::new(
                        "i",
                        DataType::Int32,
                        false,
                    )])),
                    vec![Arc::new(Int32Array::from_iter_values(0..10_i32))],
                )
                .unwrap(),
            ])
            .await
            .unwrap();

        let res = CommitBuilder::new(Arc::new(dataset))
            .with_timeout(Some(Duration::from_millis(50)))
            .execute(sample_transaction(1))
            .await;
        let err = res.expect_err("commit should time out");
        assert!(matches!(&err, Error::Timeout { .. }), "got {err:?}");
    }

    #[tokio::test]
    async fn test_commit_timeout_applies_to_execute_batch() {
        let throttled = Arc::new(ThrottledStoreWrapper {
            config: ThrottleConfig {
                wait_put_per_call: Duration::from_secs(5),
                ..Default::default()
            },
        });
        let write_params = WriteParams {
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(throttled),
                ..Default::default()
            }),
            ..Default::default()
        };
        let dataset = InsertBuilder::new("memory://batch-timeout")
            .with_params(&write_params)
            .execute(vec![
                RecordBatch::try_new(
                    Arc::new(ArrowSchema::new(vec![ArrowField::new(
                        "i",
                        DataType::Int32,
                        false,
                    )])),
                    vec![Arc::new(Int32Array::from_iter_values(0..10_i32))],
                )
                .unwrap(),
            ])
            .await
            .unwrap();

        let res = CommitBuilder::new(Arc::new(dataset))
            .with_timeout(Some(Duration::from_millis(50)))
            .execute_batch(vec![sample_transaction(1)])
            .await;
        let Err(err) = res else {
            panic!("commit should time out");
        };
        assert!(matches!(&err, Error::Timeout { .. }), "got {err:?}");
    }

    /// `with_timeout(None)` must let a commit run unbounded. Uses a throttled
    /// store so the commit takes real wall-clock time — long enough that the
    /// 50ms timeout in `test_commit_timeout_triggers` would have fired.
    #[tokio::test]
    async fn test_commit_timeout_none_disables() {
        let throttled = Arc::new(ThrottledStoreWrapper {
            config: ThrottleConfig {
                wait_put_per_call: Duration::from_millis(200),
                ..Default::default()
            },
        });
        let write_params = WriteParams {
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(throttled),
                ..Default::default()
            }),
            ..Default::default()
        };
        let dataset = InsertBuilder::new("memory://no-timeout")
            .with_params(&write_params)
            .execute(vec![
                RecordBatch::try_new(
                    Arc::new(ArrowSchema::new(vec![ArrowField::new(
                        "i",
                        DataType::Int32,
                        false,
                    )])),
                    vec![Arc::new(Int32Array::from_iter_values(0..10_i32))],
                )
                .unwrap(),
            ])
            .await
            .unwrap();

        let new_ds = CommitBuilder::new(Arc::new(dataset))
            .with_timeout(None)
            .execute(sample_transaction(1))
            .await
            .unwrap();
        assert_eq!(new_ds.manifest.version, 2);
    }

    #[tokio::test]
    async fn test_commit_retry_timeout_interrupts_conflict_backoff() {
        let dataset = InsertBuilder::new("memory://retry-timeout")
            .execute(vec![
                RecordBatch::try_new(
                    Arc::new(ArrowSchema::new(vec![ArrowField::new(
                        "i",
                        DataType::Int32,
                        false,
                    )])),
                    vec![Arc::new(Int32Array::from_iter_values(0..10_i32))],
                )
                .unwrap(),
            ])
            .await
            .unwrap();

        let result = CommitBuilder::new(Arc::new(dataset))
            .with_commit_handler(Arc::new(SlowConflictingCommitHandler))
            .with_max_retries(3)
            .with_retry_timeout(Duration::from_millis(150))
            .with_timeout(None)
            .execute(sample_transaction(1))
            .await;

        let error = result.expect_err("conflict backoff should respect retry timeout");
        assert!(
            matches!(&error, Error::TooMuchWriteContention { message, .. } if message.contains("failed on retry_timeout")),
            "got {error:?}"
        );
    }

    #[tokio::test]
    async fn test_commit_batch() {
        // Create a dataset
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..10_i32))],
        )
        .unwrap();
        let dataset = InsertBuilder::new("memory://test")
            .execute(vec![batch])
            .await
            .unwrap();
        let dataset = Arc::new(dataset);

        // Attempting to commit empty gives error
        let res = CommitBuilder::new(dataset.clone())
            .execute_batch(vec![])
            .await;
        assert!(matches!(res, Err(Error::InvalidInput { .. })));

        // Attempting to commit update gives error
        let update_transaction = Transaction {
            uuid: uuid::Uuid::new_v4().hyphenated().to_string(),
            operation: Operation::Update {
                updated_fragments: vec![],
                new_fragments: vec![],
                removed_fragment_ids: vec![],
                fields_modified: vec![],
                compacted_sstables: Vec::new(),
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: None,
                inserted_rows_filter: None,
                updated_fragment_offsets: None,
            },
            read_version: 1,
            tag: None,
            transaction_properties: None,
        };
        let res = CommitBuilder::new(dataset.clone())
            .execute_batch(vec![update_transaction])
            .await;
        assert!(matches!(res, Err(Error::NotSupported { .. })));

        // Doing multiple appends includes all.
        let append1 = sample_transaction(1);
        let append2 = sample_transaction(2);
        let mut expected_fragments = vec![];
        if let Operation::Append { fragments } = &append1.operation {
            expected_fragments.extend(fragments.clone());
        }
        if let Operation::Append { fragments } = &append2.operation {
            expected_fragments.extend(fragments.clone());
        }
        let res = CommitBuilder::new(dataset.clone())
            .execute_batch(vec![append1.clone(), append2.clone()])
            .await
            .unwrap();
        let transaction = res.merged;
        assert!(
            matches!(transaction.operation, Operation::Append { fragments } if fragments == expected_fragments)
        );
        assert_eq!(transaction.read_version, 1);
    }

    /// On non-lexically-ordered stores (e.g. S3 Express) a commit should use the
    /// version hint (a few HEAD probes, O(k)) instead of a full O(n) listing.
    #[tokio::test]
    async fn test_commit_uses_version_hint_on_non_lexical_store() {
        // Make `list` artificially slow per entry so a full listing would be
        // obvious; HEAD/GET/PUT stay fast.
        let throttled = Arc::new(ThrottledStoreWrapper {
            config: ThrottleConfig {
                wait_list_per_entry: Duration::from_millis(50),
                wait_get_per_call: Duration::from_millis(1),
                wait_put_per_call: Duration::from_millis(1),
                ..Default::default()
            },
        });
        let session = Arc::new(Session::default());
        let write_params = WriteParams {
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(throttled),
                list_is_lexically_ordered: Some(false),
                ..Default::default()
            }),
            session: Some(session.clone()),
            enable_v2_manifest_paths: true,
            ..Default::default()
        };

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..10_i32))],
        )
        .unwrap();
        let mut dataset = Arc::new(
            InsertBuilder::new("memory://test_version_hint")
                .with_params(&write_params)
                .execute(vec![batch])
                .await
                .unwrap(),
        );

        // Build up many versions so a full listing would be expensive.
        for _ in 0..50 {
            dataset = Arc::new(
                CommitBuilder::new(dataset.clone())
                    .execute(sample_transaction(dataset.manifest().version))
                    .await
                    .unwrap(),
            );
        }
        assert_eq!(dataset.manifest().version, 51);

        dataset.object_store.as_ref().io_stats_incremental();

        let start = std::time::Instant::now();
        let new_ds = CommitBuilder::new(dataset.clone())
            .execute(sample_transaction(dataset.manifest().version))
            .await
            .unwrap();
        let elapsed = start.elapsed();

        // A full listing of ~52 entries at 50ms each would take ~2.6s.
        assert!(
            elapsed < Duration::from_secs(1),
            "commit took {elapsed:?}; the version hint path was likely not used"
        );

        let io_stats = new_ds.object_store.as_ref().io_stats_incremental();
        assert!(
            io_stats.read_iops < 10,
            "read_iops = {}; a full listing was likely used",
            io_stats.read_iops
        );
    }
}
