// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use lance_file::version::LanceFileVersion;
use lance_io::object_store::{ObjectStore, ObjectStoreParams};
use lance_select::RowAddrTreeMap;
use lance_table::{
    format::{
        DataStorageFormat, Fragment, RowAddressLayoutDelta, RowAddressPlacementDelta,
        RowAddressPlacementKind, RowAddressTargetFragment, RowAddressTargetRange,
        is_detached_version,
    },
    io::commit::{CommitConfig, CommitHandler, ManifestNamingScheme},
};

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
use crate::io::commit::{read_transaction_file, same_transaction_request};
use lance_core::utils::tracing::{DATASET_COMMITTED_EVENT, TRACE_DATASET_EVENTS};
use tracing::info;
use uuid::Uuid;

/// Create a new commit from a [`Transaction`].
///
/// Transactions can be created using a write method like [`super::InsertBuilder::execute_uncommitted`].
#[derive(Debug, Clone)]
pub struct CommitBuilder<'a> {
    dest: WriteDestination<'a>,
    use_stable_row_ids: Option<bool>,
    enable_v2_manifest_paths: bool,
    storage_format: Option<LanceFileVersion>,
    commit_handler: Option<Arc<dyn CommitHandler>>,
    store_params: Option<ObjectStoreParams>,
    object_store: Option<Arc<ObjectStore>>,
    session: Option<Arc<Session>>,
    detached: bool,
    commit_config: CommitConfig,
    affected_rows: Option<RowAddrTreeMap>,
    transaction_properties: Option<Arc<HashMap<String, String>>>,
    timeout: Option<Duration>,
}

/// Default timeout applied to [`CommitBuilder::execute`] when none is set.
pub const DEFAULT_COMMIT_TIMEOUT: Duration = Duration::from_secs(1800);

fn source_free_direct_placements(fragments: &[Fragment]) -> Result<Vec<RowAddressPlacementDelta>> {
    fragments
        .iter()
        .enumerate()
        .map(|(ordinal, fragment)| {
            let row_count = u32::try_from(fragment.physical_rows.ok_or_else(|| {
                Error::invalid_input(format!(
                    "storage-version-2.3 source-free fragment {} is missing physical_rows",
                    fragment.id
                ))
            })?)
            .map_err(|_| {
                Error::format_capacity_exceeded(format!(
                    "storage-version-2.3 fragment {} has more than u32::MAX physical rows",
                    fragment.id
                ))
            })?;
            Ok(RowAddressPlacementDelta {
                source_selections: Vec::new(),
                target: RowAddressTargetRange {
                    fragment: RowAddressTargetFragment::NewFragmentOrdinal(
                        u32::try_from(ordinal).map_err(|_| {
                            Error::format_capacity_exceeded(
                                "storage-version-2.3 transaction has more than u32::MAX fragments",
                            )
                        })?,
                    ),
                    start_offset: 0,
                    end_offset: row_count,
                },
                placement_kind: RowAddressPlacementKind::Direct,
                output_cardinality: u64::from(row_count),
                output_row_sequence_fingerprint: Vec::new(),
            })
        })
        .collect()
}

async fn prepare_v2_3_transaction(
    transaction: &mut Transaction,
    dataset: Option<&Dataset>,
    requested_storage_format: Option<LanceFileVersion>,
) -> Result<()> {
    if dataset.is_none() && !matches!(transaction.operation, Operation::Overwrite { .. }) {
        return Ok(());
    }
    let target_version = transaction.target_file_version(
        dataset.map(|dataset| dataset.manifest.as_ref()),
        requested_storage_format,
    )?;
    if target_version != LanceFileVersion::V2_3 {
        return Ok(());
    }
    let source_manifest = match dataset {
        Some(dataset)
            if transaction.read_version != 0
                && transaction.read_version != dataset.version().version =>
        {
            Some(
                dataset
                    .checkout_version(transaction.read_version)
                    .await?
                    .manifest,
            )
        }
        Some(dataset) => Some(dataset.manifest.clone()),
        None => None,
    };

    match &transaction.operation {
        Operation::Append { fragments } | Operation::Overwrite { fragments, .. } => {
            if transaction.row_address_layout_delta.is_none() {
                let mut delta = if let Some(source_manifest) = &source_manifest {
                    let layout = source_manifest.row_address_layout.as_ref().ok_or_else(|| {
                        Error::invalid_input(
                            "storage-version-2.3 manifest is missing RowAddressLayout",
                        )
                    })?;
                    RowAddressLayoutDelta {
                        expected_layout_fingerprint: layout.fingerprint.clone(),
                        ..RowAddressLayoutDelta::default()
                    }
                } else {
                    let namespace_uuid = Uuid::parse_str(&transaction.uuid).map_err(|error| {
                        Error::invalid_input(format!(
                            "storage-version-2.3 create transaction UUID {} is invalid: {error}",
                            transaction.uuid
                        ))
                    })?;
                    if namespace_uuid.is_nil() {
                        return Err(Error::invalid_input(
                            "storage-version-2.3 create transaction UUID must not be nil",
                        ));
                    }
                    RowAddressLayoutDelta::for_create(namespace_uuid)
                };
                delta.placements = source_free_direct_placements(fragments)?;
                transaction.row_address_layout_delta = Some(delta);
            }
        }
        Operation::Merge { fragments, .. } => {
            let source_manifest = source_manifest.as_ref().ok_or_else(|| {
                Error::invalid_input("storage-version-2.3 Merge requires an existing dataset")
            })?;
            let current_fragment_ids = source_manifest
                .fragments
                .iter()
                .map(|fragment| fragment.id)
                .collect::<std::collections::HashSet<_>>();
            let new_fragments = fragments
                .iter()
                .filter(|fragment| !current_fragment_ids.contains(&fragment.id))
                .cloned()
                .collect::<Vec<_>>();
            if !new_fragments.is_empty()
                && transaction
                    .row_address_layout_delta
                    .as_ref()
                    .is_none_or(|delta| delta.placements.is_empty())
            {
                let layout = source_manifest.row_address_layout.as_ref().ok_or_else(|| {
                    Error::invalid_input("storage-version-2.3 manifest is missing RowAddressLayout")
                })?;
                let delta = transaction
                    .row_address_layout_delta
                    .get_or_insert_with(RowAddressLayoutDelta::default);
                delta.expected_layout_fingerprint = layout.fingerprint.clone();
                delta.placements = source_free_direct_placements(&new_fragments)?;
            }
        }
        Operation::Delete { .. } | Operation::Update { .. } | Operation::Rewrite { .. }
            if transaction.row_address_layout_delta.is_none() =>
        {
            return Err(Error::invalid_input(format!(
                "storage-version-2.3 {} requires core-authored row-address provenance; preserve the opaque transaction metadata returned by Lance or use the native dataset maintenance API",
                transaction.operation.name()
            )));
        }
        _ => {}
    }
    Ok(())
}

async fn reconcile_v2_3_sentinel_overwrite_retry(
    dataset: &Dataset,
    requested: &Transaction,
) -> Result<Option<Dataset>> {
    if requested.read_version != 0
        || !matches!(requested.operation, Operation::Overwrite { .. })
        || !dataset.manifest.uses_stable_logical_row_addresses()
    {
        return Ok(None);
    }
    let transaction_file = format!("0-{}.txn", requested.uuid);
    let transaction_path = dataset.transactions_dir().join(transaction_file.as_str());
    if !dataset.object_store.exists(&transaction_path).await? {
        return Ok(None);
    }
    let persisted = read_transaction_file(
        dataset.object_store.as_ref(),
        &dataset.base,
        &transaction_file,
    )
    .await?;
    if !same_transaction_request(requested, &persisted) {
        return Err(Error::invalid_input(format!(
            "transaction UUID {} already names a different source-free overwrite request",
            requested.uuid
        )));
    }

    for version in (1..=dataset.version().version).rev() {
        let candidate = dataset.checkout_version(version).await?;
        if candidate.manifest.transaction_file.as_deref() == Some(transaction_file.as_str()) {
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

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
            session: None,
            detached: false,
            commit_config: Default::default(),
            affected_rows: None,
            transaction_properties: None,
            timeout: Some(DEFAULT_COMMIT_TIMEOUT),
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
        self.storage_format = Some(storage_format);
        self
    }

    /// Pass an object store to use.
    pub fn with_object_store(mut self, object_store: Arc<ObjectStore>) -> Self {
        self.object_store = Some(object_store);
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

    async fn execute_inner(self, mut transaction: Transaction) -> Result<Dataset> {
        let session = self
            .session
            .or_else(|| self.dest.dataset().map(|ds| ds.session.clone()))
            .unwrap_or_default();

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

        if let Some(dataset) = dest.dataset()
            && let Some(committed) =
                reconcile_v2_3_sentinel_overwrite_retry(dataset, &transaction).await?
        {
            return Ok(committed);
        }

        prepare_v2_3_transaction(&mut transaction, dest.dataset(), self.storage_format).await?;

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

        let use_stable_row_ids = if let Some(ds) = dest.dataset() {
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
            ..Default::default()
        };

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
            WriteDestination::Dataset(dataset) => Ok(Dataset {
                manifest: Arc::new(manifest),
                manifest_location,
                session,
                fragment_bitmap,
                row_address_router: Arc::new(std::sync::OnceLock::new()),
                explicit_row_address_cache: Arc::new(Default::default()),
                ..dataset.as_ref().clone()
            }),
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
                    row_address_router: Arc::new(std::sync::OnceLock::new()),
                    explicit_row_address_cache: Arc::new(Default::default()),
                    metadata_cache,
                    file_reader_options: None,
                    store_params: self.store_params.clone().map(Box::new),
                    base_store_params: None,
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
        let row_address_layout_delta = merge_batch_append_row_address_deltas(&transactions)?;

        let mut batch_request_name = b"lance-batch-commit-v1\0".to_vec();
        for transaction in &transactions {
            batch_request_name.extend_from_slice(
                &u64::try_from(transaction.uuid.len())
                    .map_err(|_| {
                        Error::format_capacity_exceeded("batch transaction UUID is too long")
                    })?
                    .to_le_bytes(),
            );
            batch_request_name.extend_from_slice(transaction.uuid.as_bytes());
        }
        let merged = Transaction {
            uuid: uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, &batch_request_name)
                .hyphenated()
                .to_string(),
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
            row_address_layout_delta,
            original_request_fingerprint: None,
        };
        let requested_uuid = merged.uuid.clone();
        let dataset = self.execute(merged).await?;
        let merged = dataset
            .read_transaction()
            .await?
            .filter(|transaction| transaction.uuid == requested_uuid)
            .ok_or_else(|| {
                Error::internal(
                    "batch commit did not persist the canonical merged transaction metadata",
                )
            })?;
        Ok(BatchCommitResult { dataset, merged })
    }
}

fn merge_batch_append_row_address_deltas(
    transactions: &[Transaction],
) -> Result<Option<RowAddressLayoutDelta>> {
    let delta_count = transactions
        .iter()
        .filter(|transaction| transaction.row_address_layout_delta.is_some())
        .count();
    if delta_count == 0 {
        return Ok(None);
    }
    if delta_count != transactions.len() {
        return Err(Error::invalid_input(
            "batch append cannot mix storage-version-2.3 and legacy transactions",
        ));
    }

    let mut merged = RowAddressLayoutDelta::default();
    let mut ordinal_base = 0_u32;
    for transaction in transactions {
        let Operation::Append { fragments } = &transaction.operation else {
            unreachable!("batch append operations were validated by the caller");
        };
        let delta = transaction
            .row_address_layout_delta
            .as_ref()
            .expect("delta presence was validated above");
        if delta.create_namespace_uuid.is_some()
            || !delta.source_domains.is_empty()
            || !delta.retired_selections.is_empty()
            || !delta.field_changes.is_empty()
            || !delta.source_floors.is_empty()
            || !delta.replaced_generations.is_empty()
            || !delta.row_aligned_rewrite_proofs.is_empty()
            || !delta.explicit_map_placements.is_empty()
        {
            return Err(Error::invalid_input(
                "batch append requires source-free successor Direct row-address deltas",
            ));
        }
        if merged.expected_layout_fingerprint.is_empty() {
            merged.expected_layout_fingerprint = delta.expected_layout_fingerprint.clone();
        }
        if delta.expected_layout_fingerprint.is_empty() {
            return Err(Error::invalid_input(
                "storage-version-2.3 batch append transaction is missing its expected layout fingerprint",
            ));
        }

        let fragment_count = checked_batch_append_fragment_count(fragments.len())?;
        let next_ordinal_base = checked_batch_append_ordinal(ordinal_base, fragment_count)?;
        if delta.placements.len() != fragments.len() {
            return Err(Error::invalid_input(
                "batch append Direct placement count does not match fragment count",
            ));
        }
        for placement in &delta.placements {
            if placement.placement_kind != RowAddressPlacementKind::Direct
                || !placement.source_selections.is_empty()
                || !placement.output_row_sequence_fingerprint.is_empty()
            {
                return Err(Error::invalid_input(
                    "batch append only accepts source-free Direct placements",
                ));
            }
            let RowAddressTargetFragment::NewFragmentOrdinal(ordinal) = placement.target.fragment
            else {
                return Err(Error::invalid_input(
                    "batch append Direct placement must target a new fragment ordinal",
                ));
            };
            if ordinal >= fragment_count {
                return Err(Error::invalid_input(
                    "batch append Direct placement references a missing transaction fragment",
                ));
            }
            let mut placement = placement.clone();
            placement.target.fragment = RowAddressTargetFragment::NewFragmentOrdinal(
                checked_batch_append_ordinal(ordinal_base, ordinal)?,
            );
            merged.placements.push(placement);
        }
        ordinal_base = next_ordinal_base;
    }
    Ok(Some(merged))
}

fn checked_batch_append_fragment_count(fragment_count: usize) -> Result<u32> {
    u32::try_from(fragment_count).map_err(|_| {
        Error::format_capacity_exceeded(format!(
            "batch append transaction fragment count {fragment_count} exceeds u32 capacity"
        ))
    })
}

fn checked_batch_append_ordinal(ordinal_base: u32, ordinal: u32) -> Result<u32> {
    ordinal_base.checked_add(ordinal).ok_or_else(|| {
        Error::format_capacity_exceeded(format!(
            "batch append fragment ordinal {ordinal_base} + {ordinal} exceeds u32 capacity"
        ))
    })
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

    use lance_core::datatypes::Schema;
    use lance_core::error::FormatCapacityExceeded;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_io::utils::CachedFileSize;
    use lance_io::{assert_io_eq, assert_io_gt};
    use lance_table::format::{
        DataFile, Fragment, RowAddressPlacementDelta, RowAddressTargetRange,
    };
    use std::time::Duration;

    use object_store::throttle::ThrottleConfig;

    use crate::utils::test::ThrottledStoreWrapper;

    use crate::dataset::{InsertBuilder, TransactionBuilder, WriteParams};

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

    fn sample_fragment() -> Fragment {
        let (major_version, minor_version) = LanceFileVersion::Stable.to_numbers();
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
            deletion_file: None,
            row_id_meta: None,
            physical_rows: Some(10),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            native_logical_domain: None,
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
            row_address_layout_delta: None,
            original_request_fingerprint: None,
        }
    }

    fn source_free_v2_3_fragment(path: &str, physical_rows: usize) -> Fragment {
        let mut fragment = sample_fragment().with_physical_rows(physical_rows);
        fragment.files[0].path = path.to_owned();
        let (major, minor) = LanceFileVersion::V2_3.to_numbers();
        fragment.files[0].file_major_version = major;
        fragment.files[0].file_minor_version = minor;
        fragment
    }

    #[tokio::test]
    async fn v2_3_source_free_create_uses_transaction_uuid_as_namespace() {
        let namespace_uuid = Uuid::new_v4();
        let arrow_schema = ArrowSchema::new(vec![ArrowField::new("i", DataType::Int32, false)]);
        let mut fragment = sample_fragment().with_physical_rows(7);
        let (major, minor) = LanceFileVersion::V2_3.to_numbers();
        fragment.files[0].file_major_version = major;
        fragment.files[0].file_minor_version = minor;
        let mut transaction = TransactionBuilder::new(
            0,
            Operation::Overwrite {
                fragments: vec![fragment],
                schema: Schema::try_from(&arrow_schema).unwrap(),
                config_upsert_values: None,
                initial_bases: None,
            },
        )
        .uuid(namespace_uuid.to_string())
        .build();

        prepare_v2_3_transaction(&mut transaction, None, Some(LanceFileVersion::V2_3))
            .await
            .unwrap();

        let delta = transaction.row_address_layout_delta.unwrap();
        assert_eq!(delta.create_namespace_uuid, Some(namespace_uuid));
        assert_eq!(delta.placements.len(), 1);
        assert_eq!(delta.placements[0].output_cardinality, 7);
        assert_eq!(
            delta.placements[0].target.fragment,
            RowAddressTargetFragment::NewFragmentOrdinal(0)
        );
    }

    #[tokio::test]
    async fn v2_3_source_free_overwrite_and_batch_retries_are_idempotent() {
        let uri = TempStrDir::default();
        let arrow_schema = ArrowSchema::new(vec![ArrowField::new("i", DataType::Int32, false)]);
        let schema = Schema::try_from(&arrow_schema).unwrap();
        let create = TransactionBuilder::new(
            0,
            Operation::Overwrite {
                fragments: vec![source_free_v2_3_fragment("create.lance", 3)],
                schema: schema.clone(),
                config_upsert_values: None,
                initial_bases: None,
            },
        )
        .build();

        let created = CommitBuilder::new(uri.as_str())
            .with_storage_format(LanceFileVersion::V2_3)
            .execute(create.clone())
            .await
            .unwrap();
        let create_retry = CommitBuilder::new(uri.as_str())
            .with_storage_format(LanceFileVersion::V2_3)
            .execute(create)
            .await
            .unwrap();
        assert_eq!(created.version_id(), 1);
        assert_eq!(create_retry.version_id(), created.version_id());

        let overwrite = TransactionBuilder::new(
            0,
            Operation::Overwrite {
                fragments: vec![source_free_v2_3_fragment("overwrite.lance", 5)],
                schema,
                config_upsert_values: None,
                initial_bases: None,
            },
        )
        .build();
        let overwritten = CommitBuilder::new(uri.as_str())
            .execute(overwrite.clone())
            .await
            .unwrap();
        let overwrite_retry = CommitBuilder::new(uri.as_str())
            .execute(overwrite)
            .await
            .unwrap();
        assert_eq!(overwritten.version_id(), 2);
        assert_eq!(overwrite_retry.version_id(), overwritten.version_id());

        let append = |path: &str| {
            TransactionBuilder::new(
                overwritten.version_id(),
                Operation::Append {
                    fragments: vec![source_free_v2_3_fragment(path, 2)],
                },
            )
            .build()
        };
        let batch = vec![append("batch-1.lance"), append("batch-2.lance")];
        let first = CommitBuilder::new(Arc::new(overwritten.clone()))
            .execute_batch(batch.clone())
            .await
            .unwrap();
        let retry = CommitBuilder::new(Arc::new(overwritten))
            .execute_batch(batch.clone())
            .await
            .unwrap();
        let uri_retry = CommitBuilder::new(uri.as_str())
            .execute_batch(batch)
            .await
            .unwrap();
        assert_eq!(first.dataset.version_id(), 3);
        assert_eq!(retry.dataset.version_id(), first.dataset.version_id());
        assert_eq!(retry.merged.uuid, first.merged.uuid);
        assert_eq!(uri_retry.dataset.version_id(), first.dataset.version_id());
        assert_eq!(uri_retry.merged.uuid, first.merged.uuid);
        assert!(first.merged.row_address_layout_delta.is_some());
        assert_eq!(retry.dataset.manifest.fragments.len(), 3);
    }

    #[tokio::test]
    async fn concurrent_source_free_create_with_same_uuid_converges() {
        let uri = TempStrDir::default();
        let arrow_schema = ArrowSchema::new(vec![ArrowField::new("i", DataType::Int32, false)]);
        let transaction = TransactionBuilder::new(
            0,
            Operation::Overwrite {
                fragments: vec![source_free_v2_3_fragment("create.lance", 3)],
                schema: Schema::try_from(&arrow_schema).unwrap(),
                config_upsert_values: None,
                initial_bases: None,
            },
        )
        .build();
        let first = CommitBuilder::new(uri.as_str())
            .with_storage_format(LanceFileVersion::V2_3)
            .execute(transaction.clone());
        let second = CommitBuilder::new(uri.as_str())
            .with_storage_format(LanceFileVersion::V2_3)
            .execute(transaction);

        let (first, second) = tokio::join!(first, second);
        let first = first.unwrap();
        let second = second.unwrap();
        assert_eq!(first.version_id(), 1);
        assert_eq!(second.version_id(), first.version_id());
        assert_eq!(
            second
                .manifest
                .row_address_layout
                .as_ref()
                .unwrap()
                .namespace_uuid,
            first
                .manifest
                .row_address_layout
                .as_ref()
                .unwrap()
                .namespace_uuid
        );
    }

    fn sample_v2_3_append_transaction(read_version: u64, row_counts: &[u32]) -> Transaction {
        let mut transaction = sample_transaction(read_version);
        transaction.operation = Operation::Append {
            fragments: row_counts
                .iter()
                .map(|row_count| sample_fragment().with_physical_rows(*row_count as usize))
                .collect(),
        };
        transaction.row_address_layout_delta = Some(RowAddressLayoutDelta {
            expected_layout_fingerprint: vec![read_version as u8; 16],
            placements: row_counts
                .iter()
                .enumerate()
                .map(|(ordinal, row_count)| RowAddressPlacementDelta {
                    source_selections: Vec::new(),
                    target: RowAddressTargetRange {
                        fragment: RowAddressTargetFragment::NewFragmentOrdinal(ordinal as u32),
                        start_offset: 0,
                        end_offset: *row_count,
                    },
                    placement_kind: RowAddressPlacementKind::Direct,
                    output_cardinality: *row_count as u64,
                    output_row_sequence_fingerprint: Vec::new(),
                })
                .collect(),
            ..Default::default()
        });
        transaction
    }

    #[test]
    fn batch_append_rebases_v2_3_fragment_ordinals() {
        let first = sample_v2_3_append_transaction(4, &[3, 5]);
        let second = sample_v2_3_append_transaction(6, &[7]);

        let merged = merge_batch_append_row_address_deltas(&[first, second])
            .unwrap()
            .unwrap();

        assert_eq!(
            merged
                .placements
                .iter()
                .map(|placement| placement.target.fragment)
                .collect::<Vec<_>>(),
            vec![
                RowAddressTargetFragment::NewFragmentOrdinal(0),
                RowAddressTargetFragment::NewFragmentOrdinal(1),
                RowAddressTargetFragment::NewFragmentOrdinal(2),
            ]
        );
        assert_eq!(merged.expected_layout_fingerprint, vec![4; 16]);
    }

    #[test]
    fn batch_append_rejects_mixed_row_address_contracts() {
        let legacy = sample_transaction(4);
        let logical = sample_v2_3_append_transaction(4, &[3]);
        assert!(merge_batch_append_row_address_deltas(&[legacy, logical]).is_err());
    }

    #[test]
    fn batch_append_ordinal_overflow_is_typed() {
        assert_format_capacity(&checked_batch_append_ordinal(u32::MAX, 1).unwrap_err());
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn batch_append_fragment_count_overflow_is_typed() {
        assert_format_capacity(
            &checked_batch_append_fragment_count(u32::MAX as usize + 1).unwrap_err(),
        );
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
                merged_generations: Vec::new(),
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: None,
                inserted_rows_filter: None,
                updated_fragment_offsets: None,
            },
            read_version: 1,
            tag: None,
            transaction_properties: None,
            row_address_layout_delta: None,
            original_request_fingerprint: None,
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
