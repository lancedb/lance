// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Directory-based Lance Namespace implementation.
//!
//! This module provides a directory-based implementation of the Lance namespace
//! that stores tables as Lance datasets in a filesystem directory structure.

pub mod manifest;

use arrow::record_batch::RecordBatchIterator;
use arrow_ipc::reader::StreamReader;
use async_trait::async_trait;
use bytes::Bytes;
use lance::dataset::{Dataset, WriteParams};
use lance::session::Session;
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use object_store::path::Path;
use object_store::{Error as ObjectStoreError, ObjectStore as OSObjectStore, PutMode, PutOptions};
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

use lance_namespace::models::{
    CreateEmptyTableRequest, CreateEmptyTableResponse, CreateNamespaceRequest,
    CreateNamespaceResponse, CreateTableRequest, CreateTableResponse, DeclareTableRequest,
    DeclareTableResponse, DescribeNamespaceRequest, DescribeNamespaceResponse,
    DescribeTableRequest, DescribeTableResponse, DropNamespaceRequest, DropNamespaceResponse,
    DropTableRequest, DropTableResponse, ListNamespacesRequest, ListNamespacesResponse,
    ListTablesRequest, ListTablesResponse, NamespaceExistsRequest, TableExistsRequest,
};

use lance_core::{box_error, Error, Result};
use lance_namespace::schema::arrow_schema_to_json;
use lance_namespace::LanceNamespace;

use crate::credentials::{
    create_credential_vendor_for_location, has_credential_vendor_config, CredentialVendor,
};

/// Result of checking table status atomically.
///
/// This struct captures the state of a table directory in a single snapshot,
/// avoiding race conditions between checking existence and other status flags.
pub(crate) struct TableStatus {
    /// Whether the table directory exists (has any files)
    pub(crate) exists: bool,
    /// Whether the table has a `.lance-deregistered` marker file
    pub(crate) is_deregistered: bool,
    /// Whether the table has a `.lance-reserved` marker file (declared but not written)
    pub(crate) has_reserved_file: bool,
}

/// Builder for creating a DirectoryNamespace.
///
/// This builder provides a fluent API for configuring and establishing
/// connections to directory-based Lance namespaces.
///
/// # Examples
///
/// ```no_run
/// # use lance_namespace_impls::DirectoryNamespaceBuilder;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a local directory namespace
/// let namespace = DirectoryNamespaceBuilder::new("/path/to/data")
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
///
/// ```no_run
/// # use lance_namespace_impls::DirectoryNamespaceBuilder;
/// # use lance::session::Session;
/// # use std::sync::Arc;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Create with custom storage options and session
/// let session = Arc::new(Session::default());
/// let namespace = DirectoryNamespaceBuilder::new("s3://bucket/path")
///     .storage_option("region", "us-west-2")
///     .storage_option("access_key_id", "key")
///     .session(session)
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct DirectoryNamespaceBuilder {
    root: String,
    storage_options: Option<HashMap<String, String>>,
    session: Option<Arc<Session>>,
    manifest_enabled: bool,
    dir_listing_enabled: bool,
    inline_optimization_enabled: bool,
    credential_vendor_properties: HashMap<String, String>,
}

impl DirectoryNamespaceBuilder {
    /// Create a new DirectoryNamespaceBuilder with the specified root path.
    ///
    /// # Arguments
    ///
    /// * `root` - Root directory path (local path or cloud URI like s3://bucket/path)
    pub fn new(root: impl Into<String>) -> Self {
        Self {
            root: root.into().trim_end_matches('/').to_string(),
            storage_options: None,
            session: None,
            manifest_enabled: true,
            dir_listing_enabled: true, // Default to enabled for backwards compatibility
            inline_optimization_enabled: true,
            credential_vendor_properties: HashMap::new(),
        }
    }

    /// Enable or disable manifest-based listing.
    ///
    /// When enabled (default), the namespace uses a `__manifest` table to track tables.
    /// When disabled, relies solely on directory scanning.
    pub fn manifest_enabled(mut self, enabled: bool) -> Self {
        self.manifest_enabled = enabled;
        self
    }

    /// Enable or disable directory-based listing fallback.
    ///
    /// When enabled (default), falls back to directory scanning for tables not in the manifest.
    /// When disabled, only consults the manifest table.
    pub fn dir_listing_enabled(mut self, enabled: bool) -> Self {
        self.dir_listing_enabled = enabled;
        self
    }

    /// Enable or disable inline optimization of the __manifest table.
    ///
    /// When enabled (default), performs compaction and indexing on the __manifest table
    /// after every write operation to maintain optimal performance.
    /// When disabled, manual optimization must be performed separately.
    pub fn inline_optimization_enabled(mut self, enabled: bool) -> Self {
        self.inline_optimization_enabled = enabled;
        self
    }

    /// Create a DirectoryNamespaceBuilder from properties HashMap.
    ///
    /// This method parses a properties map into builder configuration.
    /// It expects:
    /// - `root`: The root directory path (required)
    /// - `manifest_enabled`: Enable manifest-based table tracking (optional, default: true)
    /// - `dir_listing_enabled`: Enable directory listing for table discovery (optional, default: true)
    /// - `inline_optimization_enabled`: Enable inline optimization of __manifest table (optional, default: true)
    /// - `storage.*`: Storage options (optional, prefix will be stripped)
    ///
    /// Credential vendor properties (prefixed with `credential_vendor.`, prefix is stripped):
    /// - `credential_vendor.enabled`: Set to "true" to enable credential vending (required)
    /// - `credential_vendor.permission`: Permission level: read, write, or admin (default: read)
    ///
    /// AWS-specific properties (for s3:// locations):
    /// - `credential_vendor.aws_role_arn`: AWS IAM role ARN (required for AWS)
    /// - `credential_vendor.aws_external_id`: AWS external ID (optional)
    /// - `credential_vendor.aws_region`: AWS region (optional)
    /// - `credential_vendor.aws_role_session_name`: AWS role session name (optional)
    /// - `credential_vendor.aws_duration_millis`: Credential duration in ms (default: 3600000, range: 15min-12hrs)
    ///
    /// GCP-specific properties (for gs:// locations):
    /// - `credential_vendor.gcp_service_account`: Service account to impersonate (optional)
    ///
    /// Note: GCP uses Application Default Credentials (ADC). To use a service account key file,
    /// set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable before starting.
    /// GCP token duration cannot be configured; it's determined by the STS endpoint (typically 1 hour).
    ///
    /// Azure-specific properties (for az:// locations):
    /// - `credential_vendor.azure_account_name`: Azure storage account name (required for Azure)
    /// - `credential_vendor.azure_tenant_id`: Azure tenant ID (optional)
    /// - `credential_vendor.azure_duration_millis`: Credential duration in ms (default: 3600000, up to 7 days)
    ///
    /// # Arguments
    ///
    /// * `properties` - Configuration properties
    /// * `session` - Optional Lance session to reuse object store registry
    ///
    /// # Returns
    ///
    /// Returns a `DirectoryNamespaceBuilder` instance.
    ///
    /// # Errors
    ///
    /// Returns an error if the `root` property is missing.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use lance_namespace_impls::DirectoryNamespaceBuilder;
    /// # use std::collections::HashMap;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut properties = HashMap::new();
    /// properties.insert("root".to_string(), "/path/to/data".to_string());
    /// properties.insert("manifest_enabled".to_string(), "true".to_string());
    /// properties.insert("dir_listing_enabled".to_string(), "false".to_string());
    /// properties.insert("storage.region".to_string(), "us-west-2".to_string());
    ///
    /// let namespace = DirectoryNamespaceBuilder::from_properties(properties, None)?
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_properties(
        properties: HashMap<String, String>,
        session: Option<Arc<Session>>,
    ) -> Result<Self> {
        // Extract root from properties (required)
        let root = properties
            .get("root")
            .cloned()
            .ok_or_else(|| Error::Namespace {
                source: "Missing required property 'root' for directory namespace".into(),
                location: snafu::location!(),
            })?;

        // Extract storage options (properties prefixed with "storage.")
        let storage_options: HashMap<String, String> = properties
            .iter()
            .filter_map(|(k, v)| {
                k.strip_prefix("storage.")
                    .map(|key| (key.to_string(), v.clone()))
            })
            .collect();

        let storage_options = if storage_options.is_empty() {
            None
        } else {
            Some(storage_options)
        };

        // Extract manifest_enabled (default: true)
        let manifest_enabled = properties
            .get("manifest_enabled")
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(true);

        // Extract dir_listing_enabled (default: true)
        let dir_listing_enabled = properties
            .get("dir_listing_enabled")
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(true);

        // Extract inline_optimization_enabled (default: true)
        let inline_optimization_enabled = properties
            .get("inline_optimization_enabled")
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(true);

        // Extract credential vendor properties (properties prefixed with "credential_vendor.")
        // The prefix is stripped to get short property names
        // The build() method will check if enabled=true before creating the vendor
        let credential_vendor_properties: HashMap<String, String> = properties
            .iter()
            .filter_map(|(k, v)| {
                k.strip_prefix("credential_vendor.")
                    .map(|key| (key.to_string(), v.clone()))
            })
            .collect();

        Ok(Self {
            root: root.trim_end_matches('/').to_string(),
            storage_options,
            session,
            manifest_enabled,
            dir_listing_enabled,
            inline_optimization_enabled,
            credential_vendor_properties,
        })
    }

    /// Add a storage option.
    ///
    /// # Arguments
    ///
    /// * `key` - Storage option key (e.g., "region", "access_key_id")
    /// * `value` - Storage option value
    pub fn storage_option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.storage_options
            .get_or_insert_with(HashMap::new)
            .insert(key.into(), value.into());
        self
    }

    /// Add multiple storage options.
    ///
    /// # Arguments
    ///
    /// * `options` - HashMap of storage options to add
    pub fn storage_options(mut self, options: HashMap<String, String>) -> Self {
        self.storage_options
            .get_or_insert_with(HashMap::new)
            .extend(options);
        self
    }

    /// Set the Lance session to use for this namespace.
    ///
    /// When a session is provided, the namespace will reuse the session's
    /// object store registry, allowing multiple namespaces and datasets
    /// to share the same underlying storage connections.
    ///
    /// # Arguments
    ///
    /// * `session` - Arc-wrapped Lance session
    pub fn session(mut self, session: Arc<Session>) -> Self {
        self.session = Some(session);
        self
    }

    /// Add a credential vendor property.
    ///
    /// Use short property names without the `credential_vendor.` prefix.
    /// Common properties: `enabled`, `permission`.
    /// AWS properties: `aws_role_arn`, `aws_external_id`, `aws_region`, `aws_role_session_name`, `aws_duration_millis`.
    /// GCP properties: `gcp_service_account`.
    /// Azure properties: `azure_account_name`, `azure_tenant_id`, `azure_duration_millis`.
    ///
    /// # Arguments
    ///
    /// * `key` - Property key (e.g., "enabled", "aws_role_arn")
    /// * `value` - Property value
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lance_namespace_impls::DirectoryNamespaceBuilder;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let namespace = DirectoryNamespaceBuilder::new("s3://my-bucket/data")
    ///     .credential_vendor_property("enabled", "true")
    ///     .credential_vendor_property("aws_role_arn", "arn:aws:iam::123456789012:role/MyRole")
    ///     .credential_vendor_property("permission", "read")
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn credential_vendor_property(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.credential_vendor_properties
            .insert(key.into(), value.into());
        self
    }

    /// Add multiple credential vendor properties.
    ///
    /// Use short property names without the `credential_vendor.` prefix.
    ///
    /// # Arguments
    ///
    /// * `properties` - HashMap of credential vendor properties to add
    pub fn credential_vendor_properties(mut self, properties: HashMap<String, String>) -> Self {
        self.credential_vendor_properties.extend(properties);
        self
    }

    /// Build the DirectoryNamespace.
    ///
    /// # Returns
    ///
    /// Returns a `DirectoryNamespace` instance.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The root path is invalid
    /// - Connection to the storage backend fails
    /// - Storage options are invalid
    pub async fn build(self) -> Result<DirectoryNamespace> {
        let (object_store, base_path) =
            Self::initialize_object_store(&self.root, &self.storage_options, &self.session).await?;

        let manifest_ns = if self.manifest_enabled {
            match manifest::ManifestNamespace::from_directory(
                self.root.clone(),
                self.storage_options.clone(),
                self.session.clone(),
                object_store.clone(),
                base_path.clone(),
                self.dir_listing_enabled,
                self.inline_optimization_enabled,
            )
            .await
            {
                Ok(ns) => Some(Arc::new(ns)),
                Err(e) => {
                    // Failed to initialize manifest namespace, fall back to directory listing only
                    log::warn!(
                        "Failed to initialize manifest namespace, falling back to directory listing only: {}",
                        e
                    );
                    None
                }
            }
        } else {
            None
        };

        // Create credential vendor once during initialization if enabled
        let credential_vendor = if has_credential_vendor_config(&self.credential_vendor_properties)
        {
            create_credential_vendor_for_location(&self.root, &self.credential_vendor_properties)
                .await?
                .map(Arc::from)
        } else {
            None
        };

        Ok(DirectoryNamespace {
            root: self.root,
            storage_options: self.storage_options,
            session: self.session,
            object_store,
            base_path,
            manifest_ns,
            dir_listing_enabled: self.dir_listing_enabled,
            credential_vendor,
        })
    }

    /// Initialize the Lance ObjectStore based on the configuration
    async fn initialize_object_store(
        root: &str,
        storage_options: &Option<HashMap<String, String>>,
        session: &Option<Arc<Session>>,
    ) -> Result<(Arc<ObjectStore>, Path)> {
        // Build ObjectStoreParams from storage options
        let params = ObjectStoreParams {
            storage_options: storage_options.clone(),
            ..Default::default()
        };

        // Use object store registry from session if provided, otherwise create a new one
        let registry = if let Some(session) = session {
            session.store_registry()
        } else {
            Arc::new(ObjectStoreRegistry::default())
        };

        // Use Lance's object store factory to create from URI
        let (object_store, base_path) = ObjectStore::from_uri_and_params(registry, root, &params)
            .await
            .map_err(|e| Error::Namespace {
                source: format!("Failed to create object store: {}", e).into(),
                location: snafu::location!(),
            })?;

        Ok((object_store, base_path))
    }
}

/// Directory-based implementation of Lance Namespace.
///
/// This implementation stores tables as Lance datasets in a directory structure.
/// It supports local filesystems and cloud storage backends through Lance's object store.
///
/// ## Manifest-based Listing
///
/// When `manifest_enabled=true`, the namespace uses a special `__manifest` Lance table to track tables
/// instead of scanning the filesystem. This provides:
/// - Better performance for listing operations
/// - Ability to track table metadata
/// - Foundation for future features like namespaces and table renaming
///
/// When `dir_listing_enabled=true`, the namespace falls back to directory scanning for tables not
/// found in the manifest, enabling gradual migration.
///
/// ## Credential Vending
///
/// When credential vendor properties are configured, `describe_table` will vend temporary
/// credentials based on the table location URI. The vendor type is auto-selected:
/// - `s3://` locations use AWS STS AssumeRole
/// - `gs://` locations use GCP OAuth2 tokens
/// - `az://` locations use Azure SAS tokens
pub struct DirectoryNamespace {
    root: String,
    storage_options: Option<HashMap<String, String>>,
    #[allow(dead_code)]
    session: Option<Arc<Session>>,
    object_store: Arc<ObjectStore>,
    base_path: Path,
    manifest_ns: Option<Arc<manifest::ManifestNamespace>>,
    dir_listing_enabled: bool,
    /// Credential vendor created once during initialization.
    /// Used to vend temporary credentials for table access.
    credential_vendor: Option<Arc<dyn CredentialVendor>>,
}

impl std::fmt::Debug for DirectoryNamespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.namespace_id())
    }
}

impl std::fmt::Display for DirectoryNamespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.namespace_id())
    }
}

impl DirectoryNamespace {
    /// Apply pagination to a list of table names
    ///
    /// Sorts the list alphabetically and applies pagination using page_token (start_after) and limit.
    ///
    /// # Arguments
    /// * `names` - The vector of table names to paginate
    /// * `page_token` - Skip items until finding one greater than this value (start_after semantics)
    /// * `limit` - Maximum number of items to keep
    fn apply_pagination(names: &mut Vec<String>, page_token: Option<String>, limit: Option<i32>) {
        // Sort alphabetically for consistent ordering
        names.sort();

        // Apply page_token filtering (start_after semantics)
        if let Some(start_after) = page_token {
            if let Some(index) = names
                .iter()
                .position(|name| name.as_str() > start_after.as_str())
            {
                names.drain(0..index);
            } else {
                names.clear();
            }
        }

        // Apply limit
        if let Some(limit) = limit {
            if limit >= 0 {
                names.truncate(limit as usize);
            }
        }
    }

    /// List tables using directory scanning (fallback method)
    async fn list_directory_tables(&self) -> Result<Vec<String>> {
        let mut tables = Vec::new();
        let entries = self
            .object_store
            .read_dir(self.base_path.clone())
            .await
            .map_err(|e| Error::IO {
                source: box_error(std::io::Error::other(format!(
                    "Failed to list directory: {}",
                    e
                ))),
                location: snafu::location!(),
            })?;

        for entry in entries {
            let path = entry.trim_end_matches('/');
            if !path.ends_with(".lance") {
                continue;
            }

            let table_name = &path[..path.len() - 6];

            // Use atomic check to skip deregistered tables and declared-but-not-written tables
            let status = self.check_table_status(table_name).await;
            if status.is_deregistered || status.has_reserved_file {
                continue;
            }

            tables.push(table_name.to_string());
        }

        Ok(tables)
    }

    /// Validate that the namespace ID represents the root namespace
    fn validate_root_namespace_id(id: &Option<Vec<String>>) -> Result<()> {
        if let Some(id) = id {
            if !id.is_empty() {
                return Err(Error::Namespace {
                    source: format!(
                        "Directory namespace only supports root namespace operations, but got namespace ID: {:?}. Expected empty ID.",
                        id
                    ).into(),
                    location: snafu::location!(),
                });
            }
        }
        Ok(())
    }

    /// Extract table name from table ID
    fn table_name_from_id(id: &Option<Vec<String>>) -> Result<String> {
        let id = id.as_ref().ok_or_else(|| Error::Namespace {
            source: "Directory namespace table ID cannot be empty".into(),
            location: snafu::location!(),
        })?;

        if id.len() != 1 {
            return Err(Error::Namespace {
                source: format!(
                    "Multi-level table IDs are only supported when manifest mode is enabled, but got: {:?}",
                    id
                )
                .into(),
                location: snafu::location!(),
            });
        }

        Ok(id[0].clone())
    }

    /// Get the full URI path for a table (for returning in responses)
    fn table_full_uri(&self, table_name: &str) -> String {
        format!("{}/{}.lance", &self.root, table_name)
    }

    /// Get the object store path for a table (relative to base_path)
    fn table_path(&self, table_name: &str) -> Path {
        self.base_path
            .child(format!("{}.lance", table_name).as_str())
    }

    /// Get the reserved file path for a table
    fn table_reserved_file_path(&self, table_name: &str) -> Path {
        self.base_path
            .child(format!("{}.lance", table_name).as_str())
            .child(".lance-reserved")
    }

    /// Get the deregistered marker file path for a table
    fn table_deregistered_file_path(&self, table_name: &str) -> Path {
        self.base_path
            .child(format!("{}.lance", table_name).as_str())
            .child(".lance-deregistered")
    }

    /// Atomically check table existence and deregistration status.
    ///
    /// This performs a single directory listing to get a consistent snapshot of the
    /// table's state, avoiding race conditions between checking existence and
    /// checking deregistration status.
    pub(crate) async fn check_table_status(&self, table_name: &str) -> TableStatus {
        let table_path = self.table_path(table_name);
        match self.object_store.read_dir(table_path).await {
            Ok(entries) => {
                let exists = !entries.is_empty();
                let is_deregistered = entries.iter().any(|e| e.ends_with(".lance-deregistered"));
                let has_reserved_file = entries.iter().any(|e| e.ends_with(".lance-reserved"));
                TableStatus {
                    exists,
                    is_deregistered,
                    has_reserved_file,
                }
            }
            Err(_) => TableStatus {
                exists: false,
                is_deregistered: false,
                has_reserved_file: false,
            },
        }
    }

    /// Atomically create a marker file using put_if_not_exists semantics.
    ///
    /// This uses `PutMode::Create` which will fail if the file already exists,
    /// providing atomic creation semantics to avoid race conditions.
    ///
    /// Returns Ok(()) if the file was created successfully.
    /// Returns Err with appropriate message if the file already exists or other error.
    async fn put_marker_file_atomic(
        &self,
        path: &Path,
        file_description: &str,
    ) -> std::result::Result<(), String> {
        let put_opts = PutOptions {
            mode: PutMode::Create,
            ..Default::default()
        };

        match self
            .object_store
            .inner
            .put_opts(path, bytes::Bytes::new().into(), put_opts)
            .await
        {
            Ok(_) => Ok(()),
            Err(ObjectStoreError::AlreadyExists { .. })
            | Err(ObjectStoreError::Precondition { .. }) => {
                Err(format!("{} already exists", file_description))
            }
            Err(e) => Err(format!("Failed to create {}: {}", file_description, e)),
        }
    }

    /// Get storage options for a table, using credential vending if configured.
    ///
    /// If credential vendor properties are configured and the table location matches
    /// a supported cloud provider, this will create an appropriate vendor and vend
    /// temporary credentials scoped to the table location. Otherwise, returns the
    /// static storage options.
    ///
    /// The vendor type is auto-selected based on the table URI:
    /// - `s3://` locations use AWS STS AssumeRole
    /// - `gs://` locations use GCP OAuth2 tokens
    /// - `az://` locations use Azure SAS tokens
    ///
    /// The permission level (Read, Write, Admin) is configured at namespace
    /// initialization time via the `credential_vendor_permission` property.
    ///
    /// # Arguments
    ///
    /// * `table_uri` - The full URI of the table
    async fn get_storage_options_for_table(
        &self,
        table_uri: &str,
    ) -> Result<Option<HashMap<String, String>>> {
        if let Some(ref vendor) = self.credential_vendor {
            let vended = vendor.vend_credentials(table_uri).await?;
            return Ok(Some(vended.storage_options));
        }
        Ok(self.storage_options.clone())
    }

    /// Migrate directory-based tables to the manifest.
    ///
    /// This is a one-time migration operation that:
    /// 1. Scans the directory for existing `.lance` tables
    /// 2. Registers any unmigrated tables in the manifest
    /// 3. Returns the count of tables that were migrated
    ///
    /// This method is safe to run multiple times - it will skip tables that are already
    /// registered in the manifest.
    ///
    /// # Usage
    ///
    /// After creating tables in directory-only mode or dual mode, you can migrate them
    /// to the manifest to enable manifest-only mode:
    ///
    /// ```no_run
    /// # use lance_namespace_impls::DirectoryNamespaceBuilder;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// // Create namespace with dual mode (manifest + directory listing)
    /// let namespace = DirectoryNamespaceBuilder::new("/path/to/data")
    ///     .manifest_enabled(true)
    ///     .dir_listing_enabled(true)
    ///     .build()
    ///     .await?;
    ///
    /// // ... tables are created and used ...
    ///
    /// // Migrate existing directory tables to manifest
    /// let migrated_count = namespace.migrate().await?;
    /// println!("Migrated {} tables", migrated_count);
    ///
    /// // Now you can disable directory listing for better performance:
    /// // (requires rebuilding the namespace)
    /// let namespace = DirectoryNamespaceBuilder::new("/path/to/data")
    ///     .manifest_enabled(true)
    ///     .dir_listing_enabled(false)  // All tables now in manifest
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Returns
    ///
    /// Returns the number of tables that were migrated to the manifest.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Manifest is not enabled
    /// - Directory listing fails
    /// - Manifest registration fails
    pub async fn migrate(&self) -> Result<usize> {
        // We only care about tables in the root namespace
        let Some(ref manifest_ns) = self.manifest_ns else {
            return Ok(0); // No manifest, nothing to migrate
        };

        // Get all table locations already in the manifest
        let manifest_locations = manifest_ns.list_manifest_table_locations().await?;

        // Get all tables from directory
        let dir_tables = self.list_directory_tables().await?;

        // Register each directory table that doesn't have an overlapping location
        // If a directory name already exists in the manifest,
        // that means the table must have already been migrated or created
        // in the manifest, so we can skip it.
        let mut migrated_count = 0;
        for table_name in dir_tables {
            // For root namespace tables, the directory name is "table_name.lance"
            let dir_name = format!("{}.lance", table_name);
            if !manifest_locations.contains(&dir_name) {
                manifest_ns.register_table(&table_name, dir_name).await?;
                migrated_count += 1;
            }
        }

        Ok(migrated_count)
    }
}

#[async_trait]
impl LanceNamespace for DirectoryNamespace {
    async fn list_namespaces(
        &self,
        request: ListNamespacesRequest,
    ) -> Result<ListNamespacesResponse> {
        if let Some(ref manifest_ns) = self.manifest_ns {
            return manifest_ns.list_namespaces(request).await;
        }

        Self::validate_root_namespace_id(&request.id)?;
        Ok(ListNamespacesResponse::new(vec![]))
    }

    async fn describe_namespace(
        &self,
        request: DescribeNamespaceRequest,
    ) -> Result<DescribeNamespaceResponse> {
        if let Some(ref manifest_ns) = self.manifest_ns {
            return manifest_ns.describe_namespace(request).await;
        }

        Self::validate_root_namespace_id(&request.id)?;
        Ok(DescribeNamespaceResponse {
            properties: Some(HashMap::new()),
        })
    }

    async fn create_namespace(
        &self,
        request: CreateNamespaceRequest,
    ) -> Result<CreateNamespaceResponse> {
        if let Some(ref manifest_ns) = self.manifest_ns {
            return manifest_ns.create_namespace(request).await;
        }

        if request.id.is_none() || request.id.as_ref().unwrap().is_empty() {
            return Err(Error::Namespace {
                source: "Root namespace already exists and cannot be created".into(),
                location: snafu::location!(),
            });
        }

        Err(Error::NotSupported {
            source: "Child namespaces are only supported when manifest mode is enabled".into(),
            location: snafu::location!(),
        })
    }

    async fn drop_namespace(&self, request: DropNamespaceRequest) -> Result<DropNamespaceResponse> {
        if let Some(ref manifest_ns) = self.manifest_ns {
            return manifest_ns.drop_namespace(request).await;
        }

        if request.id.is_none() || request.id.as_ref().unwrap().is_empty() {
            return Err(Error::Namespace {
                source: "Root namespace cannot be dropped".into(),
                location: snafu::location!(),
            });
        }

        Err(Error::NotSupported {
            source: "Child namespaces are only supported when manifest mode is enabled".into(),
            location: snafu::location!(),
        })
    }

    async fn namespace_exists(&self, request: NamespaceExistsRequest) -> Result<()> {
        if let Some(ref manifest_ns) = self.manifest_ns {
            return manifest_ns.namespace_exists(request).await;
        }

        if request.id.is_none() || request.id.as_ref().unwrap().is_empty() {
            return Ok(());
        }

        Err(Error::Namespace {
            source: "Child namespaces are only supported when manifest mode is enabled".into(),
            location: snafu::location!(),
        })
    }

    async fn list_tables(&self, request: ListTablesRequest) -> Result<ListTablesResponse> {
        // Validate that namespace ID is provided
        let namespace_id = request.id.as_ref().ok_or_else(|| Error::InvalidInput {
            source: "Namespace ID is required".into(),
            location: snafu::location!(),
        })?;

        // For child namespaces, always delegate to manifest (if enabled)
        if !namespace_id.is_empty() {
            if let Some(ref manifest_ns) = self.manifest_ns {
                return manifest_ns.list_tables(request).await;
            }
            return Err(Error::NotSupported {
                source: "Child namespaces are only supported when manifest mode is enabled".into(),
                location: snafu::location!(),
            });
        }

        // When only manifest is enabled (no directory listing), delegate directly to manifest
        if let Some(ref manifest_ns) = self.manifest_ns {
            if !self.dir_listing_enabled {
                return manifest_ns.list_tables(request).await;
            }
        }

        // When both manifest and directory listing are enabled, we need to merge and deduplicate
        let mut tables = if self.manifest_ns.is_some() && self.dir_listing_enabled {
            // Get all manifest table locations (for deduplication)
            let manifest_locations = if let Some(ref manifest_ns) = self.manifest_ns {
                manifest_ns.list_manifest_table_locations().await?
            } else {
                std::collections::HashSet::new()
            };

            // Get all manifest tables (without pagination for merging)
            let mut manifest_request = request.clone();
            manifest_request.limit = None;
            manifest_request.page_token = None;
            let manifest_tables = if let Some(ref manifest_ns) = self.manifest_ns {
                let manifest_response = manifest_ns.list_tables(manifest_request).await?;
                manifest_response.tables
            } else {
                vec![]
            };

            // Start with all manifest table names
            // Add directory tables that aren't already in the manifest (by location)
            let mut all_tables: Vec<String> = manifest_tables;
            let dir_tables = self.list_directory_tables().await?;
            for table_name in dir_tables {
                // Check if this table's location is already in the manifest
                // Manifest stores full URIs, so we need to check both formats
                let full_location = format!("{}/{}.lance", self.root, table_name);
                let relative_location = format!("{}.lance", table_name);
                if !manifest_locations.contains(&full_location)
                    && !manifest_locations.contains(&relative_location)
                {
                    all_tables.push(table_name);
                }
            }

            all_tables
        } else {
            self.list_directory_tables().await?
        };

        // Apply sorting and pagination
        Self::apply_pagination(&mut tables, request.page_token, request.limit);
        let response = ListTablesResponse::new(tables);
        Ok(response)
    }

    async fn describe_table(&self, request: DescribeTableRequest) -> Result<DescribeTableResponse> {
        if let Some(ref manifest_ns) = self.manifest_ns {
            match manifest_ns.describe_table(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(_)
                    if self.dir_listing_enabled
                        && request.id.as_ref().is_some_and(|id| id.len() == 1) =>
                {
                    // Fall through to directory check only for single-level IDs
                }
                Err(e) => return Err(e),
            }
        }

        let table_name = Self::table_name_from_id(&request.id)?;
        let table_uri = self.table_full_uri(&table_name);

        // Atomically check table existence and deregistration status
        let status = self.check_table_status(&table_name).await;

        if !status.exists {
            return Err(Error::Namespace {
                source: format!("Table does not exist: {}", table_name).into(),
                location: snafu::location!(),
            });
        }

        if status.is_deregistered {
            return Err(Error::Namespace {
                source: format!("Table is deregistered: {}", table_name).into(),
                location: snafu::location!(),
            });
        }

        // Try to load the dataset to get real information
        match Dataset::open(&table_uri).await {
            Ok(mut dataset) => {
                // If a specific version is requested, checkout that version
                if let Some(requested_version) = request.version {
                    dataset = dataset.checkout_version(requested_version as u64).await?;
                }

                let version = dataset.version().version;
                let lance_schema = dataset.schema();
                let arrow_schema: arrow_schema::Schema = lance_schema.into();
                let json_schema = arrow_schema_to_json(&arrow_schema)?;
                let storage_options = self.get_storage_options_for_table(&table_uri).await?;

                Ok(DescribeTableResponse {
                    table: Some(table_name),
                    namespace: request.id.as_ref().map(|id| {
                        if id.len() > 1 {
                            id[..id.len() - 1].to_vec()
                        } else {
                            vec![]
                        }
                    }),
                    version: Some(version as i64),
                    location: Some(table_uri.clone()),
                    table_uri: Some(table_uri),
                    schema: Some(Box::new(json_schema)),
                    storage_options,
                    stats: None,
                })
            }
            Err(err) => {
                // Use the reserved file status from the atomic check
                if status.has_reserved_file {
                    let storage_options = self.get_storage_options_for_table(&table_uri).await?;
                    Ok(DescribeTableResponse {
                        table: Some(table_name),
                        namespace: request.id.as_ref().map(|id| {
                            if id.len() > 1 {
                                id[..id.len() - 1].to_vec()
                            } else {
                                vec![]
                            }
                        }),
                        version: None,
                        location: Some(table_uri.clone()),
                        table_uri: Some(table_uri),
                        schema: None,
                        storage_options,
                        stats: None,
                    })
                } else {
                    Err(Error::Namespace {
                        source: format!(
                            "Table directory exists but cannot load dataset {}: {:?}",
                            table_name, err
                        )
                        .into(),
                        location: snafu::location!(),
                    })
                }
            }
        }
    }

    async fn table_exists(&self, request: TableExistsRequest) -> Result<()> {
        if let Some(ref manifest_ns) = self.manifest_ns {
            match manifest_ns.table_exists(request.clone()).await {
                Ok(()) => return Ok(()),
                Err(_) if self.dir_listing_enabled => {
                    // Fall through to directory check
                }
                Err(e) => return Err(e),
            }
        }

        let table_name = Self::table_name_from_id(&request.id)?;

        // Atomically check table existence and deregistration status
        let status = self.check_table_status(&table_name).await;

        if !status.exists {
            return Err(Error::Namespace {
                source: format!("Table does not exist: {}", table_name).into(),
                location: snafu::location!(),
            });
        }

        if status.is_deregistered {
            return Err(Error::Namespace {
                source: format!("Table is deregistered: {}", table_name).into(),
                location: snafu::location!(),
            });
        }

        Ok(())
    }

    async fn drop_table(&self, request: DropTableRequest) -> Result<DropTableResponse> {
        if let Some(ref manifest_ns) = self.manifest_ns {
            return manifest_ns.drop_table(request).await;
        }

        let table_name = Self::table_name_from_id(&request.id)?;
        let table_uri = self.table_full_uri(&table_name);
        let table_path = self.table_path(&table_name);

        self.object_store
            .remove_dir_all(table_path)
            .await
            .map_err(|e| Error::Namespace {
                source: format!("Failed to drop table {}: {}", table_name, e).into(),
                location: snafu::location!(),
            })?;

        Ok(DropTableResponse {
            id: request.id,
            location: Some(table_uri),
            properties: None,
            transaction_id: None,
        })
    }

    async fn create_table(
        &self,
        request: CreateTableRequest,
        request_data: Bytes,
    ) -> Result<CreateTableResponse> {
        if let Some(ref manifest_ns) = self.manifest_ns {
            return manifest_ns.create_table(request, request_data).await;
        }

        let table_name = Self::table_name_from_id(&request.id)?;
        let table_uri = self.table_full_uri(&table_name);
        if request_data.is_empty() {
            return Err(Error::Namespace {
                source: "Request data (Arrow IPC stream) is required for create_table".into(),
                location: snafu::location!(),
            });
        }

        // Parse the Arrow IPC stream from request_data
        let cursor = Cursor::new(request_data.to_vec());
        let stream_reader = StreamReader::try_new(cursor, None).map_err(|e| Error::Namespace {
            source: format!("Invalid Arrow IPC stream: {}", e).into(),
            location: snafu::location!(),
        })?;
        let arrow_schema = stream_reader.schema();

        // Collect all batches from the stream
        let mut batches = Vec::new();
        for batch_result in stream_reader {
            batches.push(batch_result.map_err(|e| Error::Namespace {
                source: format!("Failed to read batch from IPC stream: {}", e).into(),
                location: snafu::location!(),
            })?);
        }

        // Create RecordBatchReader from the batches
        let reader = if batches.is_empty() {
            let batch = arrow::record_batch::RecordBatch::new_empty(arrow_schema.clone());
            let batches = vec![Ok(batch)];
            RecordBatchIterator::new(batches, arrow_schema.clone())
        } else {
            let batch_results: Vec<_> = batches.into_iter().map(Ok).collect();
            RecordBatchIterator::new(batch_results, arrow_schema)
        };

        let store_params = self.storage_options.as_ref().map(|opts| ObjectStoreParams {
            storage_options: Some(opts.clone()),
            ..Default::default()
        });

        let write_params = WriteParams {
            mode: lance::dataset::WriteMode::Create,
            store_params,
            ..Default::default()
        };

        // Create the Lance dataset using the actual Lance API
        Dataset::write(reader, &table_uri, Some(write_params))
            .await
            .map_err(|e| Error::Namespace {
                source: format!("Failed to create Lance dataset: {}", e).into(),
                location: snafu::location!(),
            })?;

        Ok(CreateTableResponse {
            transaction_id: None,
            version: Some(1),
            location: Some(table_uri),
            storage_options: self.storage_options.clone(),
        })
    }

    async fn create_empty_table(
        &self,
        request: CreateEmptyTableRequest,
    ) -> Result<CreateEmptyTableResponse> {
        if let Some(ref manifest_ns) = self.manifest_ns {
            #[allow(deprecated)]
            return manifest_ns.create_empty_table(request).await;
        }

        let table_name = Self::table_name_from_id(&request.id)?;
        let table_uri = self.table_full_uri(&table_name);

        // Validate location if provided
        if let Some(location) = &request.location {
            let location = location.trim_end_matches('/');
            if location != table_uri {
                return Err(Error::Namespace {
                    source: format!(
                        "Cannot create table {} at location {}, must be at location {}",
                        table_name, location, table_uri
                    )
                    .into(),
                    location: snafu::location!(),
                });
            }
        }

        // Atomically create the .lance-reserved file to mark the table as existing.
        // This uses put_if_not_exists semantics to avoid race conditions.
        let reserved_file_path = self.table_reserved_file_path(&table_name);

        self.put_marker_file_atomic(&reserved_file_path, &format!("table {}", table_name))
            .await
            .map_err(|e| Error::Namespace {
                source: e.into(),
                location: snafu::location!(),
            })?;

        Ok(CreateEmptyTableResponse {
            transaction_id: None,
            location: Some(table_uri),
            storage_options: self.storage_options.clone(),
        })
    }

    async fn declare_table(&self, request: DeclareTableRequest) -> Result<DeclareTableResponse> {
        if let Some(ref manifest_ns) = self.manifest_ns {
            return manifest_ns.declare_table(request).await;
        }

        let table_name = Self::table_name_from_id(&request.id)?;
        let table_uri = self.table_full_uri(&table_name);

        // Validate location if provided
        if let Some(location) = &request.location {
            let location = location.trim_end_matches('/');
            if location != table_uri {
                return Err(Error::Namespace {
                    source: format!(
                        "Cannot declare table {} at location {}, must be at location {}",
                        table_name, location, table_uri
                    )
                    .into(),
                    location: snafu::location!(),
                });
            }
        }

        // Check if table already has data (created via create_table).
        // The atomic put only prevents races between concurrent declare_table calls,
        // not between declare_table and existing data.
        let status = self.check_table_status(&table_name).await;
        if status.exists && !status.has_reserved_file {
            // Table has data but no reserved file - it was created with data
            return Err(Error::Namespace {
                source: format!("Table already exists: {}", table_name).into(),
                location: snafu::location!(),
            });
        }

        // Atomically create the .lance-reserved file to mark the table as declared.
        // This uses put_if_not_exists semantics to avoid race conditions between
        // concurrent declare_table calls.
        let reserved_file_path = self.table_reserved_file_path(&table_name);

        self.put_marker_file_atomic(&reserved_file_path, &format!("table {}", table_name))
            .await
            .map_err(|e| Error::Namespace {
                source: e.into(),
                location: snafu::location!(),
            })?;

        Ok(DeclareTableResponse {
            transaction_id: None,
            location: Some(table_uri),
            storage_options: self.storage_options.clone(),
        })
    }

    async fn register_table(
        &self,
        request: lance_namespace::models::RegisterTableRequest,
    ) -> Result<lance_namespace::models::RegisterTableResponse> {
        // If manifest is enabled, delegate to manifest namespace
        if let Some(ref manifest_ns) = self.manifest_ns {
            return LanceNamespace::register_table(manifest_ns.as_ref(), request).await;
        }

        // Without manifest, register_table is not supported
        Err(Error::NotSupported {
            source: "register_table is only supported when manifest mode is enabled".into(),
            location: snafu::location!(),
        })
    }

    async fn deregister_table(
        &self,
        request: lance_namespace::models::DeregisterTableRequest,
    ) -> Result<lance_namespace::models::DeregisterTableResponse> {
        // If manifest is enabled, delegate to manifest namespace
        if let Some(ref manifest_ns) = self.manifest_ns {
            return LanceNamespace::deregister_table(manifest_ns.as_ref(), request).await;
        }

        // V1 mode: create a .lance-deregistered marker file in the table directory
        let table_name = Self::table_name_from_id(&request.id)?;
        let table_uri = self.table_full_uri(&table_name);

        // Check table existence and deregistration status.
        // This provides better error messages for common cases.
        let status = self.check_table_status(&table_name).await;

        if !status.exists {
            return Err(Error::Namespace {
                source: format!("Table does not exist: {}", table_name).into(),
                location: snafu::location!(),
            });
        }

        if status.is_deregistered {
            return Err(Error::Namespace {
                source: format!("Table is already deregistered: {}", table_name).into(),
                location: snafu::location!(),
            });
        }

        // Atomically create the .lance-deregistered marker file.
        // This uses put_if_not_exists semantics to prevent race conditions
        // when multiple processes try to deregister the same table concurrently.
        // If a race occurs and another process already created the file,
        // we'll get an AlreadyExists error which we convert to a proper message.
        let deregistered_path = self.table_deregistered_file_path(&table_name);
        self.put_marker_file_atomic(
            &deregistered_path,
            &format!("deregistration marker for table {}", table_name),
        )
        .await
        .map_err(|e| {
            // Convert "already exists" to "already deregistered" for better UX
            let message = if e.contains("already exists") {
                format!("Table is already deregistered: {}", table_name)
            } else {
                e
            };
            Error::Namespace {
                source: message.into(),
                location: snafu::location!(),
            }
        })?;

        Ok(lance_namespace::models::DeregisterTableResponse {
            id: request.id,
            location: Some(table_uri),
            properties: None,
            transaction_id: None,
        })
    }

    fn namespace_id(&self) -> String {
        format!("DirectoryNamespace {{ root: {:?} }}", self.root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_ipc::reader::StreamReader;
    use lance::dataset::Dataset;
    use lance_core::utils::tempfile::TempStdDir;
    use lance_namespace::models::{
        CreateTableRequest, JsonArrowDataType, JsonArrowField, JsonArrowSchema, ListTablesRequest,
    };
    use lance_namespace::schema::convert_json_arrow_schema;
    use std::io::Cursor;
    use std::sync::Arc;

    /// Helper to create a test DirectoryNamespace with a temporary directory
    async fn create_test_namespace() -> (DirectoryNamespace, TempStdDir) {
        let temp_dir = TempStdDir::default();

        let namespace = DirectoryNamespaceBuilder::new(temp_dir.to_str().unwrap())
            .build()
            .await
            .unwrap();
        (namespace, temp_dir)
    }

    /// Helper to create test IPC data from a schema
    fn create_test_ipc_data(schema: &JsonArrowSchema) -> Vec<u8> {
        use arrow::ipc::writer::StreamWriter;

        let arrow_schema = convert_json_arrow_schema(schema).unwrap();
        let arrow_schema = Arc::new(arrow_schema);
        let batch = arrow::record_batch::RecordBatch::new_empty(arrow_schema.clone());
        let mut buffer = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buffer, &arrow_schema).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }
        buffer
    }

    /// Helper to create a simple test schema
    fn create_test_schema() -> JsonArrowSchema {
        let int_type = JsonArrowDataType::new("int32".to_string());
        let string_type = JsonArrowDataType::new("utf8".to_string());

        let id_field = JsonArrowField {
            name: "id".to_string(),
            r#type: Box::new(int_type),
            nullable: false,
            metadata: None,
        };

        let name_field = JsonArrowField {
            name: "name".to_string(),
            r#type: Box::new(string_type),
            nullable: true,
            metadata: None,
        };

        JsonArrowSchema {
            fields: vec![id_field, name_field],
            metadata: None,
        }
    }

    #[tokio::test]
    async fn test_create_table() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create test IPC data
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        let mut request = CreateTableRequest::new();
        request.id = Some(vec!["test_table".to_string()]);

        let response = namespace
            .create_table(request, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        assert!(response.location.is_some());
        assert!(response.location.unwrap().ends_with("test_table.lance"));
        assert_eq!(response.version, Some(1));
    }

    #[tokio::test]
    async fn test_create_table_without_data() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        let mut request = CreateTableRequest::new();
        request.id = Some(vec!["test_table".to_string()]);

        let result = namespace.create_table(request, bytes::Bytes::new()).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Arrow IPC stream) is required"));
    }

    #[tokio::test]
    async fn test_create_table_with_invalid_id() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create test IPC data
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        // Test with empty ID
        let mut request = CreateTableRequest::new();
        request.id = Some(vec![]);

        let result = namespace
            .create_table(request, bytes::Bytes::from(ipc_data.clone()))
            .await;
        assert!(result.is_err());

        // Test with multi-level ID - should now work with manifest enabled
        // First create the parent namespace
        let mut create_ns_req = CreateNamespaceRequest::new();
        create_ns_req.id = Some(vec!["test_namespace".to_string()]);
        namespace.create_namespace(create_ns_req).await.unwrap();

        // Now create table in the namespace
        let mut request = CreateTableRequest::new();
        request.id = Some(vec!["test_namespace".to_string(), "table".to_string()]);

        let result = namespace
            .create_table(request, bytes::Bytes::from(ipc_data))
            .await;
        // Should succeed with manifest enabled
        assert!(
            result.is_ok(),
            "Multi-level table IDs should work with manifest enabled"
        );
    }

    #[tokio::test]
    async fn test_list_tables() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Initially, no tables
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 0);

        // Create test IPC data
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        // Create a table
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["table1".to_string()]);
        namespace
            .create_table(create_request, bytes::Bytes::from(ipc_data.clone()))
            .await
            .unwrap();

        // Create another table
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["table2".to_string()]);
        namespace
            .create_table(create_request, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // List tables should return both
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = namespace.list_tables(request).await.unwrap();
        let tables = response.tables;
        assert_eq!(tables.len(), 2);
        assert!(tables.contains(&"table1".to_string()));
        assert!(tables.contains(&"table2".to_string()));
    }

    #[tokio::test]
    async fn test_list_tables_with_namespace_id() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // First create a child namespace
        let mut create_ns_req = CreateNamespaceRequest::new();
        create_ns_req.id = Some(vec!["test_namespace".to_string()]);
        namespace.create_namespace(create_ns_req).await.unwrap();

        // Now list tables in the child namespace
        let mut request = ListTablesRequest::new();
        request.id = Some(vec!["test_namespace".to_string()]);

        let result = namespace.list_tables(request).await;
        // Should succeed (with manifest enabled) and return empty list (no tables yet)
        assert!(
            result.is_ok(),
            "list_tables should work with child namespace when manifest is enabled"
        );
        let response = result.unwrap();
        assert_eq!(
            response.tables.len(),
            0,
            "Namespace should have no tables yet"
        );
    }

    #[tokio::test]
    async fn test_describe_table() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create a table first
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["test_table".to_string()]);
        namespace
            .create_table(create_request, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Describe the table
        let mut request = DescribeTableRequest::new();
        request.id = Some(vec!["test_table".to_string()]);
        let response = namespace.describe_table(request).await.unwrap();

        assert!(response.location.is_some());
        assert!(response.location.unwrap().ends_with("test_table.lance"));
    }

    #[tokio::test]
    async fn test_describe_nonexistent_table() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        let mut request = DescribeTableRequest::new();
        request.id = Some(vec!["nonexistent".to_string()]);

        let result = namespace.describe_table(request).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Table does not exist"));
    }

    #[tokio::test]
    async fn test_table_exists() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create a table
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["existing_table".to_string()]);
        namespace
            .create_table(create_request, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Check existing table
        let mut request = TableExistsRequest::new();
        request.id = Some(vec!["existing_table".to_string()]);
        let result = namespace.table_exists(request).await;
        assert!(result.is_ok());

        // Check non-existent table
        let mut request = TableExistsRequest::new();
        request.id = Some(vec!["nonexistent".to_string()]);
        let result = namespace.table_exists(request).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Table does not exist"));
    }

    #[tokio::test]
    async fn test_drop_table() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create a table
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["table_to_drop".to_string()]);
        namespace
            .create_table(create_request, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Verify it exists
        let mut exists_request = TableExistsRequest::new();
        exists_request.id = Some(vec!["table_to_drop".to_string()]);
        assert!(namespace.table_exists(exists_request.clone()).await.is_ok());

        // Drop the table
        let mut drop_request = DropTableRequest::new();
        drop_request.id = Some(vec!["table_to_drop".to_string()]);
        let response = namespace.drop_table(drop_request).await.unwrap();
        assert!(response.location.is_some());

        // Verify it no longer exists
        assert!(namespace.table_exists(exists_request).await.is_err());
    }

    #[tokio::test]
    async fn test_drop_nonexistent_table() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        let mut request = DropTableRequest::new();
        request.id = Some(vec!["nonexistent".to_string()]);

        // Should not fail when dropping non-existent table (idempotent)
        let result = namespace.drop_table(request).await;
        // The operation might succeed or fail depending on implementation
        // But it should not panic
        let _ = result;
    }

    #[tokio::test]
    async fn test_root_namespace_operations() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Test list_namespaces - should return empty list for root
        let mut request = ListNamespacesRequest::new();
        request.id = Some(vec![]);
        let result = namespace.list_namespaces(request).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().namespaces.len(), 0);

        // Test describe_namespace - should succeed for root
        let mut request = DescribeNamespaceRequest::new();
        request.id = Some(vec![]);
        let result = namespace.describe_namespace(request).await;
        assert!(result.is_ok());

        // Test namespace_exists - root always exists
        let mut request = NamespaceExistsRequest::new();
        request.id = Some(vec![]);
        let result = namespace.namespace_exists(request).await;
        assert!(result.is_ok());

        // Test create_namespace - root cannot be created
        let mut request = CreateNamespaceRequest::new();
        request.id = Some(vec![]);
        let result = namespace.create_namespace(request).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));

        // Test drop_namespace - root cannot be dropped
        let mut request = DropNamespaceRequest::new();
        request.id = Some(vec![]);
        let result = namespace.drop_namespace(request).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("cannot be dropped"));
    }

    #[tokio::test]
    async fn test_non_root_namespace_operations() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // With manifest enabled (default), child namespaces are now supported
        // Test create_namespace for non-root - should succeed with manifest
        let mut request = CreateNamespaceRequest::new();
        request.id = Some(vec!["child".to_string()]);
        let result = namespace.create_namespace(request).await;
        assert!(
            result.is_ok(),
            "Child namespace creation should succeed with manifest enabled"
        );

        // Test namespace_exists for non-root - should exist after creation
        let mut request = NamespaceExistsRequest::new();
        request.id = Some(vec!["child".to_string()]);
        let result = namespace.namespace_exists(request).await;
        assert!(
            result.is_ok(),
            "Child namespace should exist after creation"
        );

        // Test drop_namespace for non-root - should succeed
        let mut request = DropNamespaceRequest::new();
        request.id = Some(vec!["child".to_string()]);
        let result = namespace.drop_namespace(request).await;
        assert!(
            result.is_ok(),
            "Child namespace drop should succeed with manifest enabled"
        );

        // Verify namespace no longer exists
        let mut request = NamespaceExistsRequest::new();
        request.id = Some(vec!["child".to_string()]);
        let result = namespace.namespace_exists(request).await;
        assert!(
            result.is_err(),
            "Child namespace should not exist after drop"
        );
    }

    #[tokio::test]
    async fn test_config_custom_root() {
        let temp_dir = TempStdDir::default();
        let custom_path = temp_dir.join("custom");
        std::fs::create_dir(&custom_path).unwrap();

        let namespace = DirectoryNamespaceBuilder::new(custom_path.to_string_lossy().to_string())
            .build()
            .await
            .unwrap();

        // Create test IPC data
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        // Create a table and verify location
        let mut request = CreateTableRequest::new();
        request.id = Some(vec!["test_table".to_string()]);

        let response = namespace
            .create_table(request, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        assert!(response.location.unwrap().contains("custom"));
    }

    #[tokio::test]
    async fn test_config_storage_options() {
        let temp_dir = TempStdDir::default();

        let namespace = DirectoryNamespaceBuilder::new(temp_dir.to_str().unwrap())
            .storage_option("option1", "value1")
            .storage_option("option2", "value2")
            .build()
            .await
            .unwrap();

        // Create test IPC data
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        // Create a table and check storage options are included
        let mut request = CreateTableRequest::new();
        request.id = Some(vec!["test_table".to_string()]);

        let response = namespace
            .create_table(request, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        let storage_options = response.storage_options.unwrap();
        assert_eq!(storage_options.get("option1"), Some(&"value1".to_string()));
        assert_eq!(storage_options.get("option2"), Some(&"value2".to_string()));
    }

    #[tokio::test]
    async fn test_from_properties_manifest_enabled() {
        let temp_dir = TempStdDir::default();

        let mut properties = HashMap::new();
        properties.insert("root".to_string(), temp_dir.to_str().unwrap().to_string());
        properties.insert("manifest_enabled".to_string(), "true".to_string());
        properties.insert("dir_listing_enabled".to_string(), "false".to_string());

        let builder = DirectoryNamespaceBuilder::from_properties(properties, None).unwrap();
        assert!(builder.manifest_enabled);
        assert!(!builder.dir_listing_enabled);

        let namespace = builder.build().await.unwrap();

        // Create test IPC data
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        // Create a table
        let mut request = CreateTableRequest::new();
        request.id = Some(vec!["test_table".to_string()]);

        let response = namespace
            .create_table(request, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        assert!(response.location.is_some());
    }

    #[tokio::test]
    async fn test_from_properties_dir_listing_enabled() {
        let temp_dir = TempStdDir::default();

        let mut properties = HashMap::new();
        properties.insert("root".to_string(), temp_dir.to_str().unwrap().to_string());
        properties.insert("manifest_enabled".to_string(), "false".to_string());
        properties.insert("dir_listing_enabled".to_string(), "true".to_string());

        let builder = DirectoryNamespaceBuilder::from_properties(properties, None).unwrap();
        assert!(!builder.manifest_enabled);
        assert!(builder.dir_listing_enabled);

        let namespace = builder.build().await.unwrap();

        // Create test IPC data
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        // Create a table
        let mut request = CreateTableRequest::new();
        request.id = Some(vec!["test_table".to_string()]);

        let response = namespace
            .create_table(request, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        assert!(response.location.is_some());
    }

    #[tokio::test]
    async fn test_from_properties_defaults() {
        let temp_dir = TempStdDir::default();

        let mut properties = HashMap::new();
        properties.insert("root".to_string(), temp_dir.to_str().unwrap().to_string());

        let builder = DirectoryNamespaceBuilder::from_properties(properties, None).unwrap();
        // Both should default to true
        assert!(builder.manifest_enabled);
        assert!(builder.dir_listing_enabled);
    }

    #[tokio::test]
    async fn test_from_properties_with_storage_options() {
        let temp_dir = TempStdDir::default();

        let mut properties = HashMap::new();
        properties.insert("root".to_string(), temp_dir.to_str().unwrap().to_string());
        properties.insert("manifest_enabled".to_string(), "true".to_string());
        properties.insert("storage.region".to_string(), "us-west-2".to_string());
        properties.insert("storage.bucket".to_string(), "my-bucket".to_string());

        let builder = DirectoryNamespaceBuilder::from_properties(properties, None).unwrap();
        assert!(builder.manifest_enabled);
        assert!(builder.storage_options.is_some());

        let storage_options = builder.storage_options.unwrap();
        assert_eq!(
            storage_options.get("region"),
            Some(&"us-west-2".to_string())
        );
        assert_eq!(
            storage_options.get("bucket"),
            Some(&"my-bucket".to_string())
        );
    }

    #[tokio::test]
    async fn test_various_arrow_types() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create schema with various types
        let fields = vec![
            JsonArrowField {
                name: "bool_col".to_string(),
                r#type: Box::new(JsonArrowDataType::new("bool".to_string())),
                nullable: true,
                metadata: None,
            },
            JsonArrowField {
                name: "int8_col".to_string(),
                r#type: Box::new(JsonArrowDataType::new("int8".to_string())),
                nullable: true,
                metadata: None,
            },
            JsonArrowField {
                name: "float64_col".to_string(),
                r#type: Box::new(JsonArrowDataType::new("float64".to_string())),
                nullable: true,
                metadata: None,
            },
            JsonArrowField {
                name: "binary_col".to_string(),
                r#type: Box::new(JsonArrowDataType::new("binary".to_string())),
                nullable: true,
                metadata: None,
            },
        ];

        let schema = JsonArrowSchema {
            fields,
            metadata: None,
        };

        // Create IPC data
        let ipc_data = create_test_ipc_data(&schema);

        let mut request = CreateTableRequest::new();
        request.id = Some(vec!["complex_table".to_string()]);

        let response = namespace
            .create_table(request, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        assert!(response.location.is_some());
    }

    #[tokio::test]
    async fn test_connect_dir() {
        let temp_dir = TempStdDir::default();

        let namespace = DirectoryNamespaceBuilder::new(temp_dir.to_str().unwrap())
            .build()
            .await
            .unwrap();

        // Test basic operation through the concrete type
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 0);
    }

    #[tokio::test]
    async fn test_create_table_with_ipc_data() {
        use arrow::array::{Int32Array, StringArray};
        use arrow::ipc::writer::StreamWriter;

        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create a schema with some fields
        let schema = create_test_schema();

        // Create some test data that matches the schema
        let arrow_schema = convert_json_arrow_schema(&schema).unwrap();
        let arrow_schema = Arc::new(arrow_schema);

        // Create a RecordBatch with actual data
        let id_array = Int32Array::from(vec![1, 2, 3]);
        let name_array = StringArray::from(vec!["Alice", "Bob", "Charlie"]);
        let batch = arrow::record_batch::RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(id_array), Arc::new(name_array)],
        )
        .unwrap();

        // Write the batch to an IPC stream
        let mut buffer = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buffer, &arrow_schema).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }

        // Create table with the IPC data
        let mut request = CreateTableRequest::new();
        request.id = Some(vec!["test_table_with_data".to_string()]);

        let response = namespace
            .create_table(request, Bytes::from(buffer))
            .await
            .unwrap();

        assert_eq!(response.version, Some(1));
        assert!(response
            .location
            .unwrap()
            .contains("test_table_with_data.lance"));

        // Verify table exists
        let mut exists_request = TableExistsRequest::new();
        exists_request.id = Some(vec!["test_table_with_data".to_string()]);
        namespace.table_exists(exists_request).await.unwrap();
    }

    #[tokio::test]
    #[allow(deprecated)]
    async fn test_create_empty_table() {
        let (namespace, temp_dir) = create_test_namespace().await;

        let mut request = CreateEmptyTableRequest::new();
        request.id = Some(vec!["empty_table".to_string()]);

        let response = namespace.create_empty_table(request).await.unwrap();

        assert!(response.location.is_some());
        assert!(response.location.unwrap().ends_with("empty_table.lance"));

        // Verify the .lance-reserved file was created in the correct location
        let table_dir = temp_dir.join("empty_table.lance");
        assert!(table_dir.exists());
        assert!(table_dir.is_dir());

        let reserved_file = table_dir.join(".lance-reserved");
        assert!(reserved_file.exists());
        assert!(reserved_file.is_file());

        // Verify file is empty
        let metadata = std::fs::metadata(&reserved_file).unwrap();
        assert_eq!(metadata.len(), 0);

        // Verify table exists by checking for .lance-reserved file
        let mut exists_request = TableExistsRequest::new();
        exists_request.id = Some(vec!["empty_table".to_string()]);
        namespace.table_exists(exists_request).await.unwrap();

        // List tables should include the empty table
        let mut list_request = ListTablesRequest::new();
        list_request.id = Some(vec![]);
        let list_response = namespace.list_tables(list_request).await.unwrap();
        assert!(list_response.tables.contains(&"empty_table".to_string()));

        // Verify describe table works for empty table
        let mut describe_request = DescribeTableRequest::new();
        describe_request.id = Some(vec!["empty_table".to_string()]);
        let describe_response = namespace.describe_table(describe_request).await.unwrap();
        assert!(describe_response.location.is_some());
        assert!(describe_response.location.unwrap().contains("empty_table"));
    }

    #[tokio::test]
    #[allow(deprecated)]
    async fn test_create_empty_table_with_wrong_location() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        let mut request = CreateEmptyTableRequest::new();
        request.id = Some(vec!["test_table".to_string()]);
        request.location = Some("/wrong/path/table.lance".to_string());

        let result = namespace.create_empty_table(request).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must be at location"));
    }

    #[tokio::test]
    #[allow(deprecated)]
    async fn test_create_empty_table_then_drop() {
        let (namespace, temp_dir) = create_test_namespace().await;

        // Create an empty table
        let mut create_request = CreateEmptyTableRequest::new();
        create_request.id = Some(vec!["empty_table_to_drop".to_string()]);

        let create_response = namespace.create_empty_table(create_request).await.unwrap();
        assert!(create_response.location.is_some());

        // Verify it exists
        let table_dir = temp_dir.join("empty_table_to_drop.lance");
        assert!(table_dir.exists());
        let reserved_file = table_dir.join(".lance-reserved");
        assert!(reserved_file.exists());

        // Drop the table
        let mut drop_request = DropTableRequest::new();
        drop_request.id = Some(vec!["empty_table_to_drop".to_string()]);
        let drop_response = namespace.drop_table(drop_request).await.unwrap();
        assert!(drop_response.location.is_some());

        // Verify table directory was removed
        assert!(!table_dir.exists());
        assert!(!reserved_file.exists());

        // Verify table no longer exists
        let mut exists_request = TableExistsRequest::new();
        exists_request.id = Some(vec!["empty_table_to_drop".to_string()]);
        let exists_result = namespace.table_exists(exists_request).await;
        assert!(exists_result.is_err());
    }

    #[tokio::test]
    async fn test_child_namespace_create_and_list() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create multiple child namespaces
        for i in 1..=3 {
            let mut create_req = CreateNamespaceRequest::new();
            create_req.id = Some(vec![format!("ns{}", i)]);
            let result = namespace.create_namespace(create_req).await;
            assert!(result.is_ok(), "Failed to create child namespace ns{}", i);
        }

        // List child namespaces
        let list_req = ListNamespacesRequest {
            id: Some(vec![]),
            page_token: None,
            limit: None,
        };
        let result = namespace.list_namespaces(list_req).await;
        assert!(result.is_ok());
        let namespaces = result.unwrap().namespaces;
        assert_eq!(namespaces.len(), 3);
        assert!(namespaces.contains(&"ns1".to_string()));
        assert!(namespaces.contains(&"ns2".to_string()));
        assert!(namespaces.contains(&"ns3".to_string()));
    }

    #[tokio::test]
    async fn test_nested_namespace_hierarchy() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create parent namespace
        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["parent".to_string()]);
        namespace.create_namespace(create_req).await.unwrap();

        // Create nested children
        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["parent".to_string(), "child1".to_string()]);
        namespace.create_namespace(create_req).await.unwrap();

        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["parent".to_string(), "child2".to_string()]);
        namespace.create_namespace(create_req).await.unwrap();

        // List children of parent
        let list_req = ListNamespacesRequest {
            id: Some(vec!["parent".to_string()]),
            page_token: None,
            limit: None,
        };
        let result = namespace.list_namespaces(list_req).await;
        assert!(result.is_ok());
        let children = result.unwrap().namespaces;
        assert_eq!(children.len(), 2);
        assert!(children.contains(&"child1".to_string()));
        assert!(children.contains(&"child2".to_string()));

        // List root should only show parent
        let list_req = ListNamespacesRequest {
            id: Some(vec![]),
            page_token: None,
            limit: None,
        };
        let result = namespace.list_namespaces(list_req).await;
        assert!(result.is_ok());
        let root_namespaces = result.unwrap().namespaces;
        assert_eq!(root_namespaces.len(), 1);
        assert_eq!(root_namespaces[0], "parent");
    }

    #[tokio::test]
    async fn test_table_in_child_namespace() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create child namespace
        let mut create_ns_req = CreateNamespaceRequest::new();
        create_ns_req.id = Some(vec!["test_ns".to_string()]);
        namespace.create_namespace(create_ns_req).await.unwrap();

        // Create table in child namespace
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_table_req = CreateTableRequest::new();
        create_table_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
        let result = namespace
            .create_table(create_table_req, bytes::Bytes::from(ipc_data))
            .await;
        assert!(result.is_ok(), "Failed to create table in child namespace");

        // List tables in child namespace
        let list_req = ListTablesRequest {
            id: Some(vec!["test_ns".to_string()]),
            page_token: None,
            limit: None,
        };
        let result = namespace.list_tables(list_req).await;
        assert!(result.is_ok());
        let tables = result.unwrap().tables;
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0], "table1");

        // Verify table exists
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
        let result = namespace.table_exists(exists_req).await;
        assert!(result.is_ok());

        // Describe table in child namespace
        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
        let result = namespace.describe_table(describe_req).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response.location.is_some());
    }

    #[tokio::test]
    async fn test_multiple_tables_in_child_namespace() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create child namespace
        let mut create_ns_req = CreateNamespaceRequest::new();
        create_ns_req.id = Some(vec!["test_ns".to_string()]);
        namespace.create_namespace(create_ns_req).await.unwrap();

        // Create multiple tables
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        for i in 1..=3 {
            let mut create_table_req = CreateTableRequest::new();
            create_table_req.id = Some(vec!["test_ns".to_string(), format!("table{}", i)]);
            namespace
                .create_table(create_table_req, bytes::Bytes::from(ipc_data.clone()))
                .await
                .unwrap();
        }

        // List tables
        let list_req = ListTablesRequest {
            id: Some(vec!["test_ns".to_string()]),
            page_token: None,
            limit: None,
        };
        let result = namespace.list_tables(list_req).await;
        assert!(result.is_ok());
        let tables = result.unwrap().tables;
        assert_eq!(tables.len(), 3);
        assert!(tables.contains(&"table1".to_string()));
        assert!(tables.contains(&"table2".to_string()));
        assert!(tables.contains(&"table3".to_string()));
    }

    #[tokio::test]
    async fn test_drop_table_in_child_namespace() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create child namespace
        let mut create_ns_req = CreateNamespaceRequest::new();
        create_ns_req.id = Some(vec!["test_ns".to_string()]);
        namespace.create_namespace(create_ns_req).await.unwrap();

        // Create table
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_table_req = CreateTableRequest::new();
        create_table_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
        namespace
            .create_table(create_table_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Drop table
        let mut drop_req = DropTableRequest::new();
        drop_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
        let result = namespace.drop_table(drop_req).await;
        assert!(result.is_ok(), "Failed to drop table in child namespace");

        // Verify table no longer exists
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
        let result = namespace.table_exists(exists_req).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    #[allow(deprecated)]
    async fn test_empty_table_in_child_namespace() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create child namespace
        let mut create_ns_req = CreateNamespaceRequest::new();
        create_ns_req.id = Some(vec!["test_ns".to_string()]);
        namespace.create_namespace(create_ns_req).await.unwrap();

        // Create empty table
        let mut create_empty_req = CreateEmptyTableRequest::new();
        create_empty_req.id = Some(vec!["test_ns".to_string(), "empty_table".to_string()]);
        let result = namespace.create_empty_table(create_empty_req).await;
        assert!(
            result.is_ok(),
            "Failed to create empty table in child namespace"
        );

        // Verify table exists
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["test_ns".to_string(), "empty_table".to_string()]);
        let result = namespace.table_exists(exists_req).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_deeply_nested_namespace() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create deeply nested namespace hierarchy
        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["level1".to_string()]);
        namespace.create_namespace(create_req).await.unwrap();

        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["level1".to_string(), "level2".to_string()]);
        namespace.create_namespace(create_req).await.unwrap();

        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec![
            "level1".to_string(),
            "level2".to_string(),
            "level3".to_string(),
        ]);
        namespace.create_namespace(create_req).await.unwrap();

        // Create table in deeply nested namespace
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_table_req = CreateTableRequest::new();
        create_table_req.id = Some(vec![
            "level1".to_string(),
            "level2".to_string(),
            "level3".to_string(),
            "table1".to_string(),
        ]);
        let result = namespace
            .create_table(create_table_req, bytes::Bytes::from(ipc_data))
            .await;
        assert!(
            result.is_ok(),
            "Failed to create table in deeply nested namespace"
        );

        // Verify table exists
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec![
            "level1".to_string(),
            "level2".to_string(),
            "level3".to_string(),
            "table1".to_string(),
        ]);
        let result = namespace.table_exists(exists_req).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_namespace_with_properties() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create namespace with properties
        let mut properties = HashMap::new();
        properties.insert("owner".to_string(), "test_user".to_string());
        properties.insert("description".to_string(), "Test namespace".to_string());

        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["test_ns".to_string()]);
        create_req.properties = Some(properties.clone());
        namespace.create_namespace(create_req).await.unwrap();

        // Describe namespace and verify properties
        let describe_req = DescribeNamespaceRequest {
            id: Some(vec!["test_ns".to_string()]),
        };
        let result = namespace.describe_namespace(describe_req).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response.properties.is_some());
        let props = response.properties.unwrap();
        assert_eq!(props.get("owner"), Some(&"test_user".to_string()));
        assert_eq!(
            props.get("description"),
            Some(&"Test namespace".to_string())
        );
    }

    #[tokio::test]
    async fn test_cannot_drop_namespace_with_tables() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create namespace
        let mut create_ns_req = CreateNamespaceRequest::new();
        create_ns_req.id = Some(vec!["test_ns".to_string()]);
        namespace.create_namespace(create_ns_req).await.unwrap();

        // Create table in namespace
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_table_req = CreateTableRequest::new();
        create_table_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
        namespace
            .create_table(create_table_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Try to drop namespace - should fail
        let mut drop_req = DropNamespaceRequest::new();
        drop_req.id = Some(vec!["test_ns".to_string()]);
        let result = namespace.drop_namespace(drop_req).await;
        assert!(
            result.is_err(),
            "Should not be able to drop namespace with tables"
        );
    }

    #[tokio::test]
    async fn test_isolation_between_namespaces() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create two namespaces
        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["ns1".to_string()]);
        namespace.create_namespace(create_req).await.unwrap();

        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["ns2".to_string()]);
        namespace.create_namespace(create_req).await.unwrap();

        // Create table with same name in both namespaces
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        let mut create_table_req = CreateTableRequest::new();
        create_table_req.id = Some(vec!["ns1".to_string(), "table1".to_string()]);
        namespace
            .create_table(create_table_req, bytes::Bytes::from(ipc_data.clone()))
            .await
            .unwrap();

        let mut create_table_req = CreateTableRequest::new();
        create_table_req.id = Some(vec!["ns2".to_string(), "table1".to_string()]);
        namespace
            .create_table(create_table_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // List tables in each namespace
        let list_req = ListTablesRequest {
            id: Some(vec!["ns1".to_string()]),
            page_token: None,
            limit: None,
        };
        let result = namespace.list_tables(list_req).await.unwrap();
        assert_eq!(result.tables.len(), 1);
        assert_eq!(result.tables[0], "table1");

        let list_req = ListTablesRequest {
            id: Some(vec!["ns2".to_string()]),
            page_token: None,
            limit: None,
        };
        let result = namespace.list_tables(list_req).await.unwrap();
        assert_eq!(result.tables.len(), 1);
        assert_eq!(result.tables[0], "table1");

        // Drop table in ns1 shouldn't affect ns2
        let mut drop_req = DropTableRequest::new();
        drop_req.id = Some(vec!["ns1".to_string(), "table1".to_string()]);
        namespace.drop_table(drop_req).await.unwrap();

        // Verify ns1 table is gone but ns2 table still exists
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["ns1".to_string(), "table1".to_string()]);
        assert!(namespace.table_exists(exists_req).await.is_err());

        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["ns2".to_string(), "table1".to_string()]);
        assert!(namespace.table_exists(exists_req).await.is_ok());
    }

    #[tokio::test]
    async fn test_migrate_directory_tables() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Step 1: Create tables in directory-only mode
        let dir_only_ns = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .dir_listing_enabled(true)
            .build()
            .await
            .unwrap();

        // Create some tables
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        for i in 1..=3 {
            let mut create_req = CreateTableRequest::new();
            create_req.id = Some(vec![format!("table{}", i)]);
            dir_only_ns
                .create_table(create_req, bytes::Bytes::from(ipc_data.clone()))
                .await
                .unwrap();
        }

        drop(dir_only_ns);

        // Step 2: Create namespace with dual mode (manifest + directory listing)
        let dual_mode_ns = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(true)
            .dir_listing_enabled(true)
            .build()
            .await
            .unwrap();

        // Before migration, tables should be visible (via directory listing fallback)
        let mut list_req = ListTablesRequest::new();
        list_req.id = Some(vec![]);
        let tables = dual_mode_ns.list_tables(list_req).await.unwrap().tables;
        assert_eq!(tables.len(), 3);

        // Run migration
        let migrated_count = dual_mode_ns.migrate().await.unwrap();
        assert_eq!(migrated_count, 3, "Should migrate all 3 tables");

        // Verify tables are now in manifest
        let mut list_req = ListTablesRequest::new();
        list_req.id = Some(vec![]);
        let tables = dual_mode_ns.list_tables(list_req).await.unwrap().tables;
        assert_eq!(tables.len(), 3);

        // Run migration again - should be idempotent
        let migrated_count = dual_mode_ns.migrate().await.unwrap();
        assert_eq!(
            migrated_count, 0,
            "Should not migrate already-migrated tables"
        );

        drop(dual_mode_ns);

        // Step 3: Create namespace with manifest-only mode
        let manifest_only_ns = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(true)
            .dir_listing_enabled(false)
            .build()
            .await
            .unwrap();

        // Tables should still be accessible (now from manifest only)
        let mut list_req = ListTablesRequest::new();
        list_req.id = Some(vec![]);
        let tables = manifest_only_ns.list_tables(list_req).await.unwrap().tables;
        assert_eq!(tables.len(), 3);
        assert!(tables.contains(&"table1".to_string()));
        assert!(tables.contains(&"table2".to_string()));
        assert!(tables.contains(&"table3".to_string()));
    }

    #[tokio::test]
    async fn test_migrate_without_manifest() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Create namespace without manifest
        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .dir_listing_enabled(true)
            .build()
            .await
            .unwrap();

        // migrate() should return 0 when manifest is not enabled
        let migrated_count = namespace.migrate().await.unwrap();
        assert_eq!(migrated_count, 0);
    }

    #[tokio::test]
    async fn test_register_table() {
        use lance_namespace::models::{RegisterTableRequest, TableExistsRequest};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .unwrap();

        // Create a physical table first using lance directly
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        let table_uri = format!("{}/external_table.lance", temp_path);
        let cursor = Cursor::new(ipc_data);
        let stream_reader = StreamReader::try_new(cursor, None).unwrap();
        let batches: Vec<_> = stream_reader
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        let schema = batches[0].schema();
        let batch_results: Vec<_> = batches.into_iter().map(Ok).collect();
        let reader = RecordBatchIterator::new(batch_results, schema);
        Dataset::write(Box::new(reader), &table_uri, None)
            .await
            .unwrap();

        // Register the table
        let mut register_req = RegisterTableRequest::new("external_table.lance".to_string());
        register_req.id = Some(vec!["registered_table".to_string()]);

        let response = namespace.register_table(register_req).await.unwrap();
        assert_eq!(response.location, Some("external_table.lance".to_string()));

        // Verify table exists in namespace
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["registered_table".to_string()]);
        assert!(namespace.table_exists(exists_req).await.is_ok());

        // Verify we can list the table
        let mut list_req = ListTablesRequest::new();
        list_req.id = Some(vec![]);
        let tables = namespace.list_tables(list_req).await.unwrap();
        assert!(tables.tables.contains(&"registered_table".to_string()));
    }

    #[tokio::test]
    async fn test_register_table_duplicate_fails() {
        use lance_namespace::models::RegisterTableRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .unwrap();

        // Register a table
        let mut register_req = RegisterTableRequest::new("test_table.lance".to_string());
        register_req.id = Some(vec!["test_table".to_string()]);

        namespace
            .register_table(register_req.clone())
            .await
            .unwrap();

        // Try to register again - should fail
        let result = namespace.register_table(register_req).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    #[tokio::test]
    async fn test_deregister_table() {
        use lance_namespace::models::{DeregisterTableRequest, TableExistsRequest};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Create namespace with manifest-only mode (no directory listing fallback)
        // This ensures deregistered tables are truly invisible
        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(true)
            .dir_listing_enabled(false)
            .build()
            .await
            .unwrap();

        // Create a table
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        namespace
            .create_table(create_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Verify table exists
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["test_table".to_string()]);
        assert!(namespace.table_exists(exists_req.clone()).await.is_ok());

        // Deregister the table
        let mut deregister_req = DeregisterTableRequest::new();
        deregister_req.id = Some(vec!["test_table".to_string()]);
        let response = namespace.deregister_table(deregister_req).await.unwrap();

        // Should return location and id
        assert!(
            response.location.is_some(),
            "Deregister should return location"
        );
        let location = response.location.as_ref().unwrap();
        // Location should be a proper file:// URI with the temp path
        // Use uri_to_url to normalize the temp path to a URL for comparison
        let expected_url = lance_io::object_store::uri_to_url(temp_path)
            .expect("Failed to convert temp path to URL");
        let expected_prefix = expected_url.to_string();
        assert!(
            location.starts_with(&expected_prefix),
            "Location should start with '{}', got: {}",
            expected_prefix,
            location
        );
        assert!(
            location.contains("test_table"),
            "Location should contain table name: {}",
            location
        );
        assert_eq!(response.id, Some(vec!["test_table".to_string()]));

        // Verify table no longer exists in namespace (removed from manifest)
        assert!(namespace.table_exists(exists_req).await.is_err());

        // Verify physical data still exists at the returned location
        let dataset = Dataset::open(location).await;
        assert!(
            dataset.is_ok(),
            "Physical table data should still exist at {}",
            location
        );
    }

    #[tokio::test]
    async fn test_deregister_table_in_child_namespace() {
        use lance_namespace::models::{
            CreateNamespaceRequest, DeregisterTableRequest, TableExistsRequest,
        };

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .unwrap();

        // Create child namespace
        let mut create_ns_req = CreateNamespaceRequest::new();
        create_ns_req.id = Some(vec!["test_ns".to_string()]);
        namespace.create_namespace(create_ns_req).await.unwrap();

        // Create a table in the child namespace
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_ns".to_string(), "test_table".to_string()]);
        namespace
            .create_table(create_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Deregister the table
        let mut deregister_req = DeregisterTableRequest::new();
        deregister_req.id = Some(vec!["test_ns".to_string(), "test_table".to_string()]);
        let response = namespace.deregister_table(deregister_req).await.unwrap();

        // Should return location and id in child namespace
        assert!(
            response.location.is_some(),
            "Deregister should return location"
        );
        let location = response.location.as_ref().unwrap();
        // Location should be a proper file:// URI with the temp path
        // Use uri_to_url to normalize the temp path to a URL for comparison
        let expected_url = lance_io::object_store::uri_to_url(temp_path)
            .expect("Failed to convert temp path to URL");
        let expected_prefix = expected_url.to_string();
        assert!(
            location.starts_with(&expected_prefix),
            "Location should start with '{}', got: {}",
            expected_prefix,
            location
        );
        assert!(
            location.contains("test_ns") && location.contains("test_table"),
            "Location should contain namespace and table name: {}",
            location
        );
        assert_eq!(
            response.id,
            Some(vec!["test_ns".to_string(), "test_table".to_string()])
        );

        // Verify table no longer exists
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["test_ns".to_string(), "test_table".to_string()]);
        assert!(namespace.table_exists(exists_req).await.is_err());
    }

    #[tokio::test]
    async fn test_register_without_manifest_fails() {
        use lance_namespace::models::RegisterTableRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Create namespace without manifest
        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .build()
            .await
            .unwrap();

        // Try to register - should fail (register requires manifest)
        let mut register_req = RegisterTableRequest::new("test_table.lance".to_string());
        register_req.id = Some(vec!["test_table".to_string()]);
        let result = namespace.register_table(register_req).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("manifest mode is enabled"));

        // Note: deregister_table now works in V1 mode via .lance-deregistered marker files
        // See test_deregister_table_v1_mode for that test case
    }

    #[tokio::test]
    async fn test_register_table_rejects_absolute_uri() {
        use lance_namespace::models::RegisterTableRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .unwrap();

        // Try to register with absolute URI - should fail
        let mut register_req = RegisterTableRequest::new("s3://bucket/table.lance".to_string());
        register_req.id = Some(vec!["test_table".to_string()]);
        let result = namespace.register_table(register_req).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Absolute URIs are not allowed"));
    }

    #[tokio::test]
    async fn test_register_table_rejects_absolute_path() {
        use lance_namespace::models::RegisterTableRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .unwrap();

        // Try to register with absolute path - should fail
        let mut register_req = RegisterTableRequest::new("/tmp/table.lance".to_string());
        register_req.id = Some(vec!["test_table".to_string()]);
        let result = namespace.register_table(register_req).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Absolute paths are not allowed"));
    }

    #[tokio::test]
    async fn test_register_table_rejects_path_traversal() {
        use lance_namespace::models::RegisterTableRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .unwrap();

        // Try to register with path traversal - should fail
        let mut register_req = RegisterTableRequest::new("../outside/table.lance".to_string());
        register_req.id = Some(vec!["test_table".to_string()]);
        let result = namespace.register_table(register_req).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Path traversal is not allowed"));
    }

    #[tokio::test]
    async fn test_namespace_write() {
        use arrow::array::Int32Array;
        use arrow::datatypes::{DataType, Field as ArrowField, Schema as ArrowSchema};
        use arrow::record_batch::{RecordBatch, RecordBatchIterator};
        use lance::dataset::{Dataset, WriteMode, WriteParams};
        use lance_namespace::LanceNamespace;

        let (namespace, _temp_dir) = create_test_namespace().await;
        let namespace = Arc::new(namespace) as Arc<dyn LanceNamespace>;

        // Use child namespace instead of root
        let table_id = vec!["test_ns".to_string(), "test_table".to_string()];
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new("b", DataType::Int32, false),
        ]));

        // Test 1: CREATE mode
        let data1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(Int32Array::from(vec![10, 20, 30])),
            ],
        )
        .unwrap();

        let reader1 = RecordBatchIterator::new(vec![data1].into_iter().map(Ok), schema.clone());
        let dataset = Dataset::write_into_namespace(
            reader1,
            namespace.clone(),
            table_id.clone(),
            None,
            false,
        )
        .await
        .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 3);
        assert_eq!(dataset.version().version, 1);

        // Test 2: APPEND mode
        let data2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![4, 5])),
                Arc::new(Int32Array::from(vec![40, 50])),
            ],
        )
        .unwrap();

        let params_append = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };

        let reader2 = RecordBatchIterator::new(vec![data2].into_iter().map(Ok), schema.clone());
        let dataset = Dataset::write_into_namespace(
            reader2,
            namespace.clone(),
            table_id.clone(),
            Some(params_append),
            false,
        )
        .await
        .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 5);
        assert_eq!(dataset.version().version, 2);

        // Test 3: OVERWRITE mode
        let data3 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![100, 200])),
                Arc::new(Int32Array::from(vec![1000, 2000])),
            ],
        )
        .unwrap();

        let params_overwrite = WriteParams {
            mode: WriteMode::Overwrite,
            ..Default::default()
        };

        let reader3 = RecordBatchIterator::new(vec![data3].into_iter().map(Ok), schema.clone());
        let dataset = Dataset::write_into_namespace(
            reader3,
            namespace.clone(),
            table_id.clone(),
            Some(params_overwrite),
            false,
        )
        .await
        .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 2);
        assert_eq!(dataset.version().version, 3);

        // Verify old data was replaced
        let result = dataset.scan().try_into_batch().await.unwrap();
        let a_col = result
            .column_by_name("a")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(a_col.values(), &[100, 200]);
    }

    // ============================================================
    // Tests for declare_table
    // ============================================================

    #[tokio::test]
    async fn test_declare_table_v1_mode() {
        use lance_namespace::models::{
            DeclareTableRequest, DescribeTableRequest, TableExistsRequest,
        };

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Create namespace in V1 mode (no manifest)
        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .build()
            .await
            .unwrap();

        // Declare a table
        let mut declare_req = DeclareTableRequest::new();
        declare_req.id = Some(vec!["test_table".to_string()]);
        let response = namespace.declare_table(declare_req).await.unwrap();

        // Should return location
        assert!(response.location.is_some());
        let location = response.location.as_ref().unwrap();
        assert!(location.ends_with("test_table.lance"));

        // Table should exist (via reserved file)
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["test_table".to_string()]);
        assert!(namespace.table_exists(exists_req).await.is_ok());

        // Describe should work but return no version/schema (not written yet)
        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["test_table".to_string()]);
        let describe_response = namespace.describe_table(describe_req).await.unwrap();
        assert!(describe_response.location.is_some());
        assert!(describe_response.version.is_none()); // Not written yet
        assert!(describe_response.schema.is_none()); // Not written yet
    }

    #[tokio::test]
    async fn test_declare_table_with_manifest() {
        use lance_namespace::models::{DeclareTableRequest, TableExistsRequest};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Create namespace with manifest
        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(true)
            .dir_listing_enabled(false)
            .build()
            .await
            .unwrap();

        // Declare a table
        let mut declare_req = DeclareTableRequest::new();
        declare_req.id = Some(vec!["test_table".to_string()]);
        let response = namespace.declare_table(declare_req).await.unwrap();

        // Should return location
        assert!(response.location.is_some());

        // Table should exist in manifest
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["test_table".to_string()]);
        assert!(namespace.table_exists(exists_req).await.is_ok());
    }

    #[tokio::test]
    async fn test_declare_table_when_table_exists() {
        use lance_namespace::models::DeclareTableRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .build()
            .await
            .unwrap();

        // First create a table with actual data
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        namespace
            .create_table(create_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Try to declare the same table - should fail because it already has data
        let mut declare_req = DeclareTableRequest::new();
        declare_req.id = Some(vec!["test_table".to_string()]);
        let result = namespace.declare_table(declare_req).await;
        assert!(result.is_err());
    }

    // ============================================================
    // Tests for deregister_table in V1 mode
    // ============================================================

    #[tokio::test]
    async fn test_deregister_table_v1_mode() {
        use lance_namespace::models::{DeregisterTableRequest, TableExistsRequest};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Create namespace in V1 mode (no manifest, with dir listing)
        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .dir_listing_enabled(true)
            .build()
            .await
            .unwrap();

        // Create a table with data
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        namespace
            .create_table(create_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Verify table exists
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["test_table".to_string()]);
        assert!(namespace.table_exists(exists_req.clone()).await.is_ok());

        // Deregister the table
        let mut deregister_req = DeregisterTableRequest::new();
        deregister_req.id = Some(vec!["test_table".to_string()]);
        let response = namespace.deregister_table(deregister_req).await.unwrap();

        // Should return location
        assert!(response.location.is_some());
        let location = response.location.as_ref().unwrap();
        assert!(location.contains("test_table"));

        // Table should no longer exist (deregistered)
        let result = namespace.table_exists(exists_req).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("deregistered"));

        // Physical data should still exist
        let dataset = Dataset::open(location).await;
        assert!(dataset.is_ok(), "Physical table data should still exist");
    }

    #[tokio::test]
    async fn test_deregister_table_v1_already_deregistered() {
        use lance_namespace::models::DeregisterTableRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .dir_listing_enabled(true)
            .build()
            .await
            .unwrap();

        // Create a table
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        namespace
            .create_table(create_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Deregister once
        let mut deregister_req = DeregisterTableRequest::new();
        deregister_req.id = Some(vec!["test_table".to_string()]);
        namespace
            .deregister_table(deregister_req.clone())
            .await
            .unwrap();

        // Try to deregister again - should fail
        let result = namespace.deregister_table(deregister_req).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("already deregistered"));
    }

    // ============================================================
    // Tests for list_tables skipping deregistered tables
    // ============================================================

    #[tokio::test]
    async fn test_list_tables_skips_deregistered_v1() {
        use lance_namespace::models::DeregisterTableRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .dir_listing_enabled(true)
            .build()
            .await
            .unwrap();

        // Create two tables
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        let mut create_req1 = CreateTableRequest::new();
        create_req1.id = Some(vec!["table1".to_string()]);
        namespace
            .create_table(create_req1, bytes::Bytes::from(ipc_data.clone()))
            .await
            .unwrap();

        let mut create_req2 = CreateTableRequest::new();
        create_req2.id = Some(vec!["table2".to_string()]);
        namespace
            .create_table(create_req2, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // List tables - should see both (root namespace = empty vec)
        let mut list_req = ListTablesRequest::new();
        list_req.id = Some(vec![]);
        let list_response = namespace.list_tables(list_req.clone()).await.unwrap();
        assert_eq!(list_response.tables.len(), 2);

        // Deregister table1
        let mut deregister_req = DeregisterTableRequest::new();
        deregister_req.id = Some(vec!["table1".to_string()]);
        namespace.deregister_table(deregister_req).await.unwrap();

        // List tables - should only see table2
        let list_response = namespace.list_tables(list_req).await.unwrap();
        assert_eq!(list_response.tables.len(), 1);
        assert!(list_response.tables.contains(&"table2".to_string()));
        assert!(!list_response.tables.contains(&"table1".to_string()));
    }

    // ============================================================
    // Tests for describe_table and table_exists with deregistered tables
    // ============================================================

    #[tokio::test]
    async fn test_describe_table_fails_for_deregistered_v1() {
        use lance_namespace::models::{DeregisterTableRequest, DescribeTableRequest};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .dir_listing_enabled(true)
            .build()
            .await
            .unwrap();

        // Create a table
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        namespace
            .create_table(create_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Describe should work before deregistration
        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = Some(vec!["test_table".to_string()]);
        assert!(namespace.describe_table(describe_req.clone()).await.is_ok());

        // Deregister
        let mut deregister_req = DeregisterTableRequest::new();
        deregister_req.id = Some(vec!["test_table".to_string()]);
        namespace.deregister_table(deregister_req).await.unwrap();

        // Describe should fail after deregistration
        let result = namespace.describe_table(describe_req).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("deregistered"));
    }

    #[tokio::test]
    async fn test_table_exists_fails_for_deregistered_v1() {
        use lance_namespace::models::{DeregisterTableRequest, TableExistsRequest};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .dir_listing_enabled(true)
            .build()
            .await
            .unwrap();

        // Create a table
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        namespace
            .create_table(create_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Table exists should work before deregistration
        let mut exists_req = TableExistsRequest::new();
        exists_req.id = Some(vec!["test_table".to_string()]);
        assert!(namespace.table_exists(exists_req.clone()).await.is_ok());

        // Deregister
        let mut deregister_req = DeregisterTableRequest::new();
        deregister_req.id = Some(vec!["test_table".to_string()]);
        namespace.deregister_table(deregister_req).await.unwrap();

        // Table exists should fail after deregistration
        let result = namespace.table_exists(exists_req).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("deregistered"));
    }

    #[tokio::test]
    async fn test_atomic_table_status_check() {
        // This test verifies that the TableStatus check is atomic
        // by ensuring a single directory listing is used

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .dir_listing_enabled(true)
            .build()
            .await
            .unwrap();

        // Create a table
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);
        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec!["test_table".to_string()]);
        namespace
            .create_table(create_req, bytes::Bytes::from(ipc_data))
            .await
            .unwrap();

        // Table status should show exists=true, is_deregistered=false
        let status = namespace.check_table_status("test_table").await;
        assert!(status.exists);
        assert!(!status.is_deregistered);
        assert!(!status.has_reserved_file);
    }
}
