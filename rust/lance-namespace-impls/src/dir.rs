// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Directory-based Lance Namespace implementation.
//!
//! This module provides a directory-based implementation of the Lance namespace
//! that stores tables as Lance datasets in a filesystem directory structure.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use lance::dataset::{Dataset, WriteParams};
use lance::session::Session;
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use object_store::path::Path;

use lance_namespace::models::{
    CreateEmptyTableRequest, CreateEmptyTableResponse, CreateNamespaceRequest,
    CreateNamespaceResponse, CreateTableRequest, CreateTableResponse, DescribeNamespaceRequest,
    DescribeNamespaceResponse, DescribeTableRequest, DescribeTableResponse, DropNamespaceRequest,
    DropNamespaceResponse, DropTableRequest, DropTableResponse, ListNamespacesRequest,
    ListNamespacesResponse, ListTablesRequest, ListTablesResponse, NamespaceExistsRequest,
    TableExistsRequest,
};

use lance_core::{box_error, Error, Result};
use lance_namespace::LanceNamespace;

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
        }
    }

    /// Create a DirectoryNamespaceBuilder from properties HashMap.
    ///
    /// This method parses a properties map into builder configuration.
    /// It expects:
    /// - `root`: The root directory path (required)
    /// - `storage.*`: Storage options (optional, prefix will be stripped)
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

        Ok(Self {
            root: root.trim_end_matches('/').to_string(),
            storage_options,
            session,
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

        Ok(DirectoryNamespace {
            root: self.root,
            storage_options: self.storage_options,
            session: self.session,
            object_store,
            base_path,
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
pub struct DirectoryNamespace {
    root: String,
    storage_options: Option<HashMap<String, String>>,
    #[allow(dead_code)]
    session: Option<Arc<Session>>,
    object_store: Arc<ObjectStore>,
    base_path: Path,
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
                    "Directory namespace only supports single-level table IDs, but got: {:?}",
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

    /// Get the versions directory path for a table
    fn table_versions_path(&self, table_name: &str) -> Path {
        // Need to chain child calls to avoid URL encoding the slash
        self.base_path
            .child(format!("{}.lance", table_name).as_str())
            .child("_versions")
    }

    /// Get the reserved file path for a table
    fn table_reserved_file_path(&self, table_name: &str) -> Path {
        // Need to chain child calls to avoid URL encoding the slash
        self.base_path
            .child(format!("{}.lance", table_name).as_str())
            .child(".lance-reserved")
    }
}

#[async_trait]
impl LanceNamespace for DirectoryNamespace {
    async fn list_namespaces(
        &self,
        request: ListNamespacesRequest,
    ) -> Result<ListNamespacesResponse> {
        // Validate this is a request for the root namespace
        Self::validate_root_namespace_id(&request.id)?;

        // Directory namespace only contains the root namespace (empty list)
        Ok(ListNamespacesResponse::new(vec![]))
    }

    async fn describe_namespace(
        &self,
        request: DescribeNamespaceRequest,
    ) -> Result<DescribeNamespaceResponse> {
        // Validate this is a request for the root namespace
        Self::validate_root_namespace_id(&request.id)?;

        // Return description of the root namespace
        Ok(DescribeNamespaceResponse {
            properties: Some(HashMap::new()),
        })
    }

    async fn create_namespace(
        &self,
        request: CreateNamespaceRequest,
    ) -> Result<CreateNamespaceResponse> {
        // Root namespace always exists and cannot be created
        if request.id.is_none() || request.id.as_ref().unwrap().is_empty() {
            return Err(Error::Namespace {
                source: "Root namespace already exists and cannot be created".into(),
                location: snafu::location!(),
            });
        }

        // Non-root namespaces are not supported
        Err(Error::NotSupported {
            source: "Directory namespace only supports the root namespace".into(),
            location: snafu::location!(),
        })
    }

    async fn drop_namespace(&self, request: DropNamespaceRequest) -> Result<DropNamespaceResponse> {
        // Root namespace always exists and cannot be dropped
        if request.id.is_none() || request.id.as_ref().unwrap().is_empty() {
            return Err(Error::Namespace {
                source: "Root namespace cannot be dropped".into(),
                location: snafu::location!(),
            });
        }

        // Non-root namespaces are not supported
        Err(Error::NotSupported {
            source: "Directory namespace only supports the root namespace".into(),
            location: snafu::location!(),
        })
    }

    async fn namespace_exists(&self, request: NamespaceExistsRequest) -> Result<()> {
        // Root namespace always exists
        if request.id.is_none() || request.id.as_ref().unwrap().is_empty() {
            return Ok(());
        }

        // Non-root namespaces don't exist
        Err(Error::Namespace {
            source: "Only root namespace exists in directory namespace".into(),
            location: snafu::location!(),
        })
    }

    async fn list_tables(&self, request: ListTablesRequest) -> Result<ListTablesResponse> {
        Self::validate_root_namespace_id(&request.id)?;

        let mut tables = Vec::new();

        // List all entries in the base directory
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

            // Only process directory-like paths that end with .lance
            if !path.ends_with(".lance") {
                continue;
            }

            // Extract table name (remove .lance suffix)
            let table_name = &path[..path.len() - 6];

            // Check if it's a valid Lance dataset or has .lance-reserved file
            let mut is_table = false;

            // First check for .lance-reserved file
            let reserved_file_path = self.table_reserved_file_path(table_name);
            if self
                .object_store
                .exists(&reserved_file_path)
                .await
                .unwrap_or(false)
            {
                is_table = true;
            }

            // If not found, check for _versions directory
            if !is_table {
                let versions_path = self.table_versions_path(table_name);
                if let Ok(version_entries) = self.object_store.read_dir(versions_path).await {
                    // If there's at least one version file, it's a valid Lance dataset
                    if !version_entries.is_empty() {
                        is_table = true;
                    }
                }
            }

            if is_table {
                tables.push(table_name.to_string());
            }
        }

        let response = ListTablesResponse::new(tables);
        Ok(response)
    }

    async fn describe_table(&self, request: DescribeTableRequest) -> Result<DescribeTableResponse> {
        let table_name = Self::table_name_from_id(&request.id)?;
        let table_uri = self.table_full_uri(&table_name);

        // Check if table exists - either as Lance dataset or with .lance-reserved file
        let mut table_exists = false;

        // First check for .lance-reserved file
        let reserved_file_path = self.table_reserved_file_path(&table_name);
        if self
            .object_store
            .exists(&reserved_file_path)
            .await
            .unwrap_or(false)
        {
            table_exists = true;
        }

        // If not found, check if it's a Lance dataset by looking for _versions directory
        if !table_exists {
            let versions_path = self.table_versions_path(&table_name);
            if let Ok(entries) = self.object_store.read_dir(versions_path).await {
                if !entries.is_empty() {
                    table_exists = true;
                }
            }
        }

        if !table_exists {
            return Err(Error::Namespace {
                source: format!("Table does not exist: {}", table_name).into(),
                location: snafu::location!(),
            });
        }

        Ok(DescribeTableResponse {
            version: None,
            location: Some(table_uri),
            schema: None,
            properties: None,
            storage_options: self.storage_options.clone(),
        })
    }

    async fn table_exists(&self, request: TableExistsRequest) -> Result<()> {
        let table_name = Self::table_name_from_id(&request.id)?;

        // Check if table exists - either as Lance dataset or with .lance-reserved file
        let mut table_exists = false;

        // First check for .lance-reserved file
        let reserved_file_path = self.table_reserved_file_path(&table_name);
        if self
            .object_store
            .exists(&reserved_file_path)
            .await
            .unwrap_or(false)
        {
            table_exists = true;
        }

        // If not found, check if it's a Lance dataset by looking for _versions directory
        if !table_exists {
            let versions_path = self.table_versions_path(&table_name);
            if let Ok(entries) = self.object_store.read_dir(versions_path).await {
                if !entries.is_empty() {
                    table_exists = true;
                }
            }
        }

        if !table_exists {
            return Err(Error::Namespace {
                source: format!("Table does not exist: {}", table_name).into(),
                location: snafu::location!(),
            });
        }

        Ok(())
    }

    async fn drop_table(&self, request: DropTableRequest) -> Result<DropTableResponse> {
        let table_name = Self::table_name_from_id(&request.id)?;
        let table_uri = self.table_full_uri(&table_name);

        // Remove the entire table directory
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
        let table_name = Self::table_name_from_id(&request.id)?;
        let table_uri = self.table_full_uri(&table_name);

        // Validate that request_data is provided and is a valid Arrow IPC stream
        if request_data.is_empty() {
            return Err(Error::Namespace {
                source: "Request data (Arrow IPC stream) is required for create_table".into(),
                location: snafu::location!(),
            });
        }

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

        // Parse the Arrow IPC stream from request_data
        use arrow::ipc::reader::StreamReader;
        use std::io::Cursor;

        let cursor = Cursor::new(request_data.to_vec());
        let stream_reader = StreamReader::try_new(cursor, None).map_err(|e| Error::Namespace {
            source: format!("Invalid Arrow IPC stream: {}", e).into(),
            location: snafu::location!(),
        })?;

        // Extract schema from the IPC stream
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
            // If no batches in the stream, create an empty batch with the schema
            let batch = arrow::record_batch::RecordBatch::new_empty(arrow_schema.clone());
            let batches = vec![Ok(batch)];
            arrow::record_batch::RecordBatchIterator::new(batches, arrow_schema.clone())
        } else {
            // Convert to RecordBatchIterator
            let batch_results: Vec<_> = batches.into_iter().map(Ok).collect();
            arrow::record_batch::RecordBatchIterator::new(batch_results, arrow_schema)
        };

        // Set up write parameters for creating a new dataset
        // Populate store_params with storage options to ensure they're forwarded to Dataset::write
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
            version: Some(1),
            location: Some(table_uri),
            properties: None,
            storage_options: self.storage_options.clone(),
        })
    }

    async fn create_empty_table(
        &self,
        request: CreateEmptyTableRequest,
    ) -> Result<CreateEmptyTableResponse> {
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

        // Create the .lance-reserved file to mark the table as existing
        let reserved_file_path = self.table_reserved_file_path(&table_name);

        self.object_store
            .create(&reserved_file_path)
            .await
            .map_err(|e| Error::Namespace {
                source: format!(
                    "Failed to create .lance-reserved file for table {}: {}",
                    table_name, e
                )
                .into(),
                location: snafu::location!(),
            })?
            .shutdown()
            .await
            .map_err(|e| Error::Namespace {
                source: format!(
                    "Failed to finalize .lance-reserved file for table {}: {}",
                    table_name, e
                )
                .into(),
                location: snafu::location!(),
            })?;

        Ok(CreateEmptyTableResponse {
            location: Some(table_uri),
            properties: None,
            storage_options: self.storage_options.clone(),
        })
    }

    fn namespace_id(&self) -> String {
        format!("DirectoryNamespace {{ root: {:?} }}", self.root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_core::utils::tempfile::TempStdDir;
    use lance_namespace::models::{JsonArrowDataType, JsonArrowField, JsonArrowSchema};
    use lance_namespace::schema::convert_json_arrow_schema;
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

        // Test with multi-level ID
        let mut request = CreateTableRequest::new();
        request.id = Some(vec!["namespace".to_string(), "table".to_string()]);

        let result = namespace
            .create_table(request, bytes::Bytes::from(ipc_data))
            .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("single-level table IDs"));
    }

    #[tokio::test]
    async fn test_create_table_with_wrong_location() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Create test IPC data
        let schema = create_test_schema();
        let ipc_data = create_test_ipc_data(&schema);

        let mut request = CreateTableRequest::new();
        request.id = Some(vec!["test_table".to_string()]);
        request.location = Some("/wrong/path/table.lance".to_string());

        let result = namespace
            .create_table(request, bytes::Bytes::from(ipc_data))
            .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must be at location"));
    }

    #[tokio::test]
    async fn test_list_tables() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        // Initially, no tables
        let request = ListTablesRequest::new();
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
        let request = ListTablesRequest::new();
        let response = namespace.list_tables(request).await.unwrap();
        let tables = response.tables;
        assert_eq!(tables.len(), 2);
        assert!(tables.contains(&"table1".to_string()));
        assert!(tables.contains(&"table2".to_string()));
    }

    #[tokio::test]
    async fn test_list_tables_with_namespace_id() {
        let (namespace, _temp_dir) = create_test_namespace().await;

        let mut request = ListTablesRequest::new();
        request.id = Some(vec!["namespace".to_string()]);

        let result = namespace.list_tables(request).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("root namespace operations"));
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
        let request = ListNamespacesRequest::new();
        let result = namespace.list_namespaces(request).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().namespaces.len(), 0);

        // Test describe_namespace - should succeed for root
        let request = DescribeNamespaceRequest::new();
        let result = namespace.describe_namespace(request).await;
        assert!(result.is_ok());

        // Test namespace_exists - root always exists
        let request = NamespaceExistsRequest::new();
        let result = namespace.namespace_exists(request).await;
        assert!(result.is_ok());

        // Test create_namespace - root cannot be created
        let request = CreateNamespaceRequest::new();
        let result = namespace.create_namespace(request).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));

        // Test drop_namespace - root cannot be dropped
        let request = DropNamespaceRequest::new();
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

        // Test create_namespace for non-root - not supported
        let mut request = CreateNamespaceRequest::new();
        request.id = Some(vec!["child".to_string()]);
        let result = namespace.create_namespace(request).await;
        assert!(matches!(result, Err(Error::NotSupported { .. })));

        // Test namespace_exists for non-root - should not exist
        let mut request = NamespaceExistsRequest::new();
        request.id = Some(vec!["child".to_string()]);
        let result = namespace.namespace_exists(request).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Only root namespace exists"));

        // Test drop_namespace for non-root - not supported
        let mut request = DropNamespaceRequest::new();
        request.id = Some(vec!["child".to_string()]);
        let result = namespace.drop_namespace(request).await;
        assert!(matches!(result, Err(Error::NotSupported { .. })));
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
        let request = ListTablesRequest::new();
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
        let list_request = ListTablesRequest::new();
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
}
