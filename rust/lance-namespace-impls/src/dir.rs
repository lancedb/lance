// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Directory-based Lance Namespace implementation.
//!
//! This module provides a directory-based implementation of the Lance namespace
//! that stores tables as Lance datasets in a filesystem directory structure.

use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use lance::dataset::{Dataset, WriteParams};
use opendal::Operator;

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

/// Connect to a directory-based namespace implementation.
///
/// This is a convenience wrapper around DirectoryNamespace::new that returns
/// the same type as connect for API consistency.
pub async fn connect_dir(properties: HashMap<String, String>) -> Result<Arc<dyn LanceNamespace>> {
    DirectoryNamespace::new(properties).map(|ns| Arc::new(ns) as Arc<dyn LanceNamespace>)
}

/// Configuration for DirectoryNamespace.
#[derive(Debug, Clone)]
pub struct DirectoryNamespaceConfig {
    /// Root directory for the namespace
    root: String,
    /// Storage options for the backend
    storage_options: HashMap<String, String>,
}

impl DirectoryNamespaceConfig {
    /// Property key for the root directory
    pub const ROOT: &'static str = "root";
    /// Prefix for storage options
    pub const STORAGE_OPTIONS_PREFIX: &'static str = "storage.";

    /// Create a new configuration from properties
    pub fn new(properties: HashMap<String, String>) -> Self {
        let root = properties
            .get(Self::ROOT)
            .cloned()
            .unwrap_or_else(|| {
                std::env::current_dir()
                    .unwrap()
                    .to_string_lossy()
                    .to_string()
            })
            .trim_end_matches('/')
            .to_string();

        let storage_options: HashMap<String, String> = properties
            .iter()
            .filter_map(|(k, v)| {
                k.strip_prefix(Self::STORAGE_OPTIONS_PREFIX)
                    .map(|key| (key.to_string(), v.clone()))
            })
            .collect();

        Self {
            root,
            storage_options,
        }
    }

    /// Get the root directory
    pub fn root(&self) -> &str {
        &self.root
    }

    /// Get the storage options
    pub fn storage_options(&self) -> &HashMap<String, String> {
        &self.storage_options
    }
}

/// Directory-based implementation of Lance Namespace.
///
/// This implementation stores tables as Lance datasets in a directory structure.
/// It supports local filesystems and cloud storage backends through OpenDAL.
pub struct DirectoryNamespace {
    config: DirectoryNamespaceConfig,
    operator: Operator,
}

impl DirectoryNamespace {
    /// Create a new DirectoryNamespace instance
    pub fn new(properties: HashMap<String, String>) -> Result<Self> {
        let config = DirectoryNamespaceConfig::new(properties);
        let operator = Self::initialize_operator(&config)?;

        Ok(Self { config, operator })
    }

    /// Initialize the OpenDAL operator based on the configuration
    fn initialize_operator(config: &DirectoryNamespaceConfig) -> Result<Operator> {
        let root = config.root();
        let storage_options = &config.storage_options;

        // Parse the root path to determine scheme and configuration
        let (scheme, opendal_config) = Self::parse_storage_path(root, storage_options)?;

        // Create the operator with the determined scheme and configuration
        let operator =
            Operator::via_iter(scheme, opendal_config).map_err(|e| Error::Namespace {
                source: format!("Failed to create operator: {}", e).into(),
                location: snafu::location!(),
            })?;

        Ok(operator)
    }

    /// Parse storage path and return scheme and configuration
    fn parse_storage_path(
        root: &str,
        storage_options: &HashMap<String, String>,
    ) -> Result<(opendal::Scheme, HashMap<String, String>)> {
        use url::Url;

        let mut config = HashMap::new();

        // Try to parse as URL, if it fails, treat as local filesystem path
        let (scheme, authority, path) = if let Ok(url) = Url::parse(root) {
            let scheme = Self::normalize_scheme(url.scheme());
            let authority = url.host_str().unwrap_or("");
            // For file:// and fs:// URLs, preserve the full path including leading slash
            // For cloud storage URLs, remove the leading slash
            let path = if scheme == "fs" || scheme == "file" {
                url.path().to_string()
            } else {
                url.path().trim_start_matches('/').to_string()
            };
            (scheme, authority.to_string(), path)
        } else {
            // Not a URL, treat as local filesystem - prepend fs://
            ("fs".to_string(), String::new(), root.to_string())
        };

        // Configure based on scheme
        let opendal_scheme = match scheme.as_str() {
            "fs" | "file" => {
                // For filesystem, use the full path (authority is empty for local paths)
                if authority.is_empty() {
                    config.insert("root".to_string(), path);
                } else {
                    // Handle file:///absolute/path or fs://hostname/path
                    config.insert("root".to_string(), format!("{}/{}", authority, path));
                }
                opendal::Scheme::Fs
            }
            "s3" => {
                config.insert("root".to_string(), path);
                config.insert("bucket".to_string(), authority);
                opendal::Scheme::S3
            }
            "gcs" => {
                config.insert("root".to_string(), path);
                config.insert("bucket".to_string(), authority);
                opendal::Scheme::Gcs
            }
            "azblob" => {
                config.insert("root".to_string(), path);
                config.insert("container".to_string(), authority);
                opendal::Scheme::Azblob
            }
            _ => {
                // For unknown schemes, try to parse as OpenDAL scheme
                config.insert("root".to_string(), path);
                if !authority.is_empty() {
                    config.insert("bucket".to_string(), authority);
                }
                opendal::Scheme::from_str(&scheme).map_err(|_| Error::Namespace {
                    source: format!("Unsupported storage scheme: {}", scheme).into(),
                    location: snafu::location!(),
                })?
            }
        };

        // Add storage options for all schemes
        config.extend(storage_options.clone());

        Ok((opendal_scheme, config))
    }

    /// Normalize scheme names with common aliases
    fn normalize_scheme(scheme: &str) -> String {
        match scheme.to_lowercase().as_str() {
            "s3a" | "s3n" => "s3".to_string(),
            "abfs" => "azblob".to_string(),
            "file" => "fs".to_string(),
            s => s.to_string(),
        }
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
                    "Directory namespace only supports single-level table IDs, but got: {:?}",
                    id
                )
                .into(),
                location: snafu::location!(),
            });
        }

        Ok(id[0].clone())
    }

    /// Get the full path for a table
    fn table_full_path(&self, table_name: &str) -> String {
        format!("{}/{}.lance", self.config.root(), table_name)
    }

    /// Get the versions path for a table
    fn table_versions_path(&self, table_name: &str) -> String {
        format!("{}.lance/_versions/", table_name)
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

        // Use non-recursive listing to avoid issues with object stores that don't have directory concept
        let entries = self.operator.list("").await.map_err(|e| Error::IO {
            source: box_error(std::io::Error::other(format!(
                "Failed to list directory: {}",
                e
            ))),
            location: snafu::location!(),
        })?;

        for entry in entries {
            let path = entry.path().trim_end_matches('/');

            // Only process directory-like paths that end with .lance
            if !path.ends_with(".lance") {
                continue;
            }

            // Extract table name (remove .lance suffix)
            let table_name = &path[..path.len() - 6];

            // Check if it's a valid Lance dataset or has .lance-reserved file
            let mut is_table = false;

            // First check for .lance-reserved file
            let reserved_file_path = format!("{}.lance/.lance-reserved", table_name);
            if self
                .operator
                .exists(&reserved_file_path)
                .await
                .unwrap_or(false)
            {
                is_table = true;
            }

            // If not found, check for _versions directory
            if !is_table {
                let versions_path = self.table_versions_path(table_name);
                if let Ok(version_entries) = self.operator.list(&versions_path).await {
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
        let table_path = self.table_full_path(&table_name);

        // Check if table exists - either as Lance dataset or with .lance-reserved file
        let mut table_exists = false;

        // First check for .lance-reserved file
        let reserved_file_path = format!("{}.lance/.lance-reserved", table_name);
        if self
            .operator
            .exists(&reserved_file_path)
            .await
            .unwrap_or(false)
        {
            table_exists = true;
        }

        // If not found, check if it's a Lance dataset by looking for _versions directory
        if !table_exists {
            let versions_path = self.table_versions_path(&table_name);
            if let Ok(entries) = self.operator.list(&versions_path).await {
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
            location: Some(table_path),
            schema: None,
            properties: None,
            storage_options: Some(self.config.storage_options.clone()),
        })
    }

    async fn table_exists(&self, request: TableExistsRequest) -> Result<()> {
        let table_name = Self::table_name_from_id(&request.id)?;

        // Check if table exists - either as Lance dataset or with .lance-reserved file
        let mut table_exists = false;

        // First check for .lance-reserved file
        let reserved_file_path = format!("{}.lance/.lance-reserved", table_name);
        if self
            .operator
            .exists(&reserved_file_path)
            .await
            .unwrap_or(false)
        {
            table_exists = true;
        }

        // If not found, check if it's a Lance dataset by looking for _versions directory
        if !table_exists {
            let versions_path = self.table_versions_path(&table_name);
            if let Ok(entries) = self.operator.list(&versions_path).await {
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

    async fn create_table(
        &self,
        request: CreateTableRequest,
        request_data: Bytes,
    ) -> Result<CreateTableResponse> {
        let table_name = Self::table_name_from_id(&request.id)?;
        let table_path = self.table_full_path(&table_name);

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
            if location != table_path {
                return Err(Error::Namespace {
                    source: format!(
                        "Cannot create table {} at location {}, must be at location {}",
                        table_name, location, table_path
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
        let write_params = WriteParams {
            mode: lance::dataset::WriteMode::Create,
            ..Default::default()
        };

        // Create the Lance dataset using the actual Lance API
        Dataset::write(reader, &table_path, Some(write_params))
            .await
            .map_err(|e| Error::Namespace {
                source: format!("Failed to create Lance dataset: {}", e).into(),
                location: snafu::location!(),
            })?;

        Ok(CreateTableResponse {
            version: Some(1),
            location: Some(table_path),
            properties: None,
            storage_options: Some(self.config.storage_options.clone()),
        })
    }

    async fn create_empty_table(
        &self,
        request: CreateEmptyTableRequest,
    ) -> Result<CreateEmptyTableResponse> {
        let table_name = Self::table_name_from_id(&request.id)?;
        let table_path = self.table_full_path(&table_name);

        // Validate location if provided
        if let Some(location) = &request.location {
            let location = location.trim_end_matches('/');
            if location != table_path {
                return Err(Error::Namespace {
                    source: format!(
                        "Cannot create table {} at location {}, must be at location {}",
                        table_name, location, table_path
                    )
                    .into(),
                    location: snafu::location!(),
                });
            }
        }

        // Create the .lance-reserved file to mark the table as existing
        let reserved_file_path = format!("{}.lance/.lance-reserved", table_name);
        self.operator
            .write(&reserved_file_path, Vec::<u8>::new())
            .await
            .map_err(|e| Error::Namespace {
                source: format!(
                    "Failed to create .lance-reserved file for table {}: {}",
                    table_name, e
                )
                .into(),
                location: snafu::location!(),
            })?;

        Ok(CreateEmptyTableResponse {
            location: Some(table_path),
            properties: None,
            storage_options: Some(self.config.storage_options.clone()),
        })
    }

    async fn drop_table(&self, request: DropTableRequest) -> Result<DropTableResponse> {
        let table_name = Self::table_name_from_id(&request.id)?;
        let table_path = self.table_full_path(&table_name);

        // Remove the entire table directory
        let table_dir = format!("{}.lance/", table_name);
        self.operator
            .remove_all(&table_dir)
            .await
            .map_err(|e| Error::Namespace {
                source: format!("Failed to drop table {}: {}", table_name, e).into(),
                location: snafu::location!(),
            })?;

        Ok(DropTableResponse {
            id: request.id,
            location: Some(table_path),
            properties: None,
            transaction_id: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_namespace::models::{JsonArrowDataType, JsonArrowField, JsonArrowSchema};
    use lance_namespace::schema::convert_json_arrow_schema;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tempfile::TempDir;

    /// Helper to create a test DirectoryNamespace with a temporary directory
    async fn create_test_namespace() -> (DirectoryNamespace, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let mut properties = HashMap::new();
        properties.insert(
            "root".to_string(),
            temp_dir.path().to_string_lossy().to_string(),
        );

        let namespace = DirectoryNamespace::new(properties).unwrap();
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
        let temp_dir = TempDir::new().unwrap();
        let custom_path = temp_dir.path().join("custom");
        std::fs::create_dir(&custom_path).unwrap();

        let mut properties = HashMap::new();
        properties.insert(
            "root".to_string(),
            custom_path.to_string_lossy().to_string(),
        );

        let namespace = DirectoryNamespace::new(properties).unwrap();

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
        let temp_dir = TempDir::new().unwrap();
        let mut properties = HashMap::new();
        properties.insert(
            "root".to_string(),
            temp_dir.path().to_string_lossy().to_string(),
        );
        properties.insert("storage.option1".to_string(), "value1".to_string());
        properties.insert("storage.option2".to_string(), "value2".to_string());

        let namespace = DirectoryNamespace::new(properties).unwrap();

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
        let temp_dir = TempDir::new().unwrap();
        let mut properties = HashMap::new();
        properties.insert(
            "root".to_string(),
            temp_dir.path().to_string_lossy().to_string(),
        );

        let namespace = connect_dir(properties).await.unwrap();

        // Test basic operation through the trait object
        let request = ListTablesRequest::new();
        let response = namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 0);
    }

    #[test]
    fn test_parse_storage_path_local() {
        let storage_options = HashMap::new();

        // Test local filesystem paths
        let (scheme, config) =
            DirectoryNamespace::parse_storage_path("/path/to/data", &storage_options).unwrap();
        assert!(matches!(scheme, opendal::Scheme::Fs));
        assert_eq!(config.get("root").unwrap(), "/path/to/data");

        // Test relative path
        let (scheme, config) =
            DirectoryNamespace::parse_storage_path("./data", &storage_options).unwrap();
        assert!(matches!(scheme, opendal::Scheme::Fs));
        assert_eq!(config.get("root").unwrap(), "./data");
    }

    #[test]
    fn test_parse_storage_path_s3() {
        let storage_options = HashMap::new();

        // Test S3 URL
        let (scheme, config) =
            DirectoryNamespace::parse_storage_path("s3://my-bucket/path/to/data", &storage_options)
                .unwrap();
        assert!(matches!(scheme, opendal::Scheme::S3));
        assert_eq!(config.get("bucket").unwrap(), "my-bucket");
        assert_eq!(config.get("root").unwrap(), "path/to/data");

        // Test S3 with just bucket
        let (scheme, config) =
            DirectoryNamespace::parse_storage_path("s3://my-bucket", &storage_options).unwrap();
        assert!(matches!(scheme, opendal::Scheme::S3));
        assert_eq!(config.get("bucket").unwrap(), "my-bucket");
        assert_eq!(config.get("root").unwrap(), "");
    }

    #[test]
    fn test_parse_storage_path_gcs() {
        let storage_options = HashMap::new();

        // Test GCS URL
        let (scheme, config) = DirectoryNamespace::parse_storage_path(
            "gcs://my-bucket/path/to/data",
            &storage_options,
        )
        .unwrap();
        assert!(matches!(scheme, opendal::Scheme::Gcs));
        assert_eq!(config.get("bucket").unwrap(), "my-bucket");
        assert_eq!(config.get("root").unwrap(), "path/to/data");
    }

    #[test]
    fn test_parse_storage_path_azblob() {
        let storage_options = HashMap::new();

        // Test Azure Blob URL
        let (scheme, config) = DirectoryNamespace::parse_storage_path(
            "azblob://my-container/path/to/data",
            &storage_options,
        )
        .unwrap();
        assert!(matches!(scheme, opendal::Scheme::Azblob));
        assert_eq!(config.get("container").unwrap(), "my-container");
        assert_eq!(config.get("root").unwrap(), "path/to/data");

        // Test with abfs alias
        let (scheme, config) =
            DirectoryNamespace::parse_storage_path("abfs://my-container/path", &storage_options)
                .unwrap();
        assert!(matches!(scheme, opendal::Scheme::Azblob));
        assert_eq!(config.get("container").unwrap(), "my-container");
        assert_eq!(config.get("root").unwrap(), "path");
    }

    #[test]
    fn test_normalize_scheme() {
        // Test scheme normalization
        assert_eq!(DirectoryNamespace::normalize_scheme("s3a"), "s3");
        assert_eq!(DirectoryNamespace::normalize_scheme("s3n"), "s3");
        assert_eq!(DirectoryNamespace::normalize_scheme("S3A"), "s3");
        assert_eq!(DirectoryNamespace::normalize_scheme("abfs"), "azblob");
        assert_eq!(DirectoryNamespace::normalize_scheme("ABFS"), "azblob");
        assert_eq!(DirectoryNamespace::normalize_scheme("file"), "fs");
        assert_eq!(DirectoryNamespace::normalize_scheme("FILE"), "fs");
        assert_eq!(DirectoryNamespace::normalize_scheme("gcs"), "gcs");
        assert_eq!(DirectoryNamespace::normalize_scheme("random"), "random");
    }

    #[test]
    fn test_fs_scheme_url() {
        let storage_options = HashMap::new();

        // Test file:// URLs
        let (scheme, config) =
            DirectoryNamespace::parse_storage_path("file:///absolute/path", &storage_options)
                .unwrap();
        assert!(matches!(scheme, opendal::Scheme::Fs));
        assert_eq!(config.get("root").unwrap(), "/absolute/path");

        // Test fs:// URLs
        let (scheme, config) =
            DirectoryNamespace::parse_storage_path("fs:///absolute/path", &storage_options)
                .unwrap();
        assert!(matches!(scheme, opendal::Scheme::Fs));
        assert_eq!(config.get("root").unwrap(), "/absolute/path");
    }

    #[test]
    fn test_storage_options_passed_through() {
        // Test that storage options are properly passed through parse_storage_path
        let mut storage_options = HashMap::new();
        storage_options.insert("aws_access_key_id".to_string(), "test_key".to_string());
        storage_options.insert(
            "aws_secret_access_key".to_string(),
            "test_secret".to_string(),
        );
        storage_options.insert("region".to_string(), "us-west-2".to_string());

        // Test with S3 - storage options should be included
        let (scheme, config) =
            DirectoryNamespace::parse_storage_path("s3://my-bucket/path", &storage_options)
                .unwrap();
        assert!(matches!(scheme, opendal::Scheme::S3));
        assert_eq!(config.get("bucket").unwrap(), "my-bucket");
        assert_eq!(config.get("root").unwrap(), "path");
        assert_eq!(config.get("aws_access_key_id").unwrap(), "test_key");
        assert_eq!(config.get("aws_secret_access_key").unwrap(), "test_secret");
        assert_eq!(config.get("region").unwrap(), "us-west-2");

        // Test with local filesystem - storage options should still be included
        let (scheme, config) =
            DirectoryNamespace::parse_storage_path("/local/path", &storage_options).unwrap();
        assert!(matches!(scheme, opendal::Scheme::Fs));
        assert_eq!(config.get("root").unwrap(), "/local/path");
        // Even for local fs, storage options should be passed through
        assert_eq!(config.get("aws_access_key_id").unwrap(), "test_key");
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
        let table_dir = temp_dir.path().join("empty_table.lance");
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
        let table_dir = temp_dir.path().join("empty_table_to_drop.lance");
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
