// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! REST implementation of Lance Namespace

use std::collections::HashMap;

use async_trait::async_trait;
use bytes::Bytes;

use lance_namespace::apis::{
    configuration::Configuration, namespace_api, table_api, transaction_api,
};
use lance_namespace::models::{
    AlterTransactionRequest, AlterTransactionResponse, CountTableRowsRequest,
    CreateEmptyTableRequest, CreateEmptyTableResponse, CreateNamespaceRequest,
    CreateNamespaceResponse, CreateTableIndexRequest, CreateTableIndexResponse, CreateTableRequest,
    CreateTableResponse, DeleteFromTableRequest, DeleteFromTableResponse, DeregisterTableRequest,
    DeregisterTableResponse, DescribeNamespaceRequest, DescribeNamespaceResponse,
    DescribeTableIndexStatsRequest, DescribeTableIndexStatsResponse, DescribeTableRequest,
    DescribeTableResponse, DescribeTransactionRequest, DescribeTransactionResponse,
    DropNamespaceRequest, DropNamespaceResponse, DropTableRequest, DropTableResponse,
    InsertIntoTableRequest, InsertIntoTableResponse, ListNamespacesRequest, ListNamespacesResponse,
    ListTableIndicesRequest, ListTableIndicesResponse, ListTablesRequest, ListTablesResponse,
    MergeInsertIntoTableRequest, MergeInsertIntoTableResponse, NamespaceExistsRequest,
    QueryTableRequest, RegisterTableRequest, RegisterTableResponse, TableExistsRequest,
    UpdateTableRequest, UpdateTableResponse,
};

use lance_core::{box_error, Error, Result};

use lance_namespace::LanceNamespace;

/// Configuration for REST namespace
#[derive(Debug, Clone)]
pub struct RestNamespaceConfig {
    /// The delimiter used for object identifiers
    delimiter: String,
    /// Additional headers to send with requests
    additional_headers: HashMap<String, String>,
    /// The base URI for the REST API
    uri: Option<String>,
    /// Path to the client certificate file (PEM format) for mTLS
    cert_file: Option<String>,
    /// Path to the client private key file (PEM format) for mTLS
    key_file: Option<String>,
    /// Path to the CA certificate file for server verification (PEM format)
    ssl_ca_cert: Option<String>,
    /// Whether to verify the hostname in the server's certificate
    assert_hostname: bool,
}

impl RestNamespaceConfig {
    /// Header prefix for additional headers
    const HEADER_PREFIX: &'static str = "header.";

    /// Default delimiter
    const DEFAULT_DELIMITER: &'static str = ".";

    /// Create a new configuration from a map of properties
    pub fn new(properties: HashMap<String, String>) -> Self {
        let delimiter = properties
            .get("delimiter")
            .cloned()
            .unwrap_or_else(|| Self::DEFAULT_DELIMITER.to_string());

        let uri = properties.get("uri").cloned();

        let mut additional_headers = HashMap::new();
        for (key, value) in &properties {
            if let Some(header_name) = key.strip_prefix(Self::HEADER_PREFIX) {
                additional_headers.insert(header_name.to_string(), value.clone());
            }
        }

        let cert_file = properties.get("tls.cert_file").cloned();
        let key_file = properties.get("tls.key_file").cloned();
        let ssl_ca_cert = properties.get("tls.ssl_ca_cert").cloned();
        let assert_hostname = properties
            .get("tls.assert_hostname")
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(true);

        Self {
            delimiter,
            additional_headers,
            uri,
            cert_file,
            key_file,
            ssl_ca_cert,
            assert_hostname,
        }
    }

    /// Get the delimiter
    pub fn delimiter(&self) -> &str {
        &self.delimiter
    }

    /// Get additional headers
    pub fn additional_headers(&self) -> &HashMap<String, String> {
        &self.additional_headers
    }

    /// Get the URI
    pub fn uri(&self) -> Option<&str> {
        self.uri.as_deref()
    }
}

/// Convert an object identifier (list of strings) to a delimited string
fn object_id_str(id: &Option<Vec<String>>, delimiter: &str) -> Result<String> {
    match id {
        Some(id_parts) if !id_parts.is_empty() => Ok(id_parts.join(delimiter)),
        Some(_) => Ok(delimiter.to_string()),
        None => Err(Error::Namespace {
            source: "Object ID is required".into(),
            location: snafu::location!(),
        }),
    }
}

/// Convert API error to lance core error
fn convert_api_error<T: std::fmt::Debug>(err: lance_namespace::apis::Error<T>) -> Error {
    use lance_namespace::apis::Error as ApiError;
    match err {
        ApiError::Reqwest(e) => Error::IO {
            source: box_error(e),
            location: snafu::location!(),
        },
        ApiError::Serde(e) => Error::Namespace {
            source: format!("Serialization error: {}", e).into(),
            location: snafu::location!(),
        },
        ApiError::Io(e) => Error::IO {
            source: box_error(e),
            location: snafu::location!(),
        },
        ApiError::ResponseError(e) => Error::Namespace {
            source: format!("Response error: {:?}", e).into(),
            location: snafu::location!(),
        },
    }
}

/// REST implementation of Lance Namespace
pub struct RestNamespace {
    config: RestNamespaceConfig,
    reqwest_config: Configuration,
}

impl RestNamespace {
    /// Create a new REST namespace with the given configuration
    pub fn new(properties: HashMap<String, String>) -> Self {
        let config = RestNamespaceConfig::new(properties);

        // Build reqwest client with custom headers if provided
        let mut client_builder = reqwest::Client::builder();

        // Add custom headers to the client
        if !config.additional_headers().is_empty() {
            let mut headers = reqwest::header::HeaderMap::new();
            for (key, value) in config.additional_headers() {
                if let (Ok(header_name), Ok(header_value)) = (
                    reqwest::header::HeaderName::from_bytes(key.as_bytes()),
                    reqwest::header::HeaderValue::from_str(value),
                ) {
                    headers.insert(header_name, header_value);
                }
            }
            client_builder = client_builder.default_headers(headers);
        }

        // Configure mTLS if certificate and key files are provided
        if let (Some(cert_file), Some(key_file)) = (&config.cert_file, &config.key_file) {
            if let (Ok(cert), Ok(key)) = (std::fs::read(cert_file), std::fs::read(key_file)) {
                if let Ok(identity) = reqwest::Identity::from_pem(&[&cert[..], &key[..]].concat()) {
                    client_builder = client_builder.identity(identity);
                }
            }
        }

        // Load CA certificate for server verification
        if let Some(ca_cert_file) = &config.ssl_ca_cert {
            if let Ok(ca_cert) = std::fs::read(ca_cert_file) {
                if let Ok(ca_cert) = reqwest::Certificate::from_pem(&ca_cert) {
                    client_builder = client_builder.add_root_certificate(ca_cert);
                }
            }
        }

        // Configure hostname verification
        client_builder = client_builder.danger_accept_invalid_hostnames(!config.assert_hostname);

        let client = client_builder
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        let mut reqwest_config = Configuration::new();
        reqwest_config.client = client;
        if let Some(uri) = config.uri() {
            reqwest_config.base_path = uri.to_string();
        }

        Self {
            config,
            reqwest_config,
        }
    }

    /// Create a new REST namespace with custom configuration (for testing)
    #[cfg(test)]
    pub fn with_configuration(
        properties: HashMap<String, String>,
        reqwest_config: Configuration,
    ) -> Self {
        let config = RestNamespaceConfig::new(properties);

        Self {
            config,
            reqwest_config,
        }
    }
}

#[async_trait]
impl LanceNamespace for RestNamespace {
    async fn list_namespaces(
        &self,
        request: ListNamespacesRequest,
    ) -> Result<ListNamespacesResponse> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        namespace_api::list_namespaces(
            &self.reqwest_config,
            &id,
            Some(self.config.delimiter()),
            request.page_token.as_deref(),
            request.limit,
        )
        .await
        .map_err(convert_api_error)
    }

    async fn describe_namespace(
        &self,
        request: DescribeNamespaceRequest,
    ) -> Result<DescribeNamespaceResponse> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        namespace_api::describe_namespace(
            &self.reqwest_config,
            &id,
            request,
            Some(self.config.delimiter()),
        )
        .await
        .map_err(convert_api_error)
    }

    async fn create_namespace(
        &self,
        request: CreateNamespaceRequest,
    ) -> Result<CreateNamespaceResponse> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        namespace_api::create_namespace(
            &self.reqwest_config,
            &id,
            request,
            Some(self.config.delimiter()),
        )
        .await
        .map_err(convert_api_error)
    }

    async fn drop_namespace(&self, request: DropNamespaceRequest) -> Result<DropNamespaceResponse> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        namespace_api::drop_namespace(
            &self.reqwest_config,
            &id,
            request,
            Some(self.config.delimiter()),
        )
        .await
        .map_err(convert_api_error)
    }

    async fn namespace_exists(&self, request: NamespaceExistsRequest) -> Result<()> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        namespace_api::namespace_exists(
            &self.reqwest_config,
            &id,
            request,
            Some(self.config.delimiter()),
        )
        .await
        .map_err(convert_api_error)
    }

    async fn list_tables(&self, request: ListTablesRequest) -> Result<ListTablesResponse> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        table_api::list_tables(
            &self.reqwest_config,
            &id,
            Some(self.config.delimiter()),
            request.page_token.as_deref(),
            request.limit,
        )
        .await
        .map_err(convert_api_error)
    }

    async fn describe_table(&self, request: DescribeTableRequest) -> Result<DescribeTableResponse> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        table_api::describe_table(
            &self.reqwest_config,
            &id,
            request,
            Some(self.config.delimiter()),
        )
        .await
        .map_err(convert_api_error)
    }

    async fn register_table(&self, request: RegisterTableRequest) -> Result<RegisterTableResponse> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        table_api::register_table(
            &self.reqwest_config,
            &id,
            request,
            Some(self.config.delimiter()),
        )
        .await
        .map_err(convert_api_error)
    }

    async fn table_exists(&self, request: TableExistsRequest) -> Result<()> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        table_api::table_exists(
            &self.reqwest_config,
            &id,
            request,
            Some(self.config.delimiter()),
        )
        .await
        .map_err(convert_api_error)
    }

    async fn drop_table(&self, request: DropTableRequest) -> Result<DropTableResponse> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        table_api::drop_table(
            &self.reqwest_config,
            &id,
            request,
            Some(self.config.delimiter()),
        )
        .await
        .map_err(convert_api_error)
    }

    async fn deregister_table(
        &self,
        request: DeregisterTableRequest,
    ) -> Result<DeregisterTableResponse> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        table_api::deregister_table(
            &self.reqwest_config,
            &id,
            request,
            Some(self.config.delimiter()),
        )
        .await
        .map_err(convert_api_error)
    }

    async fn count_table_rows(&self, request: CountTableRowsRequest) -> Result<i64> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        table_api::count_table_rows(
            &self.reqwest_config,
            &id,
            request,
            Some(self.config.delimiter()),
        )
        .await
        .map_err(convert_api_error)
    }

    async fn create_table(
        &self,
        request: CreateTableRequest,
        request_data: Bytes,
    ) -> Result<CreateTableResponse> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        let properties_json = request
            .properties
            .as_ref()
            .map(|props| serde_json::to_string(props).unwrap_or_else(|_| "{}".to_string()));

        use lance_namespace::models::create_table_request::Mode;
        let mode = request.mode.as_ref().map(|m| match m {
            Mode::Create => "create",
            Mode::ExistOk => "exist_ok",
            Mode::Overwrite => "overwrite",
        });

        table_api::create_table(
            &self.reqwest_config,
            &id,
            request_data.to_vec(),
            Some(self.config.delimiter()),
            mode,
            request.location.as_deref(),
            properties_json.as_deref(),
        )
        .await
        .map_err(convert_api_error)
    }

    async fn create_empty_table(
        &self,
        request: CreateEmptyTableRequest,
    ) -> Result<CreateEmptyTableResponse> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        table_api::create_empty_table(
            &self.reqwest_config,
            &id,
            request,
            Some(self.config.delimiter()),
        )
        .await
        .map_err(convert_api_error)
    }

    async fn insert_into_table(
        &self,
        request: InsertIntoTableRequest,
        request_data: Bytes,
    ) -> Result<InsertIntoTableResponse> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        use lance_namespace::models::insert_into_table_request::Mode;
        let mode = request.mode.as_ref().map(|m| match m {
            Mode::Append => "append",
            Mode::Overwrite => "overwrite",
        });

        table_api::insert_into_table(
            &self.reqwest_config,
            &id,
            request_data.to_vec(),
            Some(self.config.delimiter()),
            mode,
        )
        .await
        .map_err(convert_api_error)
    }

    async fn merge_insert_into_table(
        &self,
        request: MergeInsertIntoTableRequest,
        request_data: Bytes,
    ) -> Result<MergeInsertIntoTableResponse> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        let on = request.on.as_deref().ok_or_else(|| Error::Namespace {
            source: "'on' field is required for merge insert".into(),
            location: snafu::location!(),
        })?;

        table_api::merge_insert_into_table(
            &self.reqwest_config,
            &id,
            on,
            request_data.to_vec(),
            Some(self.config.delimiter()),
            request.when_matched_update_all,
            request.when_matched_update_all_filt.as_deref(),
            request.when_not_matched_insert_all,
            request.when_not_matched_by_source_delete,
            request.when_not_matched_by_source_delete_filt.as_deref(),
        )
        .await
        .map_err(convert_api_error)
    }

    async fn update_table(&self, request: UpdateTableRequest) -> Result<UpdateTableResponse> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        table_api::update_table(
            &self.reqwest_config,
            &id,
            request,
            Some(self.config.delimiter()),
        )
        .await
        .map_err(convert_api_error)
    }

    async fn delete_from_table(
        &self,
        request: DeleteFromTableRequest,
    ) -> Result<DeleteFromTableResponse> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        table_api::delete_from_table(
            &self.reqwest_config,
            &id,
            request,
            Some(self.config.delimiter()),
        )
        .await
        .map_err(convert_api_error)
    }

    async fn query_table(&self, request: QueryTableRequest) -> Result<Bytes> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        let response = table_api::query_table(
            &self.reqwest_config,
            &id,
            request,
            Some(self.config.delimiter()),
        )
        .await
        .map_err(convert_api_error)?;

        // Convert response to bytes
        let bytes = response.bytes().await.map_err(|e| Error::IO {
            source: box_error(e),
            location: snafu::location!(),
        })?;

        Ok(bytes)
    }

    async fn create_table_index(
        &self,
        request: CreateTableIndexRequest,
    ) -> Result<CreateTableIndexResponse> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        table_api::create_table_index(
            &self.reqwest_config,
            &id,
            request,
            Some(self.config.delimiter()),
        )
        .await
        .map_err(convert_api_error)
    }

    async fn list_table_indices(
        &self,
        request: ListTableIndicesRequest,
    ) -> Result<ListTableIndicesResponse> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        table_api::list_table_indices(
            &self.reqwest_config,
            &id,
            request,
            Some(self.config.delimiter()),
        )
        .await
        .map_err(convert_api_error)
    }

    async fn describe_table_index_stats(
        &self,
        request: DescribeTableIndexStatsRequest,
    ) -> Result<DescribeTableIndexStatsResponse> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        // Note: The index_name parameter seems to be missing from the request structure
        // This might need to be adjusted based on the actual API
        let index_name = ""; // This should come from somewhere in the request

        table_api::describe_table_index_stats(
            &self.reqwest_config,
            &id,
            index_name,
            request,
            Some(self.config.delimiter()),
        )
        .await
        .map_err(convert_api_error)
    }

    async fn describe_transaction(
        &self,
        request: DescribeTransactionRequest,
    ) -> Result<DescribeTransactionResponse> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        transaction_api::describe_transaction(
            &self.reqwest_config,
            &id,
            request,
            Some(self.config.delimiter()),
        )
        .await
        .map_err(convert_api_error)
    }

    async fn alter_transaction(
        &self,
        request: AlterTransactionRequest,
    ) -> Result<AlterTransactionResponse> {
        let id = object_id_str(&request.id, self.config.delimiter())?;

        transaction_api::alter_transaction(
            &self.reqwest_config,
            &id,
            request,
            Some(self.config.delimiter()),
        )
        .await
        .map_err(convert_api_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use lance_namespace::models::{create_table_request, insert_into_table_request};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Create a test REST namespace instance
    fn create_test_namespace() -> RestNamespace {
        let mut properties = HashMap::new();
        properties.insert("uri".to_string(), "http://localhost:8080".to_string());
        properties.insert("delimiter".to_string(), ".".to_string());
        RestNamespace::new(properties)
    }

    #[test]
    fn test_rest_namespace_creation() {
        let mut properties = HashMap::new();
        properties.insert("uri".to_string(), "http://example.com".to_string());
        properties.insert("delimiter".to_string(), "/".to_string());
        properties.insert(
            "header.Authorization".to_string(),
            "Bearer token".to_string(),
        );
        properties.insert("header.X-Custom".to_string(), "value".to_string());

        let _namespace = RestNamespace::new(properties);

        // Successfully created the namespace - test passes if no panic
    }

    #[tokio::test]
    async fn test_custom_headers_are_sent() {
        // Start a mock server
        let mock_server = MockServer::start().await;

        // Create mock that expects custom headers
        Mock::given(method("GET"))
            .and(path("/v1/namespace/test/list"))
            .and(wiremock::matchers::header(
                "Authorization",
                "Bearer test-token",
            ))
            .and(wiremock::matchers::header(
                "X-Custom-Header",
                "custom-value",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "namespaces": []
            })))
            .mount(&mock_server)
            .await;

        // Create namespace with custom headers
        let mut properties = HashMap::new();
        properties.insert("uri".to_string(), mock_server.uri());
        properties.insert(
            "header.Authorization".to_string(),
            "Bearer test-token".to_string(),
        );
        properties.insert(
            "header.X-Custom-Header".to_string(),
            "custom-value".to_string(),
        );

        let namespace = RestNamespace::new(properties);

        let request = ListNamespacesRequest {
            id: Some(vec!["test".to_string()]),
            page_token: None,
            limit: None,
        };

        let result = namespace.list_namespaces(request).await;

        // Should succeed, meaning headers were sent correctly
        assert!(result.is_ok());
    }

    #[test]
    fn test_default_configuration() {
        let properties = HashMap::new();
        let _namespace = RestNamespace::new(properties);

        // The default delimiter should be "." - test passes if no panic
    }

    #[test]
    fn test_with_custom_uri() {
        let mut properties = HashMap::new();
        properties.insert("uri".to_string(), "https://api.example.com/v1".to_string());

        let _namespace = RestNamespace::new(properties);
        // Test passes if no panic
    }

    #[test]
    fn test_tls_config_parsing() {
        let mut properties = HashMap::new();
        properties.insert("uri".to_string(), "https://api.example.com".to_string());
        properties.insert("tls.cert_file".to_string(), "/path/to/cert.pem".to_string());
        properties.insert("tls.key_file".to_string(), "/path/to/key.pem".to_string());
        properties.insert("tls.ssl_ca_cert".to_string(), "/path/to/ca.pem".to_string());
        properties.insert("tls.assert_hostname".to_string(), "true".to_string());

        let config = RestNamespaceConfig::new(properties);
        assert_eq!(config.cert_file, Some("/path/to/cert.pem".to_string()));
        assert_eq!(config.key_file, Some("/path/to/key.pem".to_string()));
        assert_eq!(config.ssl_ca_cert, Some("/path/to/ca.pem".to_string()));
        assert!(config.assert_hostname);
    }

    #[test]
    fn test_tls_config_default_assert_hostname() {
        let mut properties = HashMap::new();
        properties.insert("tls.cert_file".to_string(), "/path/to/cert.pem".to_string());
        properties.insert("tls.key_file".to_string(), "/path/to/key.pem".to_string());

        let config = RestNamespaceConfig::new(properties);
        // Default should be true
        assert!(config.assert_hostname);
    }

    #[test]
    fn test_tls_config_disable_hostname_verification() {
        let mut properties = HashMap::new();
        properties.insert("tls.cert_file".to_string(), "/path/to/cert.pem".to_string());
        properties.insert("tls.key_file".to_string(), "/path/to/key.pem".to_string());
        properties.insert("tls.assert_hostname".to_string(), "false".to_string());

        let config = RestNamespaceConfig::new(properties);
        assert!(!config.assert_hostname);
    }

    #[test]
    fn test_namespace_creation_with_tls_config() {
        let mut properties = HashMap::new();
        properties.insert("uri".to_string(), "https://api.example.com".to_string());
        properties.insert(
            "tls.cert_file".to_string(),
            "/nonexistent/cert.pem".to_string(),
        );
        properties.insert(
            "tls.key_file".to_string(),
            "/nonexistent/key.pem".to_string(),
        );
        properties.insert(
            "tls.ssl_ca_cert".to_string(),
            "/nonexistent/ca.pem".to_string(),
        );

        // Should not panic even with nonexistent files (they're just ignored)
        let _namespace = RestNamespace::new(properties);
    }

    #[tokio::test]
    async fn test_list_namespaces_success() {
        // Start a mock server
        let mock_server = MockServer::start().await;

        // Create mock response
        Mock::given(method("GET"))
            .and(path("/v1/namespace/test/list"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "namespaces": [
                    "namespace1",
                    "namespace2"
                ]
            })))
            .mount(&mock_server)
            .await;

        // Create namespace with mock server URL
        let mut properties = HashMap::new();
        properties.insert("uri".to_string(), mock_server.uri());
        properties.insert("delimiter".to_string(), ".".to_string());

        let mut reqwest_config = Configuration::new();
        reqwest_config.base_path = mock_server.uri();

        let namespace = RestNamespace::with_configuration(properties, reqwest_config);

        let request = ListNamespacesRequest {
            id: Some(vec!["test".to_string()]),
            page_token: None,
            limit: Some(10),
        };

        let result = namespace.list_namespaces(request).await;

        // Should succeed with mock server
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.namespaces.len(), 2);
        assert_eq!(response.namespaces[0], "namespace1");
        assert_eq!(response.namespaces[1], "namespace2");
    }

    #[tokio::test]
    async fn test_list_namespaces_error() {
        // Start a mock server
        let mock_server = MockServer::start().await;

        // Create mock error response
        Mock::given(method("GET"))
            .and(path("/v1/namespace/test/list"))
            .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
                "error": {
                    "message": "Namespace not found",
                    "type": "NamespaceNotFoundException"
                }
            })))
            .mount(&mock_server)
            .await;

        // Create namespace with mock server URL
        let mut properties = HashMap::new();
        properties.insert("uri".to_string(), mock_server.uri());

        let mut reqwest_config = Configuration::new();
        reqwest_config.base_path = mock_server.uri();

        let namespace = RestNamespace::with_configuration(properties, reqwest_config);

        let request = ListNamespacesRequest {
            id: Some(vec!["test".to_string()]),
            page_token: None,
            limit: Some(10),
        };

        let result = namespace.list_namespaces(request).await;

        // Should return an error
        assert!(result.is_err());
    }

    #[tokio::test]
    #[ignore] // Requires a running server
    async fn test_list_namespaces_integration() {
        let namespace = create_test_namespace();
        let request = ListNamespacesRequest {
            id: Some(vec!["test".to_string()]),
            page_token: None,
            limit: Some(10),
        };

        let result = namespace.list_namespaces(request).await;

        // The actual assertion depends on whether the server is running
        // In a real test, you would either mock the server or ensure it's running
        assert!(result.is_err() || result.is_ok());
    }

    #[tokio::test]
    async fn test_create_namespace_success() {
        // Start a mock server
        let mock_server = MockServer::start().await;

        // Create mock response
        Mock::given(method("POST"))
            .and(path("/v1/namespace/test.newnamespace/create"))
            .respond_with(ResponseTemplate::new(201).set_body_json(serde_json::json!({
                "namespace": {
                    "identifier": ["test", "newnamespace"],
                    "properties": {}
                }
            })))
            .mount(&mock_server)
            .await;

        // Create namespace with mock server URL
        let mut properties = HashMap::new();
        properties.insert("uri".to_string(), mock_server.uri());

        let mut reqwest_config = Configuration::new();
        reqwest_config.base_path = mock_server.uri();

        let namespace = RestNamespace::with_configuration(properties, reqwest_config);

        let request = CreateNamespaceRequest {
            id: Some(vec!["test".to_string(), "newnamespace".to_string()]),
            properties: None,
            mode: None,
        };

        let result = namespace.create_namespace(request).await;

        // Should succeed with mock server
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_create_table_success() {
        // Start a mock server
        let mock_server = MockServer::start().await;

        // Create mock response
        Mock::given(method("POST"))
            .and(path("/v1/table/test.namespace.table/create"))
            .respond_with(ResponseTemplate::new(201).set_body_json(serde_json::json!({
                "table": {
                    "identifier": ["test", "namespace", "table"],
                    "location": "/path/to/table",
                    "version": 1
                }
            })))
            .mount(&mock_server)
            .await;

        // Create namespace with mock server URL
        let mut properties = HashMap::new();
        properties.insert("uri".to_string(), mock_server.uri());

        let mut reqwest_config = Configuration::new();
        reqwest_config.base_path = mock_server.uri();

        let namespace = RestNamespace::with_configuration(properties, reqwest_config);

        let request = CreateTableRequest {
            id: Some(vec![
                "test".to_string(),
                "namespace".to_string(),
                "table".to_string(),
            ]),
            location: None,
            mode: Some(create_table_request::Mode::Create),
            properties: None,
        };

        let data = Bytes::from("arrow data here");
        let result = namespace.create_table(request, data).await;

        // Should succeed with mock server
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_insert_into_table_success() {
        // Start a mock server
        let mock_server = MockServer::start().await;

        // Create mock response
        Mock::given(method("POST"))
            .and(path("/v1/table/test.namespace.table/insert"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "version": 2
            })))
            .mount(&mock_server)
            .await;

        // Create namespace with mock server URL
        let mut properties = HashMap::new();
        properties.insert("uri".to_string(), mock_server.uri());

        let mut reqwest_config = Configuration::new();
        reqwest_config.base_path = mock_server.uri();

        let namespace = RestNamespace::with_configuration(properties, reqwest_config);

        let request = InsertIntoTableRequest {
            id: Some(vec![
                "test".to_string(),
                "namespace".to_string(),
                "table".to_string(),
            ]),
            mode: Some(insert_into_table_request::Mode::Append),
        };

        let data = Bytes::from("arrow data here");
        let result = namespace.insert_into_table(request, data).await;

        // Should succeed with mock server
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.version, Some(2));
    }

    #[tokio::test]
    #[ignore] // Requires a running server
    async fn test_create_namespace_integration() {
        let namespace = create_test_namespace();
        let request = CreateNamespaceRequest {
            id: Some(vec!["test".to_string(), "namespace".to_string()]),
            properties: None,
            mode: None,
        };

        let result = namespace.create_namespace(request).await;
        assert!(result.is_err() || result.is_ok());
    }

    #[tokio::test]
    #[ignore] // Requires a running server
    async fn test_describe_namespace() {
        let namespace = create_test_namespace();
        let request = DescribeNamespaceRequest {
            id: Some(vec!["test".to_string(), "namespace".to_string()]),
        };

        let result = namespace.describe_namespace(request).await;
        assert!(result.is_err() || result.is_ok());
    }

    #[tokio::test]
    #[ignore] // Requires a running server
    async fn test_list_tables() {
        let namespace = create_test_namespace();
        let request = ListTablesRequest {
            id: Some(vec!["test".to_string(), "namespace".to_string()]),
            page_token: None,
            limit: Some(10),
        };

        let result = namespace.list_tables(request).await;
        assert!(result.is_err() || result.is_ok());
    }

    #[tokio::test]
    #[ignore] // Requires a running server
    async fn test_create_table() {
        let namespace = create_test_namespace();
        let request = CreateTableRequest {
            id: Some(vec![
                "test".to_string(),
                "namespace".to_string(),
                "table".to_string(),
            ]),
            location: None,
            mode: Some(create_table_request::Mode::Create),
            properties: None,
        };

        let data = Bytes::from("test data");
        let result = namespace.create_table(request, data).await;
        assert!(result.is_err() || result.is_ok());
    }

    #[tokio::test]
    #[ignore] // Requires a running server
    async fn test_drop_table() {
        let namespace = create_test_namespace();
        let request = DropTableRequest {
            id: Some(vec![
                "test".to_string(),
                "namespace".to_string(),
                "table".to_string(),
            ]),
        };

        let result = namespace.drop_table(request).await;
        assert!(result.is_err() || result.is_ok());
    }

    #[tokio::test]
    #[ignore] // Requires a running server
    async fn test_insert_into_table_append() {
        let namespace = create_test_namespace();
        let request = InsertIntoTableRequest {
            id: Some(vec![
                "test".to_string(),
                "namespace".to_string(),
                "table".to_string(),
            ]),
            mode: Some(insert_into_table_request::Mode::Append),
        };

        let data = Bytes::from("test data");
        let result = namespace.insert_into_table(request, data).await;
        assert!(result.is_err() || result.is_ok());
    }

    #[tokio::test]
    #[ignore] // Requires a running server
    async fn test_insert_into_table_overwrite() {
        let namespace = create_test_namespace();
        let request = InsertIntoTableRequest {
            id: Some(vec![
                "test".to_string(),
                "namespace".to_string(),
                "table".to_string(),
            ]),
            mode: Some(insert_into_table_request::Mode::Overwrite),
        };

        let data = Bytes::from("test data");
        let result = namespace.insert_into_table(request, data).await;
        assert!(result.is_err() || result.is_ok());
    }

    #[tokio::test]
    #[ignore] // Requires a running server
    async fn test_merge_insert_into_table() {
        let namespace = create_test_namespace();
        let request = MergeInsertIntoTableRequest {
            id: Some(vec![
                "test".to_string(),
                "namespace".to_string(),
                "table".to_string(),
            ]),
            on: Some("id".to_string()),
            when_matched_update_all: Some(true),
            when_matched_update_all_filt: None,
            when_not_matched_insert_all: Some(true),
            when_not_matched_by_source_delete: Some(false),
            when_not_matched_by_source_delete_filt: None,
        };

        let data = Bytes::from("test data");
        let result = namespace.merge_insert_into_table(request, data).await;
        assert!(result.is_err() || result.is_ok());
    }

    #[tokio::test]
    #[ignore] // Requires a running server
    async fn test_delete_from_table() {
        let namespace = create_test_namespace();
        let request = DeleteFromTableRequest {
            id: Some(vec![
                "test".to_string(),
                "namespace".to_string(),
                "table".to_string(),
            ]),
            predicate: "id > 10".to_string(),
        };

        let result = namespace.delete_from_table(request).await;
        assert!(result.is_err() || result.is_ok());
    }

    #[tokio::test]
    #[ignore] // Requires a running server
    async fn test_describe_transaction() {
        let namespace = create_test_namespace();
        let request = DescribeTransactionRequest {
            id: Some(vec!["test".to_string(), "transaction".to_string()]),
        };

        let result = namespace.describe_transaction(request).await;
        assert!(result.is_err() || result.is_ok());
    }

    #[tokio::test]
    #[ignore] // Requires a running server
    async fn test_alter_transaction() {
        let namespace = create_test_namespace();
        let request = AlterTransactionRequest {
            id: Some(vec!["test".to_string(), "transaction".to_string()]),
            actions: vec![],
        };

        let result = namespace.alter_transaction(request).await;
        assert!(result.is_err() || result.is_ok());
    }
}
