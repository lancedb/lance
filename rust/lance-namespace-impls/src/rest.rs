// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! REST implementation of Lance Namespace

use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use reqwest::header::{HeaderName, HeaderValue};

use crate::context::{DynamicContextProvider, OperationInfo};

use lance_namespace::apis::urlencode;
use lance_namespace::models::{
    AlterTableAddColumnsRequest, AlterTableAddColumnsResponse, AlterTableAlterColumnsRequest,
    AlterTableAlterColumnsResponse, AlterTableDropColumnsRequest, AlterTableDropColumnsResponse,
    AlterTransactionRequest, AlterTransactionResponse, AnalyzeTableQueryPlanRequest,
    CountTableRowsRequest, CreateEmptyTableRequest, CreateEmptyTableResponse,
    CreateNamespaceRequest, CreateNamespaceResponse, CreateTableIndexRequest,
    CreateTableIndexResponse, CreateTableRequest, CreateTableResponse,
    CreateTableScalarIndexResponse, CreateTableTagRequest, CreateTableTagResponse,
    DeclareTableRequest, DeclareTableResponse, DeleteFromTableRequest, DeleteFromTableResponse,
    DeleteTableTagRequest, DeleteTableTagResponse, DeregisterTableRequest, DeregisterTableResponse,
    DescribeNamespaceRequest, DescribeNamespaceResponse, DescribeTableIndexStatsRequest,
    DescribeTableIndexStatsResponse, DescribeTableRequest, DescribeTableResponse,
    DescribeTransactionRequest, DescribeTransactionResponse, DropNamespaceRequest,
    DropNamespaceResponse, DropTableIndexRequest, DropTableIndexResponse, DropTableRequest,
    DropTableResponse, ExplainTableQueryPlanRequest, GetTableStatsRequest, GetTableStatsResponse,
    GetTableTagVersionRequest, GetTableTagVersionResponse, InsertIntoTableRequest,
    InsertIntoTableResponse, ListNamespacesRequest, ListNamespacesResponse,
    ListTableIndicesRequest, ListTableIndicesResponse, ListTableTagsRequest, ListTableTagsResponse,
    ListTableVersionsRequest, ListTableVersionsResponse, ListTablesRequest, ListTablesResponse,
    MergeInsertIntoTableRequest, MergeInsertIntoTableResponse, NamespaceExistsRequest,
    QueryTableRequest, RegisterTableRequest, RegisterTableResponse, RenameTableRequest,
    RenameTableResponse, RestoreTableRequest, RestoreTableResponse, TableExistsRequest,
    UpdateTableRequest, UpdateTableResponse, UpdateTableSchemaMetadataRequest,
    UpdateTableSchemaMetadataResponse, UpdateTableTagRequest, UpdateTableTagResponse,
};
use serde::{de::DeserializeOwned, Serialize};

use lance_core::{box_error, Error, Result};

use lance_namespace::LanceNamespace;

/// HTTP client wrapper that supports per-request header injection.
///
/// This client wraps a single `reqwest::Client` and applies dynamic headers
/// to each request without recreating the client. This is more efficient than
/// creating a new client per request when using a `DynamicContextProvider`.
///
/// The design follows lancedb's `RestfulLanceDbClient` pattern where headers
/// are applied to the built request using `headers_mut()` before execution.
#[derive(Clone)]
struct RestClient {
    client: reqwest::Client,
    base_path: String,
    base_headers: HashMap<String, String>,
    context_provider: Option<Arc<dyn DynamicContextProvider>>,
}

impl std::fmt::Debug for RestClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RestClient")
            .field("base_path", &self.base_path)
            .field("base_headers", &self.base_headers)
            .field(
                "context_provider",
                &self.context_provider.as_ref().map(|_| "Some(...)"),
            )
            .finish()
    }
}

impl RestClient {
    /// Apply base headers and dynamic context headers to a request.
    ///
    /// This method mutates the request's headers directly, which is more efficient
    /// than creating a new client with default_headers for each request.
    fn apply_headers(&self, request: &mut reqwest::Request, operation: &str, object_id: &str) {
        let request_headers = request.headers_mut();

        // First apply base headers
        for (key, value) in &self.base_headers {
            if let (Ok(header_name), Ok(header_value)) =
                (HeaderName::from_str(key), HeaderValue::from_str(value))
            {
                request_headers.insert(header_name, header_value);
            }
        }

        // Then apply context headers (override base headers if conflict)
        if let Some(provider) = &self.context_provider {
            let info = OperationInfo::new(operation, object_id);
            let context = provider.provide_context(&info);

            const HEADERS_PREFIX: &str = "headers.";
            for (key, value) in context {
                if let Some(header_name) = key.strip_prefix(HEADERS_PREFIX) {
                    if let (Ok(header_name), Ok(header_value)) = (
                        HeaderName::from_str(header_name),
                        HeaderValue::from_str(&value),
                    ) {
                        request_headers.insert(header_name, header_value);
                    }
                }
            }
        }
    }

    /// Execute a request with dynamic headers applied.
    ///
    /// This method builds the request, applies headers, and executes it.
    async fn execute(
        &self,
        req_builder: reqwest::RequestBuilder,
        operation: &str,
        object_id: &str,
    ) -> std::result::Result<reqwest::Response, reqwest::Error> {
        let mut request = req_builder.build()?;
        self.apply_headers(&mut request, operation, object_id);
        self.client.execute(request).await
    }

    /// Get the base path URL
    fn base_path(&self) -> &str {
        &self.base_path
    }

    /// Get a reference to the underlying reqwest client
    fn client(&self) -> &reqwest::Client {
        &self.client
    }
}

/// Builder for creating a RestNamespace.
///
/// This builder provides a fluent API for configuring and establishing
/// connections to REST-based Lance namespaces.
///
/// # Examples
///
/// ```no_run
/// # use lance_namespace_impls::RestNamespaceBuilder;
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a REST namespace
/// let namespace = RestNamespaceBuilder::new("http://localhost:8080")
///     .delimiter(".")
///     .header("Authorization", "Bearer token")
///     .build();
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct RestNamespaceBuilder {
    uri: String,
    delimiter: String,
    headers: HashMap<String, String>,
    cert_file: Option<String>,
    key_file: Option<String>,
    ssl_ca_cert: Option<String>,
    assert_hostname: bool,
    context_provider: Option<Arc<dyn DynamicContextProvider>>,
}

impl std::fmt::Debug for RestNamespaceBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RestNamespaceBuilder")
            .field("uri", &self.uri)
            .field("delimiter", &self.delimiter)
            .field("headers", &self.headers)
            .field("cert_file", &self.cert_file)
            .field("key_file", &self.key_file)
            .field("ssl_ca_cert", &self.ssl_ca_cert)
            .field("assert_hostname", &self.assert_hostname)
            .field(
                "context_provider",
                &self.context_provider.as_ref().map(|_| "Some(...)"),
            )
            .finish()
    }
}

impl RestNamespaceBuilder {
    /// Default delimiter for object identifiers
    const DEFAULT_DELIMITER: &'static str = "$";

    /// Create a new RestNamespaceBuilder with the specified URI.
    ///
    /// # Arguments
    ///
    /// * `uri` - Base URI for the REST API
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            delimiter: Self::DEFAULT_DELIMITER.to_string(),
            headers: HashMap::new(),
            cert_file: None,
            key_file: None,
            ssl_ca_cert: None,
            assert_hostname: true,
            context_provider: None,
        }
    }

    /// Create a RestNamespaceBuilder from properties HashMap.
    ///
    /// This method parses a properties map into builder configuration.
    /// It expects:
    /// - `uri`: The base URI for the REST API (required)
    /// - `delimiter`: Delimiter for object identifiers (optional, defaults to ".")
    /// - `header.*`: Additional headers (optional, prefix will be stripped)
    /// - `tls.cert_file`: Path to client certificate file (optional)
    /// - `tls.key_file`: Path to client private key file (optional)
    /// - `tls.ssl_ca_cert`: Path to CA certificate file (optional)
    /// - `tls.assert_hostname`: Whether to verify hostname (optional, defaults to true)
    ///
    /// # Arguments
    ///
    /// * `properties` - Configuration properties
    ///
    /// # Returns
    ///
    /// Returns a `RestNamespaceBuilder` instance.
    ///
    /// # Errors
    ///
    /// Returns an error if the `uri` property is missing.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use lance_namespace_impls::RestNamespaceBuilder;
    /// # use std::collections::HashMap;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut properties = HashMap::new();
    /// properties.insert("uri".to_string(), "http://localhost:8080".to_string());
    /// properties.insert("delimiter".to_string(), "/".to_string());
    /// properties.insert("header.Authorization".to_string(), "Bearer token".to_string());
    ///
    /// let namespace = RestNamespaceBuilder::from_properties(properties)?
    ///     .build();
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_properties(properties: HashMap<String, String>) -> Result<Self> {
        // Extract URI (required)
        let uri = properties
            .get("uri")
            .cloned()
            .ok_or_else(|| Error::Namespace {
                source: "Missing required property 'uri' for REST namespace".into(),
                location: snafu::location!(),
            })?;

        // Extract delimiter (optional)
        let delimiter = properties
            .get("delimiter")
            .cloned()
            .unwrap_or_else(|| Self::DEFAULT_DELIMITER.to_string());

        // Extract headers (properties prefixed with "header.")
        let mut headers = HashMap::new();
        for (key, value) in &properties {
            if let Some(header_name) = key.strip_prefix("header.") {
                headers.insert(header_name.to_string(), value.clone());
            }
        }

        // Extract TLS options
        let cert_file = properties.get("tls.cert_file").cloned();
        let key_file = properties.get("tls.key_file").cloned();
        let ssl_ca_cert = properties.get("tls.ssl_ca_cert").cloned();
        let assert_hostname = properties
            .get("tls.assert_hostname")
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(true);

        Ok(Self {
            uri,
            delimiter,
            headers,
            cert_file,
            key_file,
            ssl_ca_cert,
            assert_hostname,
            context_provider: None,
        })
    }

    /// Set the delimiter for object identifiers.
    ///
    /// # Arguments
    ///
    /// * `delimiter` - Delimiter string (e.g., ".", "/")
    pub fn delimiter(mut self, delimiter: impl Into<String>) -> Self {
        self.delimiter = delimiter.into();
        self
    }

    /// Add a custom header to the HTTP requests.
    ///
    /// # Arguments
    ///
    /// * `name` - Header name
    /// * `value` - Header value
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    /// Add multiple custom headers to the HTTP requests.
    ///
    /// # Arguments
    ///
    /// * `headers` - HashMap of headers to add
    pub fn headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers.extend(headers);
        self
    }

    /// Set the client certificate file for mTLS.
    ///
    /// # Arguments
    ///
    /// * `cert_file` - Path to the certificate file (PEM format)
    pub fn cert_file(mut self, cert_file: impl Into<String>) -> Self {
        self.cert_file = Some(cert_file.into());
        self
    }

    /// Set the client private key file for mTLS.
    ///
    /// # Arguments
    ///
    /// * `key_file` - Path to the private key file (PEM format)
    pub fn key_file(mut self, key_file: impl Into<String>) -> Self {
        self.key_file = Some(key_file.into());
        self
    }

    /// Set the CA certificate file for server verification.
    ///
    /// # Arguments
    ///
    /// * `ssl_ca_cert` - Path to the CA certificate file (PEM format)
    pub fn ssl_ca_cert(mut self, ssl_ca_cert: impl Into<String>) -> Self {
        self.ssl_ca_cert = Some(ssl_ca_cert.into());
        self
    }

    /// Set whether to verify the hostname in the server's certificate.
    ///
    /// # Arguments
    ///
    /// * `assert_hostname` - Whether to verify hostname
    pub fn assert_hostname(mut self, assert_hostname: bool) -> Self {
        self.assert_hostname = assert_hostname;
        self
    }

    /// Set a dynamic context provider for per-request context.
    ///
    /// The provider will be called before each HTTP request to generate
    /// additional context. Context keys that start with `headers.` are converted
    /// to HTTP headers by stripping the prefix. For example, `headers.Authorization`
    /// becomes the `Authorization` header. Keys without the `headers.` prefix are ignored.
    ///
    /// # Arguments
    ///
    /// * `provider` - The context provider implementation
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use lance_namespace_impls::{RestNamespaceBuilder, DynamicContextProvider, OperationInfo};
    /// use std::collections::HashMap;
    /// use std::sync::Arc;
    ///
    /// #[derive(Debug)]
    /// struct MyProvider;
    ///
    /// impl DynamicContextProvider for MyProvider {
    ///     fn provide_context(&self, info: &OperationInfo) -> HashMap<String, String> {
    ///         let mut ctx = HashMap::new();
    ///         ctx.insert("auth-token".to_string(), "my-token".to_string());
    ///         ctx
    ///     }
    /// }
    ///
    /// let namespace = RestNamespaceBuilder::new("http://localhost:8080")
    ///     .context_provider(Arc::new(MyProvider))
    ///     .build();
    /// ```
    pub fn context_provider(mut self, provider: Arc<dyn DynamicContextProvider>) -> Self {
        self.context_provider = Some(provider);
        self
    }

    /// Build the RestNamespace.
    ///
    /// # Returns
    ///
    /// Returns a `RestNamespace` instance.
    pub fn build(self) -> RestNamespace {
        RestNamespace::from_builder(self)
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

/// REST implementation of Lance Namespace
///
/// # Examples
///
/// ```no_run
/// # use lance_namespace_impls::RestNamespaceBuilder;
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Use the builder to create a namespace
/// let namespace = RestNamespaceBuilder::new("http://localhost:8080")
///     .build();
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct RestNamespace {
    delimiter: String,
    /// REST client that handles per-request header injection efficiently.
    rest_client: RestClient,
}

impl std::fmt::Debug for RestNamespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.namespace_id())
    }
}

impl std::fmt::Display for RestNamespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.namespace_id())
    }
}

impl RestNamespace {
    /// Create a new REST namespace from builder
    pub(crate) fn from_builder(builder: RestNamespaceBuilder) -> Self {
        // Build reqwest client WITHOUT default headers - we'll apply headers per-request
        let mut client_builder = reqwest::Client::builder();

        // Configure mTLS if certificate and key files are provided
        if let (Some(cert_file), Some(key_file)) = (&builder.cert_file, &builder.key_file) {
            if let (Ok(cert), Ok(key)) = (std::fs::read(cert_file), std::fs::read(key_file)) {
                if let Ok(identity) = reqwest::Identity::from_pem(&[&cert[..], &key[..]].concat()) {
                    client_builder = client_builder.identity(identity);
                }
            }
        }

        // Load CA certificate for server verification
        if let Some(ca_cert_file) = &builder.ssl_ca_cert {
            if let Ok(ca_cert) = std::fs::read(ca_cert_file) {
                if let Ok(ca_cert) = reqwest::Certificate::from_pem(&ca_cert) {
                    client_builder = client_builder.add_root_certificate(ca_cert);
                }
            }
        }

        // Configure hostname verification
        client_builder = client_builder.danger_accept_invalid_hostnames(!builder.assert_hostname);

        let client = client_builder
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        // Create the RestClient that handles per-request header injection
        let rest_client = RestClient {
            client,
            base_path: builder.uri,
            base_headers: builder.headers,
            context_provider: builder.context_provider,
        };

        Self {
            delimiter: builder.delimiter,
            rest_client,
        }
    }

    /// Execute a GET request and parse JSON response.
    async fn get_json<T: DeserializeOwned>(
        &self,
        path: &str,
        query: &[(&str, &str)],
        operation: &str,
        object_id: &str,
    ) -> Result<T> {
        let url = format!("{}{}", self.rest_client.base_path(), path);
        let req_builder = self.rest_client.client().get(&url).query(query);

        let resp = self
            .rest_client
            .execute(req_builder, operation, object_id)
            .await
            .map_err(|e| Error::IO {
                source: box_error(e),
                location: snafu::location!(),
            })?;

        let status = resp.status();
        let content = resp.text().await.map_err(|e| Error::IO {
            source: box_error(e),
            location: snafu::location!(),
        })?;

        if status.is_success() {
            serde_json::from_str(&content).map_err(|e| Error::Namespace {
                source: format!("Failed to parse response: {}", e).into(),
                location: snafu::location!(),
            })
        } else {
            Err(Error::Namespace {
                source: format!("Response error: status={}, content={}", status, content).into(),
                location: snafu::location!(),
            })
        }
    }

    /// Execute a POST request with JSON body and parse JSON response.
    async fn post_json<T: Serialize, R: DeserializeOwned>(
        &self,
        path: &str,
        query: &[(&str, &str)],
        body: &T,
        operation: &str,
        object_id: &str,
    ) -> Result<R> {
        let url = format!("{}{}", self.rest_client.base_path(), path);
        let req_builder = self.rest_client.client().post(&url).query(query).json(body);

        let resp = self
            .rest_client
            .execute(req_builder, operation, object_id)
            .await
            .map_err(|e| Error::IO {
                source: box_error(e),
                location: snafu::location!(),
            })?;

        let status = resp.status();
        let content = resp.text().await.map_err(|e| Error::IO {
            source: box_error(e),
            location: snafu::location!(),
        })?;

        if status.is_success() {
            serde_json::from_str(&content).map_err(|e| Error::Namespace {
                source: format!("Failed to parse response: {}", e).into(),
                location: snafu::location!(),
            })
        } else {
            Err(Error::Namespace {
                source: format!("Response error: status={}, content={}", status, content).into(),
                location: snafu::location!(),
            })
        }
    }

    /// Execute a POST request that returns nothing (204 No Content expected).
    async fn post_json_no_content<T: Serialize>(
        &self,
        path: &str,
        query: &[(&str, &str)],
        body: &T,
        operation: &str,
        object_id: &str,
    ) -> Result<()> {
        let url = format!("{}{}", self.rest_client.base_path(), path);
        let req_builder = self.rest_client.client().post(&url).query(query).json(body);

        let resp = self
            .rest_client
            .execute(req_builder, operation, object_id)
            .await
            .map_err(|e| Error::IO {
                source: box_error(e),
                location: snafu::location!(),
            })?;

        let status = resp.status();
        if status.is_success() {
            Ok(())
        } else {
            let content = resp.text().await.map_err(|e| Error::IO {
                source: box_error(e),
                location: snafu::location!(),
            })?;
            Err(Error::Namespace {
                source: format!("Response error: status={}, content={}", status, content).into(),
                location: snafu::location!(),
            })
        }
    }

    /// Execute a POST request with binary body and parse JSON response.
    async fn post_binary_json<R: DeserializeOwned>(
        &self,
        path: &str,
        query: &[(&str, &str)],
        body: Vec<u8>,
        operation: &str,
        object_id: &str,
    ) -> Result<R> {
        let url = format!("{}{}", self.rest_client.base_path(), path);
        let req_builder = self.rest_client.client().post(&url).query(query).body(body);

        let resp = self
            .rest_client
            .execute(req_builder, operation, object_id)
            .await
            .map_err(|e| Error::IO {
                source: box_error(e),
                location: snafu::location!(),
            })?;

        let status = resp.status();
        let content = resp.text().await.map_err(|e| Error::IO {
            source: box_error(e),
            location: snafu::location!(),
        })?;

        if status.is_success() {
            serde_json::from_str(&content).map_err(|e| Error::Namespace {
                source: format!("Failed to parse response: {}", e).into(),
                location: snafu::location!(),
            })
        } else {
            Err(Error::Namespace {
                source: format!("Response error: status={}, content={}", status, content).into(),
                location: snafu::location!(),
            })
        }
    }

    /// Execute a POST request with JSON body and get binary response.
    #[allow(dead_code)]
    async fn post_json_binary<T: Serialize>(
        &self,
        path: &str,
        query: &[(&str, &str)],
        body: &T,
        operation: &str,
        object_id: &str,
    ) -> Result<Bytes> {
        let url = format!("{}{}", self.rest_client.base_path(), path);
        let req_builder = self.rest_client.client().post(&url).query(query).json(body);

        let resp = self
            .rest_client
            .execute(req_builder, operation, object_id)
            .await
            .map_err(|e| Error::IO {
                source: box_error(e),
                location: snafu::location!(),
            })?;

        let status = resp.status();
        if status.is_success() {
            resp.bytes().await.map_err(|e| Error::IO {
                source: box_error(e),
                location: snafu::location!(),
            })
        } else {
            let content = resp.text().await.map_err(|e| Error::IO {
                source: box_error(e),
                location: snafu::location!(),
            })?;
            Err(Error::Namespace {
                source: format!("Response error: status={}, content={}", status, content).into(),
                location: snafu::location!(),
            })
        }
    }

    /// Get the base endpoint URL for this namespace
    pub fn endpoint(&self) -> &str {
        self.rest_client.base_path()
    }
}

#[async_trait]
impl LanceNamespace for RestNamespace {
    async fn list_namespaces(
        &self,
        request: ListNamespacesRequest,
    ) -> Result<ListNamespacesResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/namespace/{}/list", encoded_id);
        let mut query = vec![("delimiter", self.delimiter.as_str())];
        let page_token_str;
        if let Some(ref pt) = request.page_token {
            page_token_str = pt.clone();
            query.push(("page_token", page_token_str.as_str()));
        }
        let limit_str;
        if let Some(limit) = request.limit {
            limit_str = limit.to_string();
            query.push(("limit", limit_str.as_str()));
        }
        self.get_json(&path, &query, "list_namespaces", &id).await
    }

    async fn describe_namespace(
        &self,
        request: DescribeNamespaceRequest,
    ) -> Result<DescribeNamespaceResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/namespace/{}/describe", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "describe_namespace", &id)
            .await
    }

    async fn create_namespace(
        &self,
        request: CreateNamespaceRequest,
    ) -> Result<CreateNamespaceResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/namespace/{}/create", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "create_namespace", &id)
            .await
    }

    async fn drop_namespace(&self, request: DropNamespaceRequest) -> Result<DropNamespaceResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/namespace/{}/drop", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "drop_namespace", &id)
            .await
    }

    async fn namespace_exists(&self, request: NamespaceExistsRequest) -> Result<()> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/namespace/{}/exists", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json_no_content(&path, &query, &request, "namespace_exists", &id)
            .await
    }

    async fn list_tables(&self, request: ListTablesRequest) -> Result<ListTablesResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/namespace/{}/table/list", encoded_id);
        let mut query = vec![("delimiter", self.delimiter.as_str())];
        let page_token_str;
        if let Some(ref pt) = request.page_token {
            page_token_str = pt.clone();
            query.push(("page_token", page_token_str.as_str()));
        }
        let limit_str;
        if let Some(limit) = request.limit {
            limit_str = limit.to_string();
            query.push(("limit", limit_str.as_str()));
        }
        self.get_json(&path, &query, "list_tables", &id).await
    }

    async fn describe_table(&self, request: DescribeTableRequest) -> Result<DescribeTableResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/describe", encoded_id);
        let mut query = vec![("delimiter", self.delimiter.as_str())];
        let with_uri_str;
        if let Some(with_uri) = request.with_table_uri {
            with_uri_str = with_uri.to_string();
            query.push(("with_table_uri", with_uri_str.as_str()));
        }
        let detailed_str;
        if let Some(detailed) = request.load_detailed_metadata {
            detailed_str = detailed.to_string();
            query.push(("load_detailed_metadata", detailed_str.as_str()));
        }
        self.post_json(&path, &query, &request, "describe_table", &id)
            .await
    }

    async fn register_table(&self, request: RegisterTableRequest) -> Result<RegisterTableResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/register", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "register_table", &id)
            .await
    }

    async fn table_exists(&self, request: TableExistsRequest) -> Result<()> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/exists", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json_no_content(&path, &query, &request, "table_exists", &id)
            .await
    }

    async fn drop_table(&self, request: DropTableRequest) -> Result<DropTableResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/drop", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "drop_table", &id)
            .await
    }

    async fn deregister_table(
        &self,
        request: DeregisterTableRequest,
    ) -> Result<DeregisterTableResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/deregister", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "deregister_table", &id)
            .await
    }

    async fn count_table_rows(&self, request: CountTableRowsRequest) -> Result<i64> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/count_rows", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.get_json(&path, &query, "count_table_rows", &id).await
    }

    async fn create_table(
        &self,
        request: CreateTableRequest,
        request_data: Bytes,
    ) -> Result<CreateTableResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/create", encoded_id);
        let mut query = vec![("delimiter", self.delimiter.as_str())];
        let mode_str;
        if let Some(ref mode) = request.mode {
            mode_str = mode.clone();
            query.push(("mode", mode_str.as_str()));
        }
        self.post_binary_json(&path, &query, request_data.to_vec(), "create_table", &id)
            .await
    }

    async fn create_empty_table(
        &self,
        request: CreateEmptyTableRequest,
    ) -> Result<CreateEmptyTableResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/create-empty", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "create_empty_table", &id)
            .await
    }

    async fn declare_table(&self, request: DeclareTableRequest) -> Result<DeclareTableResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/declare", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "declare_table", &id)
            .await
    }

    async fn insert_into_table(
        &self,
        request: InsertIntoTableRequest,
        request_data: Bytes,
    ) -> Result<InsertIntoTableResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/insert", encoded_id);
        let mut query = vec![("delimiter", self.delimiter.as_str())];
        let mode_str;
        if let Some(ref mode) = request.mode {
            mode_str = mode.clone();
            query.push(("mode", mode_str.as_str()));
        }
        self.post_binary_json(
            &path,
            &query,
            request_data.to_vec(),
            "insert_into_table",
            &id,
        )
        .await
    }

    async fn merge_insert_into_table(
        &self,
        request: MergeInsertIntoTableRequest,
        request_data: Bytes,
    ) -> Result<MergeInsertIntoTableResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);

        let on = request.on.as_deref().ok_or_else(|| Error::Namespace {
            source: "'on' field is required for merge insert".into(),
            location: snafu::location!(),
        })?;

        let path = format!("/v1/table/{}/merge_insert", encoded_id);
        let mut query = vec![("delimiter", self.delimiter.as_str()), ("on", on)];

        let when_matched_update_all_str;
        if let Some(v) = request.when_matched_update_all {
            when_matched_update_all_str = v.to_string();
            query.push((
                "when_matched_update_all",
                when_matched_update_all_str.as_str(),
            ));
        }
        if let Some(ref v) = request.when_matched_update_all_filt {
            query.push(("when_matched_update_all_filt", v.as_str()));
        }
        let when_not_matched_insert_all_str;
        if let Some(v) = request.when_not_matched_insert_all {
            when_not_matched_insert_all_str = v.to_string();
            query.push((
                "when_not_matched_insert_all",
                when_not_matched_insert_all_str.as_str(),
            ));
        }
        let when_not_matched_by_source_delete_str;
        if let Some(v) = request.when_not_matched_by_source_delete {
            when_not_matched_by_source_delete_str = v.to_string();
            query.push((
                "when_not_matched_by_source_delete",
                when_not_matched_by_source_delete_str.as_str(),
            ));
        }
        if let Some(ref v) = request.when_not_matched_by_source_delete_filt {
            query.push(("when_not_matched_by_source_delete_filt", v.as_str()));
        }
        if let Some(ref v) = request.timeout {
            query.push(("timeout", v.as_str()));
        }
        let use_index_str;
        if let Some(v) = request.use_index {
            use_index_str = v.to_string();
            query.push(("use_index", use_index_str.as_str()));
        }

        self.post_binary_json(
            &path,
            &query,
            request_data.to_vec(),
            "merge_insert_into_table",
            &id,
        )
        .await
    }

    async fn update_table(&self, request: UpdateTableRequest) -> Result<UpdateTableResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/update", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "update_table", &id)
            .await
    }

    async fn delete_from_table(
        &self,
        request: DeleteFromTableRequest,
    ) -> Result<DeleteFromTableResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/delete", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "delete_from_table", &id)
            .await
    }

    async fn query_table(&self, request: QueryTableRequest) -> Result<Bytes> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/query", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];

        let url = format!("{}{}", self.rest_client.base_path(), path);
        let req_builder = self
            .rest_client
            .client()
            .post(&url)
            .query(&query)
            .json(&request);

        let resp = self
            .rest_client
            .execute(req_builder, "query_table", &id)
            .await
            .map_err(|e| Error::IO {
                source: box_error(e),
                location: snafu::location!(),
            })?;

        let status = resp.status();
        if status.is_success() {
            resp.bytes().await.map_err(|e| Error::IO {
                source: box_error(e),
                location: snafu::location!(),
            })
        } else {
            let content = resp.text().await.map_err(|e| Error::IO {
                source: box_error(e),
                location: snafu::location!(),
            })?;
            Err(Error::Namespace {
                source: format!("Response error: status={}, content={}", status, content).into(),
                location: snafu::location!(),
            })
        }
    }

    async fn create_table_index(
        &self,
        request: CreateTableIndexRequest,
    ) -> Result<CreateTableIndexResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/create_index", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "create_table_index", &id)
            .await
    }

    async fn list_table_indices(
        &self,
        request: ListTableIndicesRequest,
    ) -> Result<ListTableIndicesResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/index/list", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "list_table_indices", &id)
            .await
    }

    async fn describe_table_index_stats(
        &self,
        request: DescribeTableIndexStatsRequest,
    ) -> Result<DescribeTableIndexStatsResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let index_name = request.index_name.as_deref().unwrap_or("");
        let path = format!(
            "/v1/table/{}/index/{}/stats",
            encoded_id,
            urlencode(index_name)
        );
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "describe_table_index_stats", &id)
            .await
    }

    async fn describe_transaction(
        &self,
        request: DescribeTransactionRequest,
    ) -> Result<DescribeTransactionResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/transaction/{}/describe", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "describe_transaction", &id)
            .await
    }

    async fn alter_transaction(
        &self,
        request: AlterTransactionRequest,
    ) -> Result<AlterTransactionResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/transaction/{}/alter", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "alter_transaction", &id)
            .await
    }

    async fn create_table_scalar_index(
        &self,
        request: CreateTableIndexRequest,
    ) -> Result<CreateTableScalarIndexResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/create_scalar_index", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "create_table_scalar_index", &id)
            .await
    }

    async fn drop_table_index(
        &self,
        request: DropTableIndexRequest,
    ) -> Result<DropTableIndexResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let index_name = request.index_name.as_deref().unwrap_or("");
        let path = format!(
            "/v1/table/{}/index/{}/drop",
            encoded_id,
            urlencode(index_name)
        );
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "drop_table_index", &id)
            .await
    }

    async fn list_all_tables(&self, request: ListTablesRequest) -> Result<ListTablesResponse> {
        let path = "/v1/table";
        let mut query = vec![("delimiter", self.delimiter.as_str())];
        let page_token_str;
        if let Some(ref pt) = request.page_token {
            page_token_str = pt.clone();
            query.push(("page_token", page_token_str.as_str()));
        }
        let limit_str;
        if let Some(limit) = request.limit {
            limit_str = limit.to_string();
            query.push(("limit", limit_str.as_str()));
        }
        self.get_json(path, &query, "list_all_tables", "").await
    }

    async fn restore_table(&self, request: RestoreTableRequest) -> Result<RestoreTableResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/restore", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "restore_table", &id)
            .await
    }

    async fn rename_table(&self, request: RenameTableRequest) -> Result<RenameTableResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/rename", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "rename_table", &id)
            .await
    }

    async fn list_table_versions(
        &self,
        request: ListTableVersionsRequest,
    ) -> Result<ListTableVersionsResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/version/list", encoded_id);
        let mut query = vec![("delimiter", self.delimiter.as_str())];
        let page_token_str;
        if let Some(ref pt) = request.page_token {
            page_token_str = pt.clone();
            query.push(("page_token", page_token_str.as_str()));
        }
        let limit_str;
        if let Some(limit) = request.limit {
            limit_str = limit.to_string();
            query.push(("limit", limit_str.as_str()));
        }
        self.get_json(&path, &query, "list_table_versions", &id)
            .await
    }

    async fn update_table_schema_metadata(
        &self,
        request: UpdateTableSchemaMetadataRequest,
    ) -> Result<UpdateTableSchemaMetadataResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/schema_metadata/update", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        let metadata = request.metadata.unwrap_or_default();
        let result: HashMap<String, String> = self
            .post_json(
                &path,
                &query,
                &metadata,
                "update_table_schema_metadata",
                &id,
            )
            .await?;
        Ok(UpdateTableSchemaMetadataResponse {
            metadata: Some(result),
            ..Default::default()
        })
    }

    async fn get_table_stats(
        &self,
        request: GetTableStatsRequest,
    ) -> Result<GetTableStatsResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/stats", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "get_table_stats", &id)
            .await
    }

    async fn explain_table_query_plan(
        &self,
        request: ExplainTableQueryPlanRequest,
    ) -> Result<String> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/explain_plan", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "explain_table_query_plan", &id)
            .await
    }

    async fn analyze_table_query_plan(
        &self,
        request: AnalyzeTableQueryPlanRequest,
    ) -> Result<String> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/analyze_plan", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "analyze_table_query_plan", &id)
            .await
    }

    async fn alter_table_add_columns(
        &self,
        request: AlterTableAddColumnsRequest,
    ) -> Result<AlterTableAddColumnsResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/add_columns", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "alter_table_add_columns", &id)
            .await
    }

    async fn alter_table_alter_columns(
        &self,
        request: AlterTableAlterColumnsRequest,
    ) -> Result<AlterTableAlterColumnsResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/alter_columns", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "alter_table_alter_columns", &id)
            .await
    }

    async fn alter_table_drop_columns(
        &self,
        request: AlterTableDropColumnsRequest,
    ) -> Result<AlterTableDropColumnsResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/drop_columns", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "alter_table_drop_columns", &id)
            .await
    }

    async fn list_table_tags(
        &self,
        request: ListTableTagsRequest,
    ) -> Result<ListTableTagsResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/tags/list", encoded_id);
        let mut query = vec![("delimiter", self.delimiter.as_str())];
        let page_token_str;
        if let Some(ref pt) = request.page_token {
            page_token_str = pt.clone();
            query.push(("page_token", page_token_str.as_str()));
        }
        let limit_str;
        if let Some(limit) = request.limit {
            limit_str = limit.to_string();
            query.push(("limit", limit_str.as_str()));
        }
        self.get_json(&path, &query, "list_table_tags", &id).await
    }

    async fn get_table_tag_version(
        &self,
        request: GetTableTagVersionRequest,
    ) -> Result<GetTableTagVersionResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/tags/version", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "get_table_tag_version", &id)
            .await
    }

    async fn create_table_tag(
        &self,
        request: CreateTableTagRequest,
    ) -> Result<CreateTableTagResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/tags/create", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "create_table_tag", &id)
            .await
    }

    async fn delete_table_tag(
        &self,
        request: DeleteTableTagRequest,
    ) -> Result<DeleteTableTagResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/tags/delete", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "delete_table_tag", &id)
            .await
    }

    async fn update_table_tag(
        &self,
        request: UpdateTableTagRequest,
    ) -> Result<UpdateTableTagResponse> {
        let id = object_id_str(&request.id, &self.delimiter)?;
        let encoded_id = urlencode(&id);
        let path = format!("/v1/table/{}/tags/update", encoded_id);
        let query = [("delimiter", self.delimiter.as_str())];
        self.post_json(&path, &query, &request, "update_table_tag", &id)
            .await
    }

    fn namespace_id(&self) -> String {
        format!(
            "RestNamespace {{ endpoint: {:?}, delimiter: {:?} }}",
            self.rest_client.base_path(),
            self.delimiter
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

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

        let _namespace = RestNamespaceBuilder::from_properties(properties)
            .expect("Failed to create namespace builder")
            .build();

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

        let namespace = RestNamespaceBuilder::from_properties(properties)
            .expect("Failed to create namespace builder")
            .build();

        let request = ListNamespacesRequest {
            id: Some(vec!["test".to_string()]),
            ..Default::default()
        };

        let result = namespace.list_namespaces(request).await;

        // Should succeed, meaning headers were sent correctly
        assert!(result.is_ok());
    }

    #[test]
    fn test_default_configuration() {
        let mut properties = HashMap::new();
        properties.insert("uri".to_string(), "http://localhost:8080".to_string());
        let _namespace = RestNamespaceBuilder::from_properties(properties)
            .expect("Failed to create namespace builder")
            .build();

        // The default delimiter should be "$" - test passes if no panic
    }

    #[test]
    fn test_with_custom_uri() {
        let mut properties = HashMap::new();
        properties.insert("uri".to_string(), "https://api.example.com/v1".to_string());

        let _namespace = RestNamespaceBuilder::from_properties(properties)
            .expect("Failed to create namespace builder")
            .build();
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

        let builder = RestNamespaceBuilder::from_properties(properties)
            .expect("Failed to create namespace builder");
        assert_eq!(builder.cert_file, Some("/path/to/cert.pem".to_string()));
        assert_eq!(builder.key_file, Some("/path/to/key.pem".to_string()));
        assert_eq!(builder.ssl_ca_cert, Some("/path/to/ca.pem".to_string()));
        assert!(builder.assert_hostname);
    }

    #[test]
    fn test_tls_config_default_assert_hostname() {
        let mut properties = HashMap::new();
        properties.insert("uri".to_string(), "https://api.example.com".to_string());
        properties.insert("tls.cert_file".to_string(), "/path/to/cert.pem".to_string());
        properties.insert("tls.key_file".to_string(), "/path/to/key.pem".to_string());

        let builder = RestNamespaceBuilder::from_properties(properties)
            .expect("Failed to create namespace builder");
        // Default should be true
        assert!(builder.assert_hostname);
    }

    #[test]
    fn test_tls_config_disable_hostname_verification() {
        let mut properties = HashMap::new();
        properties.insert("uri".to_string(), "https://api.example.com".to_string());
        properties.insert("tls.cert_file".to_string(), "/path/to/cert.pem".to_string());
        properties.insert("tls.key_file".to_string(), "/path/to/key.pem".to_string());
        properties.insert("tls.assert_hostname".to_string(), "false".to_string());

        let builder = RestNamespaceBuilder::from_properties(properties)
            .expect("Failed to create namespace builder");
        assert!(!builder.assert_hostname);
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
        let _namespace = RestNamespaceBuilder::from_properties(properties)
            .expect("Failed to create namespace builder")
            .build();
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
        let namespace = RestNamespaceBuilder::new(mock_server.uri()).build();

        let request = ListNamespacesRequest {
            id: Some(vec!["test".to_string()]),
            limit: Some(10),
            ..Default::default()
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
        let namespace = RestNamespaceBuilder::new(mock_server.uri()).build();

        let request = ListNamespacesRequest {
            id: Some(vec!["test".to_string()]),
            limit: Some(10),
            ..Default::default()
        };

        let result = namespace.list_namespaces(request).await;

        // Should return an error
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_create_namespace_success() {
        // Start a mock server
        let mock_server = MockServer::start().await;

        // Create mock response
        let path_str = "/v1/namespace/test$newnamespace/create".replace("$", "%24");
        Mock::given(method("POST"))
            .and(path(path_str.as_str()))
            .respond_with(ResponseTemplate::new(201).set_body_json(serde_json::json!({
                "namespace": {
                    "identifier": ["test", "newnamespace"],
                    "properties": {}
                }
            })))
            .mount(&mock_server)
            .await;

        // Create namespace with mock server URL
        let namespace = RestNamespaceBuilder::new(mock_server.uri()).build();

        let request = CreateNamespaceRequest {
            id: Some(vec!["test".to_string(), "newnamespace".to_string()]),
            ..Default::default()
        };

        let result = namespace.create_namespace(request).await;

        // Should succeed with mock server
        assert!(result.is_ok(), "Failed: {:?}", result.err());
    }

    #[tokio::test]
    async fn test_create_table_success() {
        // Start a mock server
        let mock_server = MockServer::start().await;

        // Create mock response
        let path_str = "/v1/table/test$namespace$table/create".replace("$", "%24");
        Mock::given(method("POST"))
            .and(path(path_str.as_str()))
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
        let namespace = RestNamespaceBuilder::new(mock_server.uri()).build();

        let request = CreateTableRequest {
            id: Some(vec![
                "test".to_string(),
                "namespace".to_string(),
                "table".to_string(),
            ]),
            mode: Some("Create".to_string()),
            ..Default::default()
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
        let path_str = "/v1/table/test$namespace$table/insert".replace("$", "%24");
        Mock::given(method("POST"))
            .and(path(path_str.as_str()))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "transaction_id": "txn-123"
            })))
            .mount(&mock_server)
            .await;

        // Create namespace with mock server URL
        let namespace = RestNamespaceBuilder::new(mock_server.uri()).build();

        let request = InsertIntoTableRequest {
            id: Some(vec![
                "test".to_string(),
                "namespace".to_string(),
                "table".to_string(),
            ]),
            mode: Some("Append".to_string()),
            ..Default::default()
        };

        let data = Bytes::from("arrow data here");
        let result = namespace.insert_into_table(request, data).await;

        // Should succeed with mock server
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.transaction_id, Some("txn-123".to_string()));
    }

    // Integration tests for DynamicContextProvider

    #[derive(Debug)]
    struct TestContextProvider {
        headers: HashMap<String, String>,
    }

    impl DynamicContextProvider for TestContextProvider {
        fn provide_context(&self, _info: &OperationInfo) -> HashMap<String, String> {
            self.headers.clone()
        }
    }

    #[tokio::test]
    async fn test_context_provider_headers_sent() {
        let mock_server = MockServer::start().await;

        // Mock expects the context header
        Mock::given(method("GET"))
            .and(path("/v1/namespace/test/list"))
            .and(wiremock::matchers::header(
                "X-Context-Token",
                "dynamic-token",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "namespaces": []
            })))
            .mount(&mock_server)
            .await;

        // Create context provider
        let mut context_headers = HashMap::new();
        context_headers.insert(
            "headers.X-Context-Token".to_string(),
            "dynamic-token".to_string(),
        );
        let provider = Arc::new(TestContextProvider {
            headers: context_headers,
        });

        let namespace = RestNamespaceBuilder::new(mock_server.uri())
            .context_provider(provider)
            .build();

        let request = ListNamespacesRequest {
            id: Some(vec!["test".to_string()]),
            ..Default::default()
        };

        let result = namespace.list_namespaces(request).await;
        assert!(result.is_ok(), "Failed: {:?}", result.err());
    }

    #[tokio::test]
    async fn test_base_headers_merged_with_context_headers() {
        let mock_server = MockServer::start().await;

        // Mock expects BOTH base header AND context header
        Mock::given(method("GET"))
            .and(path("/v1/namespace/test/list"))
            .and(wiremock::matchers::header(
                "Authorization",
                "Bearer base-token",
            ))
            .and(wiremock::matchers::header(
                "X-Context-Token",
                "dynamic-token",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "namespaces": []
            })))
            .mount(&mock_server)
            .await;

        // Create context provider
        let mut context_headers = HashMap::new();
        context_headers.insert(
            "headers.X-Context-Token".to_string(),
            "dynamic-token".to_string(),
        );
        let provider = Arc::new(TestContextProvider {
            headers: context_headers,
        });

        // Create namespace with base header AND context provider
        let namespace = RestNamespaceBuilder::new(mock_server.uri())
            .header("Authorization", "Bearer base-token")
            .context_provider(provider)
            .build();

        let request = ListNamespacesRequest {
            id: Some(vec!["test".to_string()]),
            ..Default::default()
        };

        let result = namespace.list_namespaces(request).await;
        assert!(result.is_ok(), "Failed: {:?}", result.err());
    }

    #[tokio::test]
    async fn test_context_headers_override_base_headers() {
        let mock_server = MockServer::start().await;

        // Mock expects the CONTEXT header value (not base)
        Mock::given(method("GET"))
            .and(path("/v1/namespace/test/list"))
            .and(wiremock::matchers::header(
                "Authorization",
                "Bearer context-override-token",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "namespaces": []
            })))
            .mount(&mock_server)
            .await;

        // Context provider that overrides Authorization header
        let mut context_headers = HashMap::new();
        context_headers.insert(
            "headers.Authorization".to_string(),
            "Bearer context-override-token".to_string(),
        );
        let provider = Arc::new(TestContextProvider {
            headers: context_headers,
        });

        // Create namespace with base header that will be overridden
        let namespace = RestNamespaceBuilder::new(mock_server.uri())
            .header("Authorization", "Bearer base-token")
            .context_provider(provider)
            .build();

        let request = ListNamespacesRequest {
            id: Some(vec!["test".to_string()]),
            ..Default::default()
        };

        let result = namespace.list_namespaces(request).await;
        assert!(result.is_ok(), "Failed: {:?}", result.err());
    }

    #[tokio::test]
    async fn test_no_context_provider_uses_base_headers_only() {
        let mock_server = MockServer::start().await;

        // Mock expects only the base header
        Mock::given(method("GET"))
            .and(path("/v1/namespace/test/list"))
            .and(wiremock::matchers::header(
                "Authorization",
                "Bearer base-only",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "namespaces": []
            })))
            .mount(&mock_server)
            .await;

        // Create namespace WITHOUT context provider, only base headers
        let namespace = RestNamespaceBuilder::new(mock_server.uri())
            .header("Authorization", "Bearer base-only")
            .build();

        let request = ListNamespacesRequest {
            id: Some(vec!["test".to_string()]),
            ..Default::default()
        };

        let result = namespace.list_namespaces(request).await;
        assert!(result.is_ok(), "Failed: {:?}", result.err());
    }
}
