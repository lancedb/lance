// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! REST server adapter for Lance Namespace
//!
//! This module provides a REST API server that wraps any `LanceNamespace` implementation,
//! allowing it to be accessed via HTTP. The server implements the Lance REST Namespace
//! specification.

use std::sync::Arc;

use axum::{
    body::Bytes,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use tower_http::trace::TraceLayer;

use lance_core::{Error, Result};
use lance_namespace::models::*;
use lance_namespace::LanceNamespace;

/// Configuration for the REST server
#[derive(Debug, Clone)]
pub struct RestServerConfig {
    /// Host address to bind to
    pub host: String,
    /// Port to listen on
    pub port: u16,
}

impl Default for RestServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 2333,
        }
    }
}

/// REST server adapter that wraps a Lance Namespace implementation
pub struct RestServer {
    backend: Arc<dyn LanceNamespace>,
    config: RestServerConfig,
}

impl RestServer {
    /// Create a new REST server with the given backend namespace
    pub fn new(backend: Arc<dyn LanceNamespace>, config: RestServerConfig) -> Self {
        Self { backend, config }
    }

    /// Build the Axum router with all REST API routes
    fn router(&self) -> Router {
        Router::new()
            // Namespace operations
            .route("/v1/namespace/:id/create", post(create_namespace))
            .route("/v1/namespace/:id/list", get(list_namespaces))
            .route("/v1/namespace/:id/describe", post(describe_namespace))
            .route("/v1/namespace/:id/drop", post(drop_namespace))
            .route("/v1/namespace/:id/exists", post(namespace_exists))
            .route("/v1/namespace/:id/table/list", get(list_tables))
            // Table operations
            .route("/v1/table/:id/register", post(register_table))
            .route("/v1/table/:id/describe", post(describe_table))
            .route("/v1/table/:id/exists", post(table_exists))
            .route("/v1/table/:id/drop", post(drop_table))
            .route("/v1/table/:id/deregister", post(deregister_table))
            .route("/v1/table/:id/create", post(create_table))
            .route("/v1/table/:id/create-empty", post(create_empty_table))
            .layer(TraceLayer::new_for_http())
            .with_state(self.backend.clone())
    }

    /// Start the REST server (blocking)
    pub async fn serve(self) -> Result<()> {
        let addr = format!("{}:{}", self.config.host, self.config.port);
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .map_err(|e| Error::IO {
                source: Box::new(e),
                location: snafu::location!(),
            })?;

        axum::serve(listener, self.router())
            .await
            .map_err(|e| Error::IO {
                source: Box::new(e),
                location: snafu::location!(),
            })?;

        Ok(())
    }
}

// ============================================================================
// Query Parameters
// ============================================================================

#[derive(Debug, Deserialize)]
struct DelimiterQuery {
    delimiter: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PaginationQuery {
    delimiter: Option<String>,
    page_token: Option<String>,
    limit: Option<i32>,
}

// ============================================================================
// Error Conversion
// ============================================================================

/// Convert Lance errors to HTTP responses
fn error_to_response(err: Error) -> Response {
    match err {
        Error::Namespace { source, .. } => {
            let error_msg = source.to_string();
            if error_msg.contains("not found") || error_msg.contains("does not exist") {
                (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({
                        "error": {
                            "message": error_msg,
                            "type": "NamespaceNotFoundException"
                        }
                    })),
                )
                    .into_response()
            } else if error_msg.contains("already exists") {
                (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": {
                            "message": error_msg,
                            "type": "TableAlreadyExistsException"
                        }
                    })),
                )
                    .into_response()
            } else {
                (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": {
                            "message": error_msg,
                            "type": "NamespaceException"
                        }
                    })),
                )
                    .into_response()
            }
        }
        Error::IO { source, .. } => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": {
                    "message": source.to_string(),
                    "type": "InternalServerError"
                }
            })),
        )
            .into_response(),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": {
                    "message": err.to_string(),
                    "type": "InternalServerError"
                }
            })),
        )
            .into_response(),
    }
}

// ============================================================================
// Namespace Operation Handlers
// ============================================================================

async fn create_namespace(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<CreateNamespaceRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.create_namespace(request).await {
        Ok(response) => (StatusCode::CREATED, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn list_namespaces(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<PaginationQuery>,
) -> Response {
    let request = ListNamespacesRequest {
        id: Some(parse_id(&id, params.delimiter.as_deref())),
        page_token: params.page_token,
        limit: params.limit,
    };

    match backend.list_namespaces(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn describe_namespace(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<DescribeNamespaceRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.describe_namespace(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn drop_namespace(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<DropNamespaceRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.drop_namespace(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn namespace_exists(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<NamespaceExistsRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.namespace_exists(request).await {
        Ok(_) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error_to_response(e),
    }
}

// ============================================================================
// Table Metadata Operation Handlers
// ============================================================================

async fn list_tables(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<PaginationQuery>,
) -> Response {
    let request = ListTablesRequest {
        id: Some(parse_id(&id, params.delimiter.as_deref())),
        page_token: params.page_token,
        limit: params.limit,
    };

    match backend.list_tables(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn register_table(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<RegisterTableRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.register_table(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn describe_table(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<DescribeTableRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.describe_table(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn table_exists(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<TableExistsRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.table_exists(request).await {
        Ok(_) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn drop_table(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<DropTableRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.drop_table(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn deregister_table(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<DeregisterTableRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.deregister_table(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

// ============================================================================
// Table Data Operation Handlers
// ============================================================================

#[derive(Debug, Deserialize)]
struct CreateTableQuery {
    delimiter: Option<String>,
    mode: Option<String>,
    location: Option<String>,
    properties: Option<String>,
}

async fn create_table(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<CreateTableQuery>,
    body: Bytes,
) -> Response {
    use lance_namespace::models::create_table_request::Mode;

    let mode = params.mode.as_deref().and_then(|m| match m {
        "create" => Some(Mode::Create),
        "exist_ok" => Some(Mode::ExistOk),
        "overwrite" => Some(Mode::Overwrite),
        _ => None,
    });

    let properties = params
        .properties
        .as_ref()
        .and_then(|p| serde_json::from_str(p).ok());

    let request = CreateTableRequest {
        id: Some(parse_id(&id, params.delimiter.as_deref())),
        location: params.location,
        mode,
        properties,
    };

    match backend.create_table(request, body).await {
        Ok(response) => (StatusCode::CREATED, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn create_empty_table(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<CreateEmptyTableRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.create_empty_table(request).await {
        Ok(response) => (StatusCode::CREATED, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Parse object ID from path string using delimiter
fn parse_id(id_str: &str, delimiter: Option<&str>) -> Vec<String> {
    let delimiter = delimiter.unwrap_or(".");

    // Special case: if ID equals delimiter, it represents root namespace (empty vec)
    if id_str == delimiter {
        return vec![];
    }

    id_str
        .split(delimiter)
        .filter(|s| !s.is_empty()) // Filter out empty strings from split
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_id_default_delimiter() {
        let id = parse_id("ns1.ns2.table", None);
        assert_eq!(id, vec!["ns1", "ns2", "table"]);
    }

    #[test]
    fn test_parse_id_custom_delimiter() {
        let id = parse_id("ns1/ns2/table", Some("/"));
        assert_eq!(id, vec!["ns1", "ns2", "table"]);
    }

    #[test]
    fn test_parse_id_single_part() {
        let id = parse_id("table", None);
        assert_eq!(id, vec!["table"]);
    }

    #[test]
    fn test_parse_id_root_namespace() {
        // When ID equals delimiter, it represents root namespace
        let id = parse_id(".", None);
        assert_eq!(id, Vec::<String>::new());

        let id = parse_id("/", Some("/"));
        assert_eq!(id, Vec::<String>::new());
    }

    #[test]
    fn test_parse_id_filters_empty() {
        // Filter out empty strings from split results
        let id = parse_id("..table..", None);
        assert_eq!(id, vec!["table"]);
    }

    // ============================================================================
    // Integration Tests
    // ============================================================================

    #[cfg(feature = "rest")]
    mod integration {
        use super::super::*;
        use crate::{DirectoryNamespaceBuilder, RestNamespaceBuilder};
        use std::sync::Arc;
        use tempfile::TempDir;
        use tokio::task::JoinHandle;

        /// Test fixture that manages server lifecycle
        struct RestServerFixture {
            _temp_dir: TempDir,
            namespace: crate::RestNamespace,
            server_handle: JoinHandle<()>,
        }

        impl RestServerFixture {
            async fn new(port: u16) -> Self {
                let temp_dir = TempDir::new().unwrap();
                let temp_path = temp_dir.path().to_str().unwrap().to_string();

                // Create DirectoryNamespace backend with manifest enabled
                let backend = DirectoryNamespaceBuilder::new(&temp_path)
                    .manifest_enabled(true)
                    .build()
                    .await
                    .unwrap();
                let backend = Arc::new(backend);

                // Start REST server
                let config = RestServerConfig {
                    host: "127.0.0.1".to_string(),
                    port,
                };

                let server = RestServer::new(backend.clone(), config);
                let server_handle = tokio::spawn(async move {
                    server.serve().await.unwrap();
                });

                // Give server time to start
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

                // Create RestNamespace client
                let server_url = format!("http://127.0.0.1:{}", port);
                let namespace = RestNamespaceBuilder::new(&server_url)
                    .delimiter("$")
                    .build();

                Self {
                    _temp_dir: temp_dir,
                    namespace,
                    server_handle,
                }
            }
        }

        impl Drop for RestServerFixture {
            fn drop(&mut self) {
                self.server_handle.abort();
            }
        }

        /// Helper to create Arrow IPC data for testing
        fn create_test_arrow_data() -> Bytes {
            use arrow::array::{Int32Array, StringArray};
            use arrow::datatypes::{DataType, Field, Schema};
            use arrow::ipc::writer::StreamWriter;
            use arrow::record_batch::RecordBatch;

            let schema = Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("name", DataType::Utf8, false),
            ]);

            let batch = RecordBatch::try_new(
                Arc::new(schema),
                vec![
                    Arc::new(Int32Array::from(vec![1, 2, 3])),
                    Arc::new(StringArray::from(vec!["alice", "bob", "charlie"])),
                ],
            )
            .unwrap();

            let mut buffer = Vec::new();
            {
                let mut writer = StreamWriter::try_new(&mut buffer, &batch.schema()).unwrap();
                writer.write(&batch).unwrap();
                writer.finish().unwrap();
            }

            Bytes::from(buffer)
        }

        #[tokio::test]
        async fn test_create_and_list_child_namespaces() {
            let fixture = RestServerFixture::new(4001).await;

            // Create child namespaces
            for i in 1..=3 {
                let create_req = CreateNamespaceRequest {
                    id: Some(vec![format!("namespace{}", i)]),
                    properties: None,
                    mode: None,
                };
                let result = fixture.namespace.create_namespace(create_req).await;
                assert!(result.is_ok(), "Failed to create namespace{}", i);
            }

            // List child namespaces
            let list_req = ListNamespacesRequest {
                id: Some(vec![]),
                page_token: None,
                limit: None,
            };
            let result = fixture.namespace.list_namespaces(list_req).await;
            assert!(result.is_ok());
            let namespaces = result.unwrap();
            assert_eq!(namespaces.namespaces.len(), 3);
            assert!(namespaces.namespaces.contains(&"namespace1".to_string()));
            assert!(namespaces.namespaces.contains(&"namespace2".to_string()));
            assert!(namespaces.namespaces.contains(&"namespace3".to_string()));
        }

        #[tokio::test]
        async fn test_nested_namespace_hierarchy() {
            let fixture = RestServerFixture::new(4002).await;

            // Create parent namespace
            let create_req = CreateNamespaceRequest {
                id: Some(vec!["parent".to_string()]),
                properties: None,
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_req)
                .await
                .unwrap();

            // Create nested child namespaces
            let create_req = CreateNamespaceRequest {
                id: Some(vec!["parent".to_string(), "child1".to_string()]),
                properties: None,
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_req)
                .await
                .unwrap();

            let create_req = CreateNamespaceRequest {
                id: Some(vec!["parent".to_string(), "child2".to_string()]),
                properties: None,
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_req)
                .await
                .unwrap();

            // List children of parent
            let list_req = ListNamespacesRequest {
                id: Some(vec!["parent".to_string()]),
                page_token: None,
                limit: None,
            };
            let result = fixture.namespace.list_namespaces(list_req).await;
            assert!(result.is_ok());
            let children = result.unwrap().namespaces;
            assert_eq!(children.len(), 2);
            assert!(children.contains(&"child1".to_string()));
            assert!(children.contains(&"child2".to_string()));
        }

        #[tokio::test]
        async fn test_create_table_in_child_namespace() {
            let fixture = RestServerFixture::new(4003).await;
            let table_data = create_test_arrow_data();

            // Create child namespace first
            let create_ns_req = CreateNamespaceRequest {
                id: Some(vec!["test_namespace".to_string()]),
                properties: None,
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_ns_req)
                .await
                .unwrap();

            // Create table in child namespace
            let create_table_req = CreateTableRequest {
                id: Some(vec!["test_namespace".to_string(), "test_table".to_string()]),
                location: None,
                mode: Some(create_table_request::Mode::Create),
                properties: None,
            };

            let result = fixture
                .namespace
                .create_table(create_table_req, table_data)
                .await;

            assert!(
                result.is_ok(),
                "Failed to create table in child namespace: {:?}",
                result.err()
            );

            // Check response details
            let response = result.unwrap();
            assert!(
                response.location.is_some(),
                "Response should include location"
            );
            assert!(
                response.location.unwrap().contains("test_table"),
                "Location should contain table name"
            );
            assert_eq!(
                response.version,
                Some(1),
                "Initial table version should be 1"
            );
        }

        #[tokio::test]
        async fn test_list_tables_in_child_namespace() {
            let fixture = RestServerFixture::new(4004).await;
            let table_data = create_test_arrow_data();

            // Create child namespace
            let create_ns_req = CreateNamespaceRequest {
                id: Some(vec!["test_namespace".to_string()]),
                properties: None,
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_ns_req)
                .await
                .unwrap();

            // Create multiple tables in the namespace
            for i in 1..=3 {
                let create_table_req = CreateTableRequest {
                    id: Some(vec!["test_namespace".to_string(), format!("table{}", i)]),
                    location: None,
                    mode: Some(create_table_request::Mode::Create),
                    properties: None,
                };
                fixture
                    .namespace
                    .create_table(create_table_req, table_data.clone())
                    .await
                    .unwrap();
            }

            // List tables in the namespace
            let list_req = ListTablesRequest {
                id: Some(vec!["test_namespace".to_string()]),
                page_token: None,
                limit: None,
            };
            let result = fixture.namespace.list_tables(list_req).await;
            assert!(result.is_ok());
            let tables = result.unwrap();
            assert_eq!(tables.tables.len(), 3);
            assert!(tables.tables.contains(&"table1".to_string()));
            assert!(tables.tables.contains(&"table2".to_string()));
            assert!(tables.tables.contains(&"table3".to_string()));
        }

        #[tokio::test]
        async fn test_table_exists_in_child_namespace() {
            let fixture = RestServerFixture::new(4005).await;
            let table_data = create_test_arrow_data();

            // Create child namespace
            let create_ns_req = CreateNamespaceRequest {
                id: Some(vec!["test_namespace".to_string()]),
                properties: None,
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_ns_req)
                .await
                .unwrap();

            // Create table
            let create_table_req = CreateTableRequest {
                id: Some(vec!["test_namespace".to_string(), "test_table".to_string()]),
                location: None,
                mode: Some(create_table_request::Mode::Create),
                properties: None,
            };
            fixture
                .namespace
                .create_table(create_table_req, table_data)
                .await
                .unwrap();

            // Check table exists
            let mut exists_req = TableExistsRequest::new();
            exists_req.id = Some(vec!["test_namespace".to_string(), "test_table".to_string()]);
            let result = fixture.namespace.table_exists(exists_req).await;
            assert!(result.is_ok(), "Table should exist in child namespace");
        }

        #[tokio::test]
        async fn test_empty_table_exists_in_child_namespace() {
            let fixture = RestServerFixture::new(4015).await;

            // Create child namespace
            let create_ns_req = CreateNamespaceRequest {
                id: Some(vec!["test_namespace".to_string()]),
                properties: None,
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_ns_req)
                .await
                .unwrap();

            // Create empty table
            let mut create_req = CreateEmptyTableRequest::new();
            create_req.id = Some(vec!["test_namespace".to_string(), "test_table".to_string()]);
            fixture
                .namespace
                .create_empty_table(create_req)
                .await
                .unwrap();

            // Check table exists
            let mut exists_req = TableExistsRequest::new();
            exists_req.id = Some(vec!["test_namespace".to_string(), "test_table".to_string()]);
            let result = fixture.namespace.table_exists(exists_req).await;
            assert!(
                result.is_ok(),
                "Empty table should exist in child namespace"
            );
        }

        #[tokio::test]
        async fn test_describe_table_in_child_namespace() {
            let fixture = RestServerFixture::new(4006).await;
            let table_data = create_test_arrow_data();

            // Create child namespace
            let create_ns_req = CreateNamespaceRequest {
                id: Some(vec!["test_namespace".to_string()]),
                properties: None,
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_ns_req)
                .await
                .unwrap();

            // Create table
            let create_table_req = CreateTableRequest {
                id: Some(vec!["test_namespace".to_string(), "test_table".to_string()]),
                location: None,
                mode: Some(create_table_request::Mode::Create),
                properties: None,
            };
            fixture
                .namespace
                .create_table(create_table_req, table_data)
                .await
                .unwrap();

            // Describe the table
            let mut describe_req = DescribeTableRequest::new();
            describe_req.id = Some(vec!["test_namespace".to_string(), "test_table".to_string()]);
            let result = fixture.namespace.describe_table(describe_req).await;

            assert!(
                result.is_ok(),
                "Failed to describe table in child namespace: {:?}",
                result.err()
            );
            let response = result.unwrap();

            // Check location
            assert!(
                response.location.is_some(),
                "Response should include location"
            );
            let location = response.location.unwrap();
            assert!(
                location.contains("test_table"),
                "Location should contain table name"
            );

            // Check version (might be None for empty datasets in some implementations)
            // When version is present, it should be 1 for the first version
            if let Some(version) = response.version {
                assert_eq!(version, 1, "First table version should be 1");
            }

            // Check schema (if available)
            if let Some(schema) = response.schema {
                assert_eq!(schema.fields.len(), 2, "Schema should have 2 fields");

                // Verify field names and types
                let field_names: Vec<&str> =
                    schema.fields.iter().map(|f| f.name.as_str()).collect();
                assert!(field_names.contains(&"id"), "Schema should have 'id' field");
                assert!(
                    field_names.contains(&"name"),
                    "Schema should have 'name' field"
                );

                let id_field = schema.fields.iter().find(|f| f.name == "id").unwrap();
                assert_eq!(
                    id_field.r#type.r#type.to_lowercase(),
                    "int32",
                    "id field should be int32"
                );
                assert!(!id_field.nullable, "id field should be non-nullable");

                let name_field = schema.fields.iter().find(|f| f.name == "name").unwrap();
                assert_eq!(
                    name_field.r#type.r#type.to_lowercase(),
                    "utf8",
                    "name field should be utf8"
                );
                assert!(!name_field.nullable, "name field should be non-nullable");
            }
        }

        #[tokio::test]
        async fn test_drop_table_in_child_namespace() {
            let fixture = RestServerFixture::new(4007).await;
            let table_data = create_test_arrow_data();

            // Create child namespace
            let create_ns_req = CreateNamespaceRequest {
                id: Some(vec!["test_namespace".to_string()]),
                properties: None,
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_ns_req)
                .await
                .unwrap();

            // Create table
            let create_table_req = CreateTableRequest {
                id: Some(vec!["test_namespace".to_string(), "test_table".to_string()]),
                location: None,
                mode: Some(create_table_request::Mode::Create),
                properties: None,
            };
            fixture
                .namespace
                .create_table(create_table_req, table_data)
                .await
                .unwrap();

            // Drop the table
            let drop_req = DropTableRequest {
                id: Some(vec!["test_namespace".to_string(), "test_table".to_string()]),
            };
            let result = fixture.namespace.drop_table(drop_req).await;
            assert!(
                result.is_ok(),
                "Failed to drop table in child namespace: {:?}",
                result.err()
            );

            // Verify table no longer exists
            let mut exists_req = TableExistsRequest::new();
            exists_req.id = Some(vec!["test_namespace".to_string(), "test_table".to_string()]);
            let result = fixture.namespace.table_exists(exists_req).await;
            assert!(result.is_err(), "Table should not exist after drop");
            // After drop, accessing the table should fail
            // (error message varies depending on implementation details)
        }

        #[tokio::test]
        async fn test_create_empty_table_in_child_namespace() {
            let fixture = RestServerFixture::new(4008).await;

            // Create child namespace
            let create_ns_req = CreateNamespaceRequest {
                id: Some(vec!["test_namespace".to_string()]),
                properties: None,
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_ns_req)
                .await
                .unwrap();

            // Create empty table
            let mut create_req = CreateEmptyTableRequest::new();
            create_req.id = Some(vec![
                "test_namespace".to_string(),
                "empty_table".to_string(),
            ]);

            let result = fixture.namespace.create_empty_table(create_req).await;
            assert!(
                result.is_ok(),
                "Failed to create empty table in child namespace: {:?}",
                result.err()
            );

            // Check response details
            let response = result.unwrap();
            assert!(
                response.location.is_some(),
                "Response should include location"
            );
            assert!(
                response.location.unwrap().contains("empty_table"),
                "Location should contain table name"
            );

            // Verify the empty table exists
            let mut exists_req = TableExistsRequest::new();
            exists_req.id = Some(vec![
                "test_namespace".to_string(),
                "empty_table".to_string(),
            ]);
            let exists_result = fixture.namespace.table_exists(exists_req).await;
            assert!(
                exists_result.is_ok(),
                "Empty table should exist in child namespace"
            );
        }

        #[tokio::test]
        async fn test_describe_empty_table_in_child_namespace() {
            let fixture = RestServerFixture::new(4016).await;

            // Create child namespace
            let create_ns_req = CreateNamespaceRequest {
                id: Some(vec!["test_namespace".to_string()]),
                properties: None,
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_ns_req)
                .await
                .unwrap();

            // Create empty table
            let mut create_req = CreateEmptyTableRequest::new();
            create_req.id = Some(vec!["test_namespace".to_string(), "test_table".to_string()]);
            fixture
                .namespace
                .create_empty_table(create_req)
                .await
                .unwrap();

            // Describe the empty table
            let mut describe_req = DescribeTableRequest::new();
            describe_req.id = Some(vec!["test_namespace".to_string(), "test_table".to_string()]);
            let result = fixture.namespace.describe_table(describe_req).await;

            assert!(
                result.is_ok(),
                "Failed to describe empty table in child namespace: {:?}",
                result.err()
            );
            let response = result.unwrap();

            // Check location
            assert!(
                response.location.is_some(),
                "Response should include location"
            );
            let location = response.location.unwrap();
            assert!(
                location.contains("test_table"),
                "Location should contain table name"
            );

            // Empty tables don't have a version until data is written
            // (version is None for empty tables)

            // Empty tables don't have a schema initially
            // (schema is None until data is added)
        }

        #[tokio::test]
        async fn test_drop_empty_table_in_child_namespace() {
            let fixture = RestServerFixture::new(4017).await;

            // Create child namespace
            let create_ns_req = CreateNamespaceRequest {
                id: Some(vec!["test_namespace".to_string()]),
                properties: None,
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_ns_req)
                .await
                .unwrap();

            // Create empty table
            let mut create_req = CreateEmptyTableRequest::new();
            create_req.id = Some(vec!["test_namespace".to_string(), "test_table".to_string()]);
            fixture
                .namespace
                .create_empty_table(create_req)
                .await
                .unwrap();

            // Drop the empty table
            let drop_req = DropTableRequest {
                id: Some(vec!["test_namespace".to_string(), "test_table".to_string()]),
            };
            let result = fixture.namespace.drop_table(drop_req).await;
            assert!(
                result.is_ok(),
                "Failed to drop empty table in child namespace: {:?}",
                result.err()
            );

            // Verify table no longer exists
            let mut exists_req = TableExistsRequest::new();
            exists_req.id = Some(vec!["test_namespace".to_string(), "test_table".to_string()]);
            let result = fixture.namespace.table_exists(exists_req).await;
            assert!(result.is_err(), "Empty table should not exist after drop");
            // After drop, accessing the table should fail
            // (error message varies depending on implementation details)
        }

        #[tokio::test]
        async fn test_deeply_nested_namespace_with_empty_table() {
            let fixture = RestServerFixture::new(4018).await;

            // Create deeply nested namespace hierarchy
            let create_req = CreateNamespaceRequest {
                id: Some(vec!["level1".to_string()]),
                properties: None,
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_req)
                .await
                .unwrap();

            let create_req = CreateNamespaceRequest {
                id: Some(vec!["level1".to_string(), "level2".to_string()]),
                properties: None,
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_req)
                .await
                .unwrap();

            let create_req = CreateNamespaceRequest {
                id: Some(vec![
                    "level1".to_string(),
                    "level2".to_string(),
                    "level3".to_string(),
                ]),
                properties: None,
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_req)
                .await
                .unwrap();

            // Create empty table in deeply nested namespace
            let mut create_req = CreateEmptyTableRequest::new();
            create_req.id = Some(vec![
                "level1".to_string(),
                "level2".to_string(),
                "level3".to_string(),
                "deep_table".to_string(),
            ]);

            let result = fixture.namespace.create_empty_table(create_req).await;

            assert!(
                result.is_ok(),
                "Failed to create empty table in deeply nested namespace"
            );

            // Verify table exists
            let mut exists_req = TableExistsRequest::new();
            exists_req.id = Some(vec![
                "level1".to_string(),
                "level2".to_string(),
                "level3".to_string(),
                "deep_table".to_string(),
            ]);
            let result = fixture.namespace.table_exists(exists_req).await;
            assert!(
                result.is_ok(),
                "Empty table should exist in deeply nested namespace"
            );
        }

        #[tokio::test]
        async fn test_deeply_nested_namespace_with_table() {
            let fixture = RestServerFixture::new(4009).await;
            let table_data = create_test_arrow_data();

            // Create deeply nested namespace hierarchy
            let create_req = CreateNamespaceRequest {
                id: Some(vec!["level1".to_string()]),
                properties: None,
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_req)
                .await
                .unwrap();

            let create_req = CreateNamespaceRequest {
                id: Some(vec!["level1".to_string(), "level2".to_string()]),
                properties: None,
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_req)
                .await
                .unwrap();

            let create_req = CreateNamespaceRequest {
                id: Some(vec![
                    "level1".to_string(),
                    "level2".to_string(),
                    "level3".to_string(),
                ]),
                properties: None,
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_req)
                .await
                .unwrap();

            // Create table in deeply nested namespace
            let create_table_req = CreateTableRequest {
                id: Some(vec![
                    "level1".to_string(),
                    "level2".to_string(),
                    "level3".to_string(),
                    "deep_table".to_string(),
                ]),
                location: None,
                mode: Some(create_table_request::Mode::Create),
                properties: None,
            };

            let result = fixture
                .namespace
                .create_table(create_table_req, table_data)
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
                "deep_table".to_string(),
            ]);
            let result = fixture.namespace.table_exists(exists_req).await;
            assert!(
                result.is_ok(),
                "Table should exist in deeply nested namespace"
            );
        }

        #[tokio::test]
        async fn test_namespace_isolation() {
            let fixture = RestServerFixture::new(4010).await;
            let table_data = create_test_arrow_data();

            // Create two sibling namespaces
            let create_req = CreateNamespaceRequest {
                id: Some(vec!["namespace1".to_string()]),
                properties: None,
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_req)
                .await
                .unwrap();

            let create_req = CreateNamespaceRequest {
                id: Some(vec!["namespace2".to_string()]),
                properties: None,
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_req)
                .await
                .unwrap();

            // Create table with same name in both namespaces
            let create_table_req = CreateTableRequest {
                id: Some(vec!["namespace1".to_string(), "shared_table".to_string()]),
                location: None,
                mode: Some(create_table_request::Mode::Create),
                properties: None,
            };
            fixture
                .namespace
                .create_table(create_table_req, table_data.clone())
                .await
                .unwrap();

            let create_table_req = CreateTableRequest {
                id: Some(vec!["namespace2".to_string(), "shared_table".to_string()]),
                location: None,
                mode: Some(create_table_request::Mode::Create),
                properties: None,
            };
            fixture
                .namespace
                .create_table(create_table_req, table_data)
                .await
                .unwrap();

            // Drop table in namespace1
            let drop_req = DropTableRequest {
                id: Some(vec!["namespace1".to_string(), "shared_table".to_string()]),
            };
            fixture.namespace.drop_table(drop_req).await.unwrap();

            // Verify namespace1 table is gone but namespace2 table still exists
            let mut exists_req = TableExistsRequest::new();
            exists_req.id = Some(vec!["namespace1".to_string(), "shared_table".to_string()]);
            let result = fixture.namespace.table_exists(exists_req).await;
            assert!(
                result.is_err(),
                "Table in namespace1 should not exist after drop"
            );
            // After drop, accessing the table should fail
            // (error message varies depending on implementation details)

            let mut exists_req = TableExistsRequest::new();
            exists_req.id = Some(vec!["namespace2".to_string(), "shared_table".to_string()]);
            assert!(fixture.namespace.table_exists(exists_req).await.is_ok());
        }

        #[tokio::test]
        async fn test_drop_namespace_with_tables_fails() {
            let fixture = RestServerFixture::new(4011).await;
            let table_data = create_test_arrow_data();

            // Create namespace
            let create_ns_req = CreateNamespaceRequest {
                id: Some(vec!["test_namespace".to_string()]),
                properties: None,
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_ns_req)
                .await
                .unwrap();

            // Create table in namespace
            let create_table_req = CreateTableRequest {
                id: Some(vec!["test_namespace".to_string(), "test_table".to_string()]),
                location: None,
                mode: Some(create_table_request::Mode::Create),
                properties: None,
            };
            fixture
                .namespace
                .create_table(create_table_req, table_data)
                .await
                .unwrap();

            // Try to drop namespace with table - should fail
            let mut drop_req = DropNamespaceRequest::new();
            drop_req.id = Some(vec!["test_namespace".to_string()]);
            let result = fixture.namespace.drop_namespace(drop_req).await;
            assert!(
                result.is_err(),
                "Should not be able to drop namespace with tables"
            );
            let err_msg = result.unwrap_err().to_string();
            assert!(
                err_msg.contains("is not empty"),
                "Error should be 'is not empty', got: {}",
                err_msg
            );
        }

        #[tokio::test]
        async fn test_drop_empty_child_namespace() {
            let fixture = RestServerFixture::new(4012).await;

            // Create namespace
            let create_ns_req = CreateNamespaceRequest {
                id: Some(vec!["test_namespace".to_string()]),
                properties: None,
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_ns_req)
                .await
                .unwrap();

            // Drop empty namespace - should succeed
            let mut drop_req = DropNamespaceRequest::new();
            drop_req.id = Some(vec!["test_namespace".to_string()]);
            let result = fixture.namespace.drop_namespace(drop_req).await;
            assert!(
                result.is_ok(),
                "Should be able to drop empty child namespace"
            );

            // Verify namespace no longer exists
            let exists_req = NamespaceExistsRequest {
                id: Some(vec!["test_namespace".to_string()]),
            };
            let result = fixture.namespace.namespace_exists(exists_req).await;
            assert!(result.is_err(), "Namespace should not exist after drop");
            // After drop, namespace should not be found
            // (error message varies depending on implementation details)
        }

        #[tokio::test]
        async fn test_namespace_with_properties() {
            let fixture = RestServerFixture::new(4013).await;

            // Create namespace with properties
            let mut properties = std::collections::HashMap::new();
            properties.insert("owner".to_string(), "test_user".to_string());
            properties.insert("environment".to_string(), "production".to_string());

            let create_req = CreateNamespaceRequest {
                id: Some(vec!["test_namespace".to_string()]),
                properties: Some(properties.clone()),
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_req)
                .await
                .unwrap();

            // Describe namespace and verify properties
            let describe_req = DescribeNamespaceRequest {
                id: Some(vec!["test_namespace".to_string()]),
            };
            let result = fixture.namespace.describe_namespace(describe_req).await;
            assert!(result.is_ok());
            let response = result.unwrap();
            assert!(response.properties.is_some());
            let props = response.properties.unwrap();
            assert_eq!(props.get("owner"), Some(&"test_user".to_string()));
            assert_eq!(props.get("environment"), Some(&"production".to_string()));
        }

        #[tokio::test]
        async fn test_root_namespace_operations() {
            let fixture = RestServerFixture::new(4014).await;

            // Root namespace should always exist
            let exists_req = NamespaceExistsRequest { id: Some(vec![]) };
            let result = fixture.namespace.namespace_exists(exists_req).await;
            assert!(result.is_ok(), "Root namespace should exist");

            // Cannot create root namespace
            let create_req = CreateNamespaceRequest {
                id: Some(vec![]),
                properties: None,
                mode: None,
            };
            let result = fixture.namespace.create_namespace(create_req).await;
            assert!(result.is_err(), "Cannot create root namespace");
            let err_msg = result.unwrap_err().to_string();
            assert!(
                err_msg.contains("Root namespace already exists and cannot be created"),
                "Error should be 'Root namespace already exists and cannot be created', got: {}",
                err_msg
            );

            // Cannot drop root namespace
            let mut drop_req = DropNamespaceRequest::new();
            drop_req.id = Some(vec![]);
            let result = fixture.namespace.drop_namespace(drop_req).await;
            assert!(result.is_err(), "Cannot drop root namespace");
            let err_msg = result.unwrap_err().to_string();
            assert!(
                err_msg.contains("Root namespace cannot be dropped"),
                "Error should be 'Root namespace cannot be dropped', got: {}",
                err_msg
            );
        }
    }
}
