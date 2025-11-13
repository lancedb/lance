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
            .route("/v1/table/:id/count_rows", post(count_table_rows))
            // Table data operations
            .route("/v1/table/:id/create", post(create_table))
            .route("/v1/table/:id/create-empty", post(create_empty_table))
            .route("/v1/table/:id/insert", post(insert_into_table))
            .route("/v1/table/:id/merge_insert", post(merge_insert_into_table))
            .route("/v1/table/:id/update", post(update_table))
            .route("/v1/table/:id/delete", post(delete_from_table))
            .route("/v1/table/:id/query", post(query_table))
            // Index operations
            .route("/v1/table/:id/create_index", post(create_table_index))
            .route("/v1/table/:id/index/list", post(list_table_indices))
            .route(
                "/v1/table/:id/index/:index_name/stats",
                post(describe_table_index_stats),
            )
            // Transaction operations
            .route("/v1/transaction/:id/describe", post(describe_transaction))
            .route("/v1/transaction/:id/alter", post(alter_transaction))
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

async fn count_table_rows(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<CountTableRowsRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.count_table_rows(request).await {
        Ok(count) => (StatusCode::OK, Json(serde_json::json!({ "count": count }))).into_response(),
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

#[derive(Debug, Deserialize)]
struct InsertQuery {
    delimiter: Option<String>,
    mode: Option<String>,
}

async fn insert_into_table(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<InsertQuery>,
    body: Bytes,
) -> Response {
    use lance_namespace::models::insert_into_table_request::Mode;

    let mode = params.mode.as_deref().and_then(|m| match m {
        "append" => Some(Mode::Append),
        "overwrite" => Some(Mode::Overwrite),
        _ => None,
    });

    let request = InsertIntoTableRequest {
        id: Some(parse_id(&id, params.delimiter.as_deref())),
        mode,
    };

    match backend.insert_into_table(request, body).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

#[derive(Debug, Deserialize)]
struct MergeInsertQuery {
    delimiter: Option<String>,
    on: Option<String>,
    when_matched_update_all: Option<bool>,
    when_matched_update_all_filt: Option<String>,
    when_not_matched_insert_all: Option<bool>,
    when_not_matched_by_source_delete: Option<bool>,
    when_not_matched_by_source_delete_filt: Option<String>,
}

async fn merge_insert_into_table(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<MergeInsertQuery>,
    body: Bytes,
) -> Response {
    // Build request from query parameters
    let request = MergeInsertIntoTableRequest {
        id: Some(parse_id(&id, params.delimiter.as_deref())),
        on: params.on,
        when_matched_update_all: params.when_matched_update_all,
        when_matched_update_all_filt: params.when_matched_update_all_filt,
        when_not_matched_insert_all: params.when_not_matched_insert_all,
        when_not_matched_by_source_delete: params.when_not_matched_by_source_delete,
        when_not_matched_by_source_delete_filt: params.when_not_matched_by_source_delete_filt,
    };

    match backend.merge_insert_into_table(request, body).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn update_table(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<UpdateTableRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.update_table(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn delete_from_table(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<DeleteFromTableRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.delete_from_table(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn query_table(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<QueryTableRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.query_table(request).await {
        Ok(data) => (
            StatusCode::OK,
            [(
                axum::http::header::CONTENT_TYPE,
                "application/vnd.apache.arrow.stream",
            )],
            data,
        )
            .into_response(),
        Err(e) => error_to_response(e),
    }
}

// ============================================================================
// Index Operation Handlers
// ============================================================================

async fn create_table_index(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<CreateTableIndexRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.create_table_index(request).await {
        Ok(response) => (StatusCode::CREATED, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn list_table_indices(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<ListTableIndicesRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.list_table_indices(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

#[derive(Debug, Deserialize)]
struct IndexStatsPath {
    id: String,
    #[allow(dead_code)]
    index_name: String,
}

async fn describe_table_index_stats(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(path): Path<IndexStatsPath>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<DescribeTableIndexStatsRequest>,
) -> Response {
    request.id = Some(parse_id(&path.id, params.delimiter.as_deref()));

    match backend.describe_table_index_stats(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

// ============================================================================
// Transaction Operation Handlers
// ============================================================================

async fn describe_transaction(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<DescribeTransactionRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.describe_transaction(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn alter_transaction(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<AlterTransactionRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.alter_transaction(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
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
}
