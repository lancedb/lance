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
    extract::{Path, Query, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router, ServiceExt,
};
use serde::Deserialize;
use tokio::sync::watch;
use tower::Layer;
use tower_http::normalize_path::NormalizePathLayer;
use tower_http::trace::TraceLayer;

use lance_core::{Error, Result};
use lance_namespace::models::*;
use lance_namespace::LanceNamespace;

/// Configuration for the REST server
#[derive(Debug, Clone)]
pub struct RestAdapterConfig {
    /// Host address to bind to
    pub host: String,
    /// Port to listen on
    pub port: u16,
}

impl Default for RestAdapterConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 2333,
        }
    }
}

/// REST server adapter that wraps a Lance Namespace implementation
pub struct RestAdapter {
    backend: Arc<dyn LanceNamespace>,
    config: RestAdapterConfig,
}

impl RestAdapter {
    /// Create a new REST server with the given backend namespace
    pub fn new(backend: Arc<dyn LanceNamespace>, config: RestAdapterConfig) -> Self {
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
            // Table metadata operations
            .route("/v1/table/:id/register", post(register_table))
            .route("/v1/table/:id/describe", post(describe_table))
            .route("/v1/table/:id/exists", post(table_exists))
            .route("/v1/table/:id/drop", post(drop_table))
            .route("/v1/table/:id/deregister", post(deregister_table))
            .route("/v1/table/:id/rename", post(rename_table))
            .route("/v1/table/:id/restore", post(restore_table))
            .route("/v1/table/:id/version/list", get(list_table_versions))
            .route("/v1/table/:id/stats", get(get_table_stats))
            // Table data operations
            .route("/v1/table/:id/create", post(create_table))
            .route("/v1/table/:id/create-empty", post(create_empty_table))
            .route("/v1/table/:id/declare", post(declare_table))
            .route("/v1/table/:id/insert", post(insert_into_table))
            .route("/v1/table/:id/merge_insert", post(merge_insert_into_table))
            .route("/v1/table/:id/update", post(update_table))
            .route("/v1/table/:id/delete", post(delete_from_table))
            .route("/v1/table/:id/query", post(query_table))
            .route("/v1/table/:id/count_rows", get(count_table_rows))
            // Index operations
            .route("/v1/table/:id/create_index", post(create_table_index))
            .route(
                "/v1/table/:id/create_scalar_index",
                post(create_table_scalar_index),
            )
            .route("/v1/table/:id/index/list", get(list_table_indices))
            .route(
                "/v1/table/:id/index/:index_name/stats",
                get(describe_table_index_stats),
            )
            .route(
                "/v1/table/:id/index/:index_name/drop",
                post(drop_table_index),
            )
            // Schema operations
            .route("/v1/table/:id/add_columns", post(alter_table_add_columns))
            .route(
                "/v1/table/:id/alter_columns",
                post(alter_table_alter_columns),
            )
            .route("/v1/table/:id/drop_columns", post(alter_table_drop_columns))
            .route(
                "/v1/table/:id/schema_metadata/update",
                post(update_table_schema_metadata),
            )
            // Tag operations
            .route("/v1/table/:id/tags/list", get(list_table_tags))
            .route("/v1/table/:id/tags/version", post(get_table_tag_version))
            .route("/v1/table/:id/tags/create", post(create_table_tag))
            .route("/v1/table/:id/tags/delete", post(delete_table_tag))
            .route("/v1/table/:id/tags/update", post(update_table_tag))
            // Query plan operations
            .route("/v1/table/:id/explain_plan", post(explain_table_query_plan))
            .route("/v1/table/:id/analyze_plan", post(analyze_table_query_plan))
            // Transaction operations
            .route("/v1/transaction/:id/describe", post(describe_transaction))
            .route("/v1/transaction/:id/alter", post(alter_transaction))
            // Global table operations
            .route("/v1/table", get(list_all_tables))
            .layer(TraceLayer::new_for_http())
            .with_state(self.backend.clone())
    }

    /// Start the REST server in the background and return a handle for shutdown.
    ///
    /// This method binds to the configured address and spawns a background task
    /// to handle requests. The returned handle can be used to gracefully shut down
    /// the server.
    ///
    /// Returns an error immediately if the server fails to bind to the address.
    /// If port 0 is specified, the OS will assign an available ephemeral port.
    /// The actual port can be retrieved from the returned handle via `port()`.
    pub async fn start(self) -> Result<RestAdapterHandle> {
        let addr = format!("{}:{}", self.config.host, self.config.port);

        let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| {
            log::error!("RestAdapter::start() failed to bind to {}: {}", addr, e);
            Error::IO {
                source: Box::new(e),
                location: snafu::location!(),
            }
        })?;

        // Get the actual port (important when port 0 was specified)
        let actual_port = listener.local_addr().map(|a| a.port()).unwrap_or(0);

        let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
        let router = self.router();
        let app = NormalizePathLayer::trim_trailing_slash().layer(router);

        tokio::spawn(async move {
            let result = axum::serve(listener, ServiceExt::<Request>::into_make_service(app))
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.changed().await;
                })
                .await;

            if let Err(e) = result {
                log::error!("RestAdapter: server error: {}", e);
            }

            // Signal that server has shut down
            let _ = done_tx.send(());
        });

        Ok(RestAdapterHandle {
            shutdown_tx,
            done_rx: std::sync::Mutex::new(Some(done_rx)),
            port: actual_port,
        })
    }
}

/// Handle for controlling a running REST adapter server.
///
/// Use this handle to gracefully shut down the server when it's no longer needed.
pub struct RestAdapterHandle {
    shutdown_tx: watch::Sender<bool>,
    done_rx: std::sync::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
    port: u16,
}

impl RestAdapterHandle {
    /// Get the actual port the server is listening on.
    /// This is useful when port 0 was specified to get an OS-assigned port.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Gracefully shut down the server and wait for it to complete.
    ///
    /// This signals the server to stop accepting new connections, waits for
    /// existing connections to complete, and blocks until the server has
    /// fully shut down.
    pub fn shutdown(&self) {
        // Send shutdown signal
        let _ = self.shutdown_tx.send(true);

        // Wait for server to complete
        if let Some(done_rx) = self.done_rx.lock().unwrap().take() {
            // Use a new runtime to block on the oneshot receiver
            // This is needed because shutdown() is called from sync context
            let _ = std::thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                let _ = rt.block_on(done_rx);
            })
            .join();
        }
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
) -> Response {
    let request = DropTableRequest {
        id: Some(parse_id(&id, params.delimiter.as_deref())),
    };

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
}

async fn create_table(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<CreateTableQuery>,
    body: Bytes,
) -> Response {
    let request = CreateTableRequest {
        id: Some(parse_id(&id, params.delimiter.as_deref())),
        mode: params.mode.clone(),
    };

    match backend.create_table(request, body).await {
        Ok(response) => (StatusCode::CREATED, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

#[allow(deprecated)]
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

async fn declare_table(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<DeclareTableRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.declare_table(request).await {
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
    let request = InsertIntoTableRequest {
        id: Some(parse_id(&id, params.delimiter.as_deref())),
        mode: params.mode.clone(),
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
    timeout: Option<String>,
    use_index: Option<bool>,
}

async fn merge_insert_into_table(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<MergeInsertQuery>,
    body: Bytes,
) -> Response {
    let request = MergeInsertIntoTableRequest {
        id: Some(parse_id(&id, params.delimiter.as_deref())),
        on: params.on,
        when_matched_update_all: params.when_matched_update_all,
        when_matched_update_all_filt: params.when_matched_update_all_filt,
        when_not_matched_insert_all: params.when_not_matched_insert_all,
        when_not_matched_by_source_delete: params.when_not_matched_by_source_delete,
        when_not_matched_by_source_delete_filt: params.when_not_matched_by_source_delete_filt,
        timeout: params.timeout,
        use_index: params.use_index,
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
        Ok(bytes) => (StatusCode::OK, bytes).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn count_table_rows(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
) -> Response {
    let request = CountTableRowsRequest {
        id: Some(parse_id(&id, params.delimiter.as_deref())),
        version: None,
        predicate: None,
    };

    match backend.count_table_rows(request).await {
        Ok(count) => (StatusCode::OK, Json(serde_json::json!({ "count": count }))).into_response(),
        Err(e) => error_to_response(e),
    }
}

// ============================================================================
// Table Management Operation Handlers
// ============================================================================

async fn rename_table(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<RenameTableRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.rename_table(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn restore_table(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<RestoreTableRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.restore_table(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn list_table_versions(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<PaginationQuery>,
) -> Response {
    let request = ListTableVersionsRequest {
        id: Some(parse_id(&id, params.delimiter.as_deref())),
        page_token: params.page_token,
        limit: params.limit,
    };

    match backend.list_table_versions(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn get_table_stats(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
) -> Response {
    let request = GetTableStatsRequest {
        id: Some(parse_id(&id, params.delimiter.as_deref())),
    };

    match backend.get_table_stats(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn list_all_tables(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Query(params): Query<PaginationQuery>,
) -> Response {
    let request = ListTablesRequest {
        id: None,
        page_token: params.page_token,
        limit: params.limit,
    };

    match backend.list_all_tables(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
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

async fn create_table_scalar_index(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<CreateTableIndexRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.create_table_scalar_index(request).await {
        Ok(response) => (StatusCode::CREATED, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn list_table_indices(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
) -> Response {
    let request = ListTableIndicesRequest {
        id: Some(parse_id(&id, params.delimiter.as_deref())),
        version: None,
        page_token: None,
        limit: None,
    };

    match backend.list_table_indices(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

#[derive(Debug, Deserialize)]
struct IndexPathParams {
    id: String,
    index_name: String,
}

async fn describe_table_index_stats(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(params): Path<IndexPathParams>,
    Query(query): Query<DelimiterQuery>,
) -> Response {
    let request = DescribeTableIndexStatsRequest {
        id: Some(parse_id(&params.id, query.delimiter.as_deref())),
        version: None,
        index_name: Some(params.index_name),
    };

    match backend.describe_table_index_stats(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn drop_table_index(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(params): Path<IndexPathParams>,
    Query(query): Query<DelimiterQuery>,
) -> Response {
    let request = DropTableIndexRequest {
        id: Some(parse_id(&params.id, query.delimiter.as_deref())),
        index_name: Some(params.index_name),
    };

    match backend.drop_table_index(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

// ============================================================================
// Schema Operation Handlers
// ============================================================================

async fn alter_table_add_columns(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<AlterTableAddColumnsRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.alter_table_add_columns(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn alter_table_alter_columns(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<AlterTableAlterColumnsRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.alter_table_alter_columns(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn alter_table_drop_columns(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<AlterTableDropColumnsRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.alter_table_drop_columns(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn update_table_schema_metadata(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<UpdateTableSchemaMetadataRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.update_table_schema_metadata(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

// ============================================================================
// Tag Operation Handlers
// ============================================================================

async fn list_table_tags(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<PaginationQuery>,
) -> Response {
    let request = ListTableTagsRequest {
        id: Some(parse_id(&id, params.delimiter.as_deref())),
        page_token: params.page_token,
        limit: params.limit,
    };

    match backend.list_table_tags(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn get_table_tag_version(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<GetTableTagVersionRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.get_table_tag_version(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn create_table_tag(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<CreateTableTagRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.create_table_tag(request).await {
        Ok(response) => (StatusCode::CREATED, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn delete_table_tag(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<DeleteTableTagRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.delete_table_tag(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn update_table_tag(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<UpdateTableTagRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.update_table_tag(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

// ============================================================================
// Query Plan Operation Handlers
// ============================================================================

async fn explain_table_query_plan(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<ExplainTableQueryPlanRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.explain_table_query_plan(request).await {
        Ok(plan) => (StatusCode::OK, plan).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn analyze_table_query_plan(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(params): Query<DelimiterQuery>,
    Json(mut request): Json<AnalyzeTableQueryPlanRequest>,
) -> Response {
    request.id = Some(parse_id(&id, params.delimiter.as_deref()));

    match backend.analyze_table_query_plan(request).await {
        Ok(plan) => (StatusCode::OK, plan).into_response(),
        Err(e) => error_to_response(e),
    }
}

// ============================================================================
// Transaction Operation Handlers
// ============================================================================

async fn describe_transaction(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(_params): Query<DelimiterQuery>,
    Json(mut request): Json<DescribeTransactionRequest>,
) -> Response {
    // The path id is the transaction identifier
    // The request.id in body is the table ID (namespace path)
    // For the trait, we set request.id to include both table ID and transaction ID
    // by appending the transaction ID to the table ID path
    if let Some(ref mut table_id) = request.id {
        table_id.push(id);
    } else {
        request.id = Some(vec![id]);
    }

    match backend.describe_transaction(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => error_to_response(e),
    }
}

async fn alter_transaction(
    State(backend): State<Arc<dyn LanceNamespace>>,
    Path(id): Path<String>,
    Query(_params): Query<DelimiterQuery>,
    Json(mut request): Json<AlterTransactionRequest>,
) -> Response {
    // The path id is the transaction identifier
    // Append it to the table ID path in the request
    if let Some(ref mut table_id) = request.id {
        table_id.push(id);
    } else {
        request.id = Some(vec![id]);
    }

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
    let delimiter = delimiter.unwrap_or("$");

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
        let id = parse_id("ns1$ns2$table", None);
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
        let id = parse_id("$", None);
        assert_eq!(id, Vec::<String>::new());

        let id = parse_id("/", Some("/"));
        assert_eq!(id, Vec::<String>::new());
    }

    #[test]
    fn test_parse_id_filters_empty() {
        // Filter out empty strings from split results
        let id = parse_id("$$table$$", None);
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

        /// Test fixture that manages server lifecycle
        struct RestServerFixture {
            _temp_dir: TempDir,
            namespace: crate::RestNamespace,
            server_handle: RestAdapterHandle,
        }

        impl RestServerFixture {
            async fn new() -> Self {
                let temp_dir = TempDir::new().unwrap();
                let temp_path = temp_dir.path().to_str().unwrap().to_string();

                // Create DirectoryNamespace backend with manifest enabled
                let backend = DirectoryNamespaceBuilder::new(&temp_path)
                    .manifest_enabled(true)
                    .build()
                    .await
                    .unwrap();
                let backend = Arc::new(backend);

                // Start REST server with port 0 (OS assigns available port)
                let config = RestAdapterConfig {
                    port: 0,
                    ..Default::default()
                };

                let server = RestAdapter::new(backend.clone(), config);
                let server_handle = server.start().await.unwrap();

                // Get the actual port assigned by OS
                let actual_port = server_handle.port();

                // Create RestNamespace client
                let server_url = format!("http://127.0.0.1:{}", actual_port);
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
                self.server_handle.shutdown();
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

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_trailing_slash_handling() {
            let fixture = RestServerFixture::new().await;
            let port = fixture.server_handle.port();

            // Create a namespace using the normal API (without trailing slash)
            let create_req = CreateNamespaceRequest {
                id: Some(vec!["test_namespace".to_string()]),
                properties: None,
                mode: None,
            };
            fixture
                .namespace
                .create_namespace(create_req)
                .await
                .unwrap();

            // Test that a request with trailing slash works (using direct HTTP)
            let client = reqwest::Client::new();

            // Test POST endpoint with trailing slash
            let response = client
                .post(format!(
                    "http://127.0.0.1:{}/v1/namespace/test_namespace/exists/",
                    port
                ))
                .json(&serde_json::json!({}))
                .send()
                .await
                .unwrap();

            assert_eq!(
                response.status(),
                204,
                "POST request with trailing slash should succeed with 204 No Content"
            );

            // Test GET endpoint with trailing slash
            let response = client
                .get(format!(
                    "http://127.0.0.1:{}/v1/namespace/test_namespace/list/",
                    port
                ))
                .send()
                .await
                .unwrap();

            assert!(
                response.status().is_success(),
                "GET request with trailing slash should succeed, got status: {}",
                response.status()
            );
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_create_and_list_child_namespaces() {
            let fixture = RestServerFixture::new().await;

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

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_nested_namespace_hierarchy() {
            let fixture = RestServerFixture::new().await;

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

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_create_table_in_child_namespace() {
            let fixture = RestServerFixture::new().await;
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
                mode: Some("Create".to_string()),
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

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_list_tables_in_child_namespace() {
            let fixture = RestServerFixture::new().await;
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
                    mode: Some("Create".to_string()),
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

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_table_exists_in_child_namespace() {
            let fixture = RestServerFixture::new().await;
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
                mode: Some("Create".to_string()),
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

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        #[allow(deprecated)]
        async fn test_empty_table_exists_in_child_namespace() {
            let fixture = RestServerFixture::new().await;

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

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_describe_table_in_child_namespace() {
            let fixture = RestServerFixture::new().await;
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
                mode: Some("Create".to_string()),
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

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_drop_table_in_child_namespace() {
            let fixture = RestServerFixture::new().await;
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
                mode: Some("Create".to_string()),
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

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        #[allow(deprecated)]
        async fn test_create_empty_table_in_child_namespace() {
            let fixture = RestServerFixture::new().await;

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

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        #[allow(deprecated)]
        async fn test_describe_empty_table_in_child_namespace() {
            let fixture = RestServerFixture::new().await;

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

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        #[allow(deprecated)]
        async fn test_drop_empty_table_in_child_namespace() {
            let fixture = RestServerFixture::new().await;

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

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        #[allow(deprecated)]
        async fn test_deeply_nested_namespace_with_empty_table() {
            let fixture = RestServerFixture::new().await;

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

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_deeply_nested_namespace_with_table() {
            let fixture = RestServerFixture::new().await;
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
                mode: Some("Create".to_string()),
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

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_namespace_isolation() {
            let fixture = RestServerFixture::new().await;
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
                mode: Some("Create".to_string()),
            };
            fixture
                .namespace
                .create_table(create_table_req, table_data.clone())
                .await
                .unwrap();

            let create_table_req = CreateTableRequest {
                id: Some(vec!["namespace2".to_string(), "shared_table".to_string()]),
                mode: Some("Create".to_string()),
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

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_drop_namespace_with_tables_fails() {
            let fixture = RestServerFixture::new().await;
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
                mode: Some("Create".to_string()),
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

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_drop_empty_child_namespace() {
            let fixture = RestServerFixture::new().await;

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

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_namespace_with_properties() {
            let fixture = RestServerFixture::new().await;

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

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_root_namespace_operations() {
            let fixture = RestServerFixture::new().await;

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

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_register_table() {
            let fixture = RestServerFixture::new().await;
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

            // Create a physical table using create_table
            let create_table_req = CreateTableRequest {
                id: Some(vec![
                    "test_namespace".to_string(),
                    "physical_table".to_string(),
                ]),
                mode: Some("Create".to_string()),
            };
            fixture
                .namespace
                .create_table(create_table_req, table_data)
                .await
                .unwrap();

            // Register another table pointing to a relative path
            let register_req = RegisterTableRequest {
                id: Some(vec![
                    "test_namespace".to_string(),
                    "registered_table".to_string(),
                ]),
                location: "test_namespace$physical_table.lance".to_string(),
                mode: None,
                properties: None,
            };

            let result = fixture.namespace.register_table(register_req).await;
            assert!(
                result.is_ok(),
                "Failed to register table: {:?}",
                result.err()
            );

            let response = result.unwrap();
            assert_eq!(
                response.location,
                Some("test_namespace$physical_table.lance".to_string())
            );

            // Verify registered table exists
            let mut exists_req = TableExistsRequest::new();
            exists_req.id = Some(vec![
                "test_namespace".to_string(),
                "registered_table".to_string(),
            ]);
            let result = fixture.namespace.table_exists(exists_req).await;
            assert!(result.is_ok(), "Registered table should exist");
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_register_table_rejects_absolute_uri() {
            let fixture = RestServerFixture::new().await;

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

            // Try to register with absolute URI - should fail
            let register_req = RegisterTableRequest {
                id: Some(vec!["test_namespace".to_string(), "bad_table".to_string()]),
                location: "s3://bucket/table.lance".to_string(),
                mode: None,
                properties: None,
            };

            let result = fixture.namespace.register_table(register_req).await;
            assert!(result.is_err(), "Should reject absolute URI");
            let err_msg = result.unwrap_err().to_string();
            assert!(
                err_msg.contains("Absolute URIs are not allowed"),
                "Error should mention absolute URIs, got: {}",
                err_msg
            );
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_register_table_rejects_path_traversal() {
            let fixture = RestServerFixture::new().await;

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

            // Try to register with path traversal - should fail
            let register_req = RegisterTableRequest {
                id: Some(vec!["test_namespace".to_string(), "bad_table".to_string()]),
                location: "../outside/table.lance".to_string(),
                mode: None,
                properties: None,
            };

            let result = fixture.namespace.register_table(register_req).await;
            assert!(result.is_err(), "Should reject path traversal");
            let err_msg = result.unwrap_err().to_string();
            assert!(
                err_msg.contains("Path traversal is not allowed"),
                "Error should mention path traversal, got: {}",
                err_msg
            );
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_deregister_table() {
            let fixture = RestServerFixture::new().await;
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

            // Create a table
            let create_table_req = CreateTableRequest {
                id: Some(vec!["test_namespace".to_string(), "test_table".to_string()]),
                mode: Some("Create".to_string()),
            };
            fixture
                .namespace
                .create_table(create_table_req, table_data)
                .await
                .unwrap();

            // Verify table exists
            let mut exists_req = TableExistsRequest::new();
            exists_req.id = Some(vec!["test_namespace".to_string(), "test_table".to_string()]);
            assert!(fixture
                .namespace
                .table_exists(exists_req.clone())
                .await
                .is_ok());

            // Deregister the table
            let deregister_req = DeregisterTableRequest {
                id: Some(vec!["test_namespace".to_string(), "test_table".to_string()]),
            };
            let result = fixture.namespace.deregister_table(deregister_req).await;
            assert!(
                result.is_ok(),
                "Failed to deregister table: {:?}",
                result.err()
            );

            let response = result.unwrap();

            // Should return exact location and id
            assert!(
                response.location.is_some(),
                "Deregister response should include location"
            );
            let location = response.location.unwrap();
            assert!(
                location.ends_with("test_namespace$test_table"),
                "Location should end with test_namespace$test_table, got: {}",
                location
            );
            assert_eq!(
                response.id,
                Some(vec!["test_namespace".to_string(), "test_table".to_string()])
            );

            // Verify physical data still exists at the location
            let dataset = lance::Dataset::open(&location).await;
            assert!(
                dataset.is_ok(),
                "Physical table data should still exist at {}",
                location
            );
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_register_deregister_round_trip() {
            let fixture = RestServerFixture::new().await;
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

            // Create a physical table
            let create_table_req = CreateTableRequest {
                id: Some(vec![
                    "test_namespace".to_string(),
                    "original_table".to_string(),
                ]),
                mode: Some("Create".to_string()),
            };
            let create_response = fixture
                .namespace
                .create_table(create_table_req, table_data.clone())
                .await
                .unwrap();

            // Deregister it
            let deregister_req = DeregisterTableRequest {
                id: Some(vec![
                    "test_namespace".to_string(),
                    "original_table".to_string(),
                ]),
            };
            fixture
                .namespace
                .deregister_table(deregister_req)
                .await
                .unwrap();

            // Re-register with a different name
            let location = create_response
                .location
                .as_ref()
                .and_then(|loc| loc.strip_prefix(fixture.namespace.endpoint()))
                .unwrap_or(create_response.location.as_ref().unwrap())
                .to_string();

            let relative_location = location
                .split('/')
                .next_back()
                .unwrap_or(&location)
                .to_string();

            let register_req = RegisterTableRequest {
                id: Some(vec![
                    "test_namespace".to_string(),
                    "renamed_table".to_string(),
                ]),
                location: relative_location.clone(),
                mode: None,
                properties: None,
            };

            let register_response = fixture
                .namespace
                .register_table(register_req)
                .await
                .expect("Failed to re-register table with new name");

            // Should return the exact location we registered
            assert_eq!(register_response.location, Some(relative_location.clone()));

            // Verify new table exists
            let mut exists_req = TableExistsRequest::new();
            exists_req.id = Some(vec![
                "test_namespace".to_string(),
                "renamed_table".to_string(),
            ]);
            let result = fixture.namespace.table_exists(exists_req).await;
            assert!(result.is_ok(), "Re-registered table should exist");

            // Verify both tables point to the same physical location
            let mut describe_req = DescribeTableRequest::new();
            describe_req.id = Some(vec![
                "test_namespace".to_string(),
                "renamed_table".to_string(),
            ]);
            let describe_response = fixture
                .namespace
                .describe_table(describe_req)
                .await
                .expect("Should be able to describe renamed table");

            // Location should end with the physical table path (same as original)
            assert!(
                describe_response
                    .location
                    .as_ref()
                    .map(|loc| loc.ends_with(&relative_location))
                    .unwrap_or(false),
                "Renamed table should point to original physical location {}, got: {:?}",
                relative_location,
                describe_response.location
            );
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_namespace_write() {
            use arrow::array::Int32Array;
            use arrow::datatypes::{DataType, Field as ArrowField, Schema as ArrowSchema};
            use arrow::record_batch::{RecordBatch, RecordBatchIterator};
            use lance::dataset::{Dataset, WriteMode, WriteParams};
            use lance_namespace::LanceNamespace;

            let fixture = RestServerFixture::new().await;
            let namespace = Arc::new(fixture.namespace.clone()) as Arc<dyn LanceNamespace>;

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
    }
}
