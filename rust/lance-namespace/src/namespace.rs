// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Namespace base interface and implementations.

use async_trait::async_trait;
use bytes::Bytes;
use lance_core::{Error, Result};
use snafu::Location;

use lance_namespace_reqwest_client::models::{
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

/// Base trait for Lance Namespace implementations.
///
/// This trait defines the interface that all Lance namespace implementations
/// must provide. Each method corresponds to a specific operation on namespaces
/// or tables.
///
/// # Error Handling
///
/// All operations may return the following common errors (via [`crate::NamespaceError`]):
///
/// - [`crate::ErrorCode::Unsupported`] - Operation not supported by this backend
/// - [`crate::ErrorCode::InvalidInput`] - Invalid request parameters
/// - [`crate::ErrorCode::PermissionDenied`] - Insufficient permissions
/// - [`crate::ErrorCode::Unauthenticated`] - Invalid credentials
/// - [`crate::ErrorCode::ServiceUnavailable`] - Service temporarily unavailable
/// - [`crate::ErrorCode::Internal`] - Unexpected internal error
///
/// See individual method documentation for operation-specific errors.
#[async_trait]
pub trait LanceNamespace: Send + Sync + std::fmt::Debug {
    /// List namespaces.
    ///
    /// # Errors
    ///
    /// Returns [`crate::ErrorCode::NamespaceNotFound`] if the parent namespace does not exist.
    async fn list_namespaces(
        &self,
        _request: ListNamespacesRequest,
    ) -> Result<ListNamespacesResponse> {
        Err(Error::NotSupported {
            source: "list_namespaces not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Describe a namespace.
    ///
    /// # Errors
    ///
    /// Returns [`crate::ErrorCode::NamespaceNotFound`] if the namespace does not exist.
    async fn describe_namespace(
        &self,
        _request: DescribeNamespaceRequest,
    ) -> Result<DescribeNamespaceResponse> {
        Err(Error::NotSupported {
            source: "describe_namespace not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Create a new namespace.
    ///
    /// # Errors
    ///
    /// Returns [`crate::ErrorCode::NamespaceAlreadyExists`] if a namespace with the same name already exists.
    async fn create_namespace(
        &self,
        _request: CreateNamespaceRequest,
    ) -> Result<CreateNamespaceResponse> {
        Err(Error::NotSupported {
            source: "create_namespace not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Drop a namespace.
    ///
    /// # Errors
    ///
    /// - [`crate::ErrorCode::NamespaceNotFound`] if the namespace does not exist.
    /// - [`crate::ErrorCode::NamespaceNotEmpty`] if the namespace contains tables or child namespaces.
    async fn drop_namespace(
        &self,
        _request: DropNamespaceRequest,
    ) -> Result<DropNamespaceResponse> {
        Err(Error::NotSupported {
            source: "drop_namespace not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Check if a namespace exists.
    ///
    /// # Errors
    ///
    /// Returns [`crate::ErrorCode::NamespaceNotFound`] if the namespace does not exist.
    async fn namespace_exists(&self, _request: NamespaceExistsRequest) -> Result<()> {
        Err(Error::NotSupported {
            source: "namespace_exists not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// List tables in a namespace.
    async fn list_tables(&self, _request: ListTablesRequest) -> Result<ListTablesResponse> {
        Err(Error::NotSupported {
            source: "list_tables not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Describe a table.
    async fn describe_table(
        &self,
        _request: DescribeTableRequest,
    ) -> Result<DescribeTableResponse> {
        Err(Error::NotSupported {
            source: "describe_table not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Register a table.
    async fn register_table(
        &self,
        _request: RegisterTableRequest,
    ) -> Result<RegisterTableResponse> {
        Err(Error::NotSupported {
            source: "register_table not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Check if a table exists.
    async fn table_exists(&self, _request: TableExistsRequest) -> Result<()> {
        Err(Error::NotSupported {
            source: "table_exists not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Drop a table.
    async fn drop_table(&self, _request: DropTableRequest) -> Result<DropTableResponse> {
        Err(Error::NotSupported {
            source: "drop_table not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Deregister a table.
    async fn deregister_table(
        &self,
        _request: DeregisterTableRequest,
    ) -> Result<DeregisterTableResponse> {
        Err(Error::NotSupported {
            source: "deregister_table not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Count rows in a table.
    async fn count_table_rows(&self, _request: CountTableRowsRequest) -> Result<i64> {
        Err(Error::NotSupported {
            source: "count_table_rows not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Create a new table with data from Arrow IPC stream.
    async fn create_table(
        &self,
        _request: CreateTableRequest,
        _request_data: Bytes,
    ) -> Result<CreateTableResponse> {
        Err(Error::NotSupported {
            source: "create_table not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Declare a table (metadata only operation).
    async fn declare_table(&self, _request: DeclareTableRequest) -> Result<DeclareTableResponse> {
        Err(Error::NotSupported {
            source: "declare_table not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Create an empty table (metadata only operation).
    ///
    /// # Deprecated
    ///
    /// Use [`declare_table`](Self::declare_table) instead. Support will be removed in 3.0.0.
    #[deprecated(
        since = "2.0.0",
        note = "Use declare_table instead. Support will be removed in 3.0.0."
    )]
    async fn create_empty_table(
        &self,
        _request: CreateEmptyTableRequest,
    ) -> Result<CreateEmptyTableResponse> {
        Err(Error::NotSupported {
            source: "create_empty_table not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Insert data into a table.
    async fn insert_into_table(
        &self,
        _request: InsertIntoTableRequest,
        _request_data: Bytes,
    ) -> Result<InsertIntoTableResponse> {
        Err(Error::NotSupported {
            source: "insert_into_table not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Merge insert data into a table.
    async fn merge_insert_into_table(
        &self,
        _request: MergeInsertIntoTableRequest,
        _request_data: Bytes,
    ) -> Result<MergeInsertIntoTableResponse> {
        Err(Error::NotSupported {
            source: "merge_insert_into_table not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Update a table.
    async fn update_table(&self, _request: UpdateTableRequest) -> Result<UpdateTableResponse> {
        Err(Error::NotSupported {
            source: "update_table not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Delete from a table.
    async fn delete_from_table(
        &self,
        _request: DeleteFromTableRequest,
    ) -> Result<DeleteFromTableResponse> {
        Err(Error::NotSupported {
            source: "delete_from_table not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Query a table.
    async fn query_table(&self, _request: QueryTableRequest) -> Result<Bytes> {
        Err(Error::NotSupported {
            source: "query_table not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Create a table index.
    async fn create_table_index(
        &self,
        _request: CreateTableIndexRequest,
    ) -> Result<CreateTableIndexResponse> {
        Err(Error::NotSupported {
            source: "create_table_index not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// List table indices.
    async fn list_table_indices(
        &self,
        _request: ListTableIndicesRequest,
    ) -> Result<ListTableIndicesResponse> {
        Err(Error::NotSupported {
            source: "list_table_indices not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Describe table index statistics.
    async fn describe_table_index_stats(
        &self,
        _request: DescribeTableIndexStatsRequest,
    ) -> Result<DescribeTableIndexStatsResponse> {
        Err(Error::NotSupported {
            source: "describe_table_index_stats not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Describe a transaction.
    async fn describe_transaction(
        &self,
        _request: DescribeTransactionRequest,
    ) -> Result<DescribeTransactionResponse> {
        Err(Error::NotSupported {
            source: "describe_transaction not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Alter a transaction.
    async fn alter_transaction(
        &self,
        _request: AlterTransactionRequest,
    ) -> Result<AlterTransactionResponse> {
        Err(Error::NotSupported {
            source: "alter_transaction not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Create a scalar index on a table.
    async fn create_table_scalar_index(
        &self,
        _request: CreateTableIndexRequest,
    ) -> Result<CreateTableScalarIndexResponse> {
        Err(Error::NotSupported {
            source: "create_table_scalar_index not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Drop a table index.
    async fn drop_table_index(
        &self,
        _request: DropTableIndexRequest,
    ) -> Result<DropTableIndexResponse> {
        Err(Error::NotSupported {
            source: "drop_table_index not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// List all tables across all namespaces.
    async fn list_all_tables(&self, _request: ListTablesRequest) -> Result<ListTablesResponse> {
        Err(Error::NotSupported {
            source: "list_all_tables not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Restore a table to a specific version.
    async fn restore_table(&self, _request: RestoreTableRequest) -> Result<RestoreTableResponse> {
        Err(Error::NotSupported {
            source: "restore_table not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Rename a table.
    async fn rename_table(&self, _request: RenameTableRequest) -> Result<RenameTableResponse> {
        Err(Error::NotSupported {
            source: "rename_table not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// List all versions of a table.
    async fn list_table_versions(
        &self,
        _request: ListTableVersionsRequest,
    ) -> Result<ListTableVersionsResponse> {
        Err(Error::NotSupported {
            source: "list_table_versions not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Update table schema metadata.
    async fn update_table_schema_metadata(
        &self,
        _request: UpdateTableSchemaMetadataRequest,
    ) -> Result<UpdateTableSchemaMetadataResponse> {
        Err(Error::NotSupported {
            source: "update_table_schema_metadata not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Get table statistics.
    async fn get_table_stats(
        &self,
        _request: GetTableStatsRequest,
    ) -> Result<GetTableStatsResponse> {
        Err(Error::NotSupported {
            source: "get_table_stats not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Explain a table query plan.
    async fn explain_table_query_plan(
        &self,
        _request: ExplainTableQueryPlanRequest,
    ) -> Result<String> {
        Err(Error::NotSupported {
            source: "explain_table_query_plan not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Analyze a table query plan.
    async fn analyze_table_query_plan(
        &self,
        _request: AnalyzeTableQueryPlanRequest,
    ) -> Result<String> {
        Err(Error::NotSupported {
            source: "analyze_table_query_plan not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Add columns to a table.
    async fn alter_table_add_columns(
        &self,
        _request: AlterTableAddColumnsRequest,
    ) -> Result<AlterTableAddColumnsResponse> {
        Err(Error::NotSupported {
            source: "alter_table_add_columns not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Alter columns in a table.
    async fn alter_table_alter_columns(
        &self,
        _request: AlterTableAlterColumnsRequest,
    ) -> Result<AlterTableAlterColumnsResponse> {
        Err(Error::NotSupported {
            source: "alter_table_alter_columns not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Drop columns from a table.
    async fn alter_table_drop_columns(
        &self,
        _request: AlterTableDropColumnsRequest,
    ) -> Result<AlterTableDropColumnsResponse> {
        Err(Error::NotSupported {
            source: "alter_table_drop_columns not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// List all tags for a table.
    async fn list_table_tags(
        &self,
        _request: ListTableTagsRequest,
    ) -> Result<ListTableTagsResponse> {
        Err(Error::NotSupported {
            source: "list_table_tags not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Get the version for a specific tag.
    async fn get_table_tag_version(
        &self,
        _request: GetTableTagVersionRequest,
    ) -> Result<GetTableTagVersionResponse> {
        Err(Error::NotSupported {
            source: "get_table_tag_version not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Create a tag for a table.
    async fn create_table_tag(
        &self,
        _request: CreateTableTagRequest,
    ) -> Result<CreateTableTagResponse> {
        Err(Error::NotSupported {
            source: "create_table_tag not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Delete a tag from a table.
    async fn delete_table_tag(
        &self,
        _request: DeleteTableTagRequest,
    ) -> Result<DeleteTableTagResponse> {
        Err(Error::NotSupported {
            source: "delete_table_tag not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Update a tag for a table.
    async fn update_table_tag(
        &self,
        _request: UpdateTableTagRequest,
    ) -> Result<UpdateTableTagResponse> {
        Err(Error::NotSupported {
            source: "update_table_tag not implemented".into(),
            location: Location::new(file!(), line!(), column!()),
        })
    }

    /// Return a human-readable unique identifier for this namespace instance.
    ///
    /// This is used for equality comparison and hashing when the namespace is
    /// used as part of a storage options provider. Two namespace instances with
    /// the same ID are considered equal and will share cached resources.
    ///
    /// The ID should be human-readable for debugging and logging purposes.
    /// For example:
    /// - REST namespace: `"rest(endpoint=https://api.example.com)"`
    /// - Directory namespace: `"dir(root=/path/to/data)"`
    ///
    /// Implementations should include all configuration that uniquely identifies
    /// the namespace to provide semantic equality.
    fn namespace_id(&self) -> String;
}
