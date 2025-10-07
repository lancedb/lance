//! Lance Namespace base interface and implementations.

use async_trait::async_trait;
use bytes::Bytes;
use thiserror::Error;

use lance_namespace_reqwest_client::models::{
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

/// Error type for namespace operations
#[derive(Debug, Error)]
pub enum NamespaceError {
    #[error("Operation not supported: {0}")]
    NotSupported(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Other error: {0}")]
    Other(String),
}

/// Result type for namespace operations
pub type Result<T> = std::result::Result<T, NamespaceError>;

/// Base trait for Lance Namespace implementations.
///
/// This trait defines the interface that all Lance namespace implementations
/// must provide. Each method corresponds to a specific operation on namespaces
/// or tables.
#[async_trait]
pub trait LanceNamespace: Send + Sync {
    /// List namespaces.
    async fn list_namespaces(
        &self,
        _request: ListNamespacesRequest,
    ) -> Result<ListNamespacesResponse> {
        Err(NamespaceError::NotSupported("list_namespaces".to_string()))
    }

    /// Describe a namespace.
    async fn describe_namespace(
        &self,
        _request: DescribeNamespaceRequest,
    ) -> Result<DescribeNamespaceResponse> {
        Err(NamespaceError::NotSupported(
            "describe_namespace".to_string(),
        ))
    }

    /// Create a new namespace.
    async fn create_namespace(
        &self,
        _request: CreateNamespaceRequest,
    ) -> Result<CreateNamespaceResponse> {
        Err(NamespaceError::NotSupported("create_namespace".to_string()))
    }

    /// Drop a namespace.
    async fn drop_namespace(
        &self,
        _request: DropNamespaceRequest,
    ) -> Result<DropNamespaceResponse> {
        Err(NamespaceError::NotSupported("drop_namespace".to_string()))
    }

    /// Check if a namespace exists.
    async fn namespace_exists(&self, _request: NamespaceExistsRequest) -> Result<()> {
        Err(NamespaceError::NotSupported("namespace_exists".to_string()))
    }

    /// List tables in a namespace.
    async fn list_tables(&self, _request: ListTablesRequest) -> Result<ListTablesResponse> {
        Err(NamespaceError::NotSupported("list_tables".to_string()))
    }

    /// Describe a table.
    async fn describe_table(
        &self,
        _request: DescribeTableRequest,
    ) -> Result<DescribeTableResponse> {
        Err(NamespaceError::NotSupported("describe_table".to_string()))
    }

    /// Register a table.
    async fn register_table(
        &self,
        _request: RegisterTableRequest,
    ) -> Result<RegisterTableResponse> {
        Err(NamespaceError::NotSupported("register_table".to_string()))
    }

    /// Check if a table exists.
    async fn table_exists(&self, _request: TableExistsRequest) -> Result<()> {
        Err(NamespaceError::NotSupported("table_exists".to_string()))
    }

    /// Drop a table.
    async fn drop_table(&self, _request: DropTableRequest) -> Result<DropTableResponse> {
        Err(NamespaceError::NotSupported("drop_table".to_string()))
    }

    /// Deregister a table.
    async fn deregister_table(
        &self,
        _request: DeregisterTableRequest,
    ) -> Result<DeregisterTableResponse> {
        Err(NamespaceError::NotSupported("deregister_table".to_string()))
    }

    /// Count rows in a table.
    async fn count_table_rows(&self, _request: CountTableRowsRequest) -> Result<i64> {
        Err(NamespaceError::NotSupported("count_table_rows".to_string()))
    }

    /// Create a new table with data from Arrow IPC stream.
    async fn create_table(
        &self,
        _request: CreateTableRequest,
        _request_data: Bytes,
    ) -> Result<CreateTableResponse> {
        Err(NamespaceError::NotSupported("create_table".to_string()))
    }

    /// Create an empty table (metadata only operation).
    async fn create_empty_table(
        &self,
        _request: CreateEmptyTableRequest,
    ) -> Result<CreateEmptyTableResponse> {
        Err(NamespaceError::NotSupported(
            "create_empty_table".to_string(),
        ))
    }

    /// Insert data into a table.
    async fn insert_into_table(
        &self,
        _request: InsertIntoTableRequest,
        _request_data: Bytes,
    ) -> Result<InsertIntoTableResponse> {
        Err(NamespaceError::NotSupported(
            "insert_into_table".to_string(),
        ))
    }

    /// Merge insert data into a table.
    async fn merge_insert_into_table(
        &self,
        _request: MergeInsertIntoTableRequest,
        _request_data: Bytes,
    ) -> Result<MergeInsertIntoTableResponse> {
        Err(NamespaceError::NotSupported(
            "merge_insert_into_table".to_string(),
        ))
    }

    /// Update a table.
    async fn update_table(&self, _request: UpdateTableRequest) -> Result<UpdateTableResponse> {
        Err(NamespaceError::NotSupported("update_table".to_string()))
    }

    /// Delete from a table.
    async fn delete_from_table(
        &self,
        _request: DeleteFromTableRequest,
    ) -> Result<DeleteFromTableResponse> {
        Err(NamespaceError::NotSupported(
            "delete_from_table".to_string(),
        ))
    }

    /// Query a table.
    async fn query_table(&self, _request: QueryTableRequest) -> Result<Bytes> {
        Err(NamespaceError::NotSupported("query_table".to_string()))
    }

    /// Create a table index.
    async fn create_table_index(
        &self,
        _request: CreateTableIndexRequest,
    ) -> Result<CreateTableIndexResponse> {
        Err(NamespaceError::NotSupported(
            "create_table_index".to_string(),
        ))
    }

    /// List table indices.
    async fn list_table_indices(
        &self,
        _request: ListTableIndicesRequest,
    ) -> Result<ListTableIndicesResponse> {
        Err(NamespaceError::NotSupported(
            "list_table_indices".to_string(),
        ))
    }

    /// Describe table index statistics.
    async fn describe_table_index_stats(
        &self,
        _request: DescribeTableIndexStatsRequest,
    ) -> Result<DescribeTableIndexStatsResponse> {
        Err(NamespaceError::NotSupported(
            "describe_table_index_stats".to_string(),
        ))
    }

    /// Describe a transaction.
    async fn describe_transaction(
        &self,
        _request: DescribeTransactionRequest,
    ) -> Result<DescribeTransactionResponse> {
        Err(NamespaceError::NotSupported(
            "describe_transaction".to_string(),
        ))
    }

    /// Alter a transaction.
    async fn alter_transaction(
        &self,
        _request: AlterTransactionRequest,
    ) -> Result<AlterTransactionResponse> {
        Err(NamespaceError::NotSupported(
            "alter_transaction".to_string(),
        ))
    }
}
