// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Namespace base interface and implementations.

use async_trait::async_trait;
use bytes::Bytes;
use lance_core::{Error, Result};
use snafu::Location;

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

/// Base trait for Lance Namespace implementations.
///
/// This trait defines the interface that all Lance namespace implementations
/// must provide. Each method corresponds to a specific operation on namespaces
/// or tables.
#[async_trait]
pub trait LanceNamespace: Send + Sync + std::fmt::Debug {
    /// List namespaces.
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

    /// Create an empty table (metadata only operation).
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
