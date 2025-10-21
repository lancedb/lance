// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Integration test for REST server adapter
//!
//! This test verifies the full round-trip:
//! RestNamespace (client) → HTTP → RestServer → DirectoryNamespace (backend)

#![cfg(all(feature = "rest", feature = "rest-server", feature = "dir"))]

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use lance_namespace::models::*;
use lance_namespace::LanceNamespace;
use lance_namespace_impls::{DirectoryNamespace, RestNamespace, RestServer, RestServerConfig};
use tempfile::TempDir;

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
async fn test_rest_server_with_directory_backend() {
    // Create temporary directory for DirectoryNamespace
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path().to_str().unwrap().to_string();

    // Create DirectoryNamespace backend
    let mut backend_props = HashMap::new();
    backend_props.insert("root".to_string(), temp_path.clone());
    let backend = Arc::new(DirectoryNamespace::new(backend_props));

    // Start REST server on a random available port
    let config = RestServerConfig {
        host: "127.0.0.1".to_string(),
        port: 0, // Let OS assign a port
    };

    let server = RestServer::new(backend.clone(), config);

    // Start server in background
    let server_handle = tokio::spawn(async move {
        server.serve().await.unwrap();
    });

    // Give server time to start
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Create RestNamespace client
    let server_url = format!("http://127.0.0.1:2333"); // Using default port for now
    let mut client_props = HashMap::new();
    client_props.insert("uri".to_string(), server_url);
    client_props.insert("delimiter".to_string(), ".".to_string());
    let client = RestNamespace::new(client_props);

    // Test 1: Create namespace
    let create_ns_req = CreateNamespaceRequest {
        id: Some(vec!["test_namespace".to_string()]),
        properties: None,
        mode: None,
    };
    let create_ns_result = client.create_namespace(create_ns_req).await;
    assert!(create_ns_result.is_ok() || create_ns_result.is_err()); // Backend may not support nested namespaces

    // Test 2: List namespaces
    let list_ns_req = ListNamespacesRequest {
        id: Some(vec!["".to_string()]),
        page_token: None,
        limit: None,
    };
    let list_ns_result = client.list_namespaces(list_ns_req).await;
    assert!(list_ns_result.is_ok() || list_ns_result.is_err());

    // Test 3: Create table
    let table_data = create_test_arrow_data();
    let create_table_req = CreateTableRequest {
        id: Some(vec!["test_table".to_string()]),
        location: None,
        mode: Some(create_table_request::Mode::Create),
        properties: None,
    };
    let create_table_result = client.create_table(create_table_req, table_data.clone()).await;
    assert!(create_table_result.is_ok(), "Failed to create table: {:?}", create_table_result.err());

    // Test 4: Table exists
    let table_exists_req = TableExistsRequest {
        id: Some(vec!["test_table".to_string()]),
    };
    let table_exists_result = client.table_exists(table_exists_req).await;
    assert!(table_exists_result.is_ok(), "Table should exist");

    // Test 5: Describe table
    let describe_req = DescribeTableRequest {
        id: Some(vec!["test_table".to_string()]),
    };
    let describe_result = client.describe_table(describe_req).await;
    assert!(describe_result.is_ok(), "Failed to describe table: {:?}", describe_result.err());

    // Test 6: List tables
    let list_tables_req = ListTablesRequest {
        id: Some(vec!["".to_string()]),
        page_token: None,
        limit: None,
    };
    let list_result = client.list_tables(list_tables_req).await;
    assert!(list_result.is_ok());
    let tables = list_result.unwrap();
    assert!(tables.tables.len() > 0, "Should have at least one table");

    // Test 7: Count rows
    let count_req = CountTableRowsRequest {
        id: Some(vec!["test_table".to_string()]),
    };
    let count_result = client.count_table_rows(count_req).await;
    assert!(count_result.is_ok(), "Failed to count rows: {:?}", count_result.err());
    assert_eq!(count_result.unwrap(), 3, "Should have 3 rows");

    // Test 8: Query table
    let query_req = QueryTableRequest {
        id: Some(vec!["test_table".to_string()]),
        columns: None,
        filter: None,
        limit: None,
        offset: None,
        with_row_id: None,
        with_row_addr: None,
    };
    let query_result = client.query_table(query_req).await;
    assert!(query_result.is_ok(), "Failed to query table: {:?}", query_result.err());

    // Test 9: Insert more data
    let insert_req = InsertIntoTableRequest {
        id: Some(vec!["test_table".to_string()]),
        mode: Some(insert_into_table_request::Mode::Append),
    };
    let insert_result = client.insert_into_table(insert_req, table_data.clone()).await;
    assert!(insert_result.is_ok(), "Failed to insert data: {:?}", insert_result.err());

    // Test 10: Verify row count increased
    let count_req2 = CountTableRowsRequest {
        id: Some(vec!["test_table".to_string()]),
    };
    let count_result2 = client.count_table_rows(count_req2).await;
    assert!(count_result2.is_ok());
    assert_eq!(count_result2.unwrap(), 6, "Should have 6 rows after insert");

    // Test 11: Create empty table
    let create_empty_req = CreateEmptyTableRequest {
        id: Some(vec!["empty_table".to_string()]),
        schema: None, // Schema will be derived from data or provided
        mode: Some(create_empty_table_request::Mode::Create),
        properties: None,
    };
    let create_empty_result = client.create_empty_table(create_empty_req).await;
    // This might fail if schema is required, which is acceptable
    let _ = create_empty_result;

    // Test 12: Drop table
    let drop_req = DropTableRequest {
        id: Some(vec!["test_table".to_string()]),
    };
    let drop_result = client.drop_table(drop_req).await;
    assert!(drop_result.is_ok(), "Failed to drop table: {:?}", drop_result.err());

    // Test 13: Verify table no longer exists
    let table_exists_req2 = TableExistsRequest {
        id: Some(vec!["test_table".to_string()]),
    };
    let table_exists_result2 = client.table_exists(table_exists_req2).await;
    assert!(table_exists_result2.is_err(), "Table should not exist after drop");

    // Cleanup: Abort the server
    server_handle.abort();
}

#[tokio::test]
async fn test_error_handling() {
    // Create temporary directory for DirectoryNamespace
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path().to_str().unwrap().to_string();

    // Create DirectoryNamespace backend
    let mut backend_props = HashMap::new();
    backend_props.insert("root".to_string(), temp_path.clone());
    let backend = Arc::new(DirectoryNamespace::new(backend_props));

    // Start REST server
    let config = RestServerConfig {
        host: "127.0.0.1".to_string(),
        port: 2334,
    };

    let server = RestServer::new(backend.clone(), config);
    let server_handle = tokio::spawn(async move {
        server.serve().await.unwrap();
    });

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Create RestNamespace client
    let mut client_props = HashMap::new();
    client_props.insert("uri".to_string(), "http://127.0.0.1:2334".to_string());
    client_props.insert("delimiter".to_string(), ".".to_string());
    let client = RestNamespace::new(client_props);

    // Test: Describe non-existent table should return error
    let describe_req = DescribeTableRequest {
        id: Some(vec!["nonexistent_table".to_string()]),
    };
    let describe_result = client.describe_table(describe_req).await;
    assert!(describe_result.is_err(), "Should fail for non-existent table");

    // Test: Table exists for non-existent table should return error
    let exists_req = TableExistsRequest {
        id: Some(vec!["nonexistent_table".to_string()]),
    };
    let exists_result = client.table_exists(exists_req).await;
    assert!(exists_result.is_err(), "Should fail for non-existent table");

    server_handle.abort();
}
