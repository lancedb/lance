// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Integration test for REST server adapter with child namespace support
//!
//! This test verifies the full round-trip:
//! RestNamespace (client) → HTTP → RestServer → DirectoryNamespace (backend)
//!
//! All tests use child namespaces and tables within those namespaces to verify
//! full hierarchical namespace support.

#![cfg(all(feature = "rest", feature = "rest-server"))]

use std::sync::Arc;

use bytes::Bytes;
use lance_namespace::models::*;
use lance_namespace::LanceNamespace;
use lance_namespace_impls::{
    DirectoryNamespaceBuilder, RestNamespaceBuilder, RestServer, RestServerConfig,
};
use tempfile::TempDir;
use tokio::task::JoinHandle;

/// Test fixture that manages server lifecycle
struct RestServerFixture {
    _temp_dir: TempDir,
    client: lance_namespace_impls::RestNamespace,
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
        let client = RestNamespaceBuilder::new(&server_url)
            .delimiter("$")
            .build();

        Self {
            _temp_dir: temp_dir,
            client,
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
        let result = fixture.client.create_namespace(create_req).await;
        assert!(result.is_ok(), "Failed to create namespace{}", i);
    }

    // List child namespaces
    let list_req = ListNamespacesRequest {
        id: Some(vec![]),
        page_token: None,
        limit: None,
    };
    let result = fixture.client.list_namespaces(list_req).await;
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
    fixture.client.create_namespace(create_req).await.unwrap();

    // Create nested child namespaces
    let create_req = CreateNamespaceRequest {
        id: Some(vec!["parent".to_string(), "child1".to_string()]),
        properties: None,
        mode: None,
    };
    fixture.client.create_namespace(create_req).await.unwrap();

    let create_req = CreateNamespaceRequest {
        id: Some(vec!["parent".to_string(), "child2".to_string()]),
        properties: None,
        mode: None,
    };
    fixture.client.create_namespace(create_req).await.unwrap();

    // List children of parent
    let list_req = ListNamespacesRequest {
        id: Some(vec!["parent".to_string()]),
        page_token: None,
        limit: None,
    };
    let result = fixture.client.list_namespaces(list_req).await;
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
        .client
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
        .client
        .create_table(create_table_req, table_data)
        .await;

    assert!(
        result.is_ok(),
        "Failed to create table in child namespace: {:?}",
        result.err()
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
        .client
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
            .client
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
    let result = fixture.client.list_tables(list_req).await;
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
        .client
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
        .client
        .create_table(create_table_req, table_data)
        .await
        .unwrap();

    // Check table exists
    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(vec!["test_namespace".to_string(), "test_table".to_string()]);
    let result = fixture.client.table_exists(exists_req).await;
    assert!(result.is_ok(), "Table should exist in child namespace");
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
        .client
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
        .client
        .create_table(create_table_req, table_data)
        .await
        .unwrap();

    // Describe the table
    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["test_namespace".to_string(), "test_table".to_string()]);
    let result = fixture.client.describe_table(describe_req).await;

    assert!(
        result.is_ok(),
        "Failed to describe table in child namespace: {:?}",
        result.err()
    );
    let response = result.unwrap();
    assert!(response.location.is_some());
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
        .client
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
        .client
        .create_table(create_table_req, table_data)
        .await
        .unwrap();

    // Drop the table
    let drop_req = DropTableRequest {
        id: Some(vec!["test_namespace".to_string(), "test_table".to_string()]),
    };
    let result = fixture.client.drop_table(drop_req).await;
    assert!(
        result.is_ok(),
        "Failed to drop table in child namespace: {:?}",
        result.err()
    );

    // Verify table no longer exists
    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(vec!["test_namespace".to_string(), "test_table".to_string()]);
    let result = fixture.client.table_exists(exists_req).await;
    assert!(result.is_err(), "Table should not exist after drop");
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
        .client
        .create_namespace(create_ns_req)
        .await
        .unwrap();

    // Create empty table
    let mut create_req = CreateEmptyTableRequest::new();
    create_req.id = Some(vec![
        "test_namespace".to_string(),
        "empty_table".to_string(),
    ]);

    let result = fixture.client.create_empty_table(create_req).await;
    assert!(
        result.is_ok(),
        "Failed to create empty table in child namespace: {:?}",
        result.err()
    );
    let response = result.unwrap();
    assert!(response.location.is_some());

    // Verify the empty table exists
    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(vec![
        "test_namespace".to_string(),
        "empty_table".to_string(),
    ]);
    let exists_result = fixture.client.table_exists(exists_req).await;
    assert!(
        exists_result.is_ok(),
        "Empty table should exist in child namespace"
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
    fixture.client.create_namespace(create_req).await.unwrap();

    let create_req = CreateNamespaceRequest {
        id: Some(vec!["level1".to_string(), "level2".to_string()]),
        properties: None,
        mode: None,
    };
    fixture.client.create_namespace(create_req).await.unwrap();

    let create_req = CreateNamespaceRequest {
        id: Some(vec![
            "level1".to_string(),
            "level2".to_string(),
            "level3".to_string(),
        ]),
        properties: None,
        mode: None,
    };
    fixture.client.create_namespace(create_req).await.unwrap();

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
        .client
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
    let result = fixture.client.table_exists(exists_req).await;
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
    fixture.client.create_namespace(create_req).await.unwrap();

    let create_req = CreateNamespaceRequest {
        id: Some(vec!["namespace2".to_string()]),
        properties: None,
        mode: None,
    };
    fixture.client.create_namespace(create_req).await.unwrap();

    // Create table with same name in both namespaces
    let create_table_req = CreateTableRequest {
        id: Some(vec!["namespace1".to_string(), "shared_table".to_string()]),
        location: None,
        mode: Some(create_table_request::Mode::Create),
        properties: None,
    };
    fixture
        .client
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
        .client
        .create_table(create_table_req, table_data)
        .await
        .unwrap();

    // Drop table in namespace1
    let drop_req = DropTableRequest {
        id: Some(vec!["namespace1".to_string(), "shared_table".to_string()]),
    };
    fixture.client.drop_table(drop_req).await.unwrap();

    // Verify namespace1 table is gone but namespace2 table still exists
    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(vec!["namespace1".to_string(), "shared_table".to_string()]);
    assert!(fixture.client.table_exists(exists_req).await.is_err());

    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(vec!["namespace2".to_string(), "shared_table".to_string()]);
    assert!(fixture.client.table_exists(exists_req).await.is_ok());
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
        .client
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
        .client
        .create_table(create_table_req, table_data)
        .await
        .unwrap();

    // Try to drop namespace with table - should fail
    let mut drop_req = DropNamespaceRequest::new();
    drop_req.id = Some(vec!["test_namespace".to_string()]);
    let result = fixture.client.drop_namespace(drop_req).await;
    assert!(
        result.is_err(),
        "Should not be able to drop namespace with tables"
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
        .client
        .create_namespace(create_ns_req)
        .await
        .unwrap();

    // Drop empty namespace - should succeed
    let mut drop_req = DropNamespaceRequest::new();
    drop_req.id = Some(vec!["test_namespace".to_string()]);
    let result = fixture.client.drop_namespace(drop_req).await;
    assert!(
        result.is_ok(),
        "Should be able to drop empty child namespace"
    );

    // Verify namespace no longer exists
    let exists_req = NamespaceExistsRequest {
        id: Some(vec!["test_namespace".to_string()]),
    };
    let result = fixture.client.namespace_exists(exists_req).await;
    assert!(result.is_err(), "Namespace should not exist after drop");
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
    fixture.client.create_namespace(create_req).await.unwrap();

    // Describe namespace and verify properties
    let describe_req = DescribeNamespaceRequest {
        id: Some(vec!["test_namespace".to_string()]),
    };
    let result = fixture.client.describe_namespace(describe_req).await;
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
    let result = fixture.client.namespace_exists(exists_req).await;
    assert!(result.is_ok(), "Root namespace should exist");

    // Cannot create root namespace
    let create_req = CreateNamespaceRequest {
        id: Some(vec![]),
        properties: None,
        mode: None,
    };
    let result = fixture.client.create_namespace(create_req).await;
    assert!(result.is_err(), "Cannot create root namespace");

    // Cannot drop root namespace
    let mut drop_req = DropNamespaceRequest::new();
    drop_req.id = Some(vec![]);
    let result = fixture.client.drop_namespace(drop_req).await;
    assert!(result.is_err(), "Cannot drop root namespace");
}
