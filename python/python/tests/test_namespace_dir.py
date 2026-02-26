# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Tests for DirectoryNamespace implementation.

This module tests the DirectoryNamespace class which provides a directory-based
namespace implementation for organizing Lance tables and nested namespaces.

These tests mirror the Rust tests in rust/lance-namespace-impls/src/dir.rs
"""

import sys
import tempfile
import uuid
from threading import Lock

import lance
import lance.namespace
import pyarrow as pa
import pytest
from lance_namespace import (
    CreateEmptyTableRequest,
    CreateNamespaceRequest,
    CreateTableRequest,
    CreateTableVersionRequest,
    CreateTableVersionResponse,
    DeregisterTableRequest,
    DescribeNamespaceRequest,
    DescribeTableRequest,
    DescribeTableVersionRequest,
    DescribeTableVersionResponse,
    DropNamespaceRequest,
    DropTableRequest,
    ListNamespacesRequest,
    ListTablesRequest,
    ListTableVersionsRequest,
    ListTableVersionsResponse,
    NamespaceExistsRequest,
    RegisterTableRequest,
    TableExistsRequest,
    connect,
)


def create_test_data():
    """Create test PyArrow table data."""
    return pa.Table.from_pylist(
        [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
            {"id": 3, "name": "Charlie", "age": 35},
        ]
    )


def table_to_ipc_bytes(table):
    """Convert PyArrow table to IPC bytes."""
    import io

    sink = io.BytesIO()
    with pa.ipc.RecordBatchStreamWriter(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


@pytest.fixture
def temp_namespace():
    """Create a temporary DirectoryNamespace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use lance.namespace.connect() for consistency
        # Use high commit_retries for concurrent operation tests
        ns = connect(
            "dir", {"root": f"file://{tmpdir}", "commit_retries": "2147483647"}
        )
        yield ns


@pytest.fixture
def memory_namespace():
    """Create a memory-based DirectoryNamespace for testing."""
    unique_id = uuid.uuid4().hex[:8]
    # Use lance.namespace.connect() for consistency
    ns = connect("dir", {"root": f"memory://test_{unique_id}"})
    yield ns


class TestCreateTable:
    """Tests for create_table operation - mirrors Rust test_create_table."""

    def test_create_table(self, memory_namespace):
        """Test creating a table with data."""
        # Create parent namespace first
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        memory_namespace.create_namespace(create_ns_req)

        # Create table with data
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)

        create_req = CreateTableRequest(id=["workspace", "test_table"])
        response = memory_namespace.create_table(create_req, ipc_data)

        assert response is not None
        assert response.location is not None
        # Location format varies based on manifest implementation
        # Just check that it contains the table name
        assert "test_table" in response.location
        assert response.version == 1

    def test_create_table_without_data(self, memory_namespace):
        """Test creating a table without data should fail."""
        # Create parent namespace first
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        memory_namespace.create_namespace(create_ns_req)

        create_req = CreateTableRequest(id=["workspace", "test_table"])

        with pytest.raises(Exception) as exc_info:
            memory_namespace.create_table(create_req, b"")

        assert "Arrow IPC" in str(exc_info.value) or "required" in str(exc_info.value)

    def test_create_table_with_invalid_id(self, memory_namespace):
        """Test creating a table with invalid ID should fail."""
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)

        # Test with empty ID
        create_req = CreateTableRequest(id=[])
        with pytest.raises(Exception):
            memory_namespace.create_table(create_req, ipc_data)

    def test_create_table_in_child_namespace(self, memory_namespace):
        """Test creating table in child namespace works with manifest enabled."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["test_namespace"])
        memory_namespace.create_namespace(create_ns_req)

        # Create table in the namespace
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["test_namespace", "table"])
        response = memory_namespace.create_table(create_req, ipc_data)

        # Should succeed with manifest enabled
        assert response is not None
        assert response.location is not None


class TestListTables:
    """Tests for list_tables operation - mirrors Rust test_list_tables."""

    def test_list_tables_empty(self, memory_namespace):
        """Test listing tables in empty namespace."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        memory_namespace.create_namespace(create_ns_req)

        # Initially, no tables
        list_req = ListTablesRequest(id=["workspace"])
        response = memory_namespace.list_tables(list_req)
        assert len(response.tables) == 0

    def test_list_tables_with_tables(self, memory_namespace):
        """Test listing tables after creating them."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        memory_namespace.create_namespace(create_ns_req)

        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)

        # Create table1
        create_req = CreateTableRequest(id=["workspace", "table1"])
        memory_namespace.create_table(create_req, ipc_data)

        # Create table2
        create_req = CreateTableRequest(id=["workspace", "table2"])
        memory_namespace.create_table(create_req, ipc_data)

        # List tables should return both
        list_req = ListTablesRequest(id=["workspace"])
        response = memory_namespace.list_tables(list_req)
        assert len(response.tables) == 2

        # List tables returns table names as strings
        assert "table1" in response.tables
        assert "table2" in response.tables

    def test_list_tables_with_namespace_id(self, memory_namespace):
        """Test listing tables in a child namespace."""
        # Create child namespace
        create_ns_req = CreateNamespaceRequest(id=["test_namespace"])
        memory_namespace.create_namespace(create_ns_req)

        # List tables in the child namespace
        list_req = ListTablesRequest(id=["test_namespace"])
        response = memory_namespace.list_tables(list_req)

        # Should succeed and return empty list (no tables yet)
        assert len(response.tables) == 0


class TestDescribeTable:
    """Tests for describe_table operation - mirrors Rust test_describe_table."""

    def test_describe_table(self, memory_namespace):
        """Test describing a table."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        memory_namespace.create_namespace(create_ns_req)

        # Create a table
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["workspace", "test_table"])
        memory_namespace.create_table(create_req, ipc_data)

        # Describe the table
        describe_req = DescribeTableRequest(id=["workspace", "test_table"])
        response = memory_namespace.describe_table(describe_req)

        assert response is not None
        assert response.location is not None
        assert "test_table" in response.location

    def test_describe_nonexistent_table(self, memory_namespace):
        """Test describing a table that doesn't exist."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        memory_namespace.create_namespace(create_ns_req)

        describe_req = DescribeTableRequest(id=["workspace", "nonexistent"])

        with pytest.raises(Exception) as exc_info:
            memory_namespace.describe_table(describe_req)

        # Check for error message indicating table doesn't exist
        error_msg = str(exc_info.value).lower()
        assert "not found" in error_msg or "does not exist" in error_msg


class TestTableOperations:
    """Tests for various table operations."""

    def test_table_exists(self, memory_namespace):
        """Test checking if a table exists."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        memory_namespace.create_namespace(create_ns_req)

        # Create a table
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["workspace", "test_table"])
        memory_namespace.create_table(create_req, ipc_data)

        # Check it exists (should not raise)
        exists_req = TableExistsRequest(id=["workspace", "test_table"])
        memory_namespace.table_exists(exists_req)

    def test_table_not_exists(self, memory_namespace):
        """Test checking if a non-existent table exists."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        memory_namespace.create_namespace(create_ns_req)

        exists_req = TableExistsRequest(id=["workspace", "nonexistent"])

        with pytest.raises(Exception):
            memory_namespace.table_exists(exists_req)

    def test_drop_table(self, memory_namespace):
        """Test dropping a table."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        memory_namespace.create_namespace(create_ns_req)

        # Create table
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["workspace", "test_table"])
        memory_namespace.create_table(create_req, ipc_data)

        # Drop the table
        drop_req = DropTableRequest(id=["workspace", "test_table"])
        response = memory_namespace.drop_table(drop_req)
        assert response is not None

        # Verify table no longer exists
        exists_req = TableExistsRequest(id=["workspace", "test_table"])
        with pytest.raises(Exception):
            memory_namespace.table_exists(exists_req)

    def test_deregister_table(self, temp_namespace):
        """Test deregistering a table."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        temp_namespace.create_namespace(create_ns_req)

        # Create table using lance directly
        table_data = create_test_data()
        # Get root path from namespace
        ns_id = temp_namespace.namespace_id()
        import re

        match = re.search(r'root: "([^"]+)"', ns_id)
        assert match is not None
        root_path = match.group(1)

        # Create physical table
        table_uri = f"{root_path}/workspace/physical_table.lance"
        lance.write_dataset(table_data, table_uri)

        # Register the table with a relative location
        register_req = RegisterTableRequest(
            id=["workspace", "test_table"], location="workspace/physical_table.lance"
        )
        temp_namespace.register_table(register_req)

        # Deregister it
        deregister_req = DeregisterTableRequest(id=["workspace", "test_table"])
        response = temp_namespace.deregister_table(deregister_req)
        assert response is not None
        # Should return full URI to deregistered table
        # (use endswith to handle path canonicalization)
        assert response.location.endswith("/workspace/physical_table.lance"), (
            f"Expected location to end with '/workspace/physical_table.lance', "
            f"got {response.location}"
        )
        assert response.id == ["workspace", "test_table"]

    def test_register_table(self, temp_namespace):
        """Test registering an existing table."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        temp_namespace.create_namespace(create_ns_req)

        # Create physical table using lance
        table_data = create_test_data()
        ns_id = temp_namespace.namespace_id()
        import re

        match = re.search(r'root: "([^"]+)"', ns_id)
        assert match is not None
        root_path = match.group(1)

        # Create physical table
        table_uri = f"{root_path}/workspace/physical_table.lance"
        lance.write_dataset(table_data, table_uri)

        # Register with a different name using relative path
        register_req = RegisterTableRequest(
            id=["workspace", "registered_table"],
            location="workspace/physical_table.lance",
        )
        response = temp_namespace.register_table(register_req)
        assert response is not None
        assert response.location == "workspace/physical_table.lance"

        # Verify table exists
        exists_req = TableExistsRequest(id=["workspace", "registered_table"])
        temp_namespace.table_exists(exists_req)

        # Verify we can read from it
        describe_req = DescribeTableRequest(id=["workspace", "registered_table"])
        desc_response = temp_namespace.describe_table(describe_req)
        assert desc_response is not None
        # Should point to the same physical location
        # (use endswith to handle path canonicalization)
        assert desc_response.location.endswith("/workspace/physical_table.lance"), (
            f"Expected location to end with '/workspace/physical_table.lance', "
            f"got {desc_response.location}"
        )

    def test_register_table_rejects_absolute_uri(self, temp_namespace):
        """Test that register_table rejects absolute URIs."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        temp_namespace.create_namespace(create_ns_req)

        # Try to register with absolute URI - should fail
        register_req = RegisterTableRequest(
            id=["workspace", "test_table"], location="s3://bucket/table.lance"
        )
        with pytest.raises(Exception) as exc_info:
            temp_namespace.register_table(register_req)
        assert "Absolute URIs are not allowed" in str(exc_info.value)

    def test_register_table_rejects_absolute_path(self, temp_namespace):
        """Test that register_table rejects absolute paths."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        temp_namespace.create_namespace(create_ns_req)

        # Try to register with absolute path - should fail
        register_req = RegisterTableRequest(
            id=["workspace", "test_table"], location="/tmp/table.lance"
        )
        with pytest.raises(Exception) as exc_info:
            temp_namespace.register_table(register_req)
        assert "Absolute paths are not allowed" in str(exc_info.value)

    def test_register_table_rejects_path_traversal(self, temp_namespace):
        """Test that register_table rejects path traversal attempts."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        temp_namespace.create_namespace(create_ns_req)

        # Try to register with path traversal - should fail
        register_req = RegisterTableRequest(
            id=["workspace", "test_table"], location="../outside/table.lance"
        )
        with pytest.raises(Exception) as exc_info:
            temp_namespace.register_table(register_req)
        assert "Path traversal is not allowed" in str(exc_info.value)

    def test_create_empty_table(self, memory_namespace):
        """Test creating an empty table."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        memory_namespace.create_namespace(create_ns_req)

        # Create empty table with schema
        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("name", pa.string()),
            ]
        )

        create_req = CreateEmptyTableRequest(
            id=["workspace", "empty_table"], schema=schema
        )
        response = memory_namespace.create_empty_table(create_req)

        assert response is not None
        assert response.location is not None

        # Verify table exists
        exists_req = TableExistsRequest(id=["workspace", "empty_table"])
        memory_namespace.table_exists(exists_req)


class TestChildNamespaceOperations:
    """Tests for operations in child namespaces - mirrors Rust tests."""

    def test_create_table_in_child_namespace(self, memory_namespace):
        """Test creating multiple tables in a child namespace."""
        # Create child namespace
        create_ns_req = CreateNamespaceRequest(id=["test_ns"])
        memory_namespace.create_namespace(create_ns_req)

        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)

        # Create three tables
        for i in range(1, 4):
            create_req = CreateTableRequest(id=["test_ns", f"table{i}"])
            memory_namespace.create_table(create_req, ipc_data)

        # List tables
        list_req = ListTablesRequest(id=["test_ns"])
        response = memory_namespace.list_tables(list_req)

        assert len(response.tables) == 3
        # List tables returns table names as strings
        assert "table1" in response.tables
        assert "table2" in response.tables
        assert "table3" in response.tables

    def test_drop_table_in_child_namespace(self, memory_namespace):
        """Test dropping a table in a child namespace."""
        # Create child namespace
        create_ns_req = CreateNamespaceRequest(id=["test_ns"])
        memory_namespace.create_namespace(create_ns_req)

        # Create table
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["test_ns", "table1"])
        memory_namespace.create_table(create_req, ipc_data)

        # Drop table
        drop_req = DropTableRequest(id=["test_ns", "table1"])
        memory_namespace.drop_table(drop_req)

        # Verify table no longer exists
        exists_req = TableExistsRequest(id=["test_ns", "table1"])
        with pytest.raises(Exception):
            memory_namespace.table_exists(exists_req)

    def test_empty_table_in_child_namespace(self, memory_namespace):
        """Test creating an empty table in a child namespace."""
        # Create child namespace
        create_ns_req = CreateNamespaceRequest(id=["test_ns"])
        memory_namespace.create_namespace(create_ns_req)

        # Create empty table
        schema = pa.schema([pa.field("id", pa.int64())])
        create_req = CreateEmptyTableRequest(
            id=["test_ns", "empty_table"], schema=schema
        )
        memory_namespace.create_empty_table(create_req)

        # Verify table exists
        exists_req = TableExistsRequest(id=["test_ns", "empty_table"])
        memory_namespace.table_exists(exists_req)


class TestDeeplyNestedNamespaces:
    """Tests for deeply nested namespace hierarchies.

    Mirrors Rust test_deeply_nested_namespace.
    """

    def test_deeply_nested_namespace(self, memory_namespace):
        """Test creating deeply nested namespace hierarchy."""
        # Create deeply nested namespace hierarchy
        memory_namespace.create_namespace(CreateNamespaceRequest(id=["level1"]))
        memory_namespace.create_namespace(
            CreateNamespaceRequest(id=["level1", "level2"])
        )
        memory_namespace.create_namespace(
            CreateNamespaceRequest(id=["level1", "level2", "level3"])
        )

        # Create table in deeply nested namespace
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["level1", "level2", "level3", "table1"])
        memory_namespace.create_table(create_req, ipc_data)

        # Verify table exists
        exists_req = TableExistsRequest(id=["level1", "level2", "level3", "table1"])
        memory_namespace.table_exists(exists_req)


class TestNamespaceProperties:
    """Tests for namespace properties - mirrors Rust test_namespace_with_properties."""

    def test_namespace_with_properties(self, memory_namespace):
        """Test creating a namespace with properties."""
        # Create namespace with properties
        properties = {
            "owner": "test_user",
            "description": "Test namespace",
        }

        create_req = CreateNamespaceRequest(id=["test_ns"], properties=properties)
        memory_namespace.create_namespace(create_req)

        # Describe namespace and verify properties
        describe_req = DescribeNamespaceRequest(id=["test_ns"])
        response = memory_namespace.describe_namespace(describe_req)

        assert response.properties is not None
        assert response.properties.get("owner") == "test_user"
        assert response.properties.get("description") == "Test namespace"


class TestNamespaceConstraints:
    """Tests for namespace constraints and isolation."""

    def test_cannot_drop_namespace_with_tables(self, memory_namespace):
        """Test that dropping a namespace with tables should fail."""
        # Create namespace
        create_ns_req = CreateNamespaceRequest(id=["test_ns"])
        memory_namespace.create_namespace(create_ns_req)

        # Create table in namespace
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["test_ns", "table1"])
        memory_namespace.create_table(create_req, ipc_data)

        # Try to drop namespace - should fail
        drop_req = DropNamespaceRequest(id=["test_ns"])
        with pytest.raises(Exception) as exc_info:
            memory_namespace.drop_namespace(drop_req)

        # Should contain an error message about non-empty namespace
        assert (
            "not empty" in str(exc_info.value).lower()
            or "contains" in str(exc_info.value).lower()
        )

    def test_isolation_between_namespaces(self, memory_namespace):
        """Test that namespaces are isolated from each other."""
        # Create two namespaces
        memory_namespace.create_namespace(CreateNamespaceRequest(id=["ns1"]))
        memory_namespace.create_namespace(CreateNamespaceRequest(id=["ns2"]))

        # Create table with same name in both namespaces
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)

        create_req1 = CreateTableRequest(id=["ns1", "table1"])
        memory_namespace.create_table(create_req1, ipc_data)

        create_req2 = CreateTableRequest(id=["ns2", "table1"])
        memory_namespace.create_table(create_req2, ipc_data)

        # List tables in each namespace
        list_req = ListTablesRequest(id=["ns1"])
        response = memory_namespace.list_tables(list_req)
        assert len(response.tables) == 1
        assert "table1" in response.tables

        list_req = ListTablesRequest(id=["ns2"])
        response = memory_namespace.list_tables(list_req)
        assert len(response.tables) == 1
        assert "table1" in response.tables

        # Drop table in ns1 shouldn't affect ns2
        drop_req = DropTableRequest(id=["ns1", "table1"])
        memory_namespace.drop_table(drop_req)

        # ns1 should have no tables
        list_req = ListTablesRequest(id=["ns1"])
        response = memory_namespace.list_tables(list_req)
        assert len(response.tables) == 0

        # ns2 should still have the table
        list_req = ListTablesRequest(id=["ns2"])
        response = memory_namespace.list_tables(list_req)
        assert len(response.tables) == 1


class TestBasicNamespaceOperations:
    """Tests for basic namespace CRUD operations."""

    def test_create_and_describe_namespace(self, memory_namespace):
        """Test creating and describing a namespace."""
        # Create namespace
        create_req = CreateNamespaceRequest(id=["workspace"])
        memory_namespace.create_namespace(create_req)

        # Describe it
        describe_req = DescribeNamespaceRequest(id=["workspace"])
        response = memory_namespace.describe_namespace(describe_req)
        assert response is not None

    def test_namespace_exists(self, memory_namespace):
        """Test checking if a namespace exists."""
        # Create namespace
        create_req = CreateNamespaceRequest(id=["workspace"])
        memory_namespace.create_namespace(create_req)

        # Check it exists (should not raise)
        exists_req = NamespaceExistsRequest(id=["workspace"])
        memory_namespace.namespace_exists(exists_req)

    def test_drop_empty_namespace(self, memory_namespace):
        """Test dropping an empty namespace."""
        # Create namespace
        create_req = CreateNamespaceRequest(id=["workspace"])
        memory_namespace.create_namespace(create_req)

        # Drop it
        drop_req = DropNamespaceRequest(id=["workspace"])
        response = memory_namespace.drop_namespace(drop_req)
        assert response is not None

    def test_list_namespaces(self, memory_namespace):
        """Test listing namespaces."""
        # Create some child namespaces under a parent
        memory_namespace.create_namespace(CreateNamespaceRequest(id=["parent"]))
        memory_namespace.create_namespace(
            CreateNamespaceRequest(id=["parent", "child1"])
        )
        memory_namespace.create_namespace(
            CreateNamespaceRequest(id=["parent", "child2"])
        )

        # List namespaces under parent
        list_req = ListNamespacesRequest(id=["parent"])
        response = memory_namespace.list_namespaces(list_req)

        assert response is not None
        # Should find the child namespaces
        assert len(response.namespaces) >= 2


class TestLanceNamespaceConnect:
    """Tests for lance.namespace.connect integration."""

    def test_connect_with_properties(self):
        """Test creating DirectoryNamespace via lance.namespace.connect()."""
        import uuid

        unique_id = uuid.uuid4().hex[:8]
        properties = {
            "root": f"memory://test_connect_{unique_id}",
            "manifest_enabled": "true",
            "dir_listing_enabled": "true",
        }

        # Connect via lance.namespace.connect
        # should use lance.namespace.DirectoryNamespace
        ns = connect("dir", properties)

        # Verify it's a DirectoryNamespace instance
        assert isinstance(ns, lance.namespace.DirectoryNamespace)

        # Verify it works
        create_req = CreateTableRequest(id=["test_table"])
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        response = ns.create_table(create_req, ipc_data)
        assert response is not None

        # Verify we can list the table
        list_req = ListTablesRequest(id=[])
        list_response = ns.list_tables(list_req)
        assert len(list_response.tables) == 1
        # tables is a list of strings
        assert list_response.tables[0] == "test_table"

    def test_connect_with_storage_options(self):
        """Test creating DirectoryNamespace with storage options via connect()."""
        import uuid

        unique_id = uuid.uuid4().hex[:8]
        properties = {
            "root": f"memory://test_storage_{unique_id}",
            "storage.some_option": "value",  # Test storage.* prefix
        }

        # This should work without errors
        ns = connect("dir", properties)
        assert isinstance(ns, lance.namespace.DirectoryNamespace)


class TableVersionTrackingNamespace(lance.namespace.DirectoryNamespace):
    """Namespace wrapper that tracks table version API calls.

    Similar to the Rust TrackingNamespace and Java TableVersionTrackingNamespace,
    this extends DirectoryNamespace with table_version_tracking_enabled=true and
    counts create_table_version and describe_table_version calls.

    This class implements the JSON bridge methods that PyLanceNamespace calls,
    allowing API call tracking to work even when the calls go through Rust.

    Unlike a wrapper approach, this extends DirectoryNamespace directly so that
    Rust can detect it as a DirectoryNamespace subclass and use the native handle.
    """

    def __init__(self, root: str):
        dir_props = {
            "root": root,
            "table_version_tracking_enabled": "true",
            "manifest_enabled": "true",
        }
        super().__init__(**dir_props)
        self.create_table_version_count = 0
        self.describe_table_version_count = 0
        self.list_table_versions_count = 0
        self._lock = Lock()

    def namespace_id(self) -> str:
        return f"TableVersionTrackingNamespace {{ inner: {super().namespace_id()} }}"

    def create_table_version(
        self, request: CreateTableVersionRequest
    ) -> CreateTableVersionResponse:
        with self._lock:
            self.create_table_version_count += 1
        return super().create_table_version(request)

    def describe_table_version(
        self, request: DescribeTableVersionRequest
    ) -> DescribeTableVersionResponse:
        with self._lock:
            self.describe_table_version_count += 1
        return super().describe_table_version(request)

    def list_table_versions(
        self, request: ListTableVersionsRequest
    ) -> ListTableVersionsResponse:
        with self._lock:
            self.list_table_versions_count += 1
        return super().list_table_versions(request)

    # JSON bridge methods for Rust PyLanceNamespace callbacks
    # These call the parent's _inner (PyDirectoryNamespace) directly with dict API
    def describe_table_version_json(self, request_json: str) -> str:
        """JSON bridge that increments counter before delegating."""
        import json

        with self._lock:
            self.describe_table_version_count += 1
        request_dict = json.loads(request_json)
        response_dict = self._inner.describe_table_version(request_dict)
        return json.dumps(response_dict)

    def create_table_version_json(self, request_json: str) -> str:
        """JSON bridge that increments counter before delegating."""
        import json

        with self._lock:
            self.create_table_version_count += 1
        request_dict = json.loads(request_json)
        response_dict = self._inner.create_table_version(request_dict)
        return json.dumps(response_dict)

    def list_table_versions_json(self, request_json: str) -> str:
        """JSON bridge that increments counter before delegating."""
        import json

        with self._lock:
            self.list_table_versions_count += 1
        request_dict = json.loads(request_json)
        response_dict = self._inner.list_table_versions(request_dict)
        return json.dumps(response_dict)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="External manifest store has known issues on Windows",
)
def test_external_manifest_store_invokes_namespace_apis():
    """Test that namespace APIs are invoked correctly for managed versioning.

    This test mirrors:
    - Rust: test_external_manifest_store_invokes_namespace_apis
    - Java: testExternalManifestStoreInvokesNamespaceApis

    It verifies:
    1. list_table_versions is called when opening dataset (latest version)
    2. create_table_version is called exactly once during append
    3. describe_table_version is called when opening specific version
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        namespace = TableVersionTrackingNamespace(root=tmpdir)

        # Create parent namespace first (like Rust/Java tests)
        namespace.create_namespace(CreateNamespaceRequest(id=["workspace"]))

        table_id = ["workspace", "test_table"]

        # Create initial table
        table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
        ds = lance.write_dataset(
            table1, namespace=namespace, table_id=table_id, mode="create"
        )
        assert ds.count_rows() == 2
        assert len(ds.versions()) == 1

        # Verify describe_table returns managed_versioning=True
        describe_resp = namespace.describe_table(DescribeTableRequest(id=table_id))
        assert describe_resp.managed_versioning is True, (
            f"Expected managed_versioning=True, got {describe_resp.managed_versioning}"
        )

        # Open dataset through namespace - should call list_table_versions for latest
        initial_list_count = namespace.list_table_versions_count
        ds_from_namespace = lance.dataset(namespace=namespace, table_id=table_id)
        assert ds_from_namespace.count_rows() == 2
        assert ds_from_namespace.version == 1
        assert namespace.list_table_versions_count == initial_list_count + 1, (
            "list_table_versions should be called once when opening latest version"
        )

        # Verify create_table_version was called once during CREATE
        assert namespace.create_table_version_count == 1, (
            "create_table_version should have been called once during CREATE"
        )

        # Append data - should call create_table_version again
        table2 = pa.Table.from_pylist([{"a": 100, "b": 200}, {"a": 1000, "b": 2000}])
        ds = lance.write_dataset(
            table2, namespace=namespace, table_id=table_id, mode="append"
        )
        assert ds.count_rows() == 4
        assert len(ds.versions()) == 2

        assert namespace.create_table_version_count == 2, (
            "create_table_version should be called twice (CREATE + APPEND)"
        )

        # Open latest version - should call list_table_versions
        list_count_before_latest = namespace.list_table_versions_count
        latest_ds = lance.dataset(namespace=namespace, table_id=table_id)
        assert latest_ds.count_rows() == 4
        assert latest_ds.version == 2
        assert namespace.list_table_versions_count == list_count_before_latest + 1, (
            "list_table_versions should be called once when opening latest version"
        )

        # Open specific version (v1) - should call describe_table_version
        describe_count_before_v1 = namespace.describe_table_version_count
        v1_ds = lance.dataset(namespace=namespace, table_id=table_id, version=1)
        assert v1_ds.count_rows() == 2
        assert v1_ds.version == 1
        assert namespace.describe_table_version_count == describe_count_before_v1 + 1, (
            "describe_table_version should be called once when opening version 1"
        )


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows file locking prevents reliable concurrent filesystem operations",
)
class TestConcurrentOperations:
    """Tests for concurrent table operations.

    These tests mirror the Rust and Java concurrent tests to ensure
    the DirectoryNamespace handles concurrent create/drop operations correctly.
    """

    def test_concurrent_create_and_drop_single_instance(self, temp_namespace):
        """Test concurrent create/drop with single namespace instance."""
        import concurrent.futures

        # Initialize namespace first - create parent namespace to ensure __manifest
        # table is created before concurrent operations
        create_ns_req = CreateNamespaceRequest(id=["test_ns"])
        temp_namespace.create_namespace(create_ns_req)

        num_tables = 10
        success_count = 0
        fail_count = 0
        lock = Lock()

        def create_and_drop_table(table_index):
            nonlocal success_count, fail_count
            try:
                table_name = f"concurrent_table_{table_index}"
                table_id = ["test_ns", table_name]
                table_data = create_test_data()
                ipc_data = table_to_ipc_bytes(table_data)

                # Create table
                create_req = CreateTableRequest(id=table_id)
                temp_namespace.create_table(create_req, ipc_data)

                # Drop table
                drop_req = DropTableRequest(id=table_id)
                temp_namespace.drop_table(drop_req)

                with lock:
                    success_count += 1
            except Exception as e:
                with lock:
                    fail_count += 1
                raise e

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_tables) as executor:
            futures = [
                executor.submit(create_and_drop_table, i) for i in range(num_tables)
            ]
            concurrent.futures.wait(futures)

        assert success_count == num_tables, (
            f"Expected {num_tables} successes, got {success_count}"
        )
        assert fail_count == 0, f"Expected 0 failures, got {fail_count}"

        # Verify all tables are dropped
        list_req = ListTablesRequest(id=["test_ns"])
        response = temp_namespace.list_tables(list_req)
        assert len(response.tables) == 0, "All tables should be dropped"

    def test_concurrent_create_and_drop_multiple_instances(self):
        """Test concurrent create/drop with multiple namespace instances."""
        import concurrent.futures

        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize namespace first with a single instance to ensure __manifest
            # table is created and parent namespace exists before concurrent operations
            init_ns = connect(
                "dir",
                {"root": f"file://{tmpdir}", "commit_retries": "2147483647"},
            )
            create_ns_req = CreateNamespaceRequest(id=["test_ns"])
            init_ns.create_namespace(create_ns_req)

            num_tables = 10
            success_count = 0
            fail_count = 0
            lock = Lock()

            def create_and_drop_table(table_index):
                nonlocal success_count, fail_count
                try:
                    # Each thread creates its own namespace instance
                    # Use high commit_retries to handle version collisions
                    ns = connect(
                        "dir",
                        {"root": f"file://{tmpdir}", "commit_retries": "2147483647"},
                    )

                    table_name = f"multi_ns_table_{table_index}"
                    table_id = ["test_ns", table_name]
                    table_data = create_test_data()
                    ipc_data = table_to_ipc_bytes(table_data)

                    # Create table
                    create_req = CreateTableRequest(id=table_id)
                    ns.create_table(create_req, ipc_data)

                    # Drop table
                    drop_req = DropTableRequest(id=table_id)
                    ns.drop_table(drop_req)

                    with lock:
                        success_count += 1
                except Exception as e:
                    with lock:
                        fail_count += 1
                    raise e

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_tables
            ) as executor:
                futures = [
                    executor.submit(create_and_drop_table, i) for i in range(num_tables)
                ]
                concurrent.futures.wait(futures)

            assert success_count == num_tables, (
                f"Expected {num_tables} successes, got {success_count}"
            )
            assert fail_count == 0, f"Expected 0 failures, got {fail_count}"

            # Verify with a fresh namespace instance
            verify_ns = connect(
                "dir", {"root": f"file://{tmpdir}", "commit_retries": "2147483647"}
            )
            list_req = ListTablesRequest(id=["test_ns"])
            response = verify_ns.list_tables(list_req)
            assert len(response.tables) == 0, "All tables should be dropped"

    def test_concurrent_create_then_drop_from_different_instance(self):
        """Test creating from one set of instances, dropping from different ones."""
        import concurrent.futures

        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize namespace first with a single instance to ensure __manifest
            # table is created and parent namespace exists before concurrent operations
            init_ns = connect(
                "dir",
                {"root": f"file://{tmpdir}", "commit_retries": "2147483647"},
            )
            create_ns_req = CreateNamespaceRequest(id=["test_ns"])
            init_ns.create_namespace(create_ns_req)

            num_tables = 10

            # Phase 1: Create all tables concurrently using separate namespace instances
            create_success_count = 0
            create_fail_count = 0
            create_lock = Lock()

            def create_table(table_index):
                nonlocal create_success_count, create_fail_count
                try:
                    # Use high commit_retries to handle version collisions
                    ns = connect(
                        "dir",
                        {"root": f"file://{tmpdir}", "commit_retries": "2147483647"},
                    )

                    table_name = f"cross_instance_table_{table_index}"
                    table_id = ["test_ns", table_name]
                    table_data = create_test_data()
                    ipc_data = table_to_ipc_bytes(table_data)

                    create_req = CreateTableRequest(id=table_id)
                    ns.create_table(create_req, ipc_data)

                    with create_lock:
                        create_success_count += 1
                except Exception as e:
                    with create_lock:
                        create_fail_count += 1
                    raise e

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_tables
            ) as executor:
                futures = [executor.submit(create_table, i) for i in range(num_tables)]
                concurrent.futures.wait(futures)

            assert create_success_count == num_tables, (
                f"All creates should succeed, got {create_success_count}"
            )

            # Phase 2: Drop all tables concurrently using NEW namespace instances
            drop_success_count = 0
            drop_fail_count = 0
            drop_lock = Lock()

            def drop_table(table_index):
                nonlocal drop_success_count, drop_fail_count
                try:
                    # Use high commit_retries to handle version collisions
                    ns = connect(
                        "dir",
                        {"root": f"file://{tmpdir}", "commit_retries": "2147483647"},
                    )

                    table_name = f"cross_instance_table_{table_index}"
                    table_id = ["test_ns", table_name]

                    drop_req = DropTableRequest(id=table_id)
                    ns.drop_table(drop_req)

                    with drop_lock:
                        drop_success_count += 1
                except Exception as e:
                    with drop_lock:
                        drop_fail_count += 1
                    raise e

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_tables
            ) as executor:
                futures = [executor.submit(drop_table, i) for i in range(num_tables)]
                concurrent.futures.wait(futures)

            assert drop_success_count == num_tables, (
                f"All drops should succeed, got {drop_success_count}"
            )
            assert drop_fail_count == 0, f"No drops should fail, got {drop_fail_count}"

            # Verify all tables are dropped
            verify_ns = connect(
                "dir", {"root": f"file://{tmpdir}", "commit_retries": "2147483647"}
            )
            list_req = ListTablesRequest(id=["test_ns"])
            response = verify_ns.list_tables(list_req)
            assert len(response.tables) == 0, "All tables should be dropped"
