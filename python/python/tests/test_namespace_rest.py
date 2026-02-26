# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Tests for RestNamespace with RestAdapter server.

This module tests the RestNamespace class which provides a REST-based
namespace implementation for organizing Lance tables and nested namespaces.

These tests mirror test_namespace_dir.py to ensure parity between
DirectoryNamespace and RestNamespace implementations.
"""

import tempfile

import lance.namespace
import pyarrow as pa
import pytest
from lance_namespace import (
    CreateEmptyTableRequest,
    CreateNamespaceRequest,
    CreateTableRequest,
    DeregisterTableRequest,
    DescribeNamespaceRequest,
    DescribeTableRequest,
    DropNamespaceRequest,
    DropTableRequest,
    ListNamespacesRequest,
    ListTablesRequest,
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
def rest_namespace():
    """Create a REST namespace with adapter for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend_config = {"root": tmpdir}

        with lance.namespace.RestAdapter("dir", backend_config, port=0) as adapter:
            client = connect("rest", {"uri": f"http://127.0.0.1:{adapter.port}"})
            yield client


class TestCreateTable:
    """Tests for create_table operation - mirrors DirectoryNamespace tests."""

    def test_create_table(self, rest_namespace):
        """Test creating a table with data."""
        # Create parent namespace first
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        rest_namespace.create_namespace(create_ns_req)

        # Create table with data
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)

        create_req = CreateTableRequest(id=["workspace", "test_table"])
        response = rest_namespace.create_table(create_req, ipc_data)

        assert response is not None
        assert response.location is not None
        assert "test_table" in response.location
        assert response.version == 1

    def test_create_table_without_data(self, rest_namespace):
        """Test creating a table without data should fail."""
        # Create parent namespace first
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        rest_namespace.create_namespace(create_ns_req)

        create_req = CreateTableRequest(id=["workspace", "test_table"])

        with pytest.raises(Exception) as exc_info:
            rest_namespace.create_table(create_req, b"")

        assert "Arrow IPC" in str(exc_info.value) or "required" in str(exc_info.value)

    def test_create_table_with_invalid_id(self, rest_namespace):
        """Test creating a table with invalid ID should fail."""
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)

        # Test with empty ID
        create_req = CreateTableRequest(id=[])
        with pytest.raises(Exception):
            rest_namespace.create_table(create_req, ipc_data)

    def test_create_table_in_child_namespace(self, rest_namespace):
        """Test creating table in child namespace."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["test_namespace"])
        rest_namespace.create_namespace(create_ns_req)

        # Create table in the namespace
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["test_namespace", "table"])
        response = rest_namespace.create_table(create_req, ipc_data)

        assert response is not None
        assert response.location is not None


class TestListTables:
    """Tests for list_tables operation - mirrors DirectoryNamespace tests."""

    def test_list_tables_empty(self, rest_namespace):
        """Test listing tables in empty namespace."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        rest_namespace.create_namespace(create_ns_req)

        # Initially, no tables
        list_req = ListTablesRequest(id=["workspace"])
        response = rest_namespace.list_tables(list_req)
        assert len(response.tables) == 0

    def test_list_tables_with_tables(self, rest_namespace):
        """Test listing tables after creating them."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        rest_namespace.create_namespace(create_ns_req)

        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)

        # Create table1
        create_req = CreateTableRequest(id=["workspace", "table1"])
        rest_namespace.create_table(create_req, ipc_data)

        # Create table2
        create_req = CreateTableRequest(id=["workspace", "table2"])
        rest_namespace.create_table(create_req, ipc_data)

        # List tables should return both
        list_req = ListTablesRequest(id=["workspace"])
        response = rest_namespace.list_tables(list_req)
        assert len(response.tables) == 2
        assert "table1" in response.tables
        assert "table2" in response.tables

    def test_list_tables_with_namespace_id(self, rest_namespace):
        """Test listing tables in a child namespace."""
        # Create child namespace
        create_ns_req = CreateNamespaceRequest(id=["test_namespace"])
        rest_namespace.create_namespace(create_ns_req)

        # List tables in the child namespace
        list_req = ListTablesRequest(id=["test_namespace"])
        response = rest_namespace.list_tables(list_req)

        # Should succeed and return empty list (no tables yet)
        assert len(response.tables) == 0


class TestDescribeTable:
    """Tests for describe_table operation - mirrors DirectoryNamespace tests."""

    def test_describe_table(self, rest_namespace):
        """Test describing a table."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        rest_namespace.create_namespace(create_ns_req)

        # Create a table
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["workspace", "test_table"])
        rest_namespace.create_table(create_req, ipc_data)

        # Describe the table
        describe_req = DescribeTableRequest(id=["workspace", "test_table"])
        response = rest_namespace.describe_table(describe_req)

        assert response is not None
        assert response.location is not None
        assert "test_table" in response.location

    def test_describe_nonexistent_table(self, rest_namespace):
        """Test describing a table that doesn't exist."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        rest_namespace.create_namespace(create_ns_req)

        describe_req = DescribeTableRequest(id=["workspace", "nonexistent"])

        with pytest.raises(Exception) as exc_info:
            rest_namespace.describe_table(describe_req)

        error_msg = str(exc_info.value).lower()
        assert "not found" in error_msg or "does not exist" in error_msg


class TestTableOperations:
    """Tests for various table operations."""

    def test_table_exists(self, rest_namespace):
        """Test checking if a table exists."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        rest_namespace.create_namespace(create_ns_req)

        # Create a table
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["workspace", "test_table"])
        rest_namespace.create_table(create_req, ipc_data)

        # Check it exists (should not raise)
        exists_req = TableExistsRequest(id=["workspace", "test_table"])
        rest_namespace.table_exists(exists_req)

    def test_table_not_exists(self, rest_namespace):
        """Test checking if a non-existent table exists."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        rest_namespace.create_namespace(create_ns_req)

        exists_req = TableExistsRequest(id=["workspace", "nonexistent"])

        with pytest.raises(Exception):
            rest_namespace.table_exists(exists_req)

    def test_drop_table(self, rest_namespace):
        """Test dropping a table."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        rest_namespace.create_namespace(create_ns_req)

        # Create table
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["workspace", "test_table"])
        rest_namespace.create_table(create_req, ipc_data)

        # Drop the table
        drop_req = DropTableRequest(id=["workspace", "test_table"])
        response = rest_namespace.drop_table(drop_req)
        assert response is not None

        # Verify table no longer exists
        exists_req = TableExistsRequest(id=["workspace", "test_table"])
        with pytest.raises(Exception):
            rest_namespace.table_exists(exists_req)

    def test_create_empty_table(self, rest_namespace):
        """Test creating an empty table."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        rest_namespace.create_namespace(create_ns_req)

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
        response = rest_namespace.create_empty_table(create_req)

        assert response is not None
        assert response.location is not None

        # Verify table exists
        exists_req = TableExistsRequest(id=["workspace", "empty_table"])
        rest_namespace.table_exists(exists_req)

    def test_deregister_table(self, rest_namespace):
        """Test deregistering a table."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        rest_namespace.create_namespace(create_ns_req)

        # Create table
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["workspace", "test_table"])
        rest_namespace.create_table(create_req, ipc_data)

        # Verify table exists
        exists_req = TableExistsRequest(id=["workspace", "test_table"])
        rest_namespace.table_exists(exists_req)

        # Deregister it
        deregister_req = DeregisterTableRequest(id=["workspace", "test_table"])
        response = rest_namespace.deregister_table(deregister_req)
        assert response is not None
        assert response.location is not None
        assert response.id == ["workspace", "test_table"]

        # Verify table no longer exists in namespace
        with pytest.raises(Exception):
            rest_namespace.table_exists(exists_req)

    def test_register_table(self, rest_namespace):
        """Test registering a table."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        rest_namespace.create_namespace(create_ns_req)

        # Create physical table first
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["workspace", "physical_table"])
        rest_namespace.create_table(create_req, ipc_data)

        # Deregister it to get the physical location
        deregister_req = DeregisterTableRequest(id=["workspace", "physical_table"])
        deregister_response = rest_namespace.deregister_table(deregister_req)
        physical_location = deregister_response.location

        # Extract relative path from location (remove any URL prefix if present)
        # Location format is typically like "workspace$physical_table" or similar
        if "/" in physical_location:
            relative_location = physical_location.split("/")[-1]
        else:
            relative_location = physical_location

        # Register with a different name using relative path
        register_req = RegisterTableRequest(
            id=["workspace", "registered_table"],
            location=relative_location,
        )
        response = rest_namespace.register_table(register_req)
        assert response is not None
        assert response.location == relative_location

        # Verify table exists
        exists_req = TableExistsRequest(id=["workspace", "registered_table"])
        rest_namespace.table_exists(exists_req)

        # Verify we can describe it
        describe_req = DescribeTableRequest(id=["workspace", "registered_table"])
        desc_response = rest_namespace.describe_table(describe_req)
        assert desc_response is not None

    def test_register_table_rejects_absolute_uri(self, rest_namespace):
        """Test that register_table rejects absolute URIs."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        rest_namespace.create_namespace(create_ns_req)

        # Try to register with absolute URI - should fail
        register_req = RegisterTableRequest(
            id=["workspace", "test_table"], location="s3://bucket/table.lance"
        )
        with pytest.raises(Exception) as exc_info:
            rest_namespace.register_table(register_req)
        assert "Absolute URIs are not allowed" in str(exc_info.value)

    def test_register_table_rejects_absolute_path(self, rest_namespace):
        """Test that register_table rejects absolute paths."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        rest_namespace.create_namespace(create_ns_req)

        # Try to register with absolute path - should fail
        register_req = RegisterTableRequest(
            id=["workspace", "test_table"], location="/tmp/table.lance"
        )
        with pytest.raises(Exception) as exc_info:
            rest_namespace.register_table(register_req)
        assert "Absolute paths are not allowed" in str(exc_info.value)

    def test_register_table_rejects_path_traversal(self, rest_namespace):
        """Test that register_table rejects path traversal attempts."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        rest_namespace.create_namespace(create_ns_req)

        # Try to register with path traversal - should fail
        register_req = RegisterTableRequest(
            id=["workspace", "test_table"], location="../outside/table.lance"
        )
        with pytest.raises(Exception) as exc_info:
            rest_namespace.register_table(register_req)
        assert "Path traversal is not allowed" in str(exc_info.value)

    def test_rename_table(self, rest_namespace):
        """Test renaming a table."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        rest_namespace.create_namespace(create_ns_req)

        # Create table
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["workspace", "test_table"])
        rest_namespace.create_table(create_req, ipc_data)

        # TODO: underlying dir namespace doesn't support rename yet...

        # # Rename the table
        # rename_req = RenameTableRequest(
        #     id=["workspace", "test_table"],
        #     new_namespace_id=["workspace"],
        #     new_table_name="test_table_renamed",
        # )

        # response = rest_namespace.rename_table(rename_req)
        # assert response is not None

        # # Verify table with old name no longer exists
        # exists_req = TableExistsRequest(id=["workspace", "test_table"])
        # with pytest.raises(Exception):
        #     rest_namespace.table_exists(exists_req)

        # # Verify table with new name exists
        # exists_req = TableExistsRequest(id=["workspace", "test_table_renamed"])
        # rest_namespace.table_exists(exists_req)


class TestChildNamespaceOperations:
    """Tests for operations in child namespaces - mirrors DirectoryNamespace tests."""

    def test_create_table_in_child_namespace(self, rest_namespace):
        """Test creating multiple tables in a child namespace."""
        # Create child namespace
        create_ns_req = CreateNamespaceRequest(id=["test_ns"])
        rest_namespace.create_namespace(create_ns_req)

        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)

        # Create three tables
        for i in range(1, 4):
            create_req = CreateTableRequest(id=["test_ns", f"table{i}"])
            rest_namespace.create_table(create_req, ipc_data)

        # List tables
        list_req = ListTablesRequest(id=["test_ns"])
        response = rest_namespace.list_tables(list_req)

        assert len(response.tables) == 3
        assert "table1" in response.tables
        assert "table2" in response.tables
        assert "table3" in response.tables

    def test_drop_table_in_child_namespace(self, rest_namespace):
        """Test dropping a table in a child namespace."""
        # Create child namespace
        create_ns_req = CreateNamespaceRequest(id=["test_ns"])
        rest_namespace.create_namespace(create_ns_req)

        # Create table
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["test_ns", "table1"])
        rest_namespace.create_table(create_req, ipc_data)

        # Drop table
        drop_req = DropTableRequest(id=["test_ns", "table1"])
        rest_namespace.drop_table(drop_req)

        # Verify table no longer exists
        exists_req = TableExistsRequest(id=["test_ns", "table1"])
        with pytest.raises(Exception):
            rest_namespace.table_exists(exists_req)

    def test_empty_table_in_child_namespace(self, rest_namespace):
        """Test creating an empty table in a child namespace."""
        # Create child namespace
        create_ns_req = CreateNamespaceRequest(id=["test_ns"])
        rest_namespace.create_namespace(create_ns_req)

        # Create empty table
        schema = pa.schema([pa.field("id", pa.int64())])
        create_req = CreateEmptyTableRequest(
            id=["test_ns", "empty_table"], schema=schema
        )
        rest_namespace.create_empty_table(create_req)

        # Verify table exists
        exists_req = TableExistsRequest(id=["test_ns", "empty_table"])
        rest_namespace.table_exists(exists_req)


class TestDeeplyNestedNamespaces:
    """Tests for deeply nested namespace hierarchies."""

    def test_deeply_nested_namespace(self, rest_namespace):
        """Test creating deeply nested namespace hierarchy."""
        # Create deeply nested namespace hierarchy
        rest_namespace.create_namespace(CreateNamespaceRequest(id=["level1"]))
        rest_namespace.create_namespace(CreateNamespaceRequest(id=["level1", "level2"]))
        rest_namespace.create_namespace(
            CreateNamespaceRequest(id=["level1", "level2", "level3"])
        )

        # Create table in deeply nested namespace
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["level1", "level2", "level3", "table1"])
        rest_namespace.create_table(create_req, ipc_data)

        # Verify table exists
        exists_req = TableExistsRequest(id=["level1", "level2", "level3", "table1"])
        rest_namespace.table_exists(exists_req)


class TestNamespaceProperties:
    """Tests for namespace properties."""

    def test_namespace_with_properties(self, rest_namespace):
        """Test creating a namespace with properties."""
        # Create namespace with properties
        properties = {
            "owner": "test_user",
            "description": "Test namespace",
        }

        create_req = CreateNamespaceRequest(id=["test_ns"], properties=properties)
        rest_namespace.create_namespace(create_req)

        # Describe namespace and verify properties
        describe_req = DescribeNamespaceRequest(id=["test_ns"])
        response = rest_namespace.describe_namespace(describe_req)

        assert response.properties is not None
        assert response.properties.get("owner") == "test_user"
        assert response.properties.get("description") == "Test namespace"


class TestNamespaceConstraints:
    """Tests for namespace constraints and isolation."""

    def test_cannot_drop_namespace_with_tables(self, rest_namespace):
        """Test that dropping a namespace with tables should fail."""
        # Create namespace
        create_ns_req = CreateNamespaceRequest(id=["test_ns"])
        rest_namespace.create_namespace(create_ns_req)

        # Create table in namespace
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["test_ns", "table1"])
        rest_namespace.create_table(create_req, ipc_data)

        # Try to drop namespace - should fail
        drop_req = DropNamespaceRequest(id=["test_ns"])
        with pytest.raises(Exception) as exc_info:
            rest_namespace.drop_namespace(drop_req)

        # Should contain an error message about non-empty namespace
        assert (
            "not empty" in str(exc_info.value).lower()
            or "contains" in str(exc_info.value).lower()
        )

    def test_isolation_between_namespaces(self, rest_namespace):
        """Test that namespaces are isolated from each other."""
        # Create two namespaces
        rest_namespace.create_namespace(CreateNamespaceRequest(id=["ns1"]))
        rest_namespace.create_namespace(CreateNamespaceRequest(id=["ns2"]))

        # Create table with same name in both namespaces
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)

        create_req1 = CreateTableRequest(id=["ns1", "table1"])
        rest_namespace.create_table(create_req1, ipc_data)

        create_req2 = CreateTableRequest(id=["ns2", "table1"])
        rest_namespace.create_table(create_req2, ipc_data)

        # List tables in each namespace
        list_req = ListTablesRequest(id=["ns1"])
        response = rest_namespace.list_tables(list_req)
        assert len(response.tables) == 1
        assert "table1" in response.tables

        list_req = ListTablesRequest(id=["ns2"])
        response = rest_namespace.list_tables(list_req)
        assert len(response.tables) == 1
        assert "table1" in response.tables

        # Drop table in ns1 shouldn't affect ns2
        drop_req = DropTableRequest(id=["ns1", "table1"])
        rest_namespace.drop_table(drop_req)

        # ns1 should have no tables
        list_req = ListTablesRequest(id=["ns1"])
        response = rest_namespace.list_tables(list_req)
        assert len(response.tables) == 0

        # ns2 should still have the table
        list_req = ListTablesRequest(id=["ns2"])
        response = rest_namespace.list_tables(list_req)
        assert len(response.tables) == 1


class TestBasicNamespaceOperations:
    """Tests for basic namespace CRUD operations."""

    def test_create_and_describe_namespace(self, rest_namespace):
        """Test creating and describing a namespace."""
        # Create namespace
        create_req = CreateNamespaceRequest(id=["workspace"])
        rest_namespace.create_namespace(create_req)

        # Describe it
        describe_req = DescribeNamespaceRequest(id=["workspace"])
        response = rest_namespace.describe_namespace(describe_req)
        assert response is not None

    def test_namespace_exists(self, rest_namespace):
        """Test checking if a namespace exists."""
        # Create namespace
        create_req = CreateNamespaceRequest(id=["workspace"])
        rest_namespace.create_namespace(create_req)

        # Check it exists (should not raise)
        exists_req = NamespaceExistsRequest(id=["workspace"])
        rest_namespace.namespace_exists(exists_req)

    def test_drop_empty_namespace(self, rest_namespace):
        """Test dropping an empty namespace."""
        # Create namespace
        create_req = CreateNamespaceRequest(id=["workspace"])
        rest_namespace.create_namespace(create_req)

        # Drop it
        drop_req = DropNamespaceRequest(id=["workspace"])
        response = rest_namespace.drop_namespace(drop_req)
        assert response is not None

    def test_list_namespaces(self, rest_namespace):
        """Test listing namespaces."""
        # Create some child namespaces under a parent
        rest_namespace.create_namespace(CreateNamespaceRequest(id=["parent"]))
        rest_namespace.create_namespace(CreateNamespaceRequest(id=["parent", "child1"]))
        rest_namespace.create_namespace(CreateNamespaceRequest(id=["parent", "child2"]))

        # List namespaces under parent
        list_req = ListNamespacesRequest(id=["parent"])
        response = rest_namespace.list_namespaces(list_req)

        assert response is not None
        # Should find the child namespaces
        assert len(response.namespaces) >= 2


class TestLanceNamespaceConnect:
    """Tests for lance.namespace.connect integration."""

    def test_connect_with_rest(self):
        """Test creating RestNamespace via lance.namespace.connect()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend_config = {"root": tmpdir}

            with lance.namespace.RestAdapter("dir", backend_config, port=0) as adapter:
                properties = {"uri": f"http://127.0.0.1:{adapter.port}"}
                ns = connect("rest", properties)

                assert isinstance(ns, lance.namespace.RestNamespace)

                create_req = CreateTableRequest(id=["test_table"])
                table_data = create_test_data()
                ipc_data = table_to_ipc_bytes(table_data)
                response = ns.create_table(create_req, ipc_data)
                assert response is not None

                list_req = ListTablesRequest(id=[])
                list_response = ns.list_tables(list_req)
                assert len(list_response.tables) == 1
                assert list_response.tables[0] == "test_table"

    def test_connect_with_custom_delimiter(self):
        """Test creating RestNamespace with custom delimiter via connect()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend_config = {"root": tmpdir}

            with lance.namespace.RestAdapter("dir", backend_config, port=0) as adapter:
                properties = {
                    "uri": f"http://127.0.0.1:{adapter.port}",
                    "delimiter": "@",
                }
                ns = connect("rest", properties)

                assert isinstance(ns, lance.namespace.RestNamespace)

                create_req = CreateTableRequest(id=["test_table"])
                table_data = create_test_data()
                ipc_data = table_to_ipc_bytes(table_data)
                response = ns.create_table(create_req, ipc_data)
                assert response is not None


class TestDynamicContextProvider:
    """Tests for DynamicContextProvider with RestNamespace."""

    def test_rest_namespace_with_explicit_provider(self):
        """Test RestNamespace with an explicit context provider."""
        call_count = {"count": 0}

        class TestProvider(lance.namespace.DynamicContextProvider):
            def provide_context(self, info):
                call_count["count"] += 1
                return {
                    "headers.Authorization": "Bearer test-token",
                    "headers.X-Request-Id": f"req-{info.get('operation', 'unknown')}",
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            backend_config = {"root": tmpdir}

            with lance.namespace.RestAdapter("dir", backend_config, port=0) as adapter:
                ns = lance.namespace.RestNamespace(
                    uri=f"http://127.0.0.1:{adapter.port}",
                    context_provider=TestProvider(),
                )

                # Perform operations
                create_req = CreateNamespaceRequest(id=["workspace"])
                ns.create_namespace(create_req)

                list_req = ListNamespacesRequest(id=[])
                ns.list_namespaces(list_req)

                # Context provider should have been called
                assert call_count["count"] >= 2

    def test_explicit_provider_takes_precedence(self):
        """Test that explicit provider takes precedence over class path."""
        explicit_called = {"called": False}

        class ExplicitProvider(lance.namespace.DynamicContextProvider):
            def provide_context(self, info):
                explicit_called["called"] = True
                return {"headers.Authorization": "Bearer explicit"}

        with tempfile.TemporaryDirectory() as tmpdir:
            backend_config = {"root": tmpdir}

            with lance.namespace.RestAdapter("dir", backend_config, port=0) as adapter:
                # Pass both explicit provider and class path - explicit should win
                ns = lance.namespace.RestNamespace(
                    context_provider=ExplicitProvider(),
                    **{
                        "uri": f"http://127.0.0.1:{adapter.port}",
                        "dynamic_context_provider.impl": "nonexistent.Provider",
                    },
                )

                create_req = CreateNamespaceRequest(id=["workspace"])
                ns.create_namespace(create_req)

                # Explicit provider should have been used
                assert explicit_called["called"]
