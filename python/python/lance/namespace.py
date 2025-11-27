# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""LanceNamespace storage options integration and implementations.

This module provides:
1. LanceNamespace ABC interface for namespace implementations
2. Native Rust-backed namespace implementations (DirectoryNamespace, RestNamespace)
3. Storage options integration with LanceNamespace for automatic credential refresh
4. Plugin registry for external namespace implementations
"""

import importlib
from abc import ABC, abstractmethod
from typing import Dict, List

from lance_namespace_urllib3_client.models import (
    AlterTransactionRequest,
    AlterTransactionResponse,
    CountTableRowsRequest,
    CreateEmptyTableRequest,
    CreateEmptyTableResponse,
    CreateNamespaceRequest,
    CreateNamespaceResponse,
    CreateTableIndexRequest,
    CreateTableIndexResponse,
    CreateTableRequest,
    CreateTableResponse,
    DeleteFromTableRequest,
    DeleteFromTableResponse,
    DeregisterTableRequest,
    DeregisterTableResponse,
    DescribeNamespaceRequest,
    DescribeNamespaceResponse,
    DescribeTableIndexStatsRequest,
    DescribeTableIndexStatsResponse,
    DescribeTableRequest,
    DescribeTableResponse,
    DescribeTransactionRequest,
    DescribeTransactionResponse,
    DropNamespaceRequest,
    DropNamespaceResponse,
    DropTableRequest,
    DropTableResponse,
    InsertIntoTableRequest,
    InsertIntoTableResponse,
    ListNamespacesRequest,
    ListNamespacesResponse,
    ListTableIndicesRequest,
    ListTableIndicesResponse,
    ListTablesRequest,
    ListTablesResponse,
    MergeInsertIntoTableRequest,
    MergeInsertIntoTableResponse,
    NamespaceExistsRequest,
    QueryTableRequest,
    RegisterTableRequest,
    RegisterTableResponse,
    TableExistsRequest,
    UpdateTableRequest,
    UpdateTableResponse,
)

from .io import StorageOptionsProvider
from .lance import PyDirectoryNamespace  # Low-level Rust binding

try:
    from .lance import PyRestNamespace  # Low-level Rust binding
except ImportError:
    PyRestNamespace = None

try:
    from .lance import PyRestAdapter  # Low-level Rust binding
except ImportError:
    PyRestAdapter = None

__all__ = [
    # Interface and factory
    "LanceNamespace",
    "connect",
    "register_namespace_impl",
    # Implementations
    "DirectoryNamespace",
    "RestNamespace",
    "RestAdapter",
    "LanceNamespaceStorageOptionsProvider",
    # Request/Response types (re-exported from lance_namespace_urllib3_client)
    "AlterTransactionRequest",
    "AlterTransactionResponse",
    "CountTableRowsRequest",
    "CreateEmptyTableRequest",
    "CreateEmptyTableResponse",
    "CreateNamespaceRequest",
    "CreateNamespaceResponse",
    "CreateTableIndexRequest",
    "CreateTableIndexResponse",
    "CreateTableRequest",
    "CreateTableResponse",
    "DeleteFromTableRequest",
    "DeleteFromTableResponse",
    "DeregisterTableRequest",
    "DeregisterTableResponse",
    "DescribeNamespaceRequest",
    "DescribeNamespaceResponse",
    "DescribeTableIndexStatsRequest",
    "DescribeTableIndexStatsResponse",
    "DescribeTableRequest",
    "DescribeTableResponse",
    "DescribeTransactionRequest",
    "DescribeTransactionResponse",
    "DropNamespaceRequest",
    "DropNamespaceResponse",
    "DropTableRequest",
    "DropTableResponse",
    "InsertIntoTableRequest",
    "InsertIntoTableResponse",
    "ListNamespacesRequest",
    "ListNamespacesResponse",
    "ListTableIndicesRequest",
    "ListTableIndicesResponse",
    "ListTablesRequest",
    "ListTablesResponse",
    "MergeInsertIntoTableRequest",
    "MergeInsertIntoTableResponse",
    "NamespaceExistsRequest",
    "QueryTableRequest",
    "RegisterTableRequest",
    "RegisterTableResponse",
    "TableExistsRequest",
    "UpdateTableRequest",
    "UpdateTableResponse",
]


class LanceNamespace(ABC):
    """Base interface for Lance Namespace implementations.

    This abstract base class defines the contract for namespace implementations
    that manage Lance tables. Implementations can provide different storage backends
    (directory-based, REST API, cloud catalogs, etc.).

    To create a custom namespace implementation, subclass this ABC and implement
    at least the `namespace_id()` method. Other methods have default implementations
    that raise `NotImplementedError`.
    """

    @abstractmethod
    def namespace_id(self) -> str:
        """Return a human-readable unique identifier for this namespace instance.

        This is used for equality comparison and hashing when the namespace is
        used as part of a storage options provider. Two namespace instances with
        the same ID are considered equal and will share cached resources.

        The ID should be human-readable for debugging and logging purposes.
        For example:
        - REST namespace: "RestNamespace { uri: 'https://api.example.com' }"
        - Directory namespace: "DirectoryNamespace { root: '/path/to/data' }"

        Returns
        -------
        str
            A human-readable unique identifier string
        """
        pass

    def list_namespaces(self, request: ListNamespacesRequest) -> ListNamespacesResponse:
        """List namespaces."""
        raise NotImplementedError("Not supported: list_namespaces")

    def describe_namespace(
        self, request: DescribeNamespaceRequest
    ) -> DescribeNamespaceResponse:
        """Describe a namespace."""
        raise NotImplementedError("Not supported: describe_namespace")

    def create_namespace(
        self, request: CreateNamespaceRequest
    ) -> CreateNamespaceResponse:
        """Create a new namespace."""
        raise NotImplementedError("Not supported: create_namespace")

    def drop_namespace(self, request: DropNamespaceRequest) -> DropNamespaceResponse:
        """Drop a namespace."""
        raise NotImplementedError("Not supported: drop_namespace")

    def namespace_exists(self, request: NamespaceExistsRequest) -> None:
        """Check if a namespace exists."""
        raise NotImplementedError("Not supported: namespace_exists")

    def list_tables(self, request: ListTablesRequest) -> ListTablesResponse:
        """List tables in a namespace."""
        raise NotImplementedError("Not supported: list_tables")

    def describe_table(self, request: DescribeTableRequest) -> DescribeTableResponse:
        """Describe a table."""
        raise NotImplementedError("Not supported: describe_table")

    def register_table(self, request: RegisterTableRequest) -> RegisterTableResponse:
        """Register a table."""
        raise NotImplementedError("Not supported: register_table")

    def table_exists(self, request: TableExistsRequest) -> None:
        """Check if a table exists."""
        raise NotImplementedError("Not supported: table_exists")

    def drop_table(self, request: DropTableRequest) -> DropTableResponse:
        """Drop a table."""
        raise NotImplementedError("Not supported: drop_table")

    def deregister_table(
        self, request: DeregisterTableRequest
    ) -> DeregisterTableResponse:
        """Deregister a table."""
        raise NotImplementedError("Not supported: deregister_table")

    def count_table_rows(self, request: CountTableRowsRequest) -> int:
        """Count rows in a table."""
        raise NotImplementedError("Not supported: count_table_rows")

    def create_table(
        self, request: CreateTableRequest, request_data: bytes
    ) -> CreateTableResponse:
        """Create a new table with data from Arrow IPC stream."""
        raise NotImplementedError("Not supported: create_table")

    def create_empty_table(
        self, request: CreateEmptyTableRequest
    ) -> CreateEmptyTableResponse:
        """Create an empty table (metadata only operation)."""
        raise NotImplementedError("Not supported: create_empty_table")

    def insert_into_table(
        self, request: InsertIntoTableRequest, request_data: bytes
    ) -> InsertIntoTableResponse:
        """Insert data into a table."""
        raise NotImplementedError("Not supported: insert_into_table")

    def merge_insert_into_table(
        self, request: MergeInsertIntoTableRequest, request_data: bytes
    ) -> MergeInsertIntoTableResponse:
        """Merge insert data into a table."""
        raise NotImplementedError("Not supported: merge_insert_into_table")

    def update_table(self, request: UpdateTableRequest) -> UpdateTableResponse:
        """Update a table."""
        raise NotImplementedError("Not supported: update_table")

    def delete_from_table(
        self, request: DeleteFromTableRequest
    ) -> DeleteFromTableResponse:
        """Delete from a table."""
        raise NotImplementedError("Not supported: delete_from_table")

    def query_table(self, request: QueryTableRequest) -> bytes:
        """Query a table."""
        raise NotImplementedError("Not supported: query_table")

    def create_table_index(
        self, request: CreateTableIndexRequest
    ) -> CreateTableIndexResponse:
        """Create a table index."""
        raise NotImplementedError("Not supported: create_table_index")

    def list_table_indices(
        self, request: ListTableIndicesRequest
    ) -> ListTableIndicesResponse:
        """List table indices."""
        raise NotImplementedError("Not supported: list_table_indices")

    def describe_table_index_stats(
        self, request: DescribeTableIndexStatsRequest
    ) -> DescribeTableIndexStatsResponse:
        """Describe table index statistics."""
        raise NotImplementedError("Not supported: describe_table_index_stats")

    def describe_transaction(
        self, request: DescribeTransactionRequest
    ) -> DescribeTransactionResponse:
        """Describe a transaction."""
        raise NotImplementedError("Not supported: describe_transaction")

    def alter_transaction(
        self, request: AlterTransactionRequest
    ) -> AlterTransactionResponse:
        """Alter a transaction."""
        raise NotImplementedError("Not supported: alter_transaction")


class DirectoryNamespace(LanceNamespace):
    """Directory-based Lance Namespace implementation backed by Rust.

    This namespace stores tables as Lance datasets in a filesystem directory structure.
    It uses a manifest table to track tables and nested namespaces efficiently.

    This is a Python wrapper around the Rust PyDirectoryNamespace implementation,
    providing compatibility with the LanceNamespace ABC interface.

    Parameters
    ----------
    session : Session, optional
        Lance session for sharing object store connections. If provided,
        this namespace will reuse the session's object store registry.
    **properties : dict
        Configuration properties as key-value pairs:
        - root (required): Root directory path or URI
        - manifest_enabled (optional): Enable manifest tracking (default: "true")
        - dir_listing_enabled (optional): Enable directory listing fallback
          (default: "true")
        - storage.* (optional): Storage options with "storage." prefix
          (e.g., storage.region="us-west-2" becomes region="us-west-2" in
          storage options)

    Examples
    --------
    >>> import lance.namespace
    >>> # Create with properties dict
    >>> ns = lance.namespace.DirectoryNamespace(root="memory://test")
    >>>
    >>> # With storage options
    >>> ns = lance.namespace.DirectoryNamespace(
    ...     root="/path/to/data",
    ...     manifest_enabled="true",
    ...     **{"storage.region": "us-west-2"}
    ... )
    >>>
    >>> # Using the connect() factory function
    >>> import lance.namespace
    >>> ns = lance.namespace.connect("dir", {"root": "memory://test"})
    """

    def __init__(self, session=None, **properties):
        # Convert all values to strings as expected by Rust from_properties
        str_properties = {str(k): str(v) for k, v in properties.items()}

        # Create the underlying Rust namespace
        self._inner = PyDirectoryNamespace(session=session, **str_properties)

    def namespace_id(self) -> str:
        """Return a human-readable unique identifier for this namespace instance."""
        return self._inner.namespace_id()

    def __repr__(self) -> str:
        return f"DirectoryNamespace({self._inner.namespace_id()})"

    # Namespace operations

    def create_namespace(
        self, request: CreateNamespaceRequest
    ) -> CreateNamespaceResponse:
        response_dict = self._inner.create_namespace(request.model_dump())
        return CreateNamespaceResponse.from_dict(response_dict)

    def list_namespaces(self, request: ListNamespacesRequest) -> ListNamespacesResponse:
        response_dict = self._inner.list_namespaces(request.model_dump())
        return ListNamespacesResponse.from_dict(response_dict)

    def describe_namespace(
        self, request: DescribeNamespaceRequest
    ) -> DescribeNamespaceResponse:
        response_dict = self._inner.describe_namespace(request.model_dump())
        return DescribeNamespaceResponse.from_dict(response_dict)

    def drop_namespace(self, request: DropNamespaceRequest) -> DropNamespaceResponse:
        response_dict = self._inner.drop_namespace(request.model_dump())
        return DropNamespaceResponse.from_dict(response_dict)

    def namespace_exists(self, request: NamespaceExistsRequest) -> None:
        self._inner.namespace_exists(request.model_dump())

    # Table operations

    def list_tables(self, request: ListTablesRequest) -> ListTablesResponse:
        response_dict = self._inner.list_tables(request.model_dump())
        return ListTablesResponse.from_dict(response_dict)

    def describe_table(self, request: DescribeTableRequest) -> DescribeTableResponse:
        response_dict = self._inner.describe_table(request.model_dump())
        return DescribeTableResponse.from_dict(response_dict)

    def register_table(self, request: RegisterTableRequest) -> RegisterTableResponse:
        response_dict = self._inner.register_table(request.model_dump())
        return RegisterTableResponse.from_dict(response_dict)

    def table_exists(self, request: TableExistsRequest) -> None:
        self._inner.table_exists(request.model_dump())

    def drop_table(self, request: DropTableRequest) -> DropTableResponse:
        response_dict = self._inner.drop_table(request.model_dump())
        return DropTableResponse.from_dict(response_dict)

    def deregister_table(
        self, request: DeregisterTableRequest
    ) -> DeregisterTableResponse:
        response_dict = self._inner.deregister_table(request.model_dump())
        return DeregisterTableResponse.from_dict(response_dict)

    def create_table(
        self, request: CreateTableRequest, request_data: bytes
    ) -> CreateTableResponse:
        response_dict = self._inner.create_table(request.model_dump(), request_data)
        return CreateTableResponse.from_dict(response_dict)

    def create_empty_table(
        self, request: CreateEmptyTableRequest
    ) -> CreateEmptyTableResponse:
        response_dict = self._inner.create_empty_table(request.model_dump())
        return CreateEmptyTableResponse.from_dict(response_dict)


class RestNamespace(LanceNamespace):
    """REST-based Lance Namespace implementation backed by Rust.

    This namespace communicates with a Lance REST API server to manage
    namespaces and tables remotely.

    This is a Python wrapper around the Rust PyRestNamespace implementation,
    providing compatibility with the LanceNamespace ABC interface.

    Parameters
    ----------
    **properties : dict
        Configuration properties as key-value pairs:
        - uri (required): REST endpoint URI (e.g., "http://localhost:4099")
        - delimiter (optional): Namespace delimiter, default "$"
        - header.* (optional): HTTP headers with "header." prefix
          (e.g., header.Authorization="Bearer token" becomes
          Authorization="Bearer token" in HTTP headers)
        - tls.* (optional): TLS configuration with "tls." prefix

    Examples
    --------
    >>> import lance.namespace
    >>> # Create with properties dict
    >>> ns = lance.namespace.RestNamespace(uri="http://localhost:4099")
    >>>
    >>> # With custom delimiter and headers
    >>> ns = lance.namespace.RestNamespace(
    ...     uri="http://localhost:4099",
    ...     delimiter=".",
    ...     **{"header.Authorization": "Bearer token"}
    ... )
    >>>
    >>> # Using the connect() factory function
    >>> import lance.namespace
    >>> ns = lance.namespace.connect("rest", {"uri": "http://localhost:4099"})
    """

    def __init__(self, **properties):
        if PyRestNamespace is None:
            raise RuntimeError(
                "RestNamespace is not available. "
                "Lance was built without REST support. "
                "Please rebuild with the 'rest' feature enabled."
            )
        # Convert all values to strings as expected by Rust from_properties
        str_properties = {str(k): str(v) for k, v in properties.items()}

        # Create the underlying Rust namespace
        self._inner = PyRestNamespace(**str_properties)

    def namespace_id(self) -> str:
        """Return a human-readable unique identifier for this namespace instance."""
        return self._inner.namespace_id()

    def __repr__(self) -> str:
        return f"RestNamespace({self._inner.namespace_id()})"

    # Namespace operations

    def create_namespace(
        self, request: CreateNamespaceRequest
    ) -> CreateNamespaceResponse:
        response_dict = self._inner.create_namespace(request.model_dump())
        return CreateNamespaceResponse.from_dict(response_dict)

    def list_namespaces(self, request: ListNamespacesRequest) -> ListNamespacesResponse:
        response_dict = self._inner.list_namespaces(request.model_dump())
        return ListNamespacesResponse.from_dict(response_dict)

    def describe_namespace(
        self, request: DescribeNamespaceRequest
    ) -> DescribeNamespaceResponse:
        response_dict = self._inner.describe_namespace(request.model_dump())
        return DescribeNamespaceResponse.from_dict(response_dict)

    def drop_namespace(self, request: DropNamespaceRequest) -> DropNamespaceResponse:
        response_dict = self._inner.drop_namespace(request.model_dump())
        return DropNamespaceResponse.from_dict(response_dict)

    def namespace_exists(self, request: NamespaceExistsRequest) -> None:
        self._inner.namespace_exists(request.model_dump())

    # Table operations

    def list_tables(self, request: ListTablesRequest) -> ListTablesResponse:
        response_dict = self._inner.list_tables(request.model_dump())
        return ListTablesResponse.from_dict(response_dict)

    def describe_table(self, request: DescribeTableRequest) -> DescribeTableResponse:
        response_dict = self._inner.describe_table(request.model_dump())
        return DescribeTableResponse.from_dict(response_dict)

    def register_table(self, request: RegisterTableRequest) -> RegisterTableResponse:
        response_dict = self._inner.register_table(request.model_dump())
        return RegisterTableResponse.from_dict(response_dict)

    def table_exists(self, request: TableExistsRequest) -> None:
        self._inner.table_exists(request.model_dump())

    def drop_table(self, request: DropTableRequest) -> DropTableResponse:
        response_dict = self._inner.drop_table(request.model_dump())
        return DropTableResponse.from_dict(response_dict)

    def deregister_table(
        self, request: DeregisterTableRequest
    ) -> DeregisterTableResponse:
        response_dict = self._inner.deregister_table(request.model_dump())
        return DeregisterTableResponse.from_dict(response_dict)

    def create_table(
        self, request: CreateTableRequest, request_data: bytes
    ) -> CreateTableResponse:
        response_dict = self._inner.create_table(request.model_dump(), request_data)
        return CreateTableResponse.from_dict(response_dict)

    def create_empty_table(
        self, request: CreateEmptyTableRequest
    ) -> CreateEmptyTableResponse:
        response_dict = self._inner.create_empty_table(request.model_dump())
        return CreateEmptyTableResponse.from_dict(response_dict)


class RestAdapter:
    """REST adapter server that creates a namespace backend and exposes it via REST.

    This adapter starts an HTTP server that exposes a Lance namespace backend
    via the Lance REST API. The backend namespace can be any implementation
    (DirectoryNamespace, etc.) created from the provided configuration.
    Useful for testing RestNamespace clients.

    Parameters
    ----------
    namespace_impl : str
        Namespace implementation type ("dir", "rest", etc.)
    namespace_properties : dict, optional
        Configuration properties for the backend namespace.
        For DirectoryNamespace ("dir"):
        - root (required): Root directory path or URI
        - manifest_enabled (optional): Enable manifest tracking (default: "true")
        - dir_listing_enabled (optional): Enable directory listing fallback
        - storage.* (optional): Storage options with "storage." prefix
    session : Session, optional
        Lance session for sharing object store connections with the backend namespace.
    host : str, optional
        Host address to bind to, default "127.0.0.1"
    port : int, optional
        Port to listen on, default 2333

    Examples
    --------
    >>> import lance.namespace
    >>>
    >>> # Start REST adapter with DirectoryNamespace backend
    >>> namespace_config = {"root": "memory://test"}
    >>> with lance.namespace.RestAdapter("dir", namespace_config, port=4001) as adapter:
    ...     # Create REST client
    ...     client = lance.namespace.RestNamespace(uri="http://127.0.0.1:4001")
    ...     # Use the client...
    """

    def __init__(
        self,
        namespace_impl: str,
        namespace_properties: Dict[str, str] = None,
        session=None,
        host: str = "127.0.0.1",
        port: int = 2333,
    ):
        if PyRestAdapter is None:
            raise RuntimeError(
                "RestAdapter is not available. "
                "Lance was built without REST adapter support. "
                "Please rebuild with the 'rest-adapter' feature enabled."
            )

        # Convert to string properties
        if namespace_properties is None:
            namespace_properties = {}
        str_properties = {str(k): str(v) for k, v in namespace_properties.items()}

        # Create the underlying Rust adapter
        self._inner = PyRestAdapter(namespace_impl, str_properties, session, host, port)
        self.host = host
        self.port = port
        self.namespace_impl = namespace_impl

    def serve(self):
        """Start the REST server in the background."""
        self._inner.serve()

    def stop(self):
        """Stop the REST server."""
        self._inner.stop()

    def __enter__(self):
        """Start server when entering context."""
        self.serve()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop server when exiting context."""
        self.stop()
        return False

    def __repr__(self) -> str:
        return f"RestAdapter(host='{self.host}', port={self.port})"


class LanceNamespaceStorageOptionsProvider(StorageOptionsProvider):
    """Storage options provider that fetches storage options from a LanceNamespace.

    This provider automatically fetches fresh storage options by calling the
    namespace's describe_table() method, which returns both the table location
    and time-limited storage options. This is currently only used for refreshing
    AWS temporary access credentials.

    This is the recommended approach for LanceDB Cloud and other namespace-based
    deployments, as it handles storage options refresh automatically.

    Parameters
    ----------
    namespace : LanceNamespace
        The namespace instance to fetch storage options from. Use
        lance.namespace.connect() to create a namespace instance.
    table_id : List[str]
        The table identifier (e.g., ["workspace", "table_name"])

    Example
    -------
    This example shows how to use the storage options provider with a namespace.

    .. code-block:: python

        import lance
        import lance.namespace

        # Connect to a namespace
        namespace = lance.namespace.connect("rest", {"uri": "http://localhost:4099"})

        # Create storage options provider
        provider = lance.LanceNamespaceStorageOptionsProvider(
            namespace=namespace,
            table_id=["workspace", "table_name"]
        )

        # Use with dataset - storage options auto-refresh!
        dataset = lance.dataset(
            "s3://bucket/table.lance",
            storage_options_provider=provider
        )
    """

    def __init__(self, namespace: LanceNamespace, table_id: List[str]):
        """Initialize with namespace and table ID.

        Parameters
        ----------
        namespace : LanceNamespace
            The namespace instance with a describe_table() method
        table_id : List[str]
            The table identifier
        """
        self._namespace = namespace
        self._table_id = table_id

    def fetch_storage_options(self) -> Dict[str, str]:
        """Fetch storage options from the namespace.

        This calls namespace.describe_table() to get the latest storage options
        and their expiration time.

        Returns
        -------
        Dict[str, str]
            Flat dictionary of string key-value pairs containing storage options
            and expires_at_millis

        Raises
        ------
        RuntimeError
            If the namespace doesn't return storage options or expiration time
        """
        request = DescribeTableRequest(id=self._table_id, version=None)
        response = self._namespace.describe_table(request)
        storage_options = response.storage_options
        if storage_options is None:
            raise RuntimeError(
                "Namespace did not return storage_options. "
                "Ensure the namespace supports storage options providing."
            )

        # Verify expires_at_millis is present
        if "expires_at_millis" not in storage_options:
            raise RuntimeError(
                "Namespace storage_options missing 'expires_at_millis'. "
                "Storage options refresh will not work properly."
            )

        # Return the storage_options directly - it's already a flat Map<String, String>
        return storage_options

    def provider_id(self) -> str:
        """Return a human-readable unique identifier for this provider instance.

        This creates a semantic ID based on the namespace's ID and the table ID,
        enabling proper equality comparison and caching.

        Returns
        -------
        str
            A human-readable unique identifier string combining namespace and table info
        """
        # Try to call namespace_id() if available (lance-namespace >= 0.0.20)
        if hasattr(self._namespace, "namespace_id"):
            namespace_id = self._namespace.namespace_id()
        else:
            # Fallback for older namespace versions
            namespace_id = str(self._namespace)

        return (
            f"LanceNamespaceStorageOptionsProvider {{ "
            f"namespace: {namespace_id}, table_id: {self._table_id!r} }}"
        )


# Native implementations (Rust-backed)
NATIVE_IMPLS = {
    "rest": "lance.namespace.RestNamespace",
    "dir": "lance.namespace.DirectoryNamespace",
}

# Plugin registry for external implementations
_REGISTERED_IMPLS: Dict[str, str] = {}


def register_namespace_impl(name: str, class_path: str) -> None:
    """Register a namespace implementation with a short name.

    External libraries can use this to register their implementations,
    allowing users to use short names like "glue" instead of full class paths.

    Parameters
    ----------
    name : str
        Short name for the implementation (e.g., "glue", "hive2", "unity")
    class_path : str
        Full class path (e.g., "lance_glue.GlueNamespace")
    """
    _REGISTERED_IMPLS[name] = class_path


def connect(impl: str, properties: Dict[str, str]) -> LanceNamespace:
    """Connect to a Lance namespace implementation.

    This factory function creates namespace instances based on implementation
    aliases or full class paths. It provides a unified way to instantiate
    different namespace backends.

    Parameters
    ----------
    impl : str
        Implementation alias or full class path. Built-in aliases:
        - "rest": RestNamespace (REST API client)
        - "dir": DirectoryNamespace (local/cloud filesystem)
        You can also use full class paths like "my.custom.Namespace"
        External libraries can register additional aliases using
        `register_namespace_impl()`.
    properties : Dict[str, str]
        Configuration properties passed to the namespace constructor

    Returns
    -------
    LanceNamespace
        The connected namespace instance

    Raises
    ------
    ValueError
        If the implementation class cannot be loaded or does not
        implement LanceNamespace interface
    """
    # Check native impls first, then registered plugins, then treat as full class path
    impl_class = NATIVE_IMPLS.get(impl) or _REGISTERED_IMPLS.get(impl) or impl
    try:
        module_name, class_name = impl_class.rsplit(".", 1)
        module = importlib.import_module(module_name)
        namespace_class = getattr(module, class_name)

        if not issubclass(namespace_class, LanceNamespace):
            raise ValueError(
                f"Class {impl_class} does not implement LanceNamespace interface"
            )

        return namespace_class(**properties)
    except Exception as e:
        raise ValueError(f"Failed to construct namespace impl {impl_class}: {e}")
