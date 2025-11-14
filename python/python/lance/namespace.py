# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""LanceNamespace storage options integration and implementations.

This module provides:
1. Native Rust-backed namespace implementations (DirectoryNamespace)
2. Storage options integration with LanceNamespace for automatic credential refresh
"""

from typing import Dict, List, Optional

from lance_namespace import (
    CreateEmptyTableRequest,
    CreateEmptyTableResponse,
    CreateNamespaceRequest,
    CreateNamespaceResponse,
    CreateTableRequest,
    CreateTableResponse,
    DeregisterTableRequest,
    DeregisterTableResponse,
    DescribeNamespaceRequest,
    DescribeNamespaceResponse,
    DescribeTableRequest,
    DescribeTableResponse,
    DropNamespaceRequest,
    DropNamespaceResponse,
    DropTableRequest,
    DropTableResponse,
    LanceNamespace,
    ListNamespacesRequest,
    ListNamespacesResponse,
    ListTablesRequest,
    ListTablesResponse,
    NamespaceExistsRequest,
    RegisterTableRequest,
    RegisterTableResponse,
    TableExistsRequest,
)

from .io import StorageOptionsProvider
from .lance import PyDirectoryNamespace  # Low-level Rust binding

__all__ = ["DirectoryNamespace", "LanceNamespaceStorageOptionsProvider"]


class DirectoryNamespace(LanceNamespace):
    """Directory-based Lance Namespace implementation backed by Rust.

    This namespace stores tables as Lance datasets in a filesystem directory structure.
    It uses a manifest table to track tables and nested namespaces efficiently.

    This is a Python wrapper around the Rust PyDirectoryNamespace implementation,
    providing compatibility with the LanceNamespace ABC interface.

    Parameters
    ----------
    root : str
        Root directory path or URI (e.g., "memory://test" or "/local/path")
        Note: Cloud storage (s3://, gs://, az://) support depends on lance-io features
    storage_options : dict, optional
        Storage options for accessing the filesystem (e.g., AWS credentials)
    manifest_enabled : bool, optional
        Whether to enable manifest-based table tracking (default: True)
    dir_listing_enabled : bool, optional
        Whether to enable directory listing fallback (default: True)

    Examples
    --------
    >>> import lance.namespace
    >>> # Local directory
    >>> ns = lance.namespace.DirectoryNamespace("memory://test")
    >>>
    >>> # File system path
    >>> ns = lance.namespace.DirectoryNamespace("/path/to/data")
    """

    def __init__(
        self,
        root: str,
        storage_options: Optional[Dict[str, str]] = None,
        manifest_enabled: Optional[bool] = None,
        dir_listing_enabled: Optional[bool] = None,
    ):
        # Create the underlying Rust namespace
        self._inner = PyDirectoryNamespace(
            root,
            storage_options=storage_options,
            manifest_enabled=manifest_enabled,
            dir_listing_enabled=dir_listing_enabled,
        )

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
        lance_namespace.connect() from the lance_namespace PyPI package.
    table_id : List[str]
        The table identifier (e.g., ["workspace", "table_name"])

    Example
    -------
    This example shows how to use the storage options provider with a namespace.

    .. code-block:: python

        import lance
        import lance_namespace

        # Connect to a namespace (using the lance_namespace package)
        namespace = lance_namespace.connect("http://localhost:4099")

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
