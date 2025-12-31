# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""LanceNamespace storage options integration and implementations.

This module provides:
1. Native Rust-backed namespace implementations (DirectoryNamespace, RestNamespace)
2. Storage options integration with LanceNamespace for automatic credential refresh
3. Plugin registry for external namespace implementations

The LanceNamespace ABC interface is provided by the lance_namespace package.
"""

from typing import Dict, List

from lance_namespace import (
    CreateEmptyTableRequest,
    CreateEmptyTableResponse,
    CreateNamespaceRequest,
    CreateNamespaceResponse,
    CreateTableRequest,
    CreateTableResponse,
    DeclareTableRequest,
    DeclareTableResponse,
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

try:
    from .lance import PyRestNamespace  # Low-level Rust binding
except ImportError:
    PyRestNamespace = None

try:
    from .lance import PyRestAdapter  # Low-level Rust binding
except ImportError:
    PyRestAdapter = None

__all__ = [
    "DirectoryNamespace",
    "RestNamespace",
    "RestAdapter",
    "LanceNamespaceStorageOptionsProvider",
]


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

        Credential vendor properties (vendor is auto-selected based on table location):
            When credential vendor properties are configured, describe_table() will
            return vended temporary credentials. The vendor type is auto-selected
            based on table location URI: s3:// for AWS, gs:// for GCP, az:// for
            Azure. Requires the corresponding credential-vendor-* feature.

            Common properties:
                - credential_vendor.enabled (required): Set to "true" to enable
                - credential_vendor.permission (optional): read, write, or admin

            AWS-specific properties (for s3:// locations):
                - credential_vendor.aws_role_arn (required): IAM role ARN to assume
                - credential_vendor.aws_external_id (optional): External ID
                - credential_vendor.aws_region (optional): AWS region
                - credential_vendor.aws_role_session_name (optional): Session name
                - credential_vendor.aws_duration_millis (optional): Duration in ms
                  (default: 3600000, range: 15min-12hrs)

            GCP-specific properties (for gs:// locations):
                - credential_vendor.gcp_service_account (optional): Service account
                  to impersonate using IAM Credentials API

                Note: GCP uses Application Default Credentials (ADC). To use a service
                account key file, set the GOOGLE_APPLICATION_CREDENTIALS environment
                variable before starting. GCP token duration cannot be configured;
                it's determined by the STS endpoint (typically 1 hour).

            Azure-specific properties (for az:// locations):
                - credential_vendor.azure_account_name (required): Azure storage
                  account name
                - credential_vendor.azure_tenant_id (optional): Azure tenant ID
                - credential_vendor.azure_duration_millis (optional): Duration in ms
                  (default: 3600000, up to 7 days)

    Examples
    --------
    >>> import lance.namespace
    >>> # Create with properties dict
    >>> ns = lance.namespace.DirectoryNamespace(root="memory://test")
    >>>
    >>> # Using the connect() factory function from lance_namespace
    >>> import lance_namespace
    >>> ns = lance_namespace.connect("dir", {"root": "memory://test"})
    >>>
    >>> # With AWS credential vending (requires credential-vendor-aws feature)
    >>> # Use **dict to pass property names with dots
    >>> ns = lance.namespace.DirectoryNamespace(**{
    ...     "root": "s3://my-bucket/data",
    ...     "credential_vendor.enabled": "true",
    ...     "credential_vendor.aws_role_arn": "arn:aws:iam::123456789012:role/MyRole",
    ...     "credential_vendor.aws_duration_millis": "3600000",
    ... })
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

    def declare_table(self, request: DeclareTableRequest) -> DeclareTableResponse:
        response_dict = self._inner.declare_table(request.model_dump())
        return DeclareTableResponse.from_dict(response_dict)


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
    >>> # Using the connect() factory function from lance_namespace
    >>> import lance_namespace
    >>> ns = lance_namespace.connect("rest", {"uri": "http://localhost:4099"})
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

    def declare_table(self, request: DeclareTableRequest) -> DeclareTableResponse:
        response_dict = self._inner.declare_table(request.model_dump())
        return DeclareTableResponse.from_dict(response_dict)


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
        Host address to bind to. Default "127.0.0.1".
    port : int, optional
        Port to listen on. Default 2333 per REST spec.
        Use 0 to let the OS assign an available ephemeral port.
        Use the `port` property after `start()` to get the actual port.

    Examples
    --------
    >>> import lance.namespace
    >>>
    >>> # Start REST adapter with DirectoryNamespace backend (auto port)
    >>> namespace_config = {"root": "memory://test"}
    >>> with lance.namespace.RestAdapter("dir", namespace_config) as adapter:
    ...     # Create REST client using the assigned port
    ...     client = lance.namespace.RestNamespace(uri=f"http://127.0.0.1:{adapter.port}")
    ...     # Use the client...
    """

    def __init__(
        self,
        namespace_impl: str,
        namespace_properties: Dict[str, str] = None,
        session=None,
        host: str = None,
        port: int = None,
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
        self.namespace_impl = namespace_impl

    @property
    def port(self) -> int:
        """Get the actual port the server is listening on.

        Returns 0 if the server hasn't been started yet.
        """
        return self._inner.port

    def start(self):
        """Start the REST server in the background."""
        self._inner.start()

    def stop(self):
        """Stop the REST server."""
        self._inner.stop()

    def __enter__(self):
        """Start server when entering context."""
        self.start()
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
        and optionally their expiration time.

        Returns
        -------
        Dict[str, str]
            Flat dictionary of string key-value pairs containing storage options.
            May optionally include expires_at_millis. If expires_at_millis is not
            provided, credentials are treated as non-expiring and will not be
            automatically refreshed.

        Raises
        ------
        RuntimeError
            If the namespace doesn't return storage options
        """
        request = DescribeTableRequest(id=self._table_id, version=None)
        response = self._namespace.describe_table(request)
        storage_options = response.storage_options
        if storage_options is None:
            raise RuntimeError(
                "Namespace did not return storage_options. "
                "Ensure the namespace supports storage options providing."
            )

        # Return the storage_options directly - it's already a flat Map<String, String>
        # Note: expires_at_millis is optional. If not provided, credentials are treated
        # as non-expiring and will not be automatically refreshed.
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
