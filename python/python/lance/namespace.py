# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""LanceNamespace storage options integration.

This module provides storage options integration with LanceNamespace,
enabling automatic storage options refresh for namespace-managed tables.
"""

from typing import Dict

from .io import StorageOptionsProvider


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
    namespace : any
        The namespace instance to fetch storage options from. This can be any
        object with a describe_table(table_id, version) method that returns a
        dict with 'location' and 'storage_options' keys. For example, use
        lance_namespace.connect() from the lance_namespace PyPI package.
    table_id : List[str]
        The table identifier (e.g., ["workspace", "table_name"])

    Example
    -------
    This example shows how to use the storage options provider with a namespace.

    .. code-block:: python

        import lance
        import lance_namespace

        # Connect to a namespace (e.g., LanceDB Cloud)
        namespace = lance_namespace.connect("rest", {
            "url": "https://api.lancedb.com",
            "api_key": "your-api-key"
        })

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

    def __init__(self, namespace, table_id: list):
        """Initialize with namespace and table ID.

        Parameters
        ----------
        namespace : any
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
        # Call namespace to describe the table and get storage options
        table_info = self._namespace.describe_table(
            table_id=self._table_id, version=None
        )

        # Extract storage options - should already be a flat dict of strings
        storage_options = table_info.get("storage_options")
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
