# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""LanceNamespace credential integration.

This module provides credential vending integration with LanceNamespace,
enabling automatic credential refresh for namespace-managed tables.
"""

from typing import Dict

from .io import StorageOptionsProvider


class LanceNamespaceStorageOptionsProvider(StorageOptionsProvider):
    """Credential vendor that fetches credentials from a LanceNamespace.

    This vendor automatically fetches fresh credentials by calling the
    namespace's describe_table() method, which returns both the table location
    and time-limited storage credentials.

    This is the recommended approach for LanceDB Cloud and other namespace-based
    deployments, as it handles credential refresh automatically.

    Parameters
    ----------
    namespace : any
        The namespace instance to fetch credentials from. This can be any object
        with a describe_table(table_id, version) method that returns a dict with
        'location' and 'storage_options' keys. For example, use lance_namespace.connect()
        from the lance_namespace PyPI package.
    table_id : List[str]
        The table identifier (e.g., ["workspace", "table_name"])

    Example
    -------
    This example shows how to use the credential vendor with a namespace.

    .. code-block:: python

        import lance
        import lance_namespace

        # Connect to a namespace (e.g., LanceDB Cloud)
        namespace = lance_namespace.connect("rest", {
            "url": "https://api.lancedb.com",
            "api_key": "your-api-key"
        })

        # Create credential vendor
        vendor = lance.LanceNamespaceStorageOptionsProvider(
            namespace=namespace,
            table_id=["workspace", "table_name"]
        )

        # Use with dataset - credentials auto-refresh!
        dataset = lance.dataset(
            "s3://bucket/table.lance",
            storage_options_provider=vendor
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

    def get_storage_options(self) -> Dict[str, str]:
        """Fetch credentials from the namespace.

        This calls namespace.describe_table() to get the latest credentials
        and their expiration time.

        Returns
        -------
        Dict[str, str]
            Flat dictionary of string key-value pairs containing credentials
            and expires_at_millis

        Raises
        ------
        RuntimeError
            If the namespace doesn't return storage credentials or expiration time
        """
        # Call namespace to describe the table and get credentials
        table_info = self._namespace.describe_table(
            table_id=self._table_id, version=None
        )

        # Extract storage options - should already be a flat dict of strings
        storage_options = table_info.get("storage_options")
        if storage_options is None:
            raise RuntimeError(
                "Namespace did not return storage_options. "
                "Ensure the namespace supports credential vending."
            )

        # Verify expires_at_millis is present
        if "expires_at_millis" not in storage_options:
            raise RuntimeError(
                "Namespace storage_options missing 'expires_at_millis'. "
                "Credential refresh will not work properly."
            )

        # Return the storage_options directly - it's already a flat Map<String, String>
        return storage_options
