# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""LanceNamespace credential integration.

This module provides credential vending integration with LanceNamespace,
enabling automatic credential refresh for namespace-managed tables.
"""

from typing import Dict

from .io import CredentialVendor


class LanceNamespaceCredentialVendor(CredentialVendor):
    """Credential vendor that fetches credentials from a LanceNamespace.

    This vendor automatically fetches fresh credentials by calling the
    namespace's describe_table() method, which returns both the table location
    and time-limited storage credentials.

    This is the recommended approach for LanceDB Cloud and other namespace-based
    deployments, as it handles credential refresh automatically.

    Parameters
    ----------
    namespace : lance._lib._Namespace
        The namespace instance to fetch credentials from
    table_id : List[str]
        The table identifier (e.g., ["workspace", "table_name"])

    Example
    -------
    >>> import lance
    >>> # Connect to a namespace (e.g., LanceDB Cloud)
    >>> namespace = lance._lib.connect_namespace("rest", {
    ...     "url": "https://api.lancedb.com",
    ...     "api_key": "your-api-key"
    ... })
    >>>
    >>> # Create credential vendor
    >>> vendor = lance.LanceNamespaceCredentialVendor(
    ...     namespace=namespace,
    ...     table_id=["workspace", "table_name"]
    ... )
    >>>
    >>> # Use with dataset - credentials auto-refresh!
    >>> dataset = lance.dataset(
    ...     "s3://bucket/table.lance",
    ...     credential_vendor=vendor
    ... )
    """

    def __init__(self, namespace, table_id: list):
        """Initialize with namespace and table ID.

        Parameters
        ----------
        namespace : lance._lib._Namespace
            The namespace instance
        table_id : List[str]
            The table identifier
        """
        self._namespace = namespace
        self._table_id = table_id

    def get_credentials(self) -> Dict:
        """Fetch credentials from the namespace.

        This calls namespace.describe_table() to get the latest credentials
        and their expiration time.

        Returns
        -------
        dict
            Credentials dictionary with "storage_options" and "expires_at_millis"

        Raises
        ------
        RuntimeError
            If the namespace doesn't return storage credentials or expiration time
        """
        # Call namespace to describe the table and get credentials
        table_info = self._namespace.describe_table(
            table_id=self._table_id,
            version=None
        )

        # Extract storage options
        storage_options = table_info.get("storage_options")
        if storage_options is None:
            raise RuntimeError(
                "Namespace did not return storage_options. "
                "Ensure the namespace supports credential vending."
            )

        # Extract expiration time from storage_options
        # The expires_at_millis is included in storage_options by the namespace
        expires_at_millis_str = storage_options.get("expires_at_millis")
        if expires_at_millis_str is None:
            raise RuntimeError(
                "Namespace storage_options missing 'expires_at_millis'. "
                "Credential refresh will not work properly."
            )

        expires_at_millis = int(expires_at_millis_str)

        return {
            "storage_options": storage_options,
            "expires_at_millis": expires_at_millis,
        }
