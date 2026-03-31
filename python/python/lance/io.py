# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""I/O utilities for Lance datasets.

This module provides utilities for customizing how Lance datasets interact with
cloud storage.
"""

from typing import Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class StorageOptionsProvider(Protocol):
    """Protocol for providing dynamic storage options.

    Implementations can fetch storage options from various sources (credential
    vending services, secret managers, etc.) with support for expiration tracking.

    This is a Protocol for type checking - the actual implementation details are
    handled in Rust. Any class implementing this protocol just needs to provide
    the required methods.

    Example
    -------
    >>> class MyCredentialProvider:
    ...     def fetch_storage_options(self) -> Optional[Dict[str, str]]:
    ...         # Fetch credentials from your secret manager
    ...         return {
    ...             "aws_access_key_id": "...",
    ...             "aws_secret_access_key": "...",
    ...             "expires_at_millis": "...",  # epoch millis when creds expire
    ...         }
    ...
    ...     def provider_id(self) -> str:
    ...         return "my-credential-provider"
    """

    def fetch_storage_options(self) -> Optional[Dict[str, str]]:
        """Fetch fresh storage options.

        Returns
        -------
        Optional[Dict[str, str]]
            Storage options as a dictionary, or None if no options are available.

            If the `expires_at_millis` key is present in the dictionary, it should
            contain the epoch time in milliseconds when the options expire, and
            credentials will automatically refresh before expiration.
        """
        ...

    def provider_id(self) -> str:
        """Return a unique identifier for this provider instance.

        This is used for equality comparison and caching purposes.
        Two providers with the same ID will be treated as equal.

        Returns
        -------
        str
            A human-readable unique identifier for this provider.
            For example: "my-namespace-provider[table=my_table]"
        """
        ...
