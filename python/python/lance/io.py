# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""I/O utilities for Lance datasets.

This module provides utilities for customizing how Lance datasets interact with
cloud storage, including credential management for long-running operations.
"""

from abc import ABC, abstractmethod
from typing import Dict


class StorageOptionsProvider(ABC):
    """Abstract base class for providing storage options to Lance datasets.

    Storage options providers enable automatic refresh for long-running operations
    on cloud storage (S3, Azure, GCS). This is currently only used for refreshing
    AWS temporary access credentials. Implement this interface to integrate with
    custom credential management systems such as AWS STS, GCP STS, or
    proprietary credential services.

    The provider is called automatically before storage options expire, ensuring
    uninterrupted access during long-running queries, training jobs, or data processing.

    Example
    -------
    >>> import lance
    >>> class MyStorageOptionsProvider(StorageOptionsProvider):
    ...     def fetch_storage_options(self):
    ...         # Fetch from your credential service
    ...         return {
    ...             "aws_access_key_id": "ASIA...",
    ...             "aws_secret_access_key": "secret",
    ...             "aws_session_token": "token",
    ...             "expires_at_millis": "1234567890000",
    ...         }
    ...
    >>> provider = MyStorageOptionsProvider()
    >>> dataset = lance.dataset(  # doctest: +SKIP
    ...     "s3://bucket/table.lance", storage_options_provider=provider
    ... )

    Error Handling
    --------------
    If fetch_storage_options() raises an exception, operations requiring
    credentials will fail. Implementations should handle recoverable errors
    internally (e.g., retry token refresh) and only raise exceptions for
    unrecoverable errors.
    """

    @abstractmethod
    def fetch_storage_options(self) -> Dict[str, str]:
        """Get fresh storage credentials.

        This method is called automatically before each request and before existing
        credentials expire. It must return credentials in the format below.

        Returns
        -------
        Dict[str, str]
            Dictionary of string key-value pairs containing cloud storage credentials
            and expiration time. Required keys:

            - "expires_at_millis" (str): Unix timestamp in milliseconds (as string)
              when credentials expire. Lance will automatically call
              fetch_storage_options() again before this time.

            Plus provider-specific credential keys:

            AWS S3:
              - "aws_access_key_id" (str): AWS access key
              - "aws_secret_access_key" (str): AWS secret key
              - "aws_session_token" (str, optional): Session token for temporary
                credentials

            Azure Blob Storage:
              - "account_name" (str): Storage account name
              - "account_key" (str): Storage account key
              - Or "sas_token" (str): SAS token

            Google Cloud Storage:
              - "service_account_key" (str): Service account JSON key
              - Or "token" (str): OAuth token

        Raises
        ------
        Exception
            If unable to fetch credentials, the exception will be propagated
            and operations requiring credentials will fail.

        Example
        -------
        >>> def fetch_storage_options(self):
        ...     # Example: AWS temporary credentials
        ...     response = sts_client.assume_role(
        ...         RoleArn='arn:aws:iam::123456789012:role/DataReader',
        ...         RoleSessionName='lance-session'
        ...     )
        ...     creds = response['Credentials']
        ...     expires_at_millis = int(creds['Expiration'].timestamp() * 1000)
        ...     return {
        ...         "aws_access_key_id": creds['AccessKeyId'],
        ...         "aws_secret_access_key": creds['SecretAccessKey'],
        ...         "aws_session_token": creds['SessionToken'],
        ...         "expires_at_millis": str(expires_at_millis),
        ...     }
        """
        pass

    def provider_id(self) -> str:
        """Return a human-readable unique identifier for this provider instance.

        This is used for equality comparison and hashing in the object store
        registry. Two providers with the same ID will be treated as equal and
        share the same cached ObjectStore instance.

        The default implementation uses the class name and object's string
        representation. Override this method to provide semantic equality based
        on configuration.

        Returns
        -------
        str
            A human-readable unique identifier string.
            For example: "MyProvider { endpoint: 'https://api.example.com' }"

        Example
        -------
        >>> class MyProvider(StorageOptionsProvider):
        ...     def __init__(self, endpoint):
        ...         self.endpoint = endpoint
        ...
        ...     def fetch_storage_options(self):
        ...         return {"expires_at_millis": "1234567890000"}
        ...
        ...     def provider_id(self):
        ...         return f"MyProvider {{ endpoint: {self.endpoint!r} }}"
        """
        return f"{self.__class__.__name__} {{ repr: {str(self)!r} }}"
