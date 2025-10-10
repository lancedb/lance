# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""I/O utilities for Lance datasets.

This module provides utilities for customizing how Lance datasets interact with
cloud storage, including credential management for long-running operations.
"""

from abc import ABC, abstractmethod
from typing import Dict


class CredentialVendor(ABC):
    """Abstract base class for providing storage credentials to Lance datasets.

    Credential vendors enable automatic credential refresh for long-running operations
    on cloud storage (S3, Azure, GCS). Implement this interface to integrate with
    custom credential management systems such as AWS STS, GCP STS, or
    proprietary credential services.

    The vendor is called automatically before credentials expire, ensuring
    uninterrupted access during long-running queries, training jobs, or data processing.

    Example
    -------
    >>> class MyCredentialVendor(CredentialVendor):
    ...     def get_credentials(self):
    ...         # Fetch from your credential service
    ...         return {
    ...             "aws_access_key_id": "ASIA...",
    ...             "aws_secret_access_key": "secret",
    ...             "aws_session_token": "token",
    ...             "expires_at_millis": "1234567890000",
    ...         }
    ...
    >>> vendor = MyCredentialVendor()
    >>> dataset = lance.dataset("s3://bucket/table.lance", credential_vendor=vendor)

    Error Handling
    --------------
    If get_credentials() raises an exception, operations requiring credentials will
    fail. Implementations should handle recoverable errors internally (e.g., retry
    token refresh) and only raise exceptions for unrecoverable errors.
    """

    @abstractmethod
    def get_credentials(self) -> Dict[str, str]:
        """Get fresh storage credentials.

        This method is called automatically before each request and before existing
        credentials expire. It must return credentials in the format below.

        Returns
        -------
        Dict[str, str]
            Dictionary of string key-value pairs containing cloud storage credentials
            and expiration time. Required keys:

            - "expires_at_millis" (str): Unix timestamp in milliseconds (as string) when
              credentials expire. Lance will automatically call get_credentials() again
              before this time.

            Plus provider-specific credential keys:

            AWS S3:
              - "aws_access_key_id" (str): AWS access key
              - "aws_secret_access_key" (str): AWS secret key
              - "aws_session_token" (str, optional): Session token for temporary credentials

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
        >>> def get_credentials(self):
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


class StaticCredentialVendor(CredentialVendor):
    """Example implementation: Vendor that returns static credentials.

    This is useful for testing or when credentials don't expire during the
    dataset's lifetime. For production use with temporary credentials, implement
    a custom vendor that fetches fresh credentials from your service.

    Parameters
    ----------
    credentials : Dict[str, str]
        Dictionary containing storage credentials and expires_at_millis as strings

    Example
    -------
    >>> vendor = StaticCredentialVendor({
    ...     "aws_access_key_id": "ASIA...",
    ...     "aws_secret_access_key": "secret",
    ...     "aws_session_token": "token",
    ...     "expires_at_millis": "1234567890000",
    ... })
    >>> dataset = lance.dataset("s3://bucket/table.lance", credential_vendor=vendor)
    """

    def __init__(self, credentials: Dict[str, str]):
        """Initialize with static credentials.

        Parameters
        ----------
        credentials : Dict[str, str]
            Credential dictionary with all keys as strings, including expires_at_millis
        """
        self._credentials = credentials.copy()

    def get_credentials(self) -> Dict[str, str]:
        """Return the static credentials.

        Returns
        -------
        Dict[str, str]
            Credentials dictionary with string keys and values
        """
        return self._credentials.copy()
