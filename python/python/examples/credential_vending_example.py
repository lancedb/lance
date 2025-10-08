#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Example: Using Credential Vending with Lance Datasets

This example demonstrates two approaches to credential vending:
1. Custom Python implementation
2. Using LanceNamespace with automatic credential vending

Credential vending allows Lance to automatically refresh cloud storage
credentials before they expire, enabling long-running operations on
datasets stored in S3, Azure, or GCS.
"""

import time
from typing import Dict, Any


# ============================================================================
# Example 1: Custom Python Credential Vendor
# ============================================================================

class CustomCredentialVendor:
    """
    Custom credential vendor implementation.

    This example shows how to implement a credential vendor that fetches
    credentials from any source (AWS STS, HashiCorp Vault, custom service, etc.)
    """

    def __init__(self, credentials_service_url: str):
        self.service_url = credentials_service_url
        self.refresh_count = 0

    def get_credentials(self) -> Dict[str, Any]:
        """
        Fetch fresh credentials from your credential service.

        Returns:
            dict with 'storage_options' and 'expires_at_millis' keys
        """
        self.refresh_count += 1

        # In a real implementation, you would:
        # 1. Call your credential service API
        # 2. Parse the response
        # 3. Return the credentials with expiration time

        # Example: Simulating AWS STS AssumeRole response
        credentials = self._fetch_from_service()

        return {
            "storage_options": {
                "aws_access_key_id": credentials["AccessKeyId"],
                "aws_secret_access_key": credentials["SecretAccessKey"],
                "aws_session_token": credentials["SessionToken"],
                "expires_at_millis": str(credentials["ExpiresAtMillis"]),
            },
            "expires_at_millis": credentials["ExpiresAtMillis"],
        }

    def _fetch_from_service(self) -> Dict[str, str]:
        """Simulate fetching credentials from a service."""
        # In real code, this would make an HTTP request
        # For example:
        # import requests
        # response = requests.post(self.service_url, json={"role": "data-reader"})
        # return response.json()

        # Mock credentials (valid for 1 hour)
        expires_at = int((time.time() + 3600) * 1000)
        return {
            "AccessKeyId": f"ASIA_MOCK_{self.refresh_count}",
            "SecretAccessKey": "mock_secret_key",
            "SessionToken": "mock_session_token",
            "ExpiresAtMillis": expires_at,
        }


def example_custom_vendor():
    """Example: Using a custom credential vendor with Lance."""
    import lance

    print("=" * 70)
    print("Example 1: Custom Python Credential Vendor")
    print("=" * 70)

    # Create your custom vendor
    vendor = CustomCredentialVendor(
        credentials_service_url="https://my-creds-service.com/get-credentials"
    )

    # Wrap it for Lance
    lance_vendor = lance._lib._CredentialVendor(vendor)

    # Use it with a dataset
    # Note: This would be used with real S3/cloud URIs
    # dataset = lance.dataset(
    #     "s3://my-bucket/my-dataset.lance",
    #     credential_vendor=lance_vendor
    # )

    print(f"✓ Custom vendor created")
    print(f"  Service URL: {vendor.service_url}")
    print(f"  Refresh count: {vendor.refresh_count}")
    print()

    # Test credential fetching
    creds = vendor.get_credentials()
    print("✓ Credentials fetched:")
    print(f"  Access Key: {creds['storage_options']['aws_access_key_id']}")
    print(f"  Expires: {time.ctime(creds['expires_at_millis'] / 1000)}")
    print()


# ============================================================================
# Example 2: Using LanceNamespace with Credential Vending
# ============================================================================

def example_namespace_vendor():
    """
    Example: Using LanceNamespace for automatic credential vending.

    This demonstrates both directory-based and REST-based namespace services.
    """
    import lance
    import tempfile
    import os

    print("=" * 70)
    print("Example 2: LanceNamespace Credential Vendor")
    print("=" * 70)

    # Example 2a: Directory-based namespace (for testing/local use)
    print("\nExample 2a: Directory Namespace (Local/Testing)")
    print("-" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Connect to directory-based namespace
        namespace = lance._lib.connect_namespace("dir", {
            "path": tmpdir
        })

        print(f"✓ Connected to directory namespace at: {tmpdir}")
        print(f"  Namespace type: {type(namespace)}")
        print()

        # In a real scenario, you would have registered tables in the namespace
        # For demonstration purposes:
        print("  Usage pattern:")
        print("""
        # After registering a table in the namespace:
        table_info = namespace.describe_table(
            table_id=["workspace", "table_name"],
            version=None
        )

        # table_info contains:
        # - location: S3/file path to the dataset
        # - storage_options: credentials with expires_at_millis
        # - version: table version
        """)

    # Example 2b: REST namespace (production use)
    print("\nExample 2b: REST Namespace (Production)")
    print("-" * 70)
    print("""
    # Connect to LanceDB Cloud or compatible REST service
    namespace = lance._lib.connect_namespace("rest", {
        "url": "https://api.lancedb.com",
        "api_key": "your-api-key"
    })

    # Describe a table to get its location and credentials
    table_info = namespace.describe_table(
        table_id=["my_workspace", "my_table"],
        version=None  # or specific version number
    )

    # table_info = {
    #     "location": "s3://bucket/path/table.lance",
    #     "storage_options": {
    #         "aws_access_key_id": "ASIA...",
    #         "aws_secret_access_key": "...",
    #         "aws_session_token": "...",
    #         "expires_at_millis": "1234567890000"
    #     },
    #     "version": 42
    # }
    """)

    # Example 2c: Automatic integration with Dataset
    print("\nExample 2c: Automatic Credential Vending Integration")
    print("-" * 70)
    print("""
    # The simplest way - from_namespace() handles everything
    from lance.io.object_store import LanceNamespaceCredentialVendor

    namespace = lance._lib.connect_namespace("rest", {...})

    # Create credential vendor from namespace
    vendor = LanceNamespaceCredentialVendor(
        namespace=namespace,
        table_id=["workspace", "table"]
    )

    # Wrap for Lance
    lance_vendor = lance._lib._CredentialVendor(vendor)

    # Use with dataset - credentials auto-refresh!
    dataset = lance.dataset(
        "s3://bucket/table.lance",
        credential_vendor=lance_vendor
    )
    """)

    print()


# ============================================================================
# Example 3: Advanced Use Cases
# ============================================================================

class MultiAccountCredentialVendor:
    """
    Advanced example: Vendor that handles multiple AWS accounts.

    This shows how you might implement cross-account access or
    environment-specific credential management.
    """

    def __init__(self, account_mapping: Dict[str, str]):
        """
        Args:
            account_mapping: Maps dataset prefixes to AWS role ARNs
        """
        self.account_mapping = account_mapping
        self.current_account = None

    def get_credentials(self) -> Dict[str, Any]:
        """Fetch credentials for the current account."""
        # Determine which account based on dataset path
        # In real implementation, you'd use boto3.sts.assume_role()

        expires_at = int((time.time() + 3600) * 1000)
        return {
            "storage_options": {
                "aws_access_key_id": "ASIA_CROSS_ACCOUNT",
                "aws_secret_access_key": "secret",
                "aws_session_token": "token",
                "expires_at_millis": str(expires_at),
            },
            "expires_at_millis": expires_at,
        }


def example_advanced_use_cases():
    """Show advanced credential vending patterns."""
    print("=" * 70)
    print("Example 3: Advanced Use Cases")
    print("=" * 70)

    # Multi-account scenario
    vendor = MultiAccountCredentialVendor({
        "s3://prod-bucket": "arn:aws:iam::123456789012:role/ProdDataReader",
        "s3://dev-bucket": "arn:aws:iam::987654321098:role/DevDataReader",
    })

    print("✓ Multi-account vendor created")
    print("  Supports cross-account access patterns")
    print()

    # Custom refresh timing
    print("Custom refresh timing:")
    print("""
    from lance.io.object_store import CredentialVendingParams

    params = CredentialVendingParams(
        refresh_lead_time_ms=600000,  # Refresh 10 minutes before expiry
    )

    dataset = DatasetBuilder.from_namespace(
        namespace=namespace,
        table_id=["workspace", "table"],
        params=params
    ).load()
    """)
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("*" * 70)
    print("Lance Credential Vending Examples")
    print("*" * 70)
    print()

    # Run examples
    example_custom_vendor()
    example_namespace_vendor()
    example_advanced_use_cases()

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
    Key Takeaways:

    1. Custom Vendors: Implement get_credentials() to fetch from any source
    2. Namespace Vendors: Use LanceNamespace for automatic credential management
    3. Automatic Refresh: Lance refreshes credentials before they expire
    4. Flexible Configuration: Customize refresh timing and initial credentials

    Use Cases:
    - Long-running queries on S3/cloud data
    - Cross-account access in AWS
    - Integration with corporate credential services
    - LanceDB Cloud integration

    For more information:
    - Documentation: https://lancedb.github.io/lance/
    - GitHub: https://github.com/lancedb/lance
    """)
    print()
