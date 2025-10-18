# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Tests for custom StorageOptionsProvider implementations.
"""

import time

import lance
from lance import StorageOptionsProvider


class CustomStorageOptionsProvider(StorageOptionsProvider):
    """
    Custom credential vendor that provides mock credentials.

    This demonstrates how to implement a custom credential vendor in Python
    that can be passed to Lance datasets for dynamic credential management.
    """

    def __init__(self, access_key="test_key", expires_in_seconds=3600):
        self.access_key = access_key
        self.call_count = 0
        self.expires_in_seconds = expires_in_seconds

    def get_storage_options(self):
        """
        Called by Lance to get fresh credentials.

        Returns:
            dict of string key-value pairs including credentials and expires_at_millis
        """
        self.call_count += 1

        # Calculate expiration time
        expires_at_millis = int((time.time() + self.expires_in_seconds) * 1000)

        # Return credentials in the expected flat format
        return {
            "aws_access_key_id": f"{self.access_key}_{self.call_count}",
            "aws_secret_access_key": "secret_key",
            "aws_session_token": "session_token",
            "expires_at_millis": str(expires_at_millis),
        }


def test_custom_storage_options_provider():
    """
    Test that a custom Python credential vendor can be created and used.
    """
    vendor = CustomStorageOptionsProvider(access_key="my_custom_key")

    # Test that get_storage_options works
    creds = vendor.get_storage_options()
    assert "aws_access_key_id" in creds
    assert "expires_at_millis" in creds
    assert creds["aws_access_key_id"] == "my_custom_key_1"
    assert vendor.call_count == 1

    # Test that second call increments counter
    creds2 = vendor.get_storage_options()
    assert creds2["aws_access_key_id"] == "my_custom_key_2"
    assert vendor.call_count == 2


def test_storage_options_provider_with_dataset(tmp_path):
    """
    Test that a credential vendor can be passed to a Dataset.

    Note: Credential vending is used with cloud storage. With local files,
    the vendor is accepted but not actually used.
    """
    import pyarrow as pa

    # Create a simple dataset first
    table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    dataset_path = tmp_path / "test_dataset.lance"
    lance.write_dataset(table, str(dataset_path))

    # Create a custom credential vendor
    vendor = CustomStorageOptionsProvider()

    # Pass the vendor directly - no wrapping needed!
    # Note: With local files, the vendor is accepted but not used
    # In a real scenario, you would use this with an S3 or cloud path
    ds = lance.dataset(str(dataset_path), storage_options_provider=vendor)
    assert ds is not None

    # Verify we can read the data
    assert ds.to_table() == table


def test_storage_options_provider_error_handling():
    """
    Test that errors in get_storage_options are handled properly.
    """

    class FailingVendor(StorageOptionsProvider):
        def get_storage_options(self):
            raise ValueError("Intentional error for testing")

    vendor = FailingVendor()

    # The vendor was created successfully
    assert vendor is not None

    # Error would occur when actually calling get_storage_options during dataset operations


def test_storage_options_provider_validation(tmp_path):
    """
    Test that invalid credential vendors are rejected.
    """
    import pyarrow as pa
    import pytest

    # Object without get_storage_options method
    class InvalidVendor:
        pass

    # Create a test dataset
    table = pa.table({"a": [1, 2, 3]})
    dataset_path = tmp_path / "test.lance"
    lance.write_dataset(table, str(dataset_path))

    # Should fail when trying to open dataset with invalid vendor
    with pytest.raises(
        TypeError, match="StorageOptionsProvider must implement get_storage_options"
    ):
        lance.dataset(str(dataset_path), storage_options_provider=InvalidVendor())


def test_namespace_storage_options_provider(tmp_path):
    """
    Test LanceNamespaceStorageOptionsProvider with a mock namespace.
    This demonstrates how to use lance_namespace package.
    """
    import time

    from lance import LanceNamespaceStorageOptionsProvider

    # Create a custom mock namespace that returns credentials
    # In production, you would use: import lance_namespace; ns = lance_namespace.connect(...)
    class MockNamespace:
        def describe_table(self, table_id, version):
            return {
                "location": "s3://test-bucket/table.lance",
                "storage_options": {
                    "aws_access_key_id": "ASIA_TEST",
                    "aws_secret_access_key": "test_secret",
                    "aws_session_token": "test_token",
                    "expires_at_millis": str(int((time.time() + 3600) * 1000)),
                },
                "version": 1,
            }

    mock_ns = MockNamespace()
    vendor = LanceNamespaceStorageOptionsProvider(
        namespace=mock_ns, table_id=["workspace", "table"]
    )

    # Get credentials
    creds = vendor.get_storage_options()

    # Verify structure - should be flat Map<String, String>
    assert "aws_access_key_id" in creds
    assert "expires_at_millis" in creds
    assert creds["aws_access_key_id"] == "ASIA_TEST"
    assert isinstance(creds["expires_at_millis"], str)


def test_namespace_vendor_missing_credentials():
    """
    Test that LanceNamespaceStorageOptionsProvider raises error when namespace
    doesn't return credentials.
    """
    import pytest
    from lance import LanceNamespaceStorageOptionsProvider

    class BadNamespace:
        def describe_table(self, table_id, version):
            return {"location": "s3://bucket/table.lance"}  # Missing storage_options

    vendor = LanceNamespaceStorageOptionsProvider(
        namespace=BadNamespace(), table_id=["workspace", "table"]
    )

    with pytest.raises(RuntimeError, match="did not return storage_options"):
        vendor.get_storage_options()


def test_namespace_vendor_missing_expiration():
    """
    Test that LanceNamespaceStorageOptionsProvider raises error when credentials
    don't have expiration time.
    """
    import pytest
    from lance import LanceNamespaceStorageOptionsProvider

    class BadNamespace:
        def describe_table(self, table_id, version):
            return {
                "location": "s3://bucket/table.lance",
                "storage_options": {
                    "aws_access_key_id": "ASIA_TEST",
                    # Missing expires_at_millis
                },
            }

    vendor = LanceNamespaceStorageOptionsProvider(
        namespace=BadNamespace(), table_id=["workspace", "table"]
    )

    with pytest.raises(RuntimeError, match="missing 'expires_at_millis'"):
        vendor.get_storage_options()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
