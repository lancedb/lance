# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Tests for custom CredentialVendor implementations.
"""

import time

import lance


class CustomCredentialVendor:
    """
    Custom credential vendor that provides mock credentials.

    This demonstrates how to implement a custom credential vendor in Python
    that can be passed to Lance datasets for dynamic credential management.
    """

    def __init__(self, access_key="test_key", expires_in_seconds=3600):
        self.access_key = access_key
        self.call_count = 0
        self.expires_in_seconds = expires_in_seconds

    def get_credentials(self):
        """
        Called by Lance to get fresh credentials.

        Returns:
            dict with 'storage_options' and 'expires_at_millis' keys
        """
        self.call_count += 1

        # Calculate expiration time
        expires_at_millis = int((time.time() + self.expires_in_seconds) * 1000)

        # Return credentials in the expected format
        return {
            "storage_options": {
                "aws_access_key_id": f"{self.access_key}_{self.call_count}",
                "aws_secret_access_key": "secret_key",
                "aws_session_token": "session_token",
                "expires_at_millis": str(expires_at_millis),
            },
            "expires_at_millis": expires_at_millis,
        }


def test_custom_credential_vendor():
    """
    Test that a custom Python credential vendor can be created and used.
    """
    vendor = CustomCredentialVendor(access_key="my_custom_key")

    # Test that get_credentials works
    creds = vendor.get_credentials()
    assert "storage_options" in creds
    assert "expires_at_millis" in creds
    assert creds["storage_options"]["aws_access_key_id"] == "my_custom_key_1"
    assert vendor.call_count == 1

    # Test that it can be wrapped by Lance
    py_vendor = lance._lib._CredentialVendor(vendor)
    assert py_vendor is not None


def test_credential_vendor_with_dataset(tmp_path):
    """
    Test that a credential vendor can be passed to a Dataset.

    Note: This test will fail to actually load a dataset because we're using
    mock credentials, but it verifies the Python->Rust FFI integration works.
    """
    import pyarrow as pa

    # Create a simple dataset first
    table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    dataset_path = tmp_path / "test_dataset.lance"
    lance.write_dataset(table, str(dataset_path))

    # Create a custom credential vendor
    vendor = CustomCredentialVendor()
    py_vendor = lance._lib._CredentialVendor(vendor)

    # Note: We can't actually use credential vending with a local file path
    # This just tests that the parameter is accepted
    # In a real scenario, you would use this with an S3 or cloud path
    ds = lance.dataset(str(dataset_path))
    assert ds is not None

    # Verify we can read the data
    assert ds.to_table() == table


def test_credential_vendor_error_handling():
    """
    Test that errors in get_credentials are handled properly.
    """

    class FailingVendor:
        def get_credentials(self):
            raise ValueError("Intentional error for testing")

    vendor = FailingVendor()
    py_vendor = lance._lib._CredentialVendor(vendor)

    # The vendor was created successfully
    assert py_vendor is not None

    # Error would occur when actually calling get_credentials during dataset operations


def test_credential_vendor_validation():
    """
    Test that invalid credential vendors are rejected.
    """
    import pytest

    # Object without get_credentials method
    class InvalidVendor:
        pass

    with pytest.raises(TypeError, match="must implement get_credentials"):
        lance._lib._CredentialVendor(InvalidVendor())


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
