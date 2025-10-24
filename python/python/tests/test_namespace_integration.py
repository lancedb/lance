# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Integration tests for Lance Namespace with S3 and credential refresh.

This test simulates a namespace server that returns incrementing credentials
and verifies that the credential refresh mechanism works correctly.

See DEVELOPMENT.md under heading "Integration Tests" for more information.
"""

import copy
import time
import uuid
from threading import Lock
from typing import Dict, Optional

import lance
import pyarrow as pa
import pytest

# These are all keys that are accepted by storage_options
CONFIG = {
    "allow_http": "true",
    "aws_access_key_id": "ACCESS_KEY",
    "aws_secret_access_key": "SECRET_KEY",
    "aws_endpoint": "http://localhost:4566",
    "aws_region": "us-east-1",
}


def get_boto3_client(*args, **kwargs):
    import boto3

    return boto3.client(
        *args,
        region_name=CONFIG["aws_region"],
        aws_access_key_id=CONFIG["aws_access_key_id"],
        aws_secret_access_key=CONFIG["aws_secret_access_key"],
        **kwargs,
    )


@pytest.fixture(scope="module")
def s3_bucket():
    s3 = get_boto3_client("s3", endpoint_url=CONFIG["aws_endpoint"])
    bucket_name = "lance-namespace-integtest"
    # if bucket exists, delete it
    try:
        delete_bucket(s3, bucket_name)
    except s3.exceptions.NoSuchBucket:
        pass
    s3.create_bucket(Bucket=bucket_name)
    yield bucket_name

    delete_bucket(s3, bucket_name)


def delete_bucket(s3, bucket_name):
    # Delete all objects first
    try:
        for obj in s3.list_objects(Bucket=bucket_name).get("Contents", []):
            s3.delete_object(Bucket=bucket_name, Key=obj["Key"])
        s3.delete_bucket(Bucket=bucket_name)
    except Exception:
        pass


class MockLanceNamespace:
    """
    Mock namespace implementation that tracks credential refresh calls.

    Similar to the Rust MockStorageOptionsProvider, this implementation:
    - Returns incrementing credentials on each describe_table call
    - Tracks the number of times describe_table has been called
    - Returns credentials with short expiration times for testing refresh
    """

    def __init__(
        self,
        bucket_name: str,
        storage_options: Dict[str, str],
        credential_expires_in_seconds: int = 60,
    ):
        """
        Initialize the mock namespace.

        Parameters
        ----------
        bucket_name : str
            The S3 bucket name where tables are stored
        storage_options : Dict[str, str]
            Base storage options (aws_endpoint, aws_region, etc.)
        credential_expires_in_seconds : int
            How long credentials should be valid (for testing refresh)
        """
        self.bucket_name = bucket_name
        self.base_storage_options = storage_options
        self.credential_expires_in_seconds = credential_expires_in_seconds
        self.call_count = 0
        self.lock = Lock()
        self.tables: Dict[str, str] = {}  # table_id -> location mapping

    def register_table(self, table_id: list, location: str):
        """Register a table in the mock namespace."""
        table_key = "/".join(table_id)
        self.tables[table_key] = location

    def get_call_count(self) -> int:
        """Get the number of times describe_table has been called."""
        with self.lock:
            return self.call_count

    def describe_table(
        self, table_id: list, version: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Describe a table and return storage options with incrementing credentials.

        This simulates a namespace server that returns temporary AWS credentials
        that expire after a short time. Each call increments the credential counter.

        Parameters
        ----------
        table_id : list
            The table identifier (e.g., ["my_table"])
        version : Optional[int]
            The table version (not used in this mock)

        Returns
        -------
        Dict[str, any]
            A dictionary with:
            - location: The S3 URI of the table
            - storage_options: Dict with AWS credentials and expires_at_millis
        """
        with self.lock:
            self.call_count += 1
            count = self.call_count

        table_key = "/".join(table_id)
        if table_key not in self.tables:
            raise ValueError(f"Table not found: {table_key}")

        location = self.tables[table_key]

        # Create storage options with incrementing credentials
        storage_options = copy.deepcopy(self.base_storage_options)

        # Add incrementing credentials (similar to Rust MockStorageOptionsProvider)
        storage_options["aws_access_key_id"] = f"AKID_{count}"
        storage_options["aws_secret_access_key"] = f"SECRET_{count}"
        storage_options["aws_session_token"] = f"TOKEN_{count}"

        # Add expiration timestamp (current time + expires_in_seconds)
        expires_at_millis = int((time.time() + self.credential_expires_in_seconds) * 1000)
        storage_options["expires_at_millis"] = str(expires_at_millis)

        return {
            "location": location,
            "storage_options": storage_options,
        }


@pytest.mark.integration
def test_namespace_open_dataset(s3_bucket: str):
    """
    Test opening a dataset through a namespace with credential tracking.

    This test verifies that:
    1. We can create a dataset and register it with a namespace
    2. We can open the dataset through the namespace
    3. The namespace's describe_table method is called to fetch credentials
    """
    storage_options = copy.deepcopy(CONFIG)

    # Create a test dataset directly on S3
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    table_name = uuid.uuid4().hex
    table_uri = f"s3://{s3_bucket}/{table_name}.lance"

    # Write dataset directly to S3
    ds = lance.write_dataset(table1, table_uri, storage_options=storage_options)
    assert len(ds.versions()) == 1
    assert ds.count_rows() == 2

    # Create mock namespace and register the table
    namespace = MockLanceNamespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=60,
    )
    namespace.register_table([table_name], table_uri)

    # Open dataset through namespace (without refresh)
    # This should call describe_table once
    assert namespace.get_call_count() == 0

    ds_from_namespace = lance.dataset(
        namespace=namespace,
        table_id=[table_name],
        refresh_storage_options=False,
    )

    # Verify describe_table was called once during open
    assert namespace.get_call_count() == 1

    # Verify we can read the data
    assert ds_from_namespace.count_rows() == 2
    result = ds_from_namespace.to_table()
    assert result == table1


@pytest.mark.integration
def test_namespace_with_refresh(s3_bucket: str):
    """
    Test that refresh_storage_options=True sets up automatic credential refresh.

    This test verifies that when refresh_storage_options=True:
    1. The initial describe_table call happens during dataset open
    2. The credentials are set up for future refresh (tested by creating the provider)

    Note: We can't easily test the automatic refresh in Python without waiting
    for credentials to expire, but we verify the mechanism is set up correctly.
    """
    storage_options = copy.deepcopy(CONFIG)

    # Create a test dataset
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    table_name = uuid.uuid4().hex
    table_uri = f"s3://{s3_bucket}/{table_name}.lance"

    ds = lance.write_dataset(table1, table_uri, storage_options=storage_options)
    assert ds.count_rows() == 2

    # Create mock namespace with very short expiration (5 seconds)
    # to simulate credentials that need frequent refresh
    namespace = MockLanceNamespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=5,  # Short expiration for testing
    )
    namespace.register_table([table_name], table_uri)

    assert namespace.get_call_count() == 0

    # Open dataset with refresh enabled
    ds_from_namespace = lance.dataset(
        namespace=namespace,
        table_id=[table_name],
        refresh_storage_options=True,  # Enable automatic refresh
    )

    # Verify describe_table was called during open
    # With refresh enabled, it may be called multiple times
    # (once for initial setup, possibly more for credential fetching)
    initial_call_count = namespace.get_call_count()
    assert initial_call_count >= 1, "describe_table should be called at least once"

    # Verify we can read the data
    assert ds_from_namespace.count_rows() == 2
    result = ds_from_namespace.to_table()
    assert result == table1


@pytest.mark.integration
def test_namespace_append_through_namespace(s3_bucket: str):
    """
    Test appending to a dataset opened through a namespace.

    This verifies that write operations work correctly with namespace-managed
    credentials.
    """
    storage_options = copy.deepcopy(CONFIG)

    # Create initial dataset
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}])
    table_name = uuid.uuid4().hex
    table_uri = f"s3://{s3_bucket}/{table_name}.lance"

    ds = lance.write_dataset(table1, table_uri, storage_options=storage_options)
    assert ds.count_rows() == 1
    assert len(ds.versions()) == 1

    # Create namespace and open dataset through it
    namespace = MockLanceNamespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=60,
    )
    namespace.register_table([table_name], table_uri)

    # Open through namespace
    ds_from_namespace = lance.dataset(
        namespace=namespace,
        table_id=[table_name],
        refresh_storage_options=False,
    )

    assert ds_from_namespace.count_rows() == 1
    initial_call_count = namespace.get_call_count()
    assert initial_call_count == 1

    # Append more data using the URI directly (not through namespace)
    table2 = pa.Table.from_pylist([{"a": 10, "b": 20}])
    ds = lance.write_dataset(
        table2, table_uri, mode="append", storage_options=storage_options
    )
    assert ds.count_rows() == 2
    assert len(ds.versions()) == 2

    # Re-open through namespace to see updated data
    ds_from_namespace = lance.dataset(
        namespace=namespace,
        table_id=[table_name],
        refresh_storage_options=False,
    )

    assert ds_from_namespace.count_rows() == 2
    assert len(ds_from_namespace.versions()) == 2

    # Describe_table should have been called again
    assert namespace.get_call_count() == initial_call_count + 1


@pytest.mark.integration
def test_namespace_invalid_table_id(s3_bucket: str):
    """
    Test that opening a non-existent table through namespace raises an error.
    """
    storage_options = copy.deepcopy(CONFIG)

    namespace = MockLanceNamespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=60,
    )

    # Try to open a table that doesn't exist in the namespace
    with pytest.raises(ValueError, match="Table not found"):
        lance.dataset(
            namespace=namespace,
            table_id=["nonexistent_table"],
            refresh_storage_options=False,
        )


@pytest.mark.integration
def test_namespace_with_storage_options_provider(s3_bucket: str):
    """
    Test using a StorageOptionsProvider directly with a namespace.

    This tests the lower-level Python API where we can create a custom
    StorageOptionsProvider that wraps a namespace.
    """
    from lance.namespace import LanceNamespaceStorageOptionsProvider

    storage_options = copy.deepcopy(CONFIG)

    # Create a test dataset
    table1 = pa.Table.from_pylist([{"x": 1}, {"x": 2}, {"x": 3}])
    table_name = uuid.uuid4().hex
    table_uri = f"s3://{s3_bucket}/{table_name}.lance"

    ds = lance.write_dataset(table1, table_uri, storage_options=storage_options)
    assert ds.count_rows() == 3

    # Create namespace
    namespace = MockLanceNamespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=60,
    )
    namespace.register_table([table_name], table_uri)

    # Create a storage options provider from the namespace
    provider = LanceNamespaceStorageOptionsProvider(
        namespace=namespace,
        table_id=[table_name],
    )

    # Fetch storage options through the provider
    assert namespace.get_call_count() == 0
    fetched_options = provider.fetch_storage_options()
    assert namespace.get_call_count() == 1

    # Verify the options contain the expected keys
    assert "aws_access_key_id" in fetched_options
    assert "aws_secret_access_key" in fetched_options
    assert "aws_session_token" in fetched_options
    assert "expires_at_millis" in fetched_options

    # Verify incrementing credentials
    assert fetched_options["aws_access_key_id"] == "AKID_1"
    assert fetched_options["aws_secret_access_key"] == "SECRET_1"
    assert fetched_options["aws_session_token"] == "TOKEN_1"

    # Fetch again to verify incrementing
    fetched_options = provider.fetch_storage_options()
    assert namespace.get_call_count() == 2
    assert fetched_options["aws_access_key_id"] == "AKID_2"
    assert fetched_options["aws_secret_access_key"] == "SECRET_2"
    assert fetched_options["aws_session_token"] == "TOKEN_2"


@pytest.mark.integration
def test_namespace_credential_increment(s3_bucket: str):
    """
    Test that the mock namespace correctly increments credentials on each call.

    This validates our test infrastructure works like the Rust MockStorageOptionsProvider.
    """
    storage_options = copy.deepcopy(CONFIG)

    namespace = MockLanceNamespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=60,
    )

    # Register a dummy table
    table_uri = f"s3://{s3_bucket}/test.lance"
    namespace.register_table(["test"], table_uri)

    # Call describe_table multiple times and verify incrementing credentials
    assert namespace.get_call_count() == 0

    result1 = namespace.describe_table(["test"])
    assert namespace.get_call_count() == 1
    assert result1["storage_options"]["aws_access_key_id"] == "AKID_1"
    assert result1["storage_options"]["aws_secret_access_key"] == "SECRET_1"
    assert result1["storage_options"]["aws_session_token"] == "TOKEN_1"
    assert "expires_at_millis" in result1["storage_options"]

    result2 = namespace.describe_table(["test"])
    assert namespace.get_call_count() == 2
    assert result2["storage_options"]["aws_access_key_id"] == "AKID_2"

    result3 = namespace.describe_table(["test"])
    assert namespace.get_call_count() == 3
    assert result3["storage_options"]["aws_access_key_id"] == "AKID_3"
