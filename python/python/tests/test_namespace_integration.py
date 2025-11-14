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
from typing import Dict

import lance
import pyarrow as pa
import pytest
from lance_namespace import DescribeTableResponse, LanceNamespace

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


class MockLanceNamespace(LanceNamespace):
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

    def namespace_id(self) -> str:
        """Return a unique identifier for this namespace instance."""
        return "MockLanceNamespace { }"

    def describe_table(self, request) -> DescribeTableResponse:
        """
        Describe a table and return storage options with incrementing credentials.

        This simulates a namespace server that returns temporary AWS credentials
        that expire after a short time. Each call increments the credential counter.

        Parameters
        ----------
        request : DescribeTableRequest
            The describe table request.

        Returns
        -------
        DescribeTableResponse
            Response with location and storage_options
        """
        table_id = request.id

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
        expires_at_millis = int(
            (time.time() + self.credential_expires_in_seconds) * 1000
        )
        storage_options["expires_at_millis"] = str(expires_at_millis)

        return DescribeTableResponse(
            location=location,
            storage_options=storage_options,
        )


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

    # Open dataset through namespace (ignoring storage options from namespace)
    # This should call describe_table once
    assert namespace.get_call_count() == 0

    ds_from_namespace = lance.dataset(
        namespace=namespace,
        table_id=[table_name],
        ignore_namespace_table_storage_options=True,
    )

    # Verify describe_table was called once during open
    assert namespace.get_call_count() == 1

    # Verify we can read the data
    assert ds_from_namespace.count_rows() == 2
    result = ds_from_namespace.to_table()
    assert result == table1


@pytest.mark.integration
def test_namespace_with_refresh(s3_bucket: str):
    storage_options = copy.deepcopy(CONFIG)

    # Create a test dataset
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    table_name = uuid.uuid4().hex
    table_uri = f"s3://{s3_bucket}/{table_name}.lance"

    ds = lance.write_dataset(table1, table_uri, storage_options=storage_options)
    assert ds.count_rows() == 2

    # Create mock namespace with very short expiration (2 seconds)
    # to simulate credentials that need frequent refresh
    namespace = MockLanceNamespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=2,  # Short expiration for testing
    )
    namespace.register_table([table_name], table_uri)

    assert namespace.get_call_count() == 0

    # Open dataset with short refresh offset
    # Storage options from namespace are used by default
    ds_from_namespace = lance.dataset(
        namespace=namespace,
        table_id=[table_name],
        s3_credentials_refresh_offset_seconds=1,
    )

    initial_call_count = namespace.get_call_count()
    assert initial_call_count == 1

    # Verify we can read the data
    assert ds_from_namespace.count_rows() == 2
    result = ds_from_namespace.to_table()
    assert result == table1

    # Record call count after initial reads
    call_count_after_initial_reads = namespace.get_call_count()

    # Wait for credentials to expire
    time.sleep(3)

    # Perform another read operation after expiration
    # This should trigger a credential refresh since credentials have expired
    assert ds_from_namespace.count_rows() == 2
    result2 = ds_from_namespace.to_table()
    assert result2 == table1

    final_call_count = namespace.get_call_count()
    assert final_call_count == call_count_after_initial_reads + 1


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
        ignore_namespace_table_storage_options=True,
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
        ignore_namespace_table_storage_options=True,
    )

    assert ds_from_namespace.count_rows() == 2
    assert len(ds_from_namespace.versions()) == 2

    # Describe_table should have been called again
    assert namespace.get_call_count() == initial_call_count + 1
