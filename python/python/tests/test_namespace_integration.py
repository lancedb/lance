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
from lance.namespace import (
    CreateEmptyTableRequest,
    CreateEmptyTableResponse,
    DeclareTableRequest,
    DeclareTableResponse,
    DescribeTableRequest,
    DescribeTableResponse,
    LanceNamespace,
)

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


class TrackingNamespace(LanceNamespace):
    """Mock namespace that wraps DirectoryNamespace and tracks API calls."""

    def __init__(
        self,
        bucket_name: str,
        storage_options: Dict[str, str],
        credential_expires_in_seconds: int = 60,
    ):
        from lance.namespace import DirectoryNamespace

        self.bucket_name = bucket_name
        self.base_storage_options = storage_options
        self.credential_expires_in_seconds = credential_expires_in_seconds
        self.describe_call_count = 0
        self.create_call_count = 0
        self.lock = Lock()

        # Create underlying DirectoryNamespace with storage options
        dir_props = {f"storage.{k}": v for k, v in storage_options.items()}

        if bucket_name.startswith("/") or bucket_name.startswith("file://"):
            dir_props["root"] = f"{bucket_name}/namespace_root"
        else:
            dir_props["root"] = f"s3://{bucket_name}/namespace_root"

        self.inner = DirectoryNamespace(**dir_props)

    def get_describe_call_count(self) -> int:
        with self.lock:
            return self.describe_call_count

    def get_create_call_count(self) -> int:
        with self.lock:
            return self.create_call_count

    def namespace_id(self) -> str:
        return f"TrackingNamespace {{ inner: {self.inner.namespace_id()} }}"

    def _modify_storage_options(
        self, storage_options: Dict[str, str], count: int
    ) -> Dict[str, str]:
        """Add incrementing credentials with expiration timestamp."""
        modified = copy.deepcopy(storage_options) if storage_options else {}

        modified["aws_access_key_id"] = f"AKID_{count}"
        modified["aws_secret_access_key"] = f"SECRET_{count}"
        modified["aws_session_token"] = f"TOKEN_{count}"
        expires_at_millis = int(
            (time.time() + self.credential_expires_in_seconds) * 1000
        )
        modified["expires_at_millis"] = str(expires_at_millis)
        # Set refresh offset to 1 second (1000ms) for short-lived credential tests
        modified["refresh_offset_millis"] = "1000"

        return modified

    def create_empty_table(
        self, request: CreateEmptyTableRequest
    ) -> CreateEmptyTableResponse:
        with self.lock:
            self.create_call_count += 1
            count = self.create_call_count

        response = self.inner.create_empty_table(request)
        response.storage_options = self._modify_storage_options(
            response.storage_options, count
        )

        return response

    def declare_table(self, request: DeclareTableRequest) -> DeclareTableResponse:
        with self.lock:
            self.create_call_count += 1
            count = self.create_call_count

        response = self.inner.declare_table(request)
        response.storage_options = self._modify_storage_options(
            response.storage_options, count
        )

        return response

    def describe_table(self, request: DescribeTableRequest) -> DescribeTableResponse:
        with self.lock:
            self.describe_call_count += 1
            count = self.describe_call_count

        response = self.inner.describe_table(request)
        response.storage_options = self._modify_storage_options(
            response.storage_options, count
        )

        return response


@pytest.mark.integration
def test_namespace_open_dataset(s3_bucket: str):
    """Test creating and opening datasets through namespace with credential tracking."""
    storage_options = copy.deepcopy(CONFIG)

    namespace = TrackingNamespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=3600,
    )

    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    table_name = uuid.uuid4().hex
    table_id = ["test_ns", table_name]

    assert namespace.get_create_call_count() == 0
    assert namespace.get_describe_call_count() == 0

    ds = lance.write_dataset(
        table1, namespace=namespace, table_id=table_id, mode="create"
    )
    assert len(ds.versions()) == 1
    assert ds.count_rows() == 2
    assert namespace.get_create_call_count() == 1

    ds_from_namespace = lance.dataset(
        namespace=namespace,
        table_id=table_id,
    )

    assert namespace.get_describe_call_count() == 1
    assert ds_from_namespace.count_rows() == 2
    result = ds_from_namespace.to_table()
    assert result == table1

    # Test credential caching
    call_count_before_reads = namespace.get_describe_call_count()
    for _ in range(3):
        assert ds_from_namespace.count_rows() == 2
    assert namespace.get_describe_call_count() == call_count_before_reads


@pytest.mark.integration
def test_namespace_with_refresh(s3_bucket: str):
    """Test credential refresh when credentials expire."""
    storage_options = copy.deepcopy(CONFIG)

    namespace = TrackingNamespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=3,
    )

    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    table_name = uuid.uuid4().hex
    table_id = ["test_ns", table_name]

    assert namespace.get_create_call_count() == 0
    assert namespace.get_describe_call_count() == 0

    ds = lance.write_dataset(
        table1,
        namespace=namespace,
        table_id=table_id,
        mode="create",
    )
    assert ds.count_rows() == 2
    assert namespace.get_create_call_count() == 1

    ds_from_namespace = lance.dataset(
        namespace=namespace,
        table_id=table_id,
    )

    initial_call_count = namespace.get_describe_call_count()
    assert initial_call_count == 1
    assert ds_from_namespace.count_rows() == 2
    result = ds_from_namespace.to_table()
    assert result == table1

    call_count_after_initial_reads = namespace.get_describe_call_count()

    time.sleep(5)

    assert ds_from_namespace.count_rows() == 2
    result2 = ds_from_namespace.to_table()
    assert result2 == table1

    final_call_count = namespace.get_describe_call_count()
    assert final_call_count == call_count_after_initial_reads + 1


@pytest.mark.integration
def test_namespace_append_through_namespace(s3_bucket: str):
    """Test appending to dataset through namespace."""
    storage_options = copy.deepcopy(CONFIG)

    namespace = TrackingNamespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=3600,
    )

    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}])
    table_name = uuid.uuid4().hex
    table_id = ["test_ns", table_name]

    assert namespace.get_create_call_count() == 0
    assert namespace.get_describe_call_count() == 0

    ds = lance.write_dataset(
        table1, namespace=namespace, table_id=table_id, mode="create"
    )
    assert ds.count_rows() == 1
    assert len(ds.versions()) == 1
    assert namespace.get_create_call_count() == 1
    initial_describe_count = namespace.get_describe_call_count()

    table2 = pa.Table.from_pylist([{"a": 10, "b": 20}])
    ds = lance.write_dataset(
        table2, namespace=namespace, table_id=table_id, mode="append"
    )
    assert ds.count_rows() == 2
    assert len(ds.versions()) == 2
    assert namespace.get_create_call_count() == 1
    assert namespace.get_describe_call_count() == initial_describe_count + 1

    ds_from_namespace = lance.dataset(
        namespace=namespace,
        table_id=table_id,
    )

    assert ds_from_namespace.count_rows() == 2
    assert len(ds_from_namespace.versions()) == 2
    assert namespace.get_describe_call_count() == initial_describe_count + 2


@pytest.mark.integration
def test_namespace_write_create_mode(s3_bucket: str):
    """Test writing dataset through namespace in CREATE mode."""
    storage_options = copy.deepcopy(CONFIG)

    namespace = TrackingNamespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=3600,
    )

    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    table_name = uuid.uuid4().hex

    assert namespace.get_create_call_count() == 0
    assert namespace.get_describe_call_count() == 0

    ds = lance.write_dataset(
        table1,
        namespace=namespace,
        table_id=["test_ns", table_name],
        mode="create",
    )

    assert namespace.get_create_call_count() == 1
    assert ds.count_rows() == 2
    assert len(ds.versions()) == 1
    result = ds.to_table()
    assert result == table1


@pytest.mark.integration
def test_namespace_write_append_mode(s3_bucket: str):
    """Test writing dataset through namespace in APPEND mode."""
    storage_options = copy.deepcopy(CONFIG)

    namespace = TrackingNamespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=3600,
    )

    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}])
    table_name = uuid.uuid4().hex
    table_id = ["test_ns", table_name]

    assert namespace.get_create_call_count() == 0
    assert namespace.get_describe_call_count() == 0

    ds = lance.write_dataset(
        table1, namespace=namespace, table_id=table_id, mode="create"
    )
    assert ds.count_rows() == 1
    assert namespace.get_create_call_count() == 1
    assert namespace.get_describe_call_count() == 0

    table2 = pa.Table.from_pylist([{"a": 10, "b": 20}])

    ds = lance.write_dataset(
        table2,
        namespace=namespace,
        table_id=table_id,
        mode="append",
    )

    assert namespace.get_create_call_count() == 1
    describe_count_after_append = namespace.get_describe_call_count()
    assert describe_count_after_append == 1
    assert ds.count_rows() == 2
    assert len(ds.versions()) == 2

    call_count_before_reads = namespace.get_describe_call_count()
    for _ in range(3):
        assert ds.count_rows() == 2
    assert namespace.get_describe_call_count() == call_count_before_reads


@pytest.mark.integration
def test_namespace_write_overwrite_mode(s3_bucket: str):
    """Test writing dataset through namespace in OVERWRITE mode."""
    storage_options = copy.deepcopy(CONFIG)

    namespace = TrackingNamespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=3600,
    )

    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}])
    table_name = uuid.uuid4().hex
    table_id = ["test_ns", table_name]

    assert namespace.get_create_call_count() == 0
    assert namespace.get_describe_call_count() == 0

    ds = lance.write_dataset(
        table1, namespace=namespace, table_id=table_id, mode="create"
    )
    assert ds.count_rows() == 1
    assert namespace.get_create_call_count() == 1
    assert namespace.get_describe_call_count() == 0

    table2 = pa.Table.from_pylist([{"a": 10, "b": 20}, {"a": 100, "b": 200}])

    ds = lance.write_dataset(
        table2,
        namespace=namespace,
        table_id=table_id,
        mode="overwrite",
    )

    assert namespace.get_create_call_count() == 1
    describe_count_after_overwrite = namespace.get_describe_call_count()
    assert describe_count_after_overwrite == 1
    assert ds.count_rows() == 2
    assert len(ds.versions()) == 2
    result = ds.to_table()
    assert result == table2

    call_count_before_reads = namespace.get_describe_call_count()
    for _ in range(3):
        assert ds.count_rows() == 2
    assert namespace.get_describe_call_count() == call_count_before_reads


@pytest.mark.integration
def test_namespace_distributed_write(s3_bucket: str):
    """Test distributed write pattern through namespace."""
    storage_options = copy.deepcopy(CONFIG)

    namespace = TrackingNamespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=3600,
    )

    table_name = uuid.uuid4().hex
    table_id = ["test_ns", table_name]

    request = DeclareTableRequest(id=table_id, location=None)
    response = namespace.declare_table(request)

    assert namespace.get_create_call_count() == 1
    assert namespace.get_describe_call_count() == 0

    table_uri = response.location
    assert table_uri is not None

    from lance.namespace import LanceNamespaceStorageOptionsProvider

    namespace_storage_options = response.storage_options
    assert namespace_storage_options is not None

    storage_options_provider = LanceNamespaceStorageOptionsProvider(
        namespace=namespace, table_id=table_id
    )

    merged_options = dict(storage_options)
    merged_options.update(namespace_storage_options)

    from lance.fragment import write_fragments

    fragment1_data = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    fragment1 = write_fragments(
        fragment1_data,
        table_uri,
        storage_options=merged_options,
        storage_options_provider=storage_options_provider,
    )

    fragment2_data = pa.Table.from_pylist([{"a": 10, "b": 20}, {"a": 30, "b": 40}])
    fragment2 = write_fragments(
        fragment2_data,
        table_uri,
        storage_options=merged_options,
        storage_options_provider=storage_options_provider,
    )

    fragment3_data = pa.Table.from_pylist([{"a": 100, "b": 200}])
    fragment3 = write_fragments(
        fragment3_data,
        table_uri,
        storage_options=merged_options,
        storage_options_provider=storage_options_provider,
    )

    all_fragments = fragment1 + fragment2 + fragment3

    operation = lance.LanceOperation.Overwrite(fragment1_data.schema, all_fragments)

    ds = lance.LanceDataset.commit(
        table_uri,
        operation,
        storage_options=merged_options,
        storage_options_provider=storage_options_provider,
    )

    assert ds.count_rows() == 5
    assert len(ds.versions()) == 1

    result = ds.to_table().sort_by("a")
    expected = pa.Table.from_pylist(
        [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
            {"a": 10, "b": 20},
            {"a": 30, "b": 40},
            {"a": 100, "b": 200},
        ]
    )
    assert result == expected

    ds_from_namespace = lance.dataset(
        namespace=namespace,
        table_id=table_id,
    )
    assert ds_from_namespace.count_rows() == 5


@pytest.mark.integration
def test_file_writer_with_storage_options_provider(s3_bucket: str):
    """Test LanceFileWriter with storage_options_provider and credential refresh."""
    from lance import LanceNamespaceStorageOptionsProvider
    from lance.file import LanceFileReader, LanceFileWriter

    storage_options = copy.deepcopy(CONFIG)

    namespace = TrackingNamespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=3,
    )

    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    table_name = uuid.uuid4().hex
    table_id = ["test_ns", table_name]

    assert namespace.get_create_call_count() == 0
    assert namespace.get_describe_call_count() == 0

    ds = lance.write_dataset(
        table1, namespace=namespace, table_id=table_id, mode="create"
    )
    assert ds.count_rows() == 2
    assert namespace.get_create_call_count() == 1

    describe_response = namespace.describe_table(
        DescribeTableRequest(id=table_id, version=None)
    )
    namespace_storage_options = describe_response.storage_options

    provider = LanceNamespaceStorageOptionsProvider(
        namespace=namespace, table_id=table_id
    )

    initial_describe_count = namespace.get_describe_call_count()

    file_uri = f"s3://{s3_bucket}/{table_name}_file_test.lance"
    schema = pa.schema([pa.field("x", pa.int64()), pa.field("y", pa.int64())])

    writer = LanceFileWriter(
        file_uri,
        schema=schema,
        storage_options=namespace_storage_options,
        storage_options_provider=provider,
    )

    batch = pa.RecordBatch.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6]}, schema=schema)
    writer.write_batch(batch)

    batch2 = pa.RecordBatch.from_pydict(
        {"x": [7, 8, 9], "y": [10, 11, 12]}, schema=schema
    )
    writer.write_batch(batch2)
    writer.close()

    describe_count_after_write = namespace.get_describe_call_count()
    assert describe_count_after_write == initial_describe_count

    reader = LanceFileReader(
        file_uri,
        storage_options=namespace_storage_options,
        storage_options_provider=provider,
    )
    result = reader.read_all(batch_size=1024)
    result_table = result.to_table()
    assert result_table.num_rows == 6
    assert result_table.schema == schema

    expected_table = pa.table(
        {"x": [1, 2, 3, 7, 8, 9], "y": [4, 5, 6, 10, 11, 12]}, schema=schema
    )
    assert result_table == expected_table

    time.sleep(5)

    file_uri2 = f"s3://{s3_bucket}/{table_name}_file_test2.lance"
    writer2 = LanceFileWriter(
        file_uri2,
        schema=schema,
        storage_options=namespace_storage_options,
        storage_options_provider=provider,
    )

    batch3 = pa.RecordBatch.from_pydict(
        {"x": [100, 200], "y": [300, 400]}, schema=schema
    )
    writer2.write_batch(batch3)
    writer2.close()

    final_describe_count = namespace.get_describe_call_count()
    assert final_describe_count == describe_count_after_write + 1

    reader2 = LanceFileReader(
        file_uri2,
        storage_options=namespace_storage_options,
        storage_options_provider=provider,
    )
    result2 = reader2.read_all(batch_size=1024)
    result_table2 = result2.to_table()
    assert result_table2.num_rows == 2
    expected_table2 = pa.table({"x": [100, 200], "y": [300, 400]}, schema=schema)
    assert result_table2 == expected_table2


@pytest.mark.integration
def test_file_reader_with_storage_options_provider(s3_bucket: str):
    """Test LanceFileReader with storage_options_provider and credential refresh."""
    from lance import LanceNamespaceStorageOptionsProvider
    from lance.file import LanceFileReader, LanceFileWriter

    storage_options = copy.deepcopy(CONFIG)

    namespace = TrackingNamespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=3,
    )

    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    table_name = uuid.uuid4().hex
    table_id = ["test_ns", table_name]

    ds = lance.write_dataset(
        table1, namespace=namespace, table_id=table_id, mode="create"
    )
    assert ds.count_rows() == 2

    describe_response = namespace.describe_table(
        DescribeTableRequest(id=table_id, version=None)
    )
    namespace_storage_options = describe_response.storage_options

    provider = LanceNamespaceStorageOptionsProvider(
        namespace=namespace, table_id=table_id
    )

    file_uri = f"s3://{s3_bucket}/{table_name}_file_reader_test.lance"
    schema = pa.schema([pa.field("x", pa.int64()), pa.field("y", pa.int64())])

    # Write a file first (without provider to keep it simple)
    writer = LanceFileWriter(
        file_uri,
        schema=schema,
        storage_options=namespace_storage_options,
    )
    batch = pa.RecordBatch.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6]}, schema=schema)
    writer.write_batch(batch)
    writer.close()

    # Get fresh credentials for reading
    describe_response = namespace.describe_table(
        DescribeTableRequest(id=table_id, version=None)
    )
    namespace_storage_options = describe_response.storage_options

    initial_describe_count = namespace.get_describe_call_count()

    # First read should work without needing refresh
    reader = LanceFileReader(
        file_uri,
        storage_options=namespace_storage_options,
        storage_options_provider=provider,
    )
    result = reader.read_all(batch_size=1024)
    result_table = result.to_table()
    assert result_table.num_rows == 3
    assert result_table.schema == schema

    describe_count_after_first_read = namespace.get_describe_call_count()
    assert describe_count_after_first_read == initial_describe_count

    # Wait for credentials to expire
    time.sleep(5)

    # Write a second file
    file_uri2 = f"s3://{s3_bucket}/{table_name}_file_reader_test2.lance"
    writer2 = LanceFileWriter(
        file_uri2,
        schema=schema,
        storage_options=namespace_storage_options,
    )
    batch2 = pa.RecordBatch.from_pydict(
        {"x": [100, 200], "y": [300, 400]}, schema=schema
    )
    writer2.write_batch(batch2)
    writer2.close()

    # Second read should trigger credential refresh
    reader2 = LanceFileReader(
        file_uri2,
        storage_options=namespace_storage_options,
        storage_options_provider=provider,
    )
    result2 = reader2.read_all(batch_size=1024)
    result_table2 = result2.to_table()
    assert result_table2.num_rows == 2
    expected_table2 = pa.table({"x": [100, 200], "y": [300, 400]}, schema=schema)
    assert result_table2 == expected_table2

    final_describe_count = namespace.get_describe_call_count()
    assert final_describe_count == describe_count_after_first_read + 1


@pytest.mark.integration
def test_file_session_with_storage_options_provider(s3_bucket: str):
    """Test LanceFileSession with storage_options_provider and credential refresh."""
    from lance import LanceNamespaceStorageOptionsProvider
    from lance.file import LanceFileSession

    storage_options = copy.deepcopy(CONFIG)

    namespace = TrackingNamespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=3,
    )

    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    table_name = uuid.uuid4().hex
    table_id = ["test_ns", table_name]

    ds = lance.write_dataset(
        table1, namespace=namespace, table_id=table_id, mode="create"
    )
    assert ds.count_rows() == 2

    describe_response = namespace.describe_table(
        DescribeTableRequest(id=table_id, version=None)
    )
    namespace_storage_options = describe_response.storage_options

    provider = LanceNamespaceStorageOptionsProvider(
        namespace=namespace, table_id=table_id
    )

    initial_describe_count = namespace.get_describe_call_count()

    # Create session with storage_options_provider
    session = LanceFileSession(
        f"s3://{s3_bucket}/{table_name}_session",
        storage_options=namespace_storage_options,
        storage_options_provider=provider,
    )

    # Test contains method
    assert not session.contains("session_test.lance")

    # Test list method
    files = session.list()
    assert isinstance(files, list)

    schema = pa.schema([pa.field("x", pa.int64()), pa.field("y", pa.int64())])

    # Write using session - should not trigger credential refresh
    writer = session.open_writer(
        "session_test.lance",
        schema=schema,
    )
    batch = pa.RecordBatch.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6]}, schema=schema)
    writer.write_batch(batch)
    writer.close()

    describe_count_after_first_write = namespace.get_describe_call_count()
    assert describe_count_after_first_write == initial_describe_count

    # Test contains method after write
    assert session.contains("session_test.lance")

    # Read using session - should not trigger credential refresh
    reader = session.open_reader("session_test.lance")
    result = reader.read_all(batch_size=1024)
    result_table = result.to_table()
    assert result_table.num_rows == 3
    assert result_table.schema == schema

    expected_table = pa.table({"x": [1, 2, 3], "y": [4, 5, 6]}, schema=schema)
    assert result_table == expected_table

    describe_count_after_first_read = namespace.get_describe_call_count()
    assert describe_count_after_first_read == describe_count_after_first_write

    # Wait for credentials to expire
    time.sleep(5)

    # Write again, should trigger credential refresh
    writer2 = session.open_writer(
        "session_test2.lance",
        schema=schema,
    )
    batch2 = pa.RecordBatch.from_pydict(
        {"x": [100, 200], "y": [300, 400]}, schema=schema
    )
    writer2.write_batch(batch2)
    writer2.close()

    describe_count_after_second_write = namespace.get_describe_call_count()
    assert describe_count_after_second_write == describe_count_after_first_read + 1

    # Read the second file - should not trigger another refresh since we just refreshed
    reader2 = session.open_reader("session_test2.lance")
    result2 = reader2.read_all(batch_size=1024)
    result_table2 = result2.to_table()
    assert result_table2.num_rows == 2
    expected_table2 = pa.table({"x": [100, 200], "y": [300, 400]}, schema=schema)
    assert result_table2 == expected_table2

    final_describe_count = namespace.get_describe_call_count()
    assert final_describe_count == describe_count_after_second_write


def create_test_table_data():
    """Create test PyArrow table data for concurrent tests."""
    return pa.Table.from_pylist(
        [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
            {"id": 3, "name": "Charlie", "age": 35},
        ]
    )


def table_to_ipc_bytes(table):
    """Convert PyArrow table to IPC bytes."""
    import io

    sink = io.BytesIO()
    with pa.ipc.RecordBatchStreamWriter(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


@pytest.mark.integration
def test_basic_create_and_drop_on_s3(s3_bucket: str):
    """Test basic create and drop table operations on S3.

    Mirrors Java: testBasicCreateAndDropOnS3
    """
    from lance.namespace import DirectoryNamespace
    from lance_namespace import (
        DropTableRequest,
        TableExistsRequest,
    )

    test_prefix = f"test-{uuid.uuid4().hex[:8]}"
    storage_options = copy.deepcopy(CONFIG)
    dir_props = {f"storage.{k}": v for k, v in storage_options.items()}
    dir_props["root"] = f"s3://{s3_bucket}/{test_prefix}"
    namespace = DirectoryNamespace(**dir_props)

    table_name = "basic_test_table"
    table_data = create_test_table_data()
    table_id = ["test_ns", table_name]

    # Create table using lance.write_dataset (same as other passing tests)
    ds = lance.write_dataset(
        table_data, namespace=namespace, table_id=table_id, mode="create"
    )
    assert ds is not None
    assert ds.count_rows() == 3

    # Drop table
    drop_req = DropTableRequest(id=table_id)
    drop_resp = namespace.drop_table(drop_req)
    assert drop_resp is not None

    # Verify table no longer exists
    exists_req = TableExistsRequest(id=table_id)
    with pytest.raises(Exception):
        namespace.table_exists(exists_req)


@pytest.mark.integration
def test_concurrent_create_and_drop_single_instance_on_s3(s3_bucket: str):
    """Test concurrent create/drop with single namespace instance on S3."""
    import concurrent.futures

    from lance.namespace import DirectoryNamespace
    from lance_namespace import (
        CreateNamespaceRequest,
        CreateTableRequest,
        DropTableRequest,
    )

    test_prefix = f"test-{uuid.uuid4().hex[:8]}"
    storage_options = copy.deepcopy(CONFIG)
    dir_props = {f"storage.{k}": v for k, v in storage_options.items()}
    dir_props["root"] = f"s3://{s3_bucket}/{test_prefix}"
    # Very high retry count to guarantee all operations succeed
    dir_props["commit_retries"] = "2147483647"
    namespace = DirectoryNamespace(**dir_props)

    # Initialize namespace first - create parent namespace to ensure __manifest table
    # is created before concurrent operations
    create_ns_req = CreateNamespaceRequest(id=["test_ns"])
    namespace.create_namespace(create_ns_req)

    num_tables = 10
    success_count = 0
    fail_count = 0
    lock = Lock()

    def create_and_drop_table(table_index):
        nonlocal success_count, fail_count
        try:
            table_name = f"s3_concurrent_table_{table_index}"
            table_data = create_test_table_data()
            table_id = ["test_ns", table_name]
            ipc_data = table_to_ipc_bytes(table_data)

            # Create table using atomic create_table API
            create_req = CreateTableRequest(id=table_id)
            namespace.create_table(create_req, ipc_data)

            # Drop table
            drop_req = DropTableRequest(id=table_id)
            namespace.drop_table(drop_req)

            with lock:
                success_count += 1
        except Exception:
            with lock:
                fail_count += 1
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_tables) as executor:
        futures = [executor.submit(create_and_drop_table, i) for i in range(num_tables)]
        concurrent.futures.wait(futures)

    # All operations must succeed with very high retry count
    assert success_count == num_tables, (
        f"Expected {num_tables} successes, got {success_count}"
    )
    assert fail_count == 0, f"Expected 0 failures, got {fail_count}"


@pytest.mark.integration
def test_concurrent_create_and_drop_multiple_instances_on_s3(s3_bucket: str):
    """Test concurrent create/drop with multiple namespace instances on S3."""
    import concurrent.futures

    from lance.namespace import DirectoryNamespace
    from lance_namespace import (
        CreateNamespaceRequest,
        CreateTableRequest,
        DropTableRequest,
        ListTablesRequest,
    )

    test_prefix = f"test-{uuid.uuid4().hex[:8]}"
    storage_options = copy.deepcopy(CONFIG)
    base_dir_props = {f"storage.{k}": v for k, v in storage_options.items()}
    base_dir_props["root"] = f"s3://{s3_bucket}/{test_prefix}"
    # Very high retry count to guarantee all operations succeed
    base_dir_props["commit_retries"] = "2147483647"

    # Initialize namespace first with a single instance to ensure __manifest
    # table is created and parent namespace exists before concurrent operations
    init_ns = DirectoryNamespace(**base_dir_props.copy())
    create_ns_req = CreateNamespaceRequest(id=["test_ns"])
    init_ns.create_namespace(create_ns_req)

    num_tables = 10
    success_count = 0
    fail_count = 0
    lock = Lock()

    def create_and_drop_table(table_index):
        nonlocal success_count, fail_count
        try:
            # Each thread creates its own namespace instance
            ns = DirectoryNamespace(**base_dir_props.copy())

            table_name = f"s3_multi_ns_table_{table_index}"
            table_data = create_test_table_data()
            table_id = ["test_ns", table_name]
            ipc_data = table_to_ipc_bytes(table_data)

            # Create table using atomic create_table API
            create_req = CreateTableRequest(id=table_id)
            ns.create_table(create_req, ipc_data)

            # Drop table
            drop_req = DropTableRequest(id=table_id)
            ns.drop_table(drop_req)

            with lock:
                success_count += 1
        except Exception:
            with lock:
                fail_count += 1
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_tables) as executor:
        futures = [executor.submit(create_and_drop_table, i) for i in range(num_tables)]
        concurrent.futures.wait(futures)

    # All operations must succeed with very high retry count
    assert success_count == num_tables, (
        f"Expected {num_tables} successes, got {success_count}"
    )
    assert fail_count == 0, f"Expected 0 failures, got {fail_count}"

    # Verify remaining state is consistent (no corruption)
    verify_ns = DirectoryNamespace(**base_dir_props)
    list_req = ListTablesRequest(id=["test_ns"])
    _ = verify_ns.list_tables(list_req)  # Should not error


@pytest.mark.integration
def test_concurrent_create_then_drop_from_different_instance_on_s3(s3_bucket: str):
    """Test creating from one set of instances, dropping from different ones on S3."""
    import concurrent.futures

    from lance.namespace import DirectoryNamespace
    from lance_namespace import (
        CreateNamespaceRequest,
        CreateTableRequest,
        DropTableRequest,
        ListTablesRequest,
    )

    test_prefix = f"test-{uuid.uuid4().hex[:8]}"
    storage_options = copy.deepcopy(CONFIG)
    base_dir_props = {f"storage.{k}": v for k, v in storage_options.items()}
    base_dir_props["root"] = f"s3://{s3_bucket}/{test_prefix}"
    # Very high retry count to guarantee all operations succeed
    base_dir_props["commit_retries"] = "2147483647"

    # Initialize namespace first with a single instance to ensure __manifest
    # table is created and parent namespace exists before concurrent operations
    init_ns = DirectoryNamespace(**base_dir_props.copy())
    create_ns_req = CreateNamespaceRequest(id=["test_ns"])
    init_ns.create_namespace(create_ns_req)

    num_tables = 10

    # Phase 1: Create all tables concurrently using separate namespace instances
    create_success_count = 0
    create_fail_count = 0
    create_lock = Lock()

    def create_table(table_index):
        nonlocal create_success_count, create_fail_count
        table_name = f"s3_cross_instance_table_{table_index}"
        try:
            ns = DirectoryNamespace(**base_dir_props.copy())

            table_data = create_test_table_data()
            table_id = ["test_ns", table_name]
            ipc_data = table_to_ipc_bytes(table_data)

            # Create table using atomic create_table API
            create_req = CreateTableRequest(id=table_id)
            ns.create_table(create_req, ipc_data)

            with create_lock:
                create_success_count += 1
        except Exception:
            with create_lock:
                create_fail_count += 1
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_tables) as executor:
        futures = [executor.submit(create_table, i) for i in range(num_tables)]
        concurrent.futures.wait(futures)

    # All creates must succeed with very high retry count
    assert create_success_count == num_tables, (
        f"Expected {num_tables} create successes, got {create_success_count}"
    )
    assert create_fail_count == 0, (
        f"Expected 0 create failures, got {create_fail_count}"
    )

    # Phase 2: Drop all tables using NEW namespace instances
    drop_success_count = 0
    drop_fail_count = 0
    drop_lock = Lock()

    def drop_table(table_index):
        nonlocal drop_success_count, drop_fail_count
        try:
            ns = DirectoryNamespace(**base_dir_props.copy())

            table_name = f"s3_cross_instance_table_{table_index}"
            table_id = ["test_ns", table_name]

            drop_req = DropTableRequest(id=table_id)
            ns.drop_table(drop_req)

            with drop_lock:
                drop_success_count += 1
        except Exception:
            with drop_lock:
                drop_fail_count += 1
            raise

    # Drop all tables
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_tables) as executor:
        futures = [executor.submit(drop_table, i) for i in range(num_tables)]
        concurrent.futures.wait(futures)

    # All drops must succeed with very high retry count
    assert drop_success_count == num_tables, (
        f"Expected {num_tables} drop successes, got {drop_success_count}"
    )
    assert drop_fail_count == 0, f"Expected 0 drop failures, got {drop_fail_count}"

    # Verify remaining state is consistent (no corruption)
    verify_ns = DirectoryNamespace(**base_dir_props)
    list_req = ListTablesRequest(id=["test_ns"])
    _ = verify_ns.list_tables(list_req)  # Should not error
