# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""
Integration tests with S3 and DynamoDB. Also used to test storage_options are
passed correctly.

See DEVELOPMENT.md under heading "Integration Tests" for more information.
"""

import copy
import time
import uuid
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import lance
import pyarrow as pa
import pytest
from lance.dependencies import _RAY_AVAILABLE, ray
from lance.file import LanceFileReader, LanceFileWriter
from lance.fragment import write_fragments

# These are all keys that are accepted by storage_options
CONFIG = {
    "allow_http": "true",
    "aws_access_key_id": "ACCESSKEY",
    "aws_secret_access_key": "SECRETKEY",
    "aws_endpoint": "http://localhost:9000",
    "dynamodb_endpoint": "http://localhost:8000",
    "aws_region": "us-west-2",
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
    bucket_name = "lance-integtest"
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
    for obj in s3.list_objects(Bucket=bucket_name).get("Contents", []):
        s3.delete_object(Bucket=bucket_name, Key=obj["Key"])
    s3.delete_bucket(Bucket=bucket_name)


@pytest.fixture(scope="module")
def ddb_table():
    dynamodb = get_boto3_client("dynamodb", endpoint_url=CONFIG["dynamodb_endpoint"])
    table_name = "lance-integtest"
    # if table exists, delete it
    try:
        dynamodb.delete_table(TableName=table_name)
        # smh dynamodb is async
        time.sleep(0.5)
    except dynamodb.exceptions.ResourceNotFoundException:
        pass
    dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {"AttributeName": "base_uri", "KeyType": "HASH"},
            {"AttributeName": "version", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "base_uri", "AttributeType": "S"},
            {"AttributeName": "version", "AttributeType": "N"},
        ],
        ProvisionedThroughput={"ReadCapacityUnits": 10, "WriteCapacityUnits": 10},
    )

    time.sleep(1)
    yield table_name
    dynamodb.delete_table(TableName=table_name)


@pytest.mark.integration
@pytest.mark.parametrize("use_env", [True, False])
def test_s3_ddb_create_and_append(
    s3_bucket: str, ddb_table: str, use_env: bool, monkeypatch
):
    # Test with and without environment variables, so we can validate it works
    # either way you provide them.
    storage_options = copy.deepcopy(CONFIG)
    if use_env:
        for key, value in storage_options.items():
            monkeypatch.setenv(key.upper(), value)
        storage_options = None

    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    table_name = uuid.uuid4().hex
    table_dir = f"s3+ddb://{s3_bucket}/{table_name}?ddbTableName={ddb_table}"
    ds = lance.write_dataset(table1, table_dir, storage_options=storage_options)
    assert len(ds.versions()) == 1

    table2 = pa.Table.from_pylist([{"a": 100, "b": 2000}])

    # can detect existing dataset
    with pytest.raises(OSError, match="Dataset already exists"):
        lance.write_dataset(table2, table_dir, storage_options=storage_options)

    ds = lance.write_dataset(
        table2, table_dir, mode="append", storage_options=storage_options
    )

    assert len(ds.versions()) == 2
    assert ds.count_rows() == 3

    # can checkout
    ds = ds.checkout_version(1)
    assert ds.count_rows() == 2
    ds = ds.checkout_version(2)
    assert ds.count_rows() == 3

    with pytest.raises(ValueError, match="Not found"):
        ds.checkout_version(3)


@pytest.mark.integration
def test_s3_ddb_concurrent_commit(
    s3_bucket: str,
    ddb_table: str,
    monkeypatch,
):
    for key, value in CONFIG.items():
        monkeypatch.setenv(key.upper(), value)

    table = pa.Table.from_pylist([{"a": -1}])
    table_name = uuid.uuid4().hex
    table_dir = f"s3+ddb://{s3_bucket}/{table_name}?ddbTableName={ddb_table}"
    lance.write_dataset(table, table_dir)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futs = [
            executor.submit(
                lance.write_dataset,
                pa.Table.from_pylist([{"a": i}]),
                table_dir,
                mode="append",
            )
            for i in range(5)
        ]
        # surface any errors -- shouldn't be any
        [result.result() for result in futures.as_completed(futs)]

    assert len(lance.dataset(table_dir).versions()) == 6
    assert lance.dataset(table_dir).count_rows() == 6

    assert sorted(
        [item["a"] for item in lance.dataset(table_dir).to_table().to_pylist()]
    ) == [-1, 0, 1, 2, 3, 4]


@pytest.mark.integration
def test_s3_ddb_concurrent_commit_more_than_five(
    s3_bucket: str, ddb_table: str, monkeypatch
):
    for key, value in CONFIG.items():
        monkeypatch.setenv(key.upper(), value)

    table = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    table_name = uuid.uuid4().hex
    table_dir = f"s3+ddb://{s3_bucket}/{table_name}?ddbTableName={ddb_table}"
    lance.write_dataset(table, table_dir)

    failures = 0
    total_futures = 10

    # force the tests to start at the same time
    barrier = Barrier(10, timeout=5)

    def writh_dataset_with_start_barrier():
        barrier.wait()
        lance.write_dataset(table, table_dir, mode="append")

    with ThreadPoolExecutor(max_workers=6) as executor:
        futs = [
            executor.submit(writh_dataset_with_start_barrier)
            for _ in range(total_futures)
        ]
        for result in futures.as_completed(futs):
            try:
                result.result()
            except:  # noqa: E722,PERF203
                failures += 1

    assert failures > 0

    expected_version = total_futures - failures + 1

    assert len(lance.dataset(table_dir).versions()) == expected_version
    assert lance.dataset(table_dir).count_rows() == expected_version * 2


@pytest.mark.integration
def test_s3_unsafe(s3_bucket: str):
    storage_options = copy.deepcopy(CONFIG)
    del storage_options["dynamodb_endpoint"]

    uri = f"s3://{s3_bucket}/test_unsafe"
    data = pa.table({"x": [1, 2, 3]})
    ds = lance.write_dataset(data, uri, storage_options=storage_options)

    assert len(ds.versions()) == 1
    assert ds.count_rows() == 3
    assert ds.to_table() == data


@pytest.mark.integration
def test_s3_ddb_distributed_commit(s3_bucket: str, ddb_table: str):
    table_name = uuid.uuid4().hex
    table_dir = f"s3+ddb://{s3_bucket}/{table_name}?ddbTableName={ddb_table}"

    schema = pa.schema([pa.field("a", pa.int64())])
    fragments = write_fragments(
        pa.table({"a": pa.array(range(1024))}),
        f"s3+ddb://{s3_bucket}/distributed_commit?ddbTableName={ddb_table}",
        storage_options=CONFIG,
    )
    operation = lance.LanceOperation.Overwrite(schema, fragments)
    ds = lance.LanceDataset.commit(table_dir, operation, storage_options=CONFIG)
    assert ds.count_rows() == 1024


@pytest.mark.integration
@pytest.mark.skipif(not _RAY_AVAILABLE, reason="ray is not available")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_ray_committer(s3_bucket: str, ddb_table: str):
    from lance.ray.sink import write_lance

    table_name = uuid.uuid4().hex
    table_dir = f"s3+ddb://{s3_bucket}/{table_name}?ddbTableName={ddb_table}"

    schema = pa.schema([pa.field("id", pa.int64()), pa.field("str", pa.string())])

    ds = ray.data.range(10).map(lambda x: {"id": x["id"], "str": f"str-{x['id']}"})
    write_lance(ds, table_dir, schema=schema, storage_options=CONFIG)

    ds = lance.dataset(table_dir, storage_options=CONFIG)
    assert ds.count_rows() == 10
    assert ds.schema == schema

    tbl = ds.to_table()
    assert sorted(tbl["id"].to_pylist()) == list(range(10))
    assert set(tbl["str"].to_pylist()) == set([f"str-{i}" for i in range(10)])
    assert len(ds.get_fragments()) == 1


@pytest.mark.integration
def test_file_writer_reader(s3_bucket: str):
    storage_options = copy.deepcopy(CONFIG)
    del storage_options["dynamodb_endpoint"]
    table = pa.table({"a": [1, 2, 3]})
    file_path = f"s3://{s3_bucket}/foo.lance"
    global_buffer_text = "hello"
    global_buffer_bytes = bytes(global_buffer_text, "utf-8")
    with LanceFileWriter(str(file_path), storage_options=storage_options) as writer:
        writer.write_batch(table)
        global_buffer_pos = writer.add_global_buffer(global_buffer_bytes)
    reader = LanceFileReader(str(file_path), storage_options=storage_options)
    assert reader.read_all().to_table() == table
    assert reader.metadata().global_buffers[global_buffer_pos].size == len(
        global_buffer_bytes
    )
    assert (
        bytes(reader.read_global_buffer(global_buffer_pos)).decode()
        == global_buffer_text
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.integration
@pytest.mark.skipif(not _RAY_AVAILABLE, reason="ray is not available")
def test_ray_read_lance(s3_bucket: str):
    storage_options = copy.deepcopy(CONFIG)
    table = pa.table({"a": [1, 2], "b": ["a", "b"]})
    path = f"s3://{s3_bucket}/test_ray_read.lance"
    lance.write_dataset(table, path, storage_options=storage_options)
    ds = ray.data.read_lance(path, storage_options=storage_options, concurrency=1)
    ds.take(1)


@pytest.mark.integration
def test_append_fragment(s3_bucket: str):
    storage_options = copy.deepcopy(CONFIG)
    table = pa.table({"a": [1, 2], "b": ["a", "b"]})
    lance.fragment.LanceFragment.create(
        f"s3://{s3_bucket}/test_append.lance", table, storage_options=storage_options
    )


@pytest.mark.integration
def test_s3_drop(s3_bucket: str):
    storage_options = copy.deepcopy(CONFIG)
    table_name = uuid.uuid4().hex
    tmp_path = f"s3://{s3_bucket}/{table_name}.lance"
    table = pa.table({"x": [0]})
    dataset = lance.write_dataset(table, tmp_path, storage_options=storage_options)
    dataset.validate()
    lance.LanceDataset.drop(tmp_path, storage_options=storage_options)
