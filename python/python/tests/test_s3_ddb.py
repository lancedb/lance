#  Copyright (c) 2023. Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""
Integration tests with S3 and DynamoDB.

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
        # Delete all objects first
        for obj in s3.list_objects(Bucket=bucket_name).get("Contents", []):
            s3.delete_object(Bucket=bucket_name, Key=obj["Key"])
        s3.delete_bucket(Bucket=bucket_name)
    except s3.exceptions.NoSuchBucket:
        pass
    s3.create_bucket(Bucket=bucket_name)
    yield bucket_name
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
):
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

    assert sorted([
        item["a"] for item in lance.dataset(table_dir).to_table().to_pylist()
    ]) == [-1, 0, 1, 2, 3, 4]


@pytest.mark.integration
def test_s3_ddb_concurrent_commit_more_than_five(s3_bucket: str, ddb_table: str):
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
