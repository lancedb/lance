import uuid
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import lance
import pyarrow as pa
import pytest


@pytest.mark.integration
def test_s3_ddb_create_and_append(s3_bucket: str, ddb_table: str):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    table_name = uuid.uuid4().hex
    table_dir = f"s3+ddb://{s3_bucket}/{table_name}?ddbTableName={ddb_table}"
    lance.write_dataset(table1, table_dir)
    assert len(lance.dataset(table_dir).versions()) == 1

    table2 = pa.Table.from_pylist([{"a": 100, "b": 2000}])

    # can detect existing dataset
    with pytest.raises(OSError, match="Dataset already exists"):
        lance.write_dataset(table2, table_dir)

    lance.write_dataset(table2, table_dir, mode="append")

    assert len(lance.dataset(table_dir).versions()) == 2
    assert lance.dataset(table_dir).count_rows() == 3

    # can checkout
    assert lance.dataset(table_dir, version=1).count_rows() == 2
    assert lance.dataset(table_dir, version=2).count_rows() == 3

    with pytest.raises(ValueError, match="Not found"):
        lance.dataset(table_dir, version=3)


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

    assert sorted(
        [item["a"] for item in lance.dataset(table_dir).to_table().to_pylist()]
    ) == [-1, 0, 1, 2, 3, 4]


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
