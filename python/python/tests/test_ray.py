# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import copy
from pathlib import Path

import lance
import pyarrow as pa
import pytest

ray = pytest.importorskip("ray")


from lance.ray.sink import (  # noqa: E402
    LanceCommitter,
    LanceDatasink,
    LanceFragmentWriter,
    _register_hooks,
)

CONFIG = {
    "allow_http": "true",
    "aws_access_key_id": "ACCESSKEY",
    "aws_secret_access_key": "SECRETKEY",
    "aws_endpoint": "http://localhost:9000",
    "dynamodb_endpoint": "http://localhost:8000",
    "aws_region": "us-west-2",
}

# Use this hook until we have official DataSink in Ray.
_register_hooks()

ray.init()


def test_ray_sink(tmp_path: Path):
    schema = pa.schema([pa.field("id", pa.int64()), pa.field("str", pa.string())])

    sink = LanceDatasink(tmp_path)
    ray.data.range(10).map(
        lambda x: {"id": x["id"], "str": f"str-{x['id']}"}
    ).write_datasink(sink)

    ds = lance.dataset(tmp_path)
    ds.count_rows() == 10
    assert ds.schema.names == schema.names
    # The schema is platform-dependent, because numpy uses int32 on Windows.
    # So we observe the schema that is written and use that.
    schema = ds.schema

    tbl = ds.to_table()
    assert sorted(tbl["id"].to_pylist()) == list(range(10))
    assert set(tbl["str"].to_pylist()) == set([f"str-{i}" for i in range(10)])

    sink = LanceDatasink(tmp_path, mode="append")
    ray.data.range(10).map(
        lambda x: {"id": x["id"] + 10, "str": f"str-{x['id'] + 10}"}
    ).write_datasink(sink)

    ds = lance.dataset(tmp_path)
    ds.count_rows() == 20
    tbl = ds.to_table()
    assert sorted(tbl["id"].to_pylist()) == list(range(20))
    assert set(tbl["str"].to_pylist()) == set([f"str-{i}" for i in range(20)])

    sink = LanceDatasink(tmp_path, schema=schema, mode="overwrite")
    ray.data.range(10).map(
        lambda x: {"id": x["id"], "str": f"str-{x['id']}"}
    ).write_datasink(sink)

    ds = lance.dataset(tmp_path)
    ds.count_rows() == 10
    assert ds.schema == schema


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_ray_committer(tmp_path: Path):
    schema = pa.schema([pa.field("id", pa.int64()), pa.field("str", pa.string())])

    ds = (
        ray.data.range(10)
        .map(lambda x: {"id": x["id"], "str": f"str-{x['id']}"})
        .map_batches(LanceFragmentWriter(tmp_path, schema=schema), batch_size=5)
        .write_datasink(LanceCommitter(tmp_path))
    )

    ds = lance.dataset(tmp_path)
    ds.count_rows() == 10
    assert ds.schema == schema

    tbl = ds.to_table()
    assert sorted(tbl["id"].to_pylist()) == list(range(10))
    assert set(tbl["str"].to_pylist()) == set([f"str-{i}" for i in range(10)])
    assert len(ds.get_fragments()) == 2


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_ray_write_lance(tmp_path: Path):
    schema = pa.schema([pa.field("id", pa.int64()), pa.field("str", pa.string())])

    (
        ray.data.range(10)
        .map(lambda x: {"id": x["id"], "str": f"str-{x['id']}"})
        .write_lance(tmp_path, schema=schema)
    )

    ds = lance.dataset(tmp_path)
    ds.count_rows() == 10
    assert ds.schema == schema

    tbl = ds.to_table()
    assert sorted(tbl["id"].to_pylist()) == list(range(10))
    assert set(tbl["str"].to_pylist()) == set([f"str-{i}" for i in range(10)])


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_ray_empty_write_lance(tmp_path: Path):
    schema = pa.schema([pa.field("id", pa.int64()), pa.field("str", pa.string())])

    (
        ray.data.range(10)
        .filter((lambda row: row["id"] > 10))
        .map(lambda x: {"id": x["id"], "str": f"str-{x['id']}"})
        .write_lance(tmp_path, schema=schema)
    )
    # empty write would not generate dataset.
    with pytest.raises(ValueError):
        lance.dataset(tmp_path)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.integration
def test_ray_read_lance(s3_bucket: str):
    storage_options = copy.deepcopy(CONFIG)
    table = pa.table({"a": [1, 2], "b": ["a", "b"]})
    path = f"s3://{s3_bucket}/test_ray_read.lance"
    lance.write_dataset(table, path, storage_options=storage_options)
    ds = ray.data.read_lance(path, storage_options=storage_options, concurrency=1)
    ds.take(1)
