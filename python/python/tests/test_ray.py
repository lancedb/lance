# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pytest

ray = pytest.importorskip("ray")


from lance.ray import (  # noqa: E402
    LanceCommitter,
    LanceDatasink,
    LanceFragmentWriter,
    _register_hooks,
    merge_columns,
)

# Use this hook until we have offical DataSink in Ray.
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


def test_ray_merge_column(tmp_path: Path):
    schema = pa.schema([pa.field("id", pa.int64()), pa.field("str", pa.string())])

    (
        ray.data.range(10)
        .map(lambda x: {"id": x["id"], "str": f"str-{x['id']}"})
        .write_lance(tmp_path, schema=schema)
    )

    def value_func(batch):
        arrs = pc.add(batch["id"], 2)
        return pa.RecordBatch.from_arrays([arrs], ["sum"])

    merge_columns(tmp_path, value_func)

    ds = lance.dataset(tmp_path)
    schema = ds.schema
    assert schema.names == ["id", "str", "sum"]

    tbl = ds.to_table()
    assert set(tbl["sum"].to_pylist()) == set(range(2, 12))
    # Only bumped 1 version.
    assert ds.version == 2
