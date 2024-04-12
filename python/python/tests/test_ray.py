# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import pyarrow as pa
import pytest

ray = pytest.importorskip("ray")


from lance.ray.sink import (  # noqa: E402
    LanceCommitter,
    LanceDatasink,
    LanceFragmentWriter,
)

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
