# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import pyarrow as pa
import pytest

import lance

ray = pytest.importorskip("ray")


from lance.ray.sink import LanceDatasink  # noqa: E402


def test_ray_sink(tmp_path: Path):
    ray.init()
    schema = pa.schema([pa.field("id", pa.int64()), pa.field("str", pa.string())])

    sink = LanceDatasink(tmp_path, schema=schema)
    ray.data.range(10).map(
        lambda x: {"id": x["id"], "str": f"str-{x['id']}"}
    ).write_datasink(sink)

    ds = lance.dataset(tmp_path)
    ds.count_rows() == 10
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
