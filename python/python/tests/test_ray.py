# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import pandas as pd
import pyarrow as pa
import pytest

ray = pytest.importorskip("ray")

from lance.ray.distribute_task import DistributeCustomTasks  # noqa: E402
from lance.ray.in_place_api import add_columns  # noqa: E402
from lance.ray.sink import (  # noqa: E402
    LanceCommitter,
    LanceDatasink,
    LanceFragmentWriter,
    _register_hooks,
)
from ray.data._internal.datasource.lance_datasource import LanceDatasource  # noqa: E402

# Use this hook until we have official DataSink in Ray.
_register_hooks()

ray.init(ignore_reinit_error=True)


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
def test_ray_write_lance_none_str(tmp_path: Path):
    def f(row):
        return {
            "id": row["id"],
            "str": None,
        }

    schema = pa.schema([pa.field("id", pa.int64()), pa.field("str", pa.string())])
    (ray.data.range(10).map(f).write_lance(tmp_path, schema=schema))

    ds = lance.dataset(tmp_path)
    ds.count_rows() == 10
    assert ds.schema == schema

    tbl = ds.to_table()
    pylist = tbl["str"].to_pylist()
    assert len(pylist) == 10
    for item in pylist:
        assert item is None


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_ray_write_lance_none_str_datasink(tmp_path: Path):
    def f(row):
        return {
            "id": row["id"],
            "str": None,
        }

    schema = pa.schema([pa.field("id", pa.int64()), pa.field("str", pa.string())])

    sink = LanceDatasink(tmp_path, schema=schema)
    (ray.data.range(10).map(f).write_datasink(sink))
    ds = lance.dataset(tmp_path)
    ds.count_rows() == 10
    assert ds.schema == schema

    tbl = ds.to_table()
    pylist = tbl["str"].to_pylist()
    assert len(pylist) == 10
    for item in pylist:
        assert item is None


def generate_label(batch: pa.RecordBatch) -> pa.RecordBatch:
    heights = batch.column("height").to_pylist()
    tags = ["big" if height > 5 else "small" for height in heights]
    df = pd.DataFrame({"size_labels": tags})

    return pa.RecordBatch.from_pandas(
        df, schema=pa.schema([pa.field("size_labels", pa.string())])
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.skip(
    reason="This test local can run, but not in CI." "it's blocked by ray env"
)
def test_lance_parallel_merge_columns(tmp_path: Path):
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("height", pa.int64()),
            pa.field("weight", pa.int64()),
        ]
    )
    (
        ray.data.range(11)
        .repartition(1)
        .map(lambda x: {"id": x["id"], "height": (x["id"] + 5), "weight": x["id"]})
        .write_lance(tmp_path, schema=schema)
    )
    lance_ds = LanceDatasource(uri=tmp_path)
    add_columns(DistributeCustomTasks(lance_ds), generate_label, ["height"])
    ds = lance.dataset(tmp_path)
    tbl = ds.to_table()
    size_labels = sorted(tbl.column("size_labels").to_pylist())
    assert size_labels[:5] == ["big"] * 5
    assert size_labels[6:] == ["small"] * 5
