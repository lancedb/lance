# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import base64
import contextlib
import os
import pickle
import platform
import random
import re
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List
from unittest import mock

import lance
import lance.fragment
import numpy as np
import pandas as pd
import pandas.testing as tm
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
import pytest
from helper import ProgressForTest
from lance._dataset.sharded_batch_iterator import ShardedBatchIterator
from lance.commit import CommitConflictError
from lance.debug import format_fragment
from lance.util import validate_vector_index

# Various valid inputs for write_dataset
input_schema = pa.schema([pa.field("a", pa.float64()), pa.field("b", pa.int64())])
input_data = [
    # (schema, data)
    (None, pa.table({"a": [1.0, 2.0], "b": [20, 30]})),
    (None, pa.record_batch([[1.0, 2.0], [20, 30]], names=["a", "b"])),
    (None, pd.DataFrame({"a": [1.0, 2.0], "b": [20, 30]})),
    (None, pl.DataFrame({"a": [1.0, 2.0], "b": [20, 30]})),
    (
        input_schema,
        [pa.record_batch([pa.array([1.0, 2.0]), pa.array([20, 30])], names=["a", "b"])],
    ),
    # Can provide an iterator with a schema that is different but cast-able
    (
        input_schema,
        iter(
            pa.table(
                {
                    "a": [1.0, 2.0],
                    "b": pa.array([20, 30], pa.int32()),
                }
            ).to_batches()
        ),
    ),
]


@pytest.mark.parametrize("schema,data", input_data, ids=type)
def test_input_data(tmp_path: Path, schema, data):
    base_dir = tmp_path / "test"
    dataset = lance.write_dataset(data, base_dir, schema=schema)
    assert dataset.to_table() == input_data[0][1]


def test_roundtrip_types(tmp_path: Path):
    table = pa.table(
        {
            "dict": pa.array(["a", "b", "a"], pa.dictionary(pa.int8(), pa.string())),
            # PyArrow doesn't support creating large_string dictionaries easily.
            "large_dict": pa.DictionaryArray.from_arrays(
                pa.array([0, 1, 1], pa.int8()),
                pa.array(["foo", "bar"], pa.large_string()),
            ),
            "list": pa.array(
                [["a", "b"], ["c", "d"], ["e", "f"]], pa.list_(pa.string())
            ),
            "large_list": pa.array(
                [["a", "b"], ["c", "d"], ["e", "f"]], pa.large_list(pa.string())
            ),
        }
    )

    # TODO: V2 does not currently handle large_dict
    dataset = lance.write_dataset(table, tmp_path, data_storage_version="legacy")
    assert dataset.schema == table.schema
    assert dataset.to_table() == table


def test_dataset_overwrite(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    table2 = pa.Table.from_pylist([{"s": "one"}, {"s": "two"}])
    lance.write_dataset(table2, base_dir, mode="overwrite")

    dataset = lance.dataset(base_dir)
    assert dataset.to_table() == table2

    ds_v1 = lance.dataset(base_dir, version=1)
    assert ds_v1.to_table() == table1


def test_dataset_append(tmp_path: Path):
    table = pa.Table.from_pydict({"colA": [1, 2, 3], "colB": [4, 5, 6]})
    base_dir = tmp_path / "test"

    # verify append works even if no dataset existed at the uri
    lance.write_dataset(table, base_dir, mode="append")
    dataset = lance.dataset(base_dir)
    assert dataset.to_table() == table

    # verify appending batches with a different schema doesn't work
    table2 = pa.Table.from_pydict({"COLUMN-C": [1, 2, 3], "colB": [4, 5, 6]})
    with pytest.raises(OSError):
        lance.write_dataset(table2, dataset, mode="append")

    # But we can append subschemas
    table3 = pa.Table.from_pydict({"colA": [4, 5, 6]})
    dataset.insert(table3)  # Append is default
    assert dataset.to_table() == pa.table(
        {"colA": [1, 2, 3, 4, 5, 6], "colB": [4, 5, 6, None, None, None]}
    )


def test_dataset_from_record_batch_iterable(tmp_path: Path):
    base_dir = tmp_path / "test"

    test_pylist = [{"colA": "Alice", "colB": 20}, {"colA": "Blob", "colB": 30}]

    # split into two batches
    batches = [
        pa.RecordBatch.from_pylist([test_pylist[0]]),
        pa.RecordBatch.from_pylist([test_pylist[1]]),
    ]

    # define schema
    schema = pa.schema(
        [
            pa.field("colA", pa.string()),
            pa.field("colB", pa.int64()),
        ]
    )

    # write dataset with iterator
    lance.write_dataset(iter(batches), base_dir, schema)
    dataset = lance.dataset(base_dir)

    # After combined into one batch, make sure it is the same as original pylist
    assert list(dataset.to_batches())[0].to_pylist() == test_pylist

    # write dataset with list
    lance.write_dataset(batches, base_dir, schema, mode="overwrite")

    # After combined into one batch, make sure it is the same as original pylist
    assert list(dataset.to_batches())[0].to_pylist() == test_pylist


def test_versions(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    assert len(lance.dataset(base_dir).versions()) == 1

    table2 = pa.Table.from_pylist([{"s": "one"}, {"s": "two"}])
    time.sleep(1)
    lance.write_dataset(table2, base_dir, mode="overwrite")

    assert len(lance.dataset(base_dir).versions()) == 2

    v1, v2 = lance.dataset(base_dir).versions()
    assert v1["version"] == 1
    assert v2["version"] == 2
    assert isinstance(v1["timestamp"], datetime)
    assert isinstance(v2["timestamp"], datetime)
    assert v1["timestamp"] < v2["timestamp"]
    assert isinstance(v1["metadata"], dict)
    assert isinstance(v2["metadata"], dict)


def test_version_id(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    original_ds = lance.write_dataset(table1, base_dir)

    assert original_ds.version == 1
    assert original_ds.latest_version == 1

    table2 = pa.Table.from_pylist([{"s": "one"}, {"s": "two"}])
    time.sleep(1)
    updated_ds = lance.write_dataset(table2, base_dir, mode="overwrite")

    assert original_ds.version == 1
    assert original_ds.latest_version == 2

    assert updated_ds.version == 2
    assert updated_ds.latest_version == 2


def test_checkout(tmp_path: Path):
    tab = pa.table({"a": range(3)})
    ds1 = lance.write_dataset(tab, tmp_path)
    ds1.delete("a = 1")

    ds2 = ds1.checkout_version(1)

    assert ds2.version == 1
    assert ds2.to_table() == tab

    assert ds1.version == 2
    assert ds1.to_table() == pa.table({"a": [0, 2]})

    with pytest.raises(IOError):
        ds2.delete("a = 2")

    ds1.delete("a = 2")
    assert ds1.count_rows() == 1

    assert ds2.checkout_version(ds2.latest_version).version == ds1.version


def test_asof_checkout(tmp_path: Path):
    table = pa.Table.from_pydict({"colA": [1, 2, 3], "colB": [4, 5, 6]})
    base_dir = tmp_path / "test"

    lance.write_dataset(table, base_dir)
    assert len(lance.dataset(base_dir).versions()) == 1
    time.sleep(0.1)
    ts_1 = datetime.now()
    time.sleep(0.1)

    lance.write_dataset(table, base_dir, mode="append")
    assert len(lance.dataset(base_dir).versions()) == 2
    time.sleep(0.1)
    ts_2 = datetime.now()
    time.sleep(0.1)

    lance.write_dataset(table, base_dir, mode="append")
    assert len(lance.dataset(base_dir).versions()) == 3
    time.sleep(0.1)
    ts_3 = datetime.now()

    # check that only the first batch is present
    ds = lance.dataset(base_dir, asof=ts_1)
    assert ds.version == 1
    assert len(ds.to_table()) == 3

    # check that the first and second batch are present
    ds = lance.dataset(base_dir, asof=ts_2)
    assert ds.version == 2
    assert len(ds.to_table()) == 6

    # check that all batches are present
    ds = lance.dataset(base_dir, asof=ts_3)
    assert ds.version == 3
    assert len(ds.to_table()) == 9


def test_v2_manifest_paths(tmp_path: Path):
    lance.write_dataset(
        pa.table({"a": range(100)}), tmp_path, enable_v2_manifest_paths=True
    )
    manifest_path = os.listdir(tmp_path / "_versions")
    assert len(manifest_path) == 1
    assert re.match(r"\d{20}\.manifest", manifest_path[0])


def test_v2_manifest_paths_migration(tmp_path: Path):
    # Create a dataset with v1 manifest paths
    lance.write_dataset(
        pa.table({"a": range(100)}), tmp_path, enable_v2_manifest_paths=False
    )
    manifest_path = os.listdir(tmp_path / "_versions")
    assert manifest_path == ["1.manifest"]

    # Migrate to v2 manifest paths
    lance.dataset(tmp_path).migrate_manifest_paths_v2()
    manifest_path = os.listdir(tmp_path / "_versions")
    assert len(manifest_path) == 1
    assert re.match(r"\d{20}\.manifest", manifest_path[0])


def test_tag(tmp_path: Path):
    table = pa.Table.from_pydict({"colA": [1, 2, 3], "colB": [4, 5, 6]})
    base_dir = tmp_path / "test"

    lance.write_dataset(table, base_dir)
    ds = lance.write_dataset(table, base_dir, mode="append")

    assert len(ds.tags.list()) == 0

    with pytest.raises(ValueError):
        ds.tags.create("tag1", 3)

    with pytest.raises(ValueError):
        ds.tags.delete("tag1")

    ds.tags.create("tag1", 1)
    assert len(ds.tags.list()) == 1

    with pytest.raises(ValueError):
        ds.tags.create("tag1", 1)

    ds.tags.delete("tag1")

    ds.tags.create("tag1", 1)
    ds.tags.create("tag2", 1)

    assert len(ds.tags.list()) == 2

    with pytest.raises(OSError):
        ds.checkout_version("tag3")

    assert ds.checkout_version("tag1").version == 1

    ds = lance.dataset(base_dir, "tag1")
    assert ds.version == 1

    with pytest.raises(ValueError):
        lance.dataset(base_dir, "missing-tag")

    # test tag update
    with pytest.raises(
        ValueError, match="Version not found error: version 3 does not exist"
    ):
        ds.tags.update("tag1", 3)

    with pytest.raises(
        ValueError, match="Ref not found error: tag tag3 does not exist"
    ):
        ds.tags.update("tag3", 1)

    ds.tags.update("tag1", 2)
    ds = lance.dataset(base_dir, "tag1")
    assert ds.version == 2

    ds.tags.update("tag1", 1)
    ds = lance.dataset(base_dir, "tag1")
    assert ds.version == 1


def test_sample(tmp_path: Path):
    table1 = pa.Table.from_pydict({"x": [0, 10, 20, 30, 40, 50], "y": range(6)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    dataset = lance.dataset(base_dir)
    sampled = dataset.sample(3)

    assert sampled.num_rows == 3
    assert sampled.num_columns == 2

    sampled = dataset.sample(4, columns=["x"])

    assert sampled.num_rows == 4
    assert sampled.num_columns == 1

    for row in sampled.column(0).chunk(0):
        assert row.as_py() % 10 == 0


def test_take(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    dataset = lance.dataset(base_dir)
    table2 = dataset.take([0, 1])

    assert isinstance(table2, pa.Table)
    assert table2 == table1


@pytest.mark.parametrize("indices", [[], [1, 1], [1, 1, 20, 20, 21], [21, 0, 21, 1, 0]])
def test_take_duplicate_index(tmp_path: Path, indices: List[int]):
    table = pa.table({"x": range(24)})
    dataset = lance.write_dataset(
        table, tmp_path, max_rows_per_group=3, max_rows_per_file=9
    )
    expected = table.take(pa.array(indices, pa.int64()))

    assert dataset.take(indices) == expected


def test_take_with_columns(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    dataset = lance.dataset(base_dir)
    table2 = dataset.take([0], columns=["b"])

    assert table2 == pa.Table.from_pylist([{"b": 2}])


def test_take_with_projection(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    dataset = lance.dataset(base_dir)
    table2 = dataset.take([0], columns={"a2": "a*2", "bup": "UPPER(b)"})

    assert table2 == pa.Table.from_pylist([{"a2": 2, "bup": "X"}])

    table3 = dataset._take_rows([0], columns={"a2": "a*2", "bup": "UPPER(b)"})
    assert table3 == table2


def test_filter(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    actual_tab = dataset.to_table(columns=["a"], filter=(pa.compute.field("b") > 50))
    assert actual_tab == pa.Table.from_pydict({"a": range(51, 100)})


def test_filter_meta_columns(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    ds = lance.write_dataset(table, base_dir)

    rowids = ds.to_table(with_row_id=True, columns=[])
    some_row_id = random.sample(rowids.column(0).to_pylist(), 1)[0]
    filtered = ds.to_table(filter=f"_rowid = {some_row_id}", with_row_id=True)

    assert len(filtered) == 1

    rowaddrs = ds.to_table(with_row_address=True, columns=[])
    some_row_addr = random.sample(rowaddrs.column(0).to_pylist(), 1)[0]
    filtered = ds.to_table(filter=f"_rowaddr = {some_row_addr}", with_row_address=True)

    assert len(filtered) == 1


@pytest.mark.parametrize("data_storage_version", ["legacy", "stable"])
def test_limit_offset(tmp_path: Path, data_storage_version: str):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(
        table, base_dir, data_storage_version=data_storage_version, max_rows_per_file=10
    )
    dataset = lance.dataset(base_dir)

    # test just limit
    assert dataset.to_table(limit=10) == table.slice(0, 10)

    # test just offset
    assert dataset.to_table(offset=10) == table.slice(10, 100)

    # test both
    assert dataset.to_table(offset=10, limit=10) == table.slice(10, 10)

    # Slicing in the middle of fragments
    assert dataset.to_table(offset=5, limit=20) == table.slice(5, 20)

    # Slicing within a single fragment
    assert dataset.to_table(offset=5, limit=3) == table.slice(5, 3)

    # Skipping entire fragments
    assert dataset.to_table(offset=50, limit=25) == table.slice(50, 25)

    # Limit past the end
    assert dataset.to_table(offset=50, limit=100) == table.slice(50, 50)

    # Invalid limit / offset
    with pytest.raises(ValueError, match="Offset must be non-negative"):
        assert dataset.to_table(offset=-1, limit=10) == table.slice(50, 50)
    with pytest.raises(ValueError, match="Limit must be non-negative"):
        assert dataset.to_table(offset=10, limit=-1) == table.slice(50, 50)

    full_ds_version = dataset.version
    dataset.delete("a % 2 = 0")
    filt_table = table.filter((pa.compute.bit_wise_and(pa.compute.field("a"), 1)) != 0)
    assert (
        dataset.to_table(offset=10).combine_chunks()
        == filt_table.slice(10).combine_chunks()
    )

    dataset = dataset.checkout_version(full_ds_version)
    dataset.restore()
    dataset.delete("a > 2 AND a < 7")
    print(dataset.to_table(offset=3, limit=1))
    filt_table = table.slice(7, 1)

    assert dataset.to_table(offset=3, limit=1) == filt_table


def test_relative_paths(tmp_path: Path):
    # relative paths get coerced to the full absolute path
    current_dir = os.getcwd()
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    rel_uri = "test.lance"
    try:
        os.chdir(tmp_path)
        lance.write_dataset(table, rel_uri)

        # relative path works in the current dir
        ds = lance.dataset(rel_uri)
        assert ds.to_table() == table
    finally:
        os.chdir(current_dir)

    # relative path doesn't work in the context of a different dir
    with pytest.raises(ValueError):
        ds = lance.dataset(rel_uri)

    # relative path gets resolved to the right absolute path
    ds = lance.dataset(tmp_path / rel_uri)
    assert ds.to_table() == table


@pytest.mark.skipif(
    (platform.system() == "Windows"),
    reason="mocking user's home folder it not working",
)
def test_tilde_paths(tmp_path: Path):
    # tilde paths get resolved to the right absolute path
    tilde_uri = "~/test.lance"
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})

    with mock.patch.dict(
        os.environ, {"HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}
    ):
        # NOTE: the resolution logic is a bit finicky
        # link 1 - https://docs.rs/dirs/4.0.0/dirs/fn.home_dir.html
        # link 2 - https://docs.python.org/3/library/os.path.html#os.path.expanduser
        expected_abs_path = os.path.expanduser(tilde_uri)
        assert expected_abs_path == os.fspath(tmp_path / "test.lance")

        lance.write_dataset(table, tilde_uri)
        # works in the current context
        ds = lance.dataset(tilde_uri)
        assert ds.to_table() == table

    # works with the resolved absolute path
    ds = lance.dataset(expected_abs_path)
    assert ds.to_table() == table


def test_to_batches(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    batches = dataset.to_batches(batch_readahead=20)
    assert pa.Table.from_batches(batches) == table

    unordered_batches = dataset.to_batches(batch_readahead=20, scan_in_order=False)
    sorted_batches = pa.Table.from_batches(unordered_batches).sort_by("a")
    assert sorted_batches == table


def test_list_from_parquet(tmp_path: Path):
    # This is a regression for GH-1482, the parquet reader creates
    # list fields with the name 'element' instead of 'item'.  We should
    # ignore that
    tab = pa.Table.from_pydict(
        {"x": pa.array([[1, 2], [3, 4]], pa.list_(pa.float32(), 2))}
    )
    pq.write_table(tab, tmp_path / "foo.parquet")
    tab = pq.read_table(tmp_path / "foo.parquet")
    lance.write_dataset(tab, tmp_path / "foo.lance")


def test_pickle(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    pickled = pickle.dumps(dataset)
    unpickled = pickle.loads(pickled)
    assert dataset.to_table() == unpickled.to_table()


def test_nested_projection(tmp_path: Path):
    table = pa.Table.from_pydict(
        {
            "a": range(100),
            "b": range(100),
            "struct": [{"x": counter, "y": counter % 2 == 0} for counter in range(100)],
        }
    )
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)

    projected = dataset.to_table(columns=["struct.x"])
    assert projected == pa.Table.from_pydict({"struct.x": range(100)})

    projected = dataset.to_table(columns=["struct.y"])
    assert projected == pa.Table.from_pydict(
        {"struct.y": [i % 2 == 0 for i in range(100)]}
    )


def test_nested_projection_list(tmp_path: Path):
    table = pa.Table.from_pydict(
        {
            "a": range(100),
            "b": range(100),
            "list_struct": [
                [{"x": counter, "y": counter % 2 == 0}] for counter in range(100)
            ],
        }
    )
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)

    projected = dataset.to_table(columns={"list_struct": "list_struct[1]['x']"})
    assert projected == pa.Table.from_pydict({"list_struct": range(100)})

    # FIXME: sqlparser seems to ignore the .y part, but I can't create a simple
    # reproducible example for sqlparser. Possibly an issue in our dialect.
    # projected = dataset.to_table(
    #   columns={"list_struct": "array_element(list_struct, 1).y"})
    # assert projected == pa.Table.from_pydict(
    #     {"list_struct": [i % 2 == 0 for i in range(100)]}
    # )


def test_polar_scan(tmp_path: Path):
    some_structs = [{"x": counter, "y": counter} for counter in range(100)]
    table = pa.Table.from_pydict(
        {
            "a": range(100),
            "b": range(100),
            "struct": some_structs,
        }
    )
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    polars_df = pl.scan_pyarrow_dataset(dataset)
    df = dataset.to_table().to_pandas()
    tm.assert_frame_equal(polars_df.collect().to_pandas(), df)

    # Note, this doesn't verify that the filter is actually pushed down.
    # It only checks that, if the filter is pushed down, we interpret it
    # correctly.
    def check_pushdown_filt(pl_filt, sql_filt):
        polars_df = pl.scan_pyarrow_dataset(dataset).filter(pl_filt)
        df = dataset.to_table(filter=sql_filt).to_pandas()
        tm.assert_frame_equal(polars_df.collect().to_pandas(), df)

    # These three should push down (but we don't verify)
    check_pushdown_filt(pl.col("a") > 50, "a > 50")
    check_pushdown_filt(~(pl.col("a") > 50), "a <= 50")
    check_pushdown_filt(pl.col("a").is_in([50, 51, 52]), "a IN (50, 51, 52)")
    # At the current moment it seems polars cannot pushdown this
    # kind of filter
    check_pushdown_filt((pl.col("a") + 3) < 100, "(a + 3) < 100")

    # I can't seem to get struct["x"] to work in Lance but maybe there is
    # a way.  For now, let's compare it directly to the pyarrow compute version

    # Doesn't yet work today :( due to upstream issue (datafusion's substrait parser
    # doesn't yet handle nested refs)
    # if pa.cpp_version_info.major >= 14:
    #     polars_df = pl.scan_pyarrow_dataset(dataset).filter(pl.col("struct.x") < 10)
    #     df = dataset.to_table(filter=pc.field("struct", "x") < 10).to_pandas()
    #     tm.assert_frame_equal(polars_df.collect().to_pandas(), df)


def test_count_fragments(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    count_fragments = dataset.stats.dataset_stats()["num_fragments"]
    assert count_fragments == 1


def test_count_rows(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    count_rows = dataset.count_rows()
    assert count_rows == 100

    assert dataset.count_rows(filter="a < 50") == 50


def test_get_fragments(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    fragment = dataset.get_fragments()[0]
    assert fragment.count_rows() == 100
    assert fragment.physical_rows == 100
    assert fragment.num_deletions == 0

    assert fragment.metadata.id == 0

    head = fragment.head(10)
    tm.assert_frame_equal(head.to_pandas(), table.to_pandas()[0:10])

    assert fragment.to_table() == table

    taken = fragment.take([18, 20, 33, 53])
    assert taken == pa.Table.from_pydict({"a": [18, 20, 33, 53], "b": [18, 20, 33, 53]})


def test_pickle_fragment(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    fragment = dataset.get_fragments()[0]
    pickled = pickle.dumps(fragment)
    unpickled = pickle.loads(pickled)

    assert fragment.to_table() == unpickled.to_table()


def test_cleanup_old_versions(tmp_path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)
    time.sleep(0.1)
    moment = datetime.now()
    lance.write_dataset(table, base_dir, mode="overwrite")

    dataset = lance.dataset(base_dir)

    # These calls do nothing, but make sure we can call the method ok

    # Ok, defaults to two weeks ago
    stats = dataset.cleanup_old_versions()
    assert stats.bytes_removed == 0
    assert stats.old_versions == 0

    # Ok, can accept timedelta
    dataset.cleanup_old_versions(older_than=timedelta(days=14))

    # print(tmp_path)
    # for root, dirnames, filenames in os.walk(tmp_path):
    #     for filename in filenames:
    #         print(root + "/" + filename)

    # Now this call will actually delete the old version
    stats = dataset.cleanup_old_versions(older_than=(datetime.now() - moment))
    assert stats.bytes_removed > 0
    assert stats.old_versions == 1


def test_cleanup_error_when_tagged_old_versions(tmp_path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)
    lance.write_dataset(table, base_dir, mode="overwrite")
    time.sleep(0.1)
    moment = datetime.now()
    lance.write_dataset(table, base_dir, mode="overwrite")

    dataset = lance.dataset(base_dir)
    dataset.tags.create("old-tag", 1)
    dataset.tags.create("another-old-tag", 2)

    with pytest.raises(OSError):
        dataset.cleanup_old_versions(older_than=(datetime.now() - moment))
    assert len(dataset.versions()) == 3

    dataset.tags.delete("old-tag")
    with pytest.raises(OSError):
        dataset.cleanup_old_versions(older_than=(datetime.now() - moment))
    assert len(dataset.versions()) == 3

    dataset.tags.delete("another-old-tag")
    stats = dataset.cleanup_old_versions(older_than=(datetime.now() - moment))
    assert stats.bytes_removed > 0
    assert stats.old_versions == 2
    assert len(dataset.versions()) == 1


def test_cleanup_around_tagged_old_versions(tmp_path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)
    lance.write_dataset(table, base_dir, mode="overwrite")
    time.sleep(0.1)
    moment = datetime.now()
    lance.write_dataset(table, base_dir, mode="overwrite")

    dataset = lance.dataset(base_dir)
    dataset.tags.create("old-tag", 1)
    dataset.tags.create("another-old-tag", 2)
    dataset.tags.create("tag-latest", 3)

    stats = dataset.cleanup_old_versions(
        older_than=(datetime.now() - moment), error_if_tagged_old_versions=False
    )
    assert stats.bytes_removed == 0
    assert stats.old_versions == 0

    dataset.tags.delete("old-tag")
    stats = dataset.cleanup_old_versions(
        older_than=(datetime.now() - moment), error_if_tagged_old_versions=False
    )
    assert stats.bytes_removed > 0
    assert stats.old_versions == 1

    dataset.tags.delete("another-old-tag")
    stats = dataset.cleanup_old_versions(
        older_than=(datetime.now() - moment), error_if_tagged_old_versions=False
    )
    assert stats.bytes_removed > 0
    assert stats.old_versions == 1


def test_create_from_commit(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    fragment = lance.fragment.LanceFragment.create(base_dir, table)

    operation = lance.LanceOperation.Overwrite(table.schema, [fragment])
    dataset = lance.LanceDataset.commit(base_dir, operation)
    tbl = dataset.to_table()
    assert tbl == table


def test_append_with_commit(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"

    dataset = lance.write_dataset(table, base_dir)

    fragment = lance.fragment.LanceFragment.create(base_dir, table)
    append = lance.LanceOperation.Append([fragment])

    with pytest.raises(OSError):
        # Must specify read version
        dataset = lance.LanceDataset.commit(dataset, append)

    dataset = lance.LanceDataset.commit(dataset, append, read_version=1)

    tbl = dataset.to_table()

    expected = pa.Table.from_pydict(
        {
            "a": list(range(100)) + list(range(100)),
            "b": list(range(100)) + list(range(100)),
        }
    )
    assert tbl == expected


def test_delete_with_commit(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"

    lance.write_dataset(table, base_dir)
    lance.write_dataset(table, base_dir, mode="append")

    half_table = pa.Table.from_pydict({"a": range(50), "b": range(50)})

    fragments = lance.dataset(base_dir).get_fragments()

    updated_fragment = fragments[0].delete("a >= 50")
    delete = lance.LanceOperation.Delete(
        [updated_fragment], [fragments[1].fragment_id], "hello"
    )

    dataset = lance.LanceDataset.commit(base_dir, delete, read_version=2)

    tbl = dataset.to_table()
    assert tbl == half_table


def test_count_deleted_rows(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"

    dataset = lance.write_dataset(table, base_dir)
    dataset.delete("a >= 50")

    assert dataset.stats.dataset_stats()["num_deleted_rows"] == 50


def test_restore_with_commit(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"

    lance.write_dataset(table, base_dir)
    lance.write_dataset(table, base_dir, mode="append")

    restore = lance.LanceOperation.Restore(1)
    dataset = lance.LanceDataset.commit(base_dir, restore)

    tbl = dataset.to_table()
    assert tbl == table


def test_merge_with_commit(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"

    lance.write_dataset(table, base_dir)

    fragment = lance.dataset(base_dir).get_fragments()[0]
    merged = fragment.merge_columns(
        lambda _: pa.RecordBatch.from_pydict({"c": range(100)})
    )[0]

    expected = pa.Table.from_pydict({"a": range(100), "b": range(100), "c": range(100)})

    # PyArrow schema is deprecated, but should still work for now.
    with pytest.deprecated_call():
        merge = lance.LanceOperation.Merge([merged], expected.schema)
        dataset = lance.LanceDataset.commit(base_dir, merge, read_version=1)

    tbl = dataset.to_table()

    assert tbl == expected


def test_merge_with_schema_holes(tmp_path: Path):
    # Create table with 3 cols
    table = pa.table({"a": range(10)})
    dataset = lance.write_dataset(table, tmp_path)
    dataset.add_columns({"b": "a + 1"})
    dataset.add_columns({"c": "a + 2"})
    # Delete the middle column to create a hole in the field ids
    dataset.drop_columns(["b"])

    fragment = dataset.get_fragments()[0]
    merged, schema = fragment.merge_columns(
        lambda _: pa.RecordBatch.from_pydict({"d": range(10, 20)})
    )

    merge = lance.LanceOperation.Merge([merged], schema)
    dataset = lance.LanceDataset.commit(tmp_path, merge, read_version=dataset.version)

    dataset.validate()

    tbl = dataset.to_table()
    expected = pa.table(
        {
            "a": range(10),
            "c": range(2, 12),
            "d": range(10, 20),
        }
    )
    assert tbl == expected


def test_merge_search(tmp_path: Path):
    left_table = pa.Table.from_pydict({"id": [1, 2, 3], "left": ["a", "b", "c"]})
    right_table = pa.Table.from_pydict({"id": [1, 2, 3], "right": ["A", "B", "C"]})

    left_ds = lance.write_dataset(left_table, tmp_path / "left")

    right_ds = lance.write_dataset(right_table, tmp_path / "right")
    left_ds.merge(right_ds, "id")

    full = left_ds.to_table()
    full_filtered = left_ds.to_table(filter="id < 3")

    partial = left_ds.to_table(columns=["left"])

    assert full.column("left") == partial.column("left")

    partial = left_ds.to_table(columns=["left"], filter="id < 3")

    assert full_filtered.column("left") == partial.column("left")


def test_data_files(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    fragment = lance.fragment.LanceFragment.create(base_dir, table)

    data_files = fragment.data_files()
    assert len(data_files) == 1
    # it is a valid uuid
    uuid.UUID(os.path.splitext(data_files[0].path())[0])

    assert fragment.deletion_file() is None


def test_deletion_file(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    dataset = lance.write_dataset(table, base_dir)
    fragment = dataset.get_fragment(0)

    assert fragment.deletion_file() is None

    new_fragment = fragment.delete("a < 10")

    # Old fragment unchanged
    assert fragment.deletion_file() is None

    # New fragment has deletion file
    assert new_fragment.deletion_file() is not None
    assert re.match("_deletions/0-1-[0-9]{1,32}.arrow", new_fragment.deletion_file())
    operation = lance.LanceOperation.Overwrite(table.schema, [new_fragment])
    dataset = lance.LanceDataset.commit(base_dir, operation)
    assert dataset.count_rows() == 90


def test_commit_fragments_via_scanner(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    parquet_dir = tmp_path / "parquet.parquet"
    pq.write_to_dataset(table, parquet_dir)

    base_dir = tmp_path / "test"
    scanner = pa.dataset.dataset(parquet_dir).scanner()
    fragment_metadata = lance.fragment.LanceFragment.create(base_dir, scanner)

    # Pickle-able
    pickled = pickle.dumps(fragment_metadata)
    unpickled = pickle.loads(pickled)
    assert fragment_metadata == unpickled

    operation = lance.LanceOperation.Overwrite(table.schema, [fragment_metadata])
    dataset = lance.LanceDataset.commit(base_dir, operation)
    assert dataset.schema == table.schema

    tbl = dataset.to_table()
    assert tbl == table


def test_load_scanner_from_fragments(tmp_path: Path):
    tab = pa.table({"a": range(100), "b": range(100)})
    for _ in range(3):
        lance.write_dataset(tab, tmp_path / "dataset", mode="append")

    dataset = lance.dataset(tmp_path / "dataset")
    fragments = list(dataset.get_fragments())
    assert len(fragments) == 3

    scanner = dataset.scanner(fragments=fragments[0:2])
    assert scanner.to_table().num_rows == 2 * 100

    # Accepts an iterator
    scanner = dataset.scanner(fragments=iter(fragments[0:2]), scan_in_order=False)
    assert scanner.to_table().num_rows == 2 * 100


def test_merge_data(tmp_path: Path):
    tab = pa.table({"a": range(100), "b": range(100)})
    lance.write_dataset(tab, tmp_path / "dataset", mode="append")

    dataset = lance.dataset(tmp_path / "dataset")

    # rejects partial data for non-nullable types
    new_tab = pa.table({"a": range(40), "c": range(40)})
    # TODO: this should be ValueError
    with pytest.raises(
        OSError, match=".+Lance does not yet support nulls for type Int64."
    ):
        dataset.merge(new_tab, "a")

    # accepts a full merge
    new_tab = pa.table({"a": range(100), "c": range(100)})
    dataset.merge(new_tab, "a")
    assert dataset.version == 2
    assert dataset.to_table() == pa.table(
        {
            "a": range(100),
            "b": range(100),
            "c": range(100),
        }
    )

    # accepts a partial for string
    new_tab = pa.table({"a2": range(5), "d": ["a", "b", "c", "d", "e"]})
    dataset.merge(new_tab, left_on="a", right_on="a2")
    assert dataset.version == 3
    expected = pa.table(
        {
            "a": range(100),
            "b": range(100),
            "c": range(100),
            "d": ["a", "b", "c", "d", "e"] + [None] * 95,
        }
    )
    assert dataset.to_table() == expected
    # Verify we can also load from fresh instance
    dataset = lance.dataset(tmp_path / "dataset")
    assert dataset.to_table() == expected


def test_merge_from_dataset(tmp_path: Path):
    tab1 = pa.table({"a": range(100), "b": range(100)})
    ds1 = lance.write_dataset(tab1, tmp_path / "dataset1", mode="append")

    tab2 = pa.table({"a": range(100), "c": range(100)})
    ds2 = lance.write_dataset(tab2, tmp_path / "dataset2", mode="append")

    ds1.merge(ds2.to_batches(), "a", schema=ds2.schema)
    assert ds1.version == 2
    assert ds1.to_table() == pa.table(
        {
            "a": range(100),
            "b": range(100),
            "c": range(100),
        }
    )


def test_delete_data(tmp_path: Path):
    # We pass schema explicitly since we want b to be non-nullable.
    schema = pa.schema(
        [
            pa.field("a", pa.int64()),
            pa.field("b", pa.int64(), nullable=False),
        ]
    )
    tab = pa.table({"a": range(100), "b": range(100)}, schema=schema)
    lance.write_dataset(tab, tmp_path / "dataset", mode="append")

    dataset = lance.dataset(tmp_path / "dataset")

    dataset.delete("a < 10")
    dataset.delete("b in (98, 99)")
    assert dataset.version == 3
    assert dataset.to_table() == pa.table(
        {"a": range(10, 98), "b": range(10, 98)}, schema=schema
    )

    dataset.delete(pa_ds.field("a") < 20)
    assert dataset.version == 4
    assert dataset.to_table() == pa.table(
        {"a": range(20, 98), "b": range(20, 98)}, schema=schema
    )

    # These sorts of filters were previously used as a work-around for not
    # supporting "WHERE true". But now we need to make sure they still work
    # even with presence of expression simplification passes.
    old_version = dataset.version
    dataset.delete("b IS NOT NULL")
    assert dataset.count_rows() == 0

    dataset = lance.dataset(tmp_path / "dataset", version=old_version)
    dataset.restore()
    assert dataset.count_rows() > 0
    dataset.delete("true")
    assert dataset.count_rows() == 0


def check_merge_stats(merge_dict, expected):
    assert (
        merge_dict["num_inserted_rows"],
        merge_dict["num_updated_rows"],
        merge_dict["num_deleted_rows"],
    ) == expected


def test_merge_insert(tmp_path: Path):
    nrows = 1000
    table = pa.Table.from_pydict(
        {
            "a": range(nrows),
            "b": [1 for _ in range(nrows)],
            "c": [x % 2 for x in range(nrows)],
        }
    )
    dataset = lance.write_dataset(
        table, tmp_path / "dataset", mode="create", max_rows_per_file=100
    )
    version = dataset.version

    new_table = pa.Table.from_pydict(
        {
            "a": range(300, 300 + nrows),
            "b": [2 for _ in range(nrows)],
            "c": [0 for _ in range(nrows)],
        }
    )

    is_new = pc.field("b") == 2

    merge_dict = (
        dataset.merge_insert("a").when_not_matched_insert_all().execute(new_table)
    )
    table = dataset.to_table()
    assert table.num_rows == 1300
    assert table.filter(is_new).num_rows == 300
    check_merge_stats(merge_dict, (300, 0, 0))

    dataset = lance.dataset(tmp_path / "dataset", version=version)
    dataset.restore()
    merge_dict = dataset.merge_insert("a").when_matched_update_all().execute(new_table)
    table = dataset.to_table()
    assert table.num_rows == 1000
    assert table.filter(is_new).num_rows == 700
    check_merge_stats(merge_dict, (0, 700, 0))

    dataset = lance.dataset(tmp_path / "dataset", version=version)
    dataset.restore()
    merge_dict = (
        dataset.merge_insert("a")
        .when_not_matched_insert_all()
        .when_matched_update_all()
        .execute(new_table)
    )
    table = dataset.to_table()
    assert table.num_rows == 1300
    assert table.filter(is_new).num_rows == 1000
    check_merge_stats(merge_dict, (300, 700, 0))

    dataset = lance.dataset(tmp_path / "dataset", version=version)
    dataset.restore()
    merge_dict = (
        dataset.merge_insert("a")
        .when_not_matched_insert_all()
        .when_matched_update_all("target.c == source.c")
        .execute(new_table)
    )
    table = dataset.to_table()
    assert table.num_rows == 1300
    assert table.filter(is_new).num_rows == 650
    check_merge_stats(merge_dict, (300, 350, 0))

    dataset = lance.dataset(tmp_path / "dataset", version=version)
    dataset.restore()
    merge_dict = (
        dataset.merge_insert("a").when_not_matched_by_source_delete().execute(new_table)
    )
    table = dataset.to_table()
    assert table.num_rows == 700
    assert table.filter(is_new).num_rows == 0
    check_merge_stats(merge_dict, (0, 0, 300))

    dataset = lance.dataset(tmp_path / "dataset", version=version)
    dataset.restore()
    merge_dict = (
        dataset.merge_insert("a")
        .when_not_matched_by_source_delete("a < 100")
        .when_not_matched_insert_all()
        .execute(new_table)
    )

    table = dataset.to_table()
    assert table.num_rows == 1200
    assert table.filter(is_new).num_rows == 300
    check_merge_stats(merge_dict, (300, 0, 100))

    # If the user doesn't specify anything then the merge_insert is
    # a no-op and the operation fails
    dataset = lance.dataset(tmp_path / "dataset", version=version)
    dataset.restore()
    with pytest.raises(ValueError):
        merge_dict = dataset.merge_insert("a").execute(new_table)
        check_merge_stats(merge_dict, (None, None, None))


def test_merge_insert_subcols(tmp_path: Path):
    initial_data = pa.table(
        {
            "a": range(10),
            "b": range(10),
            "c": range(10, 20),
        }
    )
    # Split across two fragments
    dataset = lance.write_dataset(
        initial_data, tmp_path / "dataset", max_rows_per_file=5
    )
    original_fragments = dataset.get_fragments()

    new_values = pa.table(
        {
            "a": range(3, 5),
            "b": range(20, 22),
        }
    )
    (dataset.merge_insert("a").when_matched_update_all().execute(new_values))

    expected = pa.table(
        {
            "a": range(10),
            "b": [0, 1, 2, 20, 21, 5, 6, 7, 8, 9],
            "c": range(10, 20),
        }
    )
    assert dataset.to_table().sort_by("a") == expected

    # First fragment has new file
    fragments = dataset.get_fragments()
    assert fragments[0].fragment_id == original_fragments[0].fragment_id
    assert fragments[1].fragment_id == original_fragments[1].fragment_id

    assert len(fragments[0].data_files()) == 2
    assert str(fragments[0].data_files()[0]) == str(
        original_fragments[0].data_files()[0]
    )
    assert len(fragments[1].data_files()) == 1
    assert str(fragments[1].data_files()[0]) == str(
        original_fragments[1].data_files()[0]
    )


def test_flat_vector_search_with_delete(tmp_path: Path):
    table = pa.Table.from_pydict(
        {
            "id": [i for i in range(10)],
            "vector": pa.array(
                [[1.0, 1.0] for _ in range(10)], pa.list_(pa.float32(), 2)
            ),
        }
    )
    dataset = lance.write_dataset(table, tmp_path / "dataset", mode="create")
    dataset.delete("id = 0")
    assert (
        dataset.scanner(nearest={"column": "vector", "k": 10, "q": [1.0, 1.0]})
        .to_table()
        .num_rows
        == 9
    )


def test_merge_insert_conditional_upsert_example(tmp_path: Path):
    table = pa.Table.from_pydict(
        {
            "id": [1, 2, 3, 4, 5],
            "txNumber": [1, 1, 2, 2, 3],
            "vector": pa.array([[1.0, 1.0] for _ in range(5)], pa.list_(pa.float32())),
        }
    )
    dataset = lance.write_dataset(table, tmp_path / "dataset", mode="create")

    new_table = pa.Table.from_pydict(
        {
            "id": [1, 2, 3, 4, 5],
            "txNumber": [1, 2, 1, 2, 5],
            "vector": pa.array([[2.0, 2.0] for _ in range(5)], pa.list_(pa.float32())),
        }
    )

    merge_dict = (
        dataset.merge_insert("id")
        .when_matched_update_all("target.txNumber < source.txNumber")
        .execute(new_table)
    )

    table = dataset.to_table()

    expected = pa.Table.from_pydict(
        {
            "id": [1, 2, 3, 4, 5],
            "txNumber": [1, 2, 2, 2, 5],
            "vector": pa.array(
                [[1.0, 1.0], [2.0, 2.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]],
                pa.list_(pa.float32()),
            ),
        }
    )

    assert table.sort_by("id") == expected
    check_merge_stats(merge_dict, (0, 2, 0))

    # No matches

    new_table = pa.Table.from_pydict(
        {
            "id": [1, 2, 3, 4, 5],
            "txNumber": [0, 0, 0, 0, 0],
            "vector": pa.array([[2.0, 2.0] for _ in range(5)], pa.list_(pa.float32())),
        }
    )

    merge_dict = (
        dataset.merge_insert("id")
        .when_matched_update_all("target.txNumber < source.txNumber")
        .execute(new_table)
    )

    check_merge_stats(merge_dict, (0, 0, 0))


def test_merge_insert_source_is_dataset(tmp_path: Path):
    nrows = 1000
    table = pa.Table.from_pydict({"a": range(nrows), "b": [1 for _ in range(nrows)]})
    dataset = lance.write_dataset(
        table, tmp_path / "dataset", mode="create", max_rows_per_file=100
    )
    version = dataset.version

    new_table = pa.Table.from_pydict(
        {
            "a": range(300, 300 + nrows),
            "b": [2 for _ in range(nrows)],
        }
    )
    new_dataset = lance.write_dataset(
        new_table, tmp_path / "dataset2", mode="create", max_rows_per_file=80
    )

    is_new = pc.field("b") == 2

    merge_dict = (
        dataset.merge_insert("a").when_not_matched_insert_all().execute(new_dataset)
    )
    table = dataset.to_table()
    assert table.num_rows == 1300
    assert table.filter(is_new).num_rows == 300
    check_merge_stats(merge_dict, (300, 0, 0))

    dataset = lance.dataset(tmp_path / "dataset", version=version)
    dataset.restore()

    reader = new_dataset.to_batches()

    merge_dict = (
        dataset.merge_insert("a")
        .when_not_matched_insert_all()
        .execute(reader, schema=new_dataset.schema)
    )
    table = dataset.to_table()
    assert table.num_rows == 1300
    assert table.filter(is_new).num_rows == 300
    check_merge_stats(merge_dict, (300, 0, 0))


def test_merge_insert_multiple_keys(tmp_path: Path):
    nrows = 1000
    # a - [0, 1, 2, ..., 999]
    # b - [1, 1, 1, ..., 1]
    # c - [0, 1, 0, ..., 1]
    table = pa.Table.from_pydict(
        {
            "a": range(nrows),
            "b": [1 for _ in range(nrows)],
            "c": [i % 2 for i in range(nrows)],
        }
    )
    dataset = lance.write_dataset(
        table, tmp_path / "dataset", mode="create", max_rows_per_file=100
    )

    # a - [300, 301, 302, ..., 1299]
    # b - [2, 2, 2, ..., 2]
    # c - [0, 0, 0, ..., 0]
    new_table = pa.Table.from_pydict(
        {
            "a": range(300, 300 + nrows),
            "b": [2 for _ in range(nrows)],
            "c": [0 for _ in range(nrows)],
        }
    )

    is_new = pc.field("b") == 2

    merge_dict = (
        dataset.merge_insert(["a", "c"]).when_matched_update_all().execute(new_table)
    )
    table = dataset.to_table()
    assert table.num_rows == 1000
    assert table.filter(is_new).num_rows == 350
    check_merge_stats(merge_dict, (0, 350, 0))


def test_merge_insert_incompatible_schema(tmp_path: Path):
    nrows = 1000
    table = pa.Table.from_pydict(
        {
            "a": range(nrows),
            "b": [1 for _ in range(nrows)],
        }
    )
    dataset = lance.write_dataset(
        table, tmp_path / "dataset", mode="create", max_rows_per_file=100
    )

    new_table = pa.Table.from_pydict(
        {
            "a": range(300, 300 + nrows),
        }
    )

    with pytest.raises(OSError):
        merge_dict = (
            dataset.merge_insert("a")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(new_table)
        )
        check_merge_stats(merge_dict, (None, None, None))


def test_merge_insert_vector_column(tmp_path: Path):
    table = pa.Table.from_pydict(
        {
            "vec": pa.array([[1, 2, 3], [4, 5, 6]], pa.list_(pa.float32(), 3)),
            "key": [1, 2],
        }
    )

    new_table = pa.Table.from_pydict(
        {
            "vec": pa.array([[7, 8, 9], [10, 11, 12]], pa.list_(pa.float32(), 3)),
            "key": [2, 3],
        }
    )

    dataset = lance.write_dataset(
        table, tmp_path / "dataset", mode="create", max_rows_per_file=100
    )

    merge_dict = (
        dataset.merge_insert(["key"])
        .when_not_matched_insert_all()
        .when_matched_update_all()
        .execute(new_table)
    )
    expected = pa.Table.from_pydict(
        {
            "vec": pa.array(
                [[1, 2, 3], [7, 8, 9], [10, 11, 12]], pa.list_(pa.float32(), 3)
            ),
            "key": [1, 2, 3],
        }
    )

    assert dataset.to_table().sort_by("key") == expected
    check_merge_stats(merge_dict, (1, 1, 0))


def check_update_stats(update_dict, expected):
    assert (update_dict["num_rows_updated"],) == expected


def test_update_dataset(tmp_path: Path):
    nrows = 100
    vecs = pa.FixedSizeListArray.from_arrays(
        pa.array(range(2 * nrows), type=pa.float32()), 2
    )
    tab = pa.table({"a": range(nrows), "b": range(nrows), "vec": vecs})
    lance.write_dataset(tab, tmp_path / "dataset", mode="append")

    dataset = lance.dataset(tmp_path / "dataset")

    update_dict = dataset.update(dict(b="b + 1"))
    expected = pa.table({"a": range(100), "b": range(1, 101)})
    assert dataset.to_table(columns=["a", "b"]) == expected
    check_update_stats(update_dict, (100,))

    update_dict = dataset.update(dict(a="a * 2"), where="a < 50")
    expected = pa.table(
        {
            "a": [x * 2 if x < 50 else x for x in range(100)],
            "b": range(1, 101),
        }
    )
    assert dataset.to_table(columns=["a", "b"]).sort_by("b") == expected
    check_update_stats(update_dict, (50,))

    update_dict = dataset.update(dict(vec="[42.0, 43.0]"))
    expected = pa.table(
        {
            "b": range(1, 101),
            "vec": pa.array(
                [[42.0, 43.0] for _ in range(100)], pa.list_(pa.float32(), 2)
            ),
        }
    )
    assert dataset.to_table(columns=["b", "vec"]).sort_by("b") == expected
    check_update_stats(update_dict, (100,))


def test_update_dataset_all_types(tmp_path: Path):
    table = pa.table(
        {
            "int32": pa.array([1], pa.int32()),
            "int64": pa.array([1], pa.int64()),
            "uint32": pa.array([1], pa.uint32()),
            "string": pa.array(["foo"], pa.string()),
            "large_string": pa.array(["foo"], pa.large_string()),
            "float32": pa.array([1.0], pa.float32()),
            "float64": pa.array([1.0], pa.float64()),
            "bool": pa.array([True], pa.bool_()),
            "date32": pa.array([date(2021, 1, 1)], pa.date32()),
            "timestamp_ns": pa.array([datetime(2021, 1, 1)], pa.timestamp("ns")),
            "timestamp_ms": pa.array([datetime(2021, 1, 1)], pa.timestamp("ms")),
            "vec_f32": pa.array([[1.0, 2.0]], pa.list_(pa.float32(), 2)),
            "vec_f64": pa.array([[1.0, 2.0]], pa.list_(pa.float64(), 2)),
        }
    )

    dataset = lance.write_dataset(table, tmp_path)

    # One update with all matching types
    update_dict = dataset.update(
        dict(
            int32="2",
            int64="2",
            uint32="2",
            string="'bar'",
            large_string="'bar'",
            float32="2.0",
            float64="2.0",
            bool="false",
            date32="DATE '2021-01-02'",
            timestamp_ns='TIMESTAMP "2021-01-02 00:00:00"',
            timestamp_ms='TIMESTAMP "2021-01-02 00:00:00"',
            vec_f32="[3.0, 4.0]",
            vec_f64="[3.0, 4.0]",
        )
    )
    expected = pa.table(
        {
            "int32": pa.array([2], pa.int32()),
            "int64": pa.array([2], pa.int64()),
            "uint32": pa.array([2], pa.uint32()),
            "string": pa.array(["bar"], pa.string()),
            "large_string": pa.array(["bar"], pa.large_string()),
            "float32": pa.array([2.0], pa.float32()),
            "float64": pa.array([2.0], pa.float64()),
            "bool": pa.array([False], pa.bool_()),
            "date32": pa.array([date(2021, 1, 2)], pa.date32()),
            "timestamp_ns": pa.array([datetime(2021, 1, 2)], pa.timestamp("ns")),
            "timestamp_ms": pa.array([datetime(2021, 1, 2)], pa.timestamp("ms")),
            "vec_f32": pa.array([[3.0, 4.0]], pa.list_(pa.float32(), 2)),
            "vec_f64": pa.array([[3.0, 4.0]], pa.list_(pa.float64(), 2)),
        }
    )
    assert dataset.to_table() == expected
    check_update_stats(update_dict, (1,))


def test_update_with_binary_field(tmp_path: Path):
    # Create a lance dataset with binary fields
    table = pa.Table.from_pydict(
        {
            "a": [f"str-{i}" for i in range(100)],
            "b": [b"bin-{i}" for i in range(100)],
            "c": list(range(100)),
        }
    )
    dataset = lance.write_dataset(table, tmp_path)

    # Update binary field
    update_dict = dataset.update({"b": "X'616263'"}, where="c < 2")

    ds = lance.dataset(tmp_path)
    assert ds.scanner(filter="c < 2").to_table().column(
        "b"
    ).combine_chunks() == pa.array([b"abc", b"abc"])
    check_update_stats(update_dict, (2,))


def test_create_update_empty_dataset(tmp_path: Path, provide_pandas: bool):
    base_dir = tmp_path / "dataset"

    fields = [
        ("a", pa.string()),
        ("b", pa.int64()),
        ("c", pa.float64()),
    ]
    df = pd.DataFrame(columns=[name for name, _ in fields])
    tab = pa.Table.from_pandas(df, schema=pa.schema(fields))
    dataset = lance.write_dataset(tab, base_dir)

    assert dataset.count_rows() == 0
    expected_schema = pa.schema(
        [
            pa.field("a", pa.string()),
            pa.field("b", pa.int64()),
            pa.field("c", pa.float64()),
        ]
    )
    assert dataset.schema == expected_schema
    assert dataset.to_table() == pa.table(
        {"a": [], "b": [], "c": []}, schema=expected_schema
    )

    tab2 = pa.table({"a": ["foo"], "b": [1], "c": [2.0]})
    dataset = lance.write_dataset(tab2, base_dir, mode="append")

    assert dataset.count_rows() == 1
    assert dataset.to_table() == pa.table(
        {"a": ["foo"], "b": [1], "c": [2.0]}, schema=expected_schema
    )


def test_scan_with_batch_size(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    df = pd.DataFrame({"a": range(10000), "b": range(10000)})
    dataset = lance.write_dataset(df, base_dir)

    batches = dataset.scanner(batch_size=16, scan_in_order=True).to_batches()

    for idx, batch in enumerate(batches):
        assert batch.num_rows == 16
        df = batch.to_pandas()
        assert df["a"].iloc[0] == idx * 16

    os.environ["LANCE_DEFAULT_BATCH_SIZE"] = "12"
    batches = dataset.scanner(scan_in_order=True).to_batches()
    for batch in batches:
        # The last batch in each file has 4 rows
        assert batch.num_rows == 12 or batch.num_rows == 4

    del os.environ["LANCE_DEFAULT_BATCH_SIZE"]
    batches = dataset.scanner(scan_in_order=True).to_batches()
    for batch in batches:
        assert batch.num_rows != 12


@pytest.mark.slow
def test_io_buffer_size(tmp_path: Path):
    # These cases regress deadlock issues that happen when the
    # batch size was very large (in bytes) so that batches are
    # bigger than the I/O buffer size
    #
    # The test is slow (it needs to generate enough data to cover a variety
    # of cases) but it is essential to run if any changes are made to the
    # 2.0 scheduling priority / decoding strategy
    #
    # In this test we create 4 pages of data, 2 for each column.  We
    # then set the I/O buffer size to 5000 bytes so that we only read
    # 1 page at a time and set the batch size to 2M rows so that we
    # read in all 4 pages for a single batch.
    #
    # The scheduler will schedule in this order: C0P0, C1P0, C0P1, C1P1
    #
    # The deadlock would happen because the decoder was decoding in this
    # order: C0P0, C0P1, C1P0, C1P1.
    #
    # The decoder will wait for C0P1 but the scheduler will only have scheduled
    # C1P0 and so deadlock would happen.
    #
    # The fix was to change the decoder to decode in the same order as the
    # scheduler.
    base_dir = tmp_path / "dataset"

    def datagen():
        for i in range(2):
            yield pa.record_batch(
                [
                    pa.array(range(1024 * 1024), pa.uint64()),
                    pa.array(range(1024 * 1024), pa.uint64()),
                ],
                names=["a", "b"],
            )

    schema = pa.schema({"a": pa.uint64(), "b": pa.uint64()})

    dataset = lance.write_dataset(
        datagen(),
        base_dir,
        schema=schema,
        data_storage_version="stable",
        max_rows_per_file=2 * 1024 * 1024,
        mode="overwrite",
    )

    dataset.scanner(batch_size=2 * 1024 * 1024, io_buffer_size=5000).to_table()

    # This test is similar but the first column is a list column.  The I/O to grab
    # the list items will span multiple requests and it is important that all of
    # those requests share the priority of the parent list

    def datagen():
        for i in range(2):
            yield pa.record_batch(
                [
                    pa.array([[0]] * 1024 * 1024, pa.list_(pa.uint64())),
                    pa.array(range(1024 * 1024), pa.uint64()),
                ],
                names=["a", "b"],
            )

    schema = pa.schema({"a": pa.list_(pa.uint64()), "b": pa.uint64()})

    dataset = lance.write_dataset(
        datagen(),
        base_dir,
        schema=schema,
        data_storage_version="stable",
        max_rows_per_file=2 * 1024 * 1024,
        mode="overwrite",
    )

    dataset.scanner(batch_size=2 * 1024 * 1024, io_buffer_size=5000).to_table()

    # Another scenario.  Each list item is a page in itself and we have two list
    # columns

    def datagen():
        for i in range(16):
            yield pa.record_batch(
                [
                    pa.array([[0] * 5 * 1024 * 1024], pa.list_(pa.uint64())),
                    pa.array([[0] * 5 * 1024 * 1024], pa.list_(pa.uint64())),
                ],
                names=["a", "b"],
            )

    schema = pa.schema({"a": pa.list_(pa.uint64()), "b": pa.list_(pa.uint64())})

    dataset = lance.write_dataset(
        datagen(),
        base_dir,
        schema=schema,
        data_storage_version="stable",
        mode="overwrite",
    )

    dataset.scanner(batch_size=16, io_buffer_size=5000).to_table()

    # Same scenario as above except now it's a list<list<int>>

    def datagen():
        for i in range(16):
            yield pa.record_batch(
                [
                    pa.array(
                        [[[0] * 5 * 1024 * 1024]], pa.list_(pa.list_(pa.uint64()))
                    ),
                    pa.array(
                        [[[0] * 5 * 1024 * 1024]], pa.list_(pa.list_(pa.uint64()))
                    ),
                ],
                names=["a", "b"],
            )

    schema = pa.schema(
        {"a": pa.list_(pa.list_(pa.uint64())), "b": pa.list_(pa.list_(pa.uint64()))}
    )

    dataset = lance.write_dataset(
        datagen(),
        base_dir,
        schema=schema,
        data_storage_version="stable",
        mode="overwrite",
    )

    dataset.scanner(batch_size=16, io_buffer_size=5000).to_table()

    # Next we consider the case where the column is a struct column and we want to
    # make sure we don't decode too deeply into the struct child

    def datagen():
        for i in range(2):
            yield pa.record_batch(
                [
                    pa.array(
                        [{"foo": i} for i in range(1024 * 1024)],
                        pa.struct([pa.field("foo", pa.uint64())]),
                    ),
                    pa.array(range(1024 * 1024), pa.uint64()),
                ],
                names=["a", "b"],
            )

    schema = pa.schema(
        {"a": pa.struct([pa.field("foo", pa.uint64())]), "b": pa.uint64()}
    )

    dataset = lance.write_dataset(
        datagen(),
        base_dir,
        schema=schema,
        data_storage_version="stable",
        max_rows_per_file=2 * 1024 * 1024,
        mode="overwrite",
    )

    dataset.scanner(batch_size=2 * 1024 * 1024, io_buffer_size=5000).to_table()

    # This reproduces another issue we saw where there are a bunch of empty lists and
    # lance was calculating the page priority incorrectly.

    fsl_type = pa.list_(pa.uint64(), 32 * 1024 * 1024)
    list_type = pa.list_(fsl_type)

    def datagen():
        # Each item is 32
        values = pa.array(range(32 * 1024 * 1024 * 7), pa.uint64())
        fsls = pa.FixedSizeListArray.from_arrays(values, 32 * 1024 * 1024)
        # 3 items, 5 empties, 2 items
        offsets = pa.array([0, 1, 2, 3, 4, 4, 5, 6, 7], pa.int32())
        lists = pa.ListArray.from_arrays(offsets, fsls)

        values2 = pa.array(range(32 * 1024 * 1024 * 8), pa.uint64())
        fsls2 = pa.FixedSizeListArray.from_arrays(values2, 32 * 1024 * 1024)
        offsets2 = pa.array([0, 1, 2, 3, 4, 5, 6, 7, 8], pa.int32())
        lists2 = pa.ListArray.from_arrays(offsets2, fsls2)

        yield pa.record_batch(
            [
                lists,
                lists2,
            ],
            names=["a", "b"],
        )

    schema = pa.schema({"a": list_type, "b": list_type})

    dataset = lance.write_dataset(
        datagen(),
        base_dir,
        schema=schema,
        data_storage_version="stable",
        mode="overwrite",
    )

    dataset.scanner(batch_size=10, io_buffer_size=5000).to_table()

    # We need to make sure that different data files within a fragment are
    # given the same priority.  This is because we scan both files at the same
    # time in order.

    def datagen():
        for i in range(2):
            buf = pa.allocate_buffer(8 * 1024 * 1024)
            arr = pa.FixedSizeBinaryArray.from_buffers(
                pa.binary(1024 * 1024), 8, [None, buf]
            )
            yield pa.record_batch(
                [
                    arr,
                    pa.array(range(8), pa.uint64()),
                ],
                names=["a", "b"],
            )

    schema = pa.schema({"a": pa.binary(1024 * 1024), "b": pa.uint64()})

    dataset = lance.write_dataset(
        datagen(),
        base_dir,
        schema=schema,
        data_storage_version="stable",
        max_rows_per_file=2 * 1024 * 1024,
        mode="overwrite",
    )

    dataset.add_columns({"c": "b"})

    dataset.scanner(batch_size=1, io_buffer_size=5000).to_table()


def test_scan_no_columns(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    df = pd.DataFrame({"a": range(100)})
    dataset = lance.write_dataset(df, base_dir)

    # columns=[] can be used to get just the row ids
    batches = dataset.scanner(columns=[], with_row_id=True).to_batches()

    expected_schema = pa.schema([pa.field("_rowid", pa.uint64())])
    for batch in batches:
        assert batch.schema == expected_schema

    # if with_row_id is not True then columns=[] is an error
    with pytest.raises(ValueError, match="no columns were selected"):
        dataset.scanner(columns=[]).to_table()

    # also test with deleted data to make sure deleted ids not included
    dataset.delete("a = 5")
    num_rows = 0
    for batch in dataset.scanner(columns=[], with_row_id=True).to_batches():
        num_rows += batch.num_rows

    assert num_rows == 99


def test_scan_prefilter(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    vecs = pa.array(
        [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], type=pa.list_(pa.float32(), 2)
    )
    df = pa.Table.from_pydict(
        {
            "index": [1, 2, 3, 4, 5, 6],
            "type": ["a", "a", "a", "b", "b", "b"],
            "vecs": vecs,
        }
    )
    dataset = lance.write_dataset(df, base_dir)
    query = pa.array([1, 1], pa.float32())
    args = {
        "columns": ["index"],
        "filter": "type = 'b'",
        "nearest": {"column": "vecs", "q": pa.array(query), "k": 2, "use_index": False},
    }

    # With post-filter no results are returned because all
    # the closest results don't match the filter
    assert dataset.scanner(**args).to_table().num_rows == 0

    args["prefilter"] = True
    table = dataset.scanner(**args).to_table()

    expected = pa.Table.from_pydict({"index": [4, 5]})

    assert table.column("index") == expected.column("index")

    table = dataset.to_table(**args)
    assert table.column("index") == expected.column("index")

    table = pa.Table.from_batches(dataset.to_batches(**args))
    assert table.column("index") == expected.column("index")


def test_scan_count_rows(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    df = pd.DataFrame({"a": range(42), "b": range(42)})
    dataset = lance.write_dataset(df, base_dir)

    assert dataset.scanner().count_rows() == 42
    assert dataset.count_rows(filter="a < 10") == 10
    assert dataset.count_rows(filter=pa_ds.field("a") < 20) == 20


def test_scanner_schemas(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    df = pd.DataFrame({"a": range(50), "s": [f"s-{i}" for i in range(50)]})

    dataset = lance.write_dataset(df, base_dir)

    scanner = dataset.scanner(columns=["a"])
    assert scanner.dataset_schema == dataset.schema
    assert scanner.projected_schema == pa.schema([pa.field("a", pa.int64())])


def test_custom_commit_lock(tmp_path: Path):
    called_lock = False
    called_release = False

    @contextlib.contextmanager
    def commit_lock(version: int):
        nonlocal called_lock
        nonlocal called_release
        called_lock = True
        assert version == 1
        yield
        called_release = True

    lance.write_dataset(
        pa.table({"a": range(100)}), tmp_path / "test1", commit_lock=commit_lock
    )
    assert called_lock
    assert called_release

    @contextlib.contextmanager
    def commit_lock(_version: int):
        try:
            yield
        finally:
            raise Exception("hello world!")

    with pytest.raises(Exception, match="hello world!"):
        lance.write_dataset(
            pa.table({"a": range(100)}), tmp_path / "test2", commit_lock=commit_lock
        )

    @contextlib.contextmanager
    def commit_lock(_version: int):
        raise CommitConflictError()

    with pytest.raises(Exception, match="CommitConflictError"):
        lance.write_dataset(
            pa.table({"a": range(100)}), tmp_path / "test3", commit_lock=commit_lock
        )

    dataset = lance.dataset(tmp_path / "test1", commit_lock=commit_lock)
    with pytest.raises(Exception, match="CommitConflictError"):
        dataset.delete("a < 10")


def test_dataset_restore(tmp_path: Path):
    data = pa.table({"a": range(100)})
    dataset = lance.write_dataset(data, tmp_path)
    assert dataset.version == 1
    assert dataset.count_rows() == 100

    dataset.delete("a >= 50")
    assert dataset.count_rows() == 50
    assert dataset.version == 2

    dataset = lance.dataset(tmp_path, version=1)
    assert dataset.version == 1

    dataset.restore()
    assert dataset.version == 3
    assert dataset.count_rows() == 100


def test_mixed_mode_overwrite(tmp_path: Path):
    data = pa.table({"a": range(100)})
    dataset = lance.write_dataset(data, tmp_path, data_storage_version="legacy")

    assert dataset.data_storage_version == "0.1"

    dataset = lance.write_dataset(data, tmp_path, mode="overwrite")

    assert dataset.data_storage_version == "0.1"


def test_roundtrip_reader(tmp_path: Path):
    # Can roundtrip a reader
    data = pa.table({"a": range(100)})
    dataset = lance.write_dataset(data, tmp_path / "test1")
    reader = dataset.to_batches()
    lance.write_dataset(reader, tmp_path / "test2", schema=dataset.schema)

    # Can roundtrip a whole dataset
    lance.write_dataset(dataset, tmp_path / "test3")


def test_metadata(tmp_path: Path):
    data = pa.table({"a": range(100)})
    data = data.replace_schema_metadata({"foo": pickle.dumps("foo")})
    with pytest.raises(ValueError):
        lance.write_dataset(data, tmp_path)

    data = data.replace_schema_metadata({"foo": base64.b64encode(pickle.dumps("foo"))})
    lance.write_dataset(data, tmp_path)


def test_default_scan_options(tmp_path: Path):
    data = pa.table({"a": range(100), "b": range(100)})
    dataset = lance.write_dataset(data, tmp_path)
    assert dataset.schema.names == ["a", "b"]

    dataset = lance.dataset(tmp_path)
    assert dataset.schema.names == ["a", "b"]

    dataset = lance.dataset(
        tmp_path,
        default_scan_options={
            "with_row_id": True,
        },
    )
    assert dataset.schema.names == ["a", "b", "_rowid"]

    dataset = lance.dataset(tmp_path, default_scan_options={"with_row_address": True})
    assert dataset.schema.names == ["a", "b", "_rowaddr"]


def test_scan_with_row_ids(tmp_path: Path):
    """Test expose physical row ids in the scanner."""
    data = pa.table({"a": range(1000)})
    ds = lance.write_dataset(data, tmp_path, max_rows_per_file=250)

    tbl = ds.scanner(filter="a % 10 == 0 AND a < 500", with_row_id=True).to_table()
    assert "_rowid" in tbl.column_names
    row_ids = tbl["_rowid"].to_pylist()
    assert row_ids == list(range(0, 250, 10)) + list(range(2**32, 2**32 + 250, 10))

    tbl2 = ds._take_rows(row_ids)
    assert tbl2["a"] == tbl["a"]


@pytest.mark.cuda
def test_random_dataset_recall_accelerated(tmp_path: Path):
    dims = 32
    schema = pa.schema([pa.field("a", pa.list_(pa.float32(), dims), False)])
    values = pc.random(512 * dims).cast("float32")
    table = pa.Table.from_pydict(
        {"a": pa.FixedSizeListArray.from_arrays(values, dims)}, schema=schema
    )

    base_dir = tmp_path / "test"

    dataset = lance.write_dataset(table, base_dir)

    from lance.dependencies import torch

    # create index and assert no rows are uncounted
    dataset.create_index(
        "a",
        "IVF_PQ",
        num_partitions=2,
        num_sub_vectors=32,
        accelerator=torch.device("cpu"),
    )
    validate_vector_index(dataset, "a", pass_threshold=0.5)


@pytest.mark.cuda
def test_random_dataset_recall_accelerated_one_pass(tmp_path: Path):
    dims = 32
    schema = pa.schema([pa.field("a", pa.list_(pa.float32(), dims), False)])
    values = pc.random(512 * dims).cast("float32")
    table = pa.Table.from_pydict(
        {"a": pa.FixedSizeListArray.from_arrays(values, dims)}, schema=schema
    )

    base_dir = tmp_path / "test"

    dataset = lance.write_dataset(table, base_dir)

    from lance.dependencies import torch

    # create index and assert no rows are uncounted
    dataset.create_index(
        "a",
        "IVF_PQ",
        num_partitions=2,
        num_sub_vectors=32,
        accelerator=torch.device("cpu"),
        one_pass_ivfpq=True,
    )
    validate_vector_index(dataset, "a", pass_threshold=0.5)


@pytest.mark.cuda
def test_count_index_rows_accelerated(tmp_path: Path):
    dims = 32
    schema = pa.schema([pa.field("a", pa.list_(pa.float32(), dims), False)])
    values = pc.random(512 * dims).cast("float32")
    table = pa.Table.from_pydict(
        {"a": pa.FixedSizeListArray.from_arrays(values, dims)}, schema=schema
    )

    base_dir = tmp_path / "test"

    dataset = lance.write_dataset(table, base_dir)

    # assert we return None for index name that doesn't exist
    index_name = "a_idx"
    with pytest.raises(KeyError):
        dataset.stats.index_stats(index_name)["num_unindexed_rows"]
    with pytest.raises(KeyError):
        dataset.stats.index_stats(index_name)["num_indexed_rows"]

    from lance.dependencies import torch

    # create index and assert no rows are uncounted
    dataset.create_index(
        "a",
        "IVF_PQ",
        name=index_name,
        num_partitions=2,
        num_sub_vectors=1,
        accelerator=torch.device("cpu"),
    )
    assert dataset.stats.index_stats(index_name)["num_unindexed_rows"] == 0
    assert dataset.stats.index_stats(index_name)["num_indexed_rows"] == 512

    # append some data
    new_table = pa.Table.from_pydict(
        {"a": [[float(i) for i in range(32)] for _ in range(512)]}, schema=schema
    )
    dataset = lance.write_dataset(new_table, base_dir, mode="append")

    # assert rows added since index was created are uncounted
    assert dataset.stats.index_stats(index_name)["num_unindexed_rows"] == 512
    assert dataset.stats.index_stats(index_name)["num_indexed_rows"] == 512


@pytest.mark.cuda
def test_count_index_rows_accelerated_one_pass(tmp_path: Path):
    dims = 32
    schema = pa.schema([pa.field("a", pa.list_(pa.float32(), dims), False)])
    values = pc.random(512 * dims).cast("float32")
    table = pa.Table.from_pydict(
        {"a": pa.FixedSizeListArray.from_arrays(values, dims)}, schema=schema
    )

    base_dir = tmp_path / "test"

    dataset = lance.write_dataset(table, base_dir)

    # assert we return None for index name that doesn't exist
    index_name = "a_idx"
    with pytest.raises(KeyError):
        dataset.stats.index_stats(index_name)["num_unindexed_rows"]
    with pytest.raises(KeyError):
        dataset.stats.index_stats(index_name)["num_indexed_rows"]

    from lance.dependencies import torch

    # create index and assert no rows are uncounted
    dataset.create_index(
        "a",
        "IVF_PQ",
        name=index_name,
        num_partitions=2,
        num_sub_vectors=1,
        accelerator=torch.device("cpu"),
        one_pass_ivfpq=True,
    )
    assert dataset.stats.index_stats(index_name)["num_unindexed_rows"] == 0
    assert dataset.stats.index_stats(index_name)["num_indexed_rows"] == 512

    # append some data
    new_table = pa.Table.from_pydict(
        {"a": [[float(i) for i in range(32)] for _ in range(512)]}, schema=schema
    )
    dataset = lance.write_dataset(new_table, base_dir, mode="append")

    # assert rows added since index was created are uncounted
    assert dataset.stats.index_stats(index_name)["num_unindexed_rows"] == 512
    assert dataset.stats.index_stats(index_name)["num_indexed_rows"] == 512


def test_count_index_rows(tmp_path: Path):
    dims = 32
    schema = pa.schema([pa.field("a", pa.list_(pa.float32(), dims), False)])
    values = pc.random(512 * dims).cast("float32")
    table = pa.Table.from_pydict(
        {"a": pa.FixedSizeListArray.from_arrays(values, dims)}, schema=schema
    )

    base_dir = tmp_path / "test"

    dataset = lance.write_dataset(table, base_dir)

    # assert we return None for index name that doesn't exist
    index_name = "a_idx"
    with pytest.raises(KeyError):
        dataset.stats.index_stats(index_name)["num_unindexed_rows"]
    with pytest.raises(KeyError):
        dataset.stats.index_stats(index_name)["num_indexed_rows"]

    # create index and assert no rows are uncounted
    dataset.create_index(
        "a", "IVF_PQ", name=index_name, num_partitions=2, num_sub_vectors=1
    )
    assert dataset.stats.index_stats(index_name)["num_unindexed_rows"] == 0
    assert dataset.stats.index_stats(index_name)["num_indexed_rows"] == 512

    # append some data
    new_table = pa.Table.from_pydict(
        {"a": [[float(i) for i in range(32)] for _ in range(512)]}, schema=schema
    )
    dataset = lance.write_dataset(new_table, base_dir, mode="append")

    # assert rows added since index was created are uncounted
    assert dataset.stats.index_stats(index_name)["num_unindexed_rows"] == 512
    assert dataset.stats.index_stats(index_name)["num_indexed_rows"] == 512


def test_dataset_progress(tmp_path: Path):
    data = pa.table({"a": range(10)})
    progress = ProgressForTest()
    ds = lance.write_dataset(data, tmp_path, max_rows_per_file=5, progress=progress)
    assert len(ds.get_fragments()) == 2
    assert progress.begin_called == 2
    assert progress.complete_called == 2


def test_tensor_type(tmp_path: Path):
    arr = [[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400]]
    storage = pa.array(arr, pa.list_(pa.float32(), 4))
    tensor_type = pa.fixed_shape_tensor(pa.float32(), [4])
    ext_arr = pa.ExtensionArray.from_storage(tensor_type, storage)
    data = pa.table({"tensor": ext_arr})

    ds = lance.write_dataset(data, tmp_path)

    query_arr = [[10, 20, 30, 40]]
    storage = pa.array(query_arr, pa.list_(pa.float32(), 4))
    ext_arr = pa.ExtensionArray.from_storage(tensor_type, storage)
    ext_scalar = ext_arr[0]

    results = ds.to_table(
        nearest={
            "column": "tensor",
            "k": 1,
            "q": ext_scalar,
        }
    )
    assert results.num_rows == 1


def test_sharded_iterator_fragments(tmp_path: Path):
    arr = pa.array(range(1000))
    tbl = pa.table({"a": arr})
    # Write about 10 files
    lance.write_dataset(tbl, tmp_path, max_rows_per_group=20, max_rows_per_file=100)

    shard_datast = ShardedBatchIterator(tmp_path, 1, 2, columns=["a"])
    batches = pa.concat_arrays([b["a"] for b in shard_datast])
    assert batches == pa.array(
        list(range(100, 200))
        + list(range(300, 400))
        + list(range(500, 600))
        + list(range(700, 800))
        + list(range(900, 1000))
    )


def test_sharded_iterator_batches(tmp_path: Path):
    arr = pa.array(range(1000))
    tbl = pa.table({"a": arr})
    # Write about 10 files
    ds = lance.write_dataset(tbl, tmp_path, max_rows_per_file=100)

    RANK = 1
    WORLD_SIZE = 2
    BATCH_SIZE = 15
    shard_datast = ShardedBatchIterator(
        ds,
        RANK,
        WORLD_SIZE,
        columns=["a"],
        batch_size=BATCH_SIZE,
        granularity="batch",
    )
    batches = pa.concat_arrays([b["a"] for b in shard_datast])
    assert batches == pa.array(
        [
            j
            for i in range(RANK * BATCH_SIZE, 1000, WORLD_SIZE * BATCH_SIZE)
            for j in range(i, i + BATCH_SIZE)
        ]
    )


def test_sharded_iterator_non_full_batch(tmp_path: Path):
    arr = pa.array(range(1186))
    tbl = pa.table({"a": arr})

    ds = lance.write_dataset(tbl, tmp_path)
    shard_datast = ShardedBatchIterator(
        ds,
        1,
        2,
        columns=["a"],
        batch_size=100,
        granularity="batch",
    )
    batches = pa.concat_arrays([b["a"] for b in shard_datast])

    # Can read partial batches
    assert len(set(range(1100, 1186)) - set(batches.to_pylist())) == 0


def test_dynamic_projection(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    dataset = lance.dataset(base_dir)
    table2 = dataset.to_table(
        columns={
            "bool": "a > 5",
        }
    )

    expected = pa.Table.from_pylist([{"bool": False}, {"bool": True}])
    assert expected == table2


def test_migrate_manifest(tmp_path: Path):
    from lance.lance import manifest_needs_migration

    table = pa.table({"x": [1, 2, 3]})
    ds = lance.write_dataset(table, tmp_path)
    # We shouldn't need a migration for a brand new dataset.
    assert not manifest_needs_migration(ds)


def test_legacy_dataset(tmp_path: Path):
    table = pa.table({"a": range(100), "b": range(100)})
    dataset = lance.write_dataset(table, tmp_path, data_storage_version="stable")
    batches = list(dataset.to_batches())
    assert len(batches) == 1
    assert pa.Table.from_batches(batches) == table
    fragment = list(dataset.get_fragments())[0]
    assert "major_version: 2" in format_fragment(fragment.metadata, dataset)
    assert dataset.data_storage_version == "2.0"

    # Append will write v2 if dataset was originally created with v2
    dataset = lance.write_dataset(table, tmp_path, mode="append")

    assert len(dataset.get_fragments()) == 2

    fragment = list(dataset.get_fragments())[1]
    assert "major_version: 2" in format_fragment(fragment.metadata, dataset)

    dataset = lance.write_dataset(
        table, tmp_path, data_storage_version="legacy", mode="overwrite"
    )
    assert dataset.data_storage_version == "0.1"

    fragment = list(dataset.get_fragments())[0]
    assert "major_version: 2" not in format_fragment(fragment.metadata, dataset)

    # Append will write v1 if dataset was originally created with v1
    dataset = lance.write_dataset(
        table, tmp_path, data_storage_version="stable", mode="append"
    )

    fragment = list(dataset.get_fragments())[1]
    assert "major_version: 2" not in format_fragment(fragment.metadata, dataset)

    # Writing an empty table with v2 will put dataset in "v2 mode"
    dataset = lance.write_dataset(
        [],
        tmp_path,
        schema=table.schema,
        data_storage_version="stable",
        mode="overwrite",
    )

    assert len(dataset.get_fragments()) == 0

    dataset = lance.write_dataset(
        table, tmp_path, data_storage_version="legacy", mode="append"
    )

    fragment = list(dataset.get_fragments())[0]
    assert "major_version: 2" in format_fragment(fragment.metadata, dataset)

    # Writing an empty table with v1 will put dataset in "v1 mode"
    dataset = lance.write_dataset(
        [],
        tmp_path,
        schema=table.schema,
        data_storage_version="legacy",
        mode="overwrite",
    )

    assert len(dataset.get_fragments()) == 0

    dataset = lance.write_dataset(
        table, tmp_path, data_storage_version="stable", mode="append"
    )

    fragment = list(dataset.get_fragments())[0]
    assert "major_version: 2" not in format_fragment(fragment.metadata, dataset)


def test_late_materialization_param(tmp_path: Path):
    table = pa.table(
        {
            "filter": np.arange(4),
            "values": pa.array([b"abcd", b"efgh", b"ijkl", b"mnop"]),
        }
    )
    dataset = lance.write_dataset(
        table, tmp_path, data_storage_version="stable", max_rows_per_file=10000
    )
    filt = "filter % 2 == 0"

    assert "(values)" in dataset.scanner(
        filter=filt, late_materialization=None
    ).explain_plan(True)
    assert ", values" in dataset.scanner(
        filter=filt, late_materialization=False
    ).explain_plan(True)
    assert "(values)" in dataset.scanner(
        filter=filt, late_materialization=True
    ).explain_plan(True)
    assert "(values)" in dataset.scanner(
        filter=filt, late_materialization=["values"]
    ).explain_plan(True)
    assert ", values" in dataset.scanner(
        filter=filt, late_materialization=["filter"]
    ).explain_plan(True)

    # These tests just make sure we can pass in the parameter.  There's no great
    # way to know if late materialization happened or not.  That will have to be
    # for benchmarks
    expected = dataset.to_table(filter=filt)
    assert dataset.to_table(filter=filt, late_materialization=None) == expected
    assert dataset.to_table(filter=filt, late_materialization=False) == expected
    assert dataset.to_table(filter=filt, late_materialization=True) == expected
    assert dataset.to_table(filter=filt, late_materialization=["values"]) == expected

    expected = list(dataset.to_batches(filter=filt))
    assert list(dataset.to_batches(filter=filt, late_materialization=None)) == expected
    assert list(dataset.to_batches(filter=filt, late_materialization=False)) == expected
    assert list(dataset.to_batches(filter=filt, late_materialization=True)) == expected
    assert (
        list(dataset.to_batches(filter=filt, late_materialization=["values"]))
        == expected
    )


def test_late_materialization_batch_size(tmp_path: Path):
    table = pa.table({"filter": np.arange(32 * 32), "values": np.arange(32 * 32)})
    dataset = lance.write_dataset(
        table, tmp_path, data_storage_version="stable", max_rows_per_file=10000
    )
    for batch in dataset.to_batches(
        columns=["values"],
        filter="filter % 2 == 0",
        batch_size=32,
        late_materialization=True,
    ):
        assert batch.num_rows == 32


def test_use_scalar_index(tmp_path: Path):
    table = pa.table({"filter": range(100)})
    dataset = lance.write_dataset(table, tmp_path)
    dataset.create_scalar_index("filter", "BTREE")

    assert "MaterializeIndex" in dataset.scanner(filter="filter = 10").explain_plan(
        True
    )
    assert "MaterializeIndex" in dataset.scanner(
        filter="filter = 10", use_scalar_index=True
    ).explain_plan(True)
    assert "MaterializeIndex" not in dataset.scanner(
        filter="filter = 10", use_scalar_index=False
    ).explain_plan(True)


EXPECTED_DEFAULT_STORAGE_VERSION = "2.0"
EXPECTED_MAJOR_VERSION = 2
EXPECTED_MINOR_VERSION = 0


def test_default_storage_version(tmp_path: Path):
    table = pa.table({"x": [0]})
    dataset = lance.write_dataset(table, tmp_path)
    assert dataset.data_storage_version == EXPECTED_DEFAULT_STORAGE_VERSION

    frag = lance.LanceFragment.create(dataset.uri, table)
    sample_file = frag.to_json()["files"][0]
    assert sample_file["file_major_version"] == EXPECTED_MAJOR_VERSION
    assert sample_file["file_minor_version"] == EXPECTED_MINOR_VERSION

    from lance.fragment import write_fragments

    frags = write_fragments(table, dataset.uri)
    frag = frags[0]
    sample_file = frag.to_json()["files"][0]
    assert sample_file["file_major_version"] == EXPECTED_MAJOR_VERSION
    assert sample_file["file_minor_version"] == EXPECTED_MINOR_VERSION


def test_no_detached_v1(tmp_path: Path):
    table = pa.table({"x": [0]})
    dataset = lance.write_dataset(table, tmp_path)

    # Make a detached append
    table = pa.table({"x": [1]})
    frag = lance.LanceFragment.create(dataset.uri, table)
    op = lance.LanceOperation.Append([frag])
    with pytest.raises(OSError, match="v1 manifest paths"):
        dataset.commit(dataset.uri, op, read_version=dataset.version, detached=True)


def test_detached_commits(tmp_path: Path):
    table = pa.table({"x": [0]})
    dataset = lance.write_dataset(table, tmp_path, enable_v2_manifest_paths=True)

    # Make a detached append
    table = pa.table({"x": [1]})
    frag = lance.LanceFragment.create(dataset.uri, table)
    op = lance.LanceOperation.Append([frag])
    detached = dataset.commit(
        dataset.uri, op, read_version=dataset.version, detached=True
    )
    assert (detached.version & 0x8000000000000000) != 0

    assert detached.to_table() == pa.table({"x": [0, 1]})
    # Detached commit should not show up in the dataset
    dataset = lance.dataset(tmp_path)
    assert dataset.to_table() == pa.table({"x": [0]})

    # We can make more commits to dataset and they don't affect attached
    table = pa.table({"x": [2]})
    dataset = lance.write_dataset(table, tmp_path, mode="append")
    assert dataset.to_table() == pa.table({"x": [0, 2]})

    # We can check out the detached commit
    detached = dataset.checkout_version(detached.version)
    assert detached.to_table() == pa.table({"x": [0, 1]})

    # Detached commit can use detached commit as read version
    table = pa.table({"x": [3]})
    frag = lance.LanceFragment.create(detached.uri, table)
    op = lance.LanceOperation.Append([frag])

    detached2 = dataset.commit(
        dataset.uri, op, read_version=detached.version, detached=True
    )

    assert detached2.to_table() == pa.table({"x": [0, 1, 3]})
