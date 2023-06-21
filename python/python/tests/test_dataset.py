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

import os
import pickle
import platform
import time
import uuid
from datetime import datetime
from pathlib import Path
from unittest import mock

import lance
import lance.fragment
import pandas.testing as tm
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


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
        lance.write_dataset(table2, base_dir, mode="append")


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


def test_asof_checkout(tmp_path: Path):
    table = pa.Table.from_pydict({"colA": [1, 2, 3], "colB": [4, 5, 6]})
    base_dir = tmp_path / "test"

    lance.write_dataset(table, base_dir)
    assert len(lance.dataset(base_dir).versions()) == 1
    time.sleep(0.01)
    ts_1 = datetime.now()
    time.sleep(0.01)

    lance.write_dataset(table, base_dir, mode="append")
    assert len(lance.dataset(base_dir).versions()) == 2
    ts_2 = datetime.now()
    time.sleep(0.01)

    lance.write_dataset(table, base_dir, mode="append")
    assert len(lance.dataset(base_dir).versions()) == 3
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


def test_take(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    dataset = lance.dataset(base_dir)
    table2 = dataset.take([0, 1])

    assert isinstance(table2, pa.Table)
    assert table2 == table1


def test_filter(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    actual_tab = dataset.to_table(columns=["a"], filter=(pa.compute.field("b") > 50))
    assert actual_tab == pa.Table.from_pydict({"a": range(51, 100)})


def test_limit_offset(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)
    dataset = lance.dataset(base_dir)

    # test just limit
    assert dataset.to_table(limit=10) == table.slice(0, 10)

    # test just offset
    assert dataset.to_table(offset=10) == table.slice(10, 100)

    # test both
    assert dataset.to_table(offset=10, limit=10) == table.slice(10, 10)


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


def test_pickle(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    pickled = pickle.dumps(dataset)
    unpickled = pickle.loads(pickled)
    assert dataset.to_table() == unpickled.to_table()


def test_polar_scan(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    polars_df = pl.scan_pyarrow_dataset(dataset)
    df = dataset.to_table().to_pandas()
    tm.assert_frame_equal(polars_df.collect().to_pandas(), df)


def test_get_fragments(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    fragment = dataset.get_fragments()[0]
    assert fragment.count_rows() == 100

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


def test_add_columns(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    fragments = dataset.get_fragments()

    fragment = fragments[0]

    def adder(batch: pa.RecordBatch) -> pa.RecordBatch:
        c_array = pa.compute.multiply(batch.column(0), 2)
        return pa.RecordBatch.from_arrays([c_array], names=["c"])

    fragment = fragment.add_columns(adder, columns=["a"])
    schema = dataset.schema.append(pa.field("c", pa.int64()))
    assert fragment.schema() == schema

    dataset = lance.LanceDataset._commit(base_dir, schema, [fragment])
    tbl = dataset.to_table()
    assert tbl == pa.Table.from_pydict(
        {"a": range(100), "b": range(100), "c": pa.array(range(0, 200, 2), pa.int64())}
    )


def test_create_from_fragments(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    fragment = lance.fragment.LanceFragment.create(base_dir, 1, table)

    dataset = lance.LanceDataset._commit(base_dir, table.schema, [fragment])
    tbl = dataset.to_table()
    assert tbl == table


def test_data_files(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    fragment = lance.fragment.LanceFragment.create(base_dir, 1, table)

    data_files = fragment.data_files()
    assert len(data_files) == 1
    # it is a valid uuid
    uuid.UUID(os.path.splitext(data_files[0].path())[0])


def test_commit_fragments_via_scanner(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    parquet_dir = tmp_path / "parquet.parquet"
    pq.write_to_dataset(table, parquet_dir)

    base_dir = tmp_path / "test"
    scanner = pa.dataset.dataset(parquet_dir).scanner()
    fragment = lance.fragment.LanceFragment.create(base_dir, 1, scanner)
    assert fragment.schema() == table.schema

    # Pickle-able
    pickled = pickle.dumps(fragment)
    unpickled = pickle.loads(pickled)
    assert fragment == unpickled

    dataset = lance.LanceDataset._commit(base_dir, table.schema, [fragment])
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
        {"a": range(100), "b": range(100), "c": range(100)}
    )

    # accepts a partial for string
    new_tab = pa.table({"a2": range(5), "d": ["a", "b", "c", "d", "e"]})
    dataset.merge(new_tab, left_on="a", right_on="a2")
    assert dataset.version == 3
    assert dataset.to_table() == pa.table(
        {
            "a": range(100),
            "b": range(100),
            "c": range(100),
            "d": ["a", "b", "c", "d", "e"] + [None] * 95,
        }
    )
