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
import time
from datetime import datetime
from pathlib import Path
from unittest import mock

import lance
import pandas as pd
import pyarrow as pa
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
