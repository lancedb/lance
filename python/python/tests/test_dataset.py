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

import time
from datetime import datetime
from pathlib import Path

import lance
import pandas as pd
import pyarrow as pa


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

    v2, v1 = lance.dataset(base_dir).versions()
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
    ds = lance.dataset(base_dir).asof(ts_1)
    assert ds.version == 1
    assert len(ds.to_table()) == 3

    # check that the first and second batch are present
    ds = lance.dataset(base_dir).asof(ts_2)
    assert ds.version == 2
    assert len(ds.to_table()) == 6

    # check that all batches are present
    ds = lance.dataset(base_dir).asof(ts_3)
    assert ds.version == 3
    assert len(ds.to_table()) == 9


def test_version_tagging(tmp_path: Path):
    table = pa.Table.from_pydict({"colA": [1, 2, 3], "colB": [4, 5, 6]})
    base_dir = tmp_path / "test"

    # check for retrieving the dataset by tag
    lance.write_dataset(table, base_dir, mode="create", tag="release")
    ds = lance.dataset(base_dir).with_tag("release")
    assert ds.version == 1
    assert len(ds.to_table()) == 3

    # check that an older version can be retrieved by tag
    lance.write_dataset(table, base_dir, mode="append")
    ds = lance.dataset(base_dir).with_tag("release")
    assert ds.version == 1
    assert len(ds.to_table()) == 3

    # check that distinct tags can be used for the same dataset
    lance.write_dataset(table, base_dir, mode="append", tag="production")
    ds = lance.dataset(base_dir).with_tag("production")
    assert ds.version == 3
    assert len(ds.to_table()) == 9

    # check the same tag can be reused and the latest version is retrieved
    lance.write_dataset(table, base_dir, mode="append", tag="release")
    ds = lance.dataset(base_dir).with_tag("release")
    assert ds.version == 4
    assert len(ds.to_table()) == 12


def test_take(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    dataset = lance.dataset(base_dir)
    table2 = dataset.take([0, 1])

    assert isinstance(table2, pa.Table)
    assert table2 == table1
