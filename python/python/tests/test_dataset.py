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

    v1, v2 = lance.dataset(base_dir).versions()
    assert isinstance(v1["timestamp"], datetime)
    assert isinstance(v2["timestamp"], datetime)
    assert v1["timestamp"] < v2["timestamp"]
    assert isinstance(v1["metadata"], dict)
    assert isinstance(v2["metadata"], dict)

def test_take(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    dataset = lance.dataset(base_dir)
    table2 = dataset.take([0,1])

    assert isinstance(table2, pa.Table)
    assert table2 == table1