#  Copyright (c) 2022. Lance Developers
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

from pathlib import Path

import pandas as pd

import lance
import pyarrow as pa
import pyarrow.compute as pc


def test_write_versioned_dataset(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    dataset = lance.dataset(base_dir)
    new_dataset = dataset.append_column(pa.field("c", pa.utf8()), lambda x: pa.array([f"a{i}" for i in range(len(x))]))

    actual_df = new_dataset.to_table().to_pandas()
    expected_df = pd.DataFrame({"a": [1, 10], "b": [2, 20], "c": ["a0", "a1"]})
    pd.testing.assert_frame_equal(expected_df, actual_df)


def test_add_column_with_literal(tmp_path: Path):
    table = pa.Table.from_pylist([{"a": i} for i in range(10)])

    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)
    dataset = lance.dataset(base_dir)
    new_dataset = dataset.append_column(pa.field("b", pa.float64()), pc.scalar(0.5))

    assert new_dataset.version["version"] == 2
    actual_df = new_dataset.to_table().to_pandas()
    expected_df = table.to_pandas()
    expected_df["b"] = 0.5
    pd.testing.assert_frame_equal(actual_df, expected_df)


def test_add_column_with_compute(tmp_path: Path):
    table = pa.Table.from_pylist([{"a": i} for i in range(10)])

    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)
    dataset = lance.dataset(base_dir)
    new_dataset = dataset.append_column(pa.field("b", pa.int64()),
                                        pc.Expression._call("power", [pc.field("a"), pc.scalar(2)]))

    assert new_dataset.version["version"] == 2
    actual_df = new_dataset.to_table().to_pandas()
    expected_df = table.to_pandas()
    expected_df["b"] = expected_df["a"] * expected_df["a"]
    pd.testing.assert_frame_equal(actual_df, expected_df)
