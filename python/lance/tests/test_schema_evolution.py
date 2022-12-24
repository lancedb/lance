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

import datetime
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import lance


def test_write_versioned_dataset(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    dataset = lance.dataset(base_dir)
    new_dataset = dataset.append_column(
        lambda x: pa.array([f"a{i}" for i in range(len(x))]),
        field=pa.field("c", pa.utf8()),
    )

    actual_df = new_dataset.to_table().to_pandas()
    expected_df = pd.DataFrame({"a": [1, 10], "b": [2, 20], "c": ["a0", "a1"]})
    pd.testing.assert_frame_equal(expected_df, actual_df)


def test_column_projection(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    dataset = lance.dataset(base_dir)

    def value_func(x: pa.Table):
        assert x.num_columns == 1
        assert x.column_names == ["a"]
        return pa.array([str(i) for i in x.column("a")])

    new_dataset = dataset.append_column(
        value_func,
        field=pa.field("c", pa.utf8()),
        columns=["a"],
        metadata={"author": "me"},
    )

    actual_df = new_dataset.to_table().to_pandas()
    expected_df = pd.DataFrame({"a": [1, 10], "b": [2, 20], "c": ["1", "10"]})
    pd.testing.assert_frame_equal(expected_df, actual_df)
    assert new_dataset.version["metadata"] == {"author": "me"}


def test_add_column_with_literal(tmp_path: Path):
    table = pa.Table.from_pylist([{"a": i} for i in range(10)])

    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)
    dataset = lance.dataset(base_dir)
    assert dataset.version["version"] == 1
    # Created within 10 seconds, in UDT
    ts = dataset.version["timestamp"]
    with pytest.raises(TypeError):
        # can't compare offset-naive and offset-aware datetimes
        assert ts > datetime.datetime.utcnow() - datetime.timedelta(0, 10)

    assert ts > datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(
        0, 10
    )
    assert ts > datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc)

    time.sleep(1)
    new_dataset = dataset.append_column(
        pc.scalar(0.5), field=pa.field("b", pa.float64()), metadata={"track": "id"}
    )

    assert new_dataset.version["version"] == 2
    assert new_dataset.version["timestamp"] > dataset.version["timestamp"]
    assert new_dataset.version["metadata"] == {"track": "id"}
    actual_df = new_dataset.to_table().to_pandas()
    expected_df = table.to_pandas()
    expected_df["b"] = 0.5
    pd.testing.assert_frame_equal(actual_df, expected_df)


def test_add_column_with_compute(tmp_path: Path):
    table = pa.Table.from_pylist([{"a": i} for i in range(10)])

    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)
    dataset = lance.dataset(base_dir)
    new_dataset = dataset.append_column(
        pc.Expression._call("power", [pc.field("a"), pc.scalar(2)]),
        field=pa.field("b", pa.int64()),
    )

    assert new_dataset.version["version"] == 2
    actual_df = new_dataset.to_table().to_pandas()
    expected_df = table.to_pandas()
    expected_df["b"] = expected_df["a"] * expected_df["a"]
    pd.testing.assert_frame_equal(actual_df, expected_df)


def test_add_column_with_table(tmp_path: Path):
    table = pa.Table.from_pylist([{"a": i, "b": str(i)} for i in range(10)])
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)
    dataset = lance.dataset(base_dir)

    new_table = pa.Table.from_pylist([{"a": i, "c": i * 10} for i in range(10)])
    new_dataset = dataset.merge(new_table, left_on="a", right_on="a")
    assert new_dataset.version["version"] == 2
    assert new_dataset.version["metadata"] == {}
    actual_df = new_dataset.to_table().to_pandas()
    expected_df = table.to_pandas()
    expected_df["c"] = new_table.column("c").to_numpy()
    pd.testing.assert_frame_equal(actual_df, expected_df)


def test_add_column_with_table_with_metadata(tmp_path: Path):
    table = pa.Table.from_pylist([{"a": i, "b": str(i)} for i in range(10)])
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)
    dataset = lance.dataset(base_dir)

    new_table = pa.Table.from_pylist([{"a": i, "c": i * 10} for i in range(10)])
    new_dataset = dataset.merge(
        new_table, left_on="a", right_on="a", metadata={"k1": "v2"}
    )
    assert new_dataset.version["version"] == 2
    assert new_dataset.version["metadata"] == {"k1": "v2"}

    new_table = pa.Table.from_pylist([{"a": i, "d": i * 10} for i in range(10)])
    new_dataset = dataset.merge(new_table, left_on="a", right_on="a")
    assert new_dataset.version["version"] == 3
    # Version metadata is not inheritable
    assert new_dataset.version["metadata"] == {}
