# Copyright 2022 Lance Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from lance import LanceFileFormat, dataset, write_table


def test_simple_round_trips(tmp_path: Path):
    table = pa.Table.from_pandas(
        pd.DataFrame({"label": [123, 456, 789], "values": [22, 33, 2.24]})
    )
    write_table(table, tmp_path / "test.lance")

    assert (tmp_path / "test.lance").exists()

    ds = dataset(str(tmp_path / "test.lance"))
    actual = ds.to_table()

    assert table == actual


def test_head(tmp_path: Path):
    table = pa.Table.from_pandas(
        pd.DataFrame({"label": [123, 456, 789], "values": [22, 33, 2.24]})
    )
    write_table(table, tmp_path / "test.lance")
    ds = dataset(str(tmp_path / "test.lance"))
    actual = ds.head(2)
    assert table[:2] == actual


def test_write_categorical_values(tmp_path: Path):
    df = pd.DataFrame({"label": ["cat", "cat", "dog", "person"]})
    df["label"] = df["label"].astype("category")
    table = pa.Table.from_pandas(df)
    write_table(table, tmp_path / "test.lance")

    assert (tmp_path / "test.lance").exists()

    actual = dataset(str(tmp_path / "test.lance")).to_table()
    assert table == actual


def test_write_dataset(tmp_path: Path):
    table = pa.Table.from_pandas(
        pd.DataFrame(
            {
                "label": [123, 456, 789],
                "values": [22, 33, 2.24],
                "split": ["a", "b", "a"],
            }
        )
    )
    ds.write_dataset(table, tmp_path, partitioning=["split"], format=LanceFileFormat())

    part_dirs = [d.name for d in tmp_path.iterdir()]
    assert set(part_dirs) == set(["a", "b"])
    part_a = list((tmp_path / "a").glob("*.lance"))[0]
    actual = dataset(part_a).to_table()
    assert actual == pa.Table.from_pandas(
        pd.DataFrame({"label": [123, 789], "values": [22, 2.24]})
    )
