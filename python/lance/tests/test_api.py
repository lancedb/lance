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

from lance import dataset, write_table


def test_simple_round_trips(tmp_path: Path):
    table = pa.Table.from_pandas(
        pd.DataFrame({"label": [123, 456, 789], "values": [22, 33, 2.24]})
    )
    write_table(table, tmp_path / "test.lance")

    assert (tmp_path / "test.lance").exists()

    ds = dataset(str(tmp_path / "test.lance"))
    actual = ds.to_table()

    assert table == actual


def test_write_categorical_values(tmp_path: Path):
    df = pd.DataFrame({"label": ["cat", "cat", "dog", "person"]})
    df["label"] = df["label"].astype("category")
    table = pa.Table.from_pandas(df)
    write_table(table, tmp_path / "test.lance")

    assert (tmp_path / "test.lance").exists()

    actual = dataset(str(tmp_path / "test.lance")).to_table()
    assert table == actual
