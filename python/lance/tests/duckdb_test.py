#!/usr/bin/env python3
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

"""DuckDB Integration Tests"""

from pathlib import Path

import duckdb
import lance
import pandas as pd
import pyarrow as pa


def test_dictionary_type_query(tmp_path: Path):
    df = pd.DataFrame({"class": ["foo", "bar", "foo", "zoo"], "grade": ["A", "B", "B", "A"]})
    # df["class"] = df["class"].astype("category")
    # df["grade"] = df["grade"].astype("category")

    uri = tmp_path / "dict.lance"
    print(pa.Table.from_pandas(df, preserve_index=False).schema)
    lance.write_table(pa.Table.from_pandas(df), tmp_path / "dict.lance", "class")
    # print(df)

    ds = lance.dataset(uri)
    print(duckdb.query("SELECT * FROM ds"))
    print(duckdb.query("SELECT count(class), class FROM ds GROUP BY 2").to_df())
