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

import lance
import pyarrow as pa
import pandas as pd


def test_write_versioned_dataset(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    dataset = lance.dataset(base_dir)
    new_dataset = dataset.append_column(pa.field("c", pa.utf8()), lambda x: pa.array([f"a{i}" for i in range(len(x))]))

    actual_df = new_dataset.to_table().to_pandas()
    expected_df = pd.DataFrame({"a": [1, 10], "b": [2, 20], "c": ["a0", "a1"]})
    pd.testing.assert_frame_equal(expected_df, actual_df)
