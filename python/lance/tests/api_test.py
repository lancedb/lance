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


import pandas as pd
import pyarrow as pa
from pathlib import Path
from lance import write_table


def test_write_table(tmp_path: Path):
    table = pa.Table.from_pandas(pd.DataFrame({"label": [123]}))
    write_table(table, tmp_path / "test.lance", "label")

    assert (tmp_path / "test.lance").exists()