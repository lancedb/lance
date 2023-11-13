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

from pathlib import Path

import numpy as np
import pyarrow as pa

import lance
from lance.sampler import maybe_sample


def test_sample_dataset(tmp_path: Path):
    # 10240 of 32-d vectors.
    data = np.random.random(10240 * 32).astype("f")

    fsl = pa.FixedSizeListArray.from_arrays(data, 32)
    tbl = pa.Table.from_arrays([fsl], ["vec"])

    ds = lance.write_dataset(tbl, tmp_path / "data.lance")

    # Simple path
    simple_scan = list(maybe_sample(ds, 128, ["vec"]))
    assert len(simple_scan) == 1
    assert isinstance(simple_scan[0], pa.RecordBatch)
    assert simple_scan[0].schema == pa.schema([pa.field("vec", fsl.type)])
    assert simple_scan[0].num_rows == 128

    # Random path.
    large_scan = list(maybe_sample(ds, 128, ["vec"], max_takes=32))
    assert len(large_scan) == 1
    assert isinstance(large_scan[0], pa.RecordBatch)
    assert large_scan[0].schema == pa.schema([pa.field("vec", fsl.type)])
    assert large_scan[0].num_rows == 128
