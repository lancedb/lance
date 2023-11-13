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

import shutil

import lance
import numpy as np
import pyarrow as pa
import torch
from lance.torch.data import LanceDataset


def test_iter_over_dataset(tmp_path):
    # 10240 of 32-d vectors.
    data = np.random.random(10240 * 32).astype("f")

    fsl = pa.FixedSizeListArray.from_arrays(data, 32)
    ids = pa.array(range(0, 10240), type=pa.int32())
    tbl = pa.Table.from_arrays([ids, fsl], ["ids", "vec"])

    ds = lance.write_dataset(tbl, tmp_path / "data.lance")

    torch_ds = LanceDataset(
        ds, batch_size=256, samples=2048, columns=["ids", "vec"], cache=True
    )

    for batch in torch_ds:
        assert set(batch.keys()) == {"ids", "vec"}
        assert batch["ids"].dtype == torch.int32
        assert batch["vec"].shape[1] == 32

    shutil.rmtree(tmp_path / "data.lance")

    # it should read from cache this time.
    for batch in torch_ds:
        assert set(batch.keys()) == {"ids", "vec"}
        assert batch["ids"].dtype == torch.int32
        assert batch["vec"].shape[1] == 32
