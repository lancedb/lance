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
import lance
import pytest

pytest.importorskip("torch")

import torch
import pyarrow as pa
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader

from lance.pytorch.data import LanceDataset


def test_data_loader(tmp_path: Path):
    torch.Tensor([1, 2, 3])
    ids = pa.array(range(10))
    values = pa.array(range(10, 20))
    tab = pa.Table.from_arrays([ids, values], names=["id", "value"])
    print(tab)
    print(tab.take([0]))
    print(ids)

    lance.write_table(tab, tmp_path / "lance")

    dataset = LanceDataset(tmp_path / "lance", batch_size=4)
    print(dataset)
    id_batch, value_batch = next(iter(dataset))
    assert(id_batch.shape == (1, 4))


