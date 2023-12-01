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

import lance
import numpy as np
import pyarrow as pa
import torch
from lance.torch.bench_utils import ground_truth, sort_tensors
from lance.torch.distance import pairwise_l2


def test_sort_tensor():
    ids = torch.tensor([[5, 7, 3, 9, 1], [10, 2, 4, 6, 8]], dtype=torch.float32)
    values = ids.clone()

    sorted_vals, sorted_ids = sort_tensors(values, ids, 3)

    assert torch.allclose(
        sorted_vals, torch.tensor([[1, 3, 5], [2, 4, 6]], dtype=torch.float32)
    )
    assert torch.allclose(
        sorted_ids, torch.tensor([[1, 3, 5], [2, 4, 6]], dtype=torch.float32)
    )


def test_ground_truth(tmp_path: Path):
    N = 1000
    NUM_QUERIES = 50
    DIM = 128

    device = "cpu"  # Github action friendly.
    data = np.random.rand(N * DIM).astype(np.float32)
    fsl = pa.FixedSizeListArray.from_arrays(data, DIM)
    data = torch.from_numpy(data.reshape((-1, DIM))).to(device)

    tbl = pa.Table.from_arrays([fsl], ["vec"])

    ds = lance.write_dataset(tbl, tmp_path)

    idx = np.random.choice(range(N), NUM_QUERIES)
    keys = data[idx, :]

    gt = ground_truth(ds, "vec", keys, k=20, batch_size=128, device=device)
    gt, _ = torch.sort(gt, dim=1)

    actual_dists = pairwise_l2(keys, data)
    expected, _ = torch.sort(torch.argsort(actual_dists, 1)[:, :20], dim=1)

    assert torch.allclose(expected, gt)
