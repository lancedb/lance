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

import torch
from lance.torch.bench_utils import sort_tensors


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
