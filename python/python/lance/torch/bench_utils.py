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

"""Benchmark Utilities built on PyTorch"""

from typing import Optional, Union

import numpy as np
import torch

from .. import LanceDataset
from . import preferred_device


def ground_truth(
    ds: LanceDataset,
    column: str,
    query: Union[torch.Tensor, np.ndarray],
    metric_type: str = "L2",
    k: int = 100,
    batch_size: int = 10240,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Find ground truth from dataset.

    Parameters
    ----------
    ds: LanceDataset
        The dataset to test.
    column: str
        The name of the vector column.
    query: 2-D vectors
        A 2-D query vectors, with the shape of [N, dimension].
    k: int
        The number of the nearest vectors to collect for each query vector.
    metric_type: str
        Metric type. How to compute distance, accepts L2 or cosine.
    batch_size: int
        Batch size to read from the input dataset.

    Returns
    -------
    a 2-D array of row_ids for the nearest vectors from each query vector.
    """
    device = preferred_device(device=device)

    if isinstance(query, np.ndarray):
        query = torch.from_numpy(query)
    query = query.to(device)

    for batch in ds.to_batches(
        columns=[column], batch_size=batch_size, with_row_id=True
    ):
        print(batch)
        vectors = torch.from_numpy(
            np.stack(batch[column].to_numpy(zero_copy_only=False))
        )
        vectors = vectors.to(device)
        dists = l2_distance(query, vectors)
        print(dists)
        pass
