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

from typing import Optional, Union

import numpy as np
import pyarrow as pa
import torch
from torch.func import vmap

from . import preferred_device
from .distance import cosine_distance, dot_distance, l2_distance

__all__ = ["KMeans"]


# @torch.jit.script
def _random_init(data: torch.Tensor, n: int) -> torch.Tensor:
    sample_idx = torch.randint(0, data.shape[0], (n,))
    samples = data[sample_idx]
    return samples


class KMeans:
    def __init__(
        self,
        k: int,
        *,
        metric: str = "l2",
        max_iters: int = 50,
        redo: int = 2,
        init_method: str = "random",
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.k = k
        self.max_iters = max_iters

        metric = metric.lower()
        self.metric = metric
        if metric in ["l2", "euclidean"]:
            self.dist_func = l2_distance
        elif metric == "cosine":
            self.dist_func = cosine_distance
        elif metric == "dot":
            self.dist_func = dot_distance
        else:
            raise ValueError(
                f"Only l2/cosine/dot is supported as metric type, got: {metric}"
            )

        self.redo = redo
        self.centroids: Optional[torch.Tensor] = None
        self.init_method = init_method
        self.device = preferred_device(device)

    def __repr__(self):
        return f"KMeans(k={self.k}, metric={self.metric}, device={self.device})"

    def _to_tensor(self, data: Union[pa.FixedSizeListArray, np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(data, pa.FixedSizeListArray):
            data = torch.from_numpy(np.stack(data.to_numpy(zero_copy_only=False)))
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        elif isinstance(data, torch.Tensor):
            # Good type
            pass
        else:
            raise ValueError(
                "KMeans::fit accepts pyarrow FixedSizeListArray"
                + f"np.ndarray or torch.Tensor, got: {type(data)}"
            )

        data = data.to(self.device)
        return data

    def fit(self, data: Union[pa.FixedSizeListArray, np.ndarray, torch.Tensor]):
        assert self.centroids is None
        data = self._to_tensor(data)

        if self.init_method == "random":
            self.centroids = _random_init(data, self.k)

        for i in range(self.max_iters):
            self._fit_once(data)

    def _fit_once(self, data):
        assert self.centroids is not None
        part_ids = self.transform(data)
        # compute new centroids
        new_centroids = []
        for i in range(self.k):
            parted_data = data[part_ids == i]
            new_centroids.append(parted_data.mean(dim=0))

        self.centroids = torch.stack(new_centroids)

    def transform(self, data: Union[pa.Array, np.ndarray, torch.Tensor]):
        assert self.centroids is not None

        data = self._to_tensor(data)
        dists = self.dist_func(data, self.centroids)
        part_ids = dists.argmin(dim=1)
        return part_ids
