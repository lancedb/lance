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

import logging
from typing import List, Optional, Union

import numpy as np
import pyarrow as pa
import torch
from tqdm import tqdm

from . import preferred_device
from .distance import cosine_distance, dot_distance, l2_distance

__all__ = ["KMeans"]


def _random_init(data: np.ndarray, n: int, seed: Optional[int] = None) -> torch.Tensor:
    if seed is not None:
        np.random.seed(seed)
    sample_idx = np.random.randint(0, data.shape[0], (n,))
    samples = data[sample_idx]
    return samples


@torch.jit.script
def _new_centroids_mps(
    part_ids: List[torch.Tensor],
    k: int,
    data: List[torch.Tensor],
    cnts: torch.Tensor,
) -> torch.Tensor:
    # MPS does not have Torch.index_reduce_()
    # See https://github.com/pytorch/pytorch/issues/77764

    new_centroids = torch.zeros((k, data[0].shape[1]), device=data[0].device)
    for ids, chunk in zip(part_ids, data):
        new_centroids.index_add_(0, ids, chunk)
    for idx, cnt in enumerate(cnts.cpu()):
        if cnt > 0:
            new_centroids[idx, :] = new_centroids[idx, :].div(cnt)
    return new_centroids.to(data[0].device)


class KMeans:
    """K-Means trains over vectors and divide into K clusters.

    This implement is built on PyTorch, supporting CPU, GPU and Apple Silicon GPU.

    Parameters
    ----------
    k: int
        The number of clusters
    metric : str
        Metric type, support "l2", "cosine" or "dot"
    init: str
        Initialization method. Only support "random" now.
    max_iters: int
        Max number of iterations to train the kmean model.
    tolerance: float
        Relative tolerance with regards to Frobenius norm of the difference in
        the cluster centers of two consecutive iterations to declare convergence.
    seed: int, optional
        Random seed
    device: str, optional
        The device to run the PyTorch algorithms. Default we will pick
        the most performant device on the host. See `lance.torch.preferred_device()`
    """

    def __init__(
        self,
        k: int,
        *,
        metric: str = "l2",
        init: str = "random",
        max_iters: int = 50,
        tolerance: float = 1e-4,
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

        self.total_distance = 0
        self.centroids: Optional[torch.Tensor] = None
        self.init = init
        self.device = preferred_device(device)
        self.tolerance = tolerance
        self.seed = seed

    def __repr__(self):
        return f"KMeans(k={self.k}, metric={self.metric}, device={self.device})"

    def _to_tensor(
        self, data: Union[pa.FixedSizeListArray, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
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
        """Fit - Train the kmeans models.

        Parameters
        ----------
        data : pa.FixedSizeListArray, np.ndarray, or torch.Tensor
            2-D input data to train kmeans.

        """
        assert self.centroids is None

        if isinstance(data, pa.FixedSizeListArray):
            arr = np.stack(data.to_numpy(zero_copy_only=False))
        elif isinstance(data, torch.Tensor):
            arr = data.cpu().numpy()
        elif isinstance(data, np.ndarray):
            arr = data
        else:
            raise ValueError(
                "Input data can only be FixedSizeListArray"
                + f" or numpy ndarray: got {type(data)}"
            )

        if self.init == "random":
            self.centroids = self._to_tensor(_random_init(arr, self.k, self.seed))
        else:
            raise ValueError("KMeans::fit: only random initialization is supported.")

        chunks = [
            torch.from_numpy(c).to(self.device)
            for c in np.vsplit(arr, int(np.ceil(arr.shape[0] / 65536)))
        ]
        del arr
        del data

        self.total_distance = 0
        for i in tqdm(range(self.max_iters)):
            try:
                self.total_distance = self._fit_once(chunks, self.total_distance)
            except StopIteration:
                break
            if i % 10 == 0:
                logging.debug("Total distance: %s, iter: %s", self.total_distance, i)

    @staticmethod
    def _split_centroids(centroids: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        for idx, cnt in enumerate(counts.cpu()):
            if cnt == 0:
                max_idx = torch.argmax(counts).item()
                half_cnt = counts[max_idx] // 2
                counts[idx], counts[max_idx] = half_cnt, half_cnt
                centroids[idx] = centroids[max_idx] * 1.05
                centroids[max_idx] = centroids[max_idx] / 1.05
        return centroids

    @staticmethod
    def _count_rows_in_clusters(part_ids: List[torch.Tensor], k: int) -> torch.Tensor:
        max_len = max([len(ids) for ids in part_ids])
        ones = torch.ones(max_len).to(part_ids[0].device)
        num_rows = torch.zeros(k).to(part_ids[0].device)
        for part_id in part_ids:
            num_rows.scatter_add_(0, part_id, ones)
        return num_rows

    def _fit_once(self, chunks: List[torch.Tensor], last_dist=0) -> float:
        """Train KMean once and return the total distance.

        Parameters
        ----------
        chunks : List[torch.Tensor]
            A list of 2-D tensors, each tensor is a chunk of the input data.

        Returns
        -------
        float
            The total distance of the current centroids and the input data.
        """
        device = chunks[0].device
        part_ids = []
        total_dist = 0
        for chunk in chunks:
            ids, dists = self._transform(chunk)
            total_dist += dists.sum().item()
            part_ids.append(ids)

        if abs(total_dist - last_dist) / total_dist < self.tolerance:
            raise StopIteration("kmeans: converged")

        num_rows = self._count_rows_in_clusters(part_ids, self.k)
        if self.device.type == "cuda":
            new_centroids = torch.zeros_like(self.centroids, device=device)
            for ids, chunk in zip(part_ids, chunks):
                new_centroids.index_reduce_(
                    0, ids, chunk, reduce="mean", include_self=False
                )
        else:
            new_centroids = _new_centroids_mps(part_ids, self.k, chunks, num_rows)

        self.centroids = self._split_centroids(new_centroids, num_rows)
        return total_dist

    def _transform(self, data: torch.Tensor) -> torch.Tensor:
        dists = self.dist_func(data, self.centroids)
        min_idx = torch.argmin(dists, dim=1, keepdim=True)
        dists = dists.take_along_dim(min_idx, dim=1)
        return min_idx.reshape((-1)), dists.reshape(-1)

    def transform(
        self, data: Union[pa.Array, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        assert self.centroids is not None

        data = self._to_tensor(data)
        return self._transform(data)[0]
