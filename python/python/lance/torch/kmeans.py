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
import time
from typing import List, Optional, Tuple, Union, Generator

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm

import lance
from ..sampler import reservoir_sampling
from . import preferred_device
from .distance import cosine_distance, dot_distance, l2_distance

__all__ = ["KMeans"]

# @torch.jit.script
# def _random_init(data: IterableDataset, n: int) -> torch.Tensor:
def _random_init(data: lance.LanceDataset, n: int) -> torch.Tensor:
    print("Random init on : ", data)

    def iter_over_batches() -> Generator[torch.Tensor, None, None]:
        for batch in data:
            for row in batch:
                yield row.cpu()
                del row
            del batch

    choices = np.random.choice(range(0, len(data)), size=n)
    start = time.time()
    # centroids = reservoir_sampling(iter_over_batches(), n)
    tbl =data.take(sorted(choices), columns=["vec"]).combine_chunks()
    fsl = tbl.to_batches()[0]["vec"]
    centroids = np.stack(fsl.to_numpy(zero_copy_only=False))

    return torch.tensor(centroids)


class _BatchTensorDataset(IterableDataset):
    def __init__(
        self,
        tensor: Union[pa.FixedSizeListArray, torch.Tensor, np.ndarray],
        batch_size: int,
    ):
        super().__init__()
        if isinstance(tensor, pa.FixedSizeListArray):
            tensor = tensor.to_numpy(zero_copy_only=False)
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)

        self.tensor = tensor.cpu()
        self._batch_size = batch_size

    def __iter__(self):
        total_rows = self.tensor.shape[0]
        for i in range(0, total_rows, self._batch_size):
            yield self.tensor[i : min(total_rows - i, self._batch_size)]

    def __len__(self):
        return self.tensor.shape[0]


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

    def fit(
        self,
        data: Union[IterableDataset, pa.FixedSizeListArray, torch.Tensor, np.ndarray],
    ):
        """Fit - Train the kmeans models.

        Parameters
        ----------
        data : pa.FixedSizeListArray, np.ndarray, or torch.Tensor
            2-D input data to train kmeans.

        """
        start = time.time()
        assert self.centroids is None

        if isinstance(data, (pa.FixedSizeListArray, torch.Tensor, np.ndarray)):
            data = _BatchTensorDataset(data, 20480)
        elif isinstance(data, IterableDataset):
            data = data
        else:
            raise ValueError(
                "Input data can only be FixedSizeListArray"
                + f" or numpy ndarray, or torch.Dataset: got {type(data)}"
            )

        if self.init == "random":
            self.centroids = self._to_tensor(_random_init(data.dataset, self.k))
        else:
            raise ValueError("KMeans::fit: only random initialization is supported.")

        print("Start to train")
        self.total_distance = 0
        for i in tqdm(range(self.max_iters)):
            try:
                self.total_distance = self._fit_once(data, self.total_distance)
            except StopIteration:
                break
            if i % 10 == 0:
                logging.debug("Total distance: %s, iter: %s", self.total_distance, i)
        logging.info("Finish KMean training in %{}s", time.time() - start)

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

    def _fit_once(self, data: IterableDataset, last_dist=0) -> float:
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
        total_dist = 0

        new_centroids = torch.zeros_like(self.centroids, device=self.device)
        counts_per_part = torch.zeros(self.centroids.shape[0], device=self.device)
        for chunk in data:
            print("Chunk shape: {}", chunk.shape)
            chunk = chunk.to(self.device)
            ids, dists = self._transform(chunk)
            total_dist += dists.sum().item()
            ones = torch.ones(len(ids), device=self.device)

            new_centroids.index_add_(0, ids, chunk)
            counts_per_part.index_add_(0, ids, ones)
            del ids
            del dists
            del chunk

        if abs(total_dist - last_dist) / total_dist < self.tolerance:
            raise StopIteration("kmeans: converged")

        self.centroids = self._split_centroids(new_centroids, counts_per_part)
        return total_dist

    def _transform(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dist_func(data, self.centroids)

    def transform(
        self, data: Union[pa.Array, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        assert self.centroids is not None

        data = self._to_tensor(data)
        return self._transform(data)[0]
