# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import logging
import time
from typing import List, Literal, Optional, Tuple, Union

import pyarrow as pa
from tqdm import tqdm

from lance.dependencies import (
    _check_for_numpy,
    _check_for_torch,
    cuvs,
    pylibraft,
    torch,
)
from lance.dependencies import numpy as np

from cuvs.neighbors import cagra
from pylibraft.common import device_ndarray

from . import preferred_device
from .data import TensorDataset
from .distance import dot_distance, l2_distance

__all__ = ["KMeans"]


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
        Relative tolerance in regard to Frobenius norm of the difference in
        the cluster centers of two consecutive iterations to declare convergence.
    centroids : torch.Tensor, optional.
        Provide existing centroids.
    seed: int, optional
        Random seed
    device: str, optional
        The device to run the PyTorch algorithms. Default we will pick
        the most performant device on the host. See `lance.torch.preferred_device()`
    use_cuvs: bool
        Whether to use cuVS to accelerate computation (useful for many centroids).
    """

    def __init__(
        self,
        k: int,
        *,
        metric: Literal["l2", "euclidean", "cosine", "dot"] = "l2",
        init: Literal["random"] = "random",
        max_iters: int = 50,
        tolerance: float = 1e-4,
        centroids: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        use_cuvs: bool = False,
    ):
        self.k = k
        self.max_iters = max_iters
        self.use_cuvs = use_cuvs
        self.itopk_size = 4

        self.time_rebuild = 0.0
        self.time_search = 0.0

        metric = metric.lower()
        self.metric = metric
        if metric in ["l2", "euclidean", "cosine"]:
            # Cosine uses normalized unit vector and calculate l2 distance
            self.dist_func = l2_distance
        elif metric == "dot":
            self.dist_func = dot_distance
        else:
            raise ValueError(
                f"Only l2/cosine/dot is supported as metric type, got: {metric}"
            )

        if self.use_cuvs and self.metric == "dot":
            logging.warning(
                'Kmeans::__init__: metric == "dot" is incompatible" \
                " with use_cuvs == True, disabling cuvs...'
            )
            self.use_cuvs = False

        self.total_distance = 0
        self.centroids: Optional[torch.Tensor] = centroids
        self.init = init
        self.device = preferred_device(device)
        self.tolerance = tolerance
        self.seed = seed

        if self.device.type != "cuda" and self.use_cuvs:
            logging.warning(
                "Kmeans::__init__: cuda is not enabled/available, disabling cuvs..."
            )
            self.use_cuvs = False

    def __repr__(self):
        return f"KMeans(k={self.k}, metric={self.metric}, device={self.device})"

    def _to_tensor(
        self, data: Union[pa.FixedSizeListArray, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(data, pa.FixedSizeListArray):
            np_tensor = data.values.to_numpy(zero_copy_only=True).reshape(
                -1, data.type.list_size
            )
            data = torch.from_numpy(np_tensor)
        elif _check_for_numpy(data) and isinstance(data, np.ndarray):
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

    def _random_init(self, data: Union[torch.Tensor, np.ndarray]):
        """Random centroid initialization."""
        if self.centroids is not None:
            logging.debug("KMeans centroids already initialized")
            return

        is_numpy = _check_for_numpy(data) and isinstance(data, np.ndarray)
        if is_numpy or (_check_for_torch(data) and isinstance(data, torch.Tensor)):
            indices = np.random.choice(data.shape[0], self.k)
            if is_numpy:
                data = torch.from_numpy(data)
            self.centroids = data[indices]

    def fit(
        self,
        data: Union[
            torch.utils.data.IterableDataset,
            np.ndarray,
            torch.Tensor,
            pa.FixedSizeListArray,
        ],
    ) -> None:
        """Fit - Train the kmeans model.

        Parameters
        ----------
        data : pa.FixedSizeListArray, np.ndarray, or torch.Tensor
            2-D vectors to train kmeans.
        """
        self.time_rebuild = 0.0
        self.time_search = 0.0
        start = time.time()
        if isinstance(data, pa.FixedSizeListArray):
            data = np.stack(data.to_numpy(zero_copy_only=False))
        elif isinstance(data, pa.FixedShapeTensorArray):
            data = data.to_numpy_ndarray()
        if (_check_for_torch(data) and isinstance(data, torch.Tensor)) or (
            _check_for_numpy(data) and isinstance(data, np.ndarray)
        ):
            self._random_init(data)
            data = TensorDataset(data, batch_size=10240)

        assert self.centroids is not None
        self.centroids = self.centroids.to(self.device)

        logging.info(
            "Start kmean training, metric: %s, iters: %s", self.metric, self.max_iters
        )
        self.total_distance = 0
        for i in tqdm(range(self.max_iters)):
            try:
                self.total_distance = self._fit_once(
                    data, i, last_dist=self.total_distance
                )
            except StopIteration:
                break
            if i % 10 == 0:
                logging.debug("Total distance: %s, iter: %s", self.total_distance, i)
        logging.info("Finish KMean training in %s", time.time() - start)
        if self.use_cuvs:
            logging.info("Total search time: %s", self.time_search)
            logging.info("Total rebuild time: %s", self.time_rebuild)

    def _updated_centroids(
        self, centroids: torch.Tensor, counts: torch.Tensor
    ) -> torch.Tensor:
        centroids = centroids / counts[:, None]
        zero_counts = counts == 0
        for idx in zero_counts.nonzero(as_tuple=False):
            # split the largest cluster and remove empty cluster
            max_idx = torch.argmax(counts).item()
            # add 1% gassuian noise to the largest centroid
            # do this twice so we effectively split the largest cluster into 2
            # rand_like returns on [0, 1) so we need to shift it to [-0.5, 0.5)
            noise = (torch.rand_like(centroids[idx]) - 0.5) * 0.01 + 1
            centroids[idx] = centroids[max_idx] * noise
            noise = (torch.rand_like(centroids[idx]) - 0.5) * 0.01 + 1
            centroids[max_idx] = centroids[max_idx] * noise

        if self.metric == "cosine":
            # normalize the centroids
            centroids = torch.nn.functional.normalize(centroids)
        return centroids

    @staticmethod
    def _count_rows_in_clusters(part_ids: List[torch.Tensor], k: int) -> torch.Tensor:
        max_len = max([len(ids) for ids in part_ids])
        ones = torch.ones(max_len, device=part_ids[0].device)
        num_rows = torch.zeros(k, device=part_ids[0].device)
        for part_id in part_ids:
            num_rows.scatter_add_(0, part_id, ones)
        return num_rows

    def _fit_once(
        self, data: torch.utils.data.IterableDataset, epoch: int, last_dist: float = 0.0
    ) -> float:
        """Train KMean once and return the total distance.

        Parameters
        ----------
        data : List[torch.Tensor]
            A list of 2-D tensors, each tensor is a chunk of the input data.
        epoch : int
            The epoch of this training process
        last_dist : float
            The total distance of the last epoch.

        Returns
        -------
        float
            The total distance of the current centroids and the input data.
        """
        total_dist = 0

        # Use float32 to accumulate centroids, esp. if the vectors are
        # float16 / bfloat16 types.
        new_centroids = torch.zeros_like(
            self.centroids, device=self.device, dtype=torch.float32
        )
        counts_per_part = torch.zeros(self.centroids.shape[0], device=self.device)
        ones = torch.ones(1024 * 16, device=self.device)
        if (not self.use_cuvs) and self.metric in ["l2", "cosine"]:
            y2 = (self.centroids * self.centroids).sum(dim=1)
        else:
            y2 = None
        self.rebuild_index()
        for idx, chunk in enumerate(data):
            if idx % 50 == 0:
                logging.info("Kmeans::train: epoch %s, chunk %s", epoch, idx)
            chunk: torch.Tensor = chunk
            dtype = chunk.dtype
            chunk = chunk.to(self.device)
            ids, dists = self._transform(chunk, y2=y2)

            # valid_mask = ids >= 0
            # if torch.any(~valid_mask):
            # chunk = chunk[valid_mask]
            # ids = ids[valid_mask]

            total_dist += dists.nansum().item()
            if ones.shape[0] < ids.shape[0]:
                ones = torch.ones(len(ids), out=ones, device=self.device)

            new_centroids.index_add_(0, ids, chunk.type(torch.float32))
            counts_per_part.index_add_(0, ids, ones[: ids.shape[0]])
            del ids
            del dists
            del chunk
        if self.use_cuvs:
            self.itopk_size += 1
            self.itopk_size = min(self.itopk_size, 16)

        # this happens when there are too many NaNs or the data is just the same
        # vectors repeated over and over. Performance may be bad but we don't
        # want to crash.
        if total_dist == 0:
            logging.warning(
                "Kmeans::train: total_dist is 0, this is unusual."
                " This could result in bad performance during search."
            )
            raise StopIteration("kmeans: converged")

        if abs(total_dist - last_dist) / total_dist < self.tolerance:
            raise StopIteration("kmeans: converged")

        # cast to the type we get the data in
        self.centroids = self._updated_centroids(new_centroids, counts_per_part).type(
            dtype
        )
        return total_dist

    def rebuild_index(self):
        if self.use_cuvs:
            rebuild_time_start = time.time()
            cagra_metric = "sqeuclidean"
            dim = self.centroids.shape[1]
            graph_degree = max(dim // 4, 32)
            nn_descent_degree = graph_degree * 2
            index_params = cagra.IndexParams(
                metric=cagra_metric,
                intermediate_graph_degree=nn_descent_degree,
                graph_degree=graph_degree,
                build_algo="nn_descent",
                compression=None,
            )
            self.index = cagra.build(index_params, self.centroids)
            rebuild_time_end = time.time()
            self.time_rebuild += rebuild_time_end - rebuild_time_start

    def _transform(
        self,
        data: torch.Tensor,
        y2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.metric == "cosine":
            data = torch.nn.functional.normalize(data)

        if self.use_cuvs:
            search_time_start = time.time()
            device = torch.device("cuda")
            out_idx = device_ndarray.empty((data.shape[0], 1), dtype="uint32")
            out_dist = device_ndarray.empty((data.shape[0], 1), dtype="float32")
            search_params = cagra.SearchParams(itopk_size=self.itopk_size)
            cagra.search(
                search_params,
                self.index,
                data,
                1,
                neighbors=out_idx,
                distances=out_dist,
            )
            ret = (
                torch.as_tensor(out_idx, device=device)
                .squeeze(dim=1)
                .view(torch.int32),
                torch.as_tensor(out_dist, device=device),
            )
            search_time_end = time.time()
            self.time_search += search_time_end - search_time_start
            return ret

        if self.metric in ["l2", "cosine"]:
            return self.dist_func(data, self.centroids, y2=y2)
        else:
            return self.dist_func(data, self.centroids)

    def transform(
        self, data: Union[pa.Array, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """Transform the input data to cluster ids for each row."""
        assert self.centroids is not None

        data = self._to_tensor(data)
        return self._transform(data)[0]
