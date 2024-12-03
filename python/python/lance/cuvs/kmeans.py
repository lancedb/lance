# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors


import logging
import time
from typing import Literal, Optional, Tuple, Union

import pyarrow as pa

from lance.dependencies import cagra, raft_common, torch
from lance.dependencies import numpy as np
from lance.torch.kmeans import KMeans as KMeansTorch

__all__ = ["KMeans"]


class KMeans(KMeansTorch):
    """K-Means trains over vectors and divide into K clusters,
    using cuVS as accelerator.

    This implement is built on PyTorch+cuVS, supporting Nvidia GPU only.

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
        For the cuVS implementation, it will be verified this is a cuda device.
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
        itopk_size: int = 10,
    ):
        if metric == "dot":
            raise ValueError(
                'Kmeans::__init__: metric == "dot" is incompatible' " with cuVS"
            )
        super().__init__(
            k,
            metric=metric,
            init=init,
            max_iters=max_iters,
            tolerance=tolerance,
            centroids=centroids,
            seed=seed,
            device=device,
        )

        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("KMeans::__init__: cuda is not enabled/available")

        self.itopk_size = itopk_size
        self.time_rebuild = 0.0
        self.time_search = 0.0

    def fit(
        self,
        data: Union[
            torch.utils.data.IterableDataset,
            np.ndarray,
            torch.Tensor,
            pa.FixedSizeListArray,
        ],
    ) -> None:
        self.time_rebuild = 0.0
        self.time_search = 0.0
        super().fit(data)
        logging.info("Total search time: %s", self.time_search)
        logging.info("Total rebuild time: %s", self.time_rebuild)

    def rebuild_index(self):
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

        self.y2 = None

    def _transform(
        self,
        data: torch.Tensor,
        y2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.metric == "cosine":
            data = torch.nn.functional.normalize(data)

        search_time_start = time.time()
        device = torch.device("cuda")
        out_idx = raft_common.device_ndarray.empty((data.shape[0], 1), dtype="uint32")
        out_dist = raft_common.device_ndarray.empty((data.shape[0], 1), dtype="float32")
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
            torch.as_tensor(out_idx, device=device).squeeze(dim=1).view(torch.int32),
            torch.as_tensor(out_dist, device=device),
        )
        search_time_end = time.time()
        self.time_search += search_time_end - search_time_start
        return ret
