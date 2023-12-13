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


from typing import Optional, Tuple

import torch

__all__ = [
    "pairwise_cosine",
    "cosine_distance",
    "pairwise_l2",
    "l2_distance",
    "dot_distance",
]


@torch.jit.script
def _pairwise_cosine(
    x: torch.Tensor, y: torch.Tensor, y2: torch.Tensor
) -> torch.Tensor:
    x2 = torch.linalg.norm(x, dim=1).reshape((-1, 1))
    return 1 - (x @ y.T).div_(x2).div_(y2)


def pairwise_cosine(
    x: torch.Tensor, y: torch.Tensor, *, y2: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute pair-wise cosine distance between x and y.

    Parameters
    ----------
    x : torch.Tensor
        A 2-D ``[M, D]`` tensor, containing `M` vectors.
    y : torch.Tensor
        A 2-D ``[N, D]`` tensor, containing `N` vectors.

    Returns
    -------
    A ``[M, N]`` tensor with pair-wise cosine distances between x and y.
    """
    if len(x.shape) != 2 or len(y.shape) != 2:
        raise ValueError(
            f"x and y must be 2-D matrix, got: x.shape={x.shape}, y.shape={y.shape}"
        )
    if y2 is None:
        y2 = torch.linalg.norm(y, dim=1)

    return _pairwise_cosine(x, y, y2)


@torch.jit.script
def _cosine_distance(
    vectors: torch.Tensor, centroids: torch.Tensor, split_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(vectors.shape) != 2 or len(centroids.shape) != 2:
        raise ValueError(
            f"x and y must be 2-D matrix, got: vectors.shape={vectors.shape}"
            f", centroids.shape={centroids.shape}"
        )

    y2 = torch.linalg.norm(centroids.T, dim=0, keepdim=True)

    partitions = []
    distances = []

    for sub_vectors in torch.split(vectors, split_size):
        dists = _pairwise_cosine(sub_vectors, centroids, y2)
        part_ids = torch.argmin(dists, dim=1, keepdim=True)
        partitions.append(part_ids)
        distances.append(dists.take_along_dim(part_ids, dim=1))

    return torch.cat(partitions).reshape(-1), torch.cat(distances).reshape(-1)


def _suggest_batch_size(tensor: torch.Tensor) -> int:
    if torch.cuda.is_available():
        (free_mem, total_mem) = torch.cuda.mem_get_info()
        return free_mem // tensor.shape[0] // 4  # TODO: support bf16/f16
    else:
        return 1024 * 128


def cosine_distance(
    vectors: torch.Tensor, centroids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cosine pair-wise distances between two 2-D Tensors.

    Cosine distance = 1 - |xy| / ||x|| * ||y||

    Parameters
    ----------
    vectors : torch.Tensor
        A 2-D [N, D] tensor
    centroids : torch.Tensor
        A 2-D [M, D] tensor

    Returns
    -------
    A tuple of Tensors, for centroids id, and distance to the centroid.

    A 2-D [N, M] tensor of cosine distances between x and y
    """
    split = _suggest_batch_size(centroids)
    while split >= 256:
        try:
            return _cosine_distance(vectors, centroids, split_size=split)
        except RuntimeError as e:  # noqa: PERF203
            if "CUDA out of memory" in str(e):
                split //= 2
                continue
            raise

    raise RuntimeError("Cosine distance out of memory")


def pairwise_l2(
    x: torch.Tensor, y: torch.Tensor, y2: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute pair-wise L2 distance between x and y.

    Parameters
    ----------
    x : torch.Tensor
        A 2-D ``[M, D]`` tensor, containing `M` vectors.
    y : torch.Tensor
        A 2-D ``[N, D]`` tensor, containing `N` vectors.
    y2: 1-D tensor.Tensor, optional
        Optionally, the pre-computed `y^2`.

    Returns
    -------
    A ``[M, N]`` tensor with pair-wise L2 distance between x and y.
    """
    if len(x.shape) != 2 or len(y.shape) != 2:
        raise ValueError(
            f"x and y must be 2-D matrix, got: x.shape={x.shape}, y.shape={y.shape}"
        )
    if y2 is None:
        y2 = (y * y).sum(dim=1)
    x2 = (x * x).sum(dim=1)
    xy = x @ y.T
    dists = (
        x2.broadcast_to(y2.shape[0], x2.shape[0]).T
        + y2.broadcast_to(x2.shape[0], y2.shape[0])
        - 2 * xy
    )
    return dists


@torch.jit.script
def _l2_distance(
    x: torch.Tensor, y: torch.Tensor, split_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(x.shape) != 2 or len(y.shape) != 2:
        raise ValueError(
            f"x and y must be 2-D matrix, got: x.shape={x.shape}, y.shape={y.shape}"
        )

    part_ids = []
    distances = []

    y2 = (y * y).sum(dim=1)
    for sub_vectors in x.split(split_size):
        dists = pairwise_l2(sub_vectors, y, y2)
        idx = torch.argmin(dists, dim=1, keepdim=True)
        part_ids.append(idx)
        distances.append(dists.take_along_dim(idx, dim=1))

    return torch.cat(part_ids).reshape(-1), torch.cat(distances).reshape(-1)


def l2_distance(
    vectors: torch.Tensor, centroids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pair-wise L2 / Euclidean distance between two 2-D Tensors.

    Parameters
    ----------
    vectors : torch.Tensor
       A 2-D [N, D] tensor
    centroids : torch.Tensor
       A 2-D [M, D] tensor

    Returns
    -------
    A tuple of Tensors, for centroids id, and distance to the centroids.
    """
    split = _suggest_batch_size(centroids)
    while split >= 256:
        try:
            return _l2_distance(vectors, centroids, split_size=split)
        except RuntimeError as e:  # noqa: PERF203
            if "CUDA out of memory" in str(e):
                split //= 2
                continue
            raise

    raise RuntimeError("L2 distance out of memory")


@torch.jit.script
def dot_distance(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pair-wise dot distance between two 2-D Tensors.

    Parameters
    ----------
    x : torch.Tensor
        A 2-D [N, D] tensor
    y : torch.Tensor
        A 2-D [M, D] tensor

    Returns
    -------
    A 2-D [N, M] tensor of cosine distances between x and y.
    """
    if len(x.shape) != 2 or len(y.shape) != 2:
        raise ValueError(
            f"x and y must be 2-D matrix, got: x.shape={x.shape}, y.shape={y.shape}"
        )

    dists = 1 - x @ y.T
    idx = torch.argmin(dists, dim=1, keepdim=True)
    dists = dists.take_along_dim(idx, dim=1).reshape(-1)
    return idx.reshape(-1), dists.reshape(-1)
