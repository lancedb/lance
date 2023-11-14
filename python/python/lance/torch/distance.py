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


from typing import Tuple

import torch


@torch.jit.script
def _cosine_distance(
    vectors: torch.Tensor, centroids: torch.Tensor, split_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(vectors.shape) != 2 or len(centroids.shape) != 2:
        raise ValueError(
            f"x and y must be 2-D matrix, got: vectors.shape={vectors.shape}, centroids.shape={centroids.shape}"
        )

    y2 = torch.linalg.norm(centroids.T, dim=0, keepdim=True)

    partitions = []
    distances = []

    for sub_vectors in torch.split(vectors, split_size):
        x2 = torch.linalg.norm(sub_vectors, dim=1, keepdim=True)
        dists = 1 - sub_vectors @ centroids.T / (x2 * y2)
        part_ids = torch.argmin(dists, dim=1, keepdim=True)
        partitions.append(part_ids)
        distances.append(dists.take_along_dim(part_ids, dim=1))

    return torch.cat(partitions), torch.cat(distances)


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
    split = 1024 * 80
    while split > 256:
        try:
            return _cosine_distance(vectors, centroids, split_size=split)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                split //= 2
                continue
            raise

    raise RuntimeError("Cosine distance out of memory")


@torch.jit.script
def l2_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pair-wise L2 / Euclidean distance between two 2-D Tensors.

    Parameters
    ----------
    x : torch.Tensor
        A 2-D [N, D] tensor
    y : torch.Tensor
        A 2-D [M, D] tensor

    Returns
    -------
    A 2-D [N, M] tensor of L2 distances between x and y.
    """
    if len(x.shape) != 2 or len(y.shape) != 2:
        raise ValueError(
            f"x and y must be 2-D matrix, got: x.shape={x.shape}, y.shape={y.shape}"
        )
    # (x - y)^2 = x^2 + y^2 - 2xy
    x2 = (x * x).sum(dim=1)
    y2 = (y * y).sum(dim=1)
    xy = x @ y.T
    return (
        x2.broadcast_to(y2.shape[0], x2.shape[0]).T
        + y2.broadcast_to(x2.shape[0], y2.shape[0])
        - 2 * xy
    )


@torch.jit.script
def dot_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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

    return 1 - x @ y.T
