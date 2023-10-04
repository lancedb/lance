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


@torch.jit.script
def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cosine pair-wise distances between two 2-D Tensors.

    Cosine distance = 1 - |xy| / ||x|| * ||y||

    Parameters
    ----------
    x : torch.Tensor
        A 2-D [N, D] tensor
    y : torch.Tensor
        A 2-D [M, D] tensor

    Returns
    -------
    A 2-D [N, M] tensor of cosine distances between x and y
    """
    if len(x.shape) != 2 or len(y.shape) != 2:
        raise ValueError(
            f"x and y must be 2-D matrix, got: x.shape={x.shape}, y.shape={y.shape}"
        )

    x2 = torch.linalg.norm(x, dim=1, keepdim=True)
    y2 = torch.linalg.norm(y.T, dim=0, keepdim=True)
    return 1 - x @ y.T / (x2 * y2)


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
    # TODO: can we do pair-wise subtract directly?
    result = []
    for x_row in x:
        sub = x_row - y
        norms = torch.linalg.norm(sub, dim=1)
        result.append(norms)
    return torch.stack(result)


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
