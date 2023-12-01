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

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_cosine_distance():
    from lance.torch.distance import cosine_distance

    x = np.random.randn(20, 256).astype(np.float32)
    y = np.random.rand(100, 256).astype(np.float32)

    part_ids, dist = cosine_distance(torch.from_numpy(x), torch.from_numpy(y))
    assert dist.shape == (20,)
    assert part_ids.shape == (20,)

    # Brute-force / the simplest proof.
    expect = np.array(
        [
            1 - np.dot(x_row, y_row) / np.linalg.norm(x_row) / np.linalg.norm(y_row)
            for x_row in x
            for y_row in y
        ],
        dtype=np.float32,
    ).reshape(20, 100)

    expect_part_ids = np.argmin(expect, axis=1)
    expect_dists = np.take_along_axis(
        expect, expect_part_ids.reshape((-1, 1)), axis=1
    ).reshape(-1)
    assert np.allclose(part_ids.cpu(), expect_part_ids)
    assert np.allclose(dist.cpu(), expect_dists)


def test_l2_distance():
    from lance.torch.distance import l2_distance

    x = np.random.randn(20, 256).astype(np.float32)
    y = np.random.rand(100, 256).astype(np.float32)

    part_ids, dist = l2_distance(torch.from_numpy(x), torch.from_numpy(y))
    assert dist.shape == (20,)

    expect_arr = (
        np.array([[np.linalg.norm(x_row - y_row) ** 2 for y_row in y] for x_row in x])
        .astype(np.float32)
        .reshape(20, 100)
    )
    expect_part_ids = np.argmin(expect_arr, axis=1)
    expect_dists = np.take_along_axis(
        expect_arr, expect_part_ids.reshape((-1, 1)), axis=1
    ).reshape(-1)
    assert np.allclose(dist.cpu(), expect_dists)
    assert np.allclose(part_ids.cpu(), expect_part_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.cuda is not available")
def test_large_cosine_distance_cuda():
    """Test CUDA Out of memory.

    Cosine will generate a X*Y matrix as intermediate result. \
    For x=[M, D], y=[N, D], x*y^T = [M, N] float32 matrix.
    If M = 65K, N = 100K, X*Y^T of float32 = 25 GB.
    """
    from lance.torch.distance import cosine_distance

    rng = np.random.default_rng()

    x = rng.random((1024 * 100, 1536), dtype="float32")
    y = rng.random((1024 * 64, 1536), dtype="float32")

    (part_ids, dist) = cosine_distance(
        torch.from_numpy(x).to("cuda"), torch.from_numpy(y).to("cuda")
    )
    assert part_ids.shape == (1024 * 100,)
    assert dist.shape == (1024 * 100,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.cuda is not available")
def test_large_l2_distance_cuda():
    """Test CUDA Out of memory"""
    from lance.torch.distance import l2_distance

    rng = np.random.default_rng()

    x = rng.random((1024 * 100, 1536), dtype="float32")
    y = rng.random((1024 * 64, 1536), dtype="float32")

    (part_ids, dist) = l2_distance(
        torch.from_numpy(x).to("cuda"), torch.from_numpy(y).to("cuda")
    )
    assert part_ids.shape == (1024 * 100,)
    assert dist.shape == (1024 * 100,)
