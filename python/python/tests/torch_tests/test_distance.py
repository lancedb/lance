# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lance.torch.distance import pairwise_l2  # noqa: E402


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


def test_pairwise_cosine():
    from lance.torch.distance import pairwise_cosine

    x = np.random.randn(20, 256).astype(np.float32)
    y = np.random.rand(100, 256).astype(np.float32)

    dist = pairwise_cosine(torch.from_numpy(x), torch.from_numpy(y))
    assert dist.shape == (20, 100)

    expect = np.array(
        [
            1 - np.dot(x_row, y_row) / np.linalg.norm(x_row) / np.linalg.norm(y_row)
            for x_row in x
            for y_row in y
        ],
        dtype=np.float32,
    ).reshape(20, 100)
    assert np.allclose(dist.cpu(), expect)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.cuda is not available")
def test_l2_distance_f16_bf16_cuda():
    DIM = 32
    x = torch.rand(128, DIM, dtype=torch.float16).to("cuda")
    y = torch.rand(512, DIM, dtype=torch.float16).to("cuda")

    dists = pairwise_l2(x, y)
    assert dists.shape == (128, 512)

    x = x.type(torch.bfloat16)
    y = y.type(torch.bfloat16)
    dists = pairwise_l2(x, y)
    assert dists.shape == (128, 512)


def test_l2_distance_f16_bf16_cpu():
    DIM = 32
    # Make sure it happens on CPU
    x = torch.rand(128, DIM, dtype=torch.float16).to("cpu")
    y = torch.rand(512, DIM, dtype=torch.float16).to("cpu")

    dists = pairwise_l2(x, y)
    assert dists.shape == (128, 512)

    x = x.type(torch.bfloat16)
    y = y.type(torch.bfloat16)
    dists = pairwise_l2(x, y)
    assert dists.shape == (128, 512)
