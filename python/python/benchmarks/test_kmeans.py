# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import numpy as np
import pytest
from lance.util import KMeans

# This test is very large, but needs to be to make the CPU vs GPU comparison
# meaningful. GPU can be faster, but the overhead of copying data to and from
# the GPU can be significant. Only once the problem size is large enough does
# the benefit of GPU computation outweigh the overhead of copying data. Therefore,
# these tests are marked as slow, and are opt-in.
CLUSTERS = 1024
NUM_VECTORS = CLUSTERS * 256


@pytest.mark.benchmark(group="kmeans")
@pytest.mark.slow
def test_kmeans(benchmark):
    data = np.random.random((NUM_VECTORS, 1536)).astype("f")

    def _f():
        kmeans = KMeans(CLUSTERS, "cosine")
        kmeans.fit(data)

    benchmark(_f)


@pytest.mark.benchmark(group="kmeans")
@pytest.mark.slow
@pytest.mark.gpu
def test_kmeans_torch(benchmark):
    data = np.random.random((NUM_VECTORS, 1536)).astype("f")

    from lance.torch import preferred_device
    from lance.torch.kmeans import KMeans

    def _f():
        kmeans = KMeans(CLUSTERS, metric="cosine", device=preferred_device())
        kmeans.fit(data)

    benchmark(_f)
