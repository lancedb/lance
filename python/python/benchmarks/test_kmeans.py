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
from lance.util import KMeans

# This test is very large, but needs to be to make the CPU vs GPU comparison
# meaningful. GPU can be faster, but the overhead of copying data to and from
# the GPU can be significant. Only once the problem size is large enough does
# the benefit of GPU computation outweigh the overhead of copying data. Therefore,
# these tests are marked as slow, and are opt-in.
CLUSTERS = 1024 * 64
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
# @pytest.mark.slow
# @pytest.mark.gpu
def test_kmeans_torch(benchmark):
    import lance
    from lance.torch import preferred_device
    from lance.torch.data import LanceDataset
    from lance.torch.kmeans import KMeans

    ds = lance.dataset("/tmp/testdata")

    loader = LanceDataset(
        ds, columns=["vec"], batch_size=40960, cache=True, samples=NUM_VECTORS
    )

    def _f():
        import tracemalloc

        tracemalloc.start()
        try:
            kmeans = KMeans(CLUSTERS, metric="cosine", device=preferred_device())
            kmeans.fit(loader)
        except:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")
            for stat in top_stats[:10]:
                print(stat)
            raise

    benchmark(_f)
