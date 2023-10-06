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

CLUSTERS = 1024
NUM_VECTORS = CLUSTERS * 256


@pytest.mark.benchmark(group="kmeans")
def test_kmeans(benchmark):
    data = np.random.random((NUM_VECTORS, 1536)).astype("f")

    def _f():
        kmeans = KMeans(CLUSTERS, "cosine")
        kmeans.fit(data)

    benchmark(_f)


@pytest.mark.benchmark(group="kmeans")
@pytest.mark.gpu
def test_kmeans_torch(benchmark):
    data = np.random.random((NUM_VECTORS, 1536)).astype("f")

    from lance.torch import preferred_device
    from lance.torch.kmeans import KMeans

    def _f():
        kmeans = KMeans(CLUSTERS, metric="cosine", device=preferred_device())
        kmeans.fit(data)

    benchmark(_f)
