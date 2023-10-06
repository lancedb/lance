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


@pytest.mark.benchmark(group="kmeans")
def test_kmeans(benchmark):
    data = np.random.random((65535, 1536)).astype("f")

    def _f():
        kmeans = KMeans(256, "cosine")
        kmeans.fit(data)

    benchmark(_f)


@pytest.mark.benchmark(group="kmeans")
@pytest.mark.gpu
def test_kmeans_torch(benchmark):
    clusters = 8192
    data = np.random.random((clusters * 256, 1536)).astype("f")

    from lance.torch import preferred_device
    from lance.torch.kmeans import KMeans
    from torch.profiler import profile

    def _f():
        kmeans = KMeans(clusters, metric="cosine", device=preferred_device())
        # with profile(
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True,
        # ) as prof:
        #     kmeans.fit(data)
        # print(prof.key_averages().table(sort_by="self_cuda_time_total"))
        kmeans.fit(data)

    benchmark(_f)
