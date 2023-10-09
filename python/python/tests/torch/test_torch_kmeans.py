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
import torch
from lance.torch.kmeans import KMeans


def test_kmeans():
    arr = np.array(range(128)).reshape(-1, 8).astype(np.float32)
    kmeans = KMeans(4, device="cpu")
    kmeans.fit(arr)

    cluster_ids = kmeans.transform(arr)
    _, cnts = torch.unique(cluster_ids, return_counts=True)
    assert (cnts >= 1).all() and (cnts <= 8).all()
    assert len(cnts) == 4  # all cluster has data
