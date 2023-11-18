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

from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

torch = pytest.importorskip("torch")

from lance.vector import train_ivf_centroids_on_accelerator  # noqa: E402


def test_kmeans():
    from lance.torch.kmeans import KMeans

    arr = np.array(range(128)).reshape(-1, 8).astype(np.float32)
    kmeans = KMeans(4, device="cpu")
    kmeans.fit(arr)

    cluster_ids = kmeans.transform(arr)
    _, cnts = torch.unique(cluster_ids, return_counts=True)
    assert (cnts >= 1).all() and (cnts <= 8).all()
    assert len(cnts) == 4  # all cluster has data


@pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
def test_torch_kmean_accept_torch_device(tmp_path: Path, dt):
    from lance.torch import preferred_device

    device = preferred_device()
    if device == torch.device("cpu") and dt == np.float16:
        # Torch does not support float16 on CPU
        #
        # raises RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'
        pytest.skip("Skip f16 test for CPU")

    arr = np.array(range(256)).astype(dt)
    fsl = pa.FixedSizeListArray.from_arrays(arr.ravel(), list_size=8)
    tbl = pa.Table.from_arrays([fsl], ["vector"])
    ds = lance.write_dataset(tbl, tmp_path)
    # Not raising exception if pass a `torch.device()` directly
    train_ivf_centroids_on_accelerator(
        ds, "vector", 4, "L2", accelerator=preferred_device()
    )
