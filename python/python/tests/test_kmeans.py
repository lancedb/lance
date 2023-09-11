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

import lance
import numpy as np
import pyarrow as pa
import pytest
from numpy.linalg import norm


def test_train_cosine():
    kmeans = lance.util.KMeans(32, metric_type="cosine")
    data = np.random.randn(1000, 128).astype(np.float32)

    assert kmeans.centroids is None
    kmeans.fit(data)
    assert kmeans.centroids is not None
    centroids = np.stack(kmeans.centroids.to_numpy(zero_copy_only=False))
    assert centroids.shape == (32, 128)

    # test predict
    pred = kmeans.predict(data)

    # compute predict using numpy brute-force
    expected = []
    for row in data:
        # Cosine distance
        dist = 1 - np.dot(centroids, row) / (norm(centroids, axis=1) * norm(row))
        cluster_id = np.argmin(dist)
        expected.append(cluster_id)
    assert np.allclose(pred, expected)


def test_invalid_inputs():
    kmeans = lance.util.KMeans(32)
    data = pa.FixedShapeTensorArray.from_numpy_ndarray(
        np.random.randn(1000, 128, 8).astype(np.float32)
    )
    with pytest.raises(ValueError, match="must be a 1-D array"):
        kmeans.fit(data)

    data = pa.FixedShapeTensorArray.from_numpy_ndarray(
        np.random.randn(1000, 128).astype(np.float64)
    )
    with pytest.raises(ValueError, match="Array must be float32 type, got: double"):
        kmeans.fit(data)
