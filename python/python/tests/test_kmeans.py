# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import lance
import numpy as np
import pyarrow as pa
import pytest


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
