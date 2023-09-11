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

from datetime import datetime
from typing import Optional, Union

import numpy as np
import pyarrow as pa

from .lance import _KMeans

try:
    import pandas as pd

    ts_types = Union[datetime, pd.Timestamp, str]
except ImportError:
    pd = None
    ts_types = Union[datetime, str]


def sanitize_ts(ts: ts_types) -> datetime:
    """Returns a python datetime object from various timestamp input types."""
    if pd and isinstance(ts, str):
        ts = pd.to_datetime(ts).to_pydatetime()
    elif isinstance(ts, str):
        try:
            ts = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise ValueError(
                f"Failed to parse timestamp string {ts}. Try installing Pandas."
            )
    elif pd and isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    elif not isinstance(ts, datetime):
        raise TypeError(f"Unrecognized version timestamp {ts} of type {type(ts)}")
    return ts


class KMeans:
    """KMean model for clustering.

    It works with 2-D arrays of float32 type,
    and support distance metrics: "l2", "cosine", "dot".

    Note `fit()` must be called before `predict()`.

    Examples
    --------
    >>> import numpy as np
    >>> import lance
    >>> data = np.random.randn(1000, 128).astype(np.float32)
    >>> kmeans = lance.util.KMeans(8, metric_type="cosine")
    >>> kmeans.fit(data)
    >>> centroids = np.stack(kmeans.centroids.to_numpy(zero_copy_only=False))
    >>> clusters = kmeans.predict(data)
    """
    def __init__(self, k: int, metric_type: str = "l2", max_iters: int = 50):
        """Create a KMeans model.

        Parameters
        ----------
        k: int
            The number of clusters to create.
        metric_type: str, default="l2"
            The metric to use for calculating distances between vectors.
            Supported distance metrics: "l2", "cosine", "dot"
        max_iters: int
            The maximum number of iterations to run the KMeans algorithm. Default: 50.
        """
        self.k = k
        self._metric_type = metric_type
        self._kmeans = _KMeans(k, metric_type, max_iters=max_iters)

    def __repr__(self) -> str:
        return f"lance.KMeans(k={self.k}, metric_type={self._metric_type})"

    @property
    def centroids(self) -> Optional[pa.FixedSizeListArray]:
        """Returns the centroids of the model."""
        ret = self._kmeans.centroids()
        return ret

    def _to_fixed_size_list(self, data: pa.Array) -> pa.FixedSizeListArray:
        if isinstance(data, pa.FixedSizeListArray):
            if data.value_type != pa.float32():
                raise ValueError(f"Array must be float32 type, got: {data.value_type}")
            return data
        elif isinstance(data, pa.FixedShapeTensorArray):
            return pa.FixedSizeListArray.from_arrays(data.storage(), data.shape[1])
        elif isinstance(data, np.ndarray):
            if len(data.shape) != 2:
                raise ValueError(
                    f"Numpy array must be a 2-D array, got {len(data.shape)}-D"
                )
            elif data.dtype != np.float32:
                raise ValueError(f"Numpy array must be float32 type, got: {data.dtype}")

            inner_arr = pa.array(data.reshape((-1,)), type=pa.float32())
            return pa.FixedSizeListArray.from_arrays(inner_arr, data.shape[1])
        else:
            raise ValueError("Data must be a FixedSizeListArray, Tensor or numpy array")

    def fit(
        self, data: Union[pa.FixedSizeListArray, pa.FixedShapeTensorArray, np.ndarray]
    ):
        """Fit the model to the data.

        Parameters
        ----------
        data: pa.FixedSizeListArray, pa.FixedShapeTensorArray, np.ndarray
            The data to fit the model to. Must be a 2-D array of float32 type.
        """
        arr = self._to_fixed_size_list(data)
        self._kmeans.fit(arr)

    def predict(
        self, data: Union[pa.FixedSizeListArray, pa.FixedShapeTensorArray, np.ndarray]
    ) -> pa.UInt32Array:
        """Predict the cluster for each vector in the data."""
        arr = self._to_fixed_size_list(data)
        return self._kmeans.predict(arr)
