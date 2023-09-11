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
from typing import Union

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
    def __init__(self, k: int, metric_type: str = "l2"):
        self.k = k
        self._metric_type = metric_type
        self._kmeans = _KMeans(k, metric_type)

    def fit(
        self, data: Union[pa.FixedSizeListArray, pa.FixedShapeTensorArray, np.ndarray]
    ):
        """Fit the model to the data.

        Parameters
        ----------
        data: pa.FixedSizeListArray, pa.FixedShapeTensorArray, np.ndarray
            The data to fit the model to. Must be a 2-D array of float32 type.
        """
        if isinstance(data, pa.FixedShapeTensorArray):
            if len(data.shape) != 1:
                raise ValueError(
                    f"Tensor array must be a 1-D array, got {len(data.shape)}-D"
                )
            data = data.storage()
        elif isinstance(data, np.ndarray):
            if len(data.shape) != 2:
                raise ValueError(
                    f"Numpy array must be a 2-D array, got {len(data.shape)}-D"
                )
            elif data.dtype != np.float32:
                raise ValueError(f"Numpy array must be float32 type, got: {data.dtype}")

            inner_arr = pa.array(data.reshape((-1,)), type=pa.float32())
            data = pa.FixedSizeListArray.from_arrays(inner_arr, data.shape[1])
        elif isinstance(data, pa.FixedSizeListArray):
            pass
        else:
            raise ValueError("Data must be a FixedSizeListArray, Tensor or numpy array")

        batch = pa.RecordBatch.from_arrays([data], ["_kmeans_data"])
        self._kmeans.fit(batch)
