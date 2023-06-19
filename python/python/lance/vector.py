#  Copyright 2023 Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Embedding vector utilities"""

from typing import Optional, Union

import numpy as np
import pyarrow as pa


def _normalize_vectors(vectors, ndim):
    if ndim is None:
        ndim = len(next(iter(vectors)))
    values = np.array(vectors, dtype="float32").ravel()
    return pa.FixedSizeListArray.from_arrays(values, list_size=ndim)


def _validate_ndim(values, ndim):
    for v in values:
        if ndim is None:
            ndim = len(v)
        else:
            if ndim != len(v):
                raise ValueError(f"Expected {ndim} dimensions but got {len(v)} for {v}")
    return ndim


def vec_to_table(
    data: Union[dict, list, np.ndarray],
    names: Optional[Union[str, list]] = None,
    ndim: Optional[int] = None,
    check_ndim: bool = True,
) -> pa.Table:
    """
    Create a pyarrow Table containing vectors.
    Vectors are created as FixedSizeListArray's in pyarrow with Float32 values.

    Examples
    --------
    >>> import numpy as np
    >>> from lance.vector import vec_to_table
    >>> dd = {"vector0": np.random.randn(10), "vector1": np.random.randn(10)}
    >>> vec_to_table(dd)
    pyarrow.Table
    id: string
    vector: fixed_size_list<item: float>[10]
      child 0, item: float
    ----
    id: [["vector0","vector1"]]
    vector: [[[0.49671414,-0.1382643,0.64768857,1.5230298,-0.23415338,-0.23413695,\
1.5792128,0.7674347,-0.46947438,0.54256004],[-0.46341768,-0.46572974,0.24196227,\
-1.9132802,-1.7249179,-0.5622875,-1.0128311,0.31424734,-0.9080241,-1.4123037]]]
    >>> vec_to_table(dd).to_pandas()
            id                                             vector
    0  vector0  [0.49671414, -0.1382643, 0.64768857, 1.5230298...
    1  vector1  [-0.46341768, -0.46572974, 0.24196227, -1.9132...

    Parameters
    ----------
    data: dict, list, or np.ndarray
        If dict, the keys are added as "id" column
        If list, then each element is assumed to be a vector
        If ndarray, then each row is assumed to be a vector
    names: str or list, optional
        If data is dict, then names should be a list of 2 str; default ["id", "vector"]
        If data is list or ndarray, then names should be str; default "vector"
    ndim: int, optional
        Number of dimensions of the vectors. Inferred if omitted.
    check_ndim: bool, default True
        Whether to verify that all vectors have the same length

    Returns
    -------
    tbl: pa.Table
        A pyarrow Table with vectors converted to appropriate types
    """
    if isinstance(data, dict):
        if names is None:
            names = ["id", "vector"]
        elif not isinstance(names, (list, tuple)) and len(names) == 2:
            raise ValueError(
                "If data is a dict, names must be a list or tuple of 2 strings"
            )
        values = list(data.values())
        if check_ndim:
            ndim = _validate_ndim(values, ndim)
        vectors = _normalize_vectors(values, ndim)
        ids = pa.array(data.keys())
        arrays = [ids, vectors]
    elif isinstance(data, (list, np.ndarray)):
        if names is None:
            names = ["vector"]
        elif isinstance(names, str):
            names = [names]
        elif not isinstance(names, (list, tuple)) and len(names) == 1:
            raise ValueError(f"names cannot be more than 1 got {len(names)}")
        if check_ndim:
            ndim = _validate_ndim(data, ndim)
        vectors = _normalize_vectors(data, ndim)
        arrays = [vectors]
    else:
        raise NotImplementedError(
            f"data must be dict, list, or ndarray, got {type(data)} instead"
        )
    return pa.Table.from_arrays(arrays, names=names)
