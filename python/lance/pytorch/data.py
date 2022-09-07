# Copyright 2022 Lance Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import List, Optional, Union

import pyarrow as pa
import pyarrow.compute as pc
import numpy as np

try:
    import torch
    from torch.utils.data import IterableDataset
except ImportError as e:
    raise ImportError("Please install pytorch", e)

from lance import dataset, scanner

__all__ = ["LanceDataset"]


def to_numpy(arr: pa.Array):
    """Convert pyarrow array to numpy array"""
    # TODO: arrow.to_numpy(writable=True) makes a new copy of data.
    # Investigate how to directly perform zero-copy into Torch Tensor.
    np_arr = arr.to_numpy(zero_copy_only=False, writable=True)
    if pa.types.is_binary(arr.type) or pa.types.is_large_binary(arr.type):
        return np_arr.astype(np.bytes_)
    elif pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
        return np_arr.astype(np.str_)
    else:
        return np_arr


class LanceDataset(IterableDataset):
    """An PyTorch IterableDataset.

    See:
    https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    """

    def __init__(
        self,
        uri: Union[str, Path],
        columns: Optional[List[str]] = None,
        filter: Optional[pc.Expression] = None,
        batch_size: Optional[int] = None,
    ):
        self.uri = uri
        self.columns = columns if columns else []
        self.filter = filter
        self.batch_size = batch_size

    def __repr__(self):
        return f"LanceDataset(uri={self.uri})"

    def __iter__(self):
        """Yield dataset"""
        scan: pa.dataset.Scanner = scanner(
            dataset(self.uri),
            columns=self.columns,
            batch_size=self.batch_size,
            filter=self.filter,
        )
        for batch in scan.to_reader():
            yield [to_numpy(arr) for arr in batch.columns]
