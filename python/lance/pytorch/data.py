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
from typing import Union, Optional

import pyarrow as pa
import pyarrow.dataset

try:
    from torch.utils.data import IterableDataset
except ImportError:
    raise ImportError("Please install pytorch via pip install lance[pytorch]")

from lance import dataset, scanner

__all__ = ["LanceDataset"]


class LanceDataset(IterableDataset):
    """An PyTorch IterableDataset.

    See:
    https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    """

    def __init__(
        self,
        uri: Union[str, Path],
        columns: Optional[list[str]] = None,
        batch_size: Optional[int] = None,
    ):
        self.uri = uri
        self.columns = columns if columns else []
        self.batch_size = batch_size
        print(f"Init lance dataset: batch size={batch_size}")
        self.scanner: pa.dataset.Scanner = scanner(
            dataset(self.uri), columns=columns, batch_size=batch_size
        )

    def __repr__(self):
        return f"LanceDataset(uri={self.uri})"

    def __iter__(self):
        """Yield dataset"""
        import torch

        for batch in self.scanner.scan_batches():
            # TODO: arrow.to_numpy(writable=True) makes a new copy of data.
            yield [
                torch.from_numpy(arr.to_numpy(zero_copy_only=False, writable=True))
                for arr in batch.record_batch.columns
            ]
