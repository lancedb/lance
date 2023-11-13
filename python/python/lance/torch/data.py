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

"""Read Lance dataset as torch DataPipe.
"""

# PEP-585. Can be removed after deprecating python 3.8 support.
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from torchdata.datapipes.iter import IterDataPipe
import pyarrow as pa

from ..sampler import maybe_sample

if TYPE_CHECKING:
    from .. import LanceDataset

__all__ = ["LanceDataLoader"]


def _to_tensor(batch: pa.RecordBatch) -> dict[str, torch.Tensor]:
    ret = {}
    for col in batch.column_names:
        arr: pa.Array = batch[col]
        if pa.types.is_fixed_size_list(arr.type):
            tensor = torch.tensor(np.stack(arr.to_numpy(zero_copy_only=False)))
        else:
            tensor = torch.from_numpy(arr.to_numpy(zero_copy_only=False))
        ret[col] = tensor
    return ret


class LanceDataLoader(IterDataPipe):
    def __init__(self, dataset: LanceDataset, batch_size: int, *args, columns: list[str] = None,
                 samples: Optional[int] = 0, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.columns = columns
        self.batch_size = batch_size
        self.samples: Optional[int] = samples

    def __iter__(self):
        if self.samples:
            for batch in maybe_sample(self.dataset, n=self.samples, columns=self.columns):
                yield _to_tensor(batch)
        else:
            for batch in self.dataset.to_batches(columns=self.columns, batch_size=self.batch_size):
                yield _to_tensor(batch)
