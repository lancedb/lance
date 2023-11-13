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

from typing import TYPE_CHECKING, Iterable, Optional, Union

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import IterableDataset

from ..cache import CachedDataset
from ..sampler import maybe_sample

if TYPE_CHECKING:
    from .. import LanceDataset as Dataset

__all__ = ["LanceDataset"]


def _to_tensor(batch: pa.RecordBatch) -> dict[str, torch.Tensor]:
    ret = {}
    for col in batch.column_names:
        arr: pa.Array = batch[col]
        if pa.types.is_fixed_size_list(arr.type) and pa.types.is_floating(
            arr.type.value_type
        ):
            tensor = torch.tensor(np.stack(arr.to_numpy(zero_copy_only=False)))
        elif (
            pa.types.is_integer(arr.type)
            or pa.types.is_floating(arr.type)
            or pa.types.is_boolean(arr.type)
        ):
            tensor = torch.from_numpy(arr.to_numpy(zero_copy_only=False))
        else:
            raise ValueError(
                "Only support FixedSizeList<f16/f32/f64> or "
                + f"numeric values, got: {arr.type}"
            )
        ret[col] = tensor
    return ret


class LanceDataset(IterableDataset):
    """PyTorch IterableDataset over LanceDataset."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        *args,
        columns: Optional[list[str]] = None,
        filter: Optional[str] = None,
        samples: Optional[int] = 0,
        cache: Optional[Union[str, bool]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.columns = columns
        self.batch_size = batch_size
        self.samples: Optional[int] = samples

        if samples is not None and filter is not None:
            raise ValueError("Does not support sampling over filtered dataset")

        self.cache = cache
        self.cached_ds: Optional[CachedDataset] = None

    def __iter__(self):
        stream: Iterable[pa.RecordBatch]
        if self.cached_ds:
            stream = self.cached_ds
        else:
            if self.samples:
                stream = maybe_sample(
                    self.dataset,
                    n=self.samples,
                    columns=self.columns,
                    batch_size=self.batch_size,
                )
            else:
                stream = self.dataset.to_batches(
                    columns=self.columns,
                    batch_size=self.batch_size,
                    filter=filter,
                )

            if self.cache:
                self.cached_ds = CachedDataset(stream, cache=self.cache)
                stream = self.cached_ds

        for batch in stream:
            yield _to_tensor(batch)
