#  Copyright (c) 2024. Lance Developers
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

import logging
import math
from typing import TYPE_CHECKING, Iterable, Literal, Optional, Union

import pyarrow as pa

from lance._dataset.cache import CachedDataset
from lance._dataset.sharded_batch_iterator import ShardedBatchIterator
from lance.dependencies import _check_for_numpy, torch
from lance.dependencies import numpy as np

from ..sampler import maybe_sample

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["LanceDataset"]


def _to_tensor(
    batch: pa.RecordBatch, *, uint64_as_int64: bool = True
) -> Union[dict[str, torch.Tensor], torch.Tensor]:
    """Convert a pyarrow RecordBatch to torch Tensor."""
    ret = {}
    for col in batch.schema.names:
        arr: pa.Array = batch[col]
        if pa.types.is_uint64(arr.type) and uint64_as_int64:
            arr = arr.cast(pa.int64())

        if (
            pa.types.is_fixed_size_list(arr.type)
            or isinstance(arr.type, pa.FixedShapeTensorType)
        ) and pa.types.is_floating(arr.type.value_type):
            np_arrs = arr.to_numpy(zero_copy_only=False)
            np_tensor = np.stack(np_arrs)
            del np_arrs
            tensor = torch.tensor(np_tensor)
            del np_tensor
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
        del arr
        ret[col] = tensor
    if len(ret) == 1:
        t = next(iter(ret.values()))
        del ret
        return t
    return ret


class TensorDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset that wraps over a tensor, returns in batches.

    Unlike `torch.utils.data.TensorDataset`, this has the same behavior as LanceDataset
    that it yields tensor in batches.
    """

    def __init__(
        self, data: Union[torch.Tensor, np.ndarray], batch_size: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if _check_for_numpy(data) and isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        self._data: torch.Tensor = data
        self._batch_size = batch_size

    def __repr__(self):
        return "LanceTensorDataset"

    def __len__(self) -> int:
        return math.ceil(self._data.shape[0] / self._batch_size)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx >= len(self):
            raise StopIteration
        start = idx * self._batch_size
        end = min((idx + 1) * self._batch_size, self._data.shape[0])
        return self._data[start:end, :]


def concat_batches(bs):
    return pa.RecordBatch.from_arrays(
        [
            pa.concat_arrays([b.columns[i] for b in bs])
            for i in range(bs[0].num_columns)
        ],
        schema=bs[0].schema,
    )


def _buffer_arrow_batches(
    it: Iterable[pa.RecordBatch],
    buffer_size: int = 10240,
) -> Iterable[pa.RecordBatch]:
    buffer = []
    cur_size = 0
    for item in it:
        if cur_size > 0 and cur_size + item.num_rows > buffer_size:
            yield concat_batches(buffer)
            buffer = []
            cur_size = 0

        buffer.append(item)
        cur_size += item.num_rows
    if buffer:
        yield concat_batches(buffer)


class LanceDataset(torch.utils.data.IterableDataset):
    """PyTorch IterableDataset over LanceDataset."""

    def __init__(
        self,
        dataset: Union[torch.utils.data.Dataset, str, Path],
        batch_size: int,
        *args,
        columns: Optional[list[str]] = None,
        filter: Optional[str] = None,
        samples: Optional[int] = 0,
        cache: Optional[Union[str, bool]] = None,
        with_row_id: bool = False,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        shard_granularity: Optional[Literal["fragment", "batch"]] = "fragment",
        to_tensor_fn: callable[
            [pa.RecordBatch], Union[dict[str, torch.Tensor], torch.Tensor]
        ] = _to_tensor,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.columns = columns
        self.batch_size = batch_size
        self.samples: Optional[int] = samples
        self.filter = filter
        self.with_row_id = with_row_id

        # As Shared Dataset
        self.rank = rank
        self.world_size = world_size
        self.shard_granularity = shard_granularity
        self._to_tensor_fn = to_tensor_fn

        if samples is not None and filter is not None:
            raise ValueError("Does not support sampling over filtered dataset")

        self.cache = cache
        self.cached_ds: Optional[CachedDataset] = None

    def __repr__(self) -> str:
        return f"LanceTorchDataset({self.dataset.uri}, size={self.samples})"

    def __iter__(self):
        stream: Iterable[pa.RecordBatch]
        if self.cached_ds:
            stream = self.cached_ds
        else:
            if self.samples:
                raw_stream = maybe_sample(
                    self.dataset,
                    n=self.samples,
                    columns=self.columns,
                    batch_size=self.batch_size,
                )
            elif self.rank is not None and self.world_size is not None:
                logging.info(
                    "Sharded Torch Dataset: rank=%s, world_size=%s, granularity=%s",
                    self.rank,
                    self.world_size,
                    self.shard_granularity,
                )
                raw_stream = ShardedBatchIterator(
                    self.dataset,
                    self.rank,
                    self.world_size,
                    columns=self.columns,
                    batch_size=self.batch_size,
                    with_row_id=self.with_row_id,
                    granularity=self.shard_granularity,
                )
            else:
                raw_stream = self.dataset.to_batches(
                    columns=self.columns,
                    batch_size=self.batch_size,
                    filter=self.filter,
                    with_row_id=self.with_row_id,
                )

            stream = _buffer_arrow_batches(raw_stream, buffer_size=self.batch_size)

            if self.cache:
                self.cached_ds = CachedDataset(stream, cache=self.cache)
                stream = self.cached_ds

        for batch in stream:
            if self._to_tensor_fn is not None:
                batch = self._to_tensor_fn(batch)
            yield batch
            del batch
