# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Read Lance dataset as torch DataPipe."""

# PEP-585. Can be removed after deprecating python 3.8 support.
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Union

import pyarrow as pa

import lance
from lance._dataset.cache import CachedDataset
from lance.dependencies import _check_for_numpy, torch
from lance.dependencies import numpy as np

from ..sampler import (
    FullScanSampler,
    Sampler,
    ShardedBatchSampler,
    ShardedFragmentSampler,
    maybe_sample,
)

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
        ) and (
            pa.types.is_floating(arr.type.value_type)
            or pa.types.is_integer(arr.type.value_type)
        ):
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
    """PyTorch :class:`torch.utils.data.IterableDataset` over lance dataset."""

    def __init__(
        self,
        dataset: Union[torch.utils.data.Dataset, str, Path],
        batch_size: int,
        *args,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        filter: Optional[str] = None,
        samples: Optional[int] = 0,
        cache: Optional[Union[str, bool]] = None,
        with_row_id: bool = False,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        shard_granularity: Optional[Literal["fragment", "batch"]] = None,
        batch_readahead: int = 16,
        to_tensor_fn: Optional[
            callable[[pa.RecordBatch], Union[dict[str, torch.Tensor], torch.Tensor]]
        ] = None,
        sampler: Optional[Sampler] = None,
        **kwargs,
    ):
        """Use PyTorch Dataset API to read Lance dataset.

        Parameters
        ----------
        dataset : Union[torch.utils.data.Dataset, str, Path]
            Lance dataset to read. Can be URI, path, or an initialized Lance Dataset.
        batch_size : int
            Batch size to yield for each iteration.
        columns : list of str, optional
            The names of the column to read, by default None, which means reading all
            columns.
        filter : str, optional
            If set, only rows that match the filter will be read.  Currently, this
            can only be used when doing a full scan (`sampler` is None and
            shard_granularity is None or "fragment" and `samples` is None)
        cache : str or bool, optional
            If set true, the dataset will be cached on disk from the first iteration.
            The following iterations will read from the cache.
        with_row_id : bool, optional
            If set true, the returned batch will have an additional column named
            `_rowid` that contains the row id of the batch.
        rank: int, optional
            If set, the rank (idx) of this process in distributed training / inference.
        world_size: int, optional
            If set, the total number of processes in distributed training / inference.
        shard_granularity: str, optional
            The basic unit of sharding data. If set to "fragment", each worker will get
            the a subset of fragments.
            If set to "batch", it will read the "batch" interleave with the
            same fragments.
        batch_readahead: int, optional
            The number of batches to read ahead in different (Rust) threads for each
            fragment.
        sampler: callable, optional
            A function that samples the dataset.
        to_tensor_fn : callable, optional
            A function that converts a pyarrow RecordBatch to torch.Tensor.
        """
        super().__init__(*args, **kwargs)
        if isinstance(dataset, (str, Path)):
            dataset = lance.dataset(dataset)
        self.dataset = dataset
        self.columns = columns
        self.batch_size = batch_size
        self.samples: Optional[int] = samples
        self.filter = filter
        self.with_row_id = with_row_id
        self.batch_readahead = batch_readahead
        if to_tensor_fn is None:
            to_tensor_fn = _to_tensor
        self._to_tensor_fn = to_tensor_fn

        # As Shared Dataset
        self.rank = rank
        self.world_size = world_size
        self.shard_granularity = shard_granularity
        if sampler is None:
            if shard_granularity is None:
                if rank is not None or world_size is not None:
                    warnings.warn(
                        "rank and world_size are deprecated,"
                        + " use ShardedFragmentSampler instead.",
                    )
                    sampler = ShardedFragmentSampler(rank=rank, world_size=world_size)
                else:
                    sampler = FullScanSampler()
            elif shard_granularity == "batch":
                sampler = ShardedBatchSampler(rank, world_size)
            elif shard_granularity == "fragment":
                sampler = ShardedFragmentSampler(rank, world_size)
            else:
                raise ValueError("Invalid shard_granularity: {}")

        if filter is not None and self.samples > 0 or self.samples is None:
            raise ValueError("`filter` is not supported with `samples`")

        self.sampler: Sampler = sampler

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
            else:
                raw_stream = self.sampler(
                    self.dataset,
                    columns=self.columns,
                    filter=self.filter,
                    batch_size=self.batch_size,
                    with_row_id=self.with_row_id,
                    batch_readahead=self.batch_readahead,
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
