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
#

# PEP-585. Can be removed after deprecating python 3.8 support.
from __future__ import annotations

import gc
import logging
import random
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from heapq import heappush, heappushpop
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, TypeVar, Union

import pyarrow as pa

import lance
from lance.dependencies import numpy as np

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = ["maybe_sample"]


def _efficient_sample(
    dataset: lance.LanceDataset,
    n: int,
    columns: list[str],
    batch_size: int,
    max_takes: int,
) -> Generator[pa.RecordBatch, None, None]:
    """Sample n records from the dataset.

    Parameters
    ----------
    dataset : lance.LanceDataset
        The dataset to sample from.
    n : int
        The number of records to sample.
    columns : list[str]
        The columns to load.
    batch_size : int
        The batch size to use when loading the data.
    max_takes : int
        The maximum number of takes to perform. This is used to limit the number of
        random reads. Large enough value can give a good random sample without
        having to issue too many random reads.

    Returns
    -------
    Generator of a RecordBatch.
    """
    buf: list[pa.RecordBatch] = []
    total_records = len(dataset)
    assert total_records > n
    chunk_size = total_records // max_takes
    chunk_sample_size = n // max_takes

    num_sampled = 0

    for idx, i in enumerate(range(0, total_records, chunk_size)):
        # If we have already sampled enough, break. This can happen if there
        # is a remainder in the division.
        if num_sampled >= n:
            break
        num_sampled += chunk_sample_size

        # If we are at the last chunk, we may not have enough records to sample.
        local_size = min(chunk_size, total_records - i)
        local_sample_size = min(chunk_sample_size, local_size)

        if local_sample_size < local_size:
            # Add more randomness within each chunk, if there is room.
            offset = i + np.random.randint(0, local_size - local_sample_size)
        else:
            offset = i

        buf.extend(
            dataset.take(
                list(range(offset, offset + local_sample_size)),
                columns=columns,
            ).to_batches()
        )
        if idx % 50 == 0:
            logging.info("Sampled at offset=%s, len=%s", offset, chunk_sample_size)
        if sum(len(b) for b in buf) >= batch_size:
            tbl = pa.Table.from_batches(buf)
            buf.clear()
            tbl = tbl.combine_chunks()
            yield tbl.to_batches()[0]
            del tbl
    if buf:
        tbl = pa.Table.from_batches(buf).combine_chunks()
        yield tbl.to_batches()[0]
        del tbl


def maybe_sample(
    dataset: Union[str, Path, lance.LanceDataset],
    n: int,
    columns: Union[list[str], str],
    batch_size: int = 10240,
    max_takes: int = 2048,
) -> Generator[pa.RecordBatch, None, None]:
    """Sample n records from the dataset.

    Parameters
    ----------
    dataset : Union[str, Path, lance.LanceDataset]
        The dataset to sample from.
    n : int
        The number of records to sample.
    columns : Union[list[str], str]
        The columns to load.
    batch_size : int, optional
        The batch size to use when loading the data, by default 10240.
    max_takes : int, optional
        The maximum number of takes to perform, by default 2048.
        This is employed to minimize the number of random reads necessary for sampling.
        A sufficiently large value can provide an effective random sample without
        the need for excessive random reads.

    Returns
    -------
    Generator[pa.RecordBatch]
        A generator that yields [RecordBatch] of data.
    """
    if isinstance(dataset, (str, Path)):
        dataset = lance.dataset(dataset)

    if isinstance(columns, str):
        columns = [columns]

    if n >= len(dataset):
        # Dont have enough data in the dataset. Just do a full scan
        yield from dataset.to_batches(columns=columns, batch_size=batch_size)
    else:
        if n > max_takes:
            yield from _efficient_sample(dataset, n, columns, batch_size, max_takes)
        else:
            choices = np.random.choice(len(dataset), n, replace=False)
            idx = 0
            while idx < len(choices):
                end = min(idx + batch_size, len(choices))
                tbl = dataset.take(choices[idx:end], columns=columns).combine_chunks()
                yield tbl.to_batches()[0]
                idx += batch_size


T = TypeVar("T")


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: T = field(compare=False)


def reservoir_sampling(stream: Iterable[T], k: int) -> list[T]:
    rng = np.random.default_rng()
    heap = []
    for idx, item in enumerate(stream):
        entry = PrioritizedItem(rng.integers(0, k * 2), item)
        if len(heap) < k:
            heappush(heap, entry)
        else:
            vic = heappushpop(heap, entry)
            del vic
        if idx % 10240 == 0:
            logging.info("Force Python GC")
            gc.collect()
    samples = [i.item for i in heap]
    del heap
    return samples


class Sampler(ABC):
    """Sampler over LanceDataset.

    To implement a new `Sampler`, you can implement the `__call__` method to yield
    a `pyarrow.RecordBatch`.
    """

    @abstractmethod
    def __call__(
        self,
        ds: lance.LanceDataset,
        *args,
        batch_size: int = 128,
        columns: Optional[List[str]] = None,
        batch_readahead: int = 16,
        with_row_id: bool = False,
        **kwargs,
    ) -> Generator[pa.RecordBatch, None, None]:
        """A generator to yield `pyarrow.RecordBatch` from the dataset."""
        pass


class FragmentSampler(Sampler):
    """Sampling over Fragments.

    To implement a new `FragmentSampler`, you can implement the `iter_fragments` method
    to yield fragments in desired order.
    """

    def __call__(
        self,
        dataset: lance.LanceDataset,
        *args,
        batch_size: int = 128,
        columns: Optional[List[str]] = None,
        batch_readahead: int = 16,
        with_row_id: bool = False,
        **kwargs,
    ) -> Generator[pa.RecordBatch, None, None]:
        for fragment in self.iter_fragments(dataset, *args, **kwargs):
            for batch in fragment.to_batches(
                batch_size=batch_size,
                columns=columns,
                with_row_id=with_row_id,
                batch_readahead=batch_readahead,
            ):
                yield batch

    @abstractmethod
    def iter_fragments(
        self, ds: lance.LanceDataset, *args, **kwargs
    ) -> Generator[lance.LanceFragment, None, None]:
        """Iterate over data fragments."""
        pass


class FullScanSampler(FragmentSampler):
    """Default Sampler, which scan the entire dataset sequentially."""

    def iter_fragments(
        self, dataset: lance.LanceDataset, **kwargs
    ) -> Generator[lance.LanceFragment, None, None]:
        return dataset.get_fragments()


class ShardedFragmentSampler(FragmentSampler):
    """Sharded fragments by rank and world_size.

    Each rank / process will process a subset of the fragments.
    """

    def __init__(
        self, rank: int, world_size: int, randomize: bool = False, seed: int = 0
    ):
        """Initialize the ShardedFragmentSampler.

        Parameters
        ----------
        rank : int
            The rank of the process in the distributed cluster.
        world_size : int
            The total number of processes in the distributed cluster.
        randomize : bool
            If set true, randomize
        """
        super().__init__()

        self._rank = rank
        self._world_size = world_size
        self._randomize = randomize
        self._seed = seed

    @staticmethod
    def from_torch(randomize: bool = False, seed: int = 0) -> ShardedFragmentSampler:
        """Use from a PyTorch distributed environment.

        Automatically infer `rank` and `world_size` from `torch.distributed`.

        Other parameters, see :py:meth:`ShardedBatchIterator.__init__`.
        """
        import torch

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        return ShardedFragmentSampler(rank, world_size, randomize=randomize, seed=seed)

    def iter_fragments(
        self, dataset: lance.LanceDataset, **kwargs
    ) -> Generator[lance.LanceFragment, None, None]:
        fragments = dataset.get_fragments()
        if self._randomize:
            random.seed(self._seed)
            random.shuffle(fragments)
        for idx in range(self._rank, len(fragments), self._world_size):
            yield fragments[idx]


class ShardedBatchSampler(Sampler):
    """Sharded batch sampler.

    Each rank / process will process a subset of the batches.
    """

    def __init__(
        self, rank: int, world_size: int, randomize: bool = False, seed: int = 0
    ):
        self._rank = rank
        self._world_size = world_size
        self._randomize = randomize
        self._seed = seed

    @staticmethod
    def from_torch(randomize: bool = False, seed: int = 0) -> ShardedBatchSampler:
        """Use it from a PyTorch distributed environment.

        Automatically infer `rank` and `world_size` from `torch.distributed`.
        """
        import torch

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        return ShardedBatchSampler(rank, world_size, randomize=randomize, seed=seed)

    def __call__(
        self,
        dataset: lance.LanceDataset,
        *args,
        batch_size: int = 128,
        columns: Optional[List[str]] = None,
        batch_readahead: int = 16,
        with_row_id: Optional[bool] = None,
        **kwargs,
    ) -> Generator[lance.RecordBatch, None, None]:
        total = dataset.count_rows()

        if with_row_id is not None:
            warnings.warn(
                "with_row_id is not supported for ShardedBatchSampler",
            )

        def _gen_ranges():
            for start in range(
                self._rank * batch_size,
                total,
                self._world_size * batch_size,
            ):
                yield start, min(start + batch_size, total)

        ranges = list(_gen_ranges())
        if self._randomize:
            random.seed(self._seed)
            random.shuffle(ranges)

        return dataset._ds.take_scan(
            ranges,
            columns=columns,
            batch_readahead=batch_readahead,
        )
