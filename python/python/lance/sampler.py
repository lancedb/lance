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

__all__ = [
    "maybe_sample",
    "Sampler",
    "FragmentSampler",
    "FullScanSampler",
    "ShardedFragmentSampler",
    "ShardedBatchSampler",
]


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
        filter: Optional[str] = None,
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
        filter: Optional[str] = None,
        batch_readahead: int = 16,
        with_row_id: bool = False,
        **kwargs,
    ) -> Generator[pa.RecordBatch, None, None]:
        for fragment in self.iter_fragments(dataset, *args, **kwargs):
            for batch in fragment.to_batches(
                batch_size=batch_size,
                columns=columns,
                filter=filter,
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
    It yields batches from ``ds.fragments[rank::world_size]``.

    This sampler is more efficient than `ShardedBatchSampler` when the dataset is large.

    Parameters
    ----------
    rank : int
        The rank of the process in the distributed cluster.
    world_size : int
        The total number of processes in the distributed cluster.
    randomize : bool
        If set true, randomize
    seed : int
        The random seed to use when randomize is set true.
    """

    def __init__(
        self, rank: int, world_size: int, randomize: bool = False, seed: int = 0
    ):
        super().__init__()

        self._rank = rank
        self._world_size = world_size
        self._randomize = randomize
        self._seed = seed

    @staticmethod
    def from_torch(randomize: bool = False, seed: int = 0) -> ShardedFragmentSampler:
        """Use from a PyTorch distributed environment.

        Automatically infer `rank` and `world_size` from :mod:`torch.distributed`.
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

    The input is subdivided into batches (of size `batch_size`).  Each rank / process
    takes every Nth batch (where N is the world size).  The order in which batches
    are loaded is randomized.

    When there is no filter then each process only needs to load the rows assigned to
    it but this process is still slightly less efficient than ShardedFragmentSampler
    since it requires loading rows by range instead of loading all rows for a
    given fragment.

    If there is a filter then we cannot divide the row ids ahead of time.  Instead,
    each process will load the entire filtered dataset and discard the rows that are
    not assigned to it.  The resulting stream is then randomized via a reservoir
    sampler.  This does not perfectly randomize the stream but it should generate
    a stream that is random enough for many use cases.
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

    # Performs a filtered scan of the dataset and then throws away all but the Nth
    # rows (where N is the world size)
    def _shard_scan(
        self,
        dataset: lance.LanceDataset,
        batch_size: int,
        columns: Optional[List[str]],
        batch_readahead: int,
        filter: str,
    ) -> Generator[lance.RecordBatch, None, None]:
        accumulated_batches = []
        rows_accumulated = 0
        rows_to_skip = self._rank
        for batch in dataset.scanner(
            columns=columns,
            batch_readahead=batch_readahead,
            filter=filter,
            scan_in_order=True,
        ).to_batches():
            batch = batch.slice(rows_to_skip, batch.num_rows - rows_to_skip)
            # Take every Nth row
            indices = list(range(0, batch.num_rows, self._world_size))
            rows_to_skip = (
                self._world_size - (batch.num_rows % self._world_size)
            ) % self._world_size
            batch = batch.take(indices)

            # Add to our collection
            rows_accumulated += batch.num_rows
            accumulated_batches.append(batch)

            # If we have enough to generate 1 or more batches then do so
            if rows_accumulated > batch_size:
                big_batch = (
                    pa.Table.from_batches(accumulated_batches)
                    .combine_chunks()
                    .to_batches()[0]
                )
                accumulated_batches = []
                while big_batch.num_rows > batch_size:
                    next_batch = big_batch.slice(0, batch_size)
                    big_batch = big_batch.slice(batch_size)
                    yield next_batch
                rows_accumulated = big_batch.num_rows
                if big_batch.num_rows > 0:
                    accumulated_batches.append(big_batch)

    def _sample_filtered(
        self,
        dataset: lance.LanceDataset,
        batch_size: int,
        columns: Optional[List[str]],
        batch_readahead: int,
        filter: str,
    ) -> Generator[lance.RecordBatch, None, None]:
        shard_scan = self._shard_scan(
            dataset, batch_size, columns, batch_readahead, filter
        )
        if not self._randomize:
            yield from shard_scan

        random.seed(self._seed)
        heap = []
        # We want to randomize the incoming sequence.  The normal approach
        # is to pull the whole thing in memory and run fisher-yates.  We
        # want to avoid buffering the entire input.  So, as an approximation,
        # we are using a heap + random number in a style similar to reservoir
        # sampling.
        #
        # We will keep up to k batches in the reservoir.  The higher
        # k the more randomness we will get from the reservoir shuffle
        # but the more memory we need.
        #
        # Picking 256 as a heuristic which should be 32Ki rows with
        # the default batch size
        k = 256
        for batch in shard_scan:
            priority = random.randint(0, k * 2 - 1)
            entry = PrioritizedItem(priority, batch)
            if len(heap) < k:
                heappush(heap, entry)
            else:
                next_batch = heappushpop(heap, entry)
                yield next_batch.item
        for batch in heap:
            yield batch.item

    def _sample_all(
        self,
        dataset: lance.LanceDataset,
        batch_size: int,
        columns: Optional[List[str]],
        batch_readahead: int,
    ) -> Generator[lance.RecordBatch, None, None]:
        total = dataset.count_rows()

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

    def __call__(
        self,
        dataset: lance.LanceDataset,
        *args,
        batch_size: int = 128,
        columns: Optional[List[str]] = None,
        filter: Optional[str] = None,
        batch_readahead: int = 16,
        with_row_id: Optional[bool] = None,
        **kwargs,
    ) -> Generator[lance.RecordBatch, None, None]:
        if filter is None:
            if with_row_id is not None:
                warnings.warn(
                    "with_row_id is not supported for ShardedBatchSampler",
                )
            return self._sample_all(dataset, batch_size, columns, batch_readahead)
        else:
            return self._sample_filtered(
                dataset, batch_size, columns, batch_readahead, filter
            )
