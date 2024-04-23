# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Dict, Generator, List, Literal, Optional, Union

import lance
from lance.dataset import LanceDataset
from lance.dependencies import torch

if TYPE_CHECKING:
    from pathlib import Path

    import pyarrow as pa

__all__ = ["ShardedBatchIterator"]


class ShardedBatchIterator:
    """An iterator of RecordBatches, over the sharded dataset.

    Parmeters
    ---------
    uri: str or Path
        Dataset base URI
    rank: int
        The rank (id) of the shard in total `world_rank` shards.
    world_rank: int
        Total number of shards.
    columns: list of str, or dict of str to str default None
        List of column names to be fetched.
        Or a dictionary of column names to SQL expressions.
        All columns are fetched if None or unspecified.
    batch_size: int, optional
        The batch size of each shard.
    granularity: str, optional
        The granularity of the sharding, either "fragment" or "batch".
        Defaults to "fragment", which is more performant.
    with_row_id: bool, optional

    Examples
    --------

    .. code-block:: python

        import lance
        from lance._dataset import ShardDataset

        ds = ShardedDataset("s3://my-bucket/my-dataset",
            rank=2, world_size=8, columns=["col1", "col2"])
        for batch in ds:
            print(batch)
    """

    def __init__(
        self,
        data: Union[str, Path, LanceDataset],
        rank: int,
        world_size: int,
        *,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        batch_size: int = 1024 * 10,
        granularity: Literal["fragment", "batch"] = "fragment",
        batch_readahead: int = 8,
        with_row_id: bool = False,
    ):
        warnings.warn(
            "ShardedBatchIterator is deprecated, use :class:`Sampler` instead",
        )
        self._rank = rank
        self._world_size = world_size
        self._batch_size = batch_size
        self._granularity = granularity
        self._columns = columns
        self._with_row_id = with_row_id
        self._batch_readahead = batch_readahead

        self._ds: LanceDataset = (
            data if isinstance(data, LanceDataset) else lance.dataset(data)
        )

    @staticmethod
    def from_torch(
        data: Union[str, Path, LanceDataset],
        *,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        batch_size: int = 1024 * 10,
        granularity: Literal["fragment", "batch"] = "fragment",
        batch_readahead: int = 8,
        with_row_id: bool = False,
    ) -> ShardedBatchIterator:
        """Use from a PyTorch distributed environment.

        Automatically infer `rank` and `world_size` from `torch.distributed`.

        Other parameters, see :py:meth:`ShardedBatchIterator.__init__`.
        """
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        return ShardedBatchIterator(
            data,
            rank,
            world_size,
            columns=columns,
            batch_size=batch_size,
            granularity=granularity,
            batch_readahead=batch_readahead,
            with_row_id=with_row_id,
        )

    def _iter_over_fragments(self) -> Generator[pa.RecordBatch, None, None]:
        fragments = self._ds.get_fragments()
        for idx in range(self._rank, len(fragments), self._world_size):
            frag = fragments[idx]
            for batch in frag.to_batches(
                columns=self._columns,
                batch_size=self._batch_size,
                with_row_id=self._with_row_id,
                batch_readahead=self._batch_readahead,
            ):
                yield batch

    def _iter_over_batches(self) -> Generator[pa.RecordBatch, None, None]:
        if self._with_row_id:
            raise NotImplementedError(
                "with_row_id is not supported for batch sharding yet"
            )

        total = self._ds.count_rows()

        def _gen_ranges():
            for start in range(
                self._rank * self._batch_size,
                total,
                self._world_size * self._batch_size,
            ):
                yield start, min(start + self._batch_size, total)

        return self._ds._ds.take_scan(
            _gen_ranges(),
            columns=self._columns,
        )

    def __iter__(self):
        if self._granularity == "fragment":
            return self._iter_over_fragments()
        elif self._granularity == "batch":
            return self._iter_over_batches()
        else:
            raise ValueError(f"Unrecognized granularity: {self._granularity}")
