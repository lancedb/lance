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

from pathlib import Path
from typing import Generator, List, Literal, Union

import pyarrow as pa

import lance
from lance.dataset import LanceDataset

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
    columns: list of strs, optional
        Select columns to scan.
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
        columns: List[str] = None,
        batch_size: int = 1024 * 10,
        granularity: Literal["fragment", "batch"] = "fragment",
        batch_readahead: int = 8,
        with_row_id: bool = False,
    ):
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
                yield start, start + self._batch_size

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
