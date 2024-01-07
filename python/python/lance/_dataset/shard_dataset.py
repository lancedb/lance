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
from typing import Generator, List, Literal, Optional, Union

import pyarrow as pa

import lance
from lance.dataset import LanceDataset

__all__ = ["ShardDataset"]


def _shard_fragments(
    ds: LanceDataset,
    batch_size: int,
    rank: int,
    world_size: int,
    columns: Optional[List[str]] = None,
    with_row_id: bool = False,
) -> Generator[pa.RecordBatch, None, None]:
    fragments = ds.get_fragments()
    for idx in range(rank, len(fragments), world_size):
        frag = fragments[idx]
        for batch in frag.to_batches(
            columns=columns, batch_size=batch_size, with_row_id=with_row_id
        ):
            yield batch


def _shard_batches(
    ds: LanceDataset,
    batch_size: int,
    rank: int,
    world_size: int,
    columns: Optional[List[str]] = None,
    with_row_id: bool = False,
) -> Generator[pa.RecordBatch, None, None]:
    if with_row_id:
        raise NotImplementedError("with_row_id is not supported for batch sharding yet")

    total = ds.count_rows()

    def _gen_ranges():
        for start in range(rank * batch_size, total, world_size * batch_size):
            yield start, start + batch_size

    return ds._ds.take_scan(
        _gen_ranges(),
        columns=columns,
    )


class ShardDataset:
    """A dataset that is shard over the full dataset.

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

    Examples
    --------

    .. code-block:: python

        import lance
        from lance._dataset import ShardDataset

        ds = ShardDataset("s3://my-bucket/my-dataset",
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
    ):
        self._rank = rank
        self._world_size = world_size
        self._batch_size = batch_size
        self._granularity = granularity
        self._columns = columns

        self._ds: LanceDataset = (
            data if isinstance(data, LanceDataset) else lance.dataset(data)
        )

    def __iter__(self):
        if self._granularity == "fragment":
            return _shard_fragments(
                self._ds, self._batch_size, self._rank, self._world_size
            )
        elif self._granularity == "batch":
            return _shard_batches(
                self._ds, self._batch_size, self._rank, self._world_size
            )
        else:
            raise ValueError(f"Unrecognized granularity: {self._granularity}")
