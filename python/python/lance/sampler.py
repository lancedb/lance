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

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import pyarrow as pa

import lance

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = ["maybe_sample"]


def _efficient_sample(
    dataset: lance.LanceDataset,
    n: int,
    columns: list[str],
    batch_size: int,
    max_takes: int,
) -> Generator[pa.RecordBatch]:
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
        having to issue too much random reads.
    """
    buf: list[pa.Table] = []
    total_records = len(dataset)
    assert total_records > n
    chunk_size = total_records // max_takes
    chunk_sample_size = n // max_takes
    for i in range(0, total_records, chunk_size):
        offset = i + np.random.randint(0, chunk_size - chunk_sample_size)
        buf.extend(
            dataset.to_batches(
                columns=columns,
                batch_size=chunk_size,
                offset=offset,
                limit=chunk_sample_size,
            )
        )
        if sum([len(b) for b in buf]) >= batch_size:
            yield pa.Table.from_batches(buf)
            buf.clear()
    if buf:
        yield pa.Table.from_batches(buf)


def maybe_sample(
    dataset: Union[str, Path, lance.LanceDataset],
    n: int,
    columns: Union[list[str], str],
    batch_size: int = 10240,
    max_takes: int = 8194,
) -> Generator[pa.RecordBatch]:
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
        The maximum number of takes to perform, by default 8194.
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
        dataset.to_batches(columns=columns, batch_size=batch_size)
    else:
        if n > max_takes:
            yield from _efficient_sample(dataset, n, columns, batch_size, max_takes)
        else:
            choices = np.random.choice(len(dataset), n, replace=False)
            yield dataset.take_rows(choices, columns=columns)
