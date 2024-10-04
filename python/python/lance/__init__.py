# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Union

from .blob import BlobColumn, BlobFile
from .dataset import (
    LanceDataset,
    LanceOperation,
    LanceScanner,
    MergeInsertBuilder,
    __version__,
    batch_udf,
    write_dataset,
)
from .fragment import FragmentMetadata, LanceFragment
from .schema import json_to_schema, schema_to_json
from .util import sanitize_ts

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from lance.commit import CommitLock
    from lance.dependencies import pandas as pd

    ts_types = Union[datetime, pd.Timestamp, str]


__all__ = [
    "BlobColumn",
    "BlobFile",
    "FragmentMetadata",
    "LanceDataset",
    "LanceFragment",
    "LanceOperation",
    "LanceScanner",
    "MergeInsertBuilder",
    "__version__",
    "write_dataset",
    "schema_to_json",
    "json_to_schema",
    "dataset",
    "batch_udf",
]


def dataset(
    uri: Union[str, Path],
    version: Optional[int | str] = None,
    asof: Optional[ts_types] = None,
    block_size: Optional[int] = None,
    commit_lock: Optional[CommitLock] = None,
    index_cache_size: Optional[int] = None,
    storage_options: Optional[Dict[str, str]] = None,
) -> LanceDataset:
    """
    Opens the Lance dataset from the address specified.

    Parameters
    ----------
    uri : str
        Address to the Lance dataset.
    version : optional, int | str
        If specified, load a specific version of the Lance dataset. Else, loads the
        latest version. A version number (`int`) or a tag (`str`) can be provided.
    asof : optional, datetime or str
        If specified, find the latest version created on or earlier than the given
        argument value. If a version is already specified, this arg is ignored.
    block_size : optional, int
        Block size in bytes. Provide a hint for the size of the minimal I/O request.
    commit_handler : optional, CommitLock
        If specified, use the provided commit handler to lock the table while
        committing a new version. Not necessary on object stores other than S3
        or when there are no concurrent writers.
    index_cache_size : optional, int
        Index cache size. Index cache is a LRU cache with TTL. This number specifies the
        number of index pages, for example, IVF partitions, to be cached in
        the host memory. Default value is ``256``.

        Roughly, for an ``IVF_PQ`` partition with ``n`` rows, the size of each index
        page equals the combination of the pq code (``nd.array([n,pq], dtype=uint8))``
        and the row ids (``nd.array([n], dtype=uint64)``).
        Approximately, ``n = Total Rows / number of IVF partitions``.
        ``pq = number of PQ sub-vectors``.
    storage_options : optional, dict
        Extra options that make sense for a particular storage connection. This is
        used to store connection parameters like credentials, endpoint, etc.
    """
    ds = LanceDataset(
        uri,
        version,
        block_size,
        commit_lock=commit_lock,
        index_cache_size=index_cache_size,
        storage_options=storage_options,
    )
    if version is None and asof is not None:
        ts_cutoff = sanitize_ts(asof)
        ver_cutoff = max(
            [v["version"] for v in ds.versions() if v["timestamp"] < ts_cutoff],
            default=None,
        )
        if ver_cutoff is None:
            raise ValueError(
                f"{ts_cutoff} is earlier than the first version of this dataset"
            )
        else:
            return LanceDataset(
                uri,
                ver_cutoff,
                block_size,
                commit_lock=commit_lock,
                index_cache_size=index_cache_size,
            )
    else:
        return ds
