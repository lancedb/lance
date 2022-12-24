# Copyright 2022 Lance Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union

import pandas
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs
from pyarrow._dataset import Dataset

from . import version

__version__ = version.__version__

from lance.lib import (
    FileSystemDataset,
    LanceFileFormat,
    _lance_dataset_make,
    _lance_dataset_write,
)
from lance.types import register_extension_types

__all__ = [
    "dataset",
    "write_dataset",
    "LanceFileFormat",
    "__version__",
    "diff",
    "compute_metric",
]

from .util.versioning import LanceDiff, compute_metric, get_version_asof

register_extension_types()


def _dataset_plain(uri: str, **kwargs) -> ds.FileSystemDataset:
    return ds.dataset(uri, format=LanceFileFormat(), **kwargs)


def dataset(
    uri: Union[str, Path],
    version: Optional[int] = None,
    asof: Optional[Union[datetime, pd.Timestamp, str]] = None,
    filesystem: Optional[pa.fs.FileSystem] = None,
    **kwargs,
) -> FileSystemDataset:
    """
    Create an Arrow Dataset from the given lance uri.

    It supports to read both versioned dataset, and plain (legacy) dataset.

    Parameters
    ----------
    uri: str
        The uri to the lance data
    version: optional, int
        If specified, load a specific version of the dataset.
    asof: optional datetime/Timestamp/str
        If specified find the latest version created on or earlier than
        given argument value. If version is specified, this is ignored.
    filesystem: pyarrow.fs.FileSystem
        File system instance to read.

    Other Parameters
    ----------------

    """
    if not filesystem:
        from pyarrow.fs import _resolve_filesystem_and_path

        filesystem, uri = _resolve_filesystem_and_path(uri, filesystem)

    if version is None:
        if _is_plain_dataset(filesystem, uri):
            return _dataset_plain(uri, filesystem=filesystem, **kwargs)

        if asof is not None:
            ds = _get_versioned_dataset(filesystem, uri, version)
            version = get_version_asof(ds, asof)

    return _get_versioned_dataset(filesystem, uri, version)


def diff(dataset: FileSystemDataset, v1: int, v2: int = None) -> LanceDiff:
    """
    Get the difference from v1 to v2 of this dataset

    Parameters
    ----------
    dataset: FileSystemDataset
        The dataset we want to get the diff for
    v1: int
        Start version. If negative then it is assumed to be offset from latest.
        So -1 would mean the second-to-last version (i.e., dataset.versions()[-2])
    v2: int, default None
        End version. If not specified (or 0), use the current version in the given
        dataset.
    """
    if v1 < 0:
        v1 = dataset.versions()[v1 - 1]["version"]
    if v2 is None:
        v2 = dataset.version["version"]
    if v2 < 0:
        v2 = dataset.versions()[v2 - 1]["version"]
    if v1 > v2:
        raise ValueError("v2 must not be less than v1")
    return LanceDiff(
        dataset.checkout(v1), dataset if v2 is None else dataset.checkout(v2)
    )


def _is_plain_dataset(filesystem: pa.fs.FileSystem, uri: str):
    manifest = os.path.join(uri, "_latest.manifest")
    return filesystem.get_file_info(manifest).type == pa.fs.FileType.NotFound


def _get_versioned_dataset(
    filesystem: pa.fs.FileSystem, uri: str, version: Optional[int] = None
):
    # Read the versioned dataset layout.
    has_version = True
    if version is None:
        has_version = False
        version = 0
    return _lance_dataset_make(filesystem, uri, has_version, version)


def write_dataset(
    data: Union[pa.Table, pa.dataset.Dataset, pandas.DataFrame, pandas.DataFrame],
    base_dir: Union[str, Path],
    mode: str = "create",
    filesystem: Optional[pa.fs.FileSystem] = None,
    max_rows_per_file: int = 0,
    min_rows_per_group: int = 0,
    max_rows_per_group: int = 1024,
    schema: Optional[pa.Schema] = None,
    **kwargs,
):
    """Write a dataset.

    Parameters
    ----------
    data : pyarrow.Table, pyarrow.dataset.Dataset, pandas.DataFrame
        The data to write.
    base_dir : str or Path
        The root directory to write dataset to.
    mode : str
        Write mode, accepts values are, **'create' | 'append' | 'overwrite'**
    filesystem : pyarrow.fs.FileSystem, optional
        The filesystem to write the dataset
    max_rows_per_file : int
        Maximum number of rows per file. If greater than 0 then this will limit how many rows
        are placed in any single file. Otherwise, there will be no limit and one file will be
        created in each output directory unless files need to be closed to respect max_open_files.
    min_rows_per_group : int
        Minimum number of rows per group. When the value is greater than 0,
        the dataset writer will batch incoming data and only write the row groups
        to the disk when sufficient rows have accumulated.
    max_rows_per_group : int
        Maximum number of rows per group. If the value is greater than 0,
        then the dataset writer may split up large incoming batches into multiple row groups.
        If this value is set, then min_rows_per_group should also be set.
        Otherwise, it could end up with very small row groups.
    schema : pyarrow.Schema, optional
        The expected schema of the pandas DataFrame.
        This can be used to indicate the type of columns if we cannot infer it automatically.
    """
    from pyarrow.fs import _resolve_filesystem_and_path

    if 0 < max_rows_per_file < max_rows_per_group:
        raise ValueError(
            "Max rows per file must be larger or equal to max_rows_per_group"
        )

    if isinstance(data, pandas.DataFrame):
        data = pa.Table.from_pandas(data, schema=schema)
    if isinstance(data, pa.Table):
        data = pa.dataset.InMemoryDataset(data)

    filesystem, base_dir = _resolve_filesystem_and_path(base_dir, filesystem)

    _lance_dataset_write(
        data,
        base_dir,
        filesystem,
        mode,
        max_rows_per_file,
        min_rows_per_group,
        max_rows_per_group,
    )
