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

from pathlib import Path
from typing import Optional, Union
import os

import pyarrow as pa
import pyarrow.fs
import pyarrow.dataset as ds

from . import version

__version__ = version.__version__

from lance.lib import (
    LanceFileFormat,
    _lance_dataset_write,
    _lance_dataset_make,
    FileSystemDataset,
)
from lance.types import register_extension_types

__all__ = ["dataset", "write_dataset", "LanceFileFormat", "__version__"]

register_extension_types()


def _dataset_plain(uri: str, **kwargs) -> ds.FileSystemDataset:
    return ds.dataset(uri, format=LanceFileFormat(), **kwargs)


def dataset(
    uri: Union[str, Path],
    version: Optional[int] = None,
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
    filesystem: pyarrow.fs.FileSystem
        File system instance to read.

    Other Parameters
    ----------------

    """
    if not filesystem:
        filesystem, uri = pa.fs.FileSystem.from_uri(uri)

    if version is None:
        if (
            filesystem.get_file_info(os.path.join(uri, "_latest.manifest")).type
            == pa.fs.FileType.NotFound
        ):
            return _dataset_plain(uri, filesystem=filesystem, **kwargs)

    # Read the versioned dataset layout.
    has_version = True
    if version is None:
        has_version = False
        version = 0
    return _lance_dataset_make(filesystem, uri, has_version, version)


def write_dataset(
    data: Union[pa.Table, pa.dataset.Dataset],
    base_dir: Union[str, Path],
    mode: str = "create",
    filesystem: Optional[pa.fs.FileSystem] = None,
    max_rows_per_file: int = 0,
    min_rows_per_group: int = 0,
    max_rows_per_group: int = 1024,
    **kwargs,
):
    """Write a dataset.

    Parameters
    ----------
    data : pyarrow.Table or pyarrow.dataset.Dataset
        The data to write.
    base_dir : str or Path
        The root directory to write dataset to.
    mode : str
        Write mode, accepts values are, **'create' | 'append' | 'overwrite'**
    filesystem : pyarrow.fs.FileSystem, optional
        The filesystem to write the dataset
    max_rows_per_file : int
        Maximum number of rows per file. If greater than 0 then this will limit how many rows
        are placed in any single file. Otherwise there will be no limit and one file will be
        created in each output directory unless files need to be closed to respect max_open_files.
    min_rows_per_group : int
        Minimum number of rows per group. When the value is greater than 0,
        the dataset writer will batch incoming data and only write the row groups
        to the disk when sufficient rows have accumulated.
    max_rows_per_group : int
        Maximum number of rows per group. If the value is greater than 0,
        then the dataset writer may split up large incoming batches into multiple row groups.
        If this value is set, then min_rows_per_group should also be set.
        Otherwise it could end up with very small row groups.
    """
    from pyarrow.fs import _resolve_filesystem_and_path

    if 0 < max_rows_per_file < max_rows_per_group:
        raise ValueError(
            "Max rows per file must be larger or equal to max_rows_per_group"
        )

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
