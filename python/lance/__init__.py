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

from lance.lib import LanceFileFormat, WriteTable, _lance_dataset_write, _wrap_dataset
from lance.types import register_extension_types

__all__ = ["dataset", "write_table", "write_dataset", "LanceFileFormat", "__version__"]

register_extension_types()


def _dataset_plain(uri: str, **kwargs) -> ds.FileSystemDataset:
    fmt = LanceFileFormat()
    data = ds.dataset(uri, format=fmt, **kwargs)
    return _wrap_dataset(data)


def dataset(
    uri: Union[str, Path],
    version: Optional[int] = None,
    filesystem: Optional[pa.fs.FileSystem] = None,
    **kwargs,
) -> ds.FileSystemDataset:
    """
    Create an Arrow Dataset from the given lance uri.

    It supports to read both versioned dataset, and plain (legacy) dataset.

    Parameters
    ----------
    uri: str
        The uri to the lance data
    version: optional, int
        If specified, load a specific version of the dataset.
    filesystem: pa.fs.FileSystem
        File system instance to read.
    """
    if not filesystem:
        filesystem, uri = pa.fs.FileSystem.from_uri(uri)

    if version is None:
        if (
            filesystem.get_file_info(os.path.join(uri, "_latest.manifest")).type
            == pa.fs.FileType.NotFound
        ):
            return _dataset_plain(uri, **kwargs)

    # Read new one


def write_table(table: pa.Table, destination: Union[str, Path], batch_size: int = 1024):
    """Write an Arrow Table into the destination.

    Parameters
    ----------
    table : pa.Table
        Apache Arrow Table
    destination : str or `Path`
        The destination to write dataset to.
    batch_size : int, optional
        Set the batch size to write to disk.
    """
    WriteTable(table, destination, batch_size=batch_size)


def write_dataset(
    data: Union[pa.Table, pa.dataset.Dataset],
    base_dir: Union[str, Path],
    mode: str = "create",
    filesystem: pa.fs.FileSystem = None,
    **kwargs,
):
    """Write a dataset.

    Parameters
    ----------
    data : pa.Table or pa.dataset.Dataset
        The data to write.
    base_dir : str or Path
        The root directory to write dataset to.
    mode : str, 'create' | 'append' | 'overwrite'
        Write mode.
    filesystem : pa.fs.FileSystem, optional
        The filesystem to write the dataset
    """
    from pyarrow.fs import _resolve_filesystem_and_path

    if isinstance(data, pa.Table):
        data = pa.dataset.InMemoryDataset(data)

    filesystem, base_dir = _resolve_filesystem_and_path(base_dir, filesystem)

    _lance_dataset_write(data, base_dir, filesystem, mode)
