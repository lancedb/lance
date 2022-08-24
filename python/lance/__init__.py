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

import platform
from pathlib import Path
from typing import Optional, Union

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from lance.lib import BuildScanner, LanceFileFormat, WriteTable
from lance.types import register_extension_types

if platform.system() == "Linux":
    # TODO enable on MacOS
    register_extension_types()

__all__ = ["dataset", "write_table", "scanner"]


def dataset(
    uri: str,
) -> ds.Dataset:
    """
    Create an Arrow Dataset from the given lance uri.

    Parameters
    ----------
    uri: str
        The uri to the lance data
    """
    fmt = LanceFileFormat()
    return ds.dataset(uri, format=fmt)


def scanner(
    data: Union[str, Path, ds.Dataset],
    columns: Optional[str] = None,
    filter: Optional[pc.Expression] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> ds.Scanner:
    if isinstance(data, (str, Path)):
        data = dataset(str(data))
    return BuildScanner(
        data, columns=columns, filter=filter, limit=limit, offset=offset
    )


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
