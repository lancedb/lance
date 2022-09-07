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
from typing import List, Optional, Union

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
    columns: Optional[List[str]] = None,
    filter: Optional[pc.Expression] = None,
    batch_size: Optional[int] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> ds.Scanner:
    """Build a PyArrow Dataset scanner.

    It extends PyArrow Scanner with limit pushdown.

    Parameters
    ----------
    data: uri, path or pyarrow dataset
        The Dataset
    columns: List[str], optional
        Specify the columns to read.
    filter: pc.Expression, optional
        Apply filter to the scanner.
    batch_size: int
        The maximum number of records to scan for each batch.
    limit: int
        Limit the number of records to return in total.
    offset: int
        The offset to read the data from.

    See Also
    --------
    https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Scanner.html#pyarrow.dataset.Scanner
    """
    if isinstance(data, (str, Path)):
        data = dataset(str(data))
    return BuildScanner(
        data,
        columns=columns,
        filter=filter,
        batch_size=batch_size,
        limit=limit,
        offset=offset,
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
