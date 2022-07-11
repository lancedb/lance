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

from typing import Union
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
from lance.lib import LanceFileFormat, WriteTable

__all__ = ["dataset", "write_table"]


def dataset(uri: str) -> ds.Dataset:
    """
    Create an Arrow Dataset from the given lance uri.

    Parameters
    ----------
    uri: str
        The uri to the lance data
    """
    fmt = LanceFileFormat()
    return ds.dataset(uri, format=fmt)


def write_table(table: pa.Table, destination: Union[str, Path], primary_key: str):
    """Write an Arrow Table into the destination.

    Parameters
    ----------
    table : pa.Table
        Apache Arrow Table
    destination : str or `Path`
        The destination to write dataset to.
    primary_key : str
        The column name of the primary key.
    """
    WriteTable(table, destination, primary_key)
