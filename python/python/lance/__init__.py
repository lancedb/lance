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

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

try:
    import pandas as pd

    ts_types = Union[datetime, pd.Timestamp, str]
except ImportError:
    pd = None
    ts_types = Union[datetime, str]

from .dataset import LanceDataset, LanceScanner, __version__, write_dataset
from .fragment import FragmentMetadata, LanceFragment
from .schema import json_to_schema, schema_to_json
from .util import sanitize_ts

__all__ = [
    "LanceDataset",
    "LanceScanner",
    "__version__",
    "write_dataset",
    "schema_to_json",
    "json_to_schema",
    "dataset",
    "FragmentMetadata",
    "LanceFragment",
]


def dataset(
    uri: Union[str, Path],
    version: Optional[int] = None,
    asof: Optional[ts_types] = None,
    block_size: Optional[int] = None,
) -> LanceDataset:
    """
    Opens the Lance dataset from the address specified.

    Parameters
    ----------
    uri : str
        Address to the Lance dataset.
    version : optional, int
        If specified, load a specific version of the Lance dataset. Else, loads the
        latest version.
    asof : optional, datetime or str
        If specified, find the latest version created on or earlier than the given
        argument value. If a version is already specified, this arg is ignored.
    block_size : optional, int
        Block size in bytes. Provide a hint for the size of the minimal I/O request.
    """
    ds = LanceDataset(uri, version, block_size)
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
            return LanceDataset(uri, ver_cutoff, block_size)
    else:
        return ds
