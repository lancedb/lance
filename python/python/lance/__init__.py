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

import pandas as pd

from .dataset import LanceDataset, __version__, write_dataset
from .util import sanitize_ts

__all__ = [
    "LanceDataset",
    "__version__",
    "write_dataset",
    "dataset",
]


def dataset(
    uri: Union[str, Path],
    version: Optional[int] = None,
    asof: Optional[Union[datetime, pd.Timestamp, str]] = None,
) -> LanceDataset:
    """
    Opens the Lance dataset from the address specified.

    Parameters
    ----------
    uri : str
        Address to the Lance dataset.
    version : optional, int
        If specified, load a specific version of the Lance dataset. Else, loads the latest version.
    asof : optional, datetime or str
        If specified, find the latest version created on or earlier than the given argument value.
        If a version is already specified, this arg is ignored.
    """
    ds = LanceDataset(uri, version)
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
            return LanceDataset(uri, ver_cutoff)
    else:
        return ds
