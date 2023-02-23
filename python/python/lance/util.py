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

import os
from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd


def sanitize_ts(ts: Union[datetime, pd.Timestamp, str]) -> datetime:
    """Returns a python datetime object from various timestamp input types."""
    if isinstance(ts, str):
        ts = pd.to_datetime(ts).to_pydatetime()
    elif isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    elif not isinstance(ts, datetime):
        raise TypeError(f"Unrecognized version timestamp {ts} of type {type(ts)}")
    return ts


def sanitize_file_address(path: Union[str, Path]) -> str:
    """Clean up the path provided for a Lance dataset and return it as a string.
    - Handles tilde expansion
    - Expands Path objects to absolute paths
    """
    if isinstance(path, Path):
        return os.fspath(path.expanduser().absolute())
    elif isinstance(path, str):
        return os.path.expanduser(path)
    else:
        raise TypeError(f"Unrecognized path {path} of type: {type(path)}")
