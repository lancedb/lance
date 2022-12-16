#  Copyright 2022 Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from datetime import datetime, timezone

import pandas as pd

from lance.lib import FileSystemDataset


def _sanitize_ts(ts: [datetime, pd.Timestamp, str]) -> datetime:
    if isinstance(ts, str):
        ts = pd.Timestamp.fromisoformat(ts).to_pydatetime()
    elif isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    elif not isinstance(ts, datetime):
        raise TypeError(f"Unrecognized version timestamp {ts} " f"of type {type(ts)}")
    return ts.astimezone(timezone.utc)


def get_version_asof(ds: FileSystemDataset, ts: [datetime, pd.Timestamp, str]) -> int:
    """
    Get the latest version that was accessible at the time of the given timestamp

    Parameters
    ----------
    ds: FileSystemDataset
        The versioned lance dataset
    ts: datetime, pd.Timestamp, or str
        The given timestamp

    Returns
    -------
    v: int
        Version number
    """
    ts = _sanitize_ts(ts)
    for v in reversed(ds.versions()):
        if v["timestamp"] <= ts:
            return v["version"]
    raise ValueError(f"{ts} is earlier than the first version of this dataset")
