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
from __future__ import annotations

import itertools
from datetime import datetime, timezone
from functools import cached_property
from typing import Callable, Union

import pandas as pd
import pyarrow as pa

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


def compute_metric(
    ds: FileSystemDataset,
    metric_func: Callable[[FileSystemDataset], pd.DataFrame],
    versions: list = None,
    with_version: Union[bool, str] = True,
) -> pd.DataFrame:
    """
    Compare metrics across versions of a dataset

    Parameters
    ----------
    ds: FileSystemDataset
        The base dataset we want to compute metrics across versions for.
    metric_func: Callable[[FileSystemDataset], pd.DataFrame]
        Function to compute metrics DataFrame from a given dataset version.
    versions: list, default None
        All versions if not specified.
    with_version: bool or str, default True
        If bool then controls whether to add the version as auxiliary output.
        If str then assumed to be the name of the auxiliary output column.
    """
    if versions is None:
        versions = ds.versions()
    vcol_name = "version"
    if isinstance(with_version, str):
        vcol_name = with_version

    results = []
    for v in versions:
        if isinstance(v, dict):
            v = v["version"]
        vdf = metric_func(ds.checkout(v))
        if vcol_name in vdf:
            raise ValueError(f"{vcol_name} already in output df")
        vdf[vcol_name] = v
        results.append(vdf)
    return pd.concat(results)


def _compute_metric(version, uri, func, vcol_name):
    import lance

    vdf = func(lance.dataset(uri, version=version))
    if vcol_name in vdf:
        raise ValueError(f"{vcol_name} already in output df")
    vdf[vcol_name] = version
    return vdf


class LanceDiff:
    def __init__(self, v1: FileSystemDataset, v2: FileSystemDataset):
        self.v1 = v1
        self.v2 = v2

    def __repr__(self):
        return (
            "LanceDiff\n"
            f"  Added: {self.rows_added().count_rows} rows, "
            f"{len(self.columns_added().schema)} columns"
        )

    def rows_added(self, key: str = None) -> RowDiff:
        return RowDiff(self.v1, self.v2, key)

    def columns_added(self) -> ColumnDiff:
        """
        Get the net new fields between v1 and v2. You can then
        use the `schema` property to see new fields
         and `head(n)` method to get the data in those fields
        """
        v2_fields = _flat_schema(self.v2.schema)
        v1_fields = set([f.name for f in _flat_schema(self.v1.schema)])
        new_fields = [f for f in v2_fields if f.name not in v1_fields]
        return ColumnDiff(self.v2, new_fields)


class RowDiff:
    """
    Row diff between two dataset versions using the specified join keys
    """

    def __init__(
        self,
        ds_start: FileSystemDataset,
        ds_end: FileSystemDataset,
        key: [str, list[str]],
    ):
        self.ds_start = ds_start
        self.ds_end = ds_end
        self.key = [key] if isinstance(key, str) else key

    def _query(self, projection: list[str], offset: int = 0, limit: int = 0) -> str:
        join = " AND ".join([f"v2.{k}=v1.{k}" for k in self.key])
        query = (
            f"SELECT {','.join(projection)} FROM v2 "
            f"LEFT JOIN v1 ON {join} "
            f"WHERE v1.{self.key[0]} IS NULL"
        )
        if offset > 0:
            query += f" OFFSET {offset}"
        if limit > 0:
            query += f" LIMIT {limit}"
        return query

    @cached_property
    def count_rows(self) -> int:
        """
        Return the number of rows in this diff
        """
        return self.ds_end.count_rows() - self.ds_start.count_rows()

    def head(self, n: int = 10, columns: list[str] = None) -> pa.Table:
        """
        Retrieve the rows in this diff as a pyarrow table

        Parameters
        ----------
        n: int, default 10
            Get this many rows
        columns: list[str], default None
            Get all rows if not specified
        """
        try:
            import duckdb
        except ImportError:
            print("Please `pip install duckdb` to use the Lance data diff tool")
            raise
        v1 = self.ds_start
        v2 = self.ds_end
        if columns is None:
            columns = ["v2.*"]
        qry = self._query(columns, limit=n)
        return duckdb.query(qry).to_arrow_table()


class ColumnDiff:
    def __init__(self, ds: FileSystemDataset, fields: list[pa.Field]):
        self.dataset = ds
        self.fields = fields

    @property
    def schema(self) -> pa.Schema:
        """
        Flattened schema containing fields for this diff
        """
        return pa.schema(self.fields)

    def head(self, n=10, columns=None) -> pa.Table:
        """
        Return the first `n` rows for fields in this diff as a pyarrow Table

        Parameters
        ----------
        n: int, default 10
            How many rows to return
        columns: list[str], default None
            If None then all fields are returned
        """
        if columns is None:
            columns = [f.name for f in self.fields]
        return self.dataset.head(n, columns=columns)


def _flat_schema(schema):
    return itertools.chain(*[f.flatten() for f in schema])
