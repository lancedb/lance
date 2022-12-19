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
import time
from datetime import datetime, timezone

import pyarrow as pa
import pytest
import pytz

import lance
from lance.util.versioning import (
    ColumnDiff,
    LanceDiff,
    RowDiff,
    get_version_asof,
)


def test_get_version_asof(tmp_path):
    too_early = datetime.now()
    # version timestamp resolution is at the second
    time.sleep(1)

    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    test_cases = {1: _get_test_timestamps(datetime.now())}

    # version timestamp resolution is at the second
    time.sleep(1)
    table2 = pa.Table.from_pylist([{"a": 100, "b": 200}])
    lance.write_dataset(table2, base_dir, mode="append")

    test_cases[2] = _get_test_timestamps(datetime.now())

    dataset = lance.dataset(base_dir)
    for v, ts_array in test_cases.items():
        for ts in ts_array:
            assert get_version_asof(dataset, ts) == v

    with pytest.raises(ValueError):
        get_version_asof(dataset, too_early)


def _get_test_timestamps(naive):
    return [
        naive,
        naive.astimezone(timezone.utc),
        naive.astimezone(pytz.timezone("America/Los_Angeles")),
    ]


def test_compute_metric(tmp_path):
    base_dir = tmp_path / "test"
    ds = _create_dataset(base_dir)

    def func(dataset):
        return dataset.to_table().to_pandas().max().to_frame().T

    metrics = lance.compute_metric(ds, func)
    assert "version" in metrics


def _create_dataset(base_dir):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}])
    lance.write_dataset(table1, base_dir)
    table2 = pa.Table.from_pylist([{"a": 100, "b": 200}])
    lance.write_dataset(table2, base_dir, mode="append")
    table3 = pa.Table.from_pylist([{"a": 100, "c": 100, "d": 200}])
    return lance.dataset(base_dir).merge(table3, left_on="a", right_on="a")


def test_diff(tmp_path):
    base_dir = tmp_path / "test"
    ds = _create_dataset(base_dir)

    d = lance.diff(ds, 1, 2)
    assert isinstance(d, LanceDiff)

    rows = d.rows_added(key="a")
    assert isinstance(rows, RowDiff)
    assert rows.count_rows == 1
    tbl = rows.head()
    assert isinstance(tbl, pa.Table)
    assert len(tbl) == 1

    cols = lance.diff(ds, 2, 3).columns_added()
    assert isinstance(cols, ColumnDiff)
    assert len(cols.schema) == 2
    tbl = cols.head()
    assert isinstance(tbl, pa.Table)
    assert len(tbl) == 2

    assert cols.schema == lance.diff(ds, -1).columns_added().schema