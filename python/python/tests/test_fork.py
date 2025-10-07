# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import os
import sys
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pytest

N_DIMS = 768
NUM_ROWS = 100_000
NEW_ROWS = 10_000


def create_table(num_rows) -> pa.Table:
    return pa.table(
        {
            "a": pc.random(num_rows).cast(pa.float32()),
            "b": pa.array(range(0, num_rows)),
        }
    )


@pytest.mark.skipif(sys.platform == "win32", reason="Test not applicable on Windows")
def test_table_roundtrip(tmp_path: Path):
    uri = tmp_path

    tbl = create_table(100)
    lance.write_dataset(tbl, uri)

    os.fork()
    dataset = lance.dataset(uri)
    assert dataset.uri == str(uri.absolute())
    assert tbl.schema == dataset.schema
    assert tbl == dataset.to_table()

    one_col = dataset.to_table(columns=["a"])
    assert one_col == tbl.select(["a"])

    table = dataset.to_table(columns=["a"], limit=20)
    assert len(table) == 20
