# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import lance.sql
import pyarrow as pa


def test_no_dataset():
    schema = pa.schema([pa.field("Int64(5)", pa.int64(), nullable=False)])
    assert lance.sql.query("SELECT 5").to_table() == pa.table(
        {"Int64(5)": [5]}, schema=schema
    )


def test_aggregation(tmp_path):
    ds = lance.write_dataset(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}), tmp_path)
    schema = pa.schema([pa.field("sum(d1.a)", pa.int64(), nullable=True)])
    assert lance.sql.query("SELECT SUM(a) FROM d1 WHERE b > 4").with_dataset(
        "d1", ds
    ).to_table() == pa.table({"sum(d1.a)": [5]}, schema=schema)


def test_join(tmp_path):
    ds1 = lance.write_dataset(
        pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}), tmp_path / "d1"
    )
    ds2 = lance.write_dataset(
        pa.table({"c": [4, 5, 6], "d": ["x", "y", "z"]}), tmp_path / "d2"
    )
    expected = pa.table({"a": [3, 2, 1], "d": ["z", "y", "x"]})
    assert (
        lance.sql.query(
            "SELECT d1.a, d2.d FROM d1 INNER JOIN d2 ON d1.b = d2.c ORDER BY d1.a DESC"
        )
        .with_dataset("d1", ds1)
        .with_dataset("d2", ds2)
        .to_table()
        == expected
    )
