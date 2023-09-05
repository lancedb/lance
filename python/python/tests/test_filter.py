#  Copyright 2023 Lance Developers
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


"""Tests for predicate pushdown"""

import random
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from lance.vector import vec_to_table


def create_table(nrows=100):
    intcol = pa.array(range(nrows))
    floatcol = pa.array(np.arange(nrows) * 2 / 3, type=pa.float32())
    arr = np.arange(nrows) < nrows / 2
    structcol = pa.StructArray.from_arrays(
        [
            pa.array(arr, type=pa.bool_()),
            pa.array([date(2021, 1, 1) + timedelta(days=i) for i in range(nrows)]),
            pa.array([datetime(2021, 1, 1) + timedelta(hours=i) for i in range(nrows)]),
        ],
        names=["bool", "date", "dt"],
    )
    random.seed(42)

    def gen_str(n):
        return "".join(random.choices("abc", k=n))

    string_col = pa.array([gen_str(2) for _ in range(nrows)])

    decimal_col = pa.array([Decimal(f"{str(i)}.000") for i in range(nrows)])

    tbl = pa.Table.from_arrays(
        [intcol, floatcol, structcol, string_col, decimal_col],
        names=["int", "float", "rec", "str", "decimal"],
    )
    return tbl


@pytest.fixture()
def dataset(tmp_path: Path):
    tbl = create_table()
    yield lance.write_dataset(tbl, tmp_path)


def test_simple_predicates(dataset):
    predicates = [
        pc.field("int") >= 50,
        pc.field("int") == 50,
        pc.field("int") != 50,
        pc.field("float") < 30.0,
        pc.field("float") > 30.0,
        pc.field("float") <= 30.0,
        pc.field("float") >= 30.0,
        pc.field("str") != "aa",
        pc.field("str") == "aa",
    ]
    # test simple
    for expr in predicates:
        assert dataset.to_table(filter=expr) == dataset.to_table().filter(expr)


def test_sql_predicates(dataset):
    print(dataset.to_table())
    # Predicate and expected number of rows
    predicates_nrows = [
        ("int >= 50", 50),
        ("int = 50", 1),
        ("int != 50", 99),
        ("float < 30.0", 45),
        ("str = 'aa'", 16),
        ("str in ('aa', 'bb')", 26),
        ("rec.bool", 50),
        ("rec.date = cast('2021-01-01' as date)", 1),
        ("rec.dt = cast('2021-01-01 00:00:00' as timestamp(6))", 1),
        ("rec.dt = cast('2021-01-01 00:00:00' as timestamp)", 1),
        ("rec.dt = cast('2021-01-01 00:00:00' as datetime(6))", 1),
        ("rec.dt = cast('2021-01-01 00:00:00' as datetime)", 1),
        ("rec.dt = TIMESTAMP '2021-01-01 00:00:00'", 1),
        ("rec.dt = TIMESTAMP(6) '2021-01-01 00:00:00'", 1),
        ("rec.date = DATE '2021-01-01'", 1),
        ("rec.date >= cast('2021-01-31' as date)", 70),
        ("cast(rec.date as string) = '2021-01-01'", 1),
        ("decimal = DECIMAL(5,3) '12.000'", 1),
        ("decimal >= DECIMAL(5,3) '50.000'", 50),
    ]

    for expr, expected_num_rows in predicates_nrows:
        assert dataset.to_table(filter=expr).num_rows == expected_num_rows


def test_compound(dataset):
    predicates = [
        pc.field("int") >= 50,
        pc.field("float") < 90.0,
        pc.field("str") == "aa",
    ]
    # test compound
    for expr in predicates:
        for other_expr in predicates:
            compound = expr & other_expr
            assert dataset.to_table(filter=compound) == dataset.to_table().filter(
                compound
            )
            compound = expr | other_expr
            assert dataset.to_table(filter=compound) == dataset.to_table().filter(
                compound
            )


def test_match(tmp_path: Path, provide_pandas: bool):
    array = pa.array(["aaa", "bbb", "abc", "bca", "cab", "cba"])
    table = pa.Table.from_arrays([array], names=["str"])
    dataset = lance.write_dataset(table, tmp_path / "test_match")

    result = dataset.to_table(filter="str LIKE 'a%'").to_pandas()
    pd.testing.assert_frame_equal(result, pd.DataFrame({"str": ["aaa", "abc"]}))

    result = dataset.to_table(filter="str NOT LIKE 'a%'").to_pandas()
    pd.testing.assert_frame_equal(
        result, pd.DataFrame({"str": ["bbb", "bca", "cab", "cba"]})
    )

    result = dataset.to_table(filter="regexp_match(str, 'c.+')").to_pandas()
    pd.testing.assert_frame_equal(result, pd.DataFrame({"str": ["bca", "cab", "cba"]}))


def test_escaped_name(tmp_path: Path, provide_pandas: bool):
    table = pa.table({"silly :name": pa.array([0, 1, 2])})
    dataset = lance.write_dataset(table, tmp_path / "test_escaped_name")

    dataset = lance.dataset(tmp_path / "test_escaped_name")
    result = dataset.to_table(filter="`silly :name` > 1").to_pandas()
    pd.testing.assert_frame_equal(result, pd.DataFrame({"silly :name": [2]}))

    # nested case
    table = pa.table({"outer field": pa.array([{"inner field": i} for i in range(3)])})
    dataset = lance.write_dataset(table, tmp_path / "test_escaped_name_nested")

    dataset = lance.dataset(tmp_path / "test_escaped_name_nested")
    result = dataset.to_table(filter="`outer field`.`inner field` > 1").to_pandas()
    pd.testing.assert_frame_equal(
        result, pd.DataFrame({"outer field": [{"inner field": 2}]})
    )

    # test uppercase name
    table = pa.table({"ALLCAPSNAME": pa.array([0, 1]), "other": pa.array([2, 3])})
    _ = lance.write_dataset(table, tmp_path / "test_uppercase_name")

    dataset = lance.dataset(tmp_path / "test_uppercase_name")
    result = dataset.to_table(filter="`ALLCAPSNAME` == 0").to_pandas()
    pd.testing.assert_frame_equal(
        result, pd.DataFrame([{"ALLCAPSNAME": 0, "other": 2}])
    )


def test_negative_expressions(tmp_path: Path):
    table = pa.table({"x": [-1, 0, 1, 1], "y": [1, 2, 3, 4]})
    dataset = lance.write_dataset(table, tmp_path / "test_neg_expr")
    filters_expected = [
        ("x = -1", [-1]),
        ("x > -1", [0, 1, 1]),
        ("x = 1 * -1", [-1]),
        ("x <= 2 + -2 ", [-1, 0]),
        ("x = y - 2", [-1, 0, 1]),
    ]
    for filter, expected in filters_expected:
        assert dataset.scanner(filter=filter).to_table()["x"].to_pylist() == expected


def create_table_for_duckdb(nvec=10000, ndim=768):
    mat = np.random.randn(nvec, ndim)
    price = (np.random.rand(nvec) + 1) * 100

    def gen_str(n):
        return "".join(random.choices("abc"))

    meta = np.array([gen_str(1) for _ in range(nvec)])
    tbl = (
        vec_to_table(data=mat)
        .append_column("price", pa.array(price))
        .append_column("meta", pa.array(meta))
        .append_column("id", pa.array(range(nvec)))
    )
    return tbl


def test_duckdb(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    tbl = create_table_for_duckdb()
    ds = lance.write_dataset(tbl, str(tmp_path))  # noqa: F841

    actual = duckdb.query("SELECT id, meta, price FROM ds WHERE id==1000").to_df()
    expected = duckdb.query("SELECT id, meta, price FROM ds").to_df()
    expected = expected[expected.id == 1000].reset_index(drop=True)
    tm.assert_frame_equal(actual, expected)

    actual = duckdb.query("SELECT id, meta, price FROM ds WHERE id=1000").to_df()
    expected = duckdb.query("SELECT id, meta, price FROM ds").to_df()
    expected = expected[expected.id == 1000].reset_index(drop=True)
    tm.assert_frame_equal(actual, expected)

    actual = duckdb.query(
        "SELECT id, meta, price FROM ds WHERE price>20.0 and price<=90"
    ).to_df()
    expected = duckdb.query("SELECT id, meta, price FROM ds").to_df()
    expected = expected[(expected.price > 20.0) & (expected.price <= 90)].reset_index(
        drop=True
    )
    tm.assert_frame_equal(actual, expected)

    actual = duckdb.query("SELECT id, meta, price FROM ds WHERE meta=='aa'").to_df()
    expected = duckdb.query("SELECT id, meta, price FROM ds").to_df()
    expected = expected[expected.meta == "aa"].reset_index(drop=True)
    tm.assert_frame_equal(actual, expected)
