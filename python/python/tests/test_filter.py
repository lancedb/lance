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
        [pa.array(arr, type=pa.bool_())], names=["bool"]
    )

    def gen_str(n):
        return "".join(random.choices("abc"))

    stringcol = pa.array([gen_str(2) for _ in range(nrows)])

    tbl = pa.Table.from_arrays(
        [intcol, floatcol, structcol, stringcol], names=["int", "float", "rec", "str"]
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
        pc.field("float") < 90.0,
        pc.field("float") > 90.0,
        pc.field("float") <= 90.0,
        pc.field("float") >= 90.0,
        pc.field("str") != "aa",
        pc.field("str") == "aa",
    ]
    # test simple
    for expr in predicates:
        assert dataset.to_table(filter=expr) == dataset.to_table().filter(expr)


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


def test_match(tmp_path: Path):
    array = pa.array(["aaa", "bbb", "abc", "bca", "cab", "cba"])
    table = pa.Table.from_arrays([array], names=["str"])
    dataset = lance.write_dataset(table, tmp_path / "test_match")

    dataset = lance.dataset(tmp_path / "test_match")
    result = dataset.to_table(filter="str LIKE 'a%'").to_pandas()
    pd.testing.assert_frame_equal(result, pd.DataFrame({"str": ["aaa", "abc"]}))


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
    ds = lance.write_dataset(tbl, str(tmp_path))

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
