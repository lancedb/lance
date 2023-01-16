#!/usr/bin/env python3
#

import duckdb
import lance
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.testing import assert_series_equal


def test_l2_distance(db: duckdb.DuckDBPyConnection):
    """GH-7"""
    embeddings = np.random.randn(10, 10)
    tbl = pa.Table.from_arrays([embeddings.tolist()], names=["embedding"])

    df = db.query("""
    SELECT l2_distance(embedding,
        [0.14132072948046223,-0.8578304618530145,1.03418279173152,-0.01988450184766287,0.20275013403601405,
        1.1907599349042708, 1.592254025308326, -0.5606235353210591, -1.353943627981242, 0.10803636704591536]) AS score
    FROM tbl
    """).to_df()

    v1 = np.array(
        [0.14132072948046223, -0.8578304618530145, 1.03418279173152, -0.01988450184766287, 0.20275013403601405,
         1.1907599349042708, 1.592254025308326, -0.5606235353210591, -1.353943627981242, 0.10803636704591536])
    expected = pd.Series(((embeddings - v1) ** 2).sum(axis=1))
    assert np.allclose(df.score.to_numpy(), expected)

    df = db.query("""SELECT l2_distance([1, 2], [1, 2]) as score""").to_df()
    assert_series_equal(df.score, pd.Series([0], name="score", dtype='int32'))


def test_in_rectangle(db: duckdb.DuckDBPyConnection):
    tbl = pa.Table.from_pylist([{"box": [[1, 2], [3, 4]]}, {"box": [[10, 20], [30, 45]]}])
    df = db.query("""SELECT in_rectangle([15, 35], box) AS contain FROM tbl""").to_df()
    assert_series_equal(df.contain, pd.Series([False, True]), check_names=False)

    tbl = pa.Table.from_pylist([{"point": [1, 2]}, {"point": [10, 20]}])
    df = db.query("""SELECT in_rectangle(point, [[5, 10], [30, 40]]) AS contain FROM tbl""").to_df()
    assert_series_equal(df.contain, pd.Series([False, True]), check_names=False)

    df = db.query(
        """SELECT in_rectangle([15.0, 35.5], [[5.5, 10.1], [30.3, 40.4]]) as contain""").to_df()
    assert_series_equal(df.contain, pd.Series([True]), check_names=False)


def test_list_argmax(db: duckdb.DuckDBPyConnection):
    for dtype in ["INT", "BIGINT", "FLOAT", "DOUBLE"]:
        df = db.query(f"""SELECT list_argmax([1, 2, 3, 2, 1]::{dtype}[]) as idx""").to_df()
        assert_series_equal(df.idx, pd.Series([2], name='idx', dtype='int32'))

def test_derivative(db: duckdb.DuckDBPyConnection):
    tbl = pa.Table.from_pylist([{"x": i * 0.2, "y": i * 1} for i in range(5)])
    df = db.query("SELECT dydx(y, x) as d FROM tbl").to_df()
    assert_series_equal(df.d, pd.Series([None, 5, 5, 5, 5]))
