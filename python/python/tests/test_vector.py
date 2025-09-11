# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import numpy as np
import pyarrow as pa
import pytest
from lance.vector import vec_to_table


def test_dict():
    ids, vectors = _create_data()
    dd = dict(zip(ids, vectors))
    tbl = vec_to_table(dd)
    expected = [pa.array(ids), _to_vec(vectors)]
    assert_table(tbl, expected)

    new_tbl = vec_to_table(dd, names=["foo", "bar"])
    assert new_tbl.column_names == ["foo", "bar"]

    with pytest.raises(ValueError):
        ids, vectors = _create_bad_dims()
        dd = dict(zip(ids, vectors))
        vec_to_table(dd)


def test_list():
    _, vectors = _create_data()
    tbl = vec_to_table(vectors)
    expected = [_to_vec(vectors)]
    assert_table(tbl, expected)

    with pytest.raises(ValueError):
        _, vectors = _create_bad_dims()
        vec_to_table(vectors)


def test_ndarray():
    _, vectors = _create_data()
    tbl = vec_to_table(np.array(vectors))
    expected = [_to_vec(vectors)]
    assert_table(tbl, expected)

    with pytest.raises(ValueError):
        _, vectors = _create_bad_dims()
        vec_to_table(np.array(vectors))


def assert_table(tbl, expected_arrays, names=None):
    if names is None:
        if len(expected_arrays) == 1:
            names = ["vector"]
        else:
            names = ["id", "vector"]

    for i, n in enumerate(names):
        assert_array_eq(tbl[n], expected_arrays[i])


def assert_array_eq(left: pa.Array, right: pa.Array):
    if isinstance(left, pa.ChunkedArray):
        left = left.combine_chunks()
    if isinstance(right, pa.ChunkedArray):
        right = right.combine_chunks()
    if pa.types.is_float32(left.type):
        assert np.all(
            np.abs(
                left.to_numpy(zero_copy_only=False)
                - right.to_numpy(zero_copy_only=False)
            )
            < 1e-6
        )
    if pa.types.is_fixed_size_list(left.type):
        assert_array_eq(left.values, right.values)
    else:
        assert np.all(left.to_numpy(False) == right.to_numpy(False))


def _create_data():
    ids = list(range(10))
    vectors = np.random.randn(10, 8)
    return ids, vectors


def _create_bad_dims():
    ids = list(range(10))
    vectors = [np.random.randn(8) for _ in ids]
    vectors[5] = np.random.randn(5)
    return ids, vectors


def _to_vec(lst):
    return pa.FixedSizeListArray.from_arrays(
        pa.array(np.array(lst).ravel(), type=pa.float32()), list_size=8
    )
