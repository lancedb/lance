# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import random

import lance
import pyarrow as pa

sample_size = 10


def _gen_data():
    ds = lance.write_dataset(pa.table({"x": range(1000)}), "memory://test")
    return ds


def test_take_rowid_rowaddr():
    ds = _gen_data()
    total_rows = len(ds)
    sampled_indices = random.sample(range(total_rows), min(sample_size, total_rows))

    sample_dataset = ds.take(sampled_indices, columns=["_rowid"])
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 1

    sample_dataset = ds.take(sampled_indices, columns=["_rowid", "_rowid"])
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 2

    sample_dataset = ds.take([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], columns=["_rowid"])
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 1

    sample_dataset = ds.take(sampled_indices, columns=["_rowaddr"])
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 1

    sample_dataset = ds.take([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], columns=["_rowaddr"])
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 1

    sample_dataset = ds.take(sampled_indices, columns=["_rowaddr", "_rowid"])
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 2

    sample_dataset = ds.take(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 100], columns=["_rowaddr", "_rowid"]
    )
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 2

    sample_dataset = ds.take(sampled_indices, columns=["_rowid", "_rowaddr"])
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 2

    sample_dataset = ds.take(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 100], columns=["_rowid", "_rowaddr"]
    )
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 2


def test_take():
    ds = _gen_data()
    total_rows = len(ds)
    sampled_indices = random.sample(range(total_rows), min(sample_size, total_rows))

    sample_dataset = ds.take(sampled_indices, columns=["x"])
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 1

    sample_dataset = ds.take([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], columns=["x"])
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 1

    sample_dataset = ds.take(sampled_indices, columns=["x", "_rowid", "_rowaddr"])
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 3

    sample_dataset = ds.take(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 100], columns=["x", "_rowid", "_rowaddr"]
    )
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 3

    sample_dataset = ds.take(sampled_indices, columns=["_rowid", "_rowaddr", "x"])
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 3

    sample_dataset = ds.take(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 100], columns=["_rowid", "_rowaddr", "x"]
    )
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 3
