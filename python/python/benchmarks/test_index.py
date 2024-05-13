# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pytest

N_DIMS = 512


def gen_table(num_rows):
    values = pc.random(num_rows * N_DIMS).cast(pa.float32())
    vectors = pa.FixedSizeListArray.from_arrays(values, N_DIMS)
    table = pa.table({"vector": vectors})

    return table


def gen_dataset(tmpdir_factory, num_rows):
    tmp_path = Path(tmpdir_factory.mktemp("index_dataset"))
    table = gen_table(num_rows)
    dataset = lance.write_dataset(table, tmp_path)

    return dataset


@pytest.fixture(scope="module")
def test_dataset(tmpdir_factory):
    # We are writing to this, so it's not beneficial to cache it in the data_dir.
    return gen_dataset(tmpdir_factory, 1_000)


@pytest.fixture(scope="module")
def test_large_dataset(tmpdir_factory):
    # We are writing to this, so it's not beneficial to cache it in the data_dir.
    return gen_dataset(tmpdir_factory, 1_000_000)


@pytest.mark.benchmark(group="create_index")
def test_create_ivf_pq(test_dataset, benchmark):
    benchmark(
        test_dataset.create_index,
        column="vector",
        index_type="IVF_PQ",
        metric_type="L2",
        num_partitions=8,
        num_sub_vectors=2,
        num_bits=8,
        replace=True,
    )


@pytest.mark.benchmark(group="create_index")
@pytest.mark.cuda
def test_create_ivf_pq_cuda(test_dataset, benchmark):
    benchmark(
        test_dataset.create_index,
        column="vector",
        index_type="IVF_PQ",
        metric_type="L2",
        num_partitions=8,
        num_sub_vectors=2,
        num_bits=8,
        accelerator="cuda",
        replace=True,
    )


@pytest.mark.benchmark(group="optimize_index")
def test_optimize_index(test_large_dataset, benchmark):
    # insert a smaller batch into the large dataset,
    # then benchmark the optimize_index method
    small_table = gen_table(test_large_dataset.count_rows() * 30 // 100)
    lance.write_dataset(small_table, test_large_dataset.uri, mode="append")

    benchmark(test_large_dataset.optimize.optimize_indices)
