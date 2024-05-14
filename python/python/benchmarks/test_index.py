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
@pytest.mark.parametrize("num_partitions", [512])
@pytest.mark.parametrize("ef_construction", [150, 300])
@pytest.mark.parametrize("m", [16])
@pytest.mark.parametrize("num_small_indexes", [1, 5])
@pytest.mark.parametrize("num_new_rows", [500])
def test_optimize_index(test_large_dataset, benchmark, num_partitions, ef_construction, m, num_small_indexes, num_new_rows):
    # insert smaller batch(es) into the large dataset,
    # then benchmark the optimize_index method
    m_max = m * 2
    test_large_dataset = test_large_dataset.create_index(
        column="vector",
        index_type="IVF_HNSW_SQ",
        metric_type="L2",
        num_partitions=num_partitions,
        ef_construction=ef_construction,
        m=m,
        m_max=m_max,
        num_bits=8,
        replace=True,
    )

    for _ in range(num_small_indexes):
        small_table = gen_table(num_new_rows // num_small_indexes)
        lance.write_dataset(small_table, test_large_dataset.uri, mode="append")

    benchmark(test_large_dataset.optimize.optimize_indices)
