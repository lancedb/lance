# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import multiprocessing as mp
import os
import random
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import lance
import pytest
from ci_benchmarks.datasets import is_on_google, open_dataset

# POSIX fadvise flag to drop page cache
POSIX_FADV_DONTNEED = 4

DATASETS = ["tpch", "tpch-2.1", "mem-tpch", "mem-tpch-2.1"]


def drop_cache(ds: lance.LanceDataset):
    """Drop page cache for all files in the dataset using posix_fadvise.

    This only works for file-based datasets (not memory://).
    """
    # Skip cache dropping for in-memory datasets
    parsed = urlparse(ds.uri)
    if parsed.scheme == "memory":
        return

    # Get all data files from all fragments
    for fragment in ds.get_fragments():
        for data_file in fragment.data_files():
            file_path = data_file.path

            # Convert file:// URIs to local paths
            if file_path.startswith("file://"):
                file_path = urlparse(file_path).path

            # Only process if it's a local file that exists
            if os.path.exists(file_path):
                try:
                    with open(file_path, "rb") as f:
                        os.posix_fadvise(f.fileno(), 0, 0, POSIX_FADV_DONTNEED)
                except (OSError, AttributeError):
                    # posix_fadvise might not be available on all systems
                    pass


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("rows_per_take", [1, 10, 100])
def test_simple_random_access(benchmark, dataset, rows_per_take):
    ds = open_dataset(dataset)
    num_rows = ds.count_rows()

    def bench(indices):
        return ds.take(indices)

    def setup():
        indices = random.sample(range(num_rows), rows_per_take)
        return [indices], {}

    drop_cache(ds)
    benchmark.pedantic(bench, rounds=100, setup=setup, warmup_rounds=1)


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("rows_per_take", [1, 10, 100])
@pytest.mark.skipif(is_on_google(), reason="Requires too many IOPS for cloud storage")
def test_parallel_random_access(benchmark, dataset, rows_per_take):
    TAKES_PER_ITER = 100

    ds = open_dataset(dataset)
    num_rows = ds.count_rows()

    def bench(indices):
        futures = []
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            for i in range(TAKES_PER_ITER):
                iter_indices = indices[i * rows_per_take : (i + 1) * rows_per_take]
                futures.append(executor.submit(ds.take, iter_indices))
        for future in futures:
            future.result()

    def setup():
        indices = random.sample(range(num_rows), rows_per_take * TAKES_PER_ITER)
        return [indices], {}

    drop_cache(ds)
    benchmark.pedantic(bench, rounds=100, setup=setup, warmup_rounds=1)
