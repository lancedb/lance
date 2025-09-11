# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import random

import lance
import pytest
from ci_benchmarks.datasets import get_dataset_uri

DATASETS = ["tpch"]


@pytest.mark.parametrize("dataset", DATASETS)
def test_random_access(benchmark, dataset):
    NUM_INDICES = 10
    dataset_uri = get_dataset_uri(dataset)

    ds = lance.dataset(dataset_uri)
    random_indices = [random.randint(0, ds.count_rows()) for _ in range(NUM_INDICES)]

    def bench(random_indices):
        ds.take(random_indices)

    benchmark.pedantic(bench, args=(random_indices,), rounds=5)
