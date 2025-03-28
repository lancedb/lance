# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import lance
import pytest
from ci_benchmarks.datasets import get_dataset_uri

COLUMN_LABELS = ["bools", "normals"]
COLUMNS = [["bools"], ["normals"]]
FILTERS = [None, "bools IS TRUE"]


@pytest.mark.parametrize("columns", COLUMNS, ids=COLUMN_LABELS)
@pytest.mark.parametrize("filt", FILTERS)
def test_eda_search(benchmark, columns, filt):
    dataset_uri = get_dataset_uri("image_eda")

    batch_size = 32 if columns == ["image_data"] else None
    limit = None if filter is None else 100000
    frag_readahead = (
        4
        if (columns == ["image_data"] or columns == ["strings"]) and filter is None
        else None
    )

    def bench():
        ds = lance.dataset(dataset_uri)
        ds.to_table(
            columns=columns,
            filter=filt,
            batch_size=batch_size,
            fragment_readahead=frag_readahead,
            limit=limit,
        )

    benchmark.pedantic(bench, rounds=1, iterations=1)


BTREE_FILTERS = ["image_widths = 3997", "image_widths >= 3990 AND image_widths <= 3997"]


@pytest.mark.parametrize("filt", BTREE_FILTERS)
def test_eda_btree_search(benchmark, filt):
    dataset_uri = get_dataset_uri("image_eda")
    ds = lance.dataset(dataset_uri)

    def bench():
        ds.to_table(
            columns=[],
            filter=filt,
            with_row_id=True,
        )

    # We warmup so we can test hot index performance
    benchmark.pedantic(bench, warmup_rounds=1, rounds=1, iterations=100)
