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


BTREE_FILTERS = [
    None,
    "image_widths = 3997",
    "image_widths >= 3990 AND image_widths <= 3997",
    "image_widths != 3997",
]
BTREE_FILTER_LABELS = [
    None,
    "equal",
    "small_range",
    "not_equal",
]


@pytest.mark.parametrize("filt", BTREE_FILTERS, ids=BTREE_FILTER_LABELS)
@pytest.mark.parametrize("payload", [None, "image_widths"], ids=["none", "integers"])
def test_eda_btree_search(benchmark, filt: str | None, payload: str | None):
    dataset_uri = get_dataset_uri("image_eda")
    ds = lance.dataset(dataset_uri)

    batch_size = 1024 if payload == "strings" else 32 * 1024

    columns = []
    if payload is not None:
        columns = [payload]

    def bench():
        ds.to_table(
            columns=columns,
            filter=filt,
            with_row_id=True,
            batch_size=batch_size,
        )

    iterations = 1
    if payload is None and filt != "image_widths != 3997":
        # These are fast queries against the index with no data load so we can
        # run them a lot more times
        iterations = 100

    # We warmup so we can test hot index performance
    benchmark.pedantic(bench, warmup_rounds=1, rounds=1, iterations=iterations)


BASIC_BTREE_FILTERS = [
    None,
    "row_number = 100000",
    "row_number != 100000",
    "row_number >= 100000 AND row_number <= 100007",
]

BASIC_BTREE_FILTER_LABELS = [
    "none",
    "equal",
    "not_equal",
    "small_range",
]


# Repeats the same test for the basic dataset which is easier to test with locally
# This benchmark is not part of the CI job as the EDA dataset is better for that
@pytest.mark.parametrize("filt", BASIC_BTREE_FILTERS, ids=BASIC_BTREE_FILTER_LABELS)
@pytest.mark.parametrize("payload", [None, "small_strings", "integers"])
def test_basic_btree_search(benchmark, filt: str | None, payload: str | None):
    dataset_uri = get_dataset_uri("basic")
    ds = lance.dataset(dataset_uri)

    columns = []
    if payload is not None:
        columns = [payload]

    def bench():
        ds.to_table(
            columns=columns,
            filter=filt,
            with_row_id=True,
            batch_size=32 * 1024,
        )

    benchmark.pedantic(bench, warmup_rounds=1, rounds=1, iterations=100)
