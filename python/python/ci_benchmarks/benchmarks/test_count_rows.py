# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Benchmark count_rows acceleration for IS NULL / IS NOT NULL filters.

Tests five index configurations against a 1-billion-row int32 dataset with
~1% NULL values.  Each configuration stores the same data in a separate column
so that only one index type is active per measurement:

  none        — no index, full column scan (baseline)
  BITMAP      — bitmap index
  BTREE       — btree index
  ZONEMAP     — zone-map index
  BLOOMFILTER — bloom-filter index

Two filters are exercised for each configuration:
  IS NULL     — count the ~10 M null rows
  IS NOT NULL — count the ~990 M non-null rows

The goal is to show that scalar-index-accelerated count_rows is orders of
magnitude faster than the full-scan baseline regardless of which index type is
in use.
"""

from __future__ import annotations

import lance
import pytest
from ci_benchmarks.datasets import get_dataset_uri

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# (pytest id, column name that carries this index configuration)
_INDEX_CONFIGS: list[tuple[str, str]] = [
    ("none", "value_none"),
    ("bitmap", "value_bitmap"),
    ("btree", "value_btree"),
    ("zonemap", "value_zonemap"),
    ("bloomfilter", "value_bloomfilter"),
]
_INDEX_IDS = [cfg[0] for cfg in _INDEX_CONFIGS]

_FILTERS = ["is_null", "is_not_null"]


# ---------------------------------------------------------------------------
# Dataset fixture (module-scoped so the dataset is opened once per session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def count_rows_ds() -> lance.LanceDataset:
    return lance.dataset(get_dataset_uri("count_rows"))


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filter_type", _FILTERS)
@pytest.mark.parametrize("index_id,column", _INDEX_CONFIGS, ids=_INDEX_IDS)
def test_count_rows_null_filter(
    benchmark,
    count_rows_ds: lance.LanceDataset,
    index_id: str,
    column: str,
    filter_type: str,
) -> None:
    """Benchmark count_rows with IS NULL / IS NOT NULL under each index type.

    Args:
        index_id: Human-readable index name (used only for parametrize labels).
        column: The dataset column that carries the matching index.
        filter_type: ``"is_null"`` or ``"is_not_null"``.
    """
    filt = f"{column} IS NULL" if filter_type == "is_null" else f"{column} IS NOT NULL"

    def bench() -> int:
        return count_rows_ds.count_rows(filter=filt)

    benchmark.pedantic(bench, warmup_rounds=1, rounds=5)
