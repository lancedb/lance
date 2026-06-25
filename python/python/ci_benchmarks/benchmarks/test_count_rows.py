# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Benchmark count_rows acceleration for IS NULL / IS NOT NULL filters.

Tests five index configurations against an int32 dataset with ~1% NULL
values.  Each configuration stores the same data in a separate column so
that only one index type is active per measurement:

  none        — no index, full column scan (baseline)
  BITMAP      — bitmap index
  BTREE       — btree index
  ZONEMAP     — zone-map index
  BLOOMFILTER — bloom-filter index

Two filters are exercised for each configuration:
  IS NULL     — count the ~1% null rows
  IS NOT NULL — count the ~99% non-null rows

Indexed configurations are tested in two cache states to separate first-load
latency from steady-state throughput:

  warm — one prewarm call is made before measuring; the same dataset instance
         is reused so its in-memory index cache is already populated.
  cold — a fresh ``lance.dataset()`` instance is created inside each measured
         round so the in-memory index cache starts empty every time.  No
         prewarm pass is performed.
"""

from __future__ import annotations

import lance
import pytest
from ci_benchmarks.datasets import get_dataset_uri

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# Indexed configs only (warm/cold dimension applies to these)
_INDEXED_CONFIGS: list[tuple[str, str]] = [
    ("bitmap", "value_bitmap"),
    ("btree", "value_btree"),
    ("zonemap", "value_zonemap"),
    ("bloomfilter", "value_bloomfilter"),
]
_INDEXED_IDS = [cfg[0] for cfg in _INDEXED_CONFIGS]

_FILTERS = ["is_null", "is_not_null"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def count_rows_ds() -> lance.LanceDataset:
    """Shared dataset instance (index cache persists across rounds — use for warm)."""
    return lance.dataset(get_dataset_uri("count_rows"))


@pytest.fixture(scope="module")
def count_rows_uri() -> str:
    return get_dataset_uri("count_rows")


# ---------------------------------------------------------------------------
# No-index baseline (no warm/cold — there is no index cache to speak of)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filter_type", _FILTERS)
def test_count_rows_no_index(
    benchmark,
    count_rows_ds: lance.LanceDataset,
    filter_type: str,
) -> None:
    """Full-scan baseline with no scalar index."""
    filt = (
        "value_none IS NULL" if filter_type == "is_null" else "value_none IS NOT NULL"
    )

    def bench() -> int:
        return count_rows_ds.count_rows(filter=filt)

    benchmark.pedantic(bench, warmup_rounds=1, rounds=5)


# ---------------------------------------------------------------------------
# Indexed benchmarks — warm vs cold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("warm", [True, False], ids=["warm", "cold"])
@pytest.mark.parametrize("filter_type", _FILTERS)
@pytest.mark.parametrize("index_id,column", _INDEXED_CONFIGS, ids=_INDEXED_IDS)
def test_count_rows_indexed(
    benchmark,
    count_rows_ds: lance.LanceDataset,
    count_rows_uri: str,
    index_id: str,
    column: str,
    filter_type: str,
    warm: bool,
) -> None:
    """Benchmark count_rows with a scalar index, in warm and cold cache states.

    Args:
        index_id: Human-readable index name (parametrize label only).
        column: Dataset column that carries this index type.
        filter_type: ``"is_null"`` or ``"is_not_null"``.
        warm: If True, prewarm the index cache before measuring and reuse the
              shared dataset instance.  If False, create a fresh dataset
              instance on every round so the index cache starts empty.
    """
    filt = f"{column} IS NULL" if filter_type == "is_null" else f"{column} IS NOT NULL"

    if warm:

        def bench() -> int:
            return count_rows_ds.count_rows(filter=filt)

        # warmup_rounds=1 makes one unmeasured call that populates the cache.
        benchmark.pedantic(bench, warmup_rounds=1, rounds=5)
    else:

        def bench() -> int:
            # Fresh instance → empty in-memory index cache every round.
            ds = lance.dataset(count_rows_uri)
            return ds.count_rows(filter=filt)

        benchmark.pedantic(bench, warmup_rounds=0, rounds=5)
