# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""End-to-end tests for count-from-mask pushdown.

The optimizer rule under test (`CountPushdown`) rewrites
``SELECT COUNT(*) ... WHERE indexed_col <op> v`` into
``AggregateExec(Final) → CountFromMaskExec → ScalarIndexExec`` when the
index covers every dataset fragment, or splits into a Union of a pushdown
branch over the indexed fragments and a scan branch over the rest when
coverage is partial. This is category 1 (count-from-mask) of the four
aggregate-acceleration categories; the other three (mask-to-answer,
zone-aware, dimension-keyed) are not implemented yet.

Each test exercises a different state of the dataset (clean, with deletions,
with updates that introduce unindexed fragments, with a fully-deleted indexed
fragment) and asserts:

  1. The returned count matches the ground truth (correctness), and
  2. The plan routes through ``CountFromMaskExec`` (the rule fired).

For the cases where the index covers the whole dataset, the tests also assert
no ``LanceRead`` is present in the plan — proof that the count is being
answered from index metadata, not by scanning column data. The happy-path
test additionally re-runs the query and asserts the second call performs no
I/O.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import lance
import pyarrow as pa

if TYPE_CHECKING:
    from pathlib import Path


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

# 4 fragments × 25 rows = 100 rows; values 0..99 in `x`.
NUM_FRAGMENTS = 4
ROWS_PER_FRAGMENT = 25
NUM_ROWS = NUM_FRAGMENTS * ROWS_PER_FRAGMENT  # 100


def _make_dataset(tmp_path: Path) -> lance.LanceDataset:
    """Build a 4-fragment dataset with a BTREE index on `x`."""
    table = pa.table({"x": pa.array(range(NUM_ROWS), pa.int64())})
    dataset = lance.write_dataset(
        table,
        tmp_path / "ds",
        max_rows_per_file=ROWS_PER_FRAGMENT,
    )
    assert len(dataset.get_fragments()) == NUM_FRAGMENTS
    dataset.create_scalar_index("x", "BTREE")
    return dataset


def _filtered_count_plan(dataset: lance.LanceDataset, filter: str) -> str:
    """Return the ``analyze_plan(count_rows=True)`` output for a filtered
    ``COUNT(*)`` — the same plan ``count_rows(filter=…)`` actually executes."""
    return dataset.scanner(columns=[], with_row_id=True, filter=filter).analyze_plan(
        count_rows=True
    )


def _assert_pushdown_fired(plan: str) -> None:
    assert "CountFromMask" in plan, f"expected CountFromMaskExec in plan, got:\n{plan}"


def _assert_no_column_scan(plan: str) -> None:
    """Stricter: no LanceRead anywhere. Only applies when the index covers
    every dataset fragment (no partial-coverage split branch)."""
    assert "LanceRead" not in plan, (
        f"unexpected LanceRead in plan — column data was scanned:\n{plan}"
    )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_filtered_count_with_scalar_index(tmp_path: Path):
    """Happy path: filtered count on an indexed column, run twice.

    The second call must perform zero I/O — proof the rule routed the count
    through the index/deletion-mask metadata both times and the second call
    re-used the cache. The check uses ``dataset.io_stats_incremental()``
    rather than parsing the plan's ``bytes_read=…`` so we get a direct
    accounting of every object-store read the dataset performed during the
    second call, not just what the plan happens to surface.
    """
    dataset = _make_dataset(tmp_path)
    filter = "x < 50"
    expected = 50

    # Verify the rule fires for this shape.
    _assert_pushdown_fired(_filtered_count_plan(dataset, filter))
    _assert_no_column_scan(_filtered_count_plan(dataset, filter))

    # First call warms the index + deletion-mask caches.
    assert dataset.count_rows(filter=filter) == expected
    # Reset counters so the next snapshot only reflects the second call.
    dataset.io_stats_incremental()

    # Second call: must do zero I/O.
    assert dataset.count_rows(filter=filter) == expected
    stats = dataset.io_stats_incremental()
    assert stats.read_iops == 0, f"expected 0 read_iops, got {stats.read_iops}"
    assert stats.read_bytes == 0, f"expected 0 read_bytes, got {stats.read_bytes}"


def test_filtered_count_with_deleted_rows(tmp_path: Path):
    """Some matching rows are deleted — the count must reflect the deletions.

    Deletions don't change fragment coverage, so the index still covers every
    dataset fragment and the rule emits a single pushdown branch (no scan).
    """
    dataset = _make_dataset(tmp_path)
    # Delete three rows that match the filter (x < 50).
    dataset.delete("x = 10 OR x = 20 OR x = 30")
    plan = _filtered_count_plan(dataset, "x < 50")
    _assert_pushdown_fired(plan)
    _assert_no_column_scan(plan)
    assert dataset.count_rows(filter="x < 50") == 50 - 3


def test_filtered_count_with_updated_rows(tmp_path: Path):
    """Updates move rows in/out of the filter set.

    Before:  x < 50 ⇒ 50 rows match (values 0..49).
    After:
      - x = 5 → x = 100        (one row leaves the matched set)
      - x = 7 → x = 101        (another row leaves)
      - x = 60 → x = 8         (a row joins the matched set)
      - x = 70 → x = 9         (another joins)

    Net change: −2 + 2 = 0, so the final count is still 50, but the
    underlying row identities have shifted. Each update is materialized as
    a delete + insert into a new fragment in Lance — the new fragments are
    not in the index's coverage, so the optimizer rule emits a split plan:
    pushdown for the originally-indexed fragments, plus a scan branch for
    the rewritten fragments. The final count must still be correct.
    """
    dataset = _make_dataset(tmp_path)
    dataset.update({"x": "100"}, where="x = 5")
    dataset.update({"x": "101"}, where="x = 7")
    dataset.update({"x": "8"}, where="x = 60")
    dataset.update({"x": "9"}, where="x = 70")

    plan = _filtered_count_plan(dataset, "x < 50")
    _assert_pushdown_fired(plan)
    # 50 originally matching − 2 that left + 2 that joined = 50.
    assert dataset.count_rows(filter="x < 50") == 50


def test_filtered_count_with_whole_fragment_deleted(tmp_path: Path):
    """Delete every row in one indexed fragment.

    Fragment 0 covers x ∈ [0, 25). Deleting all of those rows removes 25
    matches of `x < 50`, dropping the count from 50 to 25.

    Lance retires the now-empty fragment, so the dataset has 3 fragments
    while the index still claims 4 — the index is a strict *superset* of
    the dataset, which is safe (the extra index entries simply don't
    apply). The rule emits a single pushdown branch (no scan needed).
    """
    dataset = _make_dataset(tmp_path)
    dataset.delete("x < 25")
    plan = _filtered_count_plan(dataset, "x < 50")
    _assert_pushdown_fired(plan)
    _assert_no_column_scan(plan)
    assert dataset.count_rows(filter="x < 50") == 25
