# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Benchmark #2: cost of updating a single column for 1% of rows.

Compares writing an overlay against the existing rewrite-based mechanisms
(``update``, ``merge_insert``, full overwrite).

The rewrite mechanisms are delete-and-reinsert: they re-encode the changed rows
*across all columns* into a new fragment and add a deletion vector to each
affected fragment. An overlay instead persists only the changed column's cells
plus a coverage bitmap in the manifest. The crossover therefore depends on row
width, so the benchmark sweeps a ``width`` axis:

- ``narrow``: id + val only. Rewriting whole rows is nearly free, so an overlay
  has no byte advantage and its manifest/bitmap overhead can make it larger.
- ``wide``: id + val + a 64-d float32 payload. Rewrites re-encode the payload for
  every changed row; the overlay still writes only ``val``.

Persisted bytes are split into data (under ``data/``) and metadata (manifests,
transactions, deletion vectors), because the two families spend their bytes
differently. ``read_bytes`` (process-wide read syscalls from /proc/self/io) and
wall time are reported alongside: the rewrite mechanisms must read the affected
rows to re-encode them, which both costs IO and explains much of their extra
time, while an overlay reads essentially nothing.

NOTE: the overlay arm is hand-rolled (no ``update``-via-overlay path exists yet)
and, unlike the other arms, does not read the base column first -- it writes
fresh values for the covered cells. That no-read is the win this read metric
exposes.
"""

from time import perf_counter

import lance
import numpy as np
import pyarrow as pa
import pytest
from ci_benchmarks.overlays import (
    commit_overlay_layers,
    file_sizes,
    make_base_dataset,
    proc_io_counters,
    written_breakdown,
)

NUM_ROWS = 1_000_000
ROWS_PER_FILE = 100_000  # 10 fragments; the strided selection hits every one
UPDATE_FRACTION = 0.01
STEP = int(1 / UPDATE_FRACTION)  # select id % STEP == 0
WIDE_DIM = 64

APPROACHES = [
    "update",
    "merge_insert",
    "merge_insert_indexed",
    "overlay",
    "full_rewrite",
]


def _selected_ids() -> np.ndarray:
    return np.arange(0, NUM_ROWS, STEP, dtype=np.int64)


def _payload(n: int, rng: np.random.Generator) -> pa.Array:
    flat = rng.random(n * WIDE_DIM, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat), WIDE_DIM)


def _new_rows(ids: np.ndarray, payload_dim: int) -> pa.Table:
    """Full replacement rows for ``merge_insert`` (a row-oriented op cannot
    target a single column, so it brings whole rows)."""
    rng = np.random.default_rng(1)
    cols = {
        "id": pa.array(ids),
        "val": pa.array(rng.integers(0, 1 << 30, len(ids), np.int32)),
    }
    if payload_dim:
        cols["payload"] = _payload(len(ids), rng)
    return pa.table(cols)


@pytest.mark.parametrize("width", ["narrow", "wide"])
@pytest.mark.parametrize("approach", APPROACHES)
def test_overlay_write_cost(tmp_path, record_property, approach, width):
    payload_dim = WIDE_DIM if width == "wide" else 0
    base = str(tmp_path / "ds")
    ds = make_base_dataset(
        base, NUM_ROWS, ROWS_PER_FILE, "int32", "2.1", payload_dim=payload_dim
    )
    ids = _selected_ids()
    sel, unsel = int(ids[0]), int(ids[0]) + 1
    before = {
        r["id"]: r["val"]
        for r in ds.to_table(
            columns=["id", "val"], filter=f"id IN ({sel},{unsel})"
        ).to_pylist()
    }

    if approach == "merge_insert_indexed":
        ds.create_scalar_index("id", "BTREE")  # setup, not measured

    def run():
        nonlocal ds
        if approach == "update":
            ds.update({"val": "val + 1"}, where=f"id % {STEP} = 0")
        elif approach in ("merge_insert", "merge_insert_indexed"):
            ds.merge_insert("id").when_matched_update_all().execute(
                _new_rows(ids, payload_dim)
            )
        elif approach == "overlay":
            ds = commit_overlay_layers(ds, base, 1, UPDATE_FRACTION, "stride", "int32")
        elif approach == "full_rewrite":
            table = ds.to_table()
            val = table.column("val").to_numpy(zero_copy_only=False).copy()
            val[ids] = _new_rows(ids, 0).column("val").to_numpy()
            cols = {"id": table.column("id"), "val": pa.array(val)}
            if payload_dim:
                cols["payload"] = table.column("payload")
            ds = lance.write_dataset(
                pa.table(cols), base, mode="overwrite", data_storage_version="2.1"
            )
        else:
            raise ValueError(approach)

    snap_before = file_sizes(base)
    io_before = proc_io_counters()
    start = perf_counter()
    run()
    elapsed = perf_counter() - start
    io_after = proc_io_counters()
    snap_after = file_sizes(base)

    # Guard against silently measuring a no-op: the selected row must change and
    # the unselected row must not.
    ds = lance.dataset(base)
    after = {
        r["id"]: r["val"]
        for r in ds.to_table(
            columns=["id", "val"], filter=f"id IN ({sel},{unsel})"
        ).to_pylist()
    }
    assert after[sel] != before[sel], f"{approach}/{width}: selected row unchanged"
    assert after[unsel] == before[unsel], f"{approach}/{width}: unselected row changed"

    data_bytes, meta_bytes = written_breakdown(snap_before, snap_after)
    # rchar/wchar (syscall bytes) are cache-independent; the base is warm from
    # the build, so physical read_bytes would under-report what was read.
    read_bytes = (
        io_after["rchar"] - io_before["rchar"] if io_before and io_after else -1
    )

    record_property("data_bytes", data_bytes)
    record_property("metadata_bytes", meta_bytes)
    record_property("persisted_bytes", data_bytes + meta_bytes)
    record_property("read_bytes", read_bytes)
    record_property("seconds", elapsed)
    print(
        f"\n{approach:<22} {width:<7} read={read_bytes:>11}B  "
        f"data={data_bytes:>10}B  meta={meta_bytes:>9}B  "
        f"persisted={data_bytes + meta_bytes:>10}B  time={elapsed:>7.3f}s"
    )
