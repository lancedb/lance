# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Benchmark #3: impact of overlay files on take and scan workloads.

On read, every overlay covering a requested cell must be consulted and its value
merged over the base. This measures how take and full-scan cost scale with:
- the number of overlay layers stacked on a fragment (compaction payoff),
- coverage fraction (how many cells are overlaid),
- fragmentation (contiguous run vs. strided -> pages touched), and
- value width (a 4-byte int32 vs. a wide fixed-size-list embedding).

Wall time is measured warm via pytest-benchmark; read IO (bytes + IOPS) is
measured once cold, after dropping the page cache, via ``io_stats_incremental``.
"""

import random

import lance
import pytest
from ci_benchmarks.overlays import commit_overlay_layers, make_base_dataset
from ci_benchmarks.utils import wipe_os_cache

NUM_ROWS = 1_000_000
ROWS_PER_FILE = NUM_ROWS  # single fragment: isolates overlay-layer scaling
TAKE_ROWS = 100

# Wide value column: a 3072-d float32 embedding is 12 KiB/row, ~750x an int32
# cell. Fewer rows keep the base file to ~1.2 GiB while each cell still dominates
# read cost, so the merge/interleave a scan pays per overlay layer moves real
# payload rather than 4-byte integers.
WIDE_EMBEDDING_DIM = 3072
NUM_ROWS_WIDE = 100_000


def _take_indices(num_rows: int) -> list:
    rng = random.Random(0)
    return sorted(rng.sample(range(num_rows), TAKE_ROWS))


def _covered_value(ds: lance.LanceDataset):
    """``val`` at offset 0, which every coverage pattern includes.

    Used to guard the fixture: after committing overlays a covered cell must
    read back a new value, otherwise a read-invisible overlay would let the
    benchmark silently time plain base reads.
    """
    return ds.take([0], columns=["val"]).column("val").to_pylist()[0]


def _measure_cold_io(ds: lance.LanceDataset, base: str, work):
    """Drop the page cache, run ``work`` once, return its read IO stats."""
    wipe_os_cache(base)
    ds.io_stats_incremental()  # reset
    work()
    stats = ds.io_stats_incremental()
    return stats.read_bytes, stats.read_iops


def _run_read(benchmark, record_property, base, ds, workload, num_rows):
    if workload == "take":
        indices = _take_indices(num_rows)

        def work():
            ds.take(indices, columns=["val"])
    else:

        def work():
            ds.to_table(columns=["val"])

    read_bytes, read_iops = _measure_cold_io(ds, base, work)
    record_property("cold_read_bytes", read_bytes)
    record_property("cold_read_iops", read_iops)

    benchmark(work)


@pytest.mark.parametrize("version", ["2.0", "2.1"])
@pytest.mark.parametrize("workload", ["take", "scan"])
@pytest.mark.parametrize("num_overlays", [0, 4, 16])
@pytest.mark.parametrize(
    "fraction,pattern",
    [(0.01, "contiguous"), (0.01, "stride")],
    ids=["1pct-contiguous", "1pct-stride"],
)
def test_overlay_read_scaling(
    benchmark,
    tmp_path,
    record_property,
    version,
    workload,
    num_overlays,
    fraction,
    pattern,
):
    base = str(tmp_path / "ds")
    ds = make_base_dataset(base, NUM_ROWS, ROWS_PER_FILE, "int32", version)
    if num_overlays:
        base_val = _covered_value(ds)
        ds = commit_overlay_layers(ds, num_overlays, fraction, pattern, "int32")
        assert _covered_value(ds) != base_val, "overlay not visible on read"
    _run_read(benchmark, record_property, base, ds, workload, NUM_ROWS)


# Mirror test_overlay_read_scaling but on a wide 3072-d embedding column, so the
# take/scan-vs-layers story can be read for a fat value column rather than a
# 4-byte one. Pinned to the 2.1 format (v2.0 layer scaling is already covered by
# the narrow scan above; the wide-column question is about per-layer payload).
@pytest.mark.parametrize("workload", ["take", "scan"])
@pytest.mark.parametrize("num_overlays", [0, 4, 16])
@pytest.mark.parametrize(
    "fraction,pattern",
    [(0.01, "contiguous"), (0.01, "stride")],
    ids=["1pct-contiguous", "1pct-stride"],
)
def test_overlay_read_wide(
    benchmark,
    tmp_path,
    record_property,
    workload,
    num_overlays,
    fraction,
    pattern,
):
    base = str(tmp_path / "ds")
    ds = make_base_dataset(
        base,
        NUM_ROWS_WIDE,
        NUM_ROWS_WIDE,
        "embedding",
        "2.1",
        embedding_dim=WIDE_EMBEDDING_DIM,
    )
    if num_overlays:
        base_val = _covered_value(ds)
        ds = commit_overlay_layers(
            ds,
            num_overlays,
            fraction,
            pattern,
            "embedding",
            embedding_dim=WIDE_EMBEDDING_DIM,
        )
        assert _covered_value(ds) != base_val, "overlay not visible on read"
    _run_read(benchmark, record_property, base, ds, workload, NUM_ROWS_WIDE)
