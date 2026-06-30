# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Benchmark #3: impact of overlay files on take and scan workloads.

On read, every overlay covering a requested cell must be consulted and its value
merged over the base. This measures how take and full-scan cost scale with:
- the number of overlay layers stacked on a fragment (compaction payoff),
- coverage fraction (how many cells are overlaid),
- fragmentation (contiguous run vs. strided -> pages touched), and
- data type (int32 vs. a fixed-size-list embedding).

Wall time is measured warm via pytest-benchmark; read IO (bytes + IOPS) is
measured once cold, after dropping the page cache, via ``io_stats_incremental``.
"""

import random

import lance
import pytest
from ci_benchmarks.overlays import commit_overlay_layers, make_base_dataset
from ci_benchmarks.utils import wipe_os_cache

NUM_ROWS = 1_000_000
NUM_ROWS_EMBEDDING = 100_000
ROWS_PER_FILE = NUM_ROWS  # single fragment: isolates overlay-layer scaling
TAKE_ROWS = 100


def _take_indices(num_rows: int) -> list:
    rng = random.Random(0)
    return sorted(rng.sample(range(num_rows), TAKE_ROWS))


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
    print(f"\ncold read: {read_bytes}B over {read_iops} iops")

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
        ds = commit_overlay_layers(ds, base, num_overlays, fraction, pattern, "int32")
    _run_read(benchmark, record_property, base, ds, workload, NUM_ROWS)


@pytest.mark.parametrize("workload", ["take", "scan"])
@pytest.mark.parametrize("dtype", ["int32", "embedding"])
def test_overlay_read_dtype(benchmark, tmp_path, record_property, workload, dtype):
    num_rows = NUM_ROWS_EMBEDDING if dtype == "embedding" else NUM_ROWS
    base = str(tmp_path / "ds")
    ds = make_base_dataset(base, num_rows, num_rows, dtype, "2.1")
    ds = commit_overlay_layers(ds, base, 4, 0.01, "stride", dtype)
    _run_read(benchmark, record_property, base, ds, workload, num_rows)
