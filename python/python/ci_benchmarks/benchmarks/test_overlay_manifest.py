# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Benchmark #1: impact of overlay files on manifest size.

Each committed overlay adds a ``DataOverlayFile`` entry (a value-file pointer
plus a serialized coverage bitmap) to every overlaid fragment's metadata in the
manifest. This measures how the manifest grows with the number of overlays and
the coverage size, since manifests are read on every dataset open.
"""

import pytest
from ci_benchmarks.overlays import (
    commit_overlay_layers,
    make_base_dataset,
    manifest_size,
)

NUM_ROWS = 1_000_000
# Single fragment so every overlay lands on the same fragment's metadata.
ROWS_PER_FILE = NUM_ROWS


@pytest.mark.parametrize("num_overlays", [0, 1, 4, 16, 64])
@pytest.mark.parametrize(
    "fraction,pattern",
    [(0.01, "contiguous"), (0.01, "stride"), (0.1, "stride")],
    ids=["1pct-contiguous", "1pct-stride", "10pct-stride"],
)
def test_overlay_manifest_size(
    tmp_path, record_property, num_overlays, fraction, pattern
):
    base = str(tmp_path / "ds")
    ds = make_base_dataset(base, NUM_ROWS, ROWS_PER_FILE, "int32", "2.1")
    base_bytes = manifest_size(ds)

    if num_overlays:
        ds = commit_overlay_layers(ds, num_overlays, fraction, pattern, "int32")

    total = manifest_size(ds)
    growth = total - base_bytes
    # Guard the fixture: committed overlays must enlarge the manifest, else the
    # benchmark would report growth=0 for overlays that were never recorded.
    if num_overlays:
        assert growth > 0, "overlays did not grow the manifest"
    per_overlay = growth / num_overlays if num_overlays else 0

    record_property("manifest_bytes", total)
    record_property("manifest_growth_bytes", growth)
    record_property("bytes_per_overlay", per_overlay)
    print(
        f"\noverlays={num_overlays:>3} {fraction:>5.0%} {pattern:<10} "
        f"manifest={total:>8}B  growth={growth:>8}B  "
        f"per_overlay={per_overlay:>8.0f}B"
    )
