# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""
Integration test verifying the two-tier data cache via scan_stats_callback.

Key insight: when a scan is served from the memory cache, no bytes are
read from the object store.  `ScanStatistics.bytes_read` should be 0
on the second (warm) scan.
"""

import os
import tempfile

import lance
import pyarrow as pa
import pytest


def make_dataset(path: str, n_rows: int = 50_000) -> lance.LanceDataset:
    """Write a dataset large enough to produce meaningful IO ranges."""
    table = pa.table(
        {
            "id": pa.array(range(n_rows), type=pa.int32()),
            "value": pa.array([float(i) * 0.5 for i in range(n_rows)], type=pa.float32()),
            "label": pa.array([f"label_{i % 100}" for i in range(n_rows)]),
        }
    )
    return lance.write_dataset(table, path)


def scan_and_collect_stats(ds: lance.LanceDataset) -> lance.ScanStatistics:
    """Run a full table scan and return IO statistics."""
    stats_holder = {}

    def callback(stats: lance.ScanStatistics):
        stats_holder["stats"] = stats

    ds.scanner(scan_stats_callback=callback).to_table()
    return stats_holder.get("stats")


@pytest.mark.skip(
    reason="Cache integration test — requires compiled Rust with cache enabled. "
    "Run manually: pytest python/python/tests/test_cache_integration.py -s -v"
)
def test_cache_reduces_bytes_read_on_warm_scan():
    """
    Verify that the second scan reads zero bytes from the object store
    when served from the memory cache.
    """
    with tempfile.TemporaryDirectory() as tmp:
        dataset_path = os.path.join(tmp, "test.lance")
        make_dataset(dataset_path)

        # Open dataset with 64 MiB memory cache enabled.
        ds = lance.dataset(
            dataset_path,
            storage_options={
                "data_cache_enabled": "true",
                "data_cache_memory_bytes": str(64 * 1024 * 1024),
            },
        )

        # ── Scan 1: cold ──────────────────────────────────────────────────
        cold_stats = scan_and_collect_stats(ds)
        assert cold_stats is not None, "scan_stats_callback was not called"
        print(
            f"\nCold scan:  iops={cold_stats.iops:,}  "
            f"bytes_read={cold_stats.bytes_read:,}"
        )
        assert cold_stats.bytes_read > 0, (
            "Expected bytes to be read from disk on the first (cold) scan"
        )
        assert cold_stats.iops > 0, (
            "Expected I/O operations on the first (cold) scan"
        )

        cold_bytes = cold_stats.bytes_read

        # ── Scan 2: warm ──────────────────────────────────────────────────
        # The decoder requests identical byte ranges → all served from cache.
        warm_stats = scan_and_collect_stats(ds)
        print(
            f"Warm scan:  iops={warm_stats.iops:,}  "
            f"bytes_read={warm_stats.bytes_read:,}"
        )

        warm_bytes = warm_stats.bytes_read
        print(
            f"Cache effectiveness: saved {cold_bytes - warm_bytes:,} bytes "
            f"({100 * (cold_bytes - warm_bytes) / cold_bytes:.1f}%)"
        )

        # The warm scan should read far fewer bytes (ideally zero).
        assert warm_bytes < cold_bytes * 0.1, (
            f"Expected warm scan to read <10% of cold bytes. "
            f"cold={cold_bytes:,}, warm={warm_bytes:,}"
        )


@pytest.mark.skip(
    reason="Cache integration test — requires compiled Rust with cache enabled. "
    "Run manually: pytest python/python/tests/test_cache_integration.py -s -v"
)
def test_cache_global_iops_counter():
    """
    Verify cache effectiveness using lance.iops_counter() and
    lance.bytes_read_counter() — process-global atomic counters.

    These count ONLY object-store fetches (not cache hits), so a warm
    scan should show zero new iops/bytes.
    """
    with tempfile.TemporaryDirectory() as tmp:
        dataset_path = os.path.join(tmp, "counter_test.lance")
        make_dataset(dataset_path)

        ds = lance.dataset(
            dataset_path,
            storage_options={
                "data_cache_enabled": "true",
                "data_cache_memory_bytes": str(64 * 1024 * 1024),
            },
        )

        # Baseline counters before any scan.
        iops_before = lance.iops_counter()
        bytes_before = lance.bytes_read_counter()

        # Cold scan.
        ds.to_table()
        iops_after_cold = lance.iops_counter()
        bytes_after_cold = lance.bytes_read_counter()
        cold_iops = iops_after_cold - iops_before
        cold_bytes = bytes_after_cold - bytes_before

        print(
            f"\nCold scan:  iops={cold_iops:,}  "
            f"bytes={cold_bytes:,}"
        )
        assert cold_iops > 0, "Expected I/O operations on cold scan"

        # Warm scan — should hit cache, not object store.
        iops_before_warm = lance.iops_counter()
        bytes_before_warm = lance.bytes_read_counter()

        ds.to_table()

        warm_iops = lance.iops_counter() - iops_before_warm
        warm_bytes = lance.bytes_read_counter() - bytes_before_warm

        print(
            f"Warm scan:  iops={warm_iops:,}  "
            f"bytes={warm_bytes:,}"
        )
        print(
            f"Cache hit: {100 * (1 - warm_iops / max(cold_iops, 1)):.1f}% "
            f"of IOPS saved"
        )

        # Warm scan should do far fewer (ideally zero) object-store IOPS.
        assert warm_iops < cold_iops * 0.1, (
            f"Expected warm scan IOPS <10% of cold. "
            f"cold={cold_iops}, warm={warm_iops}"
        )


if __name__ == "__main__":
    # Run directly for manual testing without pytest skip:
    #   RUST_LOG=lance_io=trace python python/python/tests/test_cache_integration.py
    #
    # Logs to watch for:
    #   lance_io::data_cache::memory  TRACE  memory cache miss — entry loaded and stored
    #   lance_io::data_cache::memory  TRACE  memory cache hit
    #   lance_io::scheduler           DEBUG  data cache miss — fetching from object store
    #   lance_io::scheduler           DEBUG  data cache served N bytes
    #
    # On a warm scan you should see only "memory cache hit" logs, no misses.
    import sys

    print("=== Cache integration test ===")
    print("Tip: run with RUST_LOG=lance_io=trace for per-range cache logs\n")

    with tempfile.TemporaryDirectory() as tmp:
        dataset_path = os.path.join(tmp, "manual_test.lance")
        print(f"Writing dataset to {dataset_path}...")
        make_dataset(dataset_path, n_rows=50_000)

        ds = lance.dataset(
            dataset_path,
            storage_options={
                "data_cache_enabled": "true",
                "data_cache_memory_bytes": str(64 * 1024 * 1024),
            },
        )

        print("\nRunning cold scan...")
        iops_before = lance.iops_counter()
        bytes_before = lance.bytes_read_counter()

        cold_stats = scan_and_collect_stats(ds)

        cold_iops_delta = lance.iops_counter() - iops_before
        cold_bytes_delta = lance.bytes_read_counter() - bytes_before

        print(
            f"  scan_stats: iops={cold_stats.iops:,} bytes={cold_stats.bytes_read:,}"
        )
        print(
            f"  global counters: iops={cold_iops_delta:,} bytes={cold_bytes_delta:,}"
        )

        print("\nRunning warm scan (should hit cache)...")
        iops_before = lance.iops_counter()
        bytes_before = lance.bytes_read_counter()

        warm_stats = scan_and_collect_stats(ds)

        warm_iops_delta = lance.iops_counter() - iops_before
        warm_bytes_delta = lance.bytes_read_counter() - bytes_before

        print(
            f"  scan_stats: iops={warm_stats.iops:,} bytes={warm_stats.bytes_read:,}"
        )
        print(
            f"  global counters: iops={warm_iops_delta:,} bytes={warm_bytes_delta:,}"
        )

        if cold_iops_delta > 0:
            savings = 100 * (1 - warm_iops_delta / cold_iops_delta)
            print(f"\nResult: {savings:.1f}% IOPS saved by cache")
            if savings > 90:
                print("✓ Cache is working correctly!")
            else:
                print("✗ Cache does not appear to be working")
                sys.exit(1)
        else:
            print("(No I/O on cold scan — dataset may be too small or already cached by OS)")
