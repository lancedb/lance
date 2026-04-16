"""
Local cache integration test.

Uses eprintln debug output to verify the cache is active:
  [CACHE] submit_request_with_cache called  ← cache path is in use
  [CACHE MISS] ...                           ← cold scan fetching data
  [CACHE HIT]  ...                           ← warm scan served from cache

Run:
    python uber-benchmark/test_cache_local.py

Note: local files are affected by OS page cache so iops_counter()
may show improvement even without our cache. The [CACHE HIT] logs
are the reliable signal.
"""

import lance
import pyarrow as pa
import tempfile
import os
import time


def make_dataset(path: str, n_rows: int = 100_000) -> lance.LanceDataset:
    table = pa.table(
        {
            "id": pa.array(range(n_rows), type=pa.int32()),
            "value": pa.array(
                [float(i) * 0.5 for i in range(n_rows)], type=pa.float32()
            ),
            "label": pa.array([f"item_{i % 1000}" for i in range(n_rows)]),
        }
    )
    return lance.write_dataset(table, path)


def scan(ds, label: str, rows: int = 50_000):
    stats_holder = {}

    def cb(s):
        stats_holder["s"] = s

    iops_before = lance.iops_counter()
    bytes_before = lance.bytes_read_counter()
    t0 = time.time()

    ds.scanner(limit=rows, scan_stats_callback=cb).to_table()

    elapsed = time.time() - t0
    iops_delta = lance.iops_counter() - iops_before
    bytes_delta = lance.bytes_read_counter() - bytes_before
    s = stats_holder.get("s")

    print(f"\n{label}")
    print(f"  time:               {elapsed:.3f}s")
    print(f"  scan_stats.iops:    {s.iops if s else 'N/A'}")
    print(
        f"  scan_stats.bytes:   {s.bytes_read:,} B"
        if s
        else "  scan_stats.bytes:   N/A"
    )
    print(f"  global iops delta:  {iops_delta}")
    print(f"  global bytes delta: {bytes_delta:,} B")
    return {"elapsed": elapsed, "iops": iops_delta, "bytes": bytes_delta}


if __name__ == "__main__":
    print("=" * 60)
    print("Lance Cache Local Integration Test")
    print("=" * 60)
    print("Watch stderr for [CACHE] / [CACHE HIT] / [CACHE MISS] lines")
    print("These confirm the cache is in the hot path.\n")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.lance")
        print(f"Writing dataset to {path}...")
        make_dataset(path, n_rows=100_000)

        ds = lance.dataset(
            path,
            storage_options={
                "data_cache_enabled": "true",
                "data_cache_memory_bytes": str(1 * 1024 * 1024 * 1024),  # 1 GiB
            },
        )
        print(f"Opened: version={ds.version} fragments={len(ds.get_fragments())}")

        cold = scan(ds, "COLD SCAN (first read)")
        warm = scan(ds, "WARM SCAN (second read, should hit cache)")

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        if cold["elapsed"] > 0:
            speedup = cold["elapsed"] / max(warm["elapsed"], 0.001)
            print(f"  Speed improvement:  {speedup:.1f}x")

        if cold["iops"] > 0:
            saved = 100 * (1 - warm["iops"] / cold["iops"])
            print(
                f"  IOPS saved:         {saved:.1f}%  (cold={cold['iops']}, warm={warm['iops']})"
            )
            print()
            if warm["iops"] == 0:
                print("  ✓ Zero object-store IOPS on warm scan!")
            else:
                print("  Note: local files use OS page cache too — check")
                print("        [CACHE HIT] in stderr for true cache verification")

        print()
        print("Check stderr above for [CACHE HIT] lines on the warm scan.")
        print("If present → cache is working end-to-end.")
