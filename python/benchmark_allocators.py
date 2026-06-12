#!/usr/bin/env python3
"""
Benchmark allocator memory usage in pylance.
This script compares the RSS memory growth under different allocator configurations:
1. Default system allocator (glibc)
2. mimalloc (configured via #[global_allocator])
3. jemalloc (configured via #[global_allocator])

To run the comparison, build pylance with different features:
  - System Allocator: maturin develop --no-default-features
  - mimalloc:         maturin develop --features mimalloc
  - jemalloc:         maturin develop --no-default-features --features jemalloc

Usage:
  python benchmark_allocators.py --allocator [system|mimalloc|jemalloc]
"""

import argparse
import ctypes
import gc
import os
import sys
import uuid
import pyarrow as pa
import lance

def run_benchmark(allocator_name, iterations=50):
    uri = f"/tmp/lance_arena_repro_{uuid.uuid4().hex}"
    n_rows = 50_000

    print(f"Creating initial dataset at {uri}...")
    lance.write_dataset(
        pa.table({
            "uid": pa.array([str(uuid.uuid4()) for _ in range(n_rows)], type=pa.utf8()),
            "value": pa.array(["x"] * n_rows, type=pa.utf8()),
        }),
        uri,
        mode="overwrite",
        max_rows_per_file=10_000,
    )

    ds = lance.dataset(uri)
    uids = ds.scanner(columns=["uid"], limit=10_000).to_table().column("uid").to_pylist()
    key_field, val_field = ds.schema.field("uid"), ds.schema.field("value")

    # Load libc to get detailed malloc info if on Linux
    libc = None
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6")
            class Mallinfo2(ctypes.Structure):
                _fields_ = [("arena", ctypes.c_size_t)] * 2 + [("_pad", ctypes.c_size_t)] * 5 + \
                           [("uordblks", ctypes.c_size_t), ("fordblks", ctypes.c_size_t), ("_", ctypes.c_size_t)]
            libc.mallinfo2.restype = Mallinfo2
        except Exception:
            pass

    pid = os.getpid()
    print(f"Starting {iterations} iterations of merge_insert using allocator: {allocator_name}...")
    print(f"Iteration | RSS (Anon) | Glibc In Use | Glibc Free (Not Returned)")
    print("-" * 65)

    for i in range(1, iterations + 1):
        tbl = pa.table({
            "uid": pa.array(uids, type=key_field.type),
            "value": pa.array(["x"] * len(uids), type=val_field.type)
        })
        ds.merge_insert("uid").when_matched_update_all().execute(tbl)
        del tbl
        gc.collect()

        if i % 10 == 0 or i == iterations:
            rss_mb = 0
            if os.path.exists(f"/proc/{pid}/status"):
                with open(f"/proc/{pid}/status") as f:
                    for line in f:
                        if line.startswith("RssAnon:"):
                            rss_mb = int(line.split()[1]) // 1024
                            break
            
            glibc_in_use_mb = "N/A"
            glibc_free_mb = "N/A"
            if libc:
                try:
                    info = libc.mallinfo2()
                    glibc_in_use_mb = f"{info.uordblks // 1048576}MB"
                    glibc_free_mb = f"{info.fordblks // 1048576}MB"
                except Exception:
                    pass

            print(f"{i:9d} | {rss_mb:10d}MB | {glibc_in_use_mb:>12} | {glibc_free_mb:>25}")

    # Cleanup dataset
    try:
        import shutil
        shutil.rmtree(uri)
    except Exception:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark allocator memory usage.")
    parser.add_argument(
        "--allocator",
        choices=["system", "mimalloc", "jemalloc"],
        required=True,
        help="The allocator this pylance build was compiled with"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of merge_insert iterations to run"
    )
    args = parser.parse_args()
    run_benchmark(args.allocator, args.iterations)
