#!/usr/bin/env python3
#
#  Copyright (c) 2023. Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Concurrent mixed-operation benchmark for Lance on S3/TOS.
Measures throughput and latency of concurrent append + delete + update writes.

Delete and Update operations trigger TransactionRebase with initial_fragments IO,
which is the target of the caching optimization in conflict_resolver.rs.

Usage:
    export AWS_ACCESS_KEY_ID=xxx
    export AWS_SECRET_ACCESS_KEY=xxx
    export AWS_ENDPOINT=https://your-endpoint
    export AWS_REGION=your-region

    python benchmark.py <dataset_uri> \
        [--num-writers 20] [--num-deleters 10] [--num-updaters 10] \
        [--num-writes-per-writer 10] [--num-deletes-per-deleter 5] [--num-updates-per-updater 5] \
        [--rows-per-write 100] [--label LABEL]
"""
import argparse
import asyncio
import time
import os
import random

os.environ.setdefault("ORC_DEBUG", "0")

import lance
import pyarrow as pa


STORAGE_OPTIONS = {
    "access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", ""),
    "secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    "aws_endpoint": os.environ.get("AWS_ENDPOINT", ""),
    "region": os.environ.get("AWS_REGION", ""),
    "virtual_hosted_style_request": "true",
}


async def appender(dataset_uri: str, writer_id: int, num_writes: int, rows_per_write: int, results: list, storage_options: dict):
    for i in range(num_writes):
        table = pa.table({
            "id": pa.array([writer_id * 100000 + i * rows_per_write + j for j in range(rows_per_write)], type=pa.int64()),
            "value": pa.array([float(writer_id * 100 + i + j * 0.01) for j in range(rows_per_write)], type=pa.float64()),
        })
        start = time.monotonic()
        ds = lance.write_dataset(table, dataset_uri, mode="append", storage_options=storage_options)
        elapsed = time.monotonic() - start
        results.append({
            "type": "append",
            "worker_id": writer_id,
            "op_idx": i,
            "elapsed_s": elapsed,
            "version": ds.version,
        })


async def deleter(dataset_uri: str, deleter_id: int, num_deletes: int, results: list, storage_options: dict):
    for i in range(num_deletes):
        await asyncio.sleep(random.uniform(0.2, 0.8))
        ds = lance.dataset(dataset_uri, storage_options=storage_options)
        fragments = list(ds.get_fragments())
        if not fragments:
            results.append({
                "type": "delete_skip",
                "worker_id": deleter_id,
                "op_idx": i,
                "elapsed_s": 0.0,
                "version": ds.version,
            })
            continue
        target_frag = random.choice(fragments)
        frag_id = target_frag.fragment_id
        start = time.monotonic()
        try:
            ds.delete(f"id >= {frag_id * 100000} AND id < {(frag_id + 1) * 100000}")
            elapsed = time.monotonic() - start
            results.append({
                "type": "delete",
                "worker_id": deleter_id,
                "op_idx": i,
                "elapsed_s": elapsed,
                "version": ds.version,
            })
        except Exception as e:
            elapsed = time.monotonic() - start
            results.append({
                "type": "delete_error",
                "worker_id": deleter_id,
                "op_idx": i,
                "elapsed_s": elapsed,
                "error": str(e),
                "version": ds.version,
            })


async def updater(dataset_uri: str, updater_id: int, num_updates: int, results: list, storage_options: dict):
    for i in range(num_updates):
        await asyncio.sleep(random.uniform(0.2, 0.8))
        ds = lance.dataset(dataset_uri, storage_options=storage_options)
        fragments = list(ds.get_fragments())
        if not fragments:
            results.append({
                "type": "update_skip",
                "worker_id": updater_id,
                "op_idx": i,
                "elapsed_s": 0.0,
                "version": ds.version,
            })
            continue
        target_frag = random.choice(fragments)
        frag_id = target_frag.fragment_id
        start = time.monotonic()
        try:
            ds.update(
                {"value": "value + 1.0"},
                where=f"id >= {frag_id * 100000} AND id < {(frag_id + 1) * 100000}",
            )
            elapsed = time.monotonic() - start
            results.append({
                "type": "update",
                "worker_id": updater_id,
                "op_idx": i,
                "elapsed_s": elapsed,
                "version": ds.version,
            })
        except Exception as e:
            elapsed = time.monotonic() - start
            results.append({
                "type": "update_error",
                "worker_id": updater_id,
                "op_idx": i,
                "elapsed_s": elapsed,
                "error": str(e),
                "version": ds.version,
            })


def print_latency(label: str, latencies: list, total_elapsed: float):
    if not latencies:
        return
    print(f"\n--- {label} Latency ---")
    print(f"  count:  {len(latencies)}")
    print(f"  min:    {latencies[0]:.4f}s")
    print(f"  avg:    {sum(latencies)/len(latencies):.4f}s")
    print(f"  p50:    {latencies[len(latencies)//2]:.4f}s")
    print(f"  p90:    {latencies[int(len(latencies)*0.9)]:.4f}s")
    print(f"  p99:    {latencies[int(len(latencies)*0.99)]:.4f}s")
    print(f"  max:    {latencies[-1]:.4f}s")
    print(f"  throughput: {len(latencies) / total_elapsed:.1f} ops/s")


async def run_benchmark(dataset_uri: str, num_writers: int, num_deleters: int, num_updaters: int, num_writes_per_writer: int, num_deletes_per_deleter: int, num_updates_per_updater: int, rows_per_write: int, label: str, storage_options: dict):
    init_table = pa.table({
        "id": pa.array([0], type=pa.int64()),
        "value": pa.array([0.0], type=pa.float64()),
    })
    ds = lance.write_dataset(init_table, dataset_uri, mode="create", storage_options=storage_options)
    print(f"Created dataset at {dataset_uri}, version={ds.version}")

    results = []
    total_appends = num_writers * num_writes_per_writer
    total_deletes = num_deleters * num_deletes_per_deleter
    total_updates = num_updaters * num_updates_per_updater
    print(f"Starting benchmark: {num_writers} appenders x {num_writes_per_writer} + {num_deleters} deleters x {num_deletes_per_deleter} + {num_updaters} updaters x {num_updates_per_updater} = {total_appends} appends + {total_deletes} deletes + {total_updates} updates")

    start = time.monotonic()
    tasks = [
        appender(dataset_uri, wid, num_writes_per_writer, rows_per_write, results, storage_options)
        for wid in range(num_writers)
    ] + [
        deleter(dataset_uri, did, num_deletes_per_deleter, results, storage_options)
        for did in range(num_deleters)
    ] + [
        updater(dataset_uri, uid, num_updates_per_updater, results, storage_options)
        for uid in range(num_updaters)
    ]
    await asyncio.gather(*tasks)
    total_elapsed = time.monotonic() - start

    append_results = [r for r in results if r["type"] == "append"]
    delete_results = [r for r in results if r["type"] == "delete"]
    delete_errors = [r for r in results if r["type"] == "delete_error"]
    delete_skips = [r for r in results if r["type"] == "delete_skip"]
    update_results = [r for r in results if r["type"] == "update"]
    update_errors = [r for r in results if r["type"] == "update_error"]
    update_skips = [r for r in results if r["type"] == "update_skip"]

    ds = lance.dataset(dataset_uri, storage_options=storage_options)
    print(f"\n{'='*60}")
    print(f"Benchmark Results ({label})")
    print(f"{'='*60}")
    print(f"Total appends:      {len(append_results)}")
    print(f"Total deletes:      {len(delete_results)} (errors: {len(delete_errors)}, skips: {len(delete_skips)})")
    print(f"Total updates:      {len(update_results)} (errors: {len(update_errors)}, skips: {len(update_skips)})")
    print(f"Total time:          {total_elapsed:.3f}s")
    print(f"Final version:       {ds.version}")

    if append_results:
        print_latency("Append", sorted([r["elapsed_s"] for r in append_results]), total_elapsed)

    if delete_results:
        print_latency("Delete", sorted([r["elapsed_s"] for r in delete_results]), total_elapsed)

    if update_results:
        print_latency("Update", sorted([r["elapsed_s"] for r in update_results]), total_elapsed)

    all_ops = append_results + delete_results + update_results
    if all_ops:
        all_latencies = sorted([r["elapsed_s"] for r in all_ops])
        print_latency("Overall (append+delete+update)", all_latencies, total_elapsed)

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Lance concurrent mixed-operation benchmark")
    parser.add_argument("dataset_uri", help="Dataset URI (e.g. s3://bucket/path)")
    parser.add_argument("--num-writers", type=int, default=20)
    parser.add_argument("--num-deleters", type=int, default=10)
    parser.add_argument("--num-updaters", type=int, default=10)
    parser.add_argument("--num-writes-per-writer", type=int, default=10)
    parser.add_argument("--num-deletes-per-deleter", type=int, default=5)
    parser.add_argument("--num-updates-per-updater", type=int, default=5)
    parser.add_argument("--rows-per-write", type=int, default=100)
    parser.add_argument("--label", type=str, default="BENCHMARK")
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        args.dataset_uri, args.num_writers, args.num_deleters, args.num_updaters,
        args.num_writes_per_writer, args.num_deletes_per_deleter, args.num_updates_per_updater,
        args.rows_per_write, args.label, STORAGE_OPTIONS
    ))


if __name__ == "__main__":
    main()
