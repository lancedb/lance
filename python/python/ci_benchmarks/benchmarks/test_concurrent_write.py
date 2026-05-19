# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Concurrent write benchmark for Lance datasets.

Measures throughput and latency of concurrent append, delete, and update
operations under different commit strategies (Pessimistic vs Optimistic).

Usage:
    # Run with default Pessimistic strategy
    pytest python/ci_benchmarks/benchmarks/test_concurrent_write.py \
        --benchmark-only

    # Run with Optimistic strategy (set env var)
    LANCE_COMMIT_STRATEGY=optimistic \
        pytest python/ci_benchmarks/benchmarks/test_concurrent_write.py \
        --benchmark-only

    # Save results as JSON
    pytest python/ci_benchmarks/benchmarks/test_concurrent_write.py \
        --benchmark-only --benchmark-json results.json

    # Run against S3/TOS
    export AWS_ACCESS_KEY_ID=xxx
    export AWS_SECRET_ACCESS_KEY=xxx
    export AWS_ENDPOINT=https://your-endpoint
    export AWS_REGION=your-region
    export LANCE_BENCH_DATASET_URI=s3://bucket/path
    pytest python/ci_benchmarks/benchmarks/test_concurrent_write.py \
        --benchmark-only
"""

import asyncio
import os
import random
import tempfile
import time

import lance
import pyarrow as pa
import pytest


def _get_storage_options():
    key_id = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    endpoint = os.environ.get("AWS_ENDPOINT", "")
    region = os.environ.get("AWS_REGION", "")
    if key_id and secret and endpoint:
        return {
            "access_key_id": key_id,
            "secret_access_key": secret,
            "aws_endpoint": endpoint,
            "region": region,
            "virtual_hosted_style_request": "true",
        }
    return None


def _get_dataset_uri(label: str) -> str:
    uri = os.environ.get("LANCE_BENCH_DATASET_URI")
    if uri:
        return f"{uri}/concurrent_bench_{label}"
    return os.path.join(tempfile.mkdtemp(), f"concurrent_bench_{label}")


async def _run_concurrent_writes(
    dataset_uri: str,
    num_writers: int,
    num_deleters: int,
    num_updaters: int,
    num_writes_per_writer: int,
    num_deletes_per_deleter: int,
    num_updates_per_updater: int,
    rows_per_write: int,
    storage_options: dict | None,
):
    init_table = pa.table(
        {
            "id": pa.array([0], type=pa.int64()),
            "value": pa.array([0.0], type=pa.float64()),
        }
    )
    lance.write_dataset(
        init_table, dataset_uri, mode="create", storage_options=storage_options
    )

    results = []

    async def appender(writer_id):
        for i in range(num_writes_per_writer):
            table = pa.table(
                {
                    "id": pa.array(
                        [
                            writer_id * 100000 + i * rows_per_write + j
                            for j in range(rows_per_write)
                        ],
                        type=pa.int64(),
                    ),
                    "value": pa.array(
                        [
                            float(writer_id * 100 + i + j * 0.01)
                            for j in range(rows_per_write)
                        ],
                        type=pa.float64(),
                    ),
                }
            )
            start = time.monotonic()
            ds = lance.write_dataset(
                table, dataset_uri, mode="append", storage_options=storage_options
            )
            elapsed = time.monotonic() - start
            results.append(
                {"type": "append", "elapsed_s": elapsed, "version": ds.version}
            )

    async def deleter(deleter_id):
        for i in range(num_deletes_per_deleter):
            await asyncio.sleep(random.uniform(0.2, 0.8))
            ds = lance.dataset(dataset_uri, storage_options=storage_options)
            fragments = list(ds.get_fragments())
            if not fragments:
                continue
            target_frag = random.choice(fragments)
            frag_id = target_frag.fragment_id
            start = time.monotonic()
            try:
                ds.delete(f"id >= {frag_id * 100000} AND id < {(frag_id + 1) * 100000}")
                elapsed = time.monotonic() - start
                results.append(
                    {"type": "delete", "elapsed_s": elapsed, "version": ds.version}
                )
            except Exception:
                elapsed = time.monotonic() - start
                results.append({"type": "delete_error", "elapsed_s": elapsed})

    async def updater(updater_id):
        for i in range(num_updates_per_updater):
            await asyncio.sleep(random.uniform(0.2, 0.8))
            ds = lance.dataset(dataset_uri, storage_options=storage_options)
            fragments = list(ds.get_fragments())
            if not fragments:
                continue
            target_frag = random.choice(fragments)
            frag_id = target_frag.fragment_id
            start = time.monotonic()
            try:
                ds.update(
                    {"value": "value + 1.0"},
                    where=(
                        f"id >= {frag_id * 100000} AND id < {(frag_id + 1) * 100000}"
                    ),
                )
                elapsed = time.monotonic() - start
                results.append(
                    {"type": "update", "elapsed_s": elapsed, "version": ds.version}
                )
            except Exception:
                elapsed = time.monotonic() - start
                results.append({"type": "update_error", "elapsed_s": elapsed})

    start = time.monotonic()
    tasks = [appender(wid) for wid in range(num_writers)]
    tasks += [deleter(did) for did in range(num_deleters)]
    tasks += [updater(uid) for uid in range(num_updaters)]
    await asyncio.gather(*tasks)
    total_elapsed = time.monotonic() - start

    append_latencies = sorted(
        [r["elapsed_s"] for r in results if r["type"] == "append"]
    )
    delete_latencies = sorted(
        [r["elapsed_s"] for r in results if r["type"] == "delete"]
    )
    update_latencies = sorted(
        [r["elapsed_s"] for r in results if r["type"] == "update"]
    )
    all_latencies = sorted(append_latencies + delete_latencies + update_latencies)

    total_ops = len(all_latencies)
    throughput = total_ops / total_elapsed if total_elapsed > 0 else 0

    n_app = len(append_latencies)
    n_del = len(delete_latencies)
    n_upd = len(update_latencies)

    print(f"\n{'=' * 60}")
    print("Concurrent Write Benchmark Results")
    print(f"{'=' * 60}")
    print(
        f"  Total ops:     {total_ops} (append={n_app}, delete={n_del}, update={n_upd})"
    )
    print(f"  Total time:    {total_elapsed:.3f}s")
    print(f"  Throughput:    {throughput:.1f} ops/s")
    if all_latencies:
        avg = sum(all_latencies) / len(all_latencies)
        p50 = all_latencies[len(all_latencies) // 2]
        p90 = all_latencies[int(len(all_latencies) * 0.9)]
        p99 = all_latencies[int(len(all_latencies) * 0.99)]
        print(f"  Overall avg:   {avg:.4f}s")
        print(f"  Overall p50:   {p50:.4f}s")
        print(f"  Overall p90:   {p90:.4f}s")
        print(f"  Overall p99:   {p99:.4f}s")
    if append_latencies:
        avg = sum(append_latencies) / len(append_latencies)
        p50 = append_latencies[len(append_latencies) // 2]
        print(f"  Append avg:    {avg:.4f}s")
        print(f"  Append p50:    {p50:.4f}s")
    if delete_latencies:
        avg = sum(delete_latencies) / len(delete_latencies)
        p50 = delete_latencies[len(delete_latencies) // 2]
        print(f"  Delete avg:    {avg:.4f}s")
        print(f"  Delete p50:    {p50:.4f}s")
    if update_latencies:
        avg = sum(update_latencies) / len(update_latencies)
        p50 = update_latencies[len(update_latencies) // 2]
        print(f"  Update avg:    {avg:.4f}s")
        print(f"  Update p50:    {p50:.4f}s")
    print(f"{'=' * 60}")

    return throughput


def _run_benchmark(
    num_writers=20,
    num_deleters=10,
    num_updaters=10,
    num_writes_per_writer=10,
    num_deletes_per_deleter=5,
    num_updates_per_updater=5,
    rows_per_write=100,
):
    storage_options = _get_storage_options()
    label = f"w{num_writers}_d{num_deleters}_u{num_updaters}"
    dataset_uri = _get_dataset_uri(label)
    throughput = asyncio.run(
        _run_concurrent_writes(
            dataset_uri,
            num_writers,
            num_deleters,
            num_updaters,
            num_writes_per_writer,
            num_deletes_per_deleter,
            num_updates_per_updater,
            rows_per_write,
            storage_options,
        )
    )
    return throughput


@pytest.mark.benchmark(group="concurrent_write", warmup=False)
def test_concurrent_write_mixed(benchmark):
    benchmark.pedantic(
        _run_benchmark,
        kwargs=dict(
            num_writers=20,
            num_deleters=10,
            num_updaters=10,
            num_writes_per_writer=10,
            num_deletes_per_deleter=5,
            num_updates_per_updater=5,
            rows_per_write=100,
        ),
        rounds=1,
        iterations=1,
    )


@pytest.mark.benchmark(group="concurrent_write", warmup=False)
def test_concurrent_write_high_concurrency(benchmark):
    benchmark.pedantic(
        _run_benchmark,
        kwargs=dict(
            num_writers=30,
            num_deleters=15,
            num_updaters=15,
            num_writes_per_writer=10,
            num_deletes_per_deleter=5,
            num_updates_per_updater=5,
            rows_per_write=100,
        ),
        rounds=1,
        iterations=1,
    )


@pytest.mark.benchmark(group="concurrent_write", warmup=False)
def test_concurrent_write_append_only(benchmark):
    benchmark.pedantic(
        _run_benchmark,
        kwargs=dict(
            num_writers=30,
            num_deleters=0,
            num_updaters=0,
            num_writes_per_writer=10,
            num_deletes_per_deleter=0,
            num_updates_per_updater=0,
            rows_per_write=100,
        ),
        rounds=1,
        iterations=1,
    )
