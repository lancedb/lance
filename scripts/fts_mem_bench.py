#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Concurrent FTS query benchmark with peak RSS tracking.

This script runs concurrent full-text queries against a Lance dataset and reports
throughput plus peak resident memory (RSS).

Example:
  python scripts/fts_mem_bench.py \
    --uri /path/to/ds \
    --text-column text \
    --project id \
    --terms-file /path/to/terms.txt \
    --limit 1000 \
    --concurrency 100 \
    --total-queries 10000
"""

from __future__ import annotations

import argparse
import os
import random
import threading
import time
from typing import List, Optional, Tuple

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

import resource

import lance


def _load_terms(terms: List[str], terms_file: Optional[str]) -> List[str]:
    if terms_file:
        with open(terms_file, "r", encoding="utf-8") as f:
            file_terms = [line.strip() for line in f if line.strip()]
        return file_terms
    return [t for t in terms if t.strip()]


def _rss_bytes_from_resource(value: int) -> int:
    # On macOS ru_maxrss is bytes. On Linux it's KB.
    import platform

    if platform.system().lower() == "linux":
        return value * 1024
    return value


class RssMonitor:
    def __init__(self, interval_sec: float = 0.1) -> None:
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.max_rss_bytes = 0

    def start(self) -> None:
        if psutil is None:
            return
        proc = psutil.Process(os.getpid())

        def _run() -> None:
            while not self._stop.is_set():
                try:
                    rss = proc.memory_info().rss
                    if rss > self.max_rss_bytes:
                        self.max_rss_bytes = rss
                except Exception:
                    pass
                time.sleep(self.interval_sec)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def _run_queries(
    ds: lance.dataset.Dataset,
    text_column: str,
    project_columns: Optional[List[str]],
    terms: List[str],
    limit: int,
    count: int,
    seed: int,
) -> Tuple[int, float]:
    rng = random.Random(seed)
    total = 0
    total_latency = 0.0
    for _ in range(count):
        term = rng.choice(terms)
        start = time.perf_counter()
        ds.to_table(
            columns=project_columns,
            full_text_query={
                "query": term,
                "columns": [text_column],
            },
            limit=limit,
        )
        total_latency += time.perf_counter() - start
        total += 1
    return total, total_latency


def main() -> None:
    parser = argparse.ArgumentParser(description="Concurrent FTS query benchmark")
    parser.add_argument("--uri", required=True, help="Dataset URI")
    parser.add_argument("--text-column", required=True, help="FTS indexed text column")
    parser.add_argument(
        "--project",
        action="append",
        default=None,
        help="Column to project (can be repeated). If omitted, all columns are returned.",
    )
    parser.add_argument("--terms", nargs="*", default=[], help="Query terms")
    parser.add_argument("--terms-file", help="File with query terms (one per line)")
    parser.add_argument("--limit", type=int, default=1000, help="Per-query limit")
    parser.add_argument("--concurrency", type=int, default=100, help="Number of threads")
    parser.add_argument(
        "--total-queries",
        type=int,
        default=10000,
        help="Total number of queries across all threads",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--index-cache-bytes",
        type=int,
        default=None,
        help="Override session index cache size in bytes",
    )
    parser.add_argument(
        "--metadata-cache-bytes",
        type=int,
        default=None,
        help="Override session metadata cache size in bytes",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help="RSS polling interval in seconds (psutil only)",
    )

    args = parser.parse_args()
    terms = _load_terms(args.terms, args.terms_file)
    if not terms:
        raise SystemExit("No terms provided. Use --terms or --terms-file.")

    session = lance.Session(
        index_cache_size_bytes=args.index_cache_bytes,
        metadata_cache_size_bytes=args.metadata_cache_bytes,
    )
    ds = lance.dataset(args.uri, session=session)

    per_worker = args.total_queries // args.concurrency
    remainder = args.total_queries % args.concurrency

    monitor = RssMonitor(interval_sec=args.poll_interval)
    monitor.start()

    start = time.perf_counter()
    results = []

    import concurrent.futures as cf

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for i in range(args.concurrency):
            count = per_worker + (1 if i < remainder else 0)
            if count == 0:
                continue
            futures.append(
                executor.submit(
                    _run_queries,
                    ds,
                    args.text_column,
                    args.project,
                    terms,
                    args.limit,
                    count,
                    args.seed + i,
                )
            )
        for fut in cf.as_completed(futures):
            results.append(fut.result())

    elapsed = time.perf_counter() - start
    monitor.stop()

    total_queries = sum(r[0] for r in results)
    total_latency = sum(r[1] for r in results)
    avg_latency_ms = (total_latency / total_queries) * 1000 if total_queries else 0.0

    peak_rss_bytes = monitor.max_rss_bytes
    if peak_rss_bytes == 0:
        peak_rss_bytes = _rss_bytes_from_resource(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )

    print("=== FTS Concurrency Benchmark ===")
    print(f"Total queries: {total_queries}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Limit: {args.limit}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"QPS: {total_queries / elapsed:.2f}")
    print(f"Avg latency: {avg_latency_ms:.2f} ms")
    print(f"Peak RSS: {peak_rss_bytes / (1024 ** 2):.2f} MiB")


if __name__ == "__main__":
    main()
