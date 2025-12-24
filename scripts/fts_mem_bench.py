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


def _current_rss_bytes() -> Optional[int]:
    if psutil is None:
        return None
    try:
        return psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        return None


def _baseline_rss_bytes(samples: int, interval_sec: float) -> Optional[int]:
    if psutil is None:
        return None
    if samples <= 0:
        return _current_rss_bytes()
    values = []
    for _ in range(samples):
        rss = _current_rss_bytes()
        if rss is not None:
            values.append(rss)
        time.sleep(interval_sec)
    if not values:
        return None
    return min(values)


def _run_queries(
    ds: lance.dataset.Dataset,
    text_column: str,
    project_columns: Optional[List[str]],
    terms: List[str],
    limit: int,
    count: int,
    seed: int,
    terms_per_query: int,
    barrier: Optional[threading.Barrier] = None,
    term_sequence: Optional[List[str]] = None,
) -> Tuple[int, float]:
    rng = random.Random(seed)
    total = 0
    total_latency = 0.0
    if barrier is not None:
        barrier.wait()
    for i in range(count):
        if term_sequence is None:
            query_terms = rng.choices(terms, k=terms_per_query)
            term = " ".join(query_terms)
        else:
            term = term_sequence[i]
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


def _prewarm(
    ds: lance.dataset.Dataset,
    text_column: str,
    project_columns: Optional[List[str]],
    terms: List[str],
    limit: int,
    count: int,
    seed: int,
    terms_per_query: int,
) -> None:
    if count <= 0:
        return
    rng = random.Random(seed)
    for _ in range(count):
        query_terms = rng.choices(terms, k=terms_per_query)
        term = " ".join(query_terms)
        ds.to_table(
            columns=project_columns,
            full_text_query={
                "query": term,
                "columns": [text_column],
            },
            limit=limit,
        )


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
    parser.add_argument(
        "--terms-per-query",
        type=int,
        default=1,
        help="Number of terms per query",
    )
    parser.add_argument("--concurrency", type=int, default=100, help="Number of threads")
    parser.add_argument(
        "--total-queries",
        type=int,
        default=10000,
        help="Total number of queries across all threads",
    )
    parser.add_argument(
        "--prewarm-queries",
        type=int,
        default=0,
        help="Number of warmup queries before measurement",
    )
    parser.add_argument(
        "--prewarm-index",
        action="append",
        default=None,
        help="Index name to prewarm (can be repeated)",
    )
    parser.add_argument(
        "--prewarm-all",
        action="store_true",
        help="Prewarm all indices before measurement",
    )
    parser.add_argument(
        "--deterministic-queries",
        action="store_true",
        help="Use a deterministic query sequence (reduces variance)",
    )
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=5,
        help="Samples to take for baseline RSS (psutil only)",
    )
    parser.add_argument(
        "--baseline-interval",
        type=float,
        default=0.2,
        help="Seconds between baseline RSS samples",
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
        default=0.01,
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

    prewarm_names = []
    if args.prewarm_index:
        prewarm_names.extend(args.prewarm_index)
    if args.prewarm_all:
        try:
            prewarm_names.extend([d.name for d in ds.describe_indices()])
        except Exception:
            prewarm_names.extend([idx.name for idx in ds.list_indices()])
    if prewarm_names:
        for name in sorted(set(prewarm_names)):
            ds.prewarm_index(name)
    elif args.prewarm_queries > 0:
        _prewarm(
            ds,
            args.text_column,
            args.project,
            terms,
            args.limit,
            args.prewarm_queries,
            args.seed,
            args.terms_per_query,
        )

    baseline_rss = _baseline_rss_bytes(args.baseline_samples, args.baseline_interval)

    per_worker = args.total_queries // args.concurrency
    remainder = args.total_queries % args.concurrency

    monitor = RssMonitor(interval_sec=args.poll_interval)
    monitor.start()

    start = time.perf_counter()
    results = []

    import concurrent.futures as cf

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        worker_count = sum(
            1
            for i in range(args.concurrency)
            if per_worker + (1 if i < remainder else 0) > 0
        )
        barrier = threading.Barrier(parties=max(1, worker_count))
        term_sequences: Optional[List[List[str]]] = None
        if args.deterministic_queries:
            rng = random.Random(args.seed)
            query_terms = []
            for _ in range(args.total_queries):
                terms_for_query = rng.choices(terms, k=args.terms_per_query)
                query_terms.append(" ".join(terms_for_query))
            term_sequences = []
            offset = 0
            for i in range(args.concurrency):
                count = per_worker + (1 if i < remainder else 0)
                if count == 0:
                    term_sequences.append([])
                    continue
                term_sequences.append(query_terms[offset : offset + count])
                offset += count
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
                    args.terms_per_query,
                    barrier,
                    term_sequences[i] if term_sequences is not None else None,
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
    end_rss = _current_rss_bytes()

    print("=== FTS Concurrency Benchmark ===")
    print(f"Total queries: {total_queries}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Limit: {args.limit}")
    print(f"Terms per query: {args.terms_per_query}")
    print(f"Deterministic queries: {args.deterministic_queries}")
    if prewarm_names:
        print(f"Prewarm indices: {sorted(set(prewarm_names))}")
    else:
        print(f"Prewarm queries: {args.prewarm_queries}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"QPS: {total_queries / elapsed:.2f}")
    print(f"Avg latency: {avg_latency_ms:.2f} ms")
    print(f"Peak RSS: {peak_rss_bytes / (1024 ** 2):.2f} MiB")
    if baseline_rss is None:
        print("Baseline RSS: unavailable (install psutil for deltas)")
    else:
        print(f"Baseline RSS: {baseline_rss / (1024 ** 2):.2f} MiB")
        print(
            f"Peak RSS delta: {(peak_rss_bytes - baseline_rss) / (1024 ** 2):.2f} MiB"
        )
        if end_rss is not None:
            print(f"End RSS: {end_rss / (1024 ** 2):.2f} MiB")
            print(f"End RSS delta: {(end_rss - baseline_rss) / (1024 ** 2):.2f} MiB")


if __name__ == "__main__":
    main()
