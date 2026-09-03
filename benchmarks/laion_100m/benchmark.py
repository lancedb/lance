# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import platform
import re
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lance
import numpy as np
import psutil
from common import (
    DEFAULT_DATASET_URI,
    DEFAULT_ID_COLUMN,
    DEFAULT_VECTOR_COLUMN,
    ResourceSampler,
    append_jsonl,
    find_index,
    load_evaluation_data,
    open_branch,
    process_resource_snapshot,
    write_json,
)

DEFAULT_NPROBES = [16, 64, 256, 1024]
DEFAULT_CONCURRENCIES = [1, 8, 16]
DEFAULT_K_VALUES = [10, 100]
STAGE_METRICS_ENV = "LANCE_ANN_STAGE_METRICS"
PLAN_METRICS = (
    "find_partitions_calls",
    "shared_coarse_quantizer_fast_path",
    "coarse_quantizer_reused_segments",
    "partitions_ranked",
    "partitions_searched",
    "find_partitions_elapsed",
    "bytes_read",
    "iops",
    "requests",
)

STAGE_DURATION_UNITS_MS = {
    "ns": 1e-6,
    "us": 1e-3,
    "µs": 1e-3,
    "μs": 1e-3,
    "ms": 1.0,
    "s": 1000.0,
}

STAGE_COUNT_UNITS = {
    "": 1,
    "K": 1_000,
    "M": 1_000_000,
    "G": 1_000_000_000,
    "T": 1_000_000_000_000,
}
GIB = 2**30


@dataclass
class RequestResult:
    latency_seconds: float
    query_index: int
    row_ids: np.ndarray | None
    hits: int = 0
    error: str | None = None


def recall_at_k(result_ids: np.ndarray, truth_ids: np.ndarray, k: int) -> float:
    expected = set(int(value) for value in truth_ids[:k])
    actual = set(int(value) for value in result_ids[:k])
    return len(expected.intersection(actual)) / k


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    return float(np.percentile(np.asarray(values), quantile, method="linear"))


def summarize_latencies(latencies: list[float]) -> dict[str, float]:
    milliseconds = [latency * 1000 for latency in latencies]
    return {
        "latency_min_ms": min(milliseconds, default=math.nan),
        "latency_mean_ms": statistics.fmean(milliseconds) if milliseconds else math.nan,
        "latency_p50_ms": percentile(milliseconds, 50),
        "latency_p90_ms": percentile(milliseconds, 90),
        "latency_p95_ms": percentile(milliseconds, 95),
        "latency_p99_ms": percentile(milliseconds, 99),
        "latency_max_ms": max(milliseconds, default=math.nan),
    }


def find_stable_nprobes(
    recall_by_nprobes: dict[int, float],
    *,
    reference_tolerance: float = 0.002,
    doubling_gain_tolerance: float = 0.001,
) -> int | None:
    points = sorted(recall_by_nprobes)
    if len(points) < 2:
        return None
    reference = recall_by_nprobes[points[-1]]
    for current, following in zip(points, points[1:]):
        current_recall = recall_by_nprobes[current]
        following_recall = recall_by_nprobes[following]
        if (
            reference - current_recall <= reference_tolerance
            and following_recall - current_recall <= doubling_gain_tolerance
        ):
            return current
    return None


def _search(
    dataset: lance.LanceDataset,
    query: np.ndarray,
    *,
    query_index: int,
    vector_column: str,
    k: int,
    nprobes: int,
    query_parallelism: int,
    approx_mode: str,
) -> RequestResult:
    started = time.perf_counter()
    try:
        table = dataset.to_table(
            columns=["_rowid", "_distance"],
            nearest={
                "column": vector_column,
                "q": query,
                "k": k,
                "nprobes": nprobes,
                "query_parallelism": query_parallelism,
                "approx_mode": approx_mode,
            },
        )
        latency_seconds = time.perf_counter() - started
        row_ids = table["_rowid"].to_numpy(zero_copy_only=False)
        return RequestResult(latency_seconds, query_index, row_ids)
    except Exception as error:  # benchmark must report request failures
        return RequestResult(
            time.perf_counter() - started,
            query_index,
            None,
            error=f"{type(error).__name__}: {error}",
        )


def _search_with_recall(
    dataset: lance.LanceDataset,
    query: np.ndarray,
    truth: np.ndarray,
    *,
    vector_column: str,
    id_column: str,
    k: int,
    nprobes: int,
    query_parallelism: int,
    approx_mode: str,
) -> RequestResult:
    started = time.perf_counter()
    try:
        table = dataset.to_table(
            columns=[id_column, "_distance"],
            nearest={
                "column": vector_column,
                "q": query,
                "k": k,
                "nprobes": nprobes,
                "query_parallelism": query_parallelism,
                "approx_mode": approx_mode,
            },
        )
        result_ids = table[id_column].to_numpy(zero_copy_only=False)
        recall = recall_at_k(result_ids, truth, k)
        return RequestResult(
            time.perf_counter() - started,
            -1,
            None,
            hits=round(recall * k),
        )
    except Exception as error:  # benchmark must report request failures
        return RequestResult(
            time.perf_counter() - started,
            -1,
            None,
            error=f"{type(error).__name__}: {error}",
        )


def recall_from_timed_results(
    dataset: lance.LanceDataset,
    results: list[RequestResult],
    ground_truth: list[np.ndarray],
    *,
    id_column: str,
    k: int,
    max_queries: int,
) -> tuple[float, int, float]:
    if max_queries <= 0:
        raise ValueError("recall sample query count must be positive")
    first_result_by_query: dict[int, RequestResult] = {}
    for result in results:
        if result.error is None and result.row_ids is not None:
            first_result_by_query.setdefault(result.query_index, result)
    sampled = [
        first_result_by_query[query_index]
        for query_index in sorted(first_result_by_query)[:max_queries]
    ]
    if not sampled:
        return math.nan, 0, 0.0

    row_counts = [
        len(result.row_ids) for result in sampled if result.row_ids is not None
    ]
    flat_row_ids = np.concatenate(
        [result.row_ids for result in sampled if result.row_ids is not None]
    )
    started = time.perf_counter()
    result_ids = dataset._take_rows(flat_row_ids.tolist(), columns=[id_column])[
        id_column
    ].to_numpy(zero_copy_only=False)
    backfill_seconds = time.perf_counter() - started

    total_hits = 0
    offset = 0
    for result, row_count in zip(sampled, row_counts):
        ids = result_ids[offset : offset + row_count]
        total_hits += round(recall_at_k(ids, ground_truth[result.query_index], k) * k)
        offset += row_count
    return total_hits / (len(sampled) * k), len(sampled), backfill_seconds


def run_closed_loop(
    dataset: lance.LanceDataset,
    queries: list[np.ndarray],
    *,
    ground_truth: list[np.ndarray] | None,
    id_column: str,
    recall_sample_queries: int,
    vector_column: str,
    k: int,
    nprobes: int,
    concurrency: int,
    duration_seconds: float,
    query_parallelism: int,
    approx_mode: str,
) -> tuple[dict[str, Any], list[RequestResult]]:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if not queries:
        raise ValueError("queries must not be empty")
    if ground_truth is not None and len(queries) != len(ground_truth):
        raise ValueError("queries and ground_truth must be aligned")
    process = psutil.Process()
    ready = threading.Barrier(concurrency + 1)
    start_event = threading.Event()
    query_counter = itertools.count()
    query_lock = threading.Lock()
    start_holder: dict[str, float] = {}

    def worker() -> list[RequestResult]:
        ready.wait()
        start_event.wait()
        local: list[RequestResult] = []
        while time.perf_counter() < start_holder["deadline"]:
            with query_lock:
                query_index = next(query_counter) % len(queries)
            local.append(
                _search(
                    dataset,
                    queries[query_index],
                    query_index=query_index,
                    vector_column=vector_column,
                    k=k,
                    nprobes=nprobes,
                    query_parallelism=query_parallelism,
                    approx_mode=approx_mode,
                )
            )
        return local

    cpu_before, rss_before = process_resource_snapshot(process)
    resources = ResourceSampler(process)
    resources.start()
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker) for _ in range(concurrency)]
            ready.wait()
            wall_started = time.perf_counter()
            start_holder["deadline"] = wall_started + duration_seconds
            start_event.set()
            results = [result for future in futures for result in future.result()]
    finally:
        peak_rss = resources.stop()
    wall_seconds = time.perf_counter() - wall_started
    cpu_after, rss_after = process_resource_snapshot(process)

    successes = [result for result in results if result.error is None]
    failures = [result for result in results if result.error is not None]
    latencies = [result.latency_seconds for result in successes]
    if ground_truth is None:
        recall, recall_queries, recall_backfill_seconds = math.nan, 0, 0.0
    else:
        recall, recall_queries, recall_backfill_seconds = recall_from_timed_results(
            dataset,
            successes,
            ground_truth,
            id_column=id_column,
            k=k,
            max_queries=recall_sample_queries,
        )
    summary = {
        "requests": len(results),
        "successful_requests": len(successes),
        "failed_requests": len(failures),
        "error_rate": len(failures) / len(results) if results else 1.0,
        "wall_seconds": wall_seconds,
        "qps": len(successes) / wall_seconds,
        "recall": recall,
        "recall_queries": recall_queries,
        "recall_backfill_seconds": recall_backfill_seconds,
        "process_cpu_seconds": cpu_after - cpu_before,
        "average_cpu_cores": (cpu_after - cpu_before) / wall_seconds,
        "rss_before_gib": rss_before / 2**30,
        "rss_after_gib": rss_after / 2**30,
        "rss_peak_gib": peak_rss / 2**30,
        "latency_first_ms": latencies[0] * 1000 if latencies else math.nan,
        **summarize_latencies(latencies),
    }
    if failures:
        summary["errors"] = dict(
            Counter(result.error for result in failures).most_common(20)
        )
    return summary, results


def warm_up(
    dataset: lance.LanceDataset,
    queries: list[np.ndarray],
    args: argparse.Namespace,
    *,
    k: int,
    nprobes: int,
    concurrency: int,
) -> None:
    run_closed_loop(
        dataset,
        queries,
        ground_truth=None,
        id_column=args.id_column,
        recall_sample_queries=args.recall_sample_queries,
        vector_column=args.vector_column,
        k=k,
        nprobes=nprobes,
        concurrency=concurrency,
        duration_seconds=args.warmup_seconds,
        query_parallelism=args.query_parallelism,
        approx_mode=args.approx_mode,
    )


def analyze_query_plan(
    dataset: lance.LanceDataset,
    query: np.ndarray,
    args: argparse.Namespace,
    *,
    k: int,
    nprobes: int,
) -> tuple[str, dict[str, list[str]]]:
    scanner = dataset.scanner(
        columns=["_rowid", "_distance"],
        nearest={
            "column": args.vector_column,
            "q": query,
            "k": k,
            "nprobes": nprobes,
            "query_parallelism": args.query_parallelism,
            "approx_mode": args.approx_mode,
        },
    )
    plan = scanner.analyze_plan()
    parsed: dict[str, list[str]] = {}
    for metric in PLAN_METRICS:
        parsed[metric] = re.findall(rf"\b{re.escape(metric)}=([^,\]\s]+)", plan)
    return plan, parsed


def validate_ann_only_plan(plan: str) -> None:
    if re.search(r"^\s*LanceRead(?::|\s)", plan, flags=re.MULTILINE):
        raise ValueError(
            "Timed ANN plan contains LanceRead; result projection would fetch "
            "base dataset columns"
        )


def parse_duration_ms(value: str) -> float:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(ns|us|µs|μs|ms|s)", value.strip())
    if match is None:
        raise ValueError(f"Unsupported duration value {value!r}")
    magnitude, unit = match.groups()
    return float(magnitude) * STAGE_DURATION_UNITS_MS[unit]


def parse_metric_count(value: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?)", value.strip())
    if match is None:
        raise ValueError(f"Unsupported count value {value!r}")
    magnitude, unit = match.groups()
    return round(float(magnitude) * STAGE_COUNT_UNITS[unit])


def parse_plan_node_metrics(
    plan: str,
    node: str,
    metrics: dict[str, str],
) -> dict[str, int | float]:
    node_lines = [
        line.strip()
        for line in plan.splitlines()
        if line.strip().startswith(f"{node}:")
    ]
    if not node_lines:
        raise ValueError(f"Analyze plan does not contain a {node!r} node")

    parsed: dict[str, int | float] = {}
    for metric, metric_type in metrics.items():
        values = []
        for line in node_lines:
            match = re.search(rf"\b{re.escape(metric)}=([^,\]]+)", line)
            if match is None:
                raise ValueError(
                    f"Analyze plan node {node!r} is missing metric {metric!r}: {line}"
                )
            values.append(match.group(1))
        if metric_type == "duration":
            parsed[metric] = sum(parse_duration_ms(value) for value in values)
        elif metric_type == "count":
            try:
                parsed[metric] = sum(parse_metric_count(value) for value in values)
            except ValueError as error:
                raise ValueError(
                    f"Analyze plan node {node!r} has an invalid {metric!r}: {values}"
                ) from error
        else:
            raise ValueError(
                f"Unsupported metric type {metric_type!r} for metric {metric!r}"
            )
    return parsed


def parse_stage_plan(plan: str) -> dict[str, int | float]:
    coarse = parse_plan_node_metrics(
        plan,
        "ANNIvfPartition",
        {
            "find_partitions_elapsed": "duration",
            "find_partitions_calls": "count",
            "partitions_ranked": "count",
            "bytes_read": "count",
            "iops": "count",
            "requests": "count",
        },
    )
    bucket = parse_plan_node_metrics(
        plan,
        "ANNSubIndex",
        {
            "search_partitions_elapsed": "duration",
            "search_partitions_calls": "count",
            "partitions_searched": "count",
            "bytes_read": "count",
            "iops": "count",
            "requests": "count",
        },
    )
    coarse_task_ms = float(coarse["find_partitions_elapsed"])
    bucket_task_ms = float(bucket["search_partitions_elapsed"])
    total_task_ms = coarse_task_ms + bucket_task_ms
    return {
        "coarse_task_ms": coarse_task_ms,
        "coarse_calls": int(coarse["find_partitions_calls"]),
        "partitions_ranked": int(coarse["partitions_ranked"]),
        "coarse_bytes_read": int(coarse["bytes_read"]),
        "coarse_iops": int(coarse["iops"]),
        "coarse_requests": int(coarse["requests"]),
        "bucket_task_ms": bucket_task_ms,
        "bucket_calls": int(bucket["search_partitions_calls"]),
        "partitions_searched": int(bucket["partitions_searched"]),
        "bucket_bytes_read": int(bucket["bytes_read"]),
        "bucket_iops": int(bucket["iops"]),
        "bucket_requests": int(bucket["requests"]),
        "coarse_task_share": coarse_task_ms / total_task_ms
        if total_task_ms
        else math.nan,
        "bucket_task_share": bucket_task_ms / total_task_ms
        if total_task_ms
        else math.nan,
    }


def validate_plan_metrics(
    metrics_by_mode: dict[str, dict[str, list[str]]], expected_segments: int
) -> None:
    for mode in ("off", "on"):
        validate_mode_plan_metrics(metrics_by_mode[mode], mode, expected_segments)


def validate_mode_plan_metrics(
    metrics: dict[str, list[str]], mode: str, expected_segments: int
) -> None:
    def total(metric: str) -> int:
        values = metrics[metric]
        if not values or any(not value.isdigit() for value in values):
            raise ValueError(f"Could not parse {metric!r} from {mode} analyze_plan")
        return sum(int(value) for value in values)

    expected = {
        "off": {
            "find_partitions_calls": expected_segments,
            "shared_coarse_quantizer_fast_path": 0,
            "coarse_quantizer_reused_segments": 0,
        },
        "on": {
            "find_partitions_calls": 1,
            "shared_coarse_quantizer_fast_path": 1,
            "coarse_quantizer_reused_segments": expected_segments - 1,
        },
    }
    if mode not in expected:
        raise ValueError(f"Unsupported benchmark mode {mode!r}")
    for metric, expected_value in expected[mode].items():
        actual = total(metric)
        if actual != expected_value:
            raise ValueError(
                f"Unexpected {mode} {metric}: {actual}, expected {expected_value}"
            )


def _metadata(args: argparse.Namespace) -> dict[str, Any]:
    arguments = {key: value for key, value in vars(args).items() if not callable(value)}
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "argv": arguments,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "lance_version": lance.__version__,
    }


def _configured_index_cache_size_bytes(args: argparse.Namespace) -> int | None:
    size_gib = args.index_cache_size_gib
    if size_gib is None:
        if args.prewarm_index:
            raise ValueError("--index-cache-size-gib is required with --prewarm-index")
        return None
    if size_gib <= 0:
        raise ValueError("--index-cache-size-gib must be positive")
    return round(size_gib * GIB)


def _open_benchmark_branch(args: argparse.Namespace, branch: str) -> lance.LanceDataset:
    return open_branch(
        args.dataset_uri,
        branch,
        index_cache_size_bytes=_configured_index_cache_size_bytes(args),
    )


def _prewarm_index(
    dataset: lance.LanceDataset,
    args: argparse.Namespace,
    *,
    output_name: str,
) -> dict[str, Any] | None:
    if not args.prewarm_index:
        return None

    index = find_index(dataset, args.index_name)
    configured_size = _configured_index_cache_size_bytes(args)
    index_size = int(index.total_size_bytes)
    if configured_size is None or configured_size < index_size:
        raise ValueError(
            "Configured index cache is smaller than the on-disk index: "
            f"{configured_size} < {index_size} bytes"
        )

    process = psutil.Process()
    cpu_before, rss_before = process_resource_snapshot(process)
    cache_before = dataset.session().index_cache_size_bytes()
    resources = ResourceSampler(process)
    started = time.perf_counter()
    resources.start()
    try:
        dataset.prewarm_index(args.index_name)
    finally:
        peak_rss = resources.stop()
    elapsed_seconds = time.perf_counter() - started
    cpu_after, rss_after = process_resource_snapshot(process)
    cache_after = dataset.session().index_cache_size_bytes()

    metrics = {
        "index_name": args.index_name,
        "segments": len(index.segments),
        "index_total_size_bytes": index_size,
        "configured_index_cache_size_bytes": configured_size,
        "cache_size_bytes_before": cache_before,
        "cache_size_bytes_after": cache_after,
        "cache_size_bytes_delta": cache_after - cache_before,
        "prewarm_seconds": elapsed_seconds,
        "process_cpu_seconds": cpu_after - cpu_before,
        "average_cpu_cores": (cpu_after - cpu_before) / elapsed_seconds,
        "rss_before_gib": rss_before / GIB,
        "rss_after_gib": rss_after / GIB,
        "rss_peak_gib": peak_rss / GIB,
        "fully_resident_by_size": cache_after >= index_size,
    }
    write_json(args.output_dir / output_name, metrics)
    if cache_after < index_size:
        raise RuntimeError(
            "Index prewarm completed but the resident index cache is smaller than "
            f"the index: {cache_after} < {index_size} bytes"
        )
    return metrics


def _result_ids(
    dataset: lance.LanceDataset,
    query: np.ndarray,
    args: argparse.Namespace,
    *,
    k: int,
    nprobes: int,
) -> set[int]:
    table = dataset.to_table(
        columns=["_rowid", "_distance"],
        nearest={
            "column": args.vector_column,
            "q": query,
            "k": k,
            "nprobes": nprobes,
            "query_parallelism": args.query_parallelism,
            "approx_mode": args.approx_mode,
        },
    )
    return set(int(value) for value in table["_rowid"].to_pylist())


def validate_comparison_inputs(
    datasets: dict[str, lance.LanceDataset],
    queries: list[np.ndarray],
    args: argparse.Namespace,
) -> None:
    baseline = find_index(datasets["off"], args.index_name)
    optimized = find_index(datasets["on"], args.index_name)
    if len(baseline.segments) != args.expected_segments:
        raise ValueError(
            f"Baseline has {len(baseline.segments)} index segments, "
            f"expected {args.expected_segments}"
        )
    if len(optimized.segments) != args.expected_segments:
        raise ValueError(
            f"Optimized branch has {len(optimized.segments)} index segments, "
            f"expected {args.expected_segments}"
        )
    if "coarse_quantizer_fingerprint" in baseline.details:
        raise ValueError("Baseline index unexpectedly enables coarse-quantizer reuse")
    if "coarse_quantizer_fingerprint" not in optimized.details:
        raise ValueError("Optimized index does not enable coarse-quantizer reuse")
    if datasets["off"].count_rows() != datasets["on"].count_rows():
        raise ValueError("The two Lance branches have different row counts")
    if datasets["off"].schema != datasets["on"].schema:
        raise ValueError("The two Lance branches have different schemas")

    validation_k = max(args.k)
    validation_nprobes = max(args.nprobes)
    for query_index, query in enumerate(queries[: args.preflight_queries]):
        baseline_ids = _result_ids(
            datasets["off"],
            query,
            args,
            k=validation_k,
            nprobes=validation_nprobes,
        )
        optimized_ids = _result_ids(
            datasets["on"],
            query,
            args,
            k=validation_k,
            nprobes=validation_nprobes,
        )
        if baseline_ids != optimized_ids:
            raise ValueError(
                "A/B result sets differ during preflight for query "
                f"{query_index}; verify that both builds used the same model artifacts"
            )


def profile_query_plan(
    dataset: lance.LanceDataset,
    query: np.ndarray,
    args: argparse.Namespace,
    *,
    k: int,
    nprobes: int,
) -> tuple[str, dict[str, int | float]]:
    scanner = dataset.scanner(
        columns=["_rowid", "_distance"],
        nearest={
            "column": args.vector_column,
            "q": query,
            "k": k,
            "nprobes": nprobes,
            "query_parallelism": args.query_parallelism,
            "approx_mode": args.approx_mode,
        },
    )
    started = time.perf_counter()
    plan = scanner.analyze_plan()
    profile_wall_ms = (time.perf_counter() - started) * 1000
    return plan, {"profile_wall_ms": profile_wall_ms, **parse_stage_plan(plan)}


def _write_summary_csv(
    output_dir: Path, rows: list[dict[str, Any]], filename: str = "summary.csv"
) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row if key != "errors"})
    with (output_dir / filename).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        key = (row["k"], row["nprobes"], row["concurrency"])
        grouped.setdefault(key, {}).setdefault(row["mode"], []).append(row)

    summaries = []
    lower_is_better = (
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
    )
    for (k, nprobes, concurrency), modes in sorted(grouped.items()):
        if set(modes) != {"off", "on"}:
            raise ValueError(f"Incomplete A/B results for {(k, nprobes, concurrency)}")

        def median(mode: str, metric: str) -> float:
            return statistics.median(row[metric] for row in modes[mode])

        row = {"k": k, "nprobes": nprobes, "concurrency": concurrency}
        for metric in (
            "qps",
            "recall",
            "error_rate",
            "average_cpu_cores",
            "rss_peak_gib",
            *lower_is_better,
        ):
            row[f"off_{metric}"] = median("off", metric)
            row[f"on_{metric}"] = median("on", metric)

        off_qps = row["off_qps"]
        row["qps_gain_percent"] = (
            (row["on_qps"] / off_qps - 1) * 100 if off_qps else math.nan
        )
        for metric in lower_is_better:
            baseline = row[f"off_{metric}"]
            row[f"{metric}_reduction_percent"] = (
                (1 - row[f"on_{metric}"] / baseline) * 100 if baseline else math.nan
            )
        row["recall_delta"] = row["on_recall"] - row["off_recall"]
        summaries.append(row)
    return summaries


def summarize_single_mode(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["k"], row["nprobes"], row["concurrency"])
        grouped.setdefault(key, []).append(row)

    summaries = []
    metrics = (
        "qps",
        "recall",
        "error_rate",
        "average_cpu_cores",
        "rss_peak_gib",
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
    )
    for (k, nprobes, concurrency), repeats in sorted(grouped.items()):
        summary = {
            "k": k,
            "nprobes": nprobes,
            "concurrency": concurrency,
            "repeats": len(repeats),
        }
        for metric in metrics:
            summary[metric] = statistics.median(row[metric] for row in repeats)
        summaries.append(summary)
    return summaries


def summarize_stage_profile(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[int, dict[str, dict[str, Any]]]] = {}
    for row in rows:
        key = (row["k"], row["nprobes"])
        grouped.setdefault(key, {}).setdefault(row["query_id"], {})[row["mode"]] = row

    summary_metrics = (
        "profile_wall_ms",
        "coarse_task_ms",
        "bucket_task_ms",
        "coarse_task_share",
        "bucket_task_share",
        "coarse_calls",
        "bucket_calls",
        "partitions_ranked",
        "partitions_searched",
        "coarse_bytes_read",
        "coarse_iops",
        "coarse_requests",
        "bucket_bytes_read",
        "bucket_iops",
        "bucket_requests",
    )
    summaries: list[dict[str, Any]] = []
    for (k, nprobes), queries in sorted(grouped.items()):
        if any(set(modes) != {"off", "on"} for modes in queries.values()):
            raise ValueError(f"Incomplete stage-profile A/B results for {(k, nprobes)}")
        summary: dict[str, Any] = {
            "k": k,
            "nprobes": nprobes,
            "profile_queries": len(queries),
        }
        for mode in ("off", "on"):
            mode_rows = [modes[mode] for modes in queries.values()]
            for metric in summary_metrics:
                values = [float(row[metric]) for row in mode_rows]
                summary[f"{mode}_{metric}_median"] = statistics.median(values)
                summary[f"{mode}_{metric}_p95"] = percentile(values, 95)

        paired = [modes for modes in queries.values()]

        def median_change(metric: str, *, lower_is_better: bool) -> float:
            changes = []
            for modes in paired:
                baseline = float(modes["off"][metric])
                optimized = float(modes["on"][metric])
                if baseline == 0:
                    continue
                if lower_is_better:
                    changes.append((1 - optimized / baseline) * 100)
                else:
                    changes.append((optimized / baseline - 1) * 100)
            return statistics.median(changes) if changes else math.nan

        summary["coarse_work_reduction_percent"] = median_change(
            "coarse_task_ms", lower_is_better=True
        )
        summary["bucket_work_delta_percent"] = median_change(
            "bucket_task_ms", lower_is_better=False
        )
        summary["profile_wall_reduction_percent"] = median_change(
            "profile_wall_ms", lower_is_better=True
        )
        summaries.append(summary)
    return summaries


def run_calibration(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metadata.json", _metadata(args))
    queries, ground_truth = load_evaluation_data(args.queries, args.ground_truth)
    dataset = _open_benchmark_branch(args, args.branch)
    find_index(dataset, args.index_name)
    _prewarm_index(dataset, args, output_name="prewarm.json")
    rows: list[dict[str, Any]] = []

    for k in args.k:
        for nprobes in args.nprobes:
            warm_up(
                dataset,
                queries,
                args,
                k=k,
                nprobes=nprobes,
                concurrency=1,
            )
            started = time.perf_counter()
            results = [
                _search_with_recall(
                    dataset,
                    query,
                    truth,
                    vector_column=args.vector_column,
                    id_column=args.id_column,
                    k=k,
                    nprobes=nprobes,
                    query_parallelism=args.query_parallelism,
                    approx_mode=args.approx_mode,
                )
                for query, truth in zip(queries, ground_truth)
            ]
            wall_seconds = time.perf_counter() - started
            successful = [result for result in results if result.error is None]
            row = {
                "phase": "calibrate",
                "branch": args.branch,
                "k": k,
                "nprobes": nprobes,
                "concurrency": 1,
                "requests": len(results),
                "successful_requests": len(successful),
                "failed_requests": len(results) - len(successful),
                "wall_seconds": wall_seconds,
                "qps": len(successful) / wall_seconds,
                "recall": sum(result.hits for result in successful)
                / (len(successful) * k)
                if successful
                else 0.0,
                **summarize_latencies(
                    [result.latency_seconds for result in successful]
                ),
            }
            rows.append(row)
            append_jsonl(output_dir / "runs.jsonl", row)

    stable_by_k = {
        k: find_stable_nprobes(
            {row["nprobes"]: row["recall"] for row in rows if row["k"] == k}
        )
        for k in args.k
    }
    stable_values = [value for value in stable_by_k.values() if value is not None]
    selected = max(stable_values) if len(stable_values) == len(args.k) else None
    write_json(
        output_dir / "calibration.json",
        {"stable_nprobes_by_k": stable_by_k, "selected_nprobes": selected},
    )
    _write_summary_csv(output_dir, rows)
    if selected is None:
        raise RuntimeError(
            "Recall did not reach a stable plateau for every k; add nprobes=4096"
        )


def _run_timed_point(
    dataset: lance.LanceDataset,
    queries: list[np.ndarray],
    ground_truth: list[np.ndarray],
    args: argparse.Namespace,
    *,
    mode: str,
    branch: str,
    repeat: int,
    k: int,
    nprobes: int,
    concurrency: int,
) -> dict[str, Any]:
    warm_up(
        dataset,
        queries,
        args,
        k=k,
        nprobes=nprobes,
        concurrency=concurrency,
    )
    summary, _ = run_closed_loop(
        dataset,
        queries,
        ground_truth=ground_truth,
        id_column=args.id_column,
        recall_sample_queries=args.recall_sample_queries,
        vector_column=args.vector_column,
        k=k,
        nprobes=nprobes,
        concurrency=concurrency,
        duration_seconds=args.duration_seconds,
        query_parallelism=args.query_parallelism,
        approx_mode=args.approx_mode,
    )
    return {
        "phase": "baseline",
        "mode": mode,
        "branch": branch,
        "repeat": repeat,
        "k": k,
        "nprobes": nprobes,
        "concurrency": concurrency,
        **summary,
    }


def run_baseline(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metadata.json", _metadata(args))
    queries, ground_truth = load_evaluation_data(args.queries, args.ground_truth)
    dataset = _open_benchmark_branch(args, args.branch)
    index = find_index(dataset, args.index_name)
    if len(index.segments) != args.expected_segments:
        raise ValueError(
            f"Baseline has {len(index.segments)} index segments, "
            f"expected {args.expected_segments}"
        )
    has_fingerprint = "coarse_quantizer_fingerprint" in index.details
    if args.mode == "off" and has_fingerprint:
        raise ValueError("Reuse-off index unexpectedly enables coarse-quantizer reuse")
    if args.mode == "on" and not has_fingerprint:
        raise ValueError("Reuse-on index does not enable coarse-quantizer reuse")

    _prewarm_index(dataset, args, output_name="prewarm.json")

    plan, metrics = analyze_query_plan(
        dataset,
        queries[0],
        args,
        k=max(args.k),
        nprobes=args.nprobes[len(args.nprobes) // 2],
    )
    (output_dir / f"analyze_plan_{args.mode}.txt").write_text(plan, encoding="utf-8")
    write_json(output_dir / f"analyze_plan_{args.mode}.json", metrics)
    validate_ann_only_plan(plan)
    validate_mode_plan_metrics(metrics, args.mode, args.expected_segments)

    rows: list[dict[str, Any]] = []
    for k in args.k:
        for nprobes in args.nprobes:
            for concurrency in args.concurrency:
                for repeat in range(args.repeats):
                    row = _run_timed_point(
                        dataset,
                        queries,
                        ground_truth,
                        args,
                        mode=args.mode,
                        branch=args.branch,
                        repeat=repeat,
                        k=k,
                        nprobes=nprobes,
                        concurrency=concurrency,
                    )
                    rows.append(row)
                    append_jsonl(output_dir / "runs.jsonl", row)

    _write_summary_csv(output_dir, rows)
    aggregate = summarize_single_mode(rows)
    write_json(output_dir / "baseline.json", aggregate)
    _write_summary_csv(output_dir, aggregate, "baseline.csv")


def run_comparison(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metadata.json", _metadata(args))
    queries, ground_truth = load_evaluation_data(args.queries, args.ground_truth)
    datasets = {
        "off": _open_benchmark_branch(args, args.baseline_branch),
        "on": _open_benchmark_branch(args, args.optimized_branch),
    }
    _prewarm_index(datasets["off"], args, output_name="prewarm_off.json")
    _prewarm_index(datasets["on"], args, output_name="prewarm_on.json")
    validate_comparison_inputs(datasets, queries, args)
    plan_metrics = {}
    for mode, dataset in datasets.items():
        plan, metrics = analyze_query_plan(
            dataset,
            queries[0],
            args,
            k=max(args.k),
            nprobes=args.nprobes[len(args.nprobes) // 2],
        )
        (output_dir / f"analyze_plan_{mode}.txt").write_text(plan, encoding="utf-8")
        write_json(output_dir / f"analyze_plan_{mode}.json", metrics)
        validate_ann_only_plan(plan)
        plan_metrics[mode] = metrics
    validate_plan_metrics(plan_metrics, args.expected_segments)
    rows: list[dict[str, Any]] = []

    for k in args.k:
        for nprobes in args.nprobes:
            for concurrency in args.concurrency:
                for repeat in range(args.repeats):
                    order = ["off", "on"] if repeat % 2 == 0 else ["on", "off"]
                    for mode in order:
                        dataset = datasets[mode]
                        row = _run_timed_point(
                            dataset,
                            queries,
                            ground_truth,
                            args,
                            mode=mode,
                            branch=args.baseline_branch
                            if mode == "off"
                            else args.optimized_branch,
                            repeat=repeat,
                            k=k,
                            nprobes=nprobes,
                            concurrency=concurrency,
                        )
                        row["phase"] = "compare"
                        rows.append(row)
                        append_jsonl(output_dir / "runs.jsonl", row)

    _write_summary_csv(output_dir, rows)
    aggregate = summarize_comparison(rows)
    write_json(output_dir / "comparison.json", aggregate)
    _write_summary_csv(output_dir, aggregate, "comparison.csv")


def run_profile(args: argparse.Namespace) -> None:
    if args.profile_queries <= 0:
        raise ValueError("profile_queries must be positive")

    # Rust reads this setting before the first ANN sub-index execution. Keep the
    # normal calibration and comparison commands on the uninstrumented path.
    os.environ[STAGE_METRICS_ENV] = "1"

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(args)
    metadata["stage_metrics_env"] = STAGE_METRICS_ENV
    write_json(output_dir / "metadata.json", metadata)
    queries, _ = load_evaluation_data(args.queries, args.ground_truth)
    datasets = {
        "off": _open_benchmark_branch(args, args.baseline_branch),
        "on": _open_benchmark_branch(args, args.optimized_branch),
    }
    _prewarm_index(datasets["off"], args, output_name="prewarm_off.json")
    _prewarm_index(datasets["on"], args, output_name="prewarm_on.json")
    validate_comparison_inputs(datasets, queries, args)

    profile_query_count = min(args.profile_queries, len(queries))
    rows: list[dict[str, Any]] = []
    plan_dir = output_dir / "stage_plans"
    plan_dir.mkdir(parents=True, exist_ok=True)

    for k in args.k:
        for nprobes in args.nprobes:
            for mode in ("off", "on"):
                warm_up(
                    datasets[mode],
                    queries,
                    args,
                    k=k,
                    nprobes=nprobes,
                    concurrency=1,
                )

            for query_id, query in enumerate(queries[:profile_query_count]):
                order = ("off", "on") if query_id % 2 == 0 else ("on", "off")
                for mode in order:
                    plan, metrics = profile_query_plan(
                        datasets[mode],
                        query,
                        args,
                        k=k,
                        nprobes=nprobes,
                    )
                    validate_ann_only_plan(plan)
                    if query_id == 0:
                        (plan_dir / f"{mode}_k{k}_nprobes{nprobes}.txt").write_text(
                            plan, encoding="utf-8"
                        )
                    row = {
                        "phase": "profile",
                        "mode": mode,
                        "branch": args.baseline_branch
                        if mode == "off"
                        else args.optimized_branch,
                        "query_id": query_id,
                        "k": k,
                        "nprobes": nprobes,
                        **metrics,
                    }
                    rows.append(row)
                    append_jsonl(output_dir / "stage_profile.jsonl", row)

    _write_summary_csv(output_dir, rows, "stage_profile.csv")
    summary = summarize_stage_profile(rows)
    write_json(output_dir / "stage_summary.json", summary)
    _write_summary_csv(output_dir, summary, "stage_summary.csv")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-uri", default=DEFAULT_DATASET_URI)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vector-column", default=DEFAULT_VECTOR_COLUMN)
    parser.add_argument("--id-column", default=DEFAULT_ID_COLUMN)
    parser.add_argument("--index-name", default="emb_ivf_rq")
    parser.add_argument("--k", type=int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--nprobes", type=int, nargs="+", default=DEFAULT_NPROBES)
    parser.add_argument("--query-parallelism", type=int, default=1)
    parser.add_argument(
        "--recall-sample-queries",
        type=int,
        default=100,
        help=(
            "Timed ANN results to batch-materialize after each run for recall "
            "measurement (default: 100)"
        ),
    )
    parser.add_argument(
        "--approx-mode", choices=["fast", "normal", "accurate"], default="normal"
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=5.0,
        help="Per-configuration query warm-up after the full index prewarm",
    )
    parser.add_argument(
        "--prewarm-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Load every index partition into memory before testing (default: enabled)"
        ),
    )
    parser.add_argument(
        "--index-cache-size-gib",
        type=float,
        default=128.0,
        help="Index cache capacity in GiB (default: 128)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark LAION 100M with PyLance")
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser("calibrate")
    _add_common_arguments(calibrate)
    calibrate.add_argument("--branch", required=True)
    calibrate.set_defaults(run=run_calibration)

    baseline = subparsers.add_parser("baseline")
    _add_common_arguments(baseline)
    baseline.add_argument("--branch", required=True)
    baseline.add_argument(
        "--mode",
        choices=("off", "on"),
        default="off",
        help="Expected shared coarse-quantizer mode for this branch (default: off)",
    )
    baseline.add_argument(
        "--concurrency", type=int, nargs="+", default=DEFAULT_CONCURRENCIES
    )
    baseline.add_argument("--duration-seconds", type=float, default=10.0)
    baseline.add_argument("--repeats", type=int, default=3)
    baseline.add_argument("--expected-segments", type=int, default=6)
    baseline.set_defaults(run=run_baseline)

    compare = subparsers.add_parser("compare")
    _add_common_arguments(compare)
    compare.add_argument("--baseline-branch", required=True)
    compare.add_argument("--optimized-branch", required=True)
    compare.add_argument(
        "--concurrency", type=int, nargs="+", default=DEFAULT_CONCURRENCIES
    )
    compare.add_argument("--duration-seconds", type=float, default=10.0)
    compare.add_argument("--repeats", type=int, default=3)
    compare.add_argument("--expected-segments", type=int, default=6)
    compare.add_argument("--preflight-queries", type=int, default=10)
    compare.set_defaults(run=run_comparison)

    profile = subparsers.add_parser("profile")
    _add_common_arguments(profile)
    profile.add_argument("--baseline-branch", required=True)
    profile.add_argument("--optimized-branch", required=True)
    profile.add_argument("--profile-queries", type=int, default=50)
    profile.add_argument("--expected-segments", type=int, default=6)
    profile.add_argument("--preflight-queries", type=int, default=10)
    profile.set_defaults(run=run_profile)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.run(arguments)
