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

DEFAULT_NPROBES = [128, 256, 512, 1024, 2048]
DEFAULT_CONCURRENCIES = [1, 8, 32]
DEFAULT_K_VALUES = [10, 100]
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


@dataclass
class RequestResult:
    latency_seconds: float
    hits: int
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
            columns=[id_column],
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
        return RequestResult(time.perf_counter() - started, round(recall * k))
    except Exception as error:  # benchmark must report request failures
        return RequestResult(
            time.perf_counter() - started,
            0,
            f"{type(error).__name__}: {error}",
        )


def run_closed_loop(
    dataset: lance.LanceDataset,
    queries: list[np.ndarray],
    ground_truth: list[np.ndarray],
    *,
    vector_column: str,
    id_column: str,
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
    if not queries or len(queries) != len(ground_truth):
        raise ValueError("queries and ground_truth must be non-empty and aligned")
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
                    ground_truth[query_index],
                    vector_column=vector_column,
                    id_column=id_column,
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
    summary = {
        "requests": len(results),
        "successful_requests": len(successes),
        "failed_requests": len(failures),
        "error_rate": len(failures) / len(results) if results else 1.0,
        "wall_seconds": wall_seconds,
        "qps": len(successes) / wall_seconds,
        "recall": sum(result.hits for result in successes) / (len(successes) * k)
        if successes
        else 0.0,
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
    ground_truth: list[np.ndarray],
    args: argparse.Namespace,
    *,
    k: int,
    nprobes: int,
    concurrency: int,
) -> None:
    run_closed_loop(
        dataset,
        queries,
        ground_truth,
        vector_column=args.vector_column,
        id_column=args.id_column,
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
        columns=[args.id_column],
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


def validate_plan_metrics(
    metrics_by_mode: dict[str, dict[str, list[str]]], expected_segments: int
) -> None:
    def total(mode: str, metric: str) -> int:
        values = metrics_by_mode[mode][metric]
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
    for mode, expected_metrics in expected.items():
        for metric, expected_value in expected_metrics.items():
            actual = total(mode, metric)
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


def _result_ids(
    dataset: lance.LanceDataset,
    query: np.ndarray,
    args: argparse.Namespace,
    *,
    k: int,
    nprobes: int,
) -> set[int]:
    table = dataset.to_table(
        columns=[args.id_column],
        nearest={
            "column": args.vector_column,
            "q": query,
            "k": k,
            "nprobes": nprobes,
            "query_parallelism": args.query_parallelism,
            "approx_mode": args.approx_mode,
        },
    )
    return set(int(value) for value in table[args.id_column].to_pylist())


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


def run_calibration(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metadata.json", _metadata(args))
    queries, ground_truth = load_evaluation_data(args.queries, args.ground_truth)
    dataset = open_branch(args.dataset_uri, args.branch)
    find_index(dataset, args.index_name)
    rows: list[dict[str, Any]] = []

    for k in args.k:
        for nprobes in args.nprobes:
            warm_up(
                dataset,
                queries,
                ground_truth,
                args,
                k=k,
                nprobes=nprobes,
                concurrency=1,
            )
            started = time.perf_counter()
            results = [
                _search(
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


def run_comparison(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metadata.json", _metadata(args))
    queries, ground_truth = load_evaluation_data(args.queries, args.ground_truth)
    datasets = {
        "off": open_branch(args.dataset_uri, args.baseline_branch),
        "on": open_branch(args.dataset_uri, args.optimized_branch),
    }
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
                        warm_up(
                            dataset,
                            queries,
                            ground_truth,
                            args,
                            k=k,
                            nprobes=nprobes,
                            concurrency=concurrency,
                        )
                        summary, _ = run_closed_loop(
                            dataset,
                            queries,
                            ground_truth,
                            vector_column=args.vector_column,
                            id_column=args.id_column,
                            k=k,
                            nprobes=nprobes,
                            concurrency=concurrency,
                            duration_seconds=args.duration_seconds,
                            query_parallelism=args.query_parallelism,
                            approx_mode=args.approx_mode,
                        )
                        row = {
                            "phase": "compare",
                            "mode": mode,
                            "branch": args.baseline_branch
                            if mode == "off"
                            else args.optimized_branch,
                            "repeat": repeat,
                            "k": k,
                            "nprobes": nprobes,
                            "concurrency": concurrency,
                            **summary,
                        }
                        rows.append(row)
                        append_jsonl(output_dir / "runs.jsonl", row)

    _write_summary_csv(output_dir, rows)
    aggregate = summarize_comparison(rows)
    write_json(output_dir / "comparison.json", aggregate)
    _write_summary_csv(output_dir, aggregate, "comparison.csv")


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
        "--approx-mode", choices=["fast", "normal", "accurate"], default="normal"
    )
    parser.add_argument("--warmup-seconds", type=float, default=5.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark LAION 100M with PyLance")
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser("calibrate")
    _add_common_arguments(calibrate)
    calibrate.add_argument("--branch", required=True)
    calibrate.set_defaults(run=run_calibration)

    compare = subparsers.add_parser("compare")
    _add_common_arguments(compare)
    compare.add_argument("--baseline-branch", required=True)
    compare.add_argument("--optimized-branch", required=True)
    compare.add_argument(
        "--concurrency", type=int, nargs="+", default=DEFAULT_CONCURRENCIES
    )
    compare.add_argument("--duration-seconds", type=float, default=60.0)
    compare.add_argument("--repeats", type=int, default=3)
    compare.add_argument("--expected-segments", type=int, default=6)
    compare.add_argument("--preflight-queries", type=int, default=10)
    compare.set_defaults(run=run_comparison)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.run(arguments)
