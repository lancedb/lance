#!/usr/bin/env python3

"""Validate parity and summarize JSONL emitted by fts_s3_benchmark."""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

CanonicalResult = tuple[tuple[int, int], ...]
ParityResult = tuple[CanonicalResult, int | None, int]


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percent / 100.0)
    return ordered[index]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append(value)
    return records


def canonical_result(record: dict[str, Any]) -> CanonicalResult:
    row_ids = record.get("row_ids")
    score_bits = record.get("score_bits")
    if not isinstance(row_ids, list) or not isinstance(score_bits, list):
        raise ValueError("query record is missing row_ids or score_bits")
    if len(row_ids) != len(score_bits):
        raise ValueError("query record has different row and score counts")
    return tuple(sorted(zip(row_ids, score_bits)))


def score_from_bits(bits: int) -> float:
    score = struct.unpack("!f", struct.pack("!I", bits))[0]
    if not math.isfinite(score):
        raise ValueError(f"query record contains a non-finite score: {bits}")
    return score


def parity_result(
    record: dict[str, Any],
) -> ParityResult:
    result = canonical_result(record)
    if not result:
        return (), None, 0

    cutoff_score = min((score for _, score in result), key=score_from_bits)
    above_cutoff = tuple(pair for pair in result if pair[1] != cutoff_score)
    cutoff_count = len(result) - len(above_cutoff)
    return above_cutoff, cutoff_score, cutoff_count


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    queries = [record for record in records if record.get("event") == "query"]
    summaries = [record for record in records if record.get("event") == "summary"]
    latencies = [float(record["latency_ms"]) for record in queries]
    throughputs = [float(record["throughput_qps"]) for record in summaries]
    peak_rss = [
        int(record["peak_rss_kib"])
        for record in queries + summaries
        if record.get("peak_rss_kib") is not None
    ]
    return {
        "query_executions": len(queries),
        "latency_mean_ms": mean(latencies) if latencies else None,
        "latency_p50_ms": percentile(latencies, 50.0),
        "latency_p95_ms": percentile(latencies, 95.0),
        "latency_p99_ms": percentile(latencies, 99.0),
        "throughput_qps_mean": mean(throughputs) if throughputs else None,
        "peak_rss_kib_max": max(peak_rss) if peak_rss else None,
    }


def relative_change(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or baseline == 0:
        return None
    return (candidate - baseline) / baseline * 100.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for phase in ("cold", "warm", "throughput"):
        for variant in ("baseline", "candidate"):
            path = args.results / f"{phase}_{variant}.jsonl"
            if not path.is_file():
                raise ValueError(f"missing benchmark output: {path}")
            data[(phase, variant)] = read_jsonl(path)

    parity_failures: list[dict[str, Any]] = []
    cutoff_ties: list[dict[str, Any]] = []
    for phase in ("cold", "warm", "throughput"):
        by_variant: dict[str, dict[str, set[ParityResult]]] = {}
        raw_by_variant: dict[str, dict[str, set[CanonicalResult]]] = {}
        for variant in ("baseline", "candidate"):
            results: dict[str, set[ParityResult]] = defaultdict(set)
            raw_results: dict[str, set[CanonicalResult]] = defaultdict(set)
            for record in data[(phase, variant)]:
                if record.get("event") == "query":
                    query = str(record["query"])
                    results[query].add(parity_result(record))
                    raw_results[query].add(canonical_result(record))
            by_variant[variant] = results
            raw_by_variant[variant] = raw_results

        queries = set(by_variant["baseline"]) | set(by_variant["candidate"])
        for query in sorted(queries):
            baseline = by_variant["baseline"].get(query, set())
            candidate = by_variant["candidate"].get(query, set())
            if len(baseline) != 1 or len(candidate) != 1 or baseline != candidate:
                parity_failures.append(
                    {
                        "phase": phase,
                        "query": query,
                        "baseline_variants": len(baseline),
                        "candidate_variants": len(candidate),
                        "equal": baseline == candidate,
                    }
                )
            else:
                baseline_raw = raw_by_variant["baseline"].get(query, set())
                candidate_raw = raw_by_variant["candidate"].get(query, set())
                if (
                    len(baseline_raw) != 1
                    or len(candidate_raw) != 1
                    or baseline_raw != candidate_raw
                ):
                    cutoff_ties.append(
                        {
                            "phase": phase,
                            "query": query,
                            "baseline_variants": len(baseline_raw),
                            "candidate_variants": len(candidate_raw),
                        }
                    )

    phases: dict[str, Any] = {}
    for phase in ("cold", "warm", "throughput"):
        baseline = summarize(data[(phase, "baseline")])
        candidate = summarize(data[(phase, "candidate")])
        phases[phase] = {
            "baseline": baseline,
            "candidate": candidate,
            "candidate_vs_baseline_percent": {
                "latency_mean": relative_change(
                    baseline["latency_mean_ms"], candidate["latency_mean_ms"]
                ),
                "latency_p50": relative_change(
                    baseline["latency_p50_ms"], candidate["latency_p50_ms"]
                ),
                "latency_p95": relative_change(
                    baseline["latency_p95_ms"], candidate["latency_p95_ms"]
                ),
                "throughput": relative_change(
                    baseline["throughput_qps_mean"],
                    candidate["throughput_qps_mean"],
                ),
                "peak_rss": relative_change(
                    baseline["peak_rss_kib_max"], candidate["peak_rss_kib_max"]
                ),
            },
        }

    summary = {
        "parity": {
            "ok": not parity_failures,
            "failures": parity_failures,
            "cutoff_ties": cutoff_ties,
        },
        "phases": phases,
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not parity_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
