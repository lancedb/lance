#!/usr/bin/env python3
"""Validate JSONL records and report paired stable-row-address comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run import (  # noqa: E402
    FORMATS,
    OPERATIONS,
    SHA256_PATTERN,
    SHA_PATTERN,
    run_sidecar_path,
    validate_record,
)


DEFAULT_BOOTSTRAP_SAMPLES = 10_000
MIN_PAIRED_ROUNDS = {"smoke": 3, "release": 10}
RUN_SIDECAR_FIELDS = frozenset(
    {
        "schema_version",
        "suite",
        "run_id",
        "created_at_utc",
        "commit",
        "host",
        "seed",
        "mode",
        "storage",
        "formats",
        "operations",
        "rounds",
        "rows",
        "rows_per_fragment",
        "take_count",
        "dataset_root",
        "output_jsonl",
        "executable",
        "data_retention",
        "take_ids_root",
        "policy_version",
        "policy_sha256",
        "policy_canonical_json",
        "policy",
    }
)


@dataclass(frozen=True)
class SourceRecord:
    source: str
    line: int
    value: dict[str, Any]


@dataclass(frozen=True)
class Comparison:
    run_id: str
    storage: str
    operation: str
    metric: str
    baseline: str
    paired_rounds: int
    ratio: float
    ci_low: float
    ci_high: float
    threshold: float
    strict: bool

    @property
    def gate_passes(self) -> bool:
        return (
            self.ci_high < self.threshold
            if self.strict
            else self.ci_high <= self.threshold
        )


@dataclass(frozen=True)
class ReportResult:
    markdown: str
    verdict: str

    def exit_code(self, enforce_gates: bool) -> int:
        if self.verdict == "INCOMPLETE":
            return 2
        if self.verdict == "FAIL":
            return 1
        return 0


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def record_metric(record: dict[str, Any], metric: str) -> int | None:
    if metric == "duration_ns":
        return record["duration_ns"]
    if metric == "logical_requests":
        fields = (
            "get_requests",
            "head_requests",
            "list_requests",
            "put_requests",
            "delete_requests",
        )
    elif metric == "io_bytes":
        fields = ("read_bytes", "write_bytes")
    elif metric == "peak_rss_bytes":
        return record["peak_rss_bytes"]
    elif metric == "actual_http_attempts":
        fields = (
            "actual_get_attempts",
            "actual_head_attempts",
            "actual_list_attempts",
            "actual_put_attempts",
            "actual_delete_attempts",
        )
    else:
        raise ValueError(f"unsupported report metric: {metric}")
    values = [record[field] for field in fields]
    if any(value is None for value in values):
        return None
    return sum(values)


def paired_bootstrap_ratio(
    candidate: Sequence[int],
    baseline: Sequence[int],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> tuple[float, float, float]:
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("paired bootstrap requires equal non-empty samples")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if any(value <= 0 for value in candidate) or any(value <= 0 for value in baseline):
        raise ValueError("durations must be positive")
    ratios = [left / right for left, right in zip(candidate, baseline, strict=True)]
    point = float(statistics.median(ratios))
    rng = random.Random(seed)
    bootstrapped = []
    for _ in range(samples):
        draw = [ratios[rng.randrange(len(ratios))] for _ in ratios]
        bootstrapped.append(float(statistics.median(draw)))
    return point, percentile(bootstrapped, 0.025), percentile(bootstrapped, 0.975)


def load_records(paths: Iterable[Path]) -> tuple[list[SourceRecord], list[str]]:
    records: list[SourceRecord] = []
    issues: list[str] = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            issues.append(f"{path}: {error}")
            continue
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                issues.append(f"{path}:{line_number}: blank lines are not allowed")
                continue
            try:
                value = json.loads(line)
                validate_record(value)
            except (json.JSONDecodeError, ValueError) as error:
                issues.append(f"{path}:{line_number}: {error}")
                continue
            records.append(SourceRecord(str(path), line_number, value))
    return records, issues


def validate_run_sidecar(value: Any, source: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("run sidecar must be a JSON object")
    missing = sorted(RUN_SIDECAR_FIELDS - value.keys())
    unknown = sorted(value.keys() - RUN_SIDECAR_FIELDS)
    if missing or unknown:
        raise ValueError(
            f"run sidecar schema mismatch: missing={missing}, unknown={unknown}"
        )
    if value["schema_version"] != 1 or value["suite"] != "stable_row_address_e2e":
        raise ValueError("run sidecar schema_version or suite is unsupported")
    for field in (
        "run_id",
        "created_at_utc",
        "commit",
        "host",
        "mode",
        "storage",
        "dataset_root",
        "output_jsonl",
        "executable",
        "data_retention",
        "take_ids_root",
        "policy_sha256",
        "policy_canonical_json",
    ):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"run sidecar {field} must be a non-empty string")
    if SHA_PATTERN.fullmatch(value["commit"]) is None:
        raise ValueError("run sidecar commit must be a full lowercase Git SHA")
    if SHA256_PATTERN.fullmatch(value["policy_sha256"]) is None:
        raise ValueError("run sidecar policy_sha256 must be a SHA-256 digest")
    if value["mode"] not in MIN_PAIRED_ROUNDS:
        raise ValueError("run sidecar mode must be smoke or release")
    if value["storage"] not in {"ebs", "s3"}:
        raise ValueError("run sidecar storage must be ebs or s3")
    if value["formats"] != list(FORMATS):
        raise ValueError("run sidecar formats do not match the frozen comparison set")
    if value["operations"] != list(OPERATIONS):
        raise ValueError("run sidecar operations do not match the complete suite")
    for field in (
        "seed",
        "rounds",
        "rows",
        "rows_per_fragment",
        "take_count",
        "policy_version",
    ):
        if not isinstance(value[field], int) or isinstance(value[field], bool):
            raise ValueError(f"run sidecar {field} must be an integer")
    if value["rounds"] < MIN_PAIRED_ROUNDS[value["mode"]]:
        raise ValueError(
            f"run sidecar {value['mode']} mode has only {value['rounds']} rounds"
        )
    if not isinstance(value["policy"], dict):
        raise ValueError("run sidecar policy must be a JSON object")
    canonical = json.dumps(
        value["policy"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    if canonical != value["policy_canonical_json"]:
        raise ValueError("run sidecar policy_canonical_json does not match policy")
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if digest != value["policy_sha256"]:
        raise ValueError("run sidecar policy_sha256 does not match canonical policy")
    if value["policy_version"] != value["policy"].get("schema_version"):
        raise ValueError("run sidecar policy_version does not match policy")
    if value["data_retention"] != "preserve":
        raise ValueError("run sidecar must record data_retention=preserve")
    return value


def load_run_sidecars(
    paths: Iterable[Path], records: Sequence[SourceRecord]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    sidecars: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    for path in paths:
        sidecar_path = run_sidecar_path(path)
        try:
            raw = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar = validate_run_sidecar(raw, sidecar_path)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            issues.append(f"{sidecar_path}: {error}")
            continue
        sidecars[str(path)] = sidecar
        take_ids_root = sidecar_path.parent / sidecar["take_ids_root"]
        missing_take_ids = [
            str(take_ids_root / f"round-{round_index:03d}" / f"{format_name}.json")
            for round_index in range(sidecar["rounds"])
            for format_name in FORMATS
            if not (
                take_ids_root / f"round-{round_index:03d}" / f"{format_name}.json"
            ).is_file()
        ]
        if missing_take_ids:
            issues.append(
                f"{sidecar_path}: missing {len(missing_take_ids)} take-ID artifact(s); "
                f"first={missing_take_ids[0]}"
            )

    provenance_fields = (
        "run_id",
        "commit",
        "host",
        "seed",
        "mode",
        "storage",
        "rows",
        "rows_per_fragment",
        "take_count",
        "policy_version",
        "policy_sha256",
    )
    for record in records:
        sidecar = sidecars.get(record.source)
        if sidecar is None:
            continue
        mismatches = {
            field: (sidecar[field], record.value[field])
            for field in provenance_fields
            if sidecar[field] != record.value[field]
        }
        if mismatches:
            issues.append(
                f"{record.source}:{record.line}: record does not match run sidecar: "
                f"{mismatches}"
            )
    return sidecars, issues


def _scope(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["run_id"],
        record["commit"],
        record["host"],
        record["seed"],
        record["policy_sha256"],
        record["policy_version"],
        record["mode"],
        record["storage"],
        record["rows"],
        record["rows_per_fragment"],
        record["take_count"],
    )


def analyze(
    records: Sequence[SourceRecord],
    *,
    mode: str = "release",
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    expected_commit: str | None = None,
    enforce_gates: bool = False,
    initial_issues: Sequence[str] = (),
) -> ReportResult:
    issues = list(initial_issues)
    failures: list[str] = []
    comparisons: list[Comparison] = []
    if mode not in MIN_PAIRED_ROUNDS:
        raise ValueError("mode must be smoke or release")
    if expected_commit is not None and SHA_PATTERN.fullmatch(expected_commit) is None:
        raise ValueError(
            "expected_commit must be a full lowercase 40-character Git SHA"
        )
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not records:
        issues.append("no valid benchmark records")

    grouped: dict[tuple[Any, ...], list[SourceRecord]] = defaultdict(list)
    for record in records:
        if record.value["mode"] != mode:
            issues.append(
                f"{record.source}:{record.line}: record mode {record.value['mode']} "
                f"does not match report mode {mode}"
            )
        if expected_commit is not None and record.value["commit"] != expected_commit:
            issues.append(
                f"{record.source}:{record.line}: commit {record.value['commit']} "
                f"does not match expected {expected_commit}"
            )
        grouped[_scope(record.value)].append(record)

    for scope, scoped_records in sorted(grouped.items(), key=lambda item: item[0]):
        run_id = scope[0]
        storage = scope[7]
        by_operation: dict[str, list[SourceRecord]] = defaultdict(list)
        for record in scoped_records:
            by_operation[record.value["operation"]].append(record)
        missing_operations = sorted(set(OPERATIONS) - by_operation.keys())
        if missing_operations:
            issues.append(
                f"{run_id}: missing operation(s): {', '.join(missing_operations)}"
            )

        for operation, operation_records in sorted(by_operation.items()):
            by_round: dict[int, list[SourceRecord]] = defaultdict(list)
            for record in operation_records:
                by_round[record.value["round"]].append(record)
            rounds = sorted(by_round)
            minimum_rounds = MIN_PAIRED_ROUNDS[mode]
            if len(rounds) < minimum_rounds:
                issues.append(
                    f"{run_id}/{operation}: expected at least {minimum_rounds} "
                    f"paired rounds, found {len(rounds)}"
                )
                continue
            if rounds != list(range(rounds[-1] + 1)):
                issues.append(
                    f"{run_id}/{operation}: rounds are not contiguous from zero"
                )

            metric_names = [
                "duration_ns",
                "logical_requests",
                "io_bytes",
                "peak_rss_bytes",
            ]
            if storage == "s3":
                metric_names.append("actual_http_attempts")
            metric_values: dict[str, dict[str, list[int]]] = {
                metric: {name: [] for name in FORMATS} for metric in metric_names
            }
            format_positions: dict[str, Counter[int]] = {
                name: Counter() for name in FORMATS
            }
            operation_complete = True
            operation_failed = False
            for round_index in rounds:
                round_records = by_round[round_index]
                pair_ids = {record.value["pair_id"] for record in round_records}
                formats = [record.value["format"] for record in round_records]
                positions = [record.value["order_index"] for record in round_records]
                if len(pair_ids) != 1:
                    issues.append(
                        f"{run_id}/{operation}/round-{round_index}: pair_id mismatch"
                    )
                    operation_complete = False
                if Counter(formats) != Counter(FORMATS):
                    issues.append(
                        f"{run_id}/{operation}/round-{round_index}: expected one record "
                        f"per format, found {sorted(formats)}"
                    )
                    operation_complete = False
                if sorted(positions) != list(range(len(FORMATS))):
                    issues.append(
                        f"{run_id}/{operation}/round-{round_index}: order_index is not "
                        "a 0..2 permutation"
                    )
                    operation_complete = False
                if not operation_complete:
                    continue
                ordered = {
                    record.value["format"]: record.value for record in round_records
                }
                for format_name in FORMATS:
                    value = ordered[format_name]
                    format_positions[format_name][value["order_index"]] += 1
                    if value["status"] != "ok":
                        failures.append(
                            f"{run_id}/{operation}/round-{round_index}/{format_name}: "
                            f"{value['error']}"
                        )
                        operation_failed = True
                    for metric in metric_names:
                        measured = record_metric(value, metric)
                        if measured is None:
                            issues.append(
                                f"{run_id}/{operation}/round-{round_index}/{format_name}: "
                                f"metric {metric} is unavailable"
                            )
                            operation_complete = False
                        elif measured <= 0:
                            issues.append(
                                f"{run_id}/{operation}/round-{round_index}/{format_name}: "
                                f"metric {metric} must be positive, got {measured}"
                            )
                            operation_complete = False
                        else:
                            metric_values[metric][format_name].append(measured)

            if not operation_complete:
                continue
            for format_name, position_counts in format_positions.items():
                counts = [position_counts[index] for index in range(len(FORMATS))]
                if max(counts) - min(counts) > 1:
                    issues.append(
                        f"{run_id}/{operation}: {format_name} order positions are not "
                        f"balanced: {counts}"
                    )
                    operation_complete = False
            if not operation_complete or operation_failed:
                continue

            for metric in metric_names:
                for baseline, threshold, strict in (
                    ("v22_no_stable", 1.05, False),
                    ("v22_stable", 1.0, True),
                ):
                    digest = hashlib.sha256(
                        f"{run_id}\0{operation}\0{metric}\0{baseline}".encode("utf-8")
                    ).digest()
                    bootstrap_seed = int.from_bytes(digest[:8], "big")
                    ratio, ci_low, ci_high = paired_bootstrap_ratio(
                        metric_values[metric]["v23_logical"],
                        metric_values[metric][baseline],
                        samples=bootstrap_samples,
                        seed=bootstrap_seed,
                    )
                    comparisons.append(
                        Comparison(
                            run_id=run_id,
                            storage=storage,
                            operation=operation,
                            metric=metric,
                            baseline=baseline,
                            paired_rounds=len(rounds),
                            ratio=ratio,
                            ci_low=ci_low,
                            ci_high=ci_high,
                            threshold=threshold,
                            strict=strict,
                        )
                    )

    observed_gate_failures = [
        comparison for comparison in comparisons if not comparison.gate_passes
    ]
    if issues:
        verdict = "INCOMPLETE"
    elif failures or (enforce_gates and observed_gate_failures):
        verdict = "FAIL"
    else:
        verdict = "PASS"

    lines = [
        "# Stable Row Address Benchmark Report",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"Evidence mode: **{mode}** (minimum paired rounds: {MIN_PAIRED_ROUNDS[mode]})",
        "",
        f"Paired bootstrap resamples: {bootstrap_samples}",
        "",
        "| Run | Storage | Operation | Metric | Baseline | Rounds | v2.3 / baseline | 95% CI | Contract | Observed |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for comparison in comparisons:
        operator = "<" if comparison.strict else "≤"
        lines.append(
            "| {run} | {storage} | {operation} | {metric} | {baseline} | {rounds} | "
            "{ratio:.4f} | [{low:.4f}, {high:.4f}] | upper CI {operator} "
            "{threshold:.2f} | {observed} |".format(
                run=comparison.run_id,
                storage=comparison.storage,
                operation=comparison.operation,
                metric=comparison.metric,
                baseline=comparison.baseline,
                rounds=comparison.paired_rounds,
                ratio=comparison.ratio,
                low=comparison.ci_low,
                high=comparison.ci_high,
                operator=operator,
                threshold=comparison.threshold,
                observed="PASS" if comparison.gate_passes else "FAIL",
            )
        )
    if not comparisons:
        lines.append("| — | — | — | — | — | — | — | — | — | INCOMPLETE |")

    if issues:
        lines.extend(("", "## Incomplete inputs", ""))
        lines.extend(f"- {issue}" for issue in issues)
    if failures:
        lines.extend(("", "## Failed operations", ""))
        lines.extend(f"- {failure}" for failure in failures)
    if observed_gate_failures and not enforce_gates:
        lines.extend(
            (
                "",
                "Observed contract failures are diagnostic because gates were not enforced. "
                "Use `--enforce-gates` for a release-gate verdict.",
            )
        )
    return ReportResult("\n".join(lines) + "\n", verdict)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--mode", choices=("smoke", "release"), default="release")
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--enforce-gates", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.bootstrap_samples < DEFAULT_BOOTSTRAP_SAMPLES:
        raise ValueError(
            f"CLI reports require at least {DEFAULT_BOOTSTRAP_SAMPLES} bootstrap samples"
        )
    records, issues = load_records(args.inputs)
    _, sidecar_issues = load_run_sidecars(args.inputs, records)
    issues.extend(sidecar_issues)
    result = analyze(
        records,
        mode=args.mode,
        bootstrap_samples=args.bootstrap_samples,
        expected_commit=args.expected_commit,
        enforce_gates=args.enforce_gates,
        initial_issues=issues,
    )
    if args.output:
        args.output.write_text(result.markdown, encoding="utf-8")
    else:
        sys.stdout.write(result.markdown)
    return result.exit_code(args.enforce_gates)


if __name__ == "__main__":
    raise SystemExit(main())
