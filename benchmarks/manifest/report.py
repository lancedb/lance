#!/usr/bin/env python3
"""Generate the Lance manifest benchmark gate report from JSONL records."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence


SUPPORTED_SCHEMA_VERSION = 2
SUPPORTED_EXPLANATION_SCHEMA_VERSION = 2
MIN_ROUNDS = 5
FRAGMENT_SIZES = (1_000, 100_000, 1_000_000, 10_000_000)
FORMATS = ("protobuf", "lance")
SCENARIOS = ("S1", "S2")
STORAGES = ("memory", "ebs", "s3")
CODEC_OPERATIONS = ("encode", "decode", "size", "decode_rss")
E2E_OPERATIONS = ("open", "commit", "conflict_retry", "time_travel")
OPERATIONS = CODEC_OPERATIONS + E2E_OPERATIONS
SUCCESS_STATUSES = frozenset(("pass", "passed", "ok", "success"))
FAILURE_STATUSES = frozenset(("fail", "failed", "error"))
GIT_SHA_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")

REQUIRED_FIELDS = (
    "schema_version",
    "suite",
    "scenario",
    "fragments",
    "format",
    "storage",
    "operation",
    "round",
    "wall_ns",
    "bytes",
    "peak_rss_bytes",
    "get_requests",
    "put_requests",
    "read_bytes",
    "write_bytes",
    "status",
    "error",
    "commit",
    "seed",
    "host",
)
INTEGER_FIELDS = (
    "schema_version",
    "fragments",
    "round",
    "wall_ns",
    "bytes",
    "peak_rss_bytes",
    "get_requests",
    "put_requests",
    "read_bytes",
    "write_bytes",
    "seed",
)


@dataclass(frozen=True, order=True)
class RunScope:
    schema_version: int
    suite: str
    commit: str
    seed: int
    host: str


@dataclass(frozen=True, order=True)
class CaseKey:
    schema_version: int
    suite: str
    scenario: str
    fragments: int
    storage: str
    operation: str
    commit: str
    seed: int
    host: str

    @property
    def scope(self) -> RunScope:
        return RunScope(
            self.schema_version, self.suite, self.commit, self.seed, self.host
        )


@dataclass(frozen=True)
class Record:
    key: CaseKey
    format_name: str
    round: int
    wall_ns: int
    bytes: int
    peak_rss_bytes: int
    get_requests: int
    put_requests: int
    read_bytes: int
    write_bytes: int
    status: str
    error: str
    source: str
    line: int

    def metric(self, name: str) -> int:
        return int(getattr(self, name))


@dataclass(frozen=True)
class ValidationIssue:
    source: str
    line: int
    message: str
    key: CaseKey | None = None

    @property
    def location(self) -> str:
        return f"{self.source}:{self.line}" if self.line else self.source


@dataclass
class InputStats:
    path: str
    lines: int = 0
    valid: int = 0
    invalid: int = 0


@dataclass
class ReportOutcome:
    markdown: str
    verdict: str

    @property
    def exit_code(self) -> int:
        return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[self.verdict]


@dataclass
class CaseAggregate:
    key: CaseKey
    records_by_format: dict[str, list[Record]] = field(
        default_factory=lambda: defaultdict(list)
    )
    incomplete_reasons: list[str] = field(default_factory=list)

    @property
    def failed_records(self) -> list[Record]:
        return [
            record
            for records in self.records_by_format.values()
            for record in records
            if record.status not in SUCCESS_STATUSES or record.error
        ]

    @property
    def state(self) -> str:
        if self.incomplete_reasons:
            return "INCOMPLETE"
        if self.failed_records:
            return "FAIL"
        return "PASS"

    def median(self, format_name: str, metric: str) -> float:
        return float(
            statistics.median(
                record.metric(metric) for record in self.records_by_format[format_name]
            )
        )


@dataclass(frozen=True)
class GateRow:
    scope: RunScope
    scenario: str
    fragments: int
    storage: str
    operation: str
    metric: str
    protobuf_value: str
    lance_value: str
    criterion: str
    verdict: str
    note: str = ""


@dataclass(frozen=True)
class ScalingRow:
    scope: RunScope
    scenario: str
    format_name: str
    operation: str
    metric: str
    small_value: str
    large_value: str
    growth: str
    deviation: str
    verdict: str
    note: str = ""


@dataclass(frozen=True)
class GateSelector:
    gate: str
    schema_version: int
    suite: str
    commit: str
    seed: int
    host: str
    scenario: str
    operation: str
    metric: str
    fragments: int | None = None
    storage: str | None = None
    format_name: str | None = None


@dataclass(frozen=True)
class GateExplanation:
    mechanism: str
    fix: str


class ExplanationError(ValueError):
    """Raised when a gate-explanation sidecar is invalid or stale."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_git_sha(value: Any) -> bool:
    return isinstance(value, str) and GIT_SHA_PATTERN.fullmatch(value) is not None


def _expected_commit_argument(value: str) -> str:
    if not _is_git_sha(value):
        raise argparse.ArgumentTypeError(
            "expected commit must be a full lowercase 40- or 64-character Git SHA"
        )
    return value


def _require_exact_keys(
    value: dict[str, Any], expected: set[str], context: str
) -> None:
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    details = []
    if missing:
        details.append(f"missing field(s): {', '.join(missing)}")
    if unknown:
        details.append(f"unknown field(s): {', '.join(unknown)}")
    if details:
        raise ExplanationError(f"{context} is malformed: {'; '.join(details)}")


def _selector_text(selector: GateSelector) -> str:
    fields: dict[str, object] = {
        "gate": selector.gate,
        "schema_version": selector.schema_version,
        "suite": selector.suite,
        "commit": selector.commit,
        "seed": selector.seed,
        "host": selector.host,
        "scenario": selector.scenario,
        "operation": selector.operation,
        "metric": selector.metric,
    }
    if selector.gate == "case":
        fields["fragments"] = selector.fragments
        fields["storage"] = selector.storage
    else:
        fields["format"] = selector.format_name
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


def _parse_selector(raw: Any, index: int) -> GateSelector:
    context = f"explanations[{index}].selector"
    if not isinstance(raw, dict):
        raise ExplanationError(f"{context} must be an object")
    gate = raw.get("gate")
    if gate not in ("case", "scaling"):
        raise ExplanationError(f"{context}.gate must be case or scaling")
    common = {
        "gate",
        "schema_version",
        "suite",
        "commit",
        "seed",
        "host",
        "scenario",
        "operation",
        "metric",
    }
    expected = common | ({"fragments", "storage"} if gate == "case" else {"format"})
    _require_exact_keys(raw, expected, context)

    if (
        not _is_int(raw["schema_version"])
        or raw["schema_version"] != SUPPORTED_SCHEMA_VERSION
    ):
        raise ExplanationError(
            f"{context}.schema_version must be {SUPPORTED_SCHEMA_VERSION}"
        )
    if raw["suite"] not in ("codec", "e2e"):
        raise ExplanationError(f"{context}.suite must be codec or e2e")
    if gate == "scaling" and raw["suite"] != "codec":
        raise ExplanationError(f"{context}: scaling selectors require suite codec")
    if raw["scenario"] not in SCENARIOS:
        raise ExplanationError(f"{context}.scenario must be S1 or S2")
    for field_name in ("commit", "host", "operation", "metric"):
        value = raw[field_name]
        if not isinstance(value, str) or not value.strip():
            raise ExplanationError(f"{context}.{field_name} must be a non-empty string")
    if not _is_git_sha(raw["commit"]):
        raise ExplanationError(
            f"{context}.commit must be a full lowercase 40- or 64-character Git SHA"
        )
    if not _is_int(raw["seed"]) or raw["seed"] < 0:
        raise ExplanationError(f"{context}.seed must be a non-negative integer")

    fragments = None
    storage = None
    format_name = None
    if gate == "case":
        if not _is_int(raw["fragments"]) or raw["fragments"] <= 0:
            raise ExplanationError(f"{context}.fragments must be a positive integer")
        if raw["storage"] not in STORAGES:
            raise ExplanationError(f"{context}.storage must be memory, ebs, or s3")
        fragments = raw["fragments"]
        storage = raw["storage"]
    else:
        if raw["format"] not in FORMATS:
            raise ExplanationError(f"{context}.format must be protobuf or lance")
        format_name = raw["format"]

    return GateSelector(
        gate=gate,
        schema_version=raw["schema_version"],
        suite=raw["suite"],
        commit=raw["commit"],
        seed=raw["seed"],
        host=raw["host"],
        scenario=raw["scenario"],
        operation=raw["operation"],
        metric=raw["metric"],
        fragments=fragments,
        storage=storage,
        format_name=format_name,
    )


def _load_explanations(path_value: str | Path) -> dict[GateSelector, GateExplanation]:
    path = Path(path_value).expanduser().resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ExplanationError(
            f"cannot read explanation sidecar {path}: {error}"
        ) from error
    except UnicodeDecodeError as error:
        raise ExplanationError(
            f"explanation sidecar {path} is not valid UTF-8: {error.reason}"
        ) from error
    except json.JSONDecodeError as error:
        raise ExplanationError(
            f"explanation sidecar {path} contains invalid JSON: {error.msg}"
        ) from error
    if not isinstance(raw, dict):
        raise ExplanationError("explanation sidecar must be a JSON object")
    _require_exact_keys(raw, {"schema_version", "explanations"}, "sidecar")
    if (
        not _is_int(raw["schema_version"])
        or raw["schema_version"] != SUPPORTED_EXPLANATION_SCHEMA_VERSION
    ):
        raise ExplanationError(
            f"sidecar.schema_version must be {SUPPORTED_EXPLANATION_SCHEMA_VERSION}"
        )
    if not isinstance(raw["explanations"], list):
        raise ExplanationError("sidecar.explanations must be an array")

    explanations: dict[GateSelector, GateExplanation] = {}
    for index, item in enumerate(raw["explanations"]):
        context = f"explanations[{index}]"
        if not isinstance(item, dict):
            raise ExplanationError(f"{context} must be an object")
        _require_exact_keys(item, {"selector", "mechanism", "fix"}, context)
        selector = _parse_selector(item["selector"], index)
        for field_name in ("mechanism", "fix"):
            value = item[field_name]
            if not isinstance(value, str) or not value.strip():
                raise ExplanationError(
                    f"{context}.{field_name} must be a non-empty string"
                )
        if selector in explanations:
            raise ExplanationError(
                f"duplicate explanation selector: {_selector_text(selector)}"
            )
        explanations[selector] = GateExplanation(
            item["mechanism"].strip(), item["fix"].strip()
        )
    return explanations


def _case_hint(raw: dict[str, Any]) -> CaseKey | None:
    fields = (
        "schema_version",
        "suite",
        "scenario",
        "fragments",
        "storage",
        "operation",
        "commit",
        "seed",
        "host",
    )
    if any(field not in raw for field in fields):
        return None
    if not all(
        _is_int(raw[field]) for field in ("schema_version", "fragments", "seed")
    ):
        return None
    if not all(
        isinstance(raw[field], str)
        for field in ("suite", "scenario", "storage", "operation", "commit", "host")
    ):
        return None
    return CaseKey(
        raw["schema_version"],
        raw["suite"],
        raw["scenario"],
        raw["fragments"],
        raw["storage"],
        raw["operation"],
        raw["commit"],
        raw["seed"],
        raw["host"],
    )


def _parse_record(
    raw: dict[str, Any], source: str, line: int, expected_commit: str
) -> tuple[Record | None, list[ValidationIssue]]:
    key_hint = _case_hint(raw)
    errors: list[str] = []
    missing = [field for field in REQUIRED_FIELDS if field not in raw]
    if missing:
        errors.append(f"missing required field(s): {', '.join(missing)}")
    if not missing:
        for field in INTEGER_FIELDS:
            if not _is_int(raw[field]) or raw[field] < 0:
                errors.append(f"{field} must be a non-negative integer")
        if raw["schema_version"] != SUPPORTED_SCHEMA_VERSION:
            errors.append(
                f"unsupported schema_version {raw['schema_version']}; "
                f"expected {SUPPORTED_SCHEMA_VERSION}"
            )
        if _is_int(raw["fragments"]) and raw["fragments"] <= 0:
            errors.append("fragments must be greater than zero")
        if raw["suite"] not in ("codec", "e2e"):
            errors.append("suite must be codec or e2e")
        if raw["scenario"] not in SCENARIOS:
            errors.append("scenario must be S1 or S2")
        if raw["format"] not in FORMATS:
            errors.append("format must be protobuf or lance")
        if raw["storage"] not in STORAGES:
            errors.append("storage must be memory, ebs, or s3")
        if raw["operation"] not in OPERATIONS:
            errors.append(f"unknown operation {raw['operation']!r}")
        if raw["suite"] == "codec":
            if raw["operation"] not in CODEC_OPERATIONS:
                errors.append("codec records must use a codec operation")
            if raw["storage"] != "memory":
                errors.append("codec records must use memory storage")
        if raw["suite"] == "e2e":
            if raw["operation"] not in E2E_OPERATIONS:
                errors.append("e2e records must use an e2e operation")
            if raw["storage"] not in ("ebs", "s3"):
                errors.append("e2e records must use ebs or s3 storage")
        if not _is_git_sha(raw["commit"]):
            errors.append("commit must be a full lowercase 40- or 64-character Git SHA")
        elif raw["commit"] != expected_commit:
            errors.append(
                f"commit {raw['commit']} does not match expected commit {expected_commit}"
            )
        if not isinstance(raw["host"], str) or not raw["host"].strip():
            errors.append("host must be a non-empty string")
        if not isinstance(raw["status"], str):
            errors.append("status must be a string")
        elif raw["status"].lower() not in SUCCESS_STATUSES | FAILURE_STATUSES:
            errors.append(f"unknown status {raw['status']!r}")
        if raw["error"] is not None and not isinstance(raw["error"], str):
            errors.append("error must be a string or null")
        metrics_are_valid_integers = all(
            _is_int(raw[field]) and raw[field] >= 0
            for field in (
                "bytes",
                "get_requests",
                "put_requests",
                "read_bytes",
                "write_bytes",
            )
        )
        if metrics_are_valid_integers and raw["suite"] == "e2e":
            expected_bytes = raw["read_bytes"] + raw["write_bytes"]
            if raw["bytes"] != expected_bytes:
                errors.append(
                    "e2e bytes must equal read_bytes + write_bytes "
                    f"({raw['bytes']} != {expected_bytes})"
                )
        if metrics_are_valid_integers and raw["suite"] == "codec":
            if raw["get_requests"] != 0 or raw["put_requests"] != 0:
                errors.append("codec get_requests and put_requests must both be zero")
            operation = raw["operation"]
            if operation == "encode":
                if raw["read_bytes"] != 0 or raw["bytes"] != raw["write_bytes"]:
                    errors.append(
                        "codec encode requires read_bytes == 0 and bytes == write_bytes"
                    )
            elif operation == "size":
                if raw["read_bytes"] != 0 or raw["write_bytes"] != 0:
                    errors.append(
                        "codec size requires read_bytes == 0 and write_bytes == 0"
                    )
            elif operation in ("decode", "decode_rss"):
                if raw["write_bytes"] != 0 or raw["bytes"] != raw["read_bytes"]:
                    errors.append(
                        f"codec {operation} requires write_bytes == 0 and bytes == read_bytes"
                    )
    if errors:
        return None, [
            ValidationIssue(source, line, error, key_hint) for error in errors
        ]

    key = _case_hint(raw)
    assert key is not None
    return (
        Record(
            key=key,
            format_name=raw["format"],
            round=raw["round"],
            wall_ns=raw["wall_ns"],
            bytes=raw["bytes"],
            peak_rss_bytes=raw["peak_rss_bytes"],
            get_requests=raw["get_requests"],
            put_requests=raw["put_requests"],
            read_bytes=raw["read_bytes"],
            write_bytes=raw["write_bytes"],
            status=raw["status"].lower(),
            error=(raw["error"] or "").strip(),
            source=source,
            line=line,
        ),
        [],
    )


def _load_records(
    paths: Sequence[str | Path], expected_commit: str
) -> tuple[list[Record], list[ValidationIssue], list[InputStats]]:
    records: list[Record] = []
    issues: list[ValidationIssue] = []
    input_stats: list[InputStats] = []
    for input_path in paths:
        path = Path(input_path).expanduser().resolve()
        stats = InputStats(str(path))
        input_stats.append(stats)
        try:
            handle = path.open("rb")
        except OSError as error:
            stats.invalid += 1
            issues.append(ValidationIssue(str(path), 0, f"cannot read input: {error}"))
            continue
        with handle:
            for line_number, raw_line in enumerate(handle, 1):
                stats.lines += 1
                try:
                    text = raw_line.decode("utf-8")
                except UnicodeDecodeError as error:
                    stats.invalid += 1
                    issues.append(
                        ValidationIssue(
                            str(path),
                            line_number,
                            f"invalid UTF-8: {error.reason}",
                        )
                    )
                    continue
                if not text.strip():
                    stats.invalid += 1
                    issues.append(
                        ValidationIssue(str(path), line_number, "blank JSONL record")
                    )
                    continue
                try:
                    raw = json.loads(text)
                except json.JSONDecodeError as error:
                    stats.invalid += 1
                    issues.append(
                        ValidationIssue(
                            str(path), line_number, f"invalid JSON: {error.msg}"
                        )
                    )
                    continue
                if not isinstance(raw, dict):
                    stats.invalid += 1
                    issues.append(
                        ValidationIssue(
                            str(path), line_number, "JSONL record must be an object"
                        )
                    )
                    continue
                record, record_issues = _parse_record(
                    raw, str(path), line_number, expected_commit
                )
                if record is None:
                    stats.invalid += 1
                    issues.extend(record_issues)
                else:
                    stats.valid += 1
                    records.append(record)
    return records, issues, input_stats


def _expected_keys(scope: RunScope) -> set[CaseKey]:
    keys: set[CaseKey] = set()
    if scope.suite == "codec":
        combinations = (
            (scenario, fragments, "memory", operation)
            for scenario in SCENARIOS
            for fragments in FRAGMENT_SIZES
            for operation in CODEC_OPERATIONS
        )
    else:
        combinations = (
            (scenario, fragments, storage, operation)
            for scenario in SCENARIOS
            for storage in ("ebs", "s3")
            for operation in E2E_OPERATIONS
            for fragments in (
                FRAGMENT_SIZES
                if operation in ("open", "time_travel")
                else (1_000_000, 10_000_000)
            )
        )
    for scenario, fragments, storage, operation in combinations:
        keys.add(
            CaseKey(
                scope.schema_version,
                scope.suite,
                scenario,
                fragments,
                storage,
                operation,
                scope.commit,
                scope.seed,
                scope.host,
            )
        )
    return keys


def _case_id(key: CaseKey) -> str:
    return (
        f"{key.suite}/{key.scenario}/{key.fragments}/{key.storage}/"
        f"{key.operation} commit={key.commit} seed={key.seed} host={key.host}"
    )


def _append_reason(aggregate: CaseAggregate, reason: str) -> None:
    if reason not in aggregate.incomplete_reasons:
        aggregate.incomplete_reasons.append(reason)


def _build_aggregates(
    records: Sequence[Record], input_issues: Sequence[ValidationIssue]
) -> tuple[dict[CaseKey, CaseAggregate], list[CaseAggregate], list[str]]:
    aggregates: dict[CaseKey, CaseAggregate] = {}
    for record in records:
        aggregate = aggregates.setdefault(record.key, CaseAggregate(record.key))
        aggregate.records_by_format[record.format_name].append(record)

    scope_hints = {record.key.scope for record in records}
    scope_hints.update(
        issue.key.scope
        for issue in input_issues
        if issue.key is not None and issue.key.suite in ("codec", "e2e")
    )
    scopes_by_suite = {
        suite: sorted(scope for scope in scope_hints if scope.suite == suite)
        for suite in ("codec", "e2e")
    }

    completeness_issues: list[str] = []
    expected: set[CaseKey] = set()
    for suite in ("codec", "e2e"):
        if not scopes_by_suite[suite]:
            completeness_issues.append(
                f"No valid or recoverable {suite} run scope was found; "
                "the required matrix cannot be evaluated."
            )
            continue
        for scope in scopes_by_suite[suite]:
            expected.update(_expected_keys(scope))

    for key in expected:
        aggregates.setdefault(key, CaseAggregate(key))

    for key, aggregate in aggregates.items():
        if key not in expected:
            reason = "record is outside the required benchmark matrix"
            _append_reason(aggregate, reason)
            completeness_issues.append(f"{_case_id(key)}: {reason}.")

    for issue in input_issues:
        if issue.key is not None and issue.key in aggregates:
            _append_reason(
                aggregates[issue.key], f"invalid input record at {issue.location}"
            )

    for key in sorted(expected):
        aggregate = aggregates[key]
        for format_name in FORMATS:
            format_records = aggregate.records_by_format.get(format_name, [])
            if not format_records:
                _append_reason(aggregate, f"missing format {format_name}")
                continue
            round_counts = Counter(record.round for record in format_records)
            duplicate_rounds = sorted(
                round_number
                for round_number, count in round_counts.items()
                if count > 1
            )
            if duplicate_rounds:
                _append_reason(
                    aggregate,
                    f"format {format_name} has duplicate round(s): "
                    + ", ".join(str(value) for value in duplicate_rounds),
                )
            if len(round_counts) < MIN_ROUNDS:
                _append_reason(
                    aggregate,
                    f"format {format_name} has {len(round_counts)} unique rounds; "
                    f"at least {MIN_ROUNDS} required",
                )
        if all(aggregate.records_by_format.get(name) for name in FORMATS):
            protobuf_rounds = {
                record.round for record in aggregate.records_by_format["protobuf"]
            }
            lance_rounds = {
                record.round for record in aggregate.records_by_format["lance"]
            }
            if protobuf_rounds != lance_rounds:
                protobuf_only = sorted(protobuf_rounds - lance_rounds)
                lance_only = sorted(lance_rounds - protobuf_rounds)
                _append_reason(
                    aggregate,
                    "unpaired format rounds "
                    f"(protobuf-only={protobuf_only}, lance-only={lance_only})",
                )
        for reason in aggregate.incomplete_reasons:
            message = f"{_case_id(key)}: {reason}."
            if message not in completeness_issues:
                completeness_issues.append(message)

    expected_aggregates = [aggregates[key] for key in sorted(expected)]
    return aggregates, expected_aggregates, completeness_issues


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0 if numerator == 0 else math.inf
    return numerator / denominator


def _fmt_number(value: float | int) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        return "inf"
    if numeric.is_integer():
        return f"{int(numeric):,}"
    return f"{numeric:,.2f}"


def _fmt_ratio(value: float) -> str:
    return "inf" if not math.isfinite(value) else f"{value:.3f}x"


def _aggregate_note(aggregate: CaseAggregate) -> str:
    if aggregate.incomplete_reasons:
        return "; ".join(aggregate.incomplete_reasons)
    failures = aggregate.failed_records
    if failures:
        examples = []
        for record in failures[:3]:
            detail = record.error or record.status
            examples.append(f"{record.format_name} round {record.round}: {detail}")
        suffix = "" if len(failures) <= 3 else f"; {len(failures) - 3} more"
        return "; ".join(examples) + suffix
    return ""


def _median_display(aggregate: CaseAggregate, format_name: str, metric: str) -> str:
    if not aggregate.records_by_format.get(format_name):
        return "--"
    return _fmt_number(aggregate.median(format_name, metric))


def _ratio_gate(
    aggregate: CaseAggregate, metric: str, threshold: float, criterion: str
) -> GateRow:
    protobuf_value = _median_display(aggregate, "protobuf", metric)
    lance_value = _median_display(aggregate, "lance", metric)
    if aggregate.state != "PASS":
        return GateRow(
            aggregate.key.scope,
            aggregate.key.scenario,
            aggregate.key.fragments,
            aggregate.key.storage,
            aggregate.key.operation,
            metric,
            protobuf_value,
            lance_value,
            criterion,
            aggregate.state,
            _aggregate_note(aggregate),
        )
    ratio = _safe_ratio(
        aggregate.median("lance", metric), aggregate.median("protobuf", metric)
    )
    verdict = "PASS" if ratio <= threshold else "FAIL"
    note = f"Lance/protobuf = {_fmt_ratio(ratio)}"
    if verdict == "FAIL":
        note += "; gate failed, so a mechanism explanation is required"
    return GateRow(
        aggregate.key.scope,
        aggregate.key.scenario,
        aggregate.key.fragments,
        aggregate.key.storage,
        aggregate.key.operation,
        metric,
        protobuf_value,
        lance_value,
        criterion,
        verdict,
        note,
    )


def _status_gate(aggregate: CaseAggregate, criterion: str) -> GateRow:
    values = {}
    for format_name in FORMATS:
        records = aggregate.records_by_format.get(format_name, [])
        failures = [
            record
            for record in records
            if record.status not in SUCCESS_STATUSES or record.error
        ]
        values[format_name] = (
            "--"
            if not records
            else "all success"
            if not failures
            else f"{len(failures)} failed"
        )
    return GateRow(
        aggregate.key.scope,
        aggregate.key.scenario,
        aggregate.key.fragments,
        aggregate.key.storage,
        aggregate.key.operation,
        "status",
        values["protobuf"],
        values["lance"],
        criterion,
        aggregate.state,
        _aggregate_note(aggregate),
    )


def _codec_gate_rows(expected: Sequence[CaseAggregate]) -> list[GateRow]:
    gate_specs = {
        "encode": ("wall_ns", 1.2, "Lance/protobuf <= 1.20x"),
        "decode": ("wall_ns", 1.0, "Lance/protobuf <= 1.00x"),
        "size": ("bytes", 1.0, "Lance/protobuf <= 1.00x"),
        "decode_rss": (
            "peak_rss_bytes",
            1.0,
            "Lance/protobuf <= 1.00x",
        ),
    }
    rows = []
    for aggregate in expected:
        if aggregate.key.suite != "codec":
            continue
        metric, threshold, criterion = gate_specs[aggregate.key.operation]
        rows.append(_ratio_gate(aggregate, metric, threshold, criterion))
    return rows


def _codec_scaling_rows(
    aggregates: dict[CaseKey, CaseAggregate], expected: Sequence[CaseAggregate]
) -> list[ScalingRow]:
    rows: list[ScalingRow] = []
    codec_scopes = sorted(
        {
            aggregate.key.scope
            for aggregate in expected
            if aggregate.key.suite == "codec"
        }
    )
    expected_growth = FRAGMENT_SIZES[-1] / FRAGMENT_SIZES[0]
    for scope in codec_scopes:
        for scenario in SCENARIOS:
            for format_name in FORMATS:
                for operation, metric in (
                    ("encode", "wall_ns"),
                    ("decode", "wall_ns"),
                    ("size", "bytes"),
                ):
                    small_key = CaseKey(
                        scope.schema_version,
                        "codec",
                        scenario,
                        FRAGMENT_SIZES[0],
                        "memory",
                        operation,
                        scope.commit,
                        scope.seed,
                        scope.host,
                    )
                    large_key = CaseKey(
                        scope.schema_version,
                        "codec",
                        scenario,
                        FRAGMENT_SIZES[-1],
                        "memory",
                        operation,
                        scope.commit,
                        scope.seed,
                        scope.host,
                    )
                    small = aggregates[small_key]
                    large = aggregates[large_key]
                    small_value = _median_display(small, format_name, metric)
                    large_value = _median_display(large, format_name, metric)
                    if "INCOMPLETE" in (small.state, large.state):
                        verdict = "INCOMPLETE"
                        note = "; ".join(
                            value
                            for value in (
                                _aggregate_note(small),
                                _aggregate_note(large),
                            )
                            if value
                        )
                        growth = deviation = "--"
                    elif "FAIL" in (small.state, large.state):
                        verdict = "FAIL"
                        note = "; ".join(
                            value
                            for value in (
                                _aggregate_note(small),
                                _aggregate_note(large),
                            )
                            if value
                        )
                        growth = deviation = "--"
                    else:
                        observed_growth = _safe_ratio(
                            large.median(format_name, metric),
                            small.median(format_name, metric),
                        )
                        if observed_growth <= 0 or not math.isfinite(observed_growth):
                            linear_deviation = math.inf
                        else:
                            linear_deviation = max(
                                observed_growth / expected_growth,
                                expected_growth / observed_growth,
                            )
                        verdict = "PASS" if linear_deviation <= 1.5 else "FAIL"
                        growth = _fmt_ratio(observed_growth)
                        deviation = _fmt_ratio(linear_deviation)
                        note = (
                            ""
                            if verdict == "PASS"
                            else "Deviation exceeds 1.5x; a mechanism explanation is required."
                        )
                    rows.append(
                        ScalingRow(
                            scope,
                            scenario,
                            format_name,
                            operation,
                            metric,
                            small_value,
                            large_value,
                            growth,
                            deviation,
                            verdict,
                            note,
                        )
                    )
    return rows


def _codec_size_ratio(
    target: CaseAggregate, expected: Sequence[CaseAggregate]
) -> tuple[float | None, str]:
    candidates = [
        aggregate
        for aggregate in expected
        if aggregate.key.suite == "codec"
        and aggregate.key.scenario == target.key.scenario
        and aggregate.key.fragments == target.key.fragments
        and aggregate.key.operation == "size"
        and aggregate.key.schema_version == target.key.schema_version
        and aggregate.key.commit == target.key.commit
        and aggregate.key.seed == target.key.seed
    ]
    exact = [
        aggregate for aggregate in candidates if aggregate.key.host == target.key.host
    ]
    if exact:
        reference = exact[0]
        if reference.state != "PASS":
            return None, f"matching codec size case is {reference.state}"
        ratio = _safe_ratio(
            reference.median("lance", "bytes"),
            reference.median("protobuf", "bytes"),
        )
        if not math.isfinite(ratio):
            return None, "matching codec size ratio is not finite"
        return ratio, "matching codec size ratio"

    ready = [aggregate for aggregate in candidates if aggregate.state == "PASS"]
    ratios = [
        _safe_ratio(
            aggregate.median("lance", "bytes"),
            aggregate.median("protobuf", "bytes"),
        )
        for aggregate in ready
    ]
    finite_ratios = [ratio for ratio in ratios if math.isfinite(ratio)]
    if not finite_ratios:
        return None, "no complete codec size case with matching commit and seed"
    first = finite_ratios[0]
    if not all(first == ratio for ratio in finite_ratios[1:]):
        return None, "codec size ratios differ across candidate hosts"
    return first, "codec size ratio from another host (identical across candidates)"


def _request_gate(aggregate: CaseAggregate) -> GateRow:
    metric = "get_requests"
    is_s3 = aggregate.key.storage == "s3"
    display_metric = metric if is_s3 else "read I/O operations"
    operation_name = "GET request" if is_s3 else "tracked read I/O operation"
    protobuf_value = _median_display(aggregate, "protobuf", metric)
    lance_value = _median_display(aggregate, "lance", metric)
    if aggregate.state != "PASS":
        verdict = aggregate.state
        note = _aggregate_note(aggregate)
    else:
        protobuf = aggregate.median("protobuf", metric)
        lance = aggregate.median("lance", metric)
        verdict = "PASS" if lance <= protobuf else "FAIL"
        note = f"Lance/protobuf = {_fmt_ratio(_safe_ratio(lance, protobuf))}"
        if verdict == "FAIL":
            note += f"; {operation_name} increase requires a mechanism explanation"
    return GateRow(
        aggregate.key.scope,
        aggregate.key.scenario,
        aggregate.key.fragments,
        aggregate.key.storage,
        aggregate.key.operation,
        display_metric,
        protobuf_value,
        lance_value,
        f"Lance median {operation_name} count <= protobuf",
        verdict,
        note,
    )


def _io_note(aggregate: CaseAggregate, metrics: Sequence[str]) -> str:
    parts = []
    for metric in metrics:
        display_metric = metric
        if aggregate.key.storage == "ebs":
            if metric == "get_requests":
                display_metric = "read_iops (get_requests field)"
            elif metric == "put_requests":
                display_metric = "write_iops (put_requests field)"
        parts.append(
            f"{display_metric} protobuf={_median_display(aggregate, 'protobuf', metric)}, "
            f"lance={_median_display(aggregate, 'lance', metric)}"
        )
    return "I/O medians: " + "; ".join(parts)


def _add_gate_note(row: GateRow, note: str) -> GateRow:
    combined = f"{row.note}; {note}" if row.note else note
    return GateRow(
        row.scope,
        row.scenario,
        row.fragments,
        row.storage,
        row.operation,
        row.metric,
        row.protobuf_value,
        row.lance_value,
        row.criterion,
        row.verdict,
        combined,
    )


def _conflict_gate(aggregate: CaseAggregate) -> GateRow:
    is_s3 = aggregate.key.storage == "s3"
    if aggregate.state != "PASS":
        verdict = aggregate.state
        note = _aggregate_note(aggregate)
    else:
        protobuf_get = aggregate.median("protobuf", "get_requests")
        lance_get = aggregate.median("lance", "get_requests")
        protobuf_put = aggregate.median("protobuf", "put_requests")
        lance_put = aggregate.median("lance", "put_requests")
        verdict = (
            "PASS"
            if protobuf_get == lance_get and protobuf_put == lance_put
            else "FAIL"
        )
        protobuf_write = aggregate.median("protobuf", "write_bytes")
        lance_write = aggregate.median("lance", "write_bytes")
        note = (
            f"write_bytes protobuf={_fmt_number(protobuf_write)}, "
            f"lance={_fmt_number(lance_write)}, "
            f"ratio={_fmt_ratio(_safe_ratio(lance_write, protobuf_write))}"
        )
        if verdict == "FAIL":
            note += "; retry behavior differs and requires a mechanism explanation"
    protobuf_value = "--"
    lance_value = "--"
    if aggregate.records_by_format.get("protobuf"):
        protobuf_value = (
            f"{_fmt_number(aggregate.median('protobuf', 'get_requests'))}/"
            f"{_fmt_number(aggregate.median('protobuf', 'put_requests'))}"
        )
    if aggregate.records_by_format.get("lance"):
        lance_value = (
            f"{_fmt_number(aggregate.median('lance', 'get_requests'))}/"
            f"{_fmt_number(aggregate.median('lance', 'put_requests'))}"
        )
    return GateRow(
        aggregate.key.scope,
        aggregate.key.scenario,
        aggregate.key.fragments,
        aggregate.key.storage,
        aggregate.key.operation,
        "GET/PUT requests" if is_s3 else "read/write I/O operations",
        protobuf_value,
        lance_value,
        (
            "GET and PUT medians match protobuf"
            if is_s3
            else "Tracked read and write I/O medians match protobuf"
        ),
        verdict,
        note,
    )


def _e2e_gate_rows(
    expected: Sequence[CaseAggregate],
) -> list[GateRow]:
    rows: list[GateRow] = []
    for aggregate in expected:
        if aggregate.key.suite != "e2e":
            continue
        key = aggregate.key
        if key.operation == "open":
            if key.fragments == 1_000:
                rows.append(
                    _add_gate_note(
                        _ratio_gate(
                            aggregate,
                            "wall_ns",
                            1.1,
                            "Lance/protobuf p50 <= 1.10x",
                        ),
                        _io_note(aggregate, ("read_bytes",)),
                    )
                )
                rows.append(_request_gate(aggregate))
            elif key.fragments == 100_000:
                rows.append(
                    _add_gate_note(
                        _status_gate(
                            aggregate,
                            "All paired rounds succeed; no explicit performance threshold",
                        ),
                        _io_note(aggregate, ("get_requests", "read_bytes")),
                    )
                )
            else:
                codec_ratio, reference_note = _codec_size_ratio(aggregate, expected)
                if codec_ratio is None:
                    row = _ratio_gate(
                        aggregate,
                        "wall_ns",
                        math.inf,
                        "Improvement must track the codec size ratio",
                    )
                    rows.append(
                        _add_gate_note(
                            GateRow(
                                row.scope,
                                row.scenario,
                                row.fragments,
                                row.storage,
                                row.operation,
                                row.metric,
                                row.protobuf_value,
                                row.lance_value,
                                row.criterion,
                                "INCOMPLETE",
                                reference_note,
                            ),
                            _io_note(aggregate, ("get_requests", "read_bytes")),
                        )
                    )
                else:
                    threshold = min(1.0, 1.5 * codec_ratio)
                    if key.storage == "s3":
                        threshold = min(threshold, 1.0 / 1.8)
                    criterion = (
                        f"Lance/protobuf <= {threshold:.3f}x "
                        f"(codec size ratio {_fmt_ratio(codec_ratio)}"
                        + (", S3 speedup >= 1.8x" if key.storage == "s3" else "")
                        + ")"
                    )
                    row = _ratio_gate(aggregate, "wall_ns", threshold, criterion)
                    rows.append(
                        _add_gate_note(
                            GateRow(
                                row.scope,
                                row.scenario,
                                row.fragments,
                                row.storage,
                                row.operation,
                                row.metric,
                                row.protobuf_value,
                                row.lance_value,
                                row.criterion,
                                row.verdict,
                                f"{row.note}; {reference_note}"
                                if row.note
                                else reference_note,
                            ),
                            _io_note(aggregate, ("get_requests", "read_bytes")),
                        )
                    )
        elif key.operation == "commit":
            rows.append(
                _add_gate_note(
                    _ratio_gate(
                        aggregate,
                        "wall_ns",
                        1.1,
                        "Lance/protobuf <= 1.10x",
                    ),
                    _io_note(aggregate, ("put_requests", "write_bytes")),
                )
            )
        elif key.operation == "conflict_retry":
            rows.append(_conflict_gate(aggregate))
        else:
            rows.append(
                _status_gate(aggregate, "All paired time-travel rounds succeed")
            )
    return rows


def _case_gate_metric(row: GateRow) -> str:
    if row.metric == "read I/O operations":
        return "get_requests"
    if row.metric in ("GET/PUT requests", "read/write I/O operations"):
        return "request_counts"
    return row.metric


def _gate_row_selector(row: GateRow) -> GateSelector:
    return GateSelector(
        gate="case",
        schema_version=row.scope.schema_version,
        suite=row.scope.suite,
        commit=row.scope.commit,
        seed=row.scope.seed,
        host=row.scope.host,
        scenario=row.scenario,
        fragments=row.fragments,
        storage=row.storage,
        operation=row.operation,
        metric=_case_gate_metric(row),
    )


def _scaling_row_selector(row: ScalingRow) -> GateSelector:
    return GateSelector(
        gate="scaling",
        schema_version=row.scope.schema_version,
        suite=row.scope.suite,
        commit=row.scope.commit,
        seed=row.scope.seed,
        host=row.scope.host,
        scenario=row.scenario,
        operation=row.operation,
        metric=row.metric,
        format_name=row.format_name,
    )


def _annotated_note(note: str, explanation: GateExplanation | None) -> str:
    if explanation is None:
        annotation = "UNEXPLAINED — mechanism and fix are required"
    else:
        annotation = (
            f"EXPLAINED — Mechanism: {explanation.mechanism}; Fix: {explanation.fix}"
        )
    return f"{note}; {annotation}" if note else annotation


def _apply_explanations(
    codec_rows: Sequence[GateRow],
    scaling_rows: Sequence[ScalingRow],
    e2e_rows: Sequence[GateRow],
    explanations: dict[GateSelector, GateExplanation],
) -> tuple[list[GateRow], list[ScalingRow], list[GateRow]]:
    verdicts: dict[GateSelector, str] = {}
    for row in (*codec_rows, *e2e_rows):
        selector = _gate_row_selector(row)
        if selector in verdicts:
            raise RuntimeError(f"duplicate generated gate: {_selector_text(selector)}")
        verdicts[selector] = row.verdict
    for row in scaling_rows:
        selector = _scaling_row_selector(row)
        if selector in verdicts:
            raise RuntimeError(f"duplicate generated gate: {_selector_text(selector)}")
        verdicts[selector] = row.verdict

    for selector in explanations:
        actual = verdicts.get(selector)
        if actual is None:
            raise ExplanationError(
                f"unknown explanation selector: {_selector_text(selector)}"
            )
        if actual != "FAIL":
            raise ExplanationError(
                "explanation selector does not identify a FAIL gate "
                f"(actual {actual}): {_selector_text(selector)}"
            )

    def annotate_gate(row: GateRow) -> GateRow:
        if row.verdict != "FAIL":
            return row
        explanation = explanations.get(_gate_row_selector(row))
        return replace(row, note=_annotated_note(row.note, explanation))

    def annotate_scaling(row: ScalingRow) -> ScalingRow:
        if row.verdict != "FAIL":
            return row
        explanation = explanations.get(_scaling_row_selector(row))
        return replace(row, note=_annotated_note(row.note, explanation))

    return (
        [annotate_gate(row) for row in codec_rows],
        [annotate_scaling(row) for row in scaling_rows],
        [annotate_gate(row) for row in e2e_rows],
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Input JSONL file(s)")
    parser.add_argument(
        "--expected-commit",
        required=True,
        type=_expected_commit_argument,
        help="full Git SHA that every input record must match",
    )
    parser.add_argument("-o", "--output", help="Write Markdown to this path")
    parser.add_argument(
        "--explanations", help="JSON sidecar with mechanism and fix for FAIL gates"
    )
    return parser.parse_args(argv)


def _md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _scope_label(scope: RunScope) -> str:
    short_commit = scope.commit[:12]
    return f"{short_commit} / seed {scope.seed} / {scope.host}"


def _render_gate_table(rows: Sequence[GateRow]) -> list[str]:
    if not rows:
        return ["No gate rows are available.", ""]
    lines = [
        "| Scope | Scenario | Fragments | Storage | Operation | Metric | Protobuf median | Lance median | Criterion | Result | Notes |",
        "|---|---|---:|---|---|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _md(value)
                for value in (
                    _scope_label(row.scope),
                    row.scenario,
                    f"{row.fragments:,}",
                    row.storage,
                    row.operation,
                    row.metric,
                    row.protobuf_value,
                    row.lance_value,
                    row.criterion,
                    f"**{row.verdict}**",
                    row.note,
                )
            )
            + " |"
        )
    lines.append("")
    return lines


def _render_scaling_table(rows: Sequence[ScalingRow]) -> list[str]:
    if not rows:
        return ["No scaling rows are available.", ""]
    lines = [
        "The expected 1K-to-10M growth is 10,000x. Symmetric deviation from that growth must be at most 1.5x.",
        "",
        "| Scope | Scenario | Format | Operation | Metric | 1K median | 10M median | Observed growth | Linear deviation | Result | Notes |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _md(value)
                for value in (
                    _scope_label(row.scope),
                    row.scenario,
                    row.format_name,
                    row.operation,
                    row.metric,
                    row.small_value,
                    row.large_value,
                    row.growth,
                    row.deviation,
                    f"**{row.verdict}**",
                    row.note,
                )
            )
            + " |"
        )
    lines.append("")
    return lines


def _s2_cost_rows(
    aggregates: dict[CaseKey, CaseAggregate], expected: Sequence[CaseAggregate]
) -> list[list[str]]:
    rows: list[list[str]] = []
    scopes = sorted(
        {
            aggregate.key.scope
            for aggregate in expected
            if aggregate.key.suite == "codec"
        }
    )
    metrics = {
        "encode": "wall_ns",
        "decode": "wall_ns",
        "size": "bytes",
        "decode_rss": "peak_rss_bytes",
    }
    for scope in scopes:
        for fragments in FRAGMENT_SIZES:
            for operation in CODEC_OPERATIONS:
                metric = metrics[operation]
                s1_key = CaseKey(
                    scope.schema_version,
                    "codec",
                    "S1",
                    fragments,
                    "memory",
                    operation,
                    scope.commit,
                    scope.seed,
                    scope.host,
                )
                s2_key = CaseKey(
                    scope.schema_version,
                    "codec",
                    "S2",
                    fragments,
                    "memory",
                    operation,
                    scope.commit,
                    scope.seed,
                    scope.host,
                )
                s1 = aggregates[s1_key]
                s2 = aggregates[s2_key]
                for format_name in FORMATS:
                    if "INCOMPLETE" in (s1.state, s2.state):
                        s1_value = _median_display(s1, format_name, metric)
                        s2_value = _median_display(s2, format_name, metric)
                        ratio = "--"
                        status = "INCOMPLETE"
                    elif "FAIL" in (s1.state, s2.state):
                        s1_value = _median_display(s1, format_name, metric)
                        s2_value = _median_display(s2, format_name, metric)
                        ratio = "--"
                        status = "FAILED RUN"
                    else:
                        s1_median = s1.median(format_name, metric)
                        s2_median = s2.median(format_name, metric)
                        s1_value = _fmt_number(s1_median)
                        s2_value = _fmt_number(s2_median)
                        ratio = _fmt_ratio(_safe_ratio(s2_median, s1_median))
                        status = "AVAILABLE"
                    rows.append(
                        [
                            _scope_label(scope),
                            f"{fragments:,}",
                            operation,
                            metric,
                            format_name,
                            s1_value,
                            s2_value,
                            ratio,
                            status,
                        ]
                    )
    return rows


def _render_s2_cost_table(rows: Sequence[Sequence[str]]) -> list[str]:
    if not rows:
        return ["No codec data are available for the S2/S1 comparison.", ""]
    lines = [
        "This section is informational and isolates the incremental cost of the S2 nullable/variable-shape representation.",
        "",
        "| Scope | Fragments | Operation | Metric | Format | S1 median | S2 median | S2/S1 | Availability |",
        "|---|---:|---|---|---|---:|---:|---:|---|",
    ]
    lines.extend("| " + " | ".join(_md(value) for value in row) + " |" for row in rows)
    lines.append("")
    return lines


def _extrapolation_rows(expected: Sequence[CaseAggregate]) -> list[list[str]]:
    rows: list[list[str]] = []
    for aggregate in expected:
        key = aggregate.key
        if key.fragments != 10_000_000:
            continue
        if key.suite == "codec" and key.operation == "size":
            metric = "bytes"
            metric_label = "manifest bytes"
        elif key.suite == "e2e" and key.operation == "commit":
            metric = "write_bytes"
            metric_label = "commit write_bytes"
        else:
            continue
        for format_name in FORMATS:
            if aggregate.state == "PASS":
                measured = aggregate.median(format_name, metric)
                measured_display = _fmt_number(measured)
                estimate_display = _fmt_number(measured * 10)
                status = "EXTRAPOLATED"
            else:
                measured_display = _median_display(aggregate, format_name, metric)
                estimate_display = "--"
                status = aggregate.state
            rows.append(
                [
                    key.suite,
                    _scope_label(key.scope),
                    key.scenario,
                    key.storage,
                    format_name,
                    metric_label,
                    measured_display,
                    estimate_display,
                    status,
                ]
            )
    return rows


def _render_extrapolation_table(rows: Sequence[Sequence[str]]) -> list[str]:
    lines = [
        "**Not measured:** every 100M value below is exactly 10 times the measured 10M median. No wall-clock time is extrapolated. These absolute size and write-amplification values are inputs to the later delta/segment discussion.",
        "",
    ]
    if not rows:
        lines.extend(("No complete 10M source rows are available.", ""))
        return lines
    lines.extend(
        (
            "| Suite | Scope | Scenario | Storage | Format | Metric | Measured 10M median | Estimated 100M | Status |",
            "|---|---|---|---|---|---|---:|---:|---|",
        )
    )
    lines.extend("| " + " | ".join(_md(value) for value in row) + " |" for row in rows)
    lines.append("")
    return lines


def _verdict(
    input_issues: Sequence[ValidationIssue],
    completeness_issues: Sequence[str],
    codec_rows: Sequence[GateRow],
    scaling_rows: Sequence[ScalingRow],
    e2e_rows: Sequence[GateRow],
) -> str:
    row_verdicts = [row.verdict for row in codec_rows]
    row_verdicts.extend(row.verdict for row in scaling_rows)
    row_verdicts.extend(row.verdict for row in e2e_rows)
    if input_issues or completeness_issues or "INCOMPLETE" in row_verdicts:
        return "INCOMPLETE"
    if "FAIL" in row_verdicts:
        return "FAIL"
    return "PASS"


def _benchmark_output_hint(
    records: Sequence[Record], suite: str, storage: str, fallback: str
) -> str:
    sources = sorted(
        {
            record.source
            for record in records
            if record.key.suite == suite and record.key.storage == storage
        }
    )
    return sources[0] if sources else fallback


def generate_report(
    paths: Sequence[str | Path],
    expected_commit: str,
    explanations_path: str | Path | None = None,
) -> ReportOutcome:
    """Load JSONL inputs and return the rendered report and overall verdict."""
    if not _is_git_sha(expected_commit):
        raise ValueError(
            "expected_commit must be a full lowercase 40- or 64-character Git SHA"
        )
    explanations = (
        _load_explanations(explanations_path) if explanations_path is not None else {}
    )
    records, issues, input_stats = _load_records(paths, expected_commit)
    aggregates, expected, completeness_issues = _build_aggregates(records, issues)
    codec_rows = _codec_gate_rows(expected)
    scaling_rows = _codec_scaling_rows(aggregates, expected)
    e2e_rows = _e2e_gate_rows(expected)
    verdict = _verdict(issues, completeness_issues, codec_rows, scaling_rows, e2e_rows)
    codec_rows, scaling_rows, e2e_rows = _apply_explanations(
        codec_rows, scaling_rows, e2e_rows, explanations
    )

    gate_verdicts = [row.verdict for row in codec_rows]
    gate_verdicts.extend(row.verdict for row in scaling_rows)
    gate_verdicts.extend(row.verdict for row in e2e_rows)
    gate_counts = Counter(gate_verdicts)
    lines = [
        "# Lance Manifest Benchmark Report",
        "",
        f"Overall result: **{verdict}**",
        "",
        "## Gate Summary",
        "",
        "| Result | Gate rows |",
        "|---|---:|",
        f"| PASS | {gate_counts['PASS']} |",
        f"| FAIL | {gate_counts['FAIL']} |",
        f"| INCOMPLETE | {gate_counts['INCOMPLETE']} |",
        "",
        "INCOMPLETE takes precedence over FAIL because malformed, missing, or unpaired data cannot establish a release gate. FAIL rows remain visible for diagnosis.",
        "",
        "## Data Completeness",
        "",
    ]
    if issues or completeness_issues:
        lines.append("| Kind | Location | Issue |")
        lines.append("|---|---|---|")
        for issue in issues:
            lines.append(f"| input | `{_md(issue.location)}` | {_md(issue.message)} |")
        for issue in completeness_issues:
            lines.append(f"| matrix | -- | {_md(issue)} |")
    else:
        lines.append(
            f"Complete required matrices with at least {MIN_ROUNDS} paired rounds per format were loaded ({len(records)} valid records)."
        )
    lines.extend(("", "## Codec Gates", ""))
    lines.extend(_render_gate_table(codec_rows))
    lines.extend(("## Codec Scaling Gates", ""))
    lines.extend(_render_scaling_table(scaling_rows))
    lines.extend(("## End-to-End Gates", ""))
    lines.extend(_render_gate_table(e2e_rows))
    lines.extend(("## S2 Incremental Cost (Informational)", ""))
    lines.extend(_render_s2_cost_table(_s2_cost_rows(aggregates, expected)))
    lines.extend(("## 100M Linear Extrapolation (Not Measured)", ""))
    lines.extend(_render_extrapolation_table(_extrapolation_rows(expected)))
    lines.extend(
        (
            "## Reproduction Metadata",
            "",
            f"Schema version: `{SUPPORTED_SCHEMA_VERSION}`. Aggregation: median over at least {MIN_ROUNDS} exactly paired round IDs for protobuf and Lance on every common dimension.",
            "",
            f"Expected Git commit: `{expected_commit}`.",
            "",
            "### Inputs",
            "",
            "| JSONL path | Lines | Valid | Invalid |",
            "|---|---:|---:|---:|",
        )
    )
    for stats in input_stats:
        lines.append(
            f"| `{_md(stats.path)}` | {stats.lines} | {stats.valid} | {stats.invalid} |"
        )
    scope_hints = {record.key.scope for record in records}
    scope_hints.update(
        issue.key.scope
        for issue in issues
        if issue.key is not None and issue.key.suite in ("codec", "e2e")
    )
    lines.extend(
        (
            "",
            "### Run Scopes",
            "",
            "| Schema | Suite | Commit | Seed | Host |",
            "|---:|---|---|---:|---|",
        )
    )
    if scope_hints:
        for scope in sorted(scope_hints):
            lines.append(
                "| "
                + " | ".join(
                    _md(value)
                    for value in (
                        scope.schema_version,
                        scope.suite,
                        scope.commit,
                        scope.seed,
                        scope.host,
                    )
                )
                + " |"
            )
    else:
        lines.append("| -- | -- | -- | -- | -- |")
    codec_output = _benchmark_output_hint(
        records, "codec", "memory", "/absolute/path/codec.jsonl"
    )
    ebs_output = _benchmark_output_hint(
        records, "e2e", "ebs", "/absolute/path/e2e-ebs.jsonl"
    )
    s3_output = _benchmark_output_hint(
        records, "e2e", "s3", "/absolute/path/e2e-s3.jsonl"
    )
    benchmark_commands = (
        "python3 benchmarks/manifest/run_codec.py \\\n"
        f"  --output {shlex.quote(codec_output)}\n\n"
        "python3 benchmarks/manifest/run_e2e.py \\\n"
        "  --dataset-prefix /mnt/lance-manifest-bench \\\n"
        "  --storage ebs \\\n"
        f"  --output {shlex.quote(ebs_output)}\n\n"
        "python3 benchmarks/manifest/run_e2e.py \\\n"
        "  --dataset-prefix 's3://<bucket>/<prefix>/lance-manifest-bench' \\\n"
        "  --storage s3 \\\n"
        f"  --output {shlex.quote(s3_output)}"
    )
    reproduction_command = (
        "python3 benchmarks/manifest/report.py "
        f"--expected-commit {shlex.quote(expected_commit)} "
        + " ".join(
            shlex.quote(str(Path(path).expanduser().resolve())) for path in paths
        )
    )
    if explanations_path is not None:
        resolved_explanations = str(Path(explanations_path).expanduser().resolve())
        reproduction_command += f" --explanations {shlex.quote(resolved_explanations)}"
        explanation_metadata = f"`{_md(resolved_explanations)}`"
    else:
        explanation_metadata = "none; FAIL rows are marked `UNEXPLAINED`"
    lines.extend(
        (
            "",
            "### Benchmark Commands",
            "",
            "Replace the dataset-prefix placeholders with benchmark-host storage. Output paths below are taken from the loaded JSONL inputs when available.",
            "",
            "```sh",
            benchmark_commands,
            "```",
            "",
            "### Report Regeneration",
            "",
            f"Explanation sidecar: {explanation_metadata}.",
            "",
            "```sh",
            reproduction_command,
            "```",
            "",
        )
    )
    return ReportOutcome("\n".join(lines), verdict)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        outcome = generate_report(args.inputs, args.expected_commit, args.explanations)
    except ExplanationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(outcome.markdown + "\n", encoding="utf-8")
    else:
        print(outcome.markdown)
    return outcome.exit_code


if __name__ == "__main__":
    sys.exit(main())
