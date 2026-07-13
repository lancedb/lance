#!/usr/bin/env python3
"""Gate the complete stable logical row-address design protocol evidence."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import protocol  # noqa: E402
import run  # noqa: E402


BOOTSTRAP_SAMPLES = 10_000
B_FAST = 2 * 1024 * 1024
W_FAST = 64 * 1024 * 1024
SIDECAR_FIELDS = {
    "schema_version",
    "suite",
    "run_id",
    "created_at_utc",
    "commit",
    "source_provenance",
    "host",
    "seed",
    "profile",
    "tracks",
    "variants",
    "matrix_case_names",
    "storage",
    "dataset_root",
    "base_dataset_root",
    "shard_count",
    "shard_index",
    "shard_id",
    "shard_strategy",
    "output_jsonl",
    "executable",
    "data_retention",
    "storage_scope",
    "fixture_strategy",
    "fixture_lineage_jsonl",
    "checkpoint_json",
    "projected_canonical_payload_bytes",
    "projected_unique_initial_index_payload_bytes_lower_bound",
    "projected_no_dedup_logical_data_payload_bytes",
    "projected_no_dedup_logical_index_payload_bytes",
    "projected_minimum_full_scan_payload_bytes",
    "matrix_sha256",
    "matrix_canonical_json",
    "matrix",
    "policy_sha256",
    "policy_canonical_json",
    "policy",
}
COMMIT_OPERATIONS = {
    "create",
    "fixture_clone",
    "append",
    "delete",
    "update",
    "merge_insert",
    "backfill",
    "default_compaction",
    "random_delete_reclaim",
    "normalize_placement",
    "repack",
    "recluster",
    "checkpoint_generation",
    "index_build",
    "index_optimize",
}
RELOCATION_OPERATIONS = {
    "default_compaction",
    "random_delete_reclaim",
    "normalize_placement",
    "repack",
    "recluster",
}
STANDARD_METRICS = (
    "latency",
    "throughput",
    "data_read_bytes",
    "data_write_bytes",
    "index_write_bytes",
    "total_read_bytes",
    "total_write_bytes",
    "peak_rss_bytes",
    "get_requests",
    "head_requests",
    "list_requests",
)


@dataclasses.dataclass(frozen=True)
class Gate:
    track: str
    scope: str
    metric: str
    baseline: str
    samples: int
    ratio: float | None
    ci_low: float | None
    ci_high: float | None
    direction: str
    threshold: float
    strict: bool
    passed: bool
    detail: str = ""

    @property
    def contract(self) -> str:
        if self.direction == "lower":
            operator = ">" if self.strict else ">="
        else:
            operator = "<" if self.strict else "<="
        return f"{self.direction} CI {operator} {self.threshold:.2f}"

    def as_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self) | {"contract": self.contract}


@dataclasses.dataclass(frozen=True)
class ReportResult:
    verdict: str
    markdown: str
    machine: dict[str, Any]

    @property
    def exit_code(self) -> int:
        if self.verdict == "INCOMPLETE":
            return 2
        if self.verdict == "FAIL":
            return 1
        return 0


def sidecar_path(input_path: Path) -> Path:
    return Path(f"{input_path}.protocol.json")


def _strict_object(value: Any, fields: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    missing = sorted(fields - value.keys())
    unknown = sorted(value.keys() - fields)
    if missing or unknown:
        raise ValueError(
            f"{context} fields mismatch: missing={missing}, unknown={unknown}"
        )
    return value


def validate_sidecar(value: Any) -> dict[str, Any]:
    value = _strict_object(value, SIDECAR_FIELDS, "protocol sidecar")
    if (
        value["schema_version"] != 1
        or value["suite"] != "stable_row_address_design_protocol"
    ):
        raise ValueError("protocol sidecar schema or suite is unsupported")
    for field in (
        "run_id",
        "created_at_utc",
        "commit",
        "source_provenance",
        "host",
        "profile",
        "storage",
        "dataset_root",
        "base_dataset_root",
        "shard_id",
        "shard_strategy",
        "output_jsonl",
        "executable",
        "data_retention",
        "storage_scope",
        "fixture_strategy",
        "fixture_lineage_jsonl",
        "checkpoint_json",
        "matrix_sha256",
        "matrix_canonical_json",
        "policy_sha256",
        "policy_canonical_json",
    ):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"protocol sidecar {field} must be non-empty")
    if run.SHA_PATTERN.fullmatch(value["commit"]) is None:
        raise ValueError("protocol sidecar commit must be a full Git SHA")
    if value["source_provenance"] not in {
        "clean-committed-source",
        "dirty-development-override",
    }:
        raise ValueError("protocol sidecar source_provenance is invalid")
    if (
        value["profile"] == "release"
        and value["source_provenance"] != "clean-committed-source"
    ):
        raise ValueError("release evidence requires clean committed source")
    if value["profile"] not in {"smoke", "release"}:
        raise ValueError("protocol sidecar profile is unsupported")
    if value["storage"] not in {"ebs", "s3"}:
        raise ValueError("protocol sidecar storage is unsupported")
    if (
        not isinstance(value["shard_count"], int)
        or isinstance(value["shard_count"], bool)
        or value["shard_count"] <= 0
        or not isinstance(value["shard_index"], int)
        or isinstance(value["shard_index"], bool)
        or value["shard_index"] not in range(value["shard_count"])
        or value["shard_id"]
        != f"shard-{value['shard_index']:03d}-of-{value['shard_count']:03d}"
    ):
        raise ValueError("protocol sidecar shard specification is invalid")
    if value["shard_strategy"] != "schema_and_fragment_layout_fixture_locality":
        raise ValueError("protocol sidecar shard_strategy is unsupported")
    if value["data_retention"] != "preserve":
        raise ValueError("protocol sidecar must preserve datasets")
    expected_scope = (
        "same_region_s3_preserved_release"
        if value["profile"] == "release"
        else "bounded_smoke"
    )
    if value["storage_scope"] != expected_scope:
        raise ValueError("protocol sidecar storage_scope is inconsistent with profile")
    if value["profile"] == "release" and value["storage"] != "s3":
        raise ValueError("release evidence requires same-region S3")
    if value["fixture_strategy"] != (
        "canonical_base_per_format_schema_fragment_layout_then_shallow_clone"
    ):
        raise ValueError("protocol sidecar fixture_strategy is unsupported")
    if (
        not isinstance(value["projected_canonical_payload_bytes"], int)
        or isinstance(value["projected_canonical_payload_bytes"], bool)
        or value["projected_canonical_payload_bytes"] <= 0
    ):
        raise ValueError("projected_canonical_payload_bytes must be positive")
    for field in (
        "projected_unique_initial_index_payload_bytes_lower_bound",
        "projected_no_dedup_logical_data_payload_bytes",
        "projected_no_dedup_logical_index_payload_bytes",
        "projected_minimum_full_scan_payload_bytes",
    ):
        if (
            not isinstance(value[field], int)
            or isinstance(value[field], bool)
            or value[field] < 0
        ):
            raise ValueError(f"{field} must be non-negative")
    if not isinstance(value["seed"], int) or isinstance(value["seed"], bool):
        raise ValueError("protocol sidecar seed must be an integer")
    if not isinstance(value["tracks"], list) or not value["tracks"]:
        raise ValueError("protocol sidecar tracks must be non-empty")
    if len(set(value["tracks"])) != len(value["tracks"]):
        raise ValueError("protocol sidecar tracks must be unique")
    if set(value["tracks"]) - set(protocol.TRACK_FIELDS):
        raise ValueError("protocol sidecar contains an unsupported track")
    if not isinstance(value["variants"], list) or set(value["variants"]) - {
        "bare",
        "scalar",
        "vector",
    }:
        raise ValueError("protocol sidecar variants are invalid")
    if not isinstance(value["matrix_case_names"], list) or any(
        not isinstance(case, str) or not case for case in value["matrix_case_names"]
    ):
        raise ValueError("protocol sidecar matrix_case_names are invalid")

    matrix_canonical = json.dumps(
        value["matrix"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    if matrix_canonical != value["matrix_canonical_json"]:
        raise ValueError("matrix canonical JSON does not match sidecar object")
    if hashlib.sha256(matrix_canonical.encode()).hexdigest() != value["matrix_sha256"]:
        raise ValueError("matrix SHA-256 does not match canonical JSON")
    # Re-run the complete matrix validator instead of trusting the embedded object.
    temporary_matrix = value["matrix"]
    _strict_object(
        temporary_matrix,
        {"schema_version", "name", "profiles", "tracks", "measurement"},
        "embedded matrix",
    )
    profile = temporary_matrix["profiles"].get(value["profile"])
    if not isinstance(profile, dict):
        raise ValueError("selected profile is missing from matrix")
    cases_by_name = {
        case.name: case
        for case in protocol.iter_matrix_cases(
            profile, set(temporary_matrix["tracks"]["matrix"]["cases"])
        )
    }
    try:
        selected_cases = [cases_by_name[name] for name in value["matrix_case_names"]]
    except KeyError as error:
        raise ValueError(
            f"sidecar references unknown matrix case: {error.args[0]}"
        ) from error
    fixture_keys = protocol.fixture_keys_for_run(
        profile, value["tracks"], value["variants"], selected_cases
    )
    projected = protocol.projected_canonical_payload_bytes(profile, fixture_keys)
    if projected != value["projected_canonical_payload_bytes"]:
        raise ValueError(
            "projected canonical payload bytes do not match selected fixtures"
        )
    projected_unique_index = (
        protocol.projected_unique_initial_index_payload_bytes_lower_bound(
            profile, fixture_keys, selected_cases
        )
    )
    if (
        projected_unique_index
        != value["projected_unique_initial_index_payload_bytes_lower_bound"]
    ):
        raise ValueError(
            "projected unique initial index payload bytes do not match fixtures"
        )
    projected_logical_data, projected_logical_index = (
        protocol.projected_no_dedup_logical_payload_bytes(
            profile, value["tracks"], value["variants"], selected_cases
        )
    )
    if projected_logical_data != value["projected_no_dedup_logical_data_payload_bytes"]:
        raise ValueError("projected no-dedup logical data bytes do not match workloads")
    if (
        projected_logical_index
        != value["projected_no_dedup_logical_index_payload_bytes"]
    ):
        raise ValueError(
            "projected no-dedup logical index bytes do not match workloads"
        )
    projected_scan = protocol.projected_minimum_full_scan_payload_bytes(
        profile, value["tracks"], value["variants"], selected_cases
    )
    if projected_scan != value["projected_minimum_full_scan_payload_bytes"]:
        raise ValueError("projected minimum full-scan bytes do not match workloads")

    policy_canonical = json.dumps(
        value["policy"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    if policy_canonical != value["policy_canonical_json"]:
        raise ValueError("policy canonical JSON does not match sidecar object")
    if hashlib.sha256(policy_canonical.encode()).hexdigest() != value["policy_sha256"]:
        raise ValueError("policy SHA-256 does not match canonical JSON")
    # canonical_policy's strict validator is exercised by the runner. The report
    # still checks the fields it consumes so evidence cannot redefine semantics.
    if value["policy"].get("schema_version") != 1:
        raise ValueError("policy schema_version must be 1")
    return value


def load_evidence(
    input_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    try:
        sidecar = validate_sidecar(
            json.loads(sidecar_path(input_path).read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return {}, [], [f"{sidecar_path(input_path)}: {error}"]
    records: list[dict[str, Any]] = []
    try:
        lines = input_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return sidecar, [], [f"{input_path}: {error}"]
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            issues.append(f"{input_path}:{line_number}: blank line")
            continue
        try:
            record = run.validate_record(json.loads(line))
        except (json.JSONDecodeError, ValueError) as error:
            issues.append(f"{input_path}:{line_number}: {error}")
            continue
        records.append(record)
    issues.extend(audit_fixture_lineage(sidecar, records))
    issues.extend(audit_checkpoint(sidecar, input_path, records))
    issues.extend(audit_maintenance_plans(sidecar, records))
    return sidecar, records, issues


def audit_maintenance_plans(
    sidecar: dict[str, Any], records: Sequence[dict[str, Any]]
) -> list[str]:
    issues: list[str] = []
    plans = {
        (record["maintenance_plan_path"], record["maintenance_plan_sha256"])
        for record in records
        if record["maintenance_plan_path"] is not None
    }
    for path_value, expected_hash in sorted(plans):
        path = Path(path_value)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            issues.append(f"{path}: {error}")
            continue
        canonical = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        actual_hash = hashlib.sha256(canonical.encode()).hexdigest()
        if actual_hash != expected_hash:
            issues.append(
                f"{path}: maintenance plan hash mismatch: "
                f"{actual_hash} != {expected_hash}"
            )
        expected = {
            "schema_version": 1,
            "suite": "stable_row_address_physical_maintenance_plan",
            "run_id": sidecar["run_id"],
            "commit": sidecar["commit"],
            "policy_sha256": sidecar["policy_sha256"],
        }
        mismatches = {
            field: (expected_value, value.get(field))
            for field, expected_value in expected.items()
            if value.get(field) != expected_value
        }
        if mismatches:
            issues.append(f"{path}: maintenance plan provenance mismatch: {mismatches}")
        try:
            value = _strict_object(
                value,
                {
                    "schema_version",
                    "suite",
                    "run_id",
                    "commit",
                    "policy_sha256",
                    "source_format",
                    "source_dataset_uri",
                    "source_dataset_version",
                    "schema_kind",
                    "expected_rows",
                    "target_rows_per_fragment",
                    "execution_target_rows_per_fragment",
                    "target_file_size_bytes",
                    "max_source_fragments_per_group",
                    "fragment_count",
                    "groups",
                    "expected_output_live_rows",
                    "expected_output_fragment_count",
                },
                "physical maintenance plan",
            )
            groups = value["groups"]
            if not isinstance(groups, list) or len(groups) != 1:
                raise ValueError("plan must contain one contiguous source group")
            group = _strict_object(
                groups[0],
                {
                    "start_ordinal",
                    "end_ordinal",
                    "source_live_rows",
                    "source_physical_rows",
                    "source_physical_data_bytes",
                    "source_live_data_bytes",
                    "expected_output_fragments",
                },
                "physical maintenance plan group",
            )
            numeric_values = (
                value["expected_rows"],
                value["execution_target_rows_per_fragment"],
                value["target_file_size_bytes"],
                value["max_source_fragments_per_group"],
                value["fragment_count"],
                value["expected_output_fragment_count"],
                *group.values(),
            )
            if any(not isinstance(item, int) or item < 0 for item in numeric_values):
                raise ValueError("plan topology values must be non-negative integers")
            if (
                value["execution_target_rows_per_fragment"] == 0
                or value["target_file_size_bytes"] == 0
                or value["max_source_fragments_per_group"] == 0
                or group["start_ordinal"] != 0
                or group["end_ordinal"] != value["fragment_count"]
                or value["fragment_count"] > value["max_source_fragments_per_group"]
                or group["source_live_rows"] != value["expected_rows"]
                or group["source_live_rows"] > group["source_physical_rows"]
                or group["source_live_data_bytes"]
                > group["source_physical_data_bytes"]
            ):
                raise ValueError("plan source group or topology is inconsistent")
            live_rows = group["source_live_rows"]
            live_bytes = group["source_live_data_bytes"]
            expected_target = (
                max(1, live_rows)
                if live_bytes == 0
                else max(
                    1,
                    min(
                        max(1, live_rows),
                        value["target_file_size_bytes"] * live_rows // live_bytes,
                    ),
                )
            )
            expected_outputs = max(
                1, (live_rows + expected_target - 1) // expected_target
            )
            expected_output_live_rows = [expected_target] * (live_rows // expected_target)
            if live_rows % expected_target:
                expected_output_live_rows.append(live_rows % expected_target)
            if not expected_output_live_rows:
                expected_output_live_rows.append(0)
            if (
                value["execution_target_rows_per_fragment"] != expected_target
                or group["expected_output_fragments"] != expected_outputs
                or value["expected_output_live_rows"] != expected_output_live_rows
                or value["expected_output_fragment_count"] != expected_outputs
            ):
                raise ValueError("plan does not implement the frozen byte formula")
        except ValueError as error:
            issues.append(f"{path}: {error}")
    return issues


def audit_checkpoint(
    sidecar: dict[str, Any],
    input_path: Path,
    records: Sequence[dict[str, Any]],
) -> list[str]:
    checkpoint_path = Path(sidecar["checkpoint_json"])
    try:
        checkpoint = _strict_object(
            json.loads(checkpoint_path.read_text(encoding="utf-8")),
            {
                "schema_version",
                "suite",
                "run_id",
                "commit",
                "matrix_sha256",
                "policy_sha256",
                "profile",
                "seed",
                "shard_id",
                "completed_records",
                "output_size",
                "last_completed_unit",
                "inflight",
            },
            "protocol checkpoint",
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return [f"{checkpoint_path}: {error}"]
    issues = []
    expected = {
        "schema_version": 1,
        "suite": "stable_row_address_protocol_checkpoint",
        "run_id": sidecar["run_id"],
        "commit": sidecar["commit"],
        "matrix_sha256": sidecar["matrix_sha256"],
        "policy_sha256": sidecar["policy_sha256"],
        "profile": sidecar["profile"],
        "seed": sidecar["seed"],
        "shard_id": sidecar["shard_id"],
        "completed_records": len(records),
        "output_size": input_path.stat().st_size,
        "inflight": None,
    }
    mismatches = {
        field: (value, checkpoint.get(field))
        for field, value in expected.items()
        if checkpoint.get(field) != value
    }
    if mismatches:
        issues.append(f"{checkpoint_path}: checkpoint mismatch: {mismatches}")
    return issues


def audit_fixture_lineage(
    sidecar: dict[str, Any], records: Sequence[dict[str, Any]]
) -> list[str]:
    issues: list[str] = []
    path = Path(sidecar["fixture_lineage_jsonl"])
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return [f"{path}: {error}"]
    fields = {
        "schema_version",
        "run_id",
        "track",
        "case",
        "repeat",
        "format",
        "schema_kind",
        "index_kind",
        "rows",
        "rows_per_fragment",
        "source_uri",
        "target_uri",
        "pair_id",
    }
    lineage_by_target: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, line in enumerate(lines, 1):
        try:
            value = _strict_object(
                json.loads(line), fields, f"fixture lineage line {line_number}"
            )
        except (json.JSONDecodeError, ValueError) as error:
            issues.append(f"{path}:{line_number}: {error}")
            continue
        if value["schema_version"] != 1 or value["run_id"] != sidecar["run_id"]:
            issues.append(f"{path}:{line_number}: lineage provenance mismatch")
        if value["format"] not in run.FORMATS:
            issues.append(f"{path}:{line_number}: invalid format")
        if not value["source_uri"] or value["source_uri"] == value["target_uri"]:
            issues.append(f"{path}:{line_number}: invalid source/target lineage")
        key = (value["target_uri"], value["format"])
        if key in lineage_by_target:
            issues.append(f"{path}:{line_number}: duplicate clone target {key}")
        lineage_by_target[key] = value

    fixture_sources = {
        record["dataset_uri"]
        for record in records
        if "/fixtures/" in record["pair_id"] and record["status"] == "ok"
    }
    clone_records = [
        record for record in records if record["operation"] == "fixture_clone"
    ]
    if len(lineage_by_target) != len(clone_records):
        issues.append(
            "fixture lineage count does not match fixture_clone record count: "
            f"{len(lineage_by_target)} != {len(clone_records)}"
        )
    for record in clone_records:
        lineage = lineage_by_target.get((record["dataset_uri"], record["format"]))
        if lineage is None:
            issues.append(
                f"{record['pair_id']}/{record['format']}: missing fixture lineage"
            )
            continue
        if lineage["pair_id"] != record["pair_id"]:
            issues.append(
                f"{record['pair_id']}/{record['format']}: lineage pair mismatch"
            )
        if lineage["source_uri"] not in fixture_sources:
            issues.append(
                f"{record['pair_id']}/{record['format']}: source fixture create is absent"
            )
    return issues


def metric_value(record: dict[str, Any], metric: str) -> int | None:
    if metric == "latency":
        return record["duration_ns"]
    if metric == "throughput":
        return record["duration_ns"]
    if metric == "peak_rss_bytes":
        return record["peak_rss_bytes"]
    if metric == "layout_index_maintenance_ns":
        return record["layout_index_maintenance_ns"]
    if metric == "data_read_bytes":
        return record["io_by_path"]["data"]["read_bytes"]
    if metric == "data_write_bytes":
        return record["io_by_path"]["data"]["write_bytes"]
    if metric == "index_write_bytes":
        return record["io_by_path"]["index"]["write_bytes"]
    if metric == "index_read_bytes":
        return record["io_by_path"]["index"]["read_bytes"]
    if metric == "metadata_read_bytes":
        return record["io_by_path"]["metadata"]["read_bytes"]
    if metric == "metadata_write_bytes":
        return record["io_by_path"]["metadata"]["write_bytes"]
    if metric == "total_read_bytes":
        return record["read_bytes"]
    if metric == "total_write_bytes":
        return record["write_bytes"]
    if metric in {"get_requests", "head_requests", "list_requests"}:
        return record[metric]
    if metric in {
        "actual_get_attempts",
        "actual_head_attempts",
        "actual_list_attempts",
    }:
        return record[metric]
    raise ValueError(f"unsupported metric: {metric}")


def ratio_value(candidate: int, baseline: int) -> float | None:
    if baseline == 0:
        return 1.0 if candidate == 0 else None
    return candidate / baseline


def percentile_nearest(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def paired_ratio_ci(
    candidate: Sequence[int],
    baseline: Sequence[int],
    *,
    samples: int,
    seed: int,
    invert: bool = False,
    statistic: str = "median",
) -> tuple[float, float, float] | None:
    if len(candidate) != len(baseline) or not candidate:
        return None
    ratios = []
    for candidate_value, baseline_value in zip(candidate, baseline, strict=True):
        ratio = ratio_value(
            baseline_value if invert else candidate_value,
            candidate_value if invert else baseline_value,
        )
        if ratio is None:
            return None
        ratios.append(ratio)
    if statistic == "median":
        summarize = statistics.median
    elif statistic == "p95":
        def summarize(values: Sequence[float]) -> float:
            return percentile_nearest(values, 0.95)
    else:
        raise ValueError("statistic must be median or p95")
    point = float(summarize(ratios))
    rng = random.Random(seed)
    bootstrapped = []
    for _ in range(samples):
        draw = [ratios[rng.randrange(len(ratios))] for _ in ratios]
        bootstrapped.append(float(summarize(draw)))
    return (
        point,
        percentile_nearest(bootstrapped, 0.025),
        percentile_nearest(bootstrapped, 0.975),
    )


def make_gate(
    *,
    track: str,
    scope: str,
    metric: str,
    baseline_name: str,
    candidate: Sequence[int],
    baseline: Sequence[int],
    direction: str,
    threshold: float,
    strict: bool,
    samples: int,
    ratio_statistic: str = "median",
) -> Gate:
    digest = hashlib.sha256(
        f"{track}\0{scope}\0{metric}\0{baseline_name}".encode()
    ).digest()
    ci = paired_ratio_ci(
        candidate,
        baseline,
        samples=samples,
        seed=int.from_bytes(digest[:8], "big"),
        invert=direction == "lower",
        statistic=ratio_statistic,
    )
    if ci is None:
        return Gate(
            track,
            scope,
            metric,
            baseline_name,
            len(candidate),
            None,
            None,
            None,
            direction,
            threshold,
            strict,
            False,
            "ratio is undefined because baseline is zero while candidate is non-zero",
        )
    ratio, ci_low, ci_high = ci
    if direction == "upper":
        passed = ci_high < threshold if strict else ci_high <= threshold
    elif direction == "lower":
        passed = ci_low > threshold if strict else ci_low >= threshold
    else:
        raise ValueError("direction must be upper or lower")
    return Gate(
        track,
        scope,
        metric,
        baseline_name,
        len(candidate),
        ratio,
        ci_low,
        ci_high,
        direction,
        threshold,
        strict,
        passed,
    )


def group_by_pair(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["pair_id"]].append(record)
    return grouped


def expected_complete_pair_ids(sidecar: dict[str, Any]) -> set[str]:
    run_id = sidecar["run_id"]
    matrix = sidecar["matrix"]
    profile = matrix["profiles"][sidecar["profile"]]
    repeats = profile["paired_repeats"]
    expected: set[str] = set()
    fixture_keys: set[tuple[str, int, str]] = set()
    if "matrix" in sidecar["tracks"]:
        cases_by_name = {
            case.name: case
            for case in protocol.iter_matrix_cases(
                profile, set(matrix["tracks"]["matrix"]["cases"])
            )
        }
        for case_name in sidecar["matrix_case_names"]:
            case = cases_by_name.get(case_name)
            if case is None:
                # Leave an impossible expected ID so the caller reports the bad sidecar.
                expected.add(f"{run_id}/matrix/{case_name}/missing-case-definition")
                continue
            for repeat in range(repeats):
                prefix = f"{run_id}/matrix/{case_name}/repeat-{repeat:03d}"
                for step_index, step in enumerate(case.steps):
                    operation = (
                        "fixture_clone"
                        if step.operation == "create"
                        else step.operation
                    )
                    expected.add(f"{prefix}/step-{step_index:03d}/{operation}")
                    if step.operation == "random_delete_reclaim":
                        expected.update(
                            {
                                f"{prefix}/step-{step_index:03d}/cold-open",
                                f"{prefix}/step-{step_index:03d}/cold-scan",
                                f"{prefix}/step-{step_index:03d}/cold-take",
                            }
                        )
                        if step.index_kind != "none":
                            expected.add(
                                f"{prefix}/step-{step_index:03d}/cold-index-take"
                            )
                probe_step = len(case.steps)
                expected.update(
                    {
                        f"{prefix}/step-{probe_step:03d}/cold-open",
                        f"{prefix}/step-{probe_step:03d}/cold-scan",
                        f"{prefix}/step-{probe_step:03d}/cold-take",
                    }
                )
                if case.steps[-1].index_kind != "none":
                    expected.add(f"{prefix}/step-{probe_step:03d}/cold-index-take")
            fixture_keys.add(
                (case.schema_kind, case.rows_per_fragment, case.fixture_index_kind)
            )

    rounds = profile["repeated_update_rounds"]
    for track in set(sidecar["tracks"]) & {
        "sustained",
        "adversarial_natural",
        "adversarial_aligned",
    }:
        for variant in sidecar["variants"]:
            _, index_kind = protocol.variant_config(variant)
            schema_kind, _ = protocol.variant_config(variant)
            rows_per_fragment = protocol._rows_per_fragment(
                profile["rows"], profile["logical_fragment_counts"][0]
            )
            fixture_keys.add((schema_kind, rows_per_fragment, index_kind))
            for repeat in range(repeats):
                prefix = f"{run_id}/{track}/{variant}/repeat-{repeat:03d}"
                expected.add(f"{prefix}/setup/fixture-clone")
                for update_round in range(rounds):
                    round_prefix = f"{prefix}/round-{update_round:03d}"
                    if track in {"sustained", "adversarial_natural"}:
                        label = "update" if track == "sustained" else "update-attempt"
                        expected.add(f"{round_prefix}/{label}")
                    if index_kind != "none":
                        expected.add(f"{round_prefix}/index-catch-up")
                    expected.update(
                        {
                            f"{prefix}/step-{update_round:03d}/cold-open",
                            f"{prefix}/step-{update_round:03d}/cold-scan",
                            f"{prefix}/step-{update_round:03d}/cold-take",
                        }
                    )
                    if index_kind != "none":
                        expected.add(
                            f"{prefix}/step-{update_round:03d}/cold-index-take"
                        )
                    if track == "adversarial_natural":
                        post_step = rounds + update_round
                        expected.update(
                            {
                                f"{prefix}/step-{post_step:03d}/cold-open",
                                f"{prefix}/step-{post_step:03d}/cold-scan",
                                f"{prefix}/step-{post_step:03d}/cold-take",
                            }
                        )
                        if index_kind != "none":
                            expected.add(
                                f"{prefix}/step-{post_step:03d}/cold-index-take"
                            )
    expanded_fixture_keys = set(fixture_keys)
    expanded_fixture_keys.update(
        (schema_kind, rows_per_fragment, "none")
        for schema_kind, rows_per_fragment, index_kind in fixture_keys
        if index_kind != "none"
    )
    for schema_kind, rows_per_fragment, index_kind in expanded_fixture_keys:
        fixture_prefix = (
            f"{run_id}/fixtures/{schema_kind}/rows-{profile['rows']}/"
            f"rows-per-fragment-{rows_per_fragment}/index-{index_kind}"
        )
        if index_kind == "none":
            expected.add(f"{fixture_prefix}/create")
        else:
            expected.add(f"{fixture_prefix}/fixture_clone")
            expected.add(f"{fixture_prefix}/index_build")
    return expected


def audit_grid_and_correctness(
    sidecar: dict[str, Any], records: Sequence[dict[str, Any]]
) -> tuple[list[str], list[str], dict[str, dict[str, dict[str, Any]]]]:
    issues: list[str] = []
    failures: list[str] = []
    grouped = group_by_pair(records)
    expected = expected_complete_pair_ids(sidecar)
    complete: dict[str, dict[str, dict[str, Any]]] = {}
    for pair_id in sorted(expected):
        pair_records = grouped.get(pair_id, [])
        formats = [record["format"] for record in pair_records]
        if Counter(formats) != Counter(run.FORMATS):
            issues.append(
                f"{pair_id}: expected one record per format, found {sorted(formats)}"
            )
            continue
        by_format = {record["format"]: record for record in pair_records}
        complete[pair_id] = by_format
        config_fields = (
            "operation",
            "expected_rows",
            "mutation_count",
            "id_start",
            "step",
            "selection_step",
            "match_percent",
            "schema_kind",
            "index_kind",
            "selection",
        )
        for field in config_fields:
            if len({record[field] for record in pair_records}) != 1:
                issues.append(f"{pair_id}: paired {field} differs across formats")
        digests = {
            record["state_digest"]
            for record in pair_records
            if record["state_digest"] is not None
        }
        if digests and len(digests) != 1:
            failures.append(f"{pair_id}: state digest mismatch: {sorted(digests)}")

    # Policy boundaries and PMR follow-ups are data-dependent. Admit additional
    # complete three-format pairs, but subject them to the same equivalence
    # checks and statistical grouping as statically expected phases.
    for pair_id, pair_records in grouped.items():
        if pair_id in complete:
            continue
        if Counter(record["format"] for record in pair_records) != Counter(run.FORMATS):
            continue
        by_format = {record["format"]: record for record in pair_records}
        complete[pair_id] = by_format
        for field in (
            "operation",
            "expected_rows",
            "mutation_count",
            "id_start",
            "step",
            "selection_step",
            "match_percent",
            "schema_kind",
            "index_kind",
            "selection",
        ):
            if len({record[field] for record in pair_records}) != 1:
                issues.append(f"{pair_id}: paired {field} differs across formats")
        digests = {
            record["state_digest"]
            for record in pair_records
            if record["state_digest"] is not None
        }
        if digests and len(digests) != 1:
            failures.append(f"{pair_id}: state digest mismatch: {sorted(digests)}")

    for pair_id, by_format in complete.items():
        if by_format["v23_logical"]["operation"] != "index_take":
            continue
        for format_name, record in by_format.items():
            if record["coverage"] is None or record["recall"] is None:
                issues.append(
                    f"{pair_id}/{format_name}: index take is missing coverage or recall"
                )
                continue
            if record["coverage"] != 1.0:
                failures.append(
                    f"{pair_id}/{format_name}: index coverage is {record['coverage']}, expected 1.0"
                )
            minimum_recall = 0.5 if record["index_kind"] == "vector_ivf_flat" else 1.0
            if record["recall"] < minimum_recall:
                failures.append(
                    f"{pair_id}/{format_name}: index recall {record['recall']} is below "
                    f"{minimum_recall}"
                )
        candidate_recall = by_format["v23_logical"]["recall"]
        baseline_recalls = [
            by_format[format_name]["recall"]
            for format_name in ("v22_no_stable", "v22_stable")
        ]
        if candidate_recall is not None and all(
            recall is not None for recall in baseline_recalls
        ):
            best_baseline = max(baseline_recalls)
            if candidate_recall < best_baseline:
                failures.append(
                    f"{pair_id}: v23 recall {candidate_recall} regresses from "
                    f"best v22 recall {best_baseline}"
                )

    for pair_id, by_format in complete.items():
        operation = by_format["v23_logical"]["operation"]
        if operation not in RELOCATION_OPERATIONS:
            continue
        plan_hashes = {
            record["maintenance_plan_sha256"] for record in by_format.values()
        }
        plan_paths = {record["maintenance_plan_path"] for record in by_format.values()}
        if None in plan_hashes or len(plan_hashes) != 1 or len(plan_paths) != 1:
            failures.append(
                f"{pair_id}: paired relocation did not use one identical maintenance plan"
            )
        for field in ("expected_rows", "fragments", "physical_rows"):
            values = {record[field] for record in by_format.values()}
            if len(values) != 1:
                failures.append(
                    f"{pair_id}: relocation postcondition differs for {field}: "
                    f"{sorted(values, key=lambda value: -1 if value is None else value)}"
                )
        indexed = by_format["v23_logical"]["index_kind"] != "none"
        if indexed:
            for field in ("coverage", "recall"):
                values = {record[field] for record in by_format.values()}
                if len(values) != 1:
                    failures.append(
                        f"{pair_id}: relocation index postcondition differs for {field}"
                    )

    if "matrix" in sidecar["tracks"]:
        profile = sidecar["matrix"]["profiles"][sidecar["profile"]]
        cases = {
            case.name: case
            for case in protocol.iter_matrix_cases(
                profile, set(sidecar["matrix"]["tracks"]["matrix"]["cases"])
            )
        }
        for case_name in sidecar["matrix_case_names"]:
            case = cases.get(case_name)
            if case is None:
                continue
            for repeat in range(profile["paired_repeats"]):
                for step_index, step in enumerate(case.steps):
                    if step.operation == "default_compaction":
                        compaction_pair_id = (
                            f"{sidecar['run_id']}/matrix/{case_name}/"
                            f"repeat-{repeat:03d}/step-{step_index:03d}/"
                            "default_compaction"
                        )
                        for record in grouped.get(compaction_pair_id, []):
                            planned = record["compaction_groups_planned"]
                            admitted = record["compaction_groups_admitted"]
                            not_admitted = record["compaction_groups_not_admitted"]
                            if planned is None or admitted is None or not_admitted is None:
                                issues.append(
                                    f"{compaction_pair_id}/{record['format']}: "
                                    "missing compaction admission counts"
                                )
                            elif planned <= 0 or admitted != planned or not_admitted != 0:
                                failures.append(
                                    f"{compaction_pair_id}/{record['format']}: "
                                    f"fast-path admission is {admitted}/{planned} "
                                    f"with {not_admitted} rejected groups"
                                )
                    if step.operation != "random_delete_reclaim":
                        continue
                    pair_id = (
                        f"{sidecar['run_id']}/matrix/{case_name}/repeat-{repeat:03d}/"
                        f"step-{step_index:03d}/default-reclaim-preflight"
                    )
                    preflight = grouped.get(pair_id, [])
                    if [record["format"] for record in preflight] != ["v23_logical"]:
                        issues.append(
                            f"{pair_id}: expected one v23_logical reclaim preflight"
                        )
                        continue
                    record = preflight[0]
                    source_pair_id = (
                        f"{sidecar['run_id']}/matrix/{case_name}/"
                        f"repeat-{repeat:03d}/step-{step_index - 1:03d}/delete"
                    )
                    source = next(
                        (
                            source_record
                            for source_record in grouped.get(source_pair_id, [])
                            if source_record["format"] == "v23_logical"
                        ),
                        None,
                    )
                    if source is None or source["dataset_version"] is None:
                        issues.append(
                            f"{pair_id}: cannot bind not_admitted evidence to delete version"
                        )
                    planned = record["compaction_groups_planned"]
                    if (
                        record["placement_maintenance_required"] is True
                        or record["admission"] is not False
                        or planned is None
                        or planned <= 0
                        or record["compaction_groups_admitted"] != 0
                        or record["compaction_groups_not_admitted"] != planned
                    ):
                        failures.append(
                            f"{pair_id}: default reclaim was not reported as not_admitted"
                        )
                    if (
                        source is not None
                        and record["dataset_version"] != source["dataset_version"]
                    ):
                        failures.append(
                            f"{pair_id}: not_admitted preflight changed dataset version"
                        )
                    if (
                        record["io_by_path"]["data"]["write_bytes"] != 0
                        or record["io_by_path"]["data"]["put_requests"] != 0
                    ):
                        failures.append(f"{pair_id}: rejected preflight wrote data")

    provenance_fields = (
        "run_id",
        "commit",
        "host",
        "seed",
        "storage",
        "policy_sha256",
    )
    for record in records:
        expected_values = {
            "run_id": sidecar["run_id"],
            "commit": sidecar["commit"],
            "host": sidecar["host"],
            "seed": sidecar["seed"],
            "storage": sidecar["storage"],
            "policy_sha256": sidecar["policy_sha256"],
        }
        mismatches = {
            field: (expected_values[field], record[field])
            for field in provenance_fields
            if record[field] != expected_values[field]
        }
        if record["mode"] != sidecar["profile"]:
            mismatches["mode"] = (sidecar["profile"], record["mode"])
        if mismatches:
            issues.append(f"{record['pair_id']}/{record['format']}: {mismatches}")
        if record["status"] != "ok":
            failures.append(
                f"{record['pair_id']}/{record['format']}: {record['error']}"
            )
        if record["status"] == "ok" and record["operation"] == "create":
            if record["io_by_path"]["data"]["write_bytes"] == 0:
                failures.append(
                    f"{record['pair_id']}/{record['format']}: unified tracker missed data writes"
                )
        if (
            record["status"] == "ok"
            and record["operation"] in RELOCATION_OPERATIONS
            and (record["compacted_data_bytes"] or 0) > 0
        ):
            data_io = record["io_by_path"]["data"]
            if data_io["read_bytes"] == 0 or data_io["write_bytes"] == 0:
                failures.append(
                    f"{record['pair_id']}/{record['format']}: unified tracker missed "
                    "relocation data I/O"
                )
        if (
            record["status"] == "ok"
            and record["operation"] in {"index_build", "index_optimize"}
            and record["io_by_path"]["index"]["write_bytes"] == 0
        ):
            failures.append(
                f"{record['pair_id']}/{record['format']}: unified tracker missed index writes"
            )
        if (
            record["status"] == "ok"
            and record["operation"] == "index_take"
            and record["io_by_path"]["index"]["read_bytes"] == 0
        ):
            failures.append(
                f"{record['pair_id']}/{record['format']}: unified tracker missed index reads"
            )
        if (
            record["status"] == "ok"
            and record["index_kind"] != "none"
            and record["coverage"] is not None
            and record["coverage"] != 1.0
        ):
            failures.append(
                f"{record['pair_id']}/{record['format']}: effective index coverage "
                f"is {record['coverage']}, expected 1.0"
            )
        if (
            record["format"] == "v23_logical"
            and record["status"] == "ok"
            and record["placement_maintenance_required"] is not True
        ):
            missing_layout_metrics = [
                field
                for field in (
                    "manifest_bytes",
                    "placement_root_bytes",
                    "placement_delta_bytes",
                    "w_epoch_bytes",
                )
                if record[field] is None
            ]
            if missing_layout_metrics:
                issues.append(
                    f"{record['pair_id']}: missing v2.3 layout metrics "
                    f"{missing_layout_metrics}"
                )
        if record["format"] != "v23_logical" and any(
            record[field] is not None
            for field in (
                "placement_root_bytes",
                "placement_delta_bytes",
                "w_epoch_bytes",
            )
        ):
            issues.append(
                f"{record['pair_id']}/{record['format']}: legacy format claimed v2.3 layout metrics"
            )
        if record["placement_maintenance_required"] is True:
            data_metrics = record["io_by_path"]["data"]
            if data_metrics["write_bytes"] != 0 or data_metrics["put_requests"] != 0:
                failures.append(f"{record['pair_id']}: PMR wrote data before rejection")
            if (
                record["format"] == "v23_logical"
                and round_from_pair_id(record["pair_id"]) == 0
                and any(
                    f"/{track}/" in record["pair_id"]
                    for track in ("adversarial_natural", "adversarial_aligned")
                )
            ):
                failures.append(
                    f"{record['pair_id']}: isolated first update returned PMR"
                )
        elif (
            record["operation"] in COMMIT_OPERATIONS
            and record["admission"] is not True
            and not record["pair_id"].endswith("/default-reclaim-preflight")
        ):
            failures.append(
                f"{record['pair_id']}/{record['format']}: commit operation was not admitted"
            )
        if (
            record["operation"] in RELOCATION_OPERATIONS
            and record["placement_maintenance_required"] is not True
            and record["status"] == "ok"
            and record["maintenance_plan_sha256"] is None
            and not (
                record["pair_id"].endswith("/default-reclaim-preflight")
                and record["admission"] is False
            )
        ):
            failures.append(
                f"{record['pair_id']}/{record['format']}: relocation is missing its frozen plan"
            )
        if "/sustained/" in record["pair_id"] and record["operation"] in {
            "normalize_placement",
            "repack",
            "recluster",
        }:
            failures.append(
                f"{record['pair_id']}: sustained track ran explicit placement maintenance"
            )

    # Aligned update phases deliberately have one candidate preflight record and
    # two baseline update records instead of a synthetic three-format pair.
    if "adversarial_aligned" in sidecar["tracks"]:
        profile = sidecar["matrix"]["profiles"][sidecar["profile"]]
        for variant in sidecar["variants"]:
            for repeat in range(profile["paired_repeats"]):
                repeat_pmr = 0
                for update_round in range(profile["repeated_update_rounds"]):
                    prefix = (
                        f"{sidecar['run_id']}/adversarial_aligned/{variant}/"
                        f"repeat-{repeat:03d}/round-{update_round:03d}"
                    )
                    candidate = grouped.get(f"{prefix}/candidate-preflight", [])
                    if [record["format"] for record in candidate] != ["v23_logical"]:
                        issues.append(f"{prefix}: missing unique candidate preflight")
                    baselines = grouped.get(f"{prefix}/baseline-update", [])
                    if Counter(record["format"] for record in baselines) != Counter(
                        ("v22_no_stable", "v22_stable")
                    ):
                        issues.append(f"{prefix}: missing aligned baseline updates")
                    if (
                        candidate
                        and candidate[0]["placement_maintenance_required"] is True
                    ):
                        repeat_pmr += 1
                        for suffix, expected_formats in (
                            ("normalize", ("v23_logical",)),
                            ("candidate-retry", ("v23_logical",)),
                        ):
                            actual = grouped.get(f"{prefix}/{suffix}", [])
                            if Counter(
                                record["format"] for record in actual
                            ) != Counter(expected_formats):
                                issues.append(f"{prefix}: missing aligned {suffix}")
                        for format_name in ("v22_no_stable", "v22_stable"):
                            actual = grouped.get(
                                f"{prefix}/forced-baseline-maintenance/{format_name}",
                                [],
                            )
                            if [record["format"] for record in actual] != [format_name]:
                                issues.append(
                                    f"{prefix}: missing forced maintenance for {format_name}"
                                )
                        aligned_records = [
                            *grouped.get(f"{prefix}/normalize", []),
                            *[
                                record
                                for format_name in ("v22_no_stable", "v22_stable")
                                for record in grouped.get(
                                    f"{prefix}/forced-baseline-maintenance/{format_name}",
                                    [],
                                )
                            ],
                        ]
                        if len(aligned_records) == 3 and len(
                            {
                                record["maintenance_plan_sha256"]
                                for record in aligned_records
                            }
                        ) != 1:
                            failures.append(
                                f"{prefix}: aligned maintenance did not share one plan"
                            )
                if sidecar["profile"] == "release" and repeat_pmr == 0:
                    failures.append(
                        f"adversarial_aligned/{variant}/repeat-{repeat}: no PMR trigger"
                    )
    return issues, failures, complete


def normalized_pair_template(pair_id: str) -> str:
    return re.sub(r"/repeat-\d{3}/", "/repeat-*/", pair_id, count=1)


def track_of(pair_id: str, run_id: str) -> str:
    prefix = f"{run_id}/"
    if not pair_id.startswith(prefix):
        return "unknown"
    return pair_id[len(prefix) :].split("/", 1)[0]


def standard_scope_is_gated(track: str, template: str, operation: str) -> bool:
    if track == "matrix":
        return True
    if track == "sustained":
        return operation in {
            "update",
            "index_optimize",
            "default_compaction",
            "open",
            "scan",
            "take",
            "index_take",
        }
    if track == "adversarial_natural":
        return "/round-000/update-attempt" in template or any(
            probe in template
            for probe in (
                "/step-000/cold-open",
                "/step-000/cold-scan",
                "/step-000/cold-take",
                "/step-000/cold-index-take",
            )
        )
    return False


def add_standard_pair_gates(
    sidecar: dict[str, Any],
    complete: dict[str, dict[str, dict[str, Any]]],
    *,
    bootstrap_samples: int,
    issues: list[str],
) -> list[Gate]:
    gates: list[Gate] = []
    repeats = sidecar["matrix"]["profiles"][sidecar["profile"]]["paired_repeats"]
    by_template: dict[str, list[dict[str, dict[str, Any]]]] = defaultdict(list)
    for pair_id, pair in complete.items():
        track = track_of(pair_id, sidecar["run_id"])
        template = normalized_pair_template(pair_id)
        operation = next(iter(pair.values()))["operation"]
        if standard_scope_is_gated(track, template, operation):
            by_template[template].append(pair)
    for template, samples in sorted(by_template.items()):
        track = track_of(template, sidecar["run_id"])
        if len(samples) != repeats:
            issues.append(
                f"{template}: expected {repeats} paired repeats, found {len(samples)}"
            )
            continue
        samples.sort(key=lambda pair: pair["v23_logical"]["round"])
        metrics = list(STANDARD_METRICS)
        if sidecar["storage"] == "s3":
            metrics.extend(
                ("actual_get_attempts", "actual_head_attempts", "actual_list_attempts")
            )
        for metric in metrics:
            values: dict[str, list[int]] = {name: [] for name in run.FORMATS}
            unavailable = False
            for pair in samples:
                for format_name in run.FORMATS:
                    measured = metric_value(pair[format_name], metric)
                    if measured is None:
                        unavailable = True
                    else:
                        values[format_name].append(measured)
            if unavailable:
                issues.append(f"{template}: metric {metric} is unavailable")
                continue
            if metric == "throughput":
                no_stable_direction, no_stable_threshold = "lower", 0.95
                stable_direction, stable_threshold = "lower", 1.0
            elif metric == "peak_rss_bytes":
                no_stable_direction, no_stable_threshold = "upper", 1.10
                stable_direction, stable_threshold = "upper", 1.0
            elif metric in {
                "get_requests",
                "head_requests",
                "list_requests",
                "actual_get_attempts",
                "actual_head_attempts",
                "actual_list_attempts",
            }:
                no_stable_direction, no_stable_threshold = "upper", 1.0
                stable_direction, stable_threshold = "upper", 1.0
            else:
                no_stable_direction, no_stable_threshold = "upper", 1.05
                stable_direction, stable_threshold = "upper", 1.0
            gates.append(
                make_gate(
                    track=track,
                    scope=template,
                    metric="latency_p95" if metric == "latency" else metric,
                    baseline_name="v22_no_stable",
                    candidate=values["v23_logical"],
                    baseline=values["v22_no_stable"],
                    direction=no_stable_direction,
                    threshold=no_stable_threshold,
                    strict=False,
                    samples=bootstrap_samples,
                    ratio_statistic="p95" if metric == "latency" else "median",
                )
            )
            gates.append(
                make_gate(
                    track=track,
                    scope=template,
                    metric="latency_p95" if metric == "latency" else metric,
                    baseline_name="v22_stable",
                    candidate=values["v23_logical"],
                    baseline=values["v22_stable"],
                    direction=stable_direction,
                    threshold=stable_threshold,
                    strict=False,
                    samples=bootstrap_samples,
                    ratio_statistic="p95" if metric == "latency" else "median",
                )
            )
    return gates


def round_from_pair_id(pair_id: str) -> int | None:
    match = re.search(r"/round-(\d{3})/", pair_id)
    return int(match.group(1)) if match else None


def repeated_records(
    records: Sequence[dict[str, Any]], track: str, variant: str, repeat: int
) -> list[dict[str, Any]]:
    needle = f"/{track}/{variant}/repeat-{repeat:03d}/"
    return [record for record in records if needle in record["pair_id"]]


def total_metric(records: Sequence[dict[str, Any]], metric: str) -> int | None:
    values = [metric_value(record, metric) for record in records]
    if any(value is None for value in values):
        return None
    if metric == "peak_rss_bytes":
        return max(values, default=0)
    return sum(values)


def add_sustained_prefix_gates(
    sidecar: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    issues: list[str],
    failures: list[str],
) -> list[Gate]:
    if "sustained" not in sidecar["tracks"]:
        return []
    gates: list[Gate] = []
    profile = sidecar["matrix"]["profiles"][sidecar["profile"]]
    repeats = profile["paired_repeats"]
    minimum_boundaries = profile["minimum_sustained_boundaries"]
    for record in records:
        if "/sustained/" not in record["pair_id"] or record["format"] != "v23_logical":
            continue
        if record["placement_maintenance_required"] is True:
            failures.append(f"{record['pair_id']}: sustained track observed PMR")
        delta = record["placement_delta_bytes"]
        epoch = record["w_epoch_bytes"]
        if delta is not None and delta > B_FAST:
            failures.append(f"{record['pair_id']}: Delta {delta} exceeds B_fast")
        if epoch is not None and epoch > W_FAST:
            failures.append(f"{record['pair_id']}: W_epoch {epoch} exceeds W_fast")

    for variant in sidecar["variants"]:
        boundary_rounds_by_repeat: list[list[int]] = []
        samples_by_boundary: dict[int, dict[str, dict[str, list[int]]]] = defaultdict(
            lambda: defaultdict(lambda: {name: [] for name in run.FORMATS})
        )
        for repeat in range(repeats):
            scoped = repeated_records(records, "sustained", variant, repeat)
            boundary_rounds = sorted(
                {
                    round_index
                    for record in scoped
                    if record["operation"] == "default_compaction"
                    and "/policy-maintenance" in record["pair_id"]
                    and (round_index := round_from_pair_id(record["pair_id"]))
                    is not None
                }
            )
            boundary_rounds_by_repeat.append(boundary_rounds)
            if len(boundary_rounds) < minimum_boundaries:
                issues.append(
                    f"sustained/{variant}/repeat-{repeat}: expected at least "
                    f"{minimum_boundaries} boundaries, found {len(boundary_rounds)}"
                )
            for boundary_ordinal, boundary_round in enumerate(boundary_rounds):
                cumulative = [
                    record
                    for record in scoped
                    if record["operation"]
                    in {"update", "index_optimize", "default_compaction"}
                    and (record_round := round_from_pair_id(record["pair_id"]))
                    is not None
                    and record_round <= boundary_round
                ]
                for metric in (
                    "latency",
                    "data_read_bytes",
                    "data_write_bytes",
                    "get_requests",
                    "head_requests",
                    "list_requests",
                ):
                    for format_name in run.FORMATS:
                        value = total_metric(
                            [r for r in cumulative if r["format"] == format_name],
                            metric,
                        )
                        if value is None:
                            issues.append(
                                f"sustained/{variant}/repeat-{repeat}/boundary-"
                                f"{boundary_ordinal}: {metric} unavailable"
                            )
                        else:
                            samples_by_boundary[boundary_ordinal][metric][
                                format_name
                            ].append(value)
        if boundary_rounds_by_repeat and any(
            rounds != boundary_rounds_by_repeat[0]
            for rounds in boundary_rounds_by_repeat[1:]
        ):
            issues.append(
                f"sustained/{variant}: natural boundary rounds differ by repeat"
            )
        for boundary_ordinal, by_metric in sorted(samples_by_boundary.items()):
            for metric, values in by_metric.items():
                if any(len(values[name]) != repeats for name in run.FORMATS):
                    issues.append(
                        f"sustained/{variant}/boundary-{boundary_ordinal}: "
                        f"incomplete {metric} prefix samples"
                    )
                    continue
                threshold = 1.0 if metric.endswith("requests") else 1.05
                scope = f"sustained/{variant}/boundary-{boundary_ordinal}/prefix"
                gates.append(
                    make_gate(
                        track="sustained",
                        scope=scope,
                        metric=metric,
                        baseline_name="v22_no_stable",
                        candidate=values["v23_logical"],
                        baseline=values["v22_no_stable"],
                        direction="upper",
                        threshold=threshold,
                        strict=False,
                        samples=bootstrap_samples,
                    )
                )
                gates.append(
                    make_gate(
                        track="sustained",
                        scope=scope,
                        metric=metric,
                        baseline_name="v22_stable",
                        candidate=values["v23_logical"],
                        baseline=values["v22_stable"],
                        direction="upper",
                        threshold=1.0,
                        strict=False,
                        samples=bootstrap_samples,
                    )
                )
    return gates


def adversarial_natural_record_round(pair_id: str, rounds: int) -> int | None:
    round_index = round_from_pair_id(pair_id)
    if round_index is not None:
        return round_index
    match = re.search(r"/step-(\d{3})/", pair_id)
    if match is None:
        return None
    step = int(match.group(1))
    if step < rounds:
        return step
    if step < 2 * rounds:
        return step - rounds
    return None


def build_adversarial_natural_observations(
    sidecar: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    issues: list[str],
    failures: list[str],
) -> dict[str, Any]:
    if "adversarial_natural" not in sidecar["tracks"]:
        return {}
    profile = sidecar["matrix"]["profiles"][sidecar["profile"]]
    rounds = profile["repeated_update_rounds"]
    repeats = profile["paired_repeats"]
    grouped = group_by_pair(records)
    metrics = [
        "latency",
        "data_read_bytes",
        "data_write_bytes",
        "index_read_bytes",
        "index_write_bytes",
        "metadata_read_bytes",
        "metadata_write_bytes",
        "total_read_bytes",
        "total_write_bytes",
        "get_requests",
        "head_requests",
        "list_requests",
    ]
    if sidecar["storage"] == "s3":
        metrics.extend(
            ("actual_get_attempts", "actual_head_attempts", "actual_list_attempts")
        )
    result: dict[str, Any] = {"schema_version": 1, "variants": {}}
    for variant in sidecar["variants"]:
        variant_repeats = []
        for repeat in range(repeats):
            scoped = [
                record
                for record in repeated_records(
                    records, "adversarial_natural", variant, repeat
                )
                if "/setup/" not in record["pair_id"]
            ]
            records_by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for record in scoped:
                record_round = adversarial_natural_record_round(
                    record["pair_id"], rounds
                )
                if record_round is None:
                    issues.append(
                        f"{record['pair_id']}: cannot assign record to an adversarial round"
                    )
                else:
                    records_by_round[record_round].append(record)

            natural_maintenance_rounds = {name: [] for name in run.FORMATS}
            for update_round in range(rounds):
                prefix = (
                    f"{sidecar['run_id']}/adversarial_natural/{variant}/"
                    f"repeat-{repeat:03d}"
                )
                pre_scan_id = f"{prefix}/step-{update_round:03d}/cold-scan"
                pre_scans = grouped.get(pre_scan_id, [])
                by_format = {record["format"]: record for record in pre_scans}
                if Counter(by_format.keys()) != Counter(run.FORMATS):
                    issues.append(
                        f"{pre_scan_id}: policy evaluation requires one pre-maintenance "
                        "scan per format"
                    )
                    continue
                for format_name in run.FORMATS:
                    try:
                        triggered, _ = protocol.policy_triggers(
                            by_format[format_name], sidecar["policy"]
                        )
                    except ValueError as error:
                        issues.append(
                            f"{pre_scan_id}/{format_name}: policy metrics are invalid: "
                            f"{error}"
                        )
                        continue
                    maintenance_id = (
                        f"{prefix}/round-{update_round:03d}/"
                        f"natural-maintenance/{format_name}"
                    )
                    maintenance = grouped.get(maintenance_id, [])
                    matching = [
                        record
                        for record in maintenance
                        if record["format"] == format_name
                    ]
                    if triggered:
                        natural_maintenance_rounds[format_name].append(update_round)
                        if len(matching) != 1:
                            issues.append(
                                f"{maintenance_id}: frozen physical policy triggered but "
                                f"found {len(matching)} maintenance records"
                            )
                    elif maintenance:
                        failures.append(
                            f"{maintenance_id}: maintenance ran without a frozen-policy trigger"
                        )

            prefixes = []
            cumulative: list[dict[str, Any]] = []
            for update_round in range(rounds):
                cumulative.extend(records_by_round.get(update_round, []))
                totals: dict[str, dict[str, int]] = {}
                for metric in metrics:
                    values: dict[str, int] = {}
                    for format_name in run.FORMATS:
                        value = total_metric(
                            [
                                record
                                for record in cumulative
                                if record["format"] == format_name
                            ],
                            metric,
                        )
                        if value is None:
                            issues.append(
                                f"adversarial_natural/{variant}/repeat-{repeat}/"
                                f"prefix-{update_round}: {metric} unavailable for "
                                f"{format_name}"
                            )
                        else:
                            values[format_name] = value
                    totals[metric] = values
                prefixes.append({"round": update_round, "totals": totals})

            terminal_step = 2 * rounds - 1
            terminal_pair_id = (
                f"{sidecar['run_id']}/adversarial_natural/{variant}/"
                f"repeat-{repeat:03d}/step-{terminal_step:03d}/cold-scan"
            )
            terminal_records = grouped.get(terminal_pair_id, [])
            terminal_by_format = {
                record["format"]: record for record in terminal_records
            }
            if Counter(terminal_by_format.keys()) != Counter(run.FORMATS):
                issues.append(
                    f"{terminal_pair_id}: terminal debt requires one post-maintenance "
                    "scan per format"
                )
            terminal = {
                format_name: {
                    "fragments": record["fragments"],
                    "physical_rows": record["physical_rows"],
                    "physical_data_bytes": record["physical_data_bytes"],
                    "estimated_live_data_bytes": record["estimated_live_data_bytes"],
                    "scan_byte_amplification": record["scan_byte_amplification"],
                    "placement_delta_bytes": record["placement_delta_bytes"],
                    "w_epoch_bytes": record["w_epoch_bytes"],
                }
                for format_name, record in terminal_by_format.items()
            }
            pmr_rounds = sorted(
                {
                    record_round
                    for record in scoped
                    if record["format"] == "v23_logical"
                    and record["placement_maintenance_required"] is True
                    and (
                        record_round := adversarial_natural_record_round(
                            record["pair_id"], rounds
                        )
                    )
                    is not None
                }
            )
            variant_repeats.append(
                {
                    "repeat": repeat,
                    "pmr_trigger_rounds": pmr_rounds,
                    "natural_maintenance_rounds": natural_maintenance_rounds,
                    "prefixes": prefixes,
                    "terminal_debt": terminal,
                }
            )
        result["variants"][variant] = variant_repeats
    return result


def add_adversarial_epoch_gates(
    sidecar: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    issues: list[str],
    failures: list[str],
) -> list[Gate]:
    if "adversarial_natural" not in sidecar["tracks"]:
        return []
    gates: list[Gate] = []
    profile = sidecar["matrix"]["profiles"][sidecar["profile"]]
    repeats = profile["paired_repeats"]
    pmr_by_repeat: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if (
            "/adversarial_natural/" in record["pair_id"]
            and record["format"] == "v23_logical"
            and record["placement_maintenance_required"] is True
        ):
            for variant in sidecar["variants"]:
                if f"/adversarial_natural/{variant}/" in record["pair_id"]:
                    pmr_by_repeat[(variant, record["round"])].append(record)
                    break
    if sidecar["profile"] == "release":
        for variant in sidecar["variants"]:
            for repeat in range(repeats):
                if not pmr_by_repeat[(variant, repeat)]:
                    failures.append(
                        f"adversarial_natural/{variant}/repeat-{repeat}: no PMR trigger"
                    )

    grouped = group_by_pair(records)
    for (variant, repeat), pmr_records in pmr_by_repeat.items():
        for pmr in pmr_records:
            update_round = round_from_pair_id(pmr["pair_id"])
            if update_round is None:
                issues.append(f"{pmr['pair_id']}: PMR record has no update round")
                continue
            prefix = (
                f"{sidecar['run_id']}/adversarial_natural/{variant}/"
                f"repeat-{repeat:03d}/round-{update_round:03d}"
            )
            maintenance = grouped.get(f"{prefix}/pmr-maintenance", [])
            retry = grouped.get(f"{prefix}/update-retry", [])
            if [record["format"] for record in maintenance] != ["v23_logical"]:
                issues.append(f"{prefix}: PMR is missing NormalizePlacement")
            if [record["format"] for record in retry] != ["v23_logical"]:
                issues.append(f"{prefix}: PMR is missing candidate retry")

    for variant in sidecar["variants"]:
        totals: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: {name: [] for name in run.FORMATS}
        )
        epoch_metrics = [
            "latency",
            "data_read_bytes",
            "data_write_bytes",
            "total_read_bytes",
            "total_write_bytes",
            "get_requests",
            "head_requests",
            "list_requests",
            "peak_rss_bytes",
            "index_write_bytes",
        ]
        if sidecar["storage"] == "s3":
            epoch_metrics.extend(
                ("actual_get_attempts", "actual_head_attempts", "actual_list_attempts")
            )
        for repeat in range(repeats):
            scoped = [
                record
                for record in repeated_records(
                    records, "adversarial_natural", variant, repeat
                )
                if "/setup/" not in record["pair_id"]
            ]
            for metric in epoch_metrics:
                for format_name in run.FORMATS:
                    value = total_metric(
                        [
                            record
                            for record in scoped
                            if record["format"] == format_name
                        ],
                        metric,
                    )
                    if value is None:
                        issues.append(
                            f"adversarial_natural/{variant}/repeat-{repeat}: "
                            f"{metric} unavailable"
                        )
                    else:
                        totals[metric][format_name].append(value)
        if variant == "bare":
            continue
        for metric in epoch_metrics:
            values = totals[metric]
            if any(len(values[name]) != repeats for name in run.FORMATS):
                issues.append(
                    f"adversarial_natural/{variant}: incomplete {metric} epoch samples"
                )
                continue
            strict_benefit = metric in {"latency", "index_write_bytes"}
            if metric in {
                "get_requests",
                "head_requests",
                "list_requests",
                "actual_get_attempts",
                "actual_head_attempts",
                "actual_list_attempts",
            }:
                no_stable_threshold = 1.0
            elif metric == "peak_rss_bytes":
                no_stable_threshold = 1.10
            elif strict_benefit:
                no_stable_threshold = 1.0
            else:
                no_stable_threshold = 1.05
            gates.append(
                make_gate(
                    track="adversarial_natural",
                    scope=f"adversarial_natural/{variant}/full-epoch",
                    metric=metric,
                    baseline_name="v22_no_stable",
                    candidate=values["v23_logical"],
                    baseline=values["v22_no_stable"],
                    direction="upper",
                    threshold=no_stable_threshold,
                    strict=strict_benefit,
                    samples=bootstrap_samples,
                )
            )
            gates.append(
                make_gate(
                    track="adversarial_natural",
                    scope=f"adversarial_natural/{variant}/full-epoch",
                    metric=metric,
                    baseline_name="v22_stable",
                    candidate=values["v23_logical"],
                    baseline=values["v22_stable"],
                    direction="upper",
                    threshold=1.0,
                    strict=strict_benefit,
                    samples=bootstrap_samples,
                )
            )
    return gates


def add_aligned_relocation_gates(
    sidecar: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    issues: list[str],
) -> list[Gate]:
    if "adversarial_aligned" not in sidecar["tracks"]:
        return []
    grouped = group_by_pair(records)
    profile = sidecar["matrix"]["profiles"][sidecar["profile"]]
    repeats = profile["paired_repeats"]
    samples: dict[tuple[str, int, str], dict[str, list[int]]] = defaultdict(
        lambda: {name: [] for name in run.FORMATS}
    )
    for variant in sidecar["variants"]:
        for repeat in range(repeats):
            scoped = repeated_records(records, "adversarial_aligned", variant, repeat)
            pmr_rounds = sorted(
                {
                    round_index
                    for record in scoped
                    if record["format"] == "v23_logical"
                    and record["placement_maintenance_required"] is True
                    and (round_index := round_from_pair_id(record["pair_id"]))
                    is not None
                }
            )
            for ordinal, update_round in enumerate(pmr_rounds):
                prefix = (
                    f"{sidecar['run_id']}/adversarial_aligned/{variant}/"
                    f"repeat-{repeat:03d}/round-{update_round:03d}"
                )
                candidate_records = grouped.get(f"{prefix}/normalize", [])
                candidate = next(
                    (
                        record
                        for record in candidate_records
                        if record["format"] == "v23_logical"
                    ),
                    None,
                )
                baselines = {
                    format_name: grouped.get(
                        f"{prefix}/forced-baseline-maintenance/{format_name}", []
                    )
                    for format_name in ("v22_no_stable", "v22_stable")
                }
                if candidate is None or any(
                    len(records_for_format) != 1
                    for records_for_format in baselines.values()
                ):
                    issues.append(
                        f"{prefix}: aligned maintenance evidence is incomplete"
                    )
                    continue
                aligned_metrics = ["latency", "data_write_bytes"]
                if variant != "bare":
                    aligned_metrics.append("index_write_bytes")
                for metric in aligned_metrics:
                    candidate_value = metric_value(candidate, metric)
                    if candidate_value is None:
                        issues.append(f"{prefix}: candidate {metric} unavailable")
                        continue
                    samples[(variant, ordinal, metric)]["v23_logical"].append(
                        candidate_value
                    )
                    for format_name, baseline_records in baselines.items():
                        value = metric_value(baseline_records[0], metric)
                        if value is None:
                            issues.append(
                                f"{prefix}: {format_name} {metric} unavailable"
                            )
                        else:
                            samples[(variant, ordinal, metric)][format_name].append(
                                value
                            )
    gates = []
    for (variant, ordinal, metric), values in sorted(samples.items()):
        if any(len(values[name]) != repeats for name in run.FORMATS):
            issues.append(
                f"adversarial_aligned/{variant}/trigger-{ordinal}: "
                f"expected {repeats} {metric} samples"
            )
            continue
        for baseline_name in ("v22_no_stable", "v22_stable"):
            gates.append(
                make_gate(
                    track="adversarial_aligned",
                    scope=f"adversarial_aligned/{variant}/trigger-{ordinal}",
                    metric=metric,
                    baseline_name=baseline_name,
                    candidate=values["v23_logical"],
                    baseline=values[baseline_name],
                    direction="upper",
                    threshold=1.0,
                    strict=True,
                    samples=bootstrap_samples,
                )
            )
    return gates


def add_indexed_relocation_contract_gates(
    sidecar: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    issues: list[str],
    failures: list[str],
) -> list[Gate]:
    if "matrix" not in sidecar["tracks"]:
        return []
    repeats = sidecar["matrix"]["profiles"][sidecar["profile"]]["paired_repeats"]
    case_names = [
        case
        for case in sidecar["matrix_case_names"]
        if case.startswith("indexed-compact-")
        or case.startswith("indexed-repeated-compaction-")
    ]
    grouped = group_by_pair(records)
    gates: list[Gate] = []
    for case_name in case_names:
        by_template: dict[str, list[dict[str, dict[str, Any]]]] = defaultdict(list)
        needle = f"/matrix/{case_name}/"
        for pair_id, pair_records in grouped.items():
            if needle not in pair_id or not pair_id.endswith("/default_compaction"):
                continue
            if Counter(record["format"] for record in pair_records) != Counter(
                run.FORMATS
            ):
                issues.append(f"{pair_id}: indexed relocation pair is incomplete")
                continue
            by_format = {record["format"]: record for record in pair_records}
            if not any(
                (record["compacted_data_bytes"] or 0) > 0
                for record in by_format.values()
            ):
                continue
            by_template[normalized_pair_template(pair_id)].append(by_format)
        if not by_template:
            issues.append(f"matrix/{case_name}: no executed indexed relocation")
            continue
        for template, samples in sorted(by_template.items()):
            if len(samples) != repeats:
                issues.append(
                    f"{template}: expected {repeats} executed indexed relocation samples"
                )
                continue
            for sample in samples:
                candidate = sample["v23_logical"]
                if candidate["row_addresses_remapped"] != 0:
                    failures.append(f"{candidate['pair_id']}: remapped row addresses")
                if candidate["indices_remapped"] != 0:
                    failures.append(f"{candidate['pair_id']}: remapped index objects")
                if candidate["index_coverage_reuse"] != 1.0:
                    failures.append(
                        f"{candidate['pair_id']}: index coverage reuse is not 100%"
                    )
                index_io = candidate["io_by_path"]["index"]
                if any(
                    index_io[field] != 0
                    for field in (
                        "get_requests",
                        "head_requests",
                        "list_requests",
                        "put_requests",
                        "delete_requests",
                        "read_bytes",
                        "write_bytes",
                    )
                ):
                    failures.append(
                        f"{candidate['pair_id']}: order-preserving compaction accessed index objects"
                    )
                for format_name, record in sample.items():
                    index_bytes = record["index_storage_bytes_before"]
                    data_bytes = record["compacted_data_bytes"]
                    if index_bytes is None or data_bytes is None:
                        issues.append(
                            f"{record['pair_id']}/{format_name}: missing index/data size precondition"
                        )
                    elif index_bytes < data_bytes:
                        failures.append(
                            f"{record['pair_id']}/{format_name}: index bytes {index_bytes} "
                            f"are below compacted data bytes {data_bytes}"
                        )
            for metric, threshold in (
                ("layout_index_maintenance_ns", 0.10),
                ("latency", 0.50),
            ):
                values = {
                    format_name: [
                        metric_value(sample[format_name], metric) for sample in samples
                    ]
                    for format_name in run.FORMATS
                }
                if any(
                    any(value is None for value in samples)
                    for samples in values.values()
                ):
                    issues.append(f"{template}: missing {metric}")
                    continue
                for baseline_name in ("v22_no_stable", "v22_stable"):
                    gates.append(
                        make_gate(
                            track="matrix",
                            scope=f"{template}/indexed-relocation",
                            metric=metric,
                            baseline_name=baseline_name,
                            candidate=values["v23_logical"],
                            baseline=values[baseline_name],
                            direction="upper",
                            threshold=threshold,
                            strict=False,
                            samples=bootstrap_samples,
                        )
                    )
    return gates


def analyze(
    sidecar: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    initial_issues: Sequence[str] = (),
    enforce_gates: bool | None = None,
) -> ReportResult:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    issues = list(initial_issues)
    failures: list[str] = []
    if not sidecar:
        issues.append("protocol sidecar is unavailable")
        complete = {}
        gates: list[Gate] = []
        adversarial_natural = {}
    else:
        grid_issues, correctness_failures, complete = audit_grid_and_correctness(
            sidecar, records
        )
        issues.extend(grid_issues)
        failures.extend(correctness_failures)
        adversarial_natural = build_adversarial_natural_observations(
            sidecar,
            records,
            issues=issues,
            failures=failures,
        )
        gates = add_standard_pair_gates(
            sidecar,
            complete,
            bootstrap_samples=bootstrap_samples,
            issues=issues,
        )
        gates.extend(
            add_sustained_prefix_gates(
                sidecar,
                records,
                bootstrap_samples=bootstrap_samples,
                issues=issues,
                failures=failures,
            )
        )
        gates.extend(
            add_adversarial_epoch_gates(
                sidecar,
                records,
                bootstrap_samples=bootstrap_samples,
                issues=issues,
                failures=failures,
            )
        )
        gates.extend(
            add_aligned_relocation_gates(
                sidecar,
                records,
                bootstrap_samples=bootstrap_samples,
                issues=issues,
            )
        )
        gates.extend(
            add_indexed_relocation_contract_gates(
                sidecar,
                records,
                bootstrap_samples=bootstrap_samples,
                issues=issues,
                failures=failures,
            )
        )
    failed_gates = [gate for gate in gates if not gate.passed]
    if enforce_gates is None:
        enforce_gates = bool(sidecar) and sidecar.get("profile") == "release"
    if issues:
        verdict = "INCOMPLETE"
    elif failures or (enforce_gates and failed_gates):
        verdict = "FAIL"
    else:
        verdict = "PASS"
    machine = {
        "schema_version": 1,
        "suite": "stable_row_address_design_protocol_report",
        "run_id": sidecar.get("run_id") if sidecar else None,
        "verdict": verdict,
        "bootstrap_samples": bootstrap_samples,
        "performance_gates_enforced": enforce_gates,
        "records": len(records),
        "complete_pairs": len(complete),
        "storage_projections": (
            {
                field: sidecar[field]
                for field in (
                    "projected_canonical_payload_bytes",
                    "projected_unique_initial_index_payload_bytes_lower_bound",
                    "projected_no_dedup_logical_data_payload_bytes",
                    "projected_no_dedup_logical_index_payload_bytes",
                    "projected_minimum_full_scan_payload_bytes",
                )
            }
            if sidecar
            else {}
        ),
        "issues": issues,
        "failures": failures,
        "gates": [gate.as_json() for gate in gates],
        "adversarial_natural": adversarial_natural,
    }
    lines = [
        "# Stable Logical Row Address Protocol Report",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"Records: {len(records)}; complete paired phases: {len(complete)}; "
        f"bootstrap resamples: {bootstrap_samples}",
        "",
        "| Track | Scope | Metric | Baseline | Samples | Ratio | 95% CI | Contract | Result |",
        "|---|---|---|---|---:|---:|---:|---|---|",
    ]
    if sidecar:
        lines[6:6] = [
            "## Frozen storage/runtime projections",
            "",
            f"- Canonical data payload: {sidecar['projected_canonical_payload_bytes']} bytes",
            "- Unique initial index payload lower bound: "
            f"{sidecar['projected_unique_initial_index_payload_bytes_lower_bound']} bytes",
            "- No-dedup logical data/index payload: "
            f"{sidecar['projected_no_dedup_logical_data_payload_bytes']} / "
            f"{sidecar['projected_no_dedup_logical_index_payload_bytes']} bytes",
            "- Minimum full-scan payload: "
            f"{sidecar['projected_minimum_full_scan_payload_bytes']} bytes",
            "",
        ]
    for gate in gates:
        ratio = "undefined" if gate.ratio is None else f"{gate.ratio:.4f}"
        ci = (
            "undefined"
            if gate.ci_low is None or gate.ci_high is None
            else f"[{gate.ci_low:.4f}, {gate.ci_high:.4f}]"
        )
        lines.append(
            f"| {gate.track} | {gate.scope} | {gate.metric} | {gate.baseline} | "
            f"{gate.samples} | {ratio} | {ci} | {gate.contract} | "
            f"{'PASS' if gate.passed else 'FAIL'} |"
        )
    if not gates:
        lines.append("| — | — | — | — | — | — | — | — | INCOMPLETE |")
    variants = adversarial_natural.get("variants", {})
    if variants:
        lines.extend(
            (
                "",
                "## Adversarial natural-policy terminal debt",
                "",
                "All per-round prefix totals are preserved in the JSON report.",
                "",
                "| Variant | Repeat | Format | PMR rounds | Natural maintenance rounds | Fragments | Scan amplification | Delta | W_epoch |",
                "|---|---:|---|---|---|---:|---:|---:|---:|",
            )
        )
        for variant, repeat_values in variants.items():
            for repeat_value in repeat_values:
                for format_name in run.FORMATS:
                    terminal = repeat_value["terminal_debt"].get(format_name, {})
                    lines.append(
                        f"| {variant} | {repeat_value['repeat']} | {format_name} | "
                        f"{repeat_value['pmr_trigger_rounds'] if format_name == 'v23_logical' else []} | "
                        f"{repeat_value['natural_maintenance_rounds'][format_name]} | "
                        f"{terminal.get('fragments', '—')} | "
                        f"{terminal.get('scan_byte_amplification', '—')} | "
                        f"{terminal.get('placement_delta_bytes', '—')} | "
                        f"{terminal.get('w_epoch_bytes', '—')} |"
                    )
    if issues:
        lines.extend(("", "## Incomplete evidence", ""))
        lines.extend(f"- {issue}" for issue in issues)
    if failures:
        lines.extend(("", "## Correctness and protocol failures", ""))
        lines.extend(f"- {failure}" for failure in failures)
    if failed_gates:
        lines.extend(("", "## Failed performance gates", ""))
        lines.extend(
            f"- {gate.track}/{gate.scope}/{gate.metric} vs {gate.baseline}: "
            f"{gate.detail or gate.contract}"
            for gate in failed_gates
        )
        if not enforce_gates:
            lines.extend(
                (
                    "",
                    "Smoke performance failures are diagnostic; release reports always enforce them.",
                )
            )
    return ReportResult(verdict, "\n".join(lines) + "\n", machine)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--enforce-gates", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.bootstrap_samples < BOOTSTRAP_SAMPLES:
        raise ValueError(
            f"CLI reports require at least {BOOTSTRAP_SAMPLES} bootstrap samples"
        )
    sidecar, records, issues = load_evidence(args.input)
    result = analyze(
        sidecar,
        records,
        bootstrap_samples=args.bootstrap_samples,
        initial_issues=issues,
        enforce_gates=args.enforce_gates or None,
    )
    if args.markdown:
        protocol.replace_text_atomic(args.markdown, result.markdown)
    else:
        sys.stdout.write(result.markdown)
    encoded = json.dumps(result.machine, sort_keys=True, separators=(",", ":")) + "\n"
    if args.json:
        protocol.replace_text_atomic(args.json, encoded)
    else:
        sys.stderr.write(encoded)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
