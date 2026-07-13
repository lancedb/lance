#!/usr/bin/env python3
"""Build and run the process-isolated stable-row-address comparison suite."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = Path(__file__).with_name("physical_maintenance_policy.v1.json")
SCHEMA_VERSION = 1
SUITE = "stable_row_address_e2e"
FORMATS = ("v22_no_stable", "v22_stable", "v23_logical")
FORMAT_CLI_NAMES = {
    "v22_no_stable": "v22-no-stable",
    "v22_stable": "v22-stable",
    "v23_logical": "v23-logical",
}
OPERATIONS = ("create", "open", "scan", "take")
WORKER_OPERATIONS = (
    "create",
    "fixture_clone",
    "append",
    "delete",
    "update",
    "merge_insert",
    "backfill",
    "default_compaction_preflight",
    "default_compaction",
    "random_delete_reclaim",
    "normalize_placement",
    "repack",
    "recluster",
    "checkpoint_generation",
    "index_build",
    "index_take",
    "index_optimize",
    "open",
    "scan",
    "take",
)
TIMING_SCOPES = {
    "create": "dataset_write_including_bounded_synthetic_stream_generation",
    "fixture_clone": "cold_session_shallow_clone_of_canonical_fixture",
    "append": "cold_session_open_and_append_including_bounded_synthetic_stream_generation",
    "delete": "cold_session_open_and_delete_commit",
    "update": "cold_session_open_and_update_commit_including_stream_generation_when_applicable",
    "merge_insert": "cold_session_open_and_merge_insert_commit_including_bounded_synthetic_stream_generation",
    "backfill": "cold_session_open_and_row_aligned_backfill_commit",
    "default_compaction_preflight": "cold_session_open_and_default_compaction_plan_only",
    "default_compaction": "cold_session_open_and_default_compaction_commit",
    "random_delete_reclaim": "cold_session_open_and_same_postcondition_random_delete_reclaim_commit",
    "normalize_placement": "cold_session_open_and_normalize_placement_commit",
    "repack": "cold_session_open_and_repack_commit",
    "recluster": "cold_session_open_and_recluster_commit",
    "checkpoint_generation": "cold_session_open_and_generation_checkpoint_commit",
    "index_build": "cold_session_open_and_index_build_commit",
    "index_take": "cold_session_open_and_index_lookup_and_take",
    "index_optimize": "cold_session_open_and_index_optimize_commit",
    "open": "dataset_open_and_contract_validation",
    "scan": "dataset_open_contract_validation_and_full_scan",
    "take": "cold_session_open_and_take_rows_with_prepared_ids",
}
SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")

RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "suite",
        "run_id",
        "pair_id",
        "commit",
        "host",
        "seed",
        "policy_sha256",
        "policy_version",
        "mode",
        "format",
        "storage",
        "operation",
        "timing_scope",
        "round",
        "order_index",
        "dataset_uri",
        "rows",
        "rows_per_fragment",
        "take_count",
        "expected_rows",
        "mutation_count",
        "id_start",
        "step",
        "selection_step",
        "match_percent",
        "schema_kind",
        "index_kind",
        "selection",
        "implementation_path",
        "maintenance_plan_path",
        "maintenance_plan_sha256",
        "started_at_unix_ns",
        "duration_ns",
        "result_rows",
        "dataset_version",
        "fragments",
        "physical_rows",
        "physical_data_bytes",
        "estimated_live_data_bytes",
        "scan_byte_amplification",
        "dataset_bytes",
        "peak_rss_bytes",
        "get_requests",
        "head_requests",
        "list_requests",
        "put_requests",
        "delete_requests",
        "actual_get_attempts",
        "actual_head_attempts",
        "actual_list_attempts",
        "actual_put_attempts",
        "actual_delete_attempts",
        "read_bytes",
        "write_bytes",
        "data_bytes",
        "index_bytes",
        "metadata_bytes",
        "manifest_bytes",
        "placement_root_bytes",
        "placement_delta_bytes",
        "w_epoch_bytes",
        "coverage",
        "recall",
        "admission",
        "placement_maintenance_required",
        "rows_inserted",
        "rows_updated",
        "rows_deleted",
        "compacted_data_bytes",
        "index_storage_bytes_before",
        "row_addresses_remapped",
        "indices_remapped",
        "index_coverage_reuse",
        "layout_index_maintenance_ns",
        "compaction_groups_planned",
        "compaction_groups_admitted",
        "compaction_groups_not_admitted",
        "state_digest",
        "io_by_path",
        "io_metrics_status",
        "status",
        "error",
    }
)

INTEGER_FIELDS = frozenset(
    {
        "schema_version",
        "seed",
        "policy_version",
        "round",
        "order_index",
        "rows",
        "rows_per_fragment",
        "take_count",
        "expected_rows",
        "mutation_count",
        "id_start",
        "step",
        "selection_step",
        "match_percent",
        "started_at_unix_ns",
        "duration_ns",
    }
)
NULLABLE_INTEGER_FIELDS = frozenset(
    {
        "result_rows",
        "dataset_version",
        "fragments",
        "physical_rows",
        "physical_data_bytes",
        "estimated_live_data_bytes",
        "dataset_bytes",
        "peak_rss_bytes",
        "get_requests",
        "head_requests",
        "list_requests",
        "put_requests",
        "delete_requests",
        "actual_get_attempts",
        "actual_head_attempts",
        "actual_list_attempts",
        "actual_put_attempts",
        "actual_delete_attempts",
        "read_bytes",
        "write_bytes",
        "data_bytes",
        "index_bytes",
        "metadata_bytes",
        "manifest_bytes",
        "placement_root_bytes",
        "placement_delta_bytes",
        "w_epoch_bytes",
        "rows_inserted",
        "rows_updated",
        "rows_deleted",
        "compacted_data_bytes",
        "index_storage_bytes_before",
        "row_addresses_remapped",
        "indices_remapped",
        "layout_index_maintenance_ns",
        "compaction_groups_planned",
        "compaction_groups_admitted",
        "compaction_groups_not_admitted",
    }
)
NULLABLE_FLOAT_FIELDS = frozenset(
    {"coverage", "recall", "scan_byte_amplification", "index_coverage_reuse"}
)
NULLABLE_BOOLEAN_FIELDS = frozenset({"admission", "placement_maintenance_required"})
NULLABLE_STRING_FIELDS = frozenset(
    {"maintenance_plan_path", "maintenance_plan_sha256"}
)
REQUEST_FIELDS = frozenset(
    {
        "get_requests",
        "head_requests",
        "list_requests",
        "put_requests",
        "delete_requests",
        "actual_get_attempts",
        "actual_head_attempts",
        "actual_list_attempts",
        "actual_put_attempts",
        "actual_delete_attempts",
        "read_bytes",
        "write_bytes",
    }
)


class RecordValidationError(ValueError):
    """Raised when a worker emits a malformed or misattributed record."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_object(value: Any, fields: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    missing = sorted(fields - value.keys())
    unknown = sorted(value.keys() - fields)
    if missing or unknown:
        raise ValueError(
            f"{context} fields mismatch: missing={missing}, unknown={unknown}"
        )
    return value


def _require_non_empty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def canonical_policy(path: Path) -> tuple[dict[str, Any], bytes, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw = _require_object(
        raw,
        {
            "schema_version",
            "name",
            "trigger",
            "fragment_order",
            "reclaim",
            "target_topology",
            "action",
            "index_postcondition",
            "concurrency",
            "execution",
        },
        "maintenance policy",
    )
    if raw["schema_version"] != 1:
        raise ValueError("maintenance policy schema_version must be 1")
    _require_non_empty_string(raw["name"], "maintenance policy name")

    trigger = _require_object(
        raw["trigger"],
        {"semantics", "evaluation_boundary", "conditions"},
        "maintenance policy trigger",
    )
    if trigger["semantics"] != "any":
        raise ValueError("maintenance policy trigger semantics must be any")
    _require_non_empty_string(
        trigger["evaluation_boundary"], "maintenance policy trigger evaluation_boundary"
    )
    if not isinstance(trigger["conditions"], list) or not trigger["conditions"]:
        raise ValueError(
            "maintenance policy trigger conditions must be a non-empty list"
        )
    trigger_metrics: set[str] = set()
    for index, condition in enumerate(trigger["conditions"]):
        condition = _require_object(
            condition,
            {"metric", "operator", "threshold", "definition"},
            f"maintenance policy trigger condition {index}",
        )
        metric = _require_non_empty_string(
            condition["metric"], f"maintenance policy trigger condition {index} metric"
        )
        if metric in trigger_metrics:
            raise ValueError(
                f"maintenance policy trigger metric is duplicated: {metric}"
            )
        trigger_metrics.add(metric)
        if condition["operator"] not in {"lt", "gt"}:
            raise ValueError(
                f"maintenance policy trigger condition {index} operator must be lt or gt"
            )
        if not isinstance(condition["threshold"], (int, float)) or isinstance(
            condition["threshold"], bool
        ):
            raise ValueError(
                f"maintenance policy trigger condition {index} threshold must be numeric"
            )
        _require_non_empty_string(
            condition["definition"],
            f"maintenance policy trigger condition {index} definition",
        )

    fragment_order = _require_object(
        raw["fragment_order"],
        {"scope", "keys", "tie_breaker", "ambiguous_order"},
        "maintenance policy fragment_order",
    )
    _require_non_empty_string(fragment_order["scope"], "fragment_order scope")
    if not isinstance(fragment_order["keys"], list) or not fragment_order["keys"]:
        raise ValueError("fragment_order keys must be a non-empty list")
    order_fields: set[str] = set()
    for index, key in enumerate(fragment_order["keys"]):
        key = _require_object(
            key, {"field", "direction", "nulls"}, f"fragment_order key {index}"
        )
        field = _require_non_empty_string(
            key["field"], f"fragment_order key {index} field"
        )
        if field in order_fields:
            raise ValueError(f"fragment_order field is duplicated: {field}")
        order_fields.add(field)
        if key["direction"] not in {"ascending", "descending"}:
            raise ValueError(f"fragment_order key {index} direction is invalid")
        if key["nulls"] not in {"first", "last", "forbidden"}:
            raise ValueError(f"fragment_order key {index} nulls is invalid")
    if "fragment_id" not in order_fields:
        raise ValueError(
            "fragment_order must include unique fragment_id in its total order"
        )
    tie_breaker = _require_object(
        fragment_order["tie_breaker"],
        {"field", "direction"},
        "fragment_order tie_breaker",
    )
    _require_non_empty_string(tie_breaker["field"], "fragment_order tie_breaker field")
    if tie_breaker["direction"] not in {"ascending", "descending"}:
        raise ValueError("fragment_order tie_breaker direction is invalid")
    if fragment_order["ambiguous_order"] != "fail":
        raise ValueError("fragment_order ambiguous_order must be fail")

    reclaim = _require_object(
        raw["reclaim"],
        {"deleted_rows", "unreferenced_objects", "high_entropy_delete_action"},
        "maintenance policy reclaim",
    )
    for field, value in reclaim.items():
        _require_non_empty_string(value, f"maintenance policy reclaim {field}")

    topology = _require_object(
        raw["target_topology"],
        {
            "grouping",
            "target_file_size_bytes",
            "max_source_fragments_per_group",
            "output_count",
            "last_output",
            "row_order",
            "allow_cross_logical_fragment_pack",
            "allow_arbitrary_row_reorder",
        },
        "maintenance policy target_topology",
    )
    for field in ("target_file_size_bytes", "max_source_fragments_per_group"):
        if not _is_int(topology[field]) or topology[field] <= 0:
            raise ValueError(f"target_topology {field} must be a positive integer")
    for field in ("grouping", "output_count", "last_output", "row_order"):
        _require_non_empty_string(topology[field], f"target_topology {field}")
    for field in ("allow_cross_logical_fragment_pack", "allow_arbitrary_row_reorder"):
        if not isinstance(topology[field], bool):
            raise ValueError(f"target_topology {field} must be boolean")

    for section, fields in (
        (
            "action",
            {"v22_no_stable", "v22_stable", "v23_logical", "failure"},
        ),
        (
            "index_postcondition",
            {
                "coverage",
                "recall",
                "v22_no_stable",
                "v22_stable",
                "v23_logical",
            },
        ),
    ):
        value = _require_object(raw[section], fields, f"maintenance policy {section}")
        for field, field_value in value.items():
            _require_non_empty_string(
                field_value, f"maintenance policy {section} {field}"
            )

    concurrency = _require_object(
        raw["concurrency"],
        {"maintenance_groups", "writers_per_group"},
        "maintenance policy concurrency",
    )
    for field, value in concurrency.items():
        if not _is_int(value) or value <= 0:
            raise ValueError(f"maintenance policy concurrency {field} must be positive")

    execution = _require_object(
        raw["execution"], {"mode", "tracks", "retry"}, "maintenance policy execution"
    )
    _require_non_empty_string(execution["mode"], "maintenance policy execution mode")
    _require_non_empty_string(execution["retry"], "maintenance policy execution retry")
    tracks = _require_object(
        execution["tracks"],
        {"sustained", "adversarial_natural", "adversarial_aligned"},
        "maintenance policy execution tracks",
    )
    for track_name, track in tracks.items():
        track = _require_object(
            track,
            {"boundary_source", "plan_application", "gate"},
            f"maintenance policy execution track {track_name}",
        )
        for field, value in track.items():
            _require_non_empty_string(
                value, f"maintenance policy execution track {track_name} {field}"
            )

    encoded = json.dumps(
        raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return raw, encoded, hashlib.sha256(encoded).hexdigest()


def format_order(round_index: int, operation_index: int) -> tuple[str, ...]:
    """Rotate the three formats so each occupies every order position."""

    start = (round_index + operation_index) % len(FORMATS)
    return FORMATS[start:] + FORMATS[:start]


def _validate_clean_checkout(commit: str, status: str) -> str:
    commit = commit.strip()
    if SHA_PATTERN.fullmatch(commit) is None:
        raise RuntimeError(f"git returned an invalid full revision: {commit!r}")
    if status.strip():
        raise RuntimeError(
            "benchmark provenance requires a clean checkout; git status reported:\n"
            f"{status.rstrip()}"
        )
    return commit


def source_revision() -> str:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    status = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    return _validate_clean_checkout(commit, status)


def build_harness() -> Path:
    command = (
        "cargo",
        "build",
        "--profile",
        "release-with-debug",
        "--package",
        "lance",
        "--features",
        "metrics",
        "--bench",
        "stable_row_address_e2e",
        "--message-format=json-render-diagnostics",
    )
    print("Building stable_row_address_e2e", file=sys.stderr)
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
    )
    executable: Path | None = None
    diagnostics: list[str] = []
    for line in result.stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("reason") == "compiler-message":
            rendered = message.get("message", {}).get("rendered")
            if rendered:
                diagnostics.append(rendered)
        target = message.get("target", {})
        if (
            message.get("reason") == "compiler-artifact"
            and target.get("name") == "stable_row_address_e2e"
            and message.get("executable")
        ):
            executable = Path(message["executable"]).resolve()
    if result.returncode != 0:
        for diagnostic in diagnostics:
            print(diagnostic, file=sys.stderr, end="")
        raise RuntimeError(f"cargo build failed with exit status {result.returncode}")
    if executable is None or not executable.is_file():
        raise RuntimeError("cargo did not report stable_row_address_e2e executable")
    return executable


def validate_record(
    record: Any, expected: dict[str, Any] | None = None
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise RecordValidationError("worker output must be a JSON object")
    missing = sorted(RECORD_FIELDS - record.keys())
    unknown = sorted(record.keys() - RECORD_FIELDS)
    if missing or unknown:
        raise RecordValidationError(
            f"record schema mismatch: missing={missing}, unknown={unknown}"
        )
    for field in INTEGER_FIELDS:
        if not _is_int(record[field]) or record[field] < 0:
            raise RecordValidationError(f"{field} must be a non-negative integer")
    for field in NULLABLE_INTEGER_FIELDS:
        value = record[field]
        if value is not None and (not _is_int(value) or value < 0):
            raise RecordValidationError(
                f"{field} must be null or a non-negative integer"
            )
    for field in NULLABLE_FLOAT_FIELDS:
        value = record[field]
        if value is not None and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
        ):
            raise RecordValidationError(f"{field} must be null or numeric")
    for field in NULLABLE_BOOLEAN_FIELDS:
        if record[field] is not None and not isinstance(record[field], bool):
            raise RecordValidationError(f"{field} must be null or boolean")
    for field in NULLABLE_STRING_FIELDS:
        if record[field] is not None and (
            not isinstance(record[field], str) or not record[field].strip()
        ):
            raise RecordValidationError(f"{field} must be null or a non-empty string")
    if record["maintenance_plan_sha256"] is not None and SHA256_PATTERN.fullmatch(
        record["maintenance_plan_sha256"]
    ) is None:
        raise RecordValidationError(
            "maintenance_plan_sha256 must be null or a lowercase SHA-256 digest"
        )
    if (record["maintenance_plan_path"] is None) != (
        record["maintenance_plan_sha256"] is None
    ):
        raise RecordValidationError(
            "maintenance_plan_path and maintenance_plan_sha256 must be both null or both present"
        )
    io_by_path = record["io_by_path"]
    if io_by_path is not None:
        expected_categories = {"data", "index", "metadata", "other"}
        if not isinstance(io_by_path, dict) or io_by_path.keys() != expected_categories:
            raise RecordValidationError(
                "io_by_path must be null or contain data/index/metadata/other"
            )
        path_fields = {
            "get_requests",
            "head_requests",
            "list_requests",
            "put_requests",
            "delete_requests",
            "read_bytes",
            "write_bytes",
        }
        for category, values in io_by_path.items():
            if not isinstance(values, dict) or values.keys() != path_fields:
                raise RecordValidationError(
                    f"io_by_path.{category} fields are malformed"
                )
            for field, value in values.items():
                if not _is_int(value) or value < 0:
                    raise RecordValidationError(
                        f"io_by_path.{category}.{field} must be non-negative"
                    )
        for category, record_field in (
            ("data", "data_bytes"),
            ("index", "index_bytes"),
            ("metadata", "metadata_bytes"),
        ):
            expected_bytes = (
                io_by_path[category]["read_bytes"] + io_by_path[category]["write_bytes"]
            )
            if record[record_field] != expected_bytes:
                raise RecordValidationError(
                    f"{record_field} does not match io_by_path.{category}"
                )
        for field in (
            "get_requests",
            "head_requests",
            "list_requests",
            "put_requests",
            "delete_requests",
            "read_bytes",
            "write_bytes",
        ):
            expected_total = sum(values[field] for values in io_by_path.values())
            if record[field] != expected_total:
                raise RecordValidationError(f"{field} does not match io_by_path total")

    if record["schema_version"] != SCHEMA_VERSION or record["suite"] != SUITE:
        raise RecordValidationError("record schema_version or suite is unsupported")
    for field in (
        "run_id",
        "pair_id",
        "commit",
        "host",
        "policy_sha256",
        "mode",
        "format",
        "storage",
        "operation",
        "dataset_uri",
        "timing_scope",
        "schema_kind",
        "index_kind",
        "selection",
        "implementation_path",
        "io_metrics_status",
        "status",
    ):
        if not isinstance(record[field], str) or not record[field].strip():
            raise RecordValidationError(f"{field} must be a non-empty string")
    if record["format"] not in FORMATS:
        raise RecordValidationError(f"unsupported format: {record['format']!r}")
    if record["storage"] not in {"ebs", "s3"}:
        raise RecordValidationError(f"unsupported storage: {record['storage']!r}")
    if record["operation"] not in WORKER_OPERATIONS:
        raise RecordValidationError(f"unsupported operation: {record['operation']!r}")
    if record["mode"] not in {"smoke", "release"}:
        raise RecordValidationError(f"unsupported mode: {record['mode']!r}")
    if record["timing_scope"] != TIMING_SCOPES[record["operation"]]:
        raise RecordValidationError("timing_scope does not match operation")
    if record["order_index"] >= len(FORMATS):
        raise RecordValidationError("order_index must be in 0..3")
    if record["expected_rows"] < 0 or record["mutation_count"] <= 0:
        raise RecordValidationError(
            "expected_rows must be non-negative and mutation_count must be positive"
        )
    if record["match_percent"] > 100:
        raise RecordValidationError("match_percent must be in 0..=100")
    if record["schema_kind"] not in {
        "narrow_16b",
        "wide_128b",
        "vector_f32_128",
    }:
        raise RecordValidationError("schema_kind is unsupported")
    if record["index_kind"] not in {
        "none",
        "scalar_btree",
        "vector_ivf_flat",
    }:
        raise RecordValidationError("index_kind is unsupported")
    if record["selection"] not in {"range", "uniform_without_replacement"}:
        raise RecordValidationError("selection is unsupported")
    state_digest = record["state_digest"]
    if state_digest is not None and (
        not isinstance(state_digest, str)
        or re.fullmatch(r"[0-9a-f]{48}", state_digest) is None
    ):
        raise RecordValidationError(
            "state_digest must be null or 48 lowercase hex digits"
        )
    if (
        record["placement_maintenance_required"] is True
        and record["admission"] is not False
    ):
        raise RecordValidationError(
            "placement_maintenance_required=true requires admission=false"
        )
    if SHA_PATTERN.fullmatch(record["commit"]) is None:
        raise RecordValidationError("commit must be a full lowercase Git SHA")
    if SHA256_PATTERN.fullmatch(record["policy_sha256"]) is None:
        raise RecordValidationError("policy_sha256 must be a lowercase SHA-256 digest")
    if record["status"] == "ok":
        if record["error"] is not None:
            raise RecordValidationError("successful record must have error=null")
    elif record["status"] == "error":
        if not isinstance(record["error"], str) or not record["error"].strip():
            raise RecordValidationError("error record must include an error message")
    else:
        raise RecordValidationError("status must be ok or error")
    if record["io_metrics_status"] == "not_instrumented":
        non_null = sorted(
            field for field in REQUEST_FIELDS if record[field] is not None
        )
        if non_null:
            raise RecordValidationError(
                "not_instrumented record must not publish request metrics: "
                + ", ".join(non_null)
            )
        if record["io_by_path"] is not None:
            raise RecordValidationError(
                "not_instrumented record must have io_by_path=null"
            )
    elif record["io_metrics_status"] == "logical_only":
        logical_fields = REQUEST_FIELDS - {
            "actual_get_attempts",
            "actual_head_attempts",
            "actual_list_attempts",
            "actual_put_attempts",
            "actual_delete_attempts",
        }
        if any(record[field] is None for field in logical_fields):
            raise RecordValidationError(
                "logical_only record is missing logical I/O metrics"
            )
        if any(record[field] is not None for field in REQUEST_FIELDS - logical_fields):
            raise RecordValidationError(
                "logical_only record must not claim actual attempts"
            )
        if record["io_by_path"] is None:
            raise RecordValidationError("logical_only record requires io_by_path")
    elif record["io_metrics_status"] == "measured":
        if any(record[field] is None for field in REQUEST_FIELDS):
            raise RecordValidationError("measured record is missing I/O metrics")
        if record["io_by_path"] is None:
            raise RecordValidationError("measured record requires io_by_path")
    else:
        raise RecordValidationError(
            "io_metrics_status must be not_instrumented, logical_only, or measured"
        )
    if expected:
        mismatches = {
            key: (value, record.get(key))
            for key, value in expected.items()
            if record.get(key) != value
        }
        if mismatches:
            raise RecordValidationError(
                f"worker record provenance mismatch: {mismatches}"
            )
    return record


def parse_worker_stdout(stdout: str, expected: dict[str, Any]) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RecordValidationError(
            f"worker must emit exactly one non-empty stdout line, got {len(lines)}"
        )
    try:
        record = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise RecordValidationError(f"worker emitted invalid JSON: {error}") from error
    return validate_record(record, expected)


def dataset_uri(root: str, run_id: str, round_index: int, format_name: str) -> str:
    suffix = f"{run_id}/round-{round_index:03d}/{format_name}.lance"
    if root.startswith("s3://"):
        return f"{root.rstrip('/')}/{suffix}"
    return str((Path(root).expanduser().resolve() / suffix).resolve())


def run_sidecar_path(output: Path) -> Path:
    return Path(f"{output}.run.json")


def write_run_sidecar(output: Path, payload: dict[str, Any]) -> Path:
    sidecar = run_sidecar_path(output)
    if sidecar.exists():
        raise FileExistsError(f"run sidecar already exists: {sidecar}")
    temporary = sidecar.with_name(f".{sidecar.name}.tmp-{os.getpid()}")
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    )
    try:
        with temporary.open("x", encoding="utf-8") as sink:
            sink.write(encoded)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, sidecar)
    finally:
        temporary.unlink(missing_ok=True)
    return sidecar


def _worker_command(
    executable: Path,
    *,
    uri: str,
    format_name: str,
    storage: str,
    operation: str,
    mode: str,
    run_id: str,
    pair_id: str,
    round_index: int,
    order_index: int,
    rows: int,
    rows_per_fragment: int,
    take_count: int,
    expected_rows: int | None = None,
    mutation_count: int = 1,
    id_start: int = 0,
    step: int = 0,
    selection_step: int = 0,
    match_percent: int = 50,
    schema_kind: str = "narrow16",
    index_kind: str = "none",
    update_driver: str = "native",
    selection: str = "range",
    target_rows_per_fragment: int = 1_000_000,
    target_file_size_bytes: int = 134_217_728,
    max_source_fragments_per_group: int = sys.maxsize,
    seed: int,
    commit: str,
    host: str,
    policy_sha256: str,
    policy_version: int,
    source_dataset_uri: str | None = None,
    take_ids_input: Path | None = None,
    prepare_take_ids_output: Path | None = None,
    prepare_maintenance_plan_output: Path | None = None,
    maintenance_plan_input: Path | None = None,
    maintenance_plan_sha256: str | None = None,
    validate_maintenance_plan_only: bool = False,
) -> tuple[str, ...]:
    command = (
        str(executable),
        "--dataset-uri",
        uri,
        "--format",
        FORMAT_CLI_NAMES[format_name],
        "--storage",
        storage,
        "--operation",
        operation.replace("_", "-"),
        "--mode",
        mode,
        "--run-id",
        run_id,
        "--pair-id",
        pair_id,
        "--round",
        str(round_index),
        "--order-index",
        str(order_index),
        "--rows",
        str(rows),
        "--rows-per-fragment",
        str(rows_per_fragment),
        "--take-count",
        str(take_count),
        "--expected-rows",
        str(rows if expected_rows is None else expected_rows),
        "--mutation-count",
        str(mutation_count),
        "--id-start",
        str(id_start),
        "--step",
        str(step),
        "--selection-step",
        str(selection_step),
        "--match-percent",
        str(match_percent),
        "--schema-kind",
        schema_kind,
        "--index-kind",
        index_kind,
        "--update-driver",
        update_driver,
        "--selection",
        selection,
        "--target-rows-per-fragment",
        str(target_rows_per_fragment),
        "--target-file-size-bytes",
        str(target_file_size_bytes),
        "--max-source-fragments-per-group",
        str(max_source_fragments_per_group),
        "--seed",
        str(seed),
        "--commit",
        commit,
        "--host",
        host,
        "--policy-sha256",
        policy_sha256,
        "--policy-version",
        str(policy_version),
    )
    if source_dataset_uri is not None:
        command += ("--source-dataset-uri", source_dataset_uri)
    if take_ids_input is not None:
        command += ("--take-ids-input", str(take_ids_input))
    if prepare_take_ids_output is not None:
        command += ("--prepare-take-ids-output", str(prepare_take_ids_output))
    if prepare_maintenance_plan_output is not None:
        command += (
            "--prepare-maintenance-plan-output",
            str(prepare_maintenance_plan_output),
        )
    if maintenance_plan_input is not None:
        command += ("--maintenance-plan-input", str(maintenance_plan_input))
    if maintenance_plan_sha256 is not None:
        command += ("--maintenance-plan-sha256", maintenance_plan_sha256)
    if validate_maintenance_plan_only:
        command += ("--validate-maintenance-plan-only",)
    return command


def run_suite(
    executable: Path,
    output: Path,
    dataset_root: str,
    storage: str,
    mode: str,
    rounds: int,
    rows: int,
    rows_per_fragment: int,
    take_count: int,
    seed: int,
    commit: str,
    host: str,
    policy_sha256: str,
    policy_version: int,
    policy: dict[str, Any],
    policy_canonical_json: str,
    operations: Iterable[str] = OPERATIONS,
) -> int:
    operations = tuple(operations)
    if any(operation not in OPERATIONS for operation in operations):
        raise ValueError(f"unsupported operations: {operations}")
    if any(operation != "create" for operation in operations) and (
        "create" not in operations or operations[0] != "create"
    ):
        raise ValueError("read operations require create as the first operation")
    if len(set(operations)) != len(operations):
        raise ValueError("operations must not contain duplicates")
    minimum_rounds = 3 if mode == "smoke" else 10
    if mode not in {"smoke", "release"} or rounds < minimum_rounds:
        raise ValueError(
            f"{mode} mode requires at least {minimum_rounds} paired rounds"
        )
    recomputed_policy = json.dumps(
        policy, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    if recomputed_policy != policy_canonical_json:
        raise ValueError("policy_canonical_json does not match policy")
    if (
        hashlib.sha256(policy_canonical_json.encode("utf-8")).hexdigest()
        != policy_sha256
    ):
        raise ValueError("policy_sha256 does not match policy_canonical_json")

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_id = f"stable-row-address-{timestamp}-{commit[:12]}"
    output.parent.mkdir(parents=True, exist_ok=True)
    take_ids_root = output.parent / f"{output.name}.{run_id}.take_ids"
    sidecar = run_sidecar_path(output)
    if output.exists() or sidecar.exists():
        raise FileExistsError(
            f"benchmark output and sidecar must not exist: output={output}, sidecar={sidecar}"
        )
    any_errors = False
    with output.open("x", encoding="utf-8", buffering=1) as sink:
        write_run_sidecar(
            output,
            {
                "schema_version": 1,
                "suite": SUITE,
                "run_id": run_id,
                "created_at_utc": timestamp,
                "commit": commit,
                "host": host,
                "seed": seed,
                "mode": mode,
                "storage": storage,
                "formats": list(FORMATS),
                "operations": list(operations),
                "rounds": rounds,
                "rows": rows,
                "rows_per_fragment": rows_per_fragment,
                "take_count": take_count,
                "dataset_root": dataset_root,
                "output_jsonl": str(output.resolve()),
                "executable": str(executable.resolve()),
                "data_retention": "preserve",
                "take_ids_root": take_ids_root.name,
                "policy_version": policy_version,
                "policy_sha256": policy_sha256,
                "policy_canonical_json": policy_canonical_json,
                "policy": policy,
            },
        )
        for round_index in range(rounds):
            for operation_index, operation in enumerate(operations):
                pair_id = f"{run_id}/round-{round_index:03d}/{operation}"
                take_id_setups: list[tuple[str, str, int]] = []
                for order_index, format_name in enumerate(
                    format_order(round_index, operation_index)
                ):
                    uri = dataset_uri(dataset_root, run_id, round_index, format_name)
                    expected = {
                        "schema_version": SCHEMA_VERSION,
                        "suite": SUITE,
                        "run_id": run_id,
                        "pair_id": pair_id,
                        "commit": commit,
                        "host": host,
                        "seed": seed,
                        "policy_sha256": policy_sha256,
                        "policy_version": policy_version,
                        "mode": mode,
                        "format": format_name,
                        "storage": storage,
                        "operation": operation,
                        "timing_scope": TIMING_SCOPES[operation],
                        "round": round_index,
                        "order_index": order_index,
                        "dataset_uri": uri,
                        "rows": rows,
                        "rows_per_fragment": rows_per_fragment,
                        "take_count": take_count,
                    }
                    command = _worker_command(
                        executable,
                        uri=uri,
                        format_name=format_name,
                        storage=storage,
                        operation=operation,
                        mode=mode,
                        run_id=run_id,
                        pair_id=pair_id,
                        round_index=round_index,
                        order_index=order_index,
                        rows=rows,
                        rows_per_fragment=rows_per_fragment,
                        take_count=take_count,
                        seed=seed,
                        commit=commit,
                        host=host,
                        policy_sha256=policy_sha256,
                        policy_version=policy_version,
                        take_ids_input=(
                            take_ids_root
                            / f"round-{round_index:03d}"
                            / f"{format_name}.json"
                            if operation == "take"
                            else None
                        ),
                    )
                    print(
                        f"round={round_index} operation={operation} "
                        f"order={order_index} format={format_name}",
                        file=sys.stderr,
                    )
                    result = subprocess.run(
                        command,
                        cwd=REPOSITORY_ROOT,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=None,
                    )
                    if result.returncode != 0:
                        raise RuntimeError(
                            f"worker exited with status {result.returncode}: {command[0]}"
                        )
                    record = parse_worker_stdout(result.stdout, expected)
                    sink.write(
                        json.dumps(record, sort_keys=True, separators=(",", ":"))
                    )
                    sink.write("\n")
                    sink.flush()
                    os.fsync(sink.fileno())
                    any_errors |= record["status"] != "ok"
                    if operation == "create" and record["status"] == "ok":
                        take_id_setups.append((format_name, uri, order_index))
                for format_name, uri, order_index in take_id_setups:
                    take_ids_output = (
                        take_ids_root
                        / f"round-{round_index:03d}"
                        / f"{format_name}.json"
                    )
                    setup_command = _worker_command(
                        executable,
                        uri=uri,
                        format_name=format_name,
                        storage=storage,
                        operation="take",
                        mode=mode,
                        run_id=run_id,
                        pair_id=f"{run_id}/round-{round_index:03d}/prepare-take-ids",
                        round_index=round_index,
                        order_index=order_index,
                        rows=rows,
                        rows_per_fragment=rows_per_fragment,
                        take_count=take_count,
                        seed=seed,
                        commit=commit,
                        host=host,
                        policy_sha256=policy_sha256,
                        policy_version=policy_version,
                        prepare_take_ids_output=take_ids_output,
                    )
                    setup = subprocess.run(
                        setup_command,
                        cwd=REPOSITORY_ROOT,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=None,
                    )
                    if setup.returncode != 0 or setup.stdout.strip():
                        raise RuntimeError(
                            "prepare-take-ids worker failed or emitted stdout: "
                            f"status={setup.returncode}, stdout={setup.stdout!r}"
                        )
    return 1 if any_errors else 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _parse_operations(value: str) -> tuple[str, ...]:
    operations = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(operations) - set(OPERATIONS))
    if not operations or unknown:
        raise argparse.ArgumentTypeError(
            f"operations must be a comma-separated subset of {OPERATIONS}; unknown={unknown}"
        )
    return operations


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--storage", choices=("ebs", "s3"), default="ebs")
    parser.add_argument("--mode", choices=("smoke", "release"), default="release")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--rounds", type=_positive_int)
    parser.add_argument("--rows", type=_positive_int, default=65_536)
    parser.add_argument("--rows-per-fragment", type=_positive_int, default=8_192)
    parser.add_argument("--take-count", type=_positive_int, default=10_000)
    parser.add_argument("--seed", type=int, default=0x4C414E43455F3233)
    parser.add_argument("--host", default=None)
    parser.add_argument("--operations", type=_parse_operations, default=OPERATIONS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.take_count > args.rows:
        raise ValueError("--take-count must not exceed --rows")
    if args.storage == "ebs" and args.dataset_root.startswith("s3://"):
        raise ValueError("--storage=ebs requires a local dataset root")
    if args.storage == "s3" and not args.dataset_root.startswith("s3://"):
        raise ValueError("--storage=s3 requires an s3:// dataset root")

    commit_before_build = source_revision()
    executable = build_harness()
    commit_after_build = source_revision()
    if commit_after_build != commit_before_build:
        raise RuntimeError(
            "source revision changed during build: "
            f"before={commit_before_build}, after={commit_after_build}"
        )
    policy, policy_bytes, policy_sha256 = canonical_policy(args.policy)
    rounds = (
        args.rounds if args.rounds is not None else (3 if args.mode == "smoke" else 10)
    )
    minimum_rounds = 3 if args.mode == "smoke" else 10
    if rounds < minimum_rounds:
        raise ValueError(
            f"--mode={args.mode} requires --rounds >= {minimum_rounds}, got {rounds}"
        )
    return run_suite(
        executable=executable,
        output=args.output.resolve(),
        dataset_root=args.dataset_root,
        storage=args.storage,
        mode=args.mode,
        rounds=rounds,
        rows=args.rows,
        rows_per_fragment=args.rows_per_fragment,
        take_count=args.take_count,
        seed=args.seed,
        commit=commit_after_build,
        host=args.host or socket.gethostname(),
        policy_sha256=policy_sha256,
        policy_version=policy["schema_version"],
        policy=policy,
        policy_canonical_json=policy_bytes.decode("utf-8"),
        operations=args.operations,
    )


if __name__ == "__main__":
    raise SystemExit(main())
