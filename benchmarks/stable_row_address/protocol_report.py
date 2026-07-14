#!/usr/bin/env python3
"""Gate the complete stable logical row-address design protocol evidence."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import protocol  # noqa: E402
import run  # noqa: E402
import environment_attestation  # noqa: E402


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
    "development_tiny",
    "host",
    "seed",
    "profile",
    "cargo_profile",
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
    "storage_region_attestation",
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
    "bounded_recluster",
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
    "bounded_recluster",
    "recluster",
}
DEFAULT_COMPACTION_PREFLIGHT = "default_compaction_preflight"
EXPLICIT_MATRIX_DIAGNOSTIC_OPERATIONS = {
    "random_delete_reclaim",
    "repack",
    "recluster",
}
DIAGNOSTIC_ONLY_GATE_TRACKS = {"adversarial_aligned"}
REPEATED_UPDATE_TRACKS = {
    "sustained",
    "adversarial_natural",
    "adversarial_aligned",
}
PMR_DIAGNOSTIC_FIELDS = (
    "pmr_reason",
    "pmr_projected_delta_bytes",
    "pmr_delta_limit_bytes",
    "pmr_projected_epoch_bytes",
    "pmr_epoch_limit_bytes",
    "pmr_generation_delta_bytes",
    "pmr_generation_epoch_bytes",
    "pmr_blocking_indices",
)
STRUCTURAL_PMR_REASONS = {
    "extent_fanout",
    "existing_explicit_map_requires_rewrite",
    "explicit_map_metadata_required",
    "selection_subtraction_requires_rewrite",
    "packed_run_subtraction_requires_rewrite",
    "logical_order_requires_rewrite",
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
PLACEMENT_METADATA_REQUEST_METRICS = (
    "metadata_get_requests",
    "metadata_head_requests",
    "metadata_list_requests",
)
PROVENANCE_FIELDS = (
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
    "policy_version",
)
CORE_PROVENANCE_FIELDS = tuple(
    field for field in PROVENANCE_FIELDS if field != "order_index"
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
    "bounded_recluster": "cold_session_open_and_bounded_recluster_commit",
    "recluster": "cold_session_open_and_recluster_commit",
    "checkpoint_generation": "cold_session_open_and_generation_checkpoint_commit",
    "index_build": "cold_session_open_and_index_build_commit",
    "index_take": "cold_session_open_and_index_lookup_and_take",
    "index_optimize": "cold_session_open_and_index_optimize_commit",
    "open": "dataset_open_and_contract_validation",
    "scan": "dataset_open_contract_validation_and_full_scan",
    "take": "cold_session_open_and_take_rows_with_prepared_ids",
}
SCHEMA_NAMES = {
    "narrow16": "narrow_16b",
    "wide128": "wide_128b",
    "vector": "vector_f32_128",
}
INDEX_NAMES = {
    "none": "none",
    "scalar": "scalar_btree",
    "vector": "vector_ivf_flat",
}
SELECTION_NAMES = {
    "range": "range",
    "random": "uniform_without_replacement",
}


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

    @property
    def aggregate_release_gate(self) -> bool:
        return self.track not in DIAGNOSTIC_ONLY_GATE_TRACKS

    def as_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self) | {
            "contract": self.contract,
            "aggregate_release_gate": self.aggregate_release_gate,
        }


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


@functools.cache
def frozen_matrix() -> tuple[dict[str, Any], str, str]:
    matrix = json.loads(protocol.DEFAULT_MATRIX.read_text(encoding="utf-8"))
    canonical = json.dumps(
        matrix, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return matrix, canonical, hashlib.sha256(canonical.encode()).hexdigest()


@functools.cache
def frozen_release_policy() -> tuple[str, str]:
    policy = json.loads(run.DEFAULT_POLICY.read_text(encoding="utf-8"))
    canonical = json.dumps(
        policy, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return canonical, hashlib.sha256(canonical.encode()).hexdigest()


def canonical_dataset_root(root: Any, storage: Any) -> str:
    if not isinstance(root, str) or not root.strip():
        raise ValueError("dataset root must be a non-empty string")
    if storage == "s3":
        if not root.startswith("s3://"):
            raise ValueError("storage=s3 requires s3:// dataset roots")
        return root.rstrip("/")
    if storage == "ebs":
        if root.startswith("s3://"):
            raise ValueError("storage=ebs requires local dataset roots")
        return str(Path(root).expanduser().resolve())
    raise ValueError("protocol sidecar storage is unsupported")


def frozen_dataset_root(sidecar: dict[str, Any]) -> str:
    base = canonical_dataset_root(
        sidecar.get("base_dataset_root"), sidecar.get("storage")
    )
    if sidecar["shard_count"] == 1:
        return base
    return f"{base}/{sidecar['shard_id']}"


def validate_dataset_root_binding(sidecar: dict[str, Any]) -> str:
    actual = canonical_dataset_root(sidecar.get("dataset_root"), sidecar.get("storage"))
    expected = frozen_dataset_root(sidecar)
    if actual != expected:
        raise ValueError(
            "protocol sidecar dataset_root is inconsistent with its shard: "
            f"expected={expected!r}, actual={actual!r}"
        )
    return actual


def validate_storage_roots(sidecar: dict[str, Any]) -> None:
    storage = sidecar.get("storage")
    roots = (sidecar.get("base_dataset_root"), sidecar.get("dataset_root"))
    if storage == "s3" and any(
        not isinstance(root, str) or not root.startswith("s3://") for root in roots
    ):
        raise ValueError("storage=s3 requires s3:// dataset roots")
    if storage == "ebs" and any(
        isinstance(root, str) and root.startswith("s3://") for root in roots
    ):
        raise ValueError("storage=ebs requires local dataset roots")


@functools.cache
def frozen_release_shard_contract(
    shard_count: int, shard_index: int
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if shard_count != 9 or shard_index not in range(9):
        raise ValueError("release evidence requires exactly nine canonical shards")
    matrix, _, _ = frozen_matrix()
    profile = matrix["profiles"]["release"]
    release_tracks = (
        "matrix",
        "sustained",
        "adversarial_natural",
        "adversarial_aligned",
    )
    release_variants = ("bare", "scalar", "vector")
    cases = tuple(
        protocol.iter_matrix_cases(profile, set(matrix["tracks"]["matrix"]["cases"]))
    )
    fixture_keys = {protocol.fixture_key_for_case(case) for case in cases}
    repeated_rows_per_fragment = max(
        1,
        (profile["rows"] + profile["logical_fragment_counts"][0] - 1)
        // profile["logical_fragment_counts"][0],
    )
    variant_layouts = {
        "bare": (
            "narrow16",
            ((profile["rows"], repeated_rows_per_fragment),),
            "none",
        ),
        "scalar": (
            "narrow16",
            ((profile["rows"], repeated_rows_per_fragment),),
            "scalar",
        ),
        "vector": (
            "vector",
            ((profile["rows"], repeated_rows_per_fragment),),
            "vector",
        ),
    }
    fixture_keys.update(variant_layouts.values())
    data_layouts = sorted(
        {(schema_kind, segments) for schema_kind, segments, _ in fixture_keys}
    )
    selected_data_layouts = {
        layout
        for ordinal, layout in enumerate(data_layouts)
        if ordinal % shard_count == shard_index
    }
    selected_fixture_keys = {
        key for key in fixture_keys if (key[0], key[1]) in selected_data_layouts
    }
    matrix_case_names = tuple(
        case.name
        for case in cases
        if protocol.fixture_key_for_case(case) in selected_fixture_keys
    )
    variants = tuple(
        variant
        for variant in release_variants
        if variant_layouts[variant] in selected_fixture_keys
    )
    tracks = tuple(
        track
        for track in release_tracks
        if (track == "matrix" and matrix_case_names) or (track != "matrix" and variants)
    )
    return tracks, variants, matrix_case_names


def validate_frozen_release_selection(sidecar: dict[str, Any]) -> None:
    if sidecar.get("profile") != "release":
        return
    expected_tracks, expected_variants, expected_cases = frozen_release_shard_contract(
        sidecar["shard_count"], sidecar["shard_index"]
    )
    for field, expected in (
        ("tracks", expected_tracks),
        ("variants", expected_variants),
        ("matrix_case_names", expected_cases),
    ):
        actual = sidecar.get(field)
        if not isinstance(actual, list) or tuple(actual) != expected:
            raise ValueError(
                f"release {field} does not match the frozen shard allocation"
            )


def validate_frozen_release_identity(sidecar: dict[str, Any]) -> None:
    if sidecar.get("profile") != "release":
        return
    if sidecar.get("seed") != 0x4C414E43455F3233:
        raise ValueError("release evidence does not use the canonical seed")
    canonical, sha256 = frozen_release_policy()
    actual = json.dumps(
        sidecar.get("policy"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    if (
        actual != canonical
        or sidecar.get("policy_canonical_json") != canonical
        or sidecar.get("policy_sha256") != sha256
    ):
        raise ValueError("release evidence does not use the repository default policy")


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
        "cargo_profile",
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
    if not isinstance(value["development_tiny"], bool):
        raise ValueError("protocol sidecar development_tiny must be a boolean")
    if value["development_tiny"] and (
        value["profile"] != "smoke"
        or value["source_provenance"] != "dirty-development-override"
    ):
        raise ValueError(
            "development_tiny evidence requires dirty-development smoke provenance"
        )
    if (
        value["profile"] == "release"
        and value["source_provenance"] != "clean-committed-source"
    ):
        raise ValueError("release evidence requires clean committed source")
    if value["profile"] not in {"smoke", "release"}:
        raise ValueError("protocol sidecar profile is unsupported")
    if value["cargo_profile"] != run.CARGO_PROFILE:
        raise ValueError(f"protocol sidecar cargo_profile must be {run.CARGO_PROFILE}")
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
    validate_storage_roots(value)
    validate_dataset_root_binding(value)
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
    if value["profile"] == "release":
        environment_attestation.validate_same_region_s3_attestation(
            value["storage_region_attestation"], value["base_dataset_root"]
        )
    elif value["storage_region_attestation"] is not None:
        raise ValueError("smoke evidence must not claim a release region attestation")
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
    if ("matrix" in value["tracks"]) != bool(value["matrix_case_names"]):
        raise ValueError("matrix track and matrix_case_names must be present together")

    matrix_canonical = json.dumps(
        value["matrix"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    if matrix_canonical != value["matrix_canonical_json"]:
        raise ValueError("matrix canonical JSON does not match sidecar object")
    if hashlib.sha256(matrix_canonical.encode()).hexdigest() != value["matrix_sha256"]:
        raise ValueError("matrix SHA-256 does not match canonical JSON")
    frozen, frozen_canonical, frozen_sha256 = frozen_matrix()
    expected_matrix = (
        protocol.development_tiny_matrix(frozen)
        if value["development_tiny"]
        else frozen
    )
    expected_canonical = (
        json.dumps(
            expected_matrix,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        if value["development_tiny"]
        else frozen_canonical
    )
    expected_sha256 = (
        hashlib.sha256(expected_canonical.encode()).hexdigest()
        if value["development_tiny"]
        else frozen_sha256
    )
    if (
        matrix_canonical != expected_canonical
        or value["matrix_sha256"] != expected_sha256
    ):
        raise ValueError(
            "protocol sidecar does not contain the expected frozen workload matrix"
        )
    validate_frozen_release_selection(value)
    validate_frozen_release_identity(value)
    # Re-run the complete matrix validator against the repository-owned object.
    temporary_matrix = expected_matrix
    _strict_object(
        temporary_matrix,
        {"schema_version", "name", "profiles", "tracks", "measurement"},
        "embedded matrix",
    )
    profile = temporary_matrix["profiles"].get(value["profile"])
    if not isinstance(profile, dict):
        raise ValueError("selected profile is missing from matrix")
    expected_reclaim_admission = {
        "smoke": "must_admit",
        "release": "must_not_admit",
    }[value["profile"]]
    if profile.get("random_delete_reclaim_admission") != expected_reclaim_admission:
        raise ValueError(
            "selected profile random_delete_reclaim_admission does not match "
            f"{expected_reclaim_admission}"
        )
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
                or group["source_live_data_bytes"] > group["source_physical_data_bytes"]
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
            expected_output_live_rows = [expected_target] * (
                live_rows // expected_target
            )
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
    if metric.startswith("metadata_") and metric.endswith("_requests"):
        request_kind = metric.removeprefix("metadata_")
        if request_kind not in {"get_requests", "head_requests", "list_requests"}:
            raise ValueError(f"unsupported metadata request metric: {metric}")
        return record["io_by_path"]["metadata"][request_kind]
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


def _implementation_path(
    operation: str,
    format_name: str,
    index_kind: str,
    selection: str,
    compaction_mode: str,
) -> str:
    if operation == "update":
        return (
            "exact_selection_matched_merge"
            if selection == "random"
            else "native_update_builder"
        )
    if operation == "fixture_clone":
        return "canonical_fixture_shallow_clone"
    if operation in {
        "normalize_placement",
        "repack",
        "recluster",
        "checkpoint_generation",
    }:
        return "capability_gated_explicit_maintenance"
    if operation == "default_compaction_preflight":
        return "default_compaction_plan_only"
    if operation == "default_compaction" and compaction_mode == "fragment_reuse":
        return {
            "v22_no_stable": "deferred_fragment_reuse_compaction",
            "v22_stable": "inline_index_remap_compaction",
            "v23_logical": "stable_logical_zero_remap_compaction",
        }[format_name]
    if operation == "default_compaction":
        return "default_compaction"
    if operation == "random_delete_reclaim":
        if format_name == "v23_logical":
            return "explicit_repack"
        if format_name == "v22_stable" and index_kind != "none":
            return "same_postcondition_default_compaction_full_index_rebuild"
        return "same_postcondition_default_compaction"
    if operation == "bounded_recluster":
        return (
            "default_bounded_recluster_fast_path"
            if format_name == "v23_logical"
            else "same_postcondition_bounded_recluster_rewrite"
        )
    if operation in {"index_build", "index_take", "index_optimize"}:
        return INDEX_NAMES[index_kind]
    return "native_dataset_api"


def _dataset_uri(root: str, suffix: str) -> str:
    if root.startswith("s3://"):
        return f"{root.rstrip('/')}/{suffix}"
    return str((Path(root).expanduser().resolve() / suffix).resolve())


def _expected_provenance(
    *,
    operation: str,
    format_name: str,
    dataset_uri: str,
    repeat: int,
    order_index: int,
    rows: int,
    rows_per_fragment: int,
    take_count: int,
    expected_rows: int,
    mutation_count: int = 1,
    id_start: int = 0,
    step: int = 0,
    selection_step: int = 0,
    match_percent: int = 50,
    schema_kind: str = "narrow16",
    index_kind: str = "none",
    selection: str = "range",
    compaction_mode: str = "standard",
) -> dict[str, Any]:
    return {
        "operation": operation,
        "timing_scope": TIMING_SCOPES[operation],
        "round": repeat,
        "order_index": order_index,
        "dataset_uri": dataset_uri,
        "rows": rows,
        "rows_per_fragment": rows_per_fragment,
        "take_count": take_count,
        "expected_rows": expected_rows,
        "mutation_count": mutation_count,
        "id_start": id_start,
        "step": step,
        "selection_step": selection_step,
        "match_percent": match_percent,
        "schema_kind": SCHEMA_NAMES[schema_kind],
        "index_kind": INDEX_NAMES[index_kind],
        "selection": SELECTION_NAMES[selection],
        "implementation_path": _implementation_path(
            operation, format_name, index_kind, selection, compaction_mode
        ),
        "policy_version": 1,
    }


@functools.lru_cache(maxsize=None)
def _expected_record_provenance(
    run_id: str,
    dataset_root: str,
    profile_name: str,
    tracks: tuple[str, ...],
    variants: tuple[str, ...],
    matrix_case_names: tuple[str, ...],
    development_tiny: bool,
    optional_pairs: frozenset[str] | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    matrix, _, _ = frozen_matrix()
    if development_tiny:
        matrix = protocol.development_tiny_matrix(matrix)
    profile = matrix["profiles"][profile_name]
    rows = profile["rows"]
    repeats = profile["paired_repeats"]
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    seen_fixtures: set[protocol.FixtureKey] = set()
    current_take_count = profile["take_counts"][-1]

    def paired_order(repeat: int, pair_id: str) -> tuple[str, ...]:
        return run.paired_format_order(repeat, pair_id)

    def dynamic_order(repeat: int, scope: str) -> tuple[str, ...]:
        return run.dynamic_format_order(repeat, scope)

    def include_optional(pair_id: str) -> bool:
        return optional_pairs is None or pair_id in optional_pairs

    def workload_uri(track: str, case: str, repeat: int, format_name: str) -> str:
        suffix = (
            f"{run_id}/{track}/{case.replace('/', '__')}/"
            f"repeat-{repeat:03d}/{format_name}.lance"
        )
        return _dataset_uri(dataset_root, suffix)

    def fixture_uri(
        schema_kind: str,
        segments: tuple[tuple[int, int], ...],
        index_kind: str,
        format_name: str,
    ) -> str:
        suffix = (
            f"{run_id}/fixtures/{schema_kind}/{protocol.fixture_layout_path(segments)}/"
            f"index-{index_kind}/"
            f"{format_name}.lance"
        )
        return _dataset_uri(dataset_root, suffix)

    def add(
        pair_id: str,
        formats: Sequence[str],
        *,
        operation: str,
        repeat: int,
        rows_per_fragment: int,
        take_count: int,
        expected_rows: int,
        uri_for_format: Callable[[str], str],
        mutation_count: int = 1,
        id_start: int = 0,
        step: int = 0,
        selection_step: int = 0,
        match_percent: int = 50,
        schema_kind: str = "narrow16",
        index_kind: str = "none",
        selection: str = "range",
        compaction_mode: str = "standard",
        order_indices: Sequence[int] | None = None,
        configured_rows: int | None = None,
    ) -> None:
        if order_indices is None:
            order_indices = range(len(formats))
        if len(order_indices) != len(formats):
            raise ValueError(f"invalid frozen order for {pair_id}")
        for format_name, order_index in zip(formats, order_indices):
            value = _expected_provenance(
                operation=operation,
                format_name=format_name,
                dataset_uri=uri_for_format(format_name),
                repeat=repeat,
                order_index=order_index,
                rows=rows if configured_rows is None else configured_rows,
                rows_per_fragment=rows_per_fragment,
                take_count=take_count,
                expected_rows=expected_rows,
                mutation_count=mutation_count,
                id_start=id_start,
                step=step,
                selection_step=selection_step,
                match_percent=match_percent,
                schema_kind=schema_kind,
                index_kind=index_kind,
                selection=selection,
                compaction_mode=compaction_mode,
            )
            key = (pair_id, format_name)
            previous = expected.setdefault(key, value)
            if previous != value:
                raise ValueError(f"conflicting frozen provenance for {key}")

    def add_fixture(
        schema_kind: str,
        segments: tuple[tuple[int, int], ...],
        index_kind: str,
        take_count: int,
    ) -> None:
        key: protocol.FixtureKey = (schema_kind, segments, index_kind)
        if key in seen_fixtures:
            return
        total_rows = sum(segment_rows for segment_rows, _ in segments)
        rows_per_fragment = segments[0][1]
        if index_kind != "none":
            add_fixture(schema_kind, segments, "none", take_count)
            prefix = (
                f"{run_id}/fixtures/{schema_kind}/{protocol.fixture_layout_path(segments)}/"
                f"index-{index_kind}"
            )
            add(
                f"{prefix}/fixture_clone",
                paired_order(0, f"{prefix}/fixture_clone"),
                operation="fixture_clone",
                repeat=0,
                rows_per_fragment=rows_per_fragment,
                take_count=take_count,
                expected_rows=total_rows,
                schema_kind=schema_kind,
                uri_for_format=lambda format_name: fixture_uri(
                    schema_kind, segments, index_kind, format_name
                ),
            )
            add(
                f"{prefix}/index_build",
                paired_order(0, f"{prefix}/index_build"),
                operation="index_build",
                repeat=0,
                rows_per_fragment=rows_per_fragment,
                take_count=take_count,
                expected_rows=total_rows,
                schema_kind=schema_kind,
                index_kind=index_kind,
                uri_for_format=lambda format_name: fixture_uri(
                    schema_kind, segments, index_kind, format_name
                ),
            )
        else:
            prefix = (
                f"{run_id}/fixtures/{schema_kind}/{protocol.fixture_layout_path(segments)}/"
                "index-none"
            )
            cumulative_rows = 0
            for segment_index, (segment_rows, segment_rows_per_fragment) in enumerate(
                segments
            ):
                operation = "create" if segment_index == 0 else "append"
                cumulative_before = cumulative_rows
                cumulative_rows += segment_rows
                label = (
                    "create" if segment_index == 0 else f"append-{segment_index:03d}"
                )
                add(
                    f"{prefix}/{label}",
                    paired_order(0, f"{prefix}/{label}"),
                    operation=operation,
                    repeat=0,
                    rows_per_fragment=segment_rows_per_fragment,
                    take_count=take_count,
                    expected_rows=cumulative_rows,
                    mutation_count=(segment_rows if operation == "append" else 1),
                    id_start=cumulative_before,
                    schema_kind=schema_kind,
                    configured_rows=segment_rows,
                    uri_for_format=lambda format_name: fixture_uri(
                        schema_kind, segments, "none", format_name
                    ),
                )
        seen_fixtures.add(key)

    def add_probes(
        track: str,
        case: str,
        repeat: int,
        *,
        expected_rows: int,
        rows_per_fragment: int,
        take_count: int,
        schema_kind: str,
        index_kind: str,
        step: int,
    ) -> None:
        prefix = f"{run_id}/{track}/{case}/repeat-{repeat:03d}/step-{step:03d}"
        operations = [
            ("cold-open", "open"),
            ("cold-scan", "scan"),
            ("cold-take", "take"),
        ]
        if index_kind != "none":
            operations.append(("cold-index-take", "index_take"))
        for label, operation in operations:
            add(
                f"{prefix}/{label}",
                paired_order(repeat, f"{prefix}/{label}"),
                operation=operation,
                repeat=repeat,
                rows_per_fragment=rows_per_fragment,
                take_count=take_count,
                expected_rows=expected_rows,
                step=step,
                schema_kind=schema_kind,
                index_kind=index_kind,
                uri_for_format=lambda format_name: workload_uri(
                    track, case, repeat, format_name
                ),
            )

    def add_step(
        pair_id: str,
        formats: Sequence[str],
        step_value: protocol.Step,
        *,
        operation: str | None,
        repeat: int,
        rows_per_fragment: int,
        take_count: int,
        track: str,
        case: str,
        step_number: int | None = None,
    ) -> None:
        add(
            pair_id,
            formats,
            operation=operation or step_value.operation,
            repeat=repeat,
            rows_per_fragment=rows_per_fragment,
            take_count=take_count,
            expected_rows=step_value.expected_rows,
            mutation_count=step_value.mutation_count,
            id_start=step_value.id_start,
            step=step_value.step if step_number is None else step_number,
            selection_step=step_value.selection_step,
            match_percent=step_value.match_percent,
            schema_kind=step_value.schema_kind,
            index_kind=step_value.index_kind,
            selection=step_value.selection,
            compaction_mode=step_value.compaction_mode,
            uri_for_format=lambda format_name: workload_uri(
                track, case, repeat, format_name
            ),
        )

    all_cases = {
        case.name: case
        for case in protocol.iter_matrix_cases(
            profile, set(matrix["tracks"]["matrix"]["cases"])
        )
    }
    repeated_fragments = profile["logical_fragment_counts"][0]
    repeated_rows_per_fragment = max(
        1, (rows + repeated_fragments - 1) // repeated_fragments
    )
    variant_config = {
        "bare": ("narrow16", "none"),
        "scalar": ("narrow16", "scalar"),
        "vector": ("vector", "vector"),
    }

    for track in tracks:
        if track == "matrix":
            for case_name in matrix_case_names:
                case = all_cases[case_name]
                current_take_count = case.take_count
                add_fixture(
                    case.schema_kind,
                    protocol.fixture_segments_for_case(case),
                    case.fixture_index_kind,
                    current_take_count,
                )
                for repeat in range(repeats):
                    prefix = f"{run_id}/matrix/{case_name}/repeat-{repeat:03d}"
                    add(
                        f"{prefix}/step-000/fixture_clone",
                        paired_order(repeat, f"{prefix}/step-000/fixture_clone"),
                        operation="fixture_clone",
                        repeat=repeat,
                        rows_per_fragment=case.rows_per_fragment,
                        take_count=current_take_count,
                        expected_rows=rows,
                        schema_kind=case.schema_kind,
                        index_kind=case.fixture_index_kind,
                        uri_for_format=lambda format_name,
                        case_name=case_name,
                        repeat=repeat: (
                            workload_uri("matrix", case_name, repeat, format_name)
                        ),
                    )
                    for step_index, step_value in enumerate(case.steps[1:], 1):
                        if step_value.operation == "random_delete_reclaim":
                            add_probes(
                                "matrix",
                                case_name,
                                repeat,
                                expected_rows=step_value.expected_rows,
                                rows_per_fragment=case.rows_per_fragment,
                                take_count=current_take_count,
                                schema_kind=case.schema_kind,
                                index_kind=step_value.index_kind,
                                step=step_index,
                            )
                        if step_value.preflight_expected_admission is not None:
                            preflight_label = (
                                "default-reclaim-preflight"
                                if step_value.operation == "random_delete_reclaim"
                                else "default-compaction-preflight"
                            )
                            add_step(
                                f"{prefix}/step-{step_index:03d}/{preflight_label}",
                                ("v23_logical",),
                                step_value,
                                operation="default_compaction_preflight",
                                repeat=repeat,
                                rows_per_fragment=case.rows_per_fragment,
                                take_count=current_take_count,
                                track="matrix",
                                case=case_name,
                            )
                        add_step(
                            f"{prefix}/step-{step_index:03d}/{step_value.operation}",
                            (
                                ("v23_logical",)
                                if step_value.operation == "recluster"
                                else paired_order(
                                    repeat,
                                    f"{prefix}/step-{step_index:03d}/{step_value.operation}",
                                )
                            ),
                            step_value,
                            operation=None,
                            repeat=repeat,
                            rows_per_fragment=case.rows_per_fragment,
                            take_count=current_take_count,
                            track="matrix",
                            case=case_name,
                            step_number=step_index,
                        )
                    final = case.steps[-1]
                    add_probes(
                        "matrix",
                        case_name,
                        repeat,
                        expected_rows=final.expected_rows,
                        rows_per_fragment=case.rows_per_fragment,
                        take_count=current_take_count,
                        schema_kind=case.schema_kind,
                        index_kind=final.index_kind,
                        step=len(case.steps),
                    )
            continue

        if track not in {
            "sustained",
            "adversarial_natural",
            "adversarial_aligned",
        }:
            continue
        for variant in variants:
            schema_kind, index_kind = variant_config[variant]
            add_fixture(
                schema_kind,
                ((rows, repeated_rows_per_fragment),),
                index_kind,
                current_take_count,
            )
            for repeat in range(repeats):
                prefix = f"{run_id}/{track}/{variant}/repeat-{repeat:03d}"
                add(
                    f"{prefix}/setup/fixture-clone",
                    paired_order(repeat, f"{prefix}/setup/fixture-clone"),
                    operation="fixture_clone",
                    repeat=repeat,
                    rows_per_fragment=repeated_rows_per_fragment,
                    take_count=current_take_count,
                    expected_rows=rows,
                    schema_kind=schema_kind,
                    index_kind=index_kind,
                    uri_for_format=lambda format_name,
                    track=track,
                    variant=variant,
                    repeat=repeat: (workload_uri(track, variant, repeat, format_name)),
                )
                for update_round in range(profile["repeated_update_rounds"]):
                    update_selection_step = 0 if track == "sustained" else update_round
                    update_kwargs = {
                        "repeat": repeat,
                        "rows_per_fragment": repeated_rows_per_fragment,
                        "take_count": current_take_count,
                        "expected_rows": rows,
                        "mutation_count": profile["hot_set_rows"],
                        "step": update_round,
                        "selection_step": update_selection_step,
                        "schema_kind": schema_kind,
                        "index_kind": index_kind,
                        "selection": "random",
                        "uri_for_format": lambda format_name,
                        track=track,
                        variant=variant,
                        repeat=repeat: (
                            workload_uri(track, variant, repeat, format_name)
                        ),
                    }
                    if track == "sustained":
                        add(
                            f"{prefix}/round-{update_round:03d}/update",
                            paired_order(
                                repeat, f"{prefix}/round-{update_round:03d}/update"
                            ),
                            operation="update",
                            **update_kwargs,
                        )
                    elif track == "adversarial_natural":
                        add(
                            f"{prefix}/round-{update_round:03d}/update-attempt",
                            paired_order(
                                repeat,
                                f"{prefix}/round-{update_round:03d}/update-attempt",
                            ),
                            operation="update",
                            **update_kwargs,
                        )
                        pmr_pair = f"{prefix}/round-{update_round:03d}/pmr-maintenance"
                        pmr_order_scope = (
                            f"adversarial_natural/{variant}/repeat-{repeat:03d}/"
                            f"round-{update_round:03d}/pmr-maintenance"
                        )
                        pmr_order = dynamic_order(repeat, pmr_order_scope)
                        candidate_order_index = pmr_order.index("v23_logical")
                        if include_optional(pmr_pair):
                            add(
                                pmr_pair,
                                ("v23_logical",),
                                operation="normalize_placement",
                                order_indices=(candidate_order_index,),
                                **update_kwargs,
                            )
                        retry_pair = f"{prefix}/round-{update_round:03d}/update-retry"
                        if include_optional(retry_pair):
                            add(
                                retry_pair,
                                ("v23_logical",),
                                operation="update",
                                order_indices=(candidate_order_index,),
                                **update_kwargs,
                            )
                    else:
                        add(
                            f"{prefix}/round-{update_round:03d}/candidate-preflight",
                            ("v23_logical",),
                            operation="update",
                            **update_kwargs,
                        )
                        normalize_pair = f"{prefix}/round-{update_round:03d}/normalize"
                        aligned_scope = (
                            f"adversarial_aligned/{variant}/repeat-{repeat:03d}/"
                            f"round-{update_round:03d}/aligned-maintenance"
                        )
                        aligned_order = dynamic_order(repeat, aligned_scope)
                        if include_optional(normalize_pair):
                            add(
                                normalize_pair,
                                ("v23_logical",),
                                operation="normalize_placement",
                                order_indices=(aligned_order.index("v23_logical"),),
                                **update_kwargs,
                            )
                        for format_name in ("v22_no_stable", "v22_stable"):
                            baseline_pair = (
                                f"{prefix}/round-{update_round:03d}/"
                                f"forced-baseline-maintenance/{format_name}"
                            )
                            if include_optional(baseline_pair):
                                add(
                                    baseline_pair,
                                    (format_name,),
                                    operation="default_compaction",
                                    order_indices=(aligned_order.index(format_name),),
                                    **update_kwargs,
                                )
                        retry_pair = (
                            f"{prefix}/round-{update_round:03d}/candidate-retry"
                        )
                        if include_optional(retry_pair):
                            add(
                                retry_pair,
                                ("v23_logical",),
                                operation="update",
                                order_indices=(aligned_order.index("v23_logical"),),
                                **update_kwargs,
                            )
                        add(
                            f"{prefix}/round-{update_round:03d}/baseline-update",
                            ("v22_no_stable", "v22_stable"),
                            operation="update",
                            **update_kwargs,
                        )
                    if index_kind != "none":
                        add(
                            f"{prefix}/round-{update_round:03d}/index-catch-up",
                            paired_order(
                                repeat,
                                f"{prefix}/round-{update_round:03d}/index-catch-up",
                            ),
                            operation="index_optimize",
                            **update_kwargs,
                        )
                    add_probes(
                        track,
                        variant,
                        repeat,
                        expected_rows=rows,
                        rows_per_fragment=repeated_rows_per_fragment,
                        take_count=current_take_count,
                        schema_kind=schema_kind,
                        index_kind=index_kind,
                        step=update_round,
                    )
                    if track == "sustained":
                        maintenance_pair = (
                            f"{prefix}/round-{update_round:03d}/policy-maintenance"
                        )
                        if include_optional(maintenance_pair):
                            add(
                                maintenance_pair,
                                paired_order(repeat, maintenance_pair),
                                operation="default_compaction",
                                repeat=repeat,
                                rows_per_fragment=repeated_rows_per_fragment,
                                take_count=current_take_count,
                                expected_rows=rows,
                                step=update_round,
                                schema_kind=schema_kind,
                                index_kind=index_kind,
                                uri_for_format=update_kwargs["uri_for_format"],
                            )
                            add_probes(
                                track,
                                variant,
                                repeat,
                                expected_rows=rows,
                                rows_per_fragment=repeated_rows_per_fragment,
                                take_count=current_take_count,
                                schema_kind=schema_kind,
                                index_kind=index_kind,
                                step=profile["repeated_update_rounds"] + update_round,
                            )
                    elif track == "adversarial_natural":
                        natural_scope = (
                            f"{prefix}/round-{update_round:03d}/natural-maintenance"
                        )
                        natural_order_scope = (
                            f"adversarial_natural/{variant}/repeat-{repeat:03d}/"
                            f"round-{update_round:03d}/natural-maintenance"
                        )
                        natural_order = dynamic_order(repeat, natural_order_scope)
                        for format_name in run.FORMATS:
                            maintenance_pair = f"{natural_scope}/{format_name}"
                            if include_optional(maintenance_pair):
                                add(
                                    maintenance_pair,
                                    (format_name,),
                                    operation="default_compaction",
                                    repeat=repeat,
                                    rows_per_fragment=repeated_rows_per_fragment,
                                    take_count=current_take_count,
                                    expected_rows=rows,
                                    step=update_round,
                                    schema_kind=schema_kind,
                                    index_kind=index_kind,
                                    order_indices=(natural_order.index(format_name),),
                                    uri_for_format=update_kwargs["uri_for_format"],
                                )
                        add_probes(
                            track,
                            variant,
                            repeat,
                            expected_rows=rows,
                            rows_per_fragment=repeated_rows_per_fragment,
                            take_count=current_take_count,
                            schema_kind=schema_kind,
                            index_kind=index_kind,
                            step=profile["repeated_update_rounds"] + update_round,
                        )
    return expected


def _schedule_arguments(sidecar: dict[str, Any]) -> tuple[Any, ...]:
    matrix, canonical, sha256 = frozen_matrix()
    development_tiny = sidecar.get("development_tiny")
    if not isinstance(development_tiny, bool):
        raise ValueError("sidecar development_tiny must be a boolean")
    if development_tiny:
        matrix = protocol.development_tiny_matrix(matrix)
        canonical = json.dumps(
            matrix, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        sha256 = hashlib.sha256(canonical.encode()).hexdigest()
    embedded = json.dumps(
        sidecar.get("matrix"), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    if (
        embedded != canonical
        or sidecar.get("matrix_canonical_json") != canonical
        or sidecar.get("matrix_sha256") != sha256
    ):
        raise ValueError("sidecar workload does not match the frozen matrix")
    profile_name = sidecar.get("profile")
    if profile_name not in {"smoke", "release"}:
        raise ValueError("sidecar profile is unsupported")
    validate_storage_roots(sidecar)
    dataset_root = validate_dataset_root_binding(sidecar)
    if ("matrix" in sidecar["tracks"]) != bool(sidecar["matrix_case_names"]):
        raise ValueError("matrix track and matrix_case_names must be present together")
    validate_frozen_release_selection(sidecar)
    validate_frozen_release_identity(sidecar)
    return (
        sidecar["run_id"],
        dataset_root,
        profile_name,
        tuple(sidecar["tracks"]),
        tuple(sidecar["variants"]),
        tuple(sidecar["matrix_case_names"]),
        development_tiny,
    )


def expected_record_provenance(
    sidecar: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    return _expected_record_provenance(*_schedule_arguments(sidecar), None)


def _derive_optional_pairs(
    sidecar: dict[str, Any],
    valid_records: dict[tuple[str, str], dict[str, Any]],
) -> tuple[frozenset[str], list[str]]:
    matrix, _, _ = frozen_matrix()
    profile = matrix["profiles"][sidecar["profile"]]
    repeats = profile["paired_repeats"]
    rounds = profile["repeated_update_rounds"]
    optional: set[str] = set()
    issues: list[str] = []

    def policy_triggered(pair_id: str, format_name: str) -> bool:
        record = valid_records.get((pair_id, format_name))
        if record is None:
            return False
        try:
            triggered, _ = protocol.policy_triggers(record, sidecar["policy"])
        except (KeyError, TypeError, ValueError) as error:
            issues.append(
                f"{pair_id}/{format_name}: cannot derive frozen maintenance boundary: "
                f"{error}"
            )
            return False
        return triggered

    for track in sidecar["tracks"]:
        if track not in {
            "sustained",
            "adversarial_natural",
            "adversarial_aligned",
        }:
            continue
        for variant in sidecar["variants"]:
            for repeat in range(repeats):
                prefix = f"{sidecar['run_id']}/{track}/{variant}/repeat-{repeat:03d}"
                for update_round in range(rounds):
                    round_prefix = f"{prefix}/round-{update_round:03d}"
                    if track == "sustained":
                        scan_pair = f"{prefix}/step-{update_round:03d}/cold-scan"
                        if policy_triggered(scan_pair, "v22_no_stable"):
                            optional.add(f"{round_prefix}/policy-maintenance")
                        continue
                    if track == "adversarial_natural":
                        update_pair = f"{round_prefix}/update-attempt"
                        candidate = valid_records.get((update_pair, "v23_logical"))
                        if candidate is not None and (
                            candidate["placement_maintenance_required"] is True
                        ):
                            maintenance_pair = f"{round_prefix}/pmr-maintenance"
                            optional.add(maintenance_pair)
                            maintained = valid_records.get(
                                (maintenance_pair, "v23_logical")
                            )
                            if maintained is not None and maintained["status"] == "ok":
                                optional.add(f"{round_prefix}/update-retry")
                        scan_pair = f"{prefix}/step-{update_round:03d}/cold-scan"
                        for format_name in run.FORMATS:
                            if policy_triggered(scan_pair, format_name):
                                optional.add(
                                    f"{round_prefix}/natural-maintenance/{format_name}"
                                )
                        continue
                    candidate_pair = f"{round_prefix}/candidate-preflight"
                    candidate = valid_records.get((candidate_pair, "v23_logical"))
                    if candidate is not None and (
                        candidate["placement_maintenance_required"] is True
                    ):
                        normalize_pair = f"{round_prefix}/normalize"
                        baseline_pairs = tuple(
                            (
                                f"{round_prefix}/forced-baseline-maintenance/"
                                f"{format_name}",
                                format_name,
                            )
                            for format_name in ("v22_no_stable", "v22_stable")
                        )
                        optional.add(normalize_pair)
                        optional.update(pair_id for pair_id, _ in baseline_pairs)
                        predecessors = [
                            valid_records.get((normalize_pair, "v23_logical")),
                            *(
                                valid_records.get((pair_id, format_name))
                                for pair_id, format_name in baseline_pairs
                            ),
                        ]
                        if all(
                            record is not None
                            and record["status"] == "ok"
                            and record["placement_maintenance_required"] is not True
                            for record in predecessors
                        ):
                            optional.add(f"{round_prefix}/candidate-retry")
    return frozenset(optional), issues


def exact_record_provenance(
    sidecar: dict[str, Any], records: Sequence[dict[str, Any]]
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[str]]:
    allowed = expected_record_provenance(sidecar)
    valid: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (record["pair_id"], record["format"])
        record_expected = allowed.get(key)
        if record_expected is None or any(
            record.get(field) != record_expected[field]
            for field in CORE_PROVENANCE_FIELDS
        ):
            continue
        valid[key] = record
    optional_pairs, issues = _derive_optional_pairs(sidecar, valid)
    return (
        _expected_record_provenance(*_schedule_arguments(sidecar), optional_pairs),
        issues,
    )


def audit_record_provenance(
    sidecar: dict[str, Any], records: Sequence[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    try:
        allowed = expected_record_provenance(sidecar)
    except (KeyError, TypeError, ValueError) as error:
        return [f"cannot derive frozen record provenance: {error}"], []
    issues: list[str] = []
    failures: list[str] = []
    valid: dict[tuple[str, str], dict[str, Any]] = {}
    actual_keys: set[tuple[str, str]] = set()
    for record in records:
        key = (record["pair_id"], record["format"])
        if key in actual_keys:
            issues.append(f"{record['pair_id']}/{record['format']}: duplicate record")
            continue
        actual_keys.add(key)
        record_expected = allowed.get(key)
        if record_expected is None:
            issues.append(
                f"{record['pair_id']}/{record['format']}: record is not a frozen invocation"
            )
            continue
        mismatches = {
            field: (record_expected[field], record.get(field))
            for field in CORE_PROVENANCE_FIELDS
            if record.get(field) != record_expected[field]
        }
        if mismatches:
            failures.append(
                f"{record['pair_id']}/{record['format']}: frozen provenance mismatch: "
                f"{mismatches}"
            )
            continue
        valid[key] = record
    optional_pairs, decision_issues = _derive_optional_pairs(sidecar, valid)
    issues.extend(decision_issues)
    expected = _expected_record_provenance(
        *_schedule_arguments(sidecar), optional_pairs
    )
    expected_keys = set(expected)
    for pair_id, format_name in sorted(expected_keys - actual_keys):
        issues.append(f"{pair_id}/{format_name}: frozen invocation is missing")
    for pair_id, format_name in sorted(actual_keys - expected_keys):
        failures.append(f"{pair_id}/{format_name}: unexpected dynamic invocation")
    for key in sorted(actual_keys & expected_keys):
        record = valid.get(key)
        if record is None:
            continue
        record_expected = expected[key]
        mismatches = {
            field: (record_expected[field], record.get(field))
            for field in PROVENANCE_FIELDS
            if record.get(field) != record_expected[field]
        }
        if mismatches:
            failures.append(
                f"{record['pair_id']}/{record['format']}: frozen provenance mismatch: "
                f"{mismatches}"
            )
    return issues, failures


def expected_complete_pair_ids(sidecar: dict[str, Any]) -> set[str]:
    run_id = sidecar["run_id"]
    matrix = sidecar["matrix"]
    profile = matrix["profiles"][sidecar["profile"]]
    repeats = profile["paired_repeats"]
    expected: set[str] = set()
    fixture_keys: set[protocol.FixtureKey] = set()
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
                    if step.operation != "recluster":
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
            fixture_keys.add(protocol.fixture_key_for_case(case))

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
            fixture_keys.add(
                (schema_kind, ((profile["rows"], rows_per_fragment),), index_kind)
            )
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
        (schema_kind, segments, "none")
        for schema_kind, segments, index_kind in fixture_keys
        if index_kind != "none"
    )
    for schema_kind, segments, index_kind in expanded_fixture_keys:
        fixture_prefix = (
            f"{run_id}/fixtures/{schema_kind}/{protocol.fixture_layout_path(segments)}/"
            f"index-{index_kind}"
        )
        if index_kind == "none":
            for segment_index in range(len(segments)):
                label = (
                    "create" if segment_index == 0 else f"append-{segment_index:03d}"
                )
                expected.add(f"{fixture_prefix}/{label}")
        else:
            expected.add(f"{fixture_prefix}/fixture_clone")
            expected.add(f"{fixture_prefix}/index_build")
    return expected


def audit_row_address_record_contract(
    sidecar: dict[str, Any], record: dict[str, Any]
) -> list[str]:
    failures: list[str] = []
    context = f"{record['pair_id']}/{record['format']}"
    delta = record["placement_delta_bytes"]
    claimed_delta = record["placement_delta_claimed_bytes"]

    if record["format"] == "v23_logical":
        if (delta is None) != (claimed_delta is None):
            failures.append(
                f"{context}: placement Delta measurement and independent claim "
                "must be both present or both null"
            )
        elif delta is not None and delta != claimed_delta:
            failures.append(
                f"{context}: measured placement Delta {delta} does not match "
                f"independent claim {claimed_delta}"
            )
    elif delta is not None or claimed_delta is not None:
        failures.append(f"{context}: legacy format claimed v2.3 placement Delta")

    pmr = record["placement_maintenance_required"] is True
    reason = record["pmr_reason"]
    diagnostic_values = {
        field: record[field] for field in PMR_DIAGNOSTIC_FIELDS if field != "pmr_reason"
    }
    if not pmr:
        populated = [
            field for field in PMR_DIAGNOSTIC_FIELDS if record[field] is not None
        ]
        if populated:
            failures.append(
                f"{context}: non-PMR record populated PMR diagnostics {populated}"
            )
    elif record["format"] != "v23_logical":
        failures.append(f"{context}: legacy format returned v2.3 PMR")
    elif reason == "projected_delta_bytes":
        required = ("pmr_projected_delta_bytes", "pmr_delta_limit_bytes")
        unexpected = [
            field
            for field, value in diagnostic_values.items()
            if field not in required and value is not None
        ]
        missing = [field for field in required if record[field] is None]
        if missing or unexpected:
            failures.append(
                f"{context}: projected_delta_bytes PMR has missing={missing}, "
                f"unexpected={unexpected}"
            )
        else:
            projected = record["pmr_projected_delta_bytes"]
            limit = record["pmr_delta_limit_bytes"]
            if limit != B_FAST or projected <= limit:
                failures.append(
                    f"{context}: projected_delta_bytes PMR requires projected > "
                    f"limit == B_fast ({B_FAST}), observed {projected}/{limit}"
                )
    elif reason == "projected_epoch_bytes":
        required = ("pmr_projected_epoch_bytes", "pmr_epoch_limit_bytes")
        unexpected = [
            field
            for field, value in diagnostic_values.items()
            if field not in required and value is not None
        ]
        missing = [field for field in required if record[field] is None]
        if missing or unexpected:
            failures.append(
                f"{context}: projected_epoch_bytes PMR has missing={missing}, "
                f"unexpected={unexpected}"
            )
        else:
            projected = record["pmr_projected_epoch_bytes"]
            limit = record["pmr_epoch_limit_bytes"]
            if limit != W_FAST or projected <= limit:
                failures.append(
                    f"{context}: projected_epoch_bytes PMR requires projected > "
                    f"limit == W_fast ({W_FAST}), observed {projected}/{limit}"
                )
    elif reason == "index_generation_blocked":
        missing = [field for field, value in diagnostic_values.items() if value is None]
        if missing:
            failures.append(
                f"{context}: index_generation_blocked PMR is missing {missing}"
            )
        else:
            projected_delta = record["pmr_projected_delta_bytes"]
            delta_limit = record["pmr_delta_limit_bytes"]
            projected_epoch = record["pmr_projected_epoch_bytes"]
            epoch_limit = record["pmr_epoch_limit_bytes"]
            generation_delta = record["pmr_generation_delta_bytes"]
            generation_epoch = record["pmr_generation_epoch_bytes"]
            blockers = record["pmr_blocking_indices"]
            if delta_limit != B_FAST or epoch_limit != W_FAST:
                failures.append(
                    f"{context}: index_generation_blocked PMR limits must equal "
                    f"B_fast/W_fast, observed {delta_limit}/{epoch_limit}"
                )
            if projected_delta <= delta_limit and projected_epoch <= epoch_limit:
                failures.append(
                    f"{context}: index_generation_blocked PMR has no exceeded budget"
                )
            if generation_delta == 0 or generation_epoch == 0:
                failures.append(
                    f"{context}: index_generation_blocked PMR generation counters "
                    "must be positive"
                )
            if generation_delta > projected_delta or generation_epoch > projected_epoch:
                failures.append(
                    f"{context}: index_generation_blocked PMR generation bytes "
                    "exceed their enclosing totals"
                )
            if not blockers:
                failures.append(
                    f"{context}: index_generation_blocked PMR has no blocking indices"
                )
            else:
                for blocker_index, blocker in enumerate(blockers):
                    blocker_context = f"{context}: blocker[{blocker_index}]"
                    start = blocker["blocked_transaction_start"]
                    end = blocker["blocked_transaction_end"]
                    if (
                        not blocker["field_ids"]
                        or blocker["oldest_generation"] == 0
                        or blocker["region_bytes"] == 0
                    ):
                        failures.append(
                            f"{blocker_context} must identify positive generation-region debt"
                        )
                    if start != blocker["oldest_generation"] or start > end:
                        failures.append(
                            f"{blocker_context} has invalid blocked transaction range "
                            f"{start}..{end} for oldest generation "
                            f"{blocker['oldest_generation']}"
                        )
    elif reason in STRUCTURAL_PMR_REASONS:
        populated = [
            field for field, value in diagnostic_values.items() if value is not None
        ]
        if populated:
            failures.append(
                f"{context}: structural PMR {reason} populated diagnostics {populated}"
            )
    else:
        failures.append(f"{context}: unknown or missing PMR reason {reason!r}")

    if (
        record["format"] == "v23_logical"
        and record["status"] == "ok"
        and not pmr
        and standard_scope_is_gated(
            sidecar,
            track_of(record["pair_id"], sidecar["run_id"]),
            normalized_pair_template(record["pair_id"]),
            record["operation"],
        )
    ):
        if delta is not None and delta > B_FAST:
            failures.append(
                f"{context}: default-fast placement Delta {delta} exceeds B_fast "
                f"{B_FAST}"
            )
        epoch = record["w_epoch_bytes"]
        if epoch is not None and epoch > W_FAST:
            failures.append(
                f"{context}: default-fast W_epoch {epoch} exceeds W_fast {W_FAST}"
            )
    return failures


def audit_grid_and_correctness(
    sidecar: dict[str, Any], records: Sequence[dict[str, Any]]
) -> tuple[list[str], list[str], dict[str, dict[str, dict[str, Any]]]]:
    issues, failures = audit_record_provenance(sidecar, records)
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
        if any(
            record["operation"] == DEFAULT_COMPACTION_PREFLIGHT
            for record in pair_records
        ):
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
                            if (
                                planned is None
                                or admitted is None
                                or not_admitted is None
                            ):
                                issues.append(
                                    f"{compaction_pair_id}/{record['format']}: "
                                    "missing compaction admission counts"
                                )
                            elif (
                                planned <= 0 or admitted != planned or not_admitted != 0
                            ):
                                failures.append(
                                    f"{compaction_pair_id}/{record['format']}: "
                                    f"fast-path admission is {admitted}/{planned} "
                                    f"with {not_admitted} rejected groups"
                                )
                    if step.preflight_expected_admission is None:
                        continue
                    preflight_label = (
                        "default-reclaim-preflight"
                        if step.operation == "random_delete_reclaim"
                        else "default-compaction-preflight"
                    )
                    pair_id = (
                        f"{sidecar['run_id']}/matrix/{case_name}/repeat-{repeat:03d}/"
                        f"step-{step_index:03d}/{preflight_label}"
                    )
                    preflight = grouped.get(pair_id, [])
                    if [record["format"] for record in preflight] != ["v23_logical"]:
                        issues.append(
                            f"{pair_id}: expected one v23_logical compaction preflight"
                        )
                        continue
                    record = preflight[0]
                    if record["operation"] != DEFAULT_COMPACTION_PREFLIGHT:
                        issues.append(
                            f"{pair_id}: reclaim preflight used {record['operation']}, "
                            f"expected {DEFAULT_COMPACTION_PREFLIGHT}"
                        )
                        continue
                    if record["implementation_path"] != "default_compaction_plan_only":
                        failures.append(
                            f"{pair_id}: reclaim preflight used implementation path "
                            f"{record['implementation_path']}"
                        )
                    source_step = case.steps[step_index - 1]
                    source_operation = (
                        "fixture_clone"
                        if source_step.operation == "create"
                        else source_step.operation
                    )
                    source_pair_id = (
                        f"{sidecar['run_id']}/matrix/{case_name}/"
                        f"repeat-{repeat:03d}/step-{step_index - 1:03d}/"
                        f"{source_operation}"
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
                            f"{pair_id}: cannot bind plan-only evidence to delete version"
                        )
                    planned = record["compaction_groups_planned"]
                    admitted = record["compaction_groups_admitted"]
                    not_admitted = record["compaction_groups_not_admitted"]
                    expected_admission = step.preflight_expected_admission
                    if (
                        record["placement_maintenance_required"] is True
                        or planned is None
                        or planned <= 0
                        or admitted is None
                        or not_admitted is None
                        or planned != admitted + not_admitted
                        or record["admission"] is not (admitted == planned)
                    ):
                        failures.append(
                            f"{pair_id}: default compaction preflight admission counts "
                            "are inconsistent"
                        )
                    elif record["admission"] is not expected_admission:
                        failures.append(
                            f"{pair_id}: workload requires admission="
                            f"{expected_admission}, observed admission={record['admission']}"
                        )
                    if (
                        source is not None
                        and record["dataset_version"] != source["dataset_version"]
                    ):
                        failures.append(
                            f"{pair_id}: plan-only preflight changed dataset version"
                        )
                    if (
                        record["put_requests"] != 0
                        or record["delete_requests"] != 0
                        or record["write_bytes"] != 0
                        or (record["actual_put_attempts"] or 0) != 0
                        or (record["actual_delete_attempts"] or 0) != 0
                        or any(
                            metrics["put_requests"] != 0
                            or metrics["delete_requests"] != 0
                            or metrics["write_bytes"] != 0
                            for metrics in record["io_by_path"].values()
                        )
                    ):
                        failures.append(f"{pair_id}: plan-only preflight wrote objects")
                    relocation_only_fields = (
                        "maintenance_plan_path",
                        "maintenance_plan_sha256",
                        "compacted_data_bytes",
                        "index_storage_bytes_before",
                        "row_addresses_remapped",
                        "indices_remapped",
                        "index_coverage_reuse",
                        "layout_index_maintenance_ns",
                    )
                    if any(
                        record[field] is not None for field in relocation_only_fields
                    ):
                        failures.append(
                            f"{pair_id}: plan-only preflight reported relocation output"
                        )

                if any(step.operation == "bounded_recluster" for step in case.steps):
                    scan_pair_id = (
                        f"{sidecar['run_id']}/matrix/{case_name}/repeat-{repeat:03d}/"
                        f"step-{len(case.steps):03d}/cold-scan"
                    )
                    scan_records = grouped.get(scan_pair_id, [])
                    if Counter(record["format"] for record in scan_records) != Counter(
                        run.FORMATS
                    ):
                        issues.append(
                            f"{scan_pair_id}: bounded clustering postcondition scan is incomplete"
                        )
                    else:
                        physical_order_digests = {
                            record["physical_order_digest"] for record in scan_records
                        }
                        if None in physical_order_digests:
                            issues.append(
                                f"{scan_pair_id}: physical order digest is missing"
                            )
                        elif len(physical_order_digests) != 1:
                            failures.append(
                                f"{scan_pair_id}: physical row order differs across formats"
                            )

    provenance_fields = (
        "run_id",
        "commit",
        "host",
        "seed",
        "storage",
        "policy_sha256",
    )
    for record in records:
        failures.extend(audit_row_address_record_contract(sidecar, record))
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
            and index_operation_requires_write(record)
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
                "placement_delta_claimed_bytes",
                "w_epoch_bytes",
            )
        ):
            failures.append(
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
            record["operation"] in COMMIT_OPERATIONS and record["admission"] is not True
        ):
            failures.append(
                f"{record['pair_id']}/{record['format']}: commit operation was not admitted"
            )
        if (
            record["operation"] in RELOCATION_OPERATIONS
            and record["placement_maintenance_required"] is not True
            and record["status"] == "ok"
            and record["maintenance_plan_sha256"] is None
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
                        if (
                            len(aligned_records) == 3
                            and len(
                                {
                                    record["maintenance_plan_sha256"]
                                    for record in aligned_records
                                }
                            )
                            != 1
                        ):
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


def index_operation_requires_write(record: dict[str, Any]) -> bool:
    if record["operation"] == "index_build":
        return True
    if record["operation"] != "index_optimize":
        return False
    pair_id = record["pair_id"]
    repeated_vector_value_update = (
        record["schema_kind"] == "vector_f32_128"
        and record["index_kind"] == "vector_ivf_flat"
        and pair_id.endswith("/index-catch-up")
        and any(f"/{track}/" in pair_id for track in REPEATED_UPDATE_TRACKS)
    )
    return not repeated_vector_value_update


def is_explicit_matrix_diagnostic(
    sidecar: dict[str, Any], template: str, operation: str
) -> bool:
    if operation in EXPLICIT_MATRIX_DIAGNOSTIC_OPERATIONS:
        return True
    if operation not in {"open", "scan", "take", "index_take"}:
        return False
    matrix = sidecar["matrix"]
    profile = matrix["profiles"][sidecar["profile"]]
    selected = set(sidecar["matrix_case_names"])
    for case in protocol.iter_matrix_cases(
        profile, set(matrix["tracks"]["matrix"]["cases"])
    ):
        if case.name not in selected:
            continue
        if case.steps[-1].operation not in EXPLICIT_MATRIX_DIAGNOSTIC_OPERATIONS:
            continue
        post_prefix = (
            f"{sidecar['run_id']}/matrix/{case.name}/repeat-*/"
            f"step-{len(case.steps):03d}/"
        )
        if template.startswith(post_prefix):
            return True
    return False


def standard_scope_is_gated(
    sidecar: dict[str, Any], track: str, template: str, operation: str
) -> bool:
    if operation == DEFAULT_COMPACTION_PREFLIGHT:
        return False
    if track == "matrix":
        return not is_explicit_matrix_diagnostic(sidecar, template, operation)
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
        if standard_scope_is_gated(sidecar, track, template, operation):
            by_template[template].append(pair)
    for template, samples in sorted(by_template.items()):
        track = track_of(template, sidecar["run_id"])
        if len(samples) != repeats:
            issues.append(
                f"{template}: expected {repeats} paired repeats, found {len(samples)}"
            )
            continue
        samples.sort(key=lambda pair: pair["v23_logical"]["round"])
        operation = samples[0]["v23_logical"]["operation"]
        metrics = list(STANDARD_METRICS)
        if operation in {"open", "take", "index_take"}:
            metrics.extend(PLACEMENT_METADATA_REQUEST_METRICS)
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
                *PLACEMENT_METADATA_REQUEST_METRICS,
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
                    strict=(
                        operation in RELOCATION_OPERATIONS
                        and metric in {"latency", "throughput"}
                    ),
                    samples=bootstrap_samples,
                    ratio_statistic="p95" if metric == "latency" else "median",
                )
            )
    return gates


def add_indexed_repack_lookup_gates(
    sidecar: dict[str, Any],
    complete: dict[str, dict[str, dict[str, Any]]],
    *,
    bootstrap_samples: int,
    issues: list[str],
) -> list[Gate]:
    if "matrix" not in sidecar["tracks"]:
        return []
    profile = sidecar["matrix"]["profiles"][sidecar["profile"]]
    repeats = profile["paired_repeats"]
    selected = set(sidecar["matrix_case_names"])
    cases = [
        case
        for case in protocol.iter_matrix_cases(
            profile, set(sidecar["matrix"]["tracks"]["matrix"]["cases"])
        )
        if case.name in selected
        and case.name.startswith("indexed-repack-random-delete-")
    ]
    gates: list[Gate] = []
    request_metrics = {
        "get_requests",
        "head_requests",
        "list_requests",
        *PLACEMENT_METADATA_REQUEST_METRICS,
    }
    metrics = [
        "latency",
        "data_read_bytes",
        "index_read_bytes",
        "metadata_read_bytes",
        "total_read_bytes",
        "get_requests",
        "head_requests",
        "list_requests",
        *PLACEMENT_METADATA_REQUEST_METRICS,
    ]
    if sidecar["storage"] == "s3":
        metrics.extend(
            ("actual_get_attempts", "actual_head_attempts", "actual_list_attempts")
        )
        request_metrics.update(
            {"actual_get_attempts", "actual_head_attempts", "actual_list_attempts"}
        )
    for case in cases:
        scope = (
            f"{sidecar['run_id']}/matrix/{case.name}/repeat-*/"
            f"step-{len(case.steps):03d}/cold-index-take/indexed-repack-lookup"
        )
        samples = []
        for repeat in range(repeats):
            pair_id = (
                f"{sidecar['run_id']}/matrix/{case.name}/repeat-{repeat:03d}/"
                f"step-{len(case.steps):03d}/cold-index-take"
            )
            pair = complete.get(pair_id)
            if pair is not None:
                samples.append(pair)
        if len(samples) != repeats:
            issues.append(
                f"{scope}: expected {repeats} indexed Repack lookup repeats, "
                f"found {len(samples)}"
            )
            continue
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
                issues.append(f"{scope}: metric {metric} is unavailable")
                continue
            threshold = 1.0 if metric in request_metrics else 1.05
            gates.append(
                make_gate(
                    track="matrix",
                    scope=scope,
                    metric="latency_p95" if metric == "latency" else metric,
                    baseline_name="v22_no_stable",
                    candidate=values["v23_logical"],
                    baseline=values["v22_no_stable"],
                    direction="upper",
                    threshold=threshold,
                    strict=False,
                    samples=bootstrap_samples,
                    ratio_statistic="p95" if metric == "latency" else "median",
                )
            )
            gates.append(
                make_gate(
                    track="matrix",
                    scope=scope,
                    metric="latency_p95" if metric == "latency" else metric,
                    baseline_name="v22_stable",
                    candidate=values["v23_logical"],
                    baseline=values["v22_stable"],
                    direction="upper",
                    threshold=1.0,
                    strict=metric == "latency",
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
    observations: dict[str, Any],
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
        prefix_metrics = [
            "latency",
            "data_read_bytes",
            "data_write_bytes",
            "metadata_read_bytes",
            "metadata_write_bytes",
            "row_address_resident_bytes",
            "row_address_epoch_write_bytes",
            "get_requests",
            "head_requests",
            "list_requests",
        ]
        if sidecar["storage"] == "s3":
            prefix_metrics.extend(
                (
                    "actual_get_attempts",
                    "actual_head_attempts",
                    "actual_list_attempts",
                )
            )
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
                for metric in prefix_metrics:
                    for format_name in run.FORMATS:
                        format_records = [
                            r for r in cumulative if r["format"] == format_name
                        ]
                        if metric == "row_address_resident_bytes":
                            value = max(
                                (
                                    r["placement_delta_bytes"] or 0
                                    for r in format_records
                                ),
                                default=0,
                            )
                        elif metric == "row_address_epoch_write_bytes":
                            value = max(
                                (r["w_epoch_bytes"] or 0 for r in format_records),
                                default=0,
                            )
                        else:
                            value = total_metric(format_records, metric)
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
        variant_observations = []
        for boundary_ordinal, by_metric in sorted(samples_by_boundary.items()):
            complete_observation = all(
                len(values[format_name]) == repeats
                for values in by_metric.values()
                for format_name in run.FORMATS
            )
            if complete_observation:
                variant_observations.append(
                    {
                        "boundary_ordinal": boundary_ordinal,
                        "boundary_round": (
                            boundary_rounds_by_repeat[0][boundary_ordinal]
                            if boundary_rounds_by_repeat
                            and boundary_ordinal < len(boundary_rounds_by_repeat[0])
                            else None
                        ),
                        "repeats": [
                            {
                                "repeat": repeat,
                                "formats": {
                                    format_name: {
                                        metric: by_metric[metric][format_name][repeat]
                                        for metric in prefix_metrics
                                    }
                                    for format_name in run.FORMATS
                                },
                            }
                            for repeat in range(repeats)
                        ],
                    }
                )
            for metric, values in by_metric.items():
                if any(len(values[name]) != repeats for name in run.FORMATS):
                    issues.append(
                        f"sustained/{variant}/boundary-{boundary_ordinal}: "
                        f"incomplete {metric} prefix samples"
                    )
                    continue
                if metric in {
                    "metadata_read_bytes",
                    "metadata_write_bytes",
                    "row_address_resident_bytes",
                    "row_address_epoch_write_bytes",
                }:
                    continue
                threshold = (
                    1.0
                    if metric.endswith("requests") or metric.startswith("actual_")
                    else 1.05
                )
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
        observations.setdefault("variants", {})[variant] = variant_observations
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
        or case.startswith("fragment-reuse-")
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
            index_covers_compacted_data = True
            for sample in samples:
                candidate = sample["v23_logical"]
                if case_name.startswith("fragment-reuse-"):
                    expected_reuse_presence = {
                        "v22_no_stable": True,
                        "v22_stable": False,
                        "v23_logical": False,
                    }
                    for (
                        format_name,
                        expected_present,
                    ) in expected_reuse_presence.items():
                        if (
                            sample[format_name]["fragment_reuse_index_present"]
                            is not expected_present
                        ):
                            failures.append(
                                f"{sample[format_name]['pair_id']}/{format_name}: "
                                "fragment-reuse comparison did not materialize the "
                                f"expected system-index state {expected_present}"
                            )
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
                        index_covers_compacted_data = False
                    elif index_bytes < data_bytes:
                        index_covers_compacted_data = False
            if not index_covers_compacted_data:
                continue
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


def audit_placement_history_independence(
    sidecar: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    issues: list[str],
    failures: list[str],
) -> dict[str, Any]:
    if "matrix" not in sidecar["tracks"]:
        return {"comparisons": []}
    profile = sidecar["matrix"]["profiles"][sidecar["profile"]]
    source_fragments = profile["logical_fragment_counts"][-1]
    selected = set(sidecar["matrix_case_names"])
    grouped = group_by_pair(records)
    comparisons: list[dict[str, Any]] = []

    def unique_candidate(pair_id: str) -> dict[str, Any] | None:
        matches = [
            record
            for record in grouped.get(pair_id, [])
            if record["format"] == "v23_logical"
        ]
        if len(matches) != 1:
            issues.append(f"{pair_id}: expected one v23 history-independence record")
            return None
        return matches[0]

    for schema in profile["schemas"]:
        one_shot_case = f"compact-{source_fragments}-to-1/{schema}"
        for rounds in profile["repeated_compaction_rounds"]:
            repeated_case = f"repeated-compaction-{rounds}/{schema}"
            if one_shot_case not in selected and repeated_case not in selected:
                continue
            if one_shot_case not in selected or repeated_case not in selected:
                if sidecar["profile"] == "release":
                    issues.append(
                        f"matrix/{schema}: one-shot and repeated compaction must be co-located"
                    )
                continue
            for repeat in range(profile["paired_repeats"]):
                one_prefix = (
                    f"{sidecar['run_id']}/matrix/{one_shot_case}/repeat-{repeat:03d}"
                )
                repeated_prefix = (
                    f"{sidecar['run_id']}/matrix/{repeated_case}/repeat-{repeat:03d}"
                )
                one_relocation = unique_candidate(
                    f"{one_prefix}/step-001/default_compaction"
                )
                repeated_relocation = unique_candidate(
                    f"{repeated_prefix}/step-{rounds:03d}/default_compaction"
                )
                one_scan = unique_candidate(f"{one_prefix}/step-002/cold-scan")
                repeated_scan = unique_candidate(
                    f"{repeated_prefix}/step-{rounds + 1:03d}/cold-scan"
                )
                if any(
                    value is None
                    for value in (
                        one_relocation,
                        repeated_relocation,
                        one_scan,
                        repeated_scan,
                    )
                ):
                    continue
                assert one_relocation is not None
                assert repeated_relocation is not None
                assert one_scan is not None
                assert repeated_scan is not None
                projection_fields = (
                    "schema_kind",
                    "expected_rows",
                    "result_rows",
                    "fragments",
                    "physical_rows",
                    "state_digest",
                )
                one_projection = {field: one_scan[field] for field in projection_fields}
                repeated_projection = {
                    field: repeated_scan[field] for field in projection_fields
                }
                if any(value is None for value in one_projection.values()) or any(
                    value is None for value in repeated_projection.values()
                ):
                    issues.append(
                        f"{repeated_prefix}: canonical topology projection is incomplete"
                    )
                    continue
                canonical = json.dumps(
                    one_projection,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
                repeated_canonical = json.dumps(
                    repeated_projection,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                repeated_fingerprint = hashlib.sha256(
                    repeated_canonical.encode()
                ).hexdigest()
                if fingerprint != repeated_fingerprint:
                    failures.append(
                        f"{repeated_prefix}: ID-normalized final topology differs from one-shot"
                    )
                ratios: dict[str, float | None] = {}
                for field in ("placement_delta_bytes", "placement_root_bytes"):
                    one_value = one_relocation[field]
                    repeated_value = repeated_relocation[field]
                    if one_value is None or repeated_value is None:
                        issues.append(
                            f"{repeated_prefix}: {field} is missing for history comparison"
                        )
                        ratios[field] = None
                        continue
                    if one_value == 0 or repeated_value == 0:
                        ratio = 1.0 if one_value == repeated_value else math.inf
                    else:
                        ratio = max(one_value, repeated_value) / min(
                            one_value, repeated_value
                        )
                    ratios[field] = ratio
                    if ratio > 1.05:
                        failures.append(
                            f"{repeated_prefix}: {field} is history-dependent: "
                            f"one-shot={one_value}, repeated={repeated_value}, ratio={ratio:.6f}"
                        )
                comparisons.append(
                    {
                        "schema": schema,
                        "rounds": rounds,
                        "repeat": repeat,
                        "one_shot_case": one_shot_case,
                        "repeated_case": repeated_case,
                        "id_normalized_semantic_fingerprint": fingerprint,
                        "repeated_semantic_fingerprint": repeated_fingerprint,
                        "placement_byte_ratios": ratios,
                    }
                )
    return {"comparisons": comparisons}


def audit_skewed_packed_run_fixtures(
    sidecar: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    issues: list[str],
    failures: list[str],
) -> dict[str, Any]:
    if "matrix" not in sidecar["tracks"]:
        return {"fixtures": []}
    profile = sidecar["matrix"]["profiles"][sidecar["profile"]]
    cases = {
        case.name: case
        for case in protocol.iter_matrix_cases(
            profile, set(sidecar["matrix"]["tracks"]["matrix"]["cases"])
        )
        if case.fixture_segments
    }
    grouped = group_by_pair(records)
    observations: list[dict[str, Any]] = []
    for case_name in sidecar["matrix_case_names"]:
        case = cases.get(case_name)
        if case is None:
            continue
        segments = protocol.fixture_segments_for_case(case)
        expected_fragments = 0
        cumulative_rows = 0
        layout = protocol.fixture_layout_path(segments)
        for segment_index, (segment_rows, rows_per_fragment) in enumerate(segments):
            expected_fragments += segment_rows // rows_per_fragment
            cumulative_rows += segment_rows
            label = "create" if segment_index == 0 else f"append-{segment_index:03d}"
            pair_id = (
                f"{sidecar['run_id']}/fixtures/{case.schema_kind}/{layout}/"
                f"index-none/{label}"
            )
            pair_records = grouped.get(pair_id, [])
            if Counter(record["format"] for record in pair_records) != Counter(
                run.FORMATS
            ):
                issues.append(f"{pair_id}: segmented fixture phase is incomplete")
                continue
            for record in pair_records:
                if record["fragments"] != expected_fragments:
                    failures.append(
                        f"{pair_id}/{record['format']}: wrote {record['fragments']} "
                        f"fragments, expected {expected_fragments}"
                    )
                if record["result_rows"] != cumulative_rows:
                    failures.append(
                        f"{pair_id}/{record['format']}: fixture has "
                        f"{record['result_rows']} rows, expected {cumulative_rows}"
                    )
        target_fragments = sum(
            segment_rows // rows_per_fragment
            for segment_rows, rows_per_fragment in segments
        )
        if len({rows_per_fragment for _, rows_per_fragment in segments}) < 2:
            failures.append(f"matrix/{case_name}: fixture row counts are uniform")
        observations.append(
            {
                "case": case_name,
                "segments": [list(segment) for segment in segments],
                "source_fragments": target_fragments,
                "source_rows": sum(segment_rows for segment_rows, _ in segments),
            }
        )
    return {"fixtures": observations}


def build_explicit_maintenance_observations(
    sidecar: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    issues: list[str],
    failures: list[str],
) -> dict[str, Any]:
    if "matrix" not in sidecar["tracks"]:
        return {"cases": {}}
    matrix = sidecar["matrix"]
    profile = matrix["profiles"][sidecar["profile"]]
    repeats = profile["paired_repeats"]
    selected = set(sidecar["matrix_case_names"])
    cases = [
        case
        for case in protocol.iter_matrix_cases(
            profile, set(matrix["tracks"]["matrix"]["cases"])
        )
        if case.name in selected
        and case.steps[-1].operation in EXPLICIT_MATRIX_DIAGNOSTIC_OPERATIONS
    ]
    grouped = group_by_pair(records)
    observations: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        step_index = len(case.steps) - 1
        operation = case.steps[-1].operation
        case_values: list[dict[str, Any]] = []
        for repeat in range(repeats):
            prefix = f"{sidecar['run_id']}/matrix/{case.name}/repeat-{repeat:03d}"
            pair_id = f"{prefix}/step-{step_index:03d}/{operation}"
            maintenance_records = grouped.get(pair_id, [])
            expected_formats = (
                ("v23_logical",) if operation == "recluster" else run.FORMATS
            )
            if Counter(record["format"] for record in maintenance_records) != Counter(
                expected_formats
            ):
                issues.append(
                    f"{pair_id}: explicit maintenance expected formats "
                    f"{list(expected_formats)}, found "
                    f"{sorted(record['format'] for record in maintenance_records)}"
                )
                continue
            by_format = {record["format"]: record for record in maintenance_records}
            candidate = by_format["v23_logical"]
            required = {
                field: candidate[field]
                for field in (
                    "rows_updated",
                    "compacted_data_bytes",
                    "row_addresses_remapped",
                    "indices_remapped",
                    "layout_index_maintenance_ns",
                    "explicit_locator_objects_written",
                    "explicit_locator_bytes_written",
                )
            }
            missing = sorted(
                field for field, value in required.items() if value is None
            )
            if missing:
                issues.append(f"{pair_id}: explicit cost fields are missing: {missing}")
                continue
            if required["rows_updated"] != candidate["expected_rows"]:
                failures.append(
                    f"{pair_id}: explicit maintenance rewrote "
                    f"{required['rows_updated']} rows, expected {candidate['expected_rows']}"
                )
            if required["compacted_data_bytes"] <= 0:
                failures.append(
                    f"{pair_id}: explicit maintenance has no source data bytes"
                )
            if required["explicit_locator_objects_written"] <= 0:
                failures.append(
                    f"{pair_id}: explicit maintenance wrote no locator objects"
                )
            if required["explicit_locator_bytes_written"] <= 0:
                failures.append(
                    f"{pair_id}: explicit maintenance wrote no locator bytes"
                )
            if required["row_addresses_remapped"] != 0:
                failures.append(
                    f"{pair_id}: explicit maintenance remapped logical addresses"
                )
            if required["indices_remapped"] != 0:
                failures.append(
                    f"{pair_id}: explicit maintenance remapped index objects"
                )
            if candidate["index_kind"] != "none":
                if candidate["index_coverage_reuse"] != 1.0:
                    failures.append(
                        f"{pair_id}: explicit maintenance did not reuse full index coverage"
                    )
                index_io = candidate["io_by_path"]["index"]
                if any(value != 0 for value in index_io.values()):
                    failures.append(
                        f"{pair_id}: explicit maintenance accessed index objects"
                    )

            maintenance_by_format = {
                format_name: {
                    "duration_ns": record["duration_ns"],
                    "data_read_bytes": record["io_by_path"]["data"]["read_bytes"],
                    "data_write_bytes": record["io_by_path"]["data"]["write_bytes"],
                    "index_read_bytes": record["io_by_path"]["index"]["read_bytes"],
                    "index_write_bytes": record["io_by_path"]["index"]["write_bytes"],
                    "metadata_read_bytes": record["io_by_path"]["metadata"][
                        "read_bytes"
                    ],
                    "metadata_write_bytes": record["io_by_path"]["metadata"][
                        "write_bytes"
                    ],
                    "get_requests": record["get_requests"],
                    "head_requests": record["head_requests"],
                    "list_requests": record["list_requests"],
                }
                for format_name, record in by_format.items()
            }
            post_step = len(case.steps)
            post_cold_lookup: dict[str, dict[str, dict[str, int | None]]] = {}
            for label in ("cold-take", "cold-index-take"):
                if label == "cold-index-take" and case.steps[-1].index_kind == "none":
                    continue
                probe_pair_id = f"{prefix}/step-{post_step:03d}/{label}"
                probe_records = grouped.get(probe_pair_id, [])
                if Counter(record["format"] for record in probe_records) != Counter(
                    run.FORMATS
                ):
                    issues.append(
                        f"{probe_pair_id}: explicit post-maintenance lookup is incomplete"
                    )
                    continue
                post_cold_lookup[label] = {
                    record["format"]: {
                        "duration_ns": record["duration_ns"],
                        "data_read_bytes": record["io_by_path"]["data"]["read_bytes"],
                        "index_read_bytes": record["io_by_path"]["index"]["read_bytes"],
                        "metadata_read_bytes": record["io_by_path"]["metadata"][
                            "read_bytes"
                        ],
                        "get_requests": record["get_requests"],
                        "head_requests": record["head_requests"],
                        "list_requests": record["list_requests"],
                        "metadata_get_requests": record["io_by_path"]["metadata"][
                            "get_requests"
                        ],
                        "metadata_head_requests": record["io_by_path"]["metadata"][
                            "head_requests"
                        ],
                        "metadata_list_requests": record["io_by_path"]["metadata"][
                            "list_requests"
                        ],
                    }
                    for record in probe_records
                }
            rows_rewritten = required["rows_updated"]
            compacted_data_bytes = required["compacted_data_bytes"]
            data_write_bytes = candidate["io_by_path"]["data"]["write_bytes"]
            case_values.append(
                {
                    "repeat": repeat,
                    "operation": operation,
                    "mapping_bytes_per_row": (
                        required["explicit_locator_bytes_written"] / rows_rewritten
                        if rows_rewritten > 0
                        else None
                    ),
                    "data_write_amplification": (
                        data_write_bytes / compacted_data_bytes
                        if compacted_data_bytes > 0
                        else None
                    ),
                    "locator_objects_written": required[
                        "explicit_locator_objects_written"
                    ],
                    "locator_bytes_written": required["explicit_locator_bytes_written"],
                    "maintenance": maintenance_by_format,
                    "post_cold_lookup": post_cold_lookup,
                }
            )
        observations[case.name] = case_values
    return {"cases": observations}


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
        explicit_maintenance = {"cases": {}}
        sustained_prefixes = {"variants": {}}
        placement_history_independence = {"comparisons": []}
        skewed_packed_run_fixtures = {"fixtures": []}
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
        explicit_maintenance = build_explicit_maintenance_observations(
            sidecar,
            records,
            issues=issues,
            failures=failures,
        )
        placement_history_independence = audit_placement_history_independence(
            sidecar,
            records,
            issues=issues,
            failures=failures,
        )
        skewed_packed_run_fixtures = audit_skewed_packed_run_fixtures(
            sidecar,
            records,
            issues=issues,
            failures=failures,
        )
        sustained_prefixes = {"variants": {}}
        gates = add_standard_pair_gates(
            sidecar,
            complete,
            bootstrap_samples=bootstrap_samples,
            issues=issues,
        )
        gates.extend(
            add_indexed_repack_lookup_gates(
                sidecar,
                complete,
                bootstrap_samples=bootstrap_samples,
                issues=issues,
            )
        )
        gates.extend(
            add_sustained_prefix_gates(
                sidecar,
                records,
                bootstrap_samples=bootstrap_samples,
                issues=issues,
                failures=failures,
                observations=sustained_prefixes,
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
    failed_release_gates = [
        gate for gate in failed_gates if gate.aggregate_release_gate
    ]
    failed_diagnostic_gates = [
        gate for gate in failed_gates if not gate.aggregate_release_gate
    ]
    if enforce_gates is None:
        enforce_gates = bool(sidecar) and sidecar.get("profile") == "release"
    if issues:
        verdict = "INCOMPLETE"
    elif failures or (enforce_gates and failed_release_gates):
        verdict = "FAIL"
    else:
        verdict = "PASS"
    machine = {
        "schema_version": 1,
        "suite": "stable_row_address_design_protocol_report",
        "run_id": sidecar.get("run_id") if sidecar else None,
        "commit": sidecar.get("commit") if sidecar else None,
        "verdict": verdict,
        "bootstrap_samples": bootstrap_samples,
        "performance_gates_enforced": enforce_gates,
        "diagnostic_only_gate_tracks": sorted(DIAGNOSTIC_ONLY_GATE_TRACKS),
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
        "explicit_maintenance": explicit_maintenance,
        "placement_history_independence": placement_history_independence,
        "skewed_packed_run_fixtures": skewed_packed_run_fixtures,
        "sustained_prefixes": sustained_prefixes,
    }
    lines = [
        "# Stable Logical Row Address Protocol Report",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"Records: {len(records)}; complete paired phases: {len(complete)}; "
        f"bootstrap resamples: {bootstrap_samples}",
        "",
        "| Track | Scope | Metric | Baseline | Samples | Ratio | 95% CI | Contract | Role | Result |",
        "|---|---|---|---|---:|---:|---:|---|---|---|",
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
            f"{'release gate' if gate.aggregate_release_gate else 'diagnostic-only'} | "
            f"{'PASS' if gate.passed else 'FAIL'} |"
        )
    if not gates:
        lines.append("| — | — | — | — | — | — | — | — | — | INCOMPLETE |")
    sustained_variants = sustained_prefixes.get("variants", {})
    if any(sustained_variants.values()):
        lines.extend(
            (
                "",
                "## Sustained maintenance-boundary prefix totals",
                "",
                "| Variant | Boundary | Round | Repeat | Format | Metadata R/W bytes | Actual GET/HEAD/LIST | Row-address resident/epoch-write bytes |",
                "|---|---:|---:|---:|---|---:|---:|---:|",
            )
        )
        for variant, boundaries in sustained_variants.items():
            for boundary in boundaries:
                for repeat_value in boundary["repeats"]:
                    for format_name, totals in repeat_value["formats"].items():
                        actual_attempts = (
                            f"{totals['actual_get_attempts']}/"
                            f"{totals['actual_head_attempts']}/"
                            f"{totals['actual_list_attempts']}"
                            if sidecar["storage"] == "s3"
                            else "—"
                        )
                        lines.append(
                            f"| {variant} | {boundary['boundary_ordinal']} | "
                            f"{boundary['boundary_round']} | {repeat_value['repeat']} | "
                            f"{format_name} | {totals['metadata_read_bytes']}/"
                            f"{totals['metadata_write_bytes']} | {actual_attempts} | "
                            f"{totals['row_address_resident_bytes']}/"
                            f"{totals['row_address_epoch_write_bytes']} |"
                        )
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
    explicit_cases = explicit_maintenance.get("cases", {})
    if explicit_cases:
        lines.extend(
            (
                "",
                "## Explicit maintenance public cost",
                "",
                "These observations are diagnostics and are not default-operation gates.",
                "",
                "| Case | Repeat | Mapping bytes/row | Data write amplification | Locator objects | Locator bytes | Candidate latency (ns) |",
                "|---|---:|---:|---:|---:|---:|---:|",
            )
        )
        for case_name, case_values in explicit_cases.items():
            for value in case_values:
                candidate = value["maintenance"]["v23_logical"]
                mapping = value["mapping_bytes_per_row"]
                amplification = value["data_write_amplification"]
                lines.append(
                    f"| {case_name} | {value['repeat']} | "
                    f"{mapping if mapping is not None else '—'} | "
                    f"{amplification if amplification is not None else '—'} | "
                    f"{value['locator_objects_written']} | "
                    f"{value['locator_bytes_written']} | {candidate['duration_ns']} |"
                )
        lines.extend(
            (
                "",
                "### Paired maintenance and lookup cost",
                "",
                "| Case | Repeat | Format | Maintenance ns | Data R/W bytes | Index R/W bytes | Metadata R/W bytes | GET/HEAD/LIST | Post cold take ns | Post cold index take ns |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            )
        )
        for case_name, case_values in explicit_cases.items():
            for value in case_values:
                cold_take = value["post_cold_lookup"].get("cold-take", {})
                cold_index_take = value["post_cold_lookup"].get("cold-index-take", {})
                for format_name, cost in value["maintenance"].items():
                    lines.append(
                        f"| {case_name} | {value['repeat']} | {format_name} | "
                        f"{cost['duration_ns']} | "
                        f"{cost['data_read_bytes']}/{cost['data_write_bytes']} | "
                        f"{cost['index_read_bytes']}/{cost['index_write_bytes']} | "
                        f"{cost['metadata_read_bytes']}/{cost['metadata_write_bytes']} | "
                        f"{cost['get_requests']}/{cost['head_requests']}/{cost['list_requests']} | "
                        f"{cold_take.get(format_name, {}).get('duration_ns', '—')} | "
                        f"{cold_index_take.get(format_name, {}).get('duration_ns', '—')} |"
                    )
    if issues:
        lines.extend(("", "## Incomplete evidence", ""))
        lines.extend(f"- {issue}" for issue in issues)
    if failures:
        lines.extend(("", "## Correctness and protocol failures", ""))
        lines.extend(f"- {failure}" for failure in failures)
    if failed_release_gates:
        lines.extend(("", "## Failed performance gates", ""))
        lines.extend(
            f"- {gate.track}/{gate.scope}/{gate.metric} vs {gate.baseline}: "
            f"{gate.detail or gate.contract}"
            for gate in failed_release_gates
        )
        if not enforce_gates:
            lines.extend(
                (
                    "",
                    "Smoke performance failures are diagnostic; release reports always enforce them.",
                )
            )
    if failed_diagnostic_gates:
        lines.extend(("", "## Failed diagnostic-only comparisons", ""))
        lines.append(
            "These comparisons are reported but do not affect the aggregate release verdict."
        )
        lines.extend(
            f"- {gate.track}/{gate.scope}/{gate.metric} vs {gate.baseline}: "
            f"{gate.detail or gate.contract}"
            for gate in failed_diagnostic_gates
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
