#!/usr/bin/env python3
"""Execute the design-owned stable logical row-address benchmark protocol."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run  # noqa: E402


DEFAULT_MATRIX = SCRIPT_DIR / "workload_matrix.v1.json"
PROFILE_FIELDS = {
    "paired_repeats",
    "rows",
    "schemas",
    "logical_fragment_counts",
    "take_counts",
    "delete_percentages",
    "update_percentages",
    "merge_match_percentages",
    "backfill_column_counts",
    "repeated_compaction_rounds",
    "repeated_update_rounds",
    "hot_set_rows",
    "minimum_sustained_boundaries",
}
TRACK_FIELDS = {
    "matrix": {"cases"},
    "sustained": {
        "selection",
        "selection_step",
        "maintenance_boundary_source",
        "variants",
    },
    "adversarial_natural": {
        "selection",
        "maintenance_boundary_source",
        "variants",
    },
    "adversarial_aligned": {
        "selection",
        "maintenance_boundary_source",
        "variants",
    },
}
MEASUREMENT_FIELDS = {
    "process_isolation",
    "cold_probe_order",
    "state_equivalence",
    "logical_io",
    "actual_attempts",
    "rss",
    "result_format",
}
SCHEMA_RECORD_NAMES = {
    "narrow16": "narrow_16b",
    "wide128": "wide_128b",
    "vector": "vector_f32_128",
}
INDEX_RECORD_NAMES = {
    "none": "none",
    "scalar": "scalar_btree",
    "vector": "vector_ivf_flat",
}
SELECTION_RECORD_NAMES = {
    "range": "range",
    "random": "uniform_without_replacement",
}


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


def _positive_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def load_matrix(path: Path) -> tuple[dict[str, Any], str, str]:
    matrix = json.loads(path.read_text(encoding="utf-8"))
    matrix = _strict_object(
        matrix,
        {"schema_version", "name", "profiles", "tracks", "measurement"},
        "workload matrix",
    )
    if matrix["schema_version"] != 1:
        raise ValueError("workload matrix schema_version must be 1")
    if not isinstance(matrix["name"], str) or not matrix["name"].strip():
        raise ValueError("workload matrix name must be non-empty")

    profiles = _strict_object(matrix["profiles"], {"smoke", "release"}, "profiles")
    for profile_name, profile in profiles.items():
        profile = _strict_object(profile, PROFILE_FIELDS, f"profile {profile_name}")
        for field in (
            "paired_repeats",
            "rows",
            "repeated_update_rounds",
            "hot_set_rows",
            "minimum_sustained_boundaries",
        ):
            _positive_int(profile[field], f"profile {profile_name}.{field}")
        if profile["hot_set_rows"] > profile["rows"]:
            raise ValueError(f"profile {profile_name} hot_set_rows exceeds rows")
        for field in (
            "logical_fragment_counts",
            "take_counts",
            "delete_percentages",
            "update_percentages",
            "merge_match_percentages",
            "backfill_column_counts",
            "repeated_compaction_rounds",
        ):
            values = profile[field]
            if not isinstance(values, list) or not values:
                raise ValueError(f"profile {profile_name}.{field} must be non-empty")
            for index, value in enumerate(values):
                if field == "merge_match_percentages" and value == 0:
                    continue
                _positive_int(value, f"profile {profile_name}.{field}[{index}]")
        if any(value > 100 for value in profile["delete_percentages"]):
            raise ValueError("delete percentages must not exceed 100")
        if any(value > 100 for value in profile["update_percentages"]):
            raise ValueError("update percentages must not exceed 100")
        if any(value > 100 for value in profile["merge_match_percentages"]):
            raise ValueError("merge match percentages must not exceed 100")
        if profile["schemas"] != ["narrow16", "wide128", "vector"]:
            raise ValueError("profiles must freeze narrow16/wide128/vector schemas")
        if any(value > profile["rows"] for value in profile["take_counts"]):
            raise ValueError(f"profile {profile_name} take_count exceeds rows")

    tracks = _strict_object(matrix["tracks"], set(TRACK_FIELDS), "tracks")
    for track_name, fields in TRACK_FIELDS.items():
        track = _strict_object(tracks[track_name], fields, f"track {track_name}")
        list_field = "cases" if track_name == "matrix" else "variants"
        if not isinstance(track[list_field], list) or not track[list_field]:
            raise ValueError(f"track {track_name}.{list_field} must be non-empty")
    if tracks["sustained"]["selection_step"] != 0:
        raise ValueError("sustained selection_step must be frozen at zero")
    for track_name in ("sustained", "adversarial_natural", "adversarial_aligned"):
        if tracks[track_name]["variants"] != ["bare", "scalar", "vector"]:
            raise ValueError(f"track {track_name} must cover bare/scalar/vector")

    _strict_object(matrix["measurement"], MEASUREMENT_FIELDS, "measurement")
    canonical = json.dumps(
        matrix, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return matrix, canonical, digest


@dataclasses.dataclass(frozen=True)
class Step:
    operation: str
    expected_rows: int
    mutation_count: int = 1
    id_start: int = 0
    step: int = 0
    selection_step: int = 0
    match_percent: int = 50
    schema_kind: str = "narrow16"
    index_kind: str = "none"
    update_driver: str = "native"
    selection: str = "range"
    target_rows_per_fragment: int = 1_000_000

    def __post_init__(self) -> None:
        if self.operation not in run.WORKER_OPERATIONS:
            raise ValueError(f"unsupported worker operation: {self.operation}")
        if self.expected_rows < 0 or self.mutation_count <= 0:
            raise ValueError("invalid expected_rows or mutation_count")
        if self.schema_kind not in SCHEMA_RECORD_NAMES:
            raise ValueError(f"unsupported schema_kind: {self.schema_kind}")
        if self.index_kind not in INDEX_RECORD_NAMES:
            raise ValueError(f"unsupported index_kind: {self.index_kind}")
        if self.selection not in SELECTION_RECORD_NAMES:
            raise ValueError(f"unsupported selection: {self.selection}")

    @property
    def implementation_path(self) -> str:
        if self.operation == "update":
            return (
                "native_update_builder"
                if self.update_driver == "native"
                else "exact_selection_matched_merge"
            )
        if self.operation == "fixture_clone":
            return "canonical_fixture_shallow_clone"
        if self.operation in {
            "normalize_placement",
            "repack",
            "recluster",
            "checkpoint_generation",
        }:
            return "capability_gated_explicit_maintenance"
        if self.operation == "default_compaction":
            return "default_compaction"
        if self.operation == "random_delete_reclaim":
            return "same_postcondition_repack_or_default_compaction"
        if self.operation in {"index_build", "index_take", "index_optimize"}:
            return INDEX_RECORD_NAMES[self.index_kind]
        return "native_dataset_api"


@dataclasses.dataclass(frozen=True)
class MatrixCase:
    name: str
    schema_kind: str
    rows_per_fragment: int
    take_count: int
    steps: tuple[Step, ...]
    fixture_index_kind: str = "none"


def _rows_per_fragment(rows: int, fragments: int) -> int:
    return max(1, math.ceil(rows / fragments))


def iter_matrix_cases(
    profile: dict[str, Any], selected: set[str]
) -> Iterator[MatrixCase]:
    rows = profile["rows"]
    default_fragments = profile["logical_fragment_counts"][0]
    default_rows_per_fragment = _rows_per_fragment(rows, default_fragments)
    for schema in profile["schemas"]:
        for take_count in profile["take_counts"]:
            if "append" in selected:
                count = max(1, rows // 100)
                yield MatrixCase(
                    f"append/{schema}/take-{take_count}",
                    schema,
                    default_rows_per_fragment,
                    take_count,
                    (
                        Step("create", rows, schema_kind=schema),
                        Step(
                            "append",
                            rows + count,
                            mutation_count=count,
                            id_start=rows,
                            schema_kind=schema,
                        ),
                    ),
                )
            for delete_kind in ("clustered", "random"):
                if f"delete_{delete_kind}" not in selected:
                    continue
                for percentage in profile["delete_percentages"]:
                    count = rows * percentage // 100
                    reclaim = delete_kind == "random" and percentage in {50, 90}
                    steps = [
                        Step("create", rows, schema_kind=schema),
                        Step(
                            "delete",
                            rows - count,
                            mutation_count=count,
                            schema_kind=schema,
                            selection=(
                                "random" if delete_kind == "random" else "range"
                            ),
                        ),
                    ]
                    if delete_kind == "clustered":
                        steps.append(
                            Step(
                                "default_compaction",
                                rows - count,
                                mutation_count=count,
                                schema_kind=schema,
                                selection="range",
                                target_rows_per_fragment=max(1, rows - count),
                            )
                        )
                    if reclaim:
                        steps.append(
                            Step(
                                "random_delete_reclaim",
                                rows - count,
                                mutation_count=count,
                                schema_kind=schema,
                                selection="random",
                                target_rows_per_fragment=max(1, rows - count),
                            )
                        )
                    yield MatrixCase(
                        f"delete-{delete_kind}-{percentage}/{schema}/take-{take_count}",
                        schema,
                        default_rows_per_fragment,
                        min(take_count, rows - count),
                        tuple(steps),
                    )
            for update_kind in ("clustered", "random"):
                if f"update_{update_kind}" not in selected:
                    continue
                for percentage in profile["update_percentages"]:
                    count = max(1, rows * percentage // 100)
                    yield MatrixCase(
                        f"update-{update_kind}-{percentage}/{schema}/take-{take_count}",
                        schema,
                        default_rows_per_fragment,
                        take_count,
                        (
                            Step("create", rows, schema_kind=schema),
                            Step(
                                "update",
                                rows,
                                mutation_count=count,
                                schema_kind=schema,
                                selection="random"
                                if update_kind == "random"
                                else "range",
                                update_driver=(
                                    "exact-matched-merge"
                                    if update_kind == "random"
                                    else "native"
                                ),
                            ),
                        ),
                    )
            if "merge_insert" in selected:
                count = max(1, rows // 100)
                for match_percent in profile["merge_match_percentages"]:
                    matched = count * match_percent // 100
                    expected = rows + count - matched
                    yield MatrixCase(
                        f"merge-{match_percent}/{schema}/take-{take_count}",
                        schema,
                        default_rows_per_fragment,
                        take_count,
                        (
                            Step("create", rows, schema_kind=schema),
                            Step(
                                "merge_insert",
                                expected,
                                mutation_count=count,
                                id_start=rows,
                                match_percent=match_percent,
                                schema_kind=schema,
                            ),
                        ),
                    )
            if "backfill" in selected:
                for columns in profile["backfill_column_counts"]:
                    yield MatrixCase(
                        f"backfill-{columns}/{schema}/take-{take_count}",
                        schema,
                        default_rows_per_fragment,
                        take_count,
                        (
                            Step("create", rows, schema_kind=schema),
                            Step(
                                "backfill",
                                rows,
                                mutation_count=columns,
                                schema_kind=schema,
                            ),
                        ),
                    )

    if "n_to_one_compaction" in selected:
        for fragments in profile["logical_fragment_counts"]:
            for schema in profile["schemas"]:
                yield MatrixCase(
                    f"compact-{fragments}-to-1/{schema}",
                    schema,
                    _rows_per_fragment(rows, fragments),
                    min(profile["take_counts"][-1], rows),
                    (
                        Step("create", rows, schema_kind=schema),
                        Step(
                            "default_compaction",
                            rows,
                            schema_kind=schema,
                            target_rows_per_fragment=rows,
                        ),
                    ),
                )
    if "repeated_compaction" in selected:
        fragments = profile["logical_fragment_counts"][-1]
        for rounds in profile["repeated_compaction_rounds"]:
            for schema in profile["schemas"]:
                steps = [Step("create", rows, schema_kind=schema)]
                steps.extend(
                    Step(
                        "default_compaction",
                        rows,
                        step=step,
                        schema_kind=schema,
                        target_rows_per_fragment=rows,
                    )
                    for step in range(rounds)
                )
                yield MatrixCase(
                    f"repeated-compaction-{rounds}/{schema}",
                    schema,
                    _rows_per_fragment(rows, fragments),
                    min(profile["take_counts"][-1], rows),
                    tuple(steps),
                )
    if "pack_random_mutation_chain" in selected:
        fragments = profile["logical_fragment_counts"][-1]
        rows_per_fragment = _rows_per_fragment(rows, fragments)
        for schema in profile["schemas"]:
            for percentage in profile["update_percentages"]:
                count = max(1, rows * percentage // 100)
                yield MatrixCase(
                    f"pack-random-update-{percentage}/{schema}",
                    schema,
                    rows_per_fragment,
                    min(profile["take_counts"][-1], rows),
                    (
                        Step("create", rows, schema_kind=schema),
                        Step(
                            "default_compaction",
                            rows,
                            schema_kind=schema,
                            target_rows_per_fragment=rows,
                        ),
                        Step(
                            "update",
                            rows,
                            mutation_count=count,
                            schema_kind=schema,
                            update_driver="exact-matched-merge",
                            selection="random",
                        ),
                    ),
                )
            for percentage in profile["delete_percentages"]:
                count = max(1, rows * percentage // 100)
                expected = rows - count
                steps = [
                    Step("create", rows, schema_kind=schema),
                    Step(
                        "default_compaction",
                        rows,
                        schema_kind=schema,
                        target_rows_per_fragment=rows,
                    ),
                    Step(
                        "delete",
                        expected,
                        mutation_count=count,
                        schema_kind=schema,
                        selection="random",
                    ),
                ]
                if percentage in {50, 90}:
                    steps.append(
                        Step(
                            "random_delete_reclaim",
                            expected,
                            mutation_count=count,
                            schema_kind=schema,
                            selection="random",
                            target_rows_per_fragment=max(1, expected),
                        )
                    )
                yield MatrixCase(
                    f"pack-random-delete-{percentage}/{schema}",
                    schema,
                    rows_per_fragment,
                    min(profile["take_counts"][-1], expected),
                    tuple(steps),
                )
    if "indexed_n_to_one_compaction" in selected:
        for fragments in profile["logical_fragment_counts"]:
            for schema, index_kind in (("narrow16", "scalar"), ("vector", "vector")):
                yield MatrixCase(
                    f"indexed-compact-{fragments}-to-1/{index_kind}",
                    schema,
                    _rows_per_fragment(rows, fragments),
                    min(profile["take_counts"][-1], rows),
                    (
                        Step("create", rows, schema_kind=schema),
                        Step(
                            "default_compaction",
                            rows,
                            schema_kind=schema,
                            index_kind=index_kind,
                            target_rows_per_fragment=rows,
                        ),
                    ),
                    index_kind,
                )
    if "indexed_repeated_compaction" in selected:
        fragments = profile["logical_fragment_counts"][-1]
        for rounds in profile["repeated_compaction_rounds"]:
            for schema, index_kind in (("narrow16", "scalar"), ("vector", "vector")):
                steps = [
                    Step("create", rows, schema_kind=schema),
                ]
                steps.extend(
                    Step(
                        "default_compaction",
                        rows,
                        step=step,
                        schema_kind=schema,
                        index_kind=index_kind,
                        target_rows_per_fragment=rows,
                    )
                    for step in range(rounds)
                )
                yield MatrixCase(
                    f"indexed-repeated-compaction-{rounds}/{index_kind}",
                    schema,
                    _rows_per_fragment(rows, fragments),
                    min(profile["take_counts"][-1], rows),
                    tuple(steps),
                    index_kind,
                )
    if "scalar_index" in selected:
        count = max(1, rows // 100)
        yield MatrixCase(
            "scalar-index",
            "narrow16",
            default_rows_per_fragment,
            min(profile["take_counts"][-1], rows),
            (
                Step("create", rows),
                Step("index_take", rows, index_kind="scalar"),
                Step("append", rows + count, mutation_count=count, id_start=rows),
                Step("index_optimize", rows + count, index_kind="scalar"),
                Step("index_take", rows + count, index_kind="scalar"),
            ),
            "scalar",
        )
    if "vector_index" in selected:
        count = max(1, rows // 100)
        yield MatrixCase(
            "vector-index",
            "vector",
            default_rows_per_fragment,
            min(profile["take_counts"][-1], rows),
            (
                Step("create", rows, schema_kind="vector"),
                Step("index_take", rows, schema_kind="vector", index_kind="vector"),
                Step(
                    "append",
                    rows + count,
                    mutation_count=count,
                    id_start=rows,
                    schema_kind="vector",
                ),
                Step(
                    "index_optimize",
                    rows + count,
                    schema_kind="vector",
                    index_kind="vector",
                ),
                Step(
                    "index_take",
                    rows + count,
                    schema_kind="vector",
                    index_kind="vector",
                ),
            ),
            "vector",
        )


def policy_triggers(
    record: dict[str, Any], policy: dict[str, Any]
) -> tuple[bool, dict[str, float]]:
    physical = record["physical_data_bytes"]
    live = record["estimated_live_data_bytes"]
    fragments = record["fragments"]
    amplification = record["scan_byte_amplification"]
    if physical is None or live is None or fragments is None or amplification is None:
        raise ValueError("physical maintenance metrics are unavailable")
    target_bytes = policy["target_topology"]["target_file_size_bytes"]
    metrics = {
        "live_byte_ratio": live / max(1, physical),
        "physical_fragment_count_ratio": fragments
        / max(1, math.ceil(live / target_bytes)),
        "scan_byte_amplification": amplification,
    }
    decisions = []
    for condition in policy["trigger"]["conditions"]:
        value = metrics[condition["metric"]]
        if condition["operator"] == "lt":
            decisions.append(value < condition["threshold"])
        elif condition["operator"] == "gt":
            decisions.append(value > condition["threshold"])
        else:
            raise ValueError(f"unsupported policy operator: {condition['operator']}")
    if policy["trigger"]["semantics"] != "any":
        raise ValueError("runner only accepts frozen any-trigger policy")
    return any(decisions), metrics


class ProtocolRunner:
    def __init__(
        self,
        *,
        executable: Path,
        output: Path,
        dataset_root: str,
        storage: str,
        mode: str,
        commit: str,
        host: str,
        seed: int,
        policy: dict[str, Any],
        policy_sha256: str,
        policy_version: int,
        run_id: str,
        rows: int,
        rows_per_fragment: int,
        take_count: int,
        matrix_sha256: str,
        shard_id: str,
        resume: bool,
    ) -> None:
        self.executable = executable
        self.output = output
        self.dataset_root = dataset_root
        self.storage = storage
        self.mode = mode
        self.commit = commit
        self.host = host
        self.seed = seed
        self.policy = policy
        self.policy_sha256 = policy_sha256
        self.policy_version = policy_version
        self.run_id = run_id
        self.rows = rows
        self.rows_per_fragment = rows_per_fragment
        self.take_count = take_count
        self.matrix_sha256 = matrix_sha256
        self.shard_id = shard_id
        self.phase_index = 0
        self.records = 0
        self.failures: list[str] = []
        self.boundaries = 0
        self.pmr_triggers = 0
        self._existing_records: dict[tuple[str, str], dict[str, Any]] = {}
        if resume:
            existing_lines = (
                output.read_text(encoding="utf-8").splitlines()
                if output.exists()
                else []
            )
            for line_number, line in enumerate(existing_lines, 1):
                record = run.validate_record(json.loads(line))
                key = (record["pair_id"], record["format"])
                if key in self._existing_records:
                    raise ValueError(
                        f"duplicate resumed record at line {line_number}: {key}"
                    )
                self._existing_records[key] = record
            self._sink = output.open(
                "a" if output.exists() else "x", encoding="utf-8", buffering=1
            )
        else:
            self._sink = output.open("x", encoding="utf-8", buffering=1)
        self.records = len(self._existing_records)
        self.checkpoint_path = Path(f"{output}.checkpoint.json")
        try:
            self._checkpoint = self._load_or_initialize_checkpoint(resume)
        except Exception:
            self._sink.close()
            raise
        self.fixture_lineage_path = Path(f"{output}.fixture_lineage.jsonl")
        self._lineage_keys: set[tuple[str, str]] = set()
        if resume:
            existing_lineage = (
                self.fixture_lineage_path.read_text(encoding="utf-8").splitlines()
                if self.fixture_lineage_path.exists()
                else []
            )
            for line in existing_lineage:
                lineage = json.loads(line)
                self._lineage_keys.add((lineage["target_uri"], lineage["format"]))
            self._lineage_sink = self.fixture_lineage_path.open(
                "a" if self.fixture_lineage_path.exists() else "x",
                encoding="utf-8",
                buffering=1,
            )
        else:
            self._lineage_sink = self.fixture_lineage_path.open(
                "x", encoding="utf-8", buffering=1
            )
        self._fixtures: set[tuple[str, int, int, str]] = set()
        self.take_ids_root = output.parent / f"{output.name}.{run_id}.take_ids"
        self.maintenance_plans_root = (
            output.parent / f"{output.name}.{run_id}.maintenance_plans"
        )

    def close(self) -> None:
        self._sink.close()
        self._lineage_sink.close()

    def _checkpoint_identity(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "suite": "stable_row_address_protocol_checkpoint",
            "run_id": self.run_id,
            "commit": self.commit,
            "matrix_sha256": self.matrix_sha256,
            "policy_sha256": self.policy_sha256,
            "profile": self.mode,
            "seed": self.seed,
            "shard_id": self.shard_id,
        }

    def _load_or_initialize_checkpoint(self, resume: bool) -> dict[str, Any]:
        identity = self._checkpoint_identity()
        if not resume:
            checkpoint = identity | {
                "completed_records": 0,
                "output_size": 0,
                "last_completed_unit": None,
                "inflight": None,
            }
            replace_json_atomic(self.checkpoint_path, checkpoint)
            return checkpoint
        if not self.checkpoint_path.exists():
            if self._existing_records:
                raise FileNotFoundError(
                    "resume output exists without its checkpoint; restart this shard under a fresh prefix"
                )
            checkpoint = identity | {
                "completed_records": 0,
                "output_size": 0,
                "last_completed_unit": None,
                "inflight": None,
            }
            replace_json_atomic(self.checkpoint_path, checkpoint)
            return checkpoint
        checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        for field, expected in identity.items():
            if checkpoint.get(field) != expected:
                raise ValueError(
                    f"resume checkpoint {field} mismatch: "
                    f"{checkpoint.get(field)!r} != {expected!r}"
                )
        inflight = checkpoint.get("inflight")
        if inflight is not None:
            key = (inflight["pair_id"], inflight["format"])
            if key not in self._existing_records:
                raise RuntimeError(
                    "resume found an ambiguous worker outcome with no durable record: "
                    f"{key}; restart this shard under a fresh prefix"
                )
            checkpoint["inflight"] = None
        output_size = output_size_bytes(self.output)
        if checkpoint.get("output_size", 0) > output_size:
            raise ValueError("checkpoint output_size exceeds durable JSONL size")
        checkpoint["completed_records"] = len(self._existing_records)
        checkpoint["output_size"] = output_size
        replace_json_atomic(self.checkpoint_path, checkpoint)
        return checkpoint

    def _set_inflight(self, pair_id: str, format_name: str) -> None:
        self._checkpoint["inflight"] = {
            "pair_id": pair_id,
            "format": format_name,
        }
        replace_json_atomic(self.checkpoint_path, self._checkpoint)

    def _record_completed(self) -> None:
        self._checkpoint["inflight"] = None
        self._checkpoint["completed_records"] = len(self._existing_records)
        self._checkpoint["output_size"] = output_size_bytes(self.output)
        replace_json_atomic(self.checkpoint_path, self._checkpoint)

    def complete_unit(self, unit: str) -> None:
        self._checkpoint["last_completed_unit"] = unit
        replace_json_atomic(self.checkpoint_path, self._checkpoint)

    def dataset_uri(self, track: str, case: str, repeat: int, format_name: str) -> str:
        safe_case = case.replace("/", "__")
        suffix = (
            f"{self.run_id}/{track}/{safe_case}/repeat-{repeat:03d}/{format_name}.lance"
        )
        if self.dataset_root.startswith("s3://"):
            return f"{self.dataset_root.rstrip('/')}/{suffix}"
        return str((Path(self.dataset_root).expanduser().resolve() / suffix).resolve())

    def fixture_uri(
        self,
        schema_kind: str,
        rows: int,
        rows_per_fragment: int,
        index_kind: str,
        format_name: str,
    ) -> str:
        suffix = (
            f"{self.run_id}/fixtures/{schema_kind}/rows-{rows}/"
            f"rows-per-fragment-{rows_per_fragment}/index-{index_kind}/"
            f"{format_name}.lance"
        )
        if self.dataset_root.startswith("s3://"):
            return f"{self.dataset_root.rstrip('/')}/{suffix}"
        return str((Path(self.dataset_root).expanduser().resolve() / suffix).resolve())

    def _expected(self, step: Step, **identity: Any) -> dict[str, Any]:
        return {
            "schema_version": run.SCHEMA_VERSION,
            "suite": run.SUITE,
            "run_id": self.run_id,
            "commit": self.commit,
            "host": self.host,
            "seed": self.seed,
            "policy_sha256": self.policy_sha256,
            "policy_version": self.policy_version,
            "mode": self.mode,
            "storage": self.storage,
            "operation": step.operation,
            "timing_scope": run.TIMING_SCOPES[step.operation],
            "rows": self.rows,
            "rows_per_fragment": self.rows_per_fragment,
            "take_count": self.take_count,
            "expected_rows": step.expected_rows,
            "mutation_count": step.mutation_count,
            "id_start": step.id_start,
            "step": step.step,
            "selection_step": step.selection_step,
            "match_percent": step.match_percent,
            "schema_kind": SCHEMA_RECORD_NAMES[step.schema_kind],
            "index_kind": INDEX_RECORD_NAMES[step.index_kind],
            "selection": SELECTION_RECORD_NAMES[step.selection],
            "implementation_path": step.implementation_path,
            **identity,
        }

    def _command(
        self,
        step: Step,
        *,
        uri: str,
        format_name: str,
        pair_id: str,
        repeat: int,
        order_index: int,
        take_ids_input: Path | None = None,
        prepare_take_ids_output: Path | None = None,
        prepare_maintenance_plan_output: Path | None = None,
        maintenance_plan_input: Path | None = None,
        maintenance_plan_sha256: str | None = None,
        validate_maintenance_plan_only: bool = False,
        target_file_size_bytes: int | None = None,
        max_source_fragments_per_group: int | None = None,
        source_dataset_uri: str | None = None,
    ) -> tuple[str, ...]:
        return run._worker_command(
            self.executable,
            uri=uri,
            format_name=format_name,
            storage=self.storage,
            operation=step.operation,
            mode=self.mode,
            run_id=self.run_id,
            pair_id=pair_id,
            round_index=repeat,
            order_index=order_index,
            rows=self.rows,
            rows_per_fragment=self.rows_per_fragment,
            take_count=self.take_count,
            expected_rows=step.expected_rows,
            mutation_count=step.mutation_count,
            id_start=step.id_start,
            step=step.step,
            selection_step=step.selection_step,
            match_percent=step.match_percent,
            schema_kind=step.schema_kind,
            index_kind=step.index_kind,
            update_driver=step.update_driver,
            selection=step.selection,
            target_rows_per_fragment=step.target_rows_per_fragment,
            target_file_size_bytes=(
                target_file_size_bytes
                or self.policy["target_topology"]["target_file_size_bytes"]
            ),
            max_source_fragments_per_group=(
                max_source_fragments_per_group
                or self.policy["target_topology"]["max_source_fragments_per_group"]
            ),
            seed=self.seed,
            commit=self.commit,
            host=self.host,
            policy_sha256=self.policy_sha256,
            policy_version=self.policy_version,
            source_dataset_uri=source_dataset_uri,
            take_ids_input=take_ids_input,
            prepare_take_ids_output=prepare_take_ids_output,
            prepare_maintenance_plan_output=prepare_maintenance_plan_output,
            maintenance_plan_input=maintenance_plan_input,
            maintenance_plan_sha256=maintenance_plan_sha256,
            validate_maintenance_plan_only=validate_maintenance_plan_only,
        )

    def invoke_one(
        self,
        step: Step,
        *,
        track: str,
        case: str,
        repeat: int,
        format_name: str,
        pair_id: str,
        order_index: int,
        take_ids_input: Path | None = None,
        dataset_uri_override: str | None = None,
        source_dataset_uri: str | None = None,
        maintenance_plan: tuple[Path, str] | None = None,
        max_source_fragments_per_group: int | None = None,
        target_file_size_bytes: int | None = None,
    ) -> dict[str, Any]:
        uri = dataset_uri_override or self.dataset_uri(track, case, repeat, format_name)
        expected = self._expected(
            step,
            pair_id=pair_id,
            round=repeat,
            order_index=order_index,
            dataset_uri=uri,
            format=format_name,
            maintenance_plan_path=(
                str(maintenance_plan[0]) if maintenance_plan is not None else None
            ),
            maintenance_plan_sha256=(
                maintenance_plan[1] if maintenance_plan is not None else None
            ),
        )
        key = (pair_id, format_name)
        if key in self._existing_records:
            record = run.validate_record(self._existing_records[key], expected)
            if record["status"] != "ok":
                self.failures.append(
                    f"{pair_id}/{format_name}: {record['error'] or 'worker error'}"
                )
            return record
        command = self._command(
            step,
            uri=uri,
            format_name=format_name,
            pair_id=pair_id,
            repeat=repeat,
            order_index=order_index,
            take_ids_input=take_ids_input,
            source_dataset_uri=source_dataset_uri,
            maintenance_plan_input=(
                maintenance_plan[0] if maintenance_plan is not None else None
            ),
            maintenance_plan_sha256=(
                maintenance_plan[1] if maintenance_plan is not None else None
            ),
            max_source_fragments_per_group=max_source_fragments_per_group,
            target_file_size_bytes=target_file_size_bytes,
        )
        self._set_inflight(pair_id, format_name)
        print(
            f"track={track} case={case} repeat={repeat} step={step.step} "
            f"operation={step.operation} format={format_name}",
            file=sys.stderr,
        )
        result = subprocess.run(
            command,
            cwd=run.REPOSITORY_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=None,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"worker exited with status {result.returncode}: {command[0]}"
            )
        record = run.parse_worker_stdout(result.stdout, expected)
        self._sink.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
        self._sink.write("\n")
        self._sink.flush()
        os.fsync(self._sink.fileno())
        self.records += 1
        self._existing_records[key] = record
        self._record_completed()
        if record["status"] != "ok":
            self.failures.append(f"{pair_id}/{format_name}: {record['error']}")
        return record

    def ensure_fixture(
        self, schema_kind: str, rows_per_fragment: int, index_kind: str
    ) -> None:
        key = (schema_kind, self.rows, rows_per_fragment, index_kind)
        if key in self._fixtures:
            return
        if index_kind != "none":
            self.ensure_fixture(schema_kind, rows_per_fragment, "none")
            clone_step = Step("fixture_clone", self.rows, schema_kind=schema_kind)
            clone_pair_id = (
                f"{self.run_id}/fixtures/{schema_kind}/rows-{self.rows}/"
                f"rows-per-fragment-{rows_per_fragment}/index-{index_kind}/"
                "fixture_clone"
            )
            order = run.format_order(0, self.phase_index)
            self.phase_index += 1
            clone_records = []
            for order_index, format_name in enumerate(order):
                source_uri = self.fixture_uri(
                    schema_kind,
                    self.rows,
                    rows_per_fragment,
                    "none",
                    format_name,
                )
                target_uri = self.fixture_uri(
                    schema_kind,
                    self.rows,
                    rows_per_fragment,
                    index_kind,
                    format_name,
                )
                clone_records.append(
                    self.invoke_one(
                        clone_step,
                        track="fixtures",
                        case=f"{schema_kind}-{rows_per_fragment}-index-{index_kind}",
                        repeat=0,
                        format_name=format_name,
                        pair_id=clone_pair_id,
                        order_index=order_index,
                        dataset_uri_override=target_uri,
                        source_dataset_uri=source_uri,
                    )
                )
                self.write_fixture_lineage(
                    track="fixtures",
                    case=f"{schema_kind}-{rows_per_fragment}-index-{index_kind}",
                    repeat=0,
                    format_name=format_name,
                    schema_kind=schema_kind,
                    index_kind="none",
                    rows_per_fragment=rows_per_fragment,
                    source_uri=source_uri,
                    target_uri=target_uri,
                    pair_id=clone_pair_id,
                )
            self.require_success(
                clone_records,
                f"fixture/{schema_kind}/{rows_per_fragment}/{index_kind}/clone",
            )
            index_step = Step(
                "index_build",
                self.rows,
                schema_kind=schema_kind,
                index_kind=index_kind,
            )
            index_pair_id = (
                f"{self.run_id}/fixtures/{schema_kind}/rows-{self.rows}/"
                f"rows-per-fragment-{rows_per_fragment}/index-{index_kind}/index_build"
            )
            order = run.format_order(0, self.phase_index)
            self.phase_index += 1
            index_records = [
                self.invoke_one(
                    index_step,
                    track="fixtures",
                    case=f"{schema_kind}-{rows_per_fragment}-index-{index_kind}",
                    repeat=0,
                    format_name=format_name,
                    pair_id=index_pair_id,
                    order_index=order_index,
                    dataset_uri_override=self.fixture_uri(
                        schema_kind,
                        self.rows,
                        rows_per_fragment,
                        index_kind,
                        format_name,
                    ),
                )
                for order_index, format_name in enumerate(order)
            ]
            self.require_success(
                index_records,
                f"fixture/{schema_kind}/{rows_per_fragment}/{index_kind}/index",
            )
            self._fixtures.add(key)
            return
        step = Step("create", self.rows, schema_kind=schema_kind)
        pair_id = (
            f"{self.run_id}/fixtures/{schema_kind}/rows-{self.rows}/"
            f"rows-per-fragment-{rows_per_fragment}/index-none/create"
        )
        order = run.format_order(0, self.phase_index)
        self.phase_index += 1
        records = []
        for order_index, format_name in enumerate(order):
            records.append(
                self.invoke_one(
                    step,
                    track="fixtures",
                    case=f"{schema_kind}-{rows_per_fragment}",
                    repeat=0,
                    format_name=format_name,
                    pair_id=pair_id,
                    order_index=order_index,
                    dataset_uri_override=self.fixture_uri(
                        schema_kind,
                        self.rows,
                        rows_per_fragment,
                        "none",
                        format_name,
                    ),
                )
            )
        self.require_success(records, f"fixture/{schema_kind}/{rows_per_fragment}")
        self._fixtures.add(key)

    def clone_fixture_all(
        self,
        *,
        track: str,
        case: str,
        repeat: int,
        schema_kind: str,
        rows_per_fragment: int,
        index_kind: str,
        label: str,
    ) -> dict[str, dict[str, Any]]:
        self.ensure_fixture(schema_kind, rows_per_fragment, index_kind)
        step = Step(
            "fixture_clone",
            self.rows,
            schema_kind=schema_kind,
            index_kind=index_kind,
        )
        pair_id = f"{self.run_id}/{track}/{case}/repeat-{repeat:03d}/{label}"
        order = run.format_order(repeat, self.phase_index)
        self.phase_index += 1
        records: dict[str, dict[str, Any]] = {}
        for order_index, format_name in enumerate(order):
            source_uri = self.fixture_uri(
                schema_kind,
                self.rows,
                rows_per_fragment,
                index_kind,
                format_name,
            )
            target_uri = self.dataset_uri(track, case, repeat, format_name)
            record = self.invoke_one(
                step,
                track=track,
                case=case,
                repeat=repeat,
                format_name=format_name,
                pair_id=pair_id,
                order_index=order_index,
                source_dataset_uri=source_uri,
            )
            records[format_name] = record
            self.write_fixture_lineage(
                track=track,
                case=case,
                repeat=repeat,
                format_name=format_name,
                schema_kind=schema_kind,
                index_kind=index_kind,
                rows_per_fragment=rows_per_fragment,
                source_uri=source_uri,
                target_uri=target_uri,
                pair_id=pair_id,
            )
        return records

    def write_fixture_lineage(
        self,
        *,
        track: str,
        case: str,
        repeat: int,
        format_name: str,
        schema_kind: str,
        index_kind: str,
        rows_per_fragment: int,
        source_uri: str,
        target_uri: str,
        pair_id: str,
    ) -> None:
        lineage_key = (target_uri, format_name)
        if lineage_key in self._lineage_keys:
            return
        lineage = {
            "schema_version": 1,
            "run_id": self.run_id,
            "track": track,
            "case": case,
            "repeat": repeat,
            "format": format_name,
            "schema_kind": schema_kind,
            "index_kind": index_kind,
            "rows": self.rows,
            "rows_per_fragment": rows_per_fragment,
            "source_uri": source_uri,
            "target_uri": target_uri,
            "pair_id": pair_id,
        }
        self._lineage_sink.write(
            json.dumps(lineage, sort_keys=True, separators=(",", ":")) + "\n"
        )
        self._lineage_sink.flush()
        os.fsync(self._lineage_sink.fileno())
        self._lineage_keys.add(lineage_key)

    def invoke_all(
        self,
        step: Step,
        *,
        track: str,
        case: str,
        repeat: int,
        label: str,
        maintenance_plan: tuple[Path, str] | None = None,
        max_source_fragments_per_group: int | None = None,
        target_file_size_bytes: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        pair_id = f"{self.run_id}/{track}/{case}/repeat-{repeat:03d}/{label}"
        order = run.format_order(repeat, self.phase_index)
        self.phase_index += 1
        if maintenance_plan is not None:
            self.validate_maintenance_plan_formats(
                step,
                track=track,
                case=case,
                repeat=repeat,
                formats=order,
                maintenance_plan=maintenance_plan,
                max_source_fragments_per_group=max_source_fragments_per_group,
                target_file_size_bytes=target_file_size_bytes,
            )
        return {
            format_name: self.invoke_one(
                step,
                track=track,
                case=case,
                repeat=repeat,
                format_name=format_name,
                pair_id=pair_id,
                order_index=order_index,
                maintenance_plan=maintenance_plan,
                max_source_fragments_per_group=max_source_fragments_per_group,
                target_file_size_bytes=target_file_size_bytes,
            )
            for order_index, format_name in enumerate(order)
        }

    def validate_maintenance_plan_formats(
        self,
        step: Step,
        *,
        track: str,
        case: str,
        repeat: int,
        formats: Sequence[str],
        maintenance_plan: tuple[Path, str],
        max_source_fragments_per_group: int | None,
        target_file_size_bytes: int | None,
    ) -> None:
        for order_index, format_name in enumerate(formats):
            command = self._command(
                step,
                uri=self.dataset_uri(track, case, repeat, format_name),
                format_name=format_name,
                pair_id=(
                    f"{self.run_id}/{track}/{case}/repeat-{repeat:03d}/"
                    "maintenance-plan-validation"
                ),
                repeat=repeat,
                order_index=order_index,
                maintenance_plan_input=maintenance_plan[0],
                maintenance_plan_sha256=maintenance_plan[1],
                validate_maintenance_plan_only=True,
                max_source_fragments_per_group=max_source_fragments_per_group,
                target_file_size_bytes=target_file_size_bytes,
            )
            result = subprocess.run(
                command,
                cwd=run.REPOSITORY_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=None,
            )
            if result.returncode != 0 or result.stdout.strip():
                raise RuntimeError(
                    "maintenance plan is not compatible with every paired format: "
                    f"format={format_name}, status={result.returncode}, "
                    f"stdout={result.stdout!r}"
                )

    def prepare_take_ids(
        self,
        step: Step,
        *,
        track: str,
        case: str,
        repeat: int,
        format_name: str,
        label: str,
        order_index: int,
    ) -> Path:
        output = (
            self.take_ids_root
            / track
            / case.replace("/", "__")
            / f"repeat-{repeat:03d}"
            / label
            / f"{format_name}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            return output
        pair_id = f"{self.run_id}/{track}/{case}/repeat-{repeat:03d}/{label}/prepare"
        command = self._command(
            dataclasses.replace(step, operation="take"),
            uri=self.dataset_uri(track, case, repeat, format_name),
            format_name=format_name,
            pair_id=pair_id,
            repeat=repeat,
            order_index=order_index,
            prepare_take_ids_output=output,
        )
        result = subprocess.run(
            command,
            cwd=run.REPOSITORY_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=None,
        )
        if result.returncode != 0 or result.stdout.strip():
            raise RuntimeError(
                "prepare-take-ids failed or emitted stdout: "
                f"status={result.returncode}, stdout={result.stdout!r}"
            )
        return output

    def prepare_maintenance_plan(
        self,
        step: Step,
        *,
        track: str,
        case: str,
        repeat: int,
        label: str,
        source_format: str,
        max_source_fragments_per_group: int,
        target_file_size_bytes: int,
    ) -> tuple[Path, str]:
        output = (
            self.maintenance_plans_root
            / track
            / case.replace("/", "__")
            / f"repeat-{repeat:03d}"
            / label
            / f"{source_format}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            pair_id = (
                f"{self.run_id}/{track}/{case}/repeat-{repeat:03d}/{label}/prepare"
            )
            command = self._command(
                step,
                uri=self.dataset_uri(track, case, repeat, source_format),
                format_name=source_format,
                pair_id=pair_id,
                repeat=repeat,
                order_index=0,
                prepare_maintenance_plan_output=output,
                max_source_fragments_per_group=max_source_fragments_per_group,
                target_file_size_bytes=target_file_size_bytes,
            )
            result = subprocess.run(
                command,
                cwd=run.REPOSITORY_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=None,
            )
            if result.returncode != 0 or result.stdout.strip():
                raise RuntimeError(
                    "prepare-maintenance-plan failed or emitted stdout: "
                    f"status={result.returncode}, stdout={result.stdout!r}"
                )
        plan = json.loads(output.read_text(encoding="utf-8"))
        fields = {
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
        }
        plan = _strict_object(plan, fields, "physical maintenance plan")
        expected = {
            "schema_version": 1,
            "suite": "stable_row_address_physical_maintenance_plan",
            "run_id": self.run_id,
            "commit": self.commit,
            "policy_sha256": self.policy_sha256,
            "source_format": source_format,
            "source_dataset_uri": self.dataset_uri(track, case, repeat, source_format),
            "schema_kind": SCHEMA_RECORD_NAMES[step.schema_kind],
            "expected_rows": step.expected_rows,
            "target_rows_per_fragment": step.target_rows_per_fragment,
            "target_file_size_bytes": target_file_size_bytes,
            "max_source_fragments_per_group": max_source_fragments_per_group,
        }
        mismatches = {
            field: (value, plan.get(field))
            for field, value in expected.items()
            if plan.get(field) != value
        }
        if mismatches:
            raise ValueError(f"physical maintenance plan mismatch: {mismatches}")
        groups = plan["groups"]
        if not isinstance(groups, list) or len(groups) != 1:
            raise ValueError(
                "physical maintenance plan must contain one frozen contiguous group"
            )
        expected_start = 0
        expected_outputs = 0
        planned_live_rows = 0
        for ordinal, group in enumerate(groups):
            group = _strict_object(
                group,
                {
                    "start_ordinal",
                    "end_ordinal",
                    "source_live_rows",
                    "source_physical_rows",
                    "source_physical_data_bytes",
                    "source_live_data_bytes",
                    "expected_output_fragments",
                },
                f"physical maintenance plan group {ordinal}",
            )
            if any(
                not isinstance(group[field], int) or group[field] < 0
                for field in (
                    "start_ordinal",
                    "end_ordinal",
                    "source_live_rows",
                    "source_physical_rows",
                    "source_physical_data_bytes",
                    "source_live_data_bytes",
                    "expected_output_fragments",
                )
            ):
                raise ValueError("physical maintenance group values must be non-negative integers")
            if (
                group["start_ordinal"] != expected_start
                or group["end_ordinal"] <= expected_start
                or group["end_ordinal"] - expected_start
                > max_source_fragments_per_group
            ):
                raise ValueError("physical maintenance plan groups are not contiguous")
            if (
                group["source_live_rows"] > group["source_physical_rows"]
                or group["source_live_data_bytes"]
                > group["source_physical_data_bytes"]
            ):
                raise ValueError("physical maintenance plan live source exceeds physical source")
            expected_start = group["end_ordinal"]
            expected_outputs += group["expected_output_fragments"]
            planned_live_rows += group["source_live_rows"]
        if (
            not isinstance(plan["execution_target_rows_per_fragment"], int)
            or plan["execution_target_rows_per_fragment"] <= 0
        ):
            raise ValueError("maintenance execution target rows must be positive")
        if expected_start != plan["fragment_count"]:
            raise ValueError("physical maintenance plan does not cover all fragments")
        if expected_outputs != plan["expected_output_fragment_count"]:
            raise ValueError("physical maintenance plan output count is inconsistent")
        live_bytes = groups[0]["source_live_data_bytes"]
        expected_execution_target = (
            max(1, planned_live_rows)
            if live_bytes == 0
            else max(
                1,
                min(
                    max(1, planned_live_rows),
                    target_file_size_bytes * planned_live_rows // live_bytes,
                ),
            )
        )
        expected_output_count = max(
            1,
            (planned_live_rows + expected_execution_target - 1)
            // expected_execution_target,
        )
        expected_output_live_rows = [
            expected_execution_target
        ] * (planned_live_rows // expected_execution_target)
        if planned_live_rows % expected_execution_target:
            expected_output_live_rows.append(
                planned_live_rows % expected_execution_target
            )
        if not expected_output_live_rows:
            expected_output_live_rows.append(0)
        if (
            planned_live_rows != step.expected_rows
            or plan["execution_target_rows_per_fragment"]
            != expected_execution_target
            or groups[0]["expected_output_fragments"] != expected_output_count
            or plan["expected_output_live_rows"] != expected_output_live_rows
            or plan["expected_output_fragment_count"] != expected_output_count
        ):
            raise ValueError(
                "physical maintenance plan does not implement the frozen byte formula"
            )
        canonical = json.dumps(
            plan, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return output, hashlib.sha256(canonical.encode()).hexdigest()

    def probes(
        self,
        *,
        track: str,
        case: str,
        repeat: int,
        expected_rows: int,
        schema_kind: str,
        index_kind: str,
        step_index: int,
    ) -> dict[str, dict[str, Any]]:
        common = Step(
            "open",
            expected_rows,
            step=step_index,
            schema_kind=schema_kind,
            index_kind=index_kind,
        )
        self.invoke_all(
            common,
            track=track,
            case=case,
            repeat=repeat,
            label=f"step-{step_index:03d}/cold-open",
        )
        scans = self.invoke_all(
            dataclasses.replace(common, operation="scan"),
            track=track,
            case=case,
            repeat=repeat,
            label=f"step-{step_index:03d}/cold-scan",
        )
        digests = {
            record["state_digest"]
            for record in scans.values()
            if record["status"] == "ok"
        }
        if len(digests) != 1:
            self.failures.append(
                f"{track}/{case}/repeat-{repeat}/step-{step_index}: "
                f"state digest mismatch: {sorted(digests)}"
            )

        take_step = dataclasses.replace(common, operation="take")
        pair_id = (
            f"{self.run_id}/{track}/{case}/repeat-{repeat:03d}/"
            f"step-{step_index:03d}/cold-take"
        )
        order = run.format_order(repeat, self.phase_index)
        self.phase_index += 1
        take_user_ids: dict[str, tuple[int, ...]] = {}
        take_artifacts: dict[str, Path] = {}
        for order_index, format_name in enumerate(order):
            take_ids = self.prepare_take_ids(
                take_step,
                track=track,
                case=case,
                repeat=repeat,
                format_name=format_name,
                label=f"step-{step_index:03d}/cold-take",
                order_index=order_index,
            )
            artifact = json.loads(take_ids.read_text(encoding="utf-8"))
            take_user_ids[format_name] = tuple(artifact["user_ids"])
            take_artifacts[format_name] = take_ids
        if len(set(take_user_ids.values())) != 1:
            failure = (
                f"{track}/{case}/repeat-{repeat}/step-{step_index}: "
                "paired cold takes selected different user rows"
            )
            self.failures.append(failure)
            raise RuntimeError(failure)
        for order_index, format_name in enumerate(order):
            self.invoke_one(
                take_step,
                track=track,
                case=case,
                repeat=repeat,
                format_name=format_name,
                pair_id=pair_id,
                order_index=order_index,
                take_ids_input=take_artifacts[format_name],
            )
        if index_kind != "none":
            self.invoke_all(
                dataclasses.replace(common, operation="index_take"),
                track=track,
                case=case,
                repeat=repeat,
                label=f"step-{step_index:03d}/cold-index-take",
            )
        return scans

    def require_success(self, records: Iterable[dict[str, Any]], context: str) -> None:
        errors: list[str] = []
        for record in records:
            if record["status"] != "ok":
                errors.append(
                    f"{record['format']}: {record.get('error') or 'worker error'}"
                )
            if record["placement_maintenance_required"] is True:
                errors.append(f"{record['format']}: unexpected placement backpressure")
        if errors:
            failure = f"{context}: {'; '.join(errors)}"
            self.failures.append(failure)
            raise RuntimeError(failure)


def run_matrix(
    runner: ProtocolRunner,
    profile: dict[str, Any],
    cases: Sequence[MatrixCase],
) -> None:
    if not cases:
        raise ValueError("matrix selection produced no cases")
    for case in cases:
        for repeat in range(profile["paired_repeats"]):
            runner.rows = profile["rows"]
            runner.rows_per_fragment = case.rows_per_fragment
            runner.take_count = case.take_count
            for step_index, step in enumerate(case.steps):
                if step.operation == "create":
                    records = runner.clone_fixture_all(
                        track="matrix",
                        case=case.name,
                        repeat=repeat,
                        schema_kind=case.schema_kind,
                        rows_per_fragment=case.rows_per_fragment,
                        index_kind=case.fixture_index_kind,
                        label=f"step-{step_index:03d}/fixture_clone",
                    )
                    runner.require_success(
                        records.values(),
                        f"matrix/{case.name}/step-{step_index}/fixture-clone",
                    )
                    continue
                if step.operation == "random_delete_reclaim":
                    source_version = records["v23_logical"]["dataset_version"]
                    if source_version is None:
                        raise RuntimeError(
                            "uniform-random delete did not report its committed version"
                        )
                    runner.probes(
                        track="matrix",
                        case=case.name,
                        repeat=repeat,
                        expected_rows=step.expected_rows,
                        schema_kind=case.schema_kind,
                        index_kind=step.index_kind,
                        step_index=step_index,
                    )
                    preflight = runner.invoke_one(
                        dataclasses.replace(step, operation="default_compaction"),
                        track="matrix",
                        case=case.name,
                        repeat=repeat,
                        format_name="v23_logical",
                        pair_id=(
                            f"{runner.run_id}/matrix/{case.name}/repeat-{repeat:03d}/"
                            f"step-{step_index:03d}/default-reclaim-preflight"
                        ),
                        order_index=0,
                    )
                    assert_default_not_admitted(
                        preflight,
                        source_version=source_version,
                        context="uniform-random delete reclaim",
                    )
                maintenance_plan = None
                max_source_fragments = None
                target_file_size_bytes = None
                if step.operation in {"default_compaction", "random_delete_reclaim"}:
                    max_source_fragments = max(profile["logical_fragment_counts"]) * 2
                    target_file_size_bytes = sys.maxsize
                    maintenance_plan = runner.prepare_maintenance_plan(
                        dataclasses.replace(step, step=step_index),
                        track="matrix",
                        case=case.name,
                        repeat=repeat,
                        label=f"step-{step_index:03d}/{step.operation}-plan",
                        source_format="v22_no_stable",
                        max_source_fragments_per_group=max_source_fragments,
                        target_file_size_bytes=target_file_size_bytes,
                    )
                records = runner.invoke_all(
                    dataclasses.replace(step, step=step_index),
                    track="matrix",
                    case=case.name,
                    repeat=repeat,
                    label=f"step-{step_index:03d}/{step.operation}",
                    maintenance_plan=maintenance_plan,
                    max_source_fragments_per_group=max_source_fragments,
                    target_file_size_bytes=target_file_size_bytes,
                )
                runner.require_success(
                    records.values(), f"matrix/{case.name}/step-{step_index}"
                )
            final = case.steps[-1]
            runner.probes(
                track="matrix",
                case=case.name,
                repeat=repeat,
                expected_rows=final.expected_rows,
                schema_kind=case.schema_kind,
                index_kind=final.index_kind,
                step_index=len(case.steps),
            )
            runner.complete_unit(f"matrix/{case.name}/repeat-{repeat:03d}")


def variant_config(variant: str) -> tuple[str, str]:
    if variant == "bare":
        return "narrow16", "none"
    if variant == "scalar":
        return "narrow16", "scalar"
    if variant == "vector":
        return "vector", "vector"
    raise ValueError(f"unsupported variant: {variant}")


def target_rows_for_policy(schema_kind: str, policy: dict[str, Any]) -> int:
    bytes_per_row = {"narrow16": 16, "wide128": 128, "vector": 528}[schema_kind]
    return max(1, policy["target_topology"]["target_file_size_bytes"] // bytes_per_row)


def fixture_keys_for_run(
    profile: dict[str, Any],
    tracks: Sequence[str],
    variants: Sequence[str],
    matrix_cases: Sequence[MatrixCase],
) -> set[tuple[str, int, str]]:
    keys = (
        {
            (case.schema_kind, case.rows_per_fragment, case.fixture_index_kind)
            for case in matrix_cases
        }
        if "matrix" in tracks
        else set()
    )
    if set(tracks) & {"sustained", "adversarial_natural", "adversarial_aligned"}:
        rows_per_fragment = _rows_per_fragment(
            profile["rows"], profile["logical_fragment_counts"][0]
        )
        for variant in variants:
            schema_kind, index_kind = variant_config(variant)
            keys.add((schema_kind, rows_per_fragment, index_kind))
    return keys


def fixture_keys_for_shard(
    fixture_keys: Iterable[tuple[str, int, str]], shard_count: int, shard_index: int
) -> set[tuple[str, int, str]]:
    fixture_keys = set(fixture_keys)
    data_keys = sorted(
        {
            (schema_kind, rows_per_fragment)
            for schema_kind, rows_per_fragment, _ in fixture_keys
        }
    )
    selected_data_keys = {
        key
        for ordinal, key in enumerate(data_keys)
        if ordinal % shard_count == shard_index
    }
    return {key for key in fixture_keys if (key[0], key[1]) in selected_data_keys}


def projected_canonical_payload_bytes(
    profile: dict[str, Any], fixture_keys: Iterable[tuple[str, int, str]]
) -> int:
    bytes_per_row = {"narrow16": 16, "wide128": 128, "vector": 528}
    data_keys = {
        (schema_kind, rows_per_fragment)
        for schema_kind, rows_per_fragment, _ in fixture_keys
    }
    return sum(
        profile["rows"] * bytes_per_row[schema_kind] * len(run.FORMATS)
        for schema_kind, _ in data_keys
    )


def projected_unique_initial_index_payload_bytes_lower_bound(
    profile: dict[str, Any],
    fixture_keys: Iterable[tuple[str, int, str]],
    matrix_cases: Sequence[MatrixCase],
) -> int:
    index_bytes_per_row = {"scalar": 48, "vector": 520}
    canonical = sum(
        profile["rows"] * index_bytes_per_row[index_kind] * len(run.FORMATS)
        for _, _, index_kind in set(fixture_keys)
        if index_kind != "none"
    )
    measured_builds = sum(
        profile["rows"]
        * index_bytes_per_row[step.index_kind]
        * len(run.FORMATS)
        * profile["paired_repeats"]
        for case in matrix_cases
        for step in case.steps
        if step.operation == "index_build"
    )
    return canonical + measured_builds


def projected_no_dedup_logical_payload_bytes(
    profile: dict[str, Any],
    tracks: Sequence[str],
    variants: Sequence[str],
    matrix_cases: Sequence[MatrixCase],
) -> tuple[int, int]:
    bytes_per_row = {"narrow16": 16, "wide128": 128, "vector": 528}
    index_bytes_per_row = {"scalar": 48, "vector": 520}
    repeats = profile["paired_repeats"]
    data = sum(
        profile["rows"] * bytes_per_row[case.schema_kind] * len(run.FORMATS) * repeats
        for case in matrix_cases
    )
    index = sum(
        profile["rows"]
        * index_bytes_per_row[case.fixture_index_kind]
        * len(run.FORMATS)
        * repeats
        for case in matrix_cases
        if case.fixture_index_kind != "none"
    )
    index += sum(
        profile["rows"]
        * index_bytes_per_row[step.index_kind]
        * len(run.FORMATS)
        * repeats
        for case in matrix_cases
        for step in case.steps
        if step.operation == "index_build"
    )
    for _ in set(tracks) & {
        "sustained",
        "adversarial_natural",
        "adversarial_aligned",
    }:
        for variant in variants:
            schema_kind, index_kind = variant_config(variant)
            data += (
                profile["rows"]
                * bytes_per_row[schema_kind]
                * len(run.FORMATS)
                * repeats
            )
            if index_kind != "none":
                index += (
                    profile["rows"]
                    * index_bytes_per_row[index_kind]
                    * len(run.FORMATS)
                    * repeats
                )
    return data, index


def projected_minimum_full_scan_payload_bytes(
    profile: dict[str, Any],
    tracks: Sequence[str],
    variants: Sequence[str],
    matrix_cases: Sequence[MatrixCase],
) -> int:
    bytes_per_row = {"narrow16": 16, "wide128": 128, "vector": 528}
    repeats = profile["paired_repeats"]
    matrix_probe_bytes = 0
    if "matrix" in tracks:
        for case in matrix_cases:
            row_bytes = bytes_per_row[case.schema_kind]
            matrix_probe_bytes += (
                case.steps[-1].expected_rows * row_bytes * len(run.FORMATS) * repeats
            )
            matrix_probe_bytes += sum(
                step.expected_rows * row_bytes * len(run.FORMATS) * repeats
                for step in case.steps
                if step.operation == "random_delete_reclaim"
            )
    repeated_probe_bytes = 0
    for track in set(tracks) & {
        "sustained",
        "adversarial_natural",
        "adversarial_aligned",
    }:
        if track == "sustained":
            probe_rounds = (
                profile["repeated_update_rounds"]
                + profile["minimum_sustained_boundaries"]
            )
        elif track == "adversarial_natural":
            probe_rounds = 2 * profile["repeated_update_rounds"]
        else:
            probe_rounds = profile["repeated_update_rounds"]
        repeated_probe_bytes += sum(
            profile["rows"]
            * bytes_per_row[variant_config(variant)[0]]
            * len(run.FORMATS)
            * repeats
            * probe_rounds
            for variant in variants
        )
    # Each cold-take probe performs an untimed full scan to bind the same
    # business rows to each format's row references. Count that scan alongside
    # the measured cold scan so remote storage and runtime projections are not
    # understated.
    return 2 * (matrix_probe_bytes + repeated_probe_bytes)


def setup_repeated_fixture(
    runner: ProtocolRunner,
    *,
    track: str,
    variant: str,
    repeat: int,
    rows: int,
    rows_per_fragment: int,
) -> tuple[str, str]:
    schema_kind, index_kind = variant_config(variant)
    runner.rows = rows
    runner.rows_per_fragment = rows_per_fragment
    records = runner.clone_fixture_all(
        track=track,
        case=variant,
        repeat=repeat,
        schema_kind=schema_kind,
        rows_per_fragment=rows_per_fragment,
        index_kind=index_kind,
        label="setup/fixture-clone",
    )
    runner.require_success(records.values(), f"{track}/{variant}/setup/fixture-clone")
    return schema_kind, index_kind


def run_sustained(
    runner: ProtocolRunner,
    profile: dict[str, Any],
    variants: Sequence[str],
) -> None:
    rows = profile["rows"]
    hot_set_rows = profile["hot_set_rows"]
    rounds = profile["repeated_update_rounds"]
    fragments = profile["logical_fragment_counts"][0]
    rows_per_fragment = _rows_per_fragment(rows, fragments)
    for variant in variants:
        for repeat in range(profile["paired_repeats"]):
            schema_kind, index_kind = setup_repeated_fixture(
                runner,
                track="sustained",
                variant=variant,
                repeat=repeat,
                rows=rows,
                rows_per_fragment=rows_per_fragment,
            )
            repeat_boundaries = 0
            for update_round in range(rounds):
                update = Step(
                    "update",
                    rows,
                    mutation_count=hot_set_rows,
                    step=update_round,
                    selection_step=0,
                    schema_kind=schema_kind,
                    index_kind=index_kind,
                    update_driver="exact-matched-merge",
                    selection="random",
                )
                records = runner.invoke_all(
                    update,
                    track="sustained",
                    case=variant,
                    repeat=repeat,
                    label=f"round-{update_round:03d}/update",
                )
                runner.require_success(
                    records.values(),
                    f"sustained/{variant}/repeat-{repeat}/round-{update_round}",
                )
                if index_kind != "none":
                    catchup = runner.invoke_all(
                        dataclasses.replace(update, operation="index_optimize"),
                        track="sustained",
                        case=variant,
                        repeat=repeat,
                        label=f"round-{update_round:03d}/index-catch-up",
                    )
                    runner.require_success(catchup.values(), "sustained index catch-up")
                scans = runner.probes(
                    track="sustained",
                    case=variant,
                    repeat=repeat,
                    expected_rows=rows,
                    schema_kind=schema_kind,
                    index_kind=index_kind,
                    step_index=update_round,
                )
                baseline = scans["v22_no_stable"]
                triggered, _ = policy_triggers(baseline, runner.policy)
                if triggered:
                    maintenance = Step(
                        "default_compaction",
                        rows,
                        step=update_round,
                        schema_kind=schema_kind,
                        index_kind=index_kind,
                        target_rows_per_fragment=target_rows_for_policy(
                            schema_kind, runner.policy
                        ),
                    )
                    max_source_fragments = runner.policy["target_topology"][
                        "max_source_fragments_per_group"
                    ]
                    maintenance_plan = runner.prepare_maintenance_plan(
                        maintenance,
                        track="sustained",
                        case=variant,
                        repeat=repeat,
                        label=f"round-{update_round:03d}/policy-maintenance-plan",
                        source_format="v22_no_stable",
                        max_source_fragments_per_group=max_source_fragments,
                        target_file_size_bytes=runner.policy["target_topology"][
                            "target_file_size_bytes"
                        ],
                    )
                    records = runner.invoke_all(
                        maintenance,
                        track="sustained",
                        case=variant,
                        repeat=repeat,
                        label=f"round-{update_round:03d}/policy-maintenance",
                        maintenance_plan=maintenance_plan,
                        max_source_fragments_per_group=max_source_fragments,
                        target_file_size_bytes=runner.policy["target_topology"][
                            "target_file_size_bytes"
                        ],
                    )
                    runner.require_success(records.values(), "sustained maintenance")
                    repeat_boundaries += 1
                    runner.boundaries += 1
                    runner.probes(
                        track="sustained",
                        case=variant,
                        repeat=repeat,
                        expected_rows=rows,
                        schema_kind=schema_kind,
                        index_kind=index_kind,
                        step_index=rounds + update_round,
                    )
                runner.complete_unit(
                    f"sustained/{variant}/repeat-{repeat:03d}/round-{update_round:03d}"
                )
            if repeat_boundaries < profile["minimum_sustained_boundaries"]:
                runner.failures.append(
                    f"sustained/{variant}/repeat-{repeat}: only {repeat_boundaries} "
                    "natural boundaries; fixture is invalid"
                )


def assert_pmr_preflight(record: dict[str, Any], context: str) -> None:
    if record["placement_maintenance_required"] is not True:
        return
    data = record["io_by_path"]
    if data is None or data["data"]["write_bytes"] != 0:
        raise RuntimeError(
            f"{context}: PlacementMaintenanceRequired wrote data objects before rejection"
        )


def assert_default_not_admitted(
    record: dict[str, Any], *, source_version: int, context: str
) -> None:
    data = record["io_by_path"]["data"]
    planned = record["compaction_groups_planned"]
    admitted = record["compaction_groups_admitted"]
    not_admitted = record["compaction_groups_not_admitted"]
    if (
        record["status"] != "ok"
        or record["placement_maintenance_required"] is True
        or record["admission"] is not False
        or planned is None
        or planned <= 0
        or admitted != 0
        or not_admitted != planned
        or record["dataset_version"] != source_version
        or data["put_requests"] != 0
        or data["write_bytes"] != 0
    ):
        raise RuntimeError(
            f"{context}: default compaction did not produce a side-effect-free "
            "not_admitted result"
        )


def run_adversarial_natural(
    runner: ProtocolRunner,
    profile: dict[str, Any],
    variants: Sequence[str],
) -> None:
    rows = profile["rows"]
    selected_rows = profile["hot_set_rows"]
    rounds = profile["repeated_update_rounds"]
    fragments = profile["logical_fragment_counts"][0]
    rows_per_fragment = _rows_per_fragment(rows, fragments)
    for variant in variants:
        for repeat in range(profile["paired_repeats"]):
            schema_kind, index_kind = setup_repeated_fixture(
                runner,
                track="adversarial_natural",
                variant=variant,
                repeat=repeat,
                rows=rows,
                rows_per_fragment=rows_per_fragment,
            )
            for update_round in range(rounds):
                update = Step(
                    "update",
                    rows,
                    mutation_count=selected_rows,
                    step=update_round,
                    selection_step=update_round,
                    schema_kind=schema_kind,
                    index_kind=index_kind,
                    update_driver="exact-matched-merge",
                    selection="random",
                )
                records = runner.invoke_all(
                    update,
                    track="adversarial_natural",
                    case=variant,
                    repeat=repeat,
                    label=f"round-{update_round:03d}/update-attempt",
                )
                candidate = records["v23_logical"]
                assert_pmr_preflight(candidate, "adversarial natural")
                if candidate["placement_maintenance_required"] is True:
                    runner.pmr_triggers += 1
                    maintenance = dataclasses.replace(
                        update,
                        operation="normalize_placement",
                        target_rows_per_fragment=target_rows_for_policy(
                            schema_kind, runner.policy
                        ),
                    )
                    max_source_fragments = runner.policy["target_topology"][
                        "max_source_fragments_per_group"
                    ]
                    maintenance_plan = runner.prepare_maintenance_plan(
                        maintenance,
                        track="adversarial_natural",
                        case=variant,
                        repeat=repeat,
                        label=f"round-{update_round:03d}/pmr-maintenance-plan",
                        source_format="v23_logical",
                        max_source_fragments_per_group=max_source_fragments,
                        target_file_size_bytes=runner.policy["target_topology"][
                            "target_file_size_bytes"
                        ],
                    )
                    maintained = runner.invoke_one(
                        maintenance,
                        track="adversarial_natural",
                        case=variant,
                        repeat=repeat,
                        format_name="v23_logical",
                        pair_id=(
                            f"{runner.run_id}/adversarial_natural/{variant}/"
                            f"repeat-{repeat:03d}/round-{update_round:03d}/pmr-maintenance"
                        ),
                        order_index=0,
                        maintenance_plan=maintenance_plan,
                        max_source_fragments_per_group=max_source_fragments,
                        target_file_size_bytes=runner.policy["target_topology"][
                            "target_file_size_bytes"
                        ],
                    )
                    if maintained["status"] != "ok":
                        continue
                    retry = runner.invoke_one(
                        update,
                        track="adversarial_natural",
                        case=variant,
                        repeat=repeat,
                        format_name="v23_logical",
                        pair_id=(
                            f"{runner.run_id}/adversarial_natural/{variant}/"
                            f"repeat-{repeat:03d}/round-{update_round:03d}/update-retry"
                        ),
                        order_index=0,
                    )
                    runner.require_success([retry], "adversarial candidate retry")
                else:
                    runner.require_success(records.values(), "adversarial update")
                if index_kind != "none":
                    catchup = runner.invoke_all(
                        dataclasses.replace(update, operation="index_optimize"),
                        track="adversarial_natural",
                        case=variant,
                        repeat=repeat,
                        label=f"round-{update_round:03d}/index-catch-up",
                    )
                    runner.require_success(
                        catchup.values(), "adversarial index catch-up"
                    )
                scans = runner.probes(
                    track="adversarial_natural",
                    case=variant,
                    repeat=repeat,
                    expected_rows=rows,
                    schema_kind=schema_kind,
                    index_kind=index_kind,
                    step_index=update_round,
                )
                for format_name, record in scans.items():
                    triggered, _ = policy_triggers(record, runner.policy)
                    if not triggered:
                        continue
                    maintenance = Step(
                        "default_compaction",
                        rows,
                        step=update_round,
                        schema_kind=schema_kind,
                        index_kind=index_kind,
                        target_rows_per_fragment=target_rows_for_policy(
                            schema_kind, runner.policy
                        ),
                    )
                    max_source_fragments = runner.policy["target_topology"][
                        "max_source_fragments_per_group"
                    ]
                    maintenance_plan = runner.prepare_maintenance_plan(
                        maintenance,
                        track="adversarial_natural",
                        case=variant,
                        repeat=repeat,
                        label=(
                            f"round-{update_round:03d}/natural-maintenance-plan/"
                            f"{format_name}"
                        ),
                        source_format=format_name,
                        max_source_fragments_per_group=max_source_fragments,
                        target_file_size_bytes=runner.policy["target_topology"][
                            "target_file_size_bytes"
                        ],
                    )
                    maintained = runner.invoke_one(
                        maintenance,
                        track="adversarial_natural",
                        case=variant,
                        repeat=repeat,
                        format_name=format_name,
                        pair_id=(
                            f"{runner.run_id}/adversarial_natural/{variant}/"
                            f"repeat-{repeat:03d}/round-{update_round:03d}/"
                            f"natural-maintenance/{format_name}"
                        ),
                        order_index=0,
                        maintenance_plan=maintenance_plan,
                        max_source_fragments_per_group=max_source_fragments,
                        target_file_size_bytes=runner.policy["target_topology"][
                            "target_file_size_bytes"
                        ],
                    )
                    runner.require_success([maintained], "natural policy maintenance")
                runner.probes(
                    track="adversarial_natural",
                    case=variant,
                    repeat=repeat,
                    expected_rows=rows,
                    schema_kind=schema_kind,
                    index_kind=index_kind,
                    step_index=rounds + update_round,
                )
                runner.complete_unit(
                    f"adversarial_natural/{variant}/repeat-{repeat:03d}/"
                    f"round-{update_round:03d}"
                )


def run_adversarial_aligned(
    runner: ProtocolRunner,
    profile: dict[str, Any],
    variants: Sequence[str],
) -> None:
    rows = profile["rows"]
    selected_rows = profile["hot_set_rows"]
    rounds = profile["repeated_update_rounds"]
    rows_per_fragment = _rows_per_fragment(rows, profile["logical_fragment_counts"][0])
    for variant in variants:
        for repeat in range(profile["paired_repeats"]):
            schema_kind, index_kind = setup_repeated_fixture(
                runner,
                track="adversarial_aligned",
                variant=variant,
                repeat=repeat,
                rows=rows,
                rows_per_fragment=rows_per_fragment,
            )
            for update_round in range(rounds):
                update = Step(
                    "update",
                    rows,
                    mutation_count=selected_rows,
                    step=update_round,
                    selection_step=update_round,
                    schema_kind=schema_kind,
                    index_kind=index_kind,
                    update_driver="exact-matched-merge",
                    selection="random",
                )
                candidate = runner.invoke_one(
                    update,
                    track="adversarial_aligned",
                    case=variant,
                    repeat=repeat,
                    format_name="v23_logical",
                    pair_id=(
                        f"{runner.run_id}/adversarial_aligned/{variant}/"
                        f"repeat-{repeat:03d}/round-{update_round:03d}/candidate-preflight"
                    ),
                    order_index=0,
                )
                assert_pmr_preflight(candidate, "adversarial aligned")
                if candidate["placement_maintenance_required"] is True:
                    runner.pmr_triggers += 1
                    normalize = dataclasses.replace(
                        update,
                        operation="normalize_placement",
                        target_rows_per_fragment=target_rows_for_policy(
                            schema_kind, runner.policy
                        ),
                    )
                    max_source_fragments = runner.policy["target_topology"][
                        "max_source_fragments_per_group"
                    ]
                    maintenance_plan = runner.prepare_maintenance_plan(
                        normalize,
                        track="adversarial_aligned",
                        case=variant,
                        repeat=repeat,
                        label=f"round-{update_round:03d}/aligned-maintenance-plan",
                        source_format="v23_logical",
                        max_source_fragments_per_group=max_source_fragments,
                        target_file_size_bytes=runner.policy["target_topology"][
                            "target_file_size_bytes"
                        ],
                    )
                    runner.validate_maintenance_plan_formats(
                        normalize,
                        track="adversarial_aligned",
                        case=variant,
                        repeat=repeat,
                        formats=run.FORMATS,
                        maintenance_plan=maintenance_plan,
                        max_source_fragments_per_group=max_source_fragments,
                        target_file_size_bytes=runner.policy["target_topology"][
                            "target_file_size_bytes"
                        ],
                    )
                    maintained = runner.invoke_one(
                        normalize,
                        track="adversarial_aligned",
                        case=variant,
                        repeat=repeat,
                        format_name="v23_logical",
                        pair_id=(
                            f"{runner.run_id}/adversarial_aligned/{variant}/"
                            f"repeat-{repeat:03d}/round-{update_round:03d}/normalize"
                        ),
                        order_index=0,
                        maintenance_plan=maintenance_plan,
                        max_source_fragments_per_group=max_source_fragments,
                        target_file_size_bytes=runner.policy["target_topology"][
                            "target_file_size_bytes"
                        ],
                    )
                    baseline_maintenance = dataclasses.replace(
                        update,
                        operation="default_compaction",
                        target_rows_per_fragment=target_rows_for_policy(
                            schema_kind, runner.policy
                        ),
                    )
                    baseline_records = [
                        runner.invoke_one(
                            baseline_maintenance,
                            track="adversarial_aligned",
                            case=variant,
                            repeat=repeat,
                            format_name=format_name,
                            pair_id=(
                                f"{runner.run_id}/adversarial_aligned/{variant}/"
                                f"repeat-{repeat:03d}/round-{update_round:03d}/"
                                f"forced-baseline-maintenance/{format_name}"
                            ),
                            order_index=order_index,
                            maintenance_plan=maintenance_plan,
                            max_source_fragments_per_group=max_source_fragments,
                            target_file_size_bytes=runner.policy["target_topology"][
                                "target_file_size_bytes"
                            ],
                        )
                        for order_index, format_name in enumerate(
                            ("v22_no_stable", "v22_stable")
                        )
                    ]
                    runner.require_success(
                        [maintained, *baseline_records], "aligned maintenance"
                    )
                    candidate = runner.invoke_one(
                        update,
                        track="adversarial_aligned",
                        case=variant,
                        repeat=repeat,
                        format_name="v23_logical",
                        pair_id=(
                            f"{runner.run_id}/adversarial_aligned/{variant}/"
                            f"repeat-{repeat:03d}/round-{update_round:03d}/candidate-retry"
                        ),
                        order_index=0,
                    )
                runner.require_success([candidate], "aligned candidate update")
                baseline_updates = [
                    runner.invoke_one(
                        update,
                        track="adversarial_aligned",
                        case=variant,
                        repeat=repeat,
                        format_name=format_name,
                        pair_id=(
                            f"{runner.run_id}/adversarial_aligned/{variant}/"
                            f"repeat-{repeat:03d}/round-{update_round:03d}/baseline-update"
                        ),
                        order_index=order_index,
                    )
                    for order_index, format_name in enumerate(
                        ("v22_no_stable", "v22_stable")
                    )
                ]
                runner.require_success(baseline_updates, "aligned baseline update")
                if index_kind != "none":
                    catchup = runner.invoke_all(
                        dataclasses.replace(update, operation="index_optimize"),
                        track="adversarial_aligned",
                        case=variant,
                        repeat=repeat,
                        label=f"round-{update_round:03d}/index-catch-up",
                    )
                    runner.require_success(catchup.values(), "aligned index catch-up")
                runner.probes(
                    track="adversarial_aligned",
                    case=variant,
                    repeat=repeat,
                    expected_rows=rows,
                    schema_kind=schema_kind,
                    index_kind=index_kind,
                    step_index=update_round,
                )
                runner.complete_unit(
                    f"adversarial_aligned/{variant}/repeat-{repeat:03d}/"
                    f"round-{update_round:03d}"
                )


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    replace_json_atomic(path, value)


def replace_json_atomic(path: Path, value: dict[str, Any]) -> None:
    replace_text_atomic(
        path,
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
    )


def replace_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as sink:
            sink.write(value)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def output_size_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--storage", choices=("ebs", "s3"), default="ebs")
    parser.add_argument("--profile", choices=("smoke", "release"), default="smoke")
    parser.add_argument(
        "--track",
        action="append",
        choices=("matrix", "sustained", "adversarial_natural", "adversarial_aligned"),
    )
    parser.add_argument(
        "--variant", action="append", choices=("bare", "scalar", "vector")
    )
    parser.add_argument("--case", action="append")
    parser.add_argument("--case-filter")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--policy", type=Path, default=run.DEFAULT_POLICY)
    parser.add_argument("--seed", type=int, default=0x4C414E43455F3233)
    parser.add_argument("--host")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--development-executable",
        type=Path,
        help="Smoke-only dirty-tree override; never valid as release evidence.",
    )
    parser.add_argument(
        "--development-tiny",
        action="store_true",
        help="Shrink smoke fixtures for local protocol plumbing validation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must be in 0..shard-count")
    if args.storage == "ebs" and args.dataset_root.startswith("s3://"):
        raise ValueError("--storage=ebs requires a local dataset root")
    if args.storage == "s3" and not args.dataset_root.startswith("s3://"):
        raise ValueError("--storage=s3 requires an s3:// dataset root")
    if args.profile == "release" and args.storage != "s3":
        raise ValueError("release profile requires same-region S3; EBS is smoke-only")
    matrix, matrix_canonical, matrix_sha256 = load_matrix(args.matrix)
    if args.development_tiny:
        if args.development_executable is None or args.profile != "smoke":
            raise ValueError(
                "--development-tiny requires --development-executable and --profile=smoke"
            )
        smoke = matrix["profiles"]["smoke"]
        smoke.update(
            {
                "rows": 4096,
                "logical_fragment_counts": [8],
                "take_counts": [16],
                "repeated_compaction_rounds": [3],
                "repeated_update_rounds": 3,
                "hot_set_rows": 41,
                "minimum_sustained_boundaries": 3,
            }
        )
        matrix_canonical = json.dumps(
            matrix, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        matrix_sha256 = hashlib.sha256(matrix_canonical.encode()).hexdigest()
    policy, policy_bytes, policy_sha256 = run.canonical_policy(args.policy)
    profile = matrix["profiles"][args.profile]
    tracks = args.track or ["matrix"]
    if len(set(tracks)) != len(tracks):
        raise ValueError("--track values must be unique")
    variants = args.variant or ["bare", "scalar", "vector"]
    selected_matrix_cases = set(args.case or matrix["tracks"]["matrix"]["cases"])
    unknown_matrix_cases = selected_matrix_cases - set(
        matrix["tracks"]["matrix"]["cases"]
    )
    if unknown_matrix_cases:
        raise ValueError(f"unknown matrix cases: {sorted(unknown_matrix_cases)}")
    matrix_cases = [
        case
        for case in iter_matrix_cases(profile, selected_matrix_cases)
        if args.case_filter is None or args.case_filter in case.name
    ]
    if "matrix" in tracks and not matrix_cases:
        raise ValueError("matrix selection produced no cases")
    all_fixture_keys = sorted(
        fixture_keys_for_run(profile, tracks, variants, matrix_cases)
    )
    selected_fixture_keys = fixture_keys_for_shard(
        all_fixture_keys, args.shard_count, args.shard_index
    )
    matrix_cases = [
        case
        for case in matrix_cases
        if (case.schema_kind, case.rows_per_fragment, case.fixture_index_kind)
        in selected_fixture_keys
    ]
    repeated_requested = set(tracks) & {
        "sustained",
        "adversarial_natural",
        "adversarial_aligned",
    }
    repeated_rows_per_fragment = _rows_per_fragment(
        profile["rows"], profile["logical_fragment_counts"][0]
    )
    variants = [
        variant
        for variant in variants
        if (
            variant_config(variant)[0],
            repeated_rows_per_fragment,
            variant_config(variant)[1],
        )
        in selected_fixture_keys
    ]
    tracks = [
        track
        for track in tracks
        if (track == "matrix" and matrix_cases)
        or (track in repeated_requested and variants)
    ]
    if not tracks:
        raise ValueError("selected shard contains no workload units")
    matrix_case_names = [case.name for case in matrix_cases]
    shard_id = f"shard-{args.shard_index:03d}-of-{args.shard_count:03d}"
    dataset_root = (
        f"{args.dataset_root.rstrip('/')}/{shard_id}"
        if args.shard_count > 1
        else args.dataset_root
    )
    output = args.output.expanduser().resolve()
    if args.shard_count > 1:
        output = output.with_name(
            f"{output.stem}.{shard_id}{output.suffix or '.jsonl'}"
        )

    if args.development_executable is not None:
        if args.profile != "smoke":
            raise ValueError("--development-executable is only allowed for smoke")
        executable = args.development_executable.expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(executable)
        commit_after = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=run.REPOSITORY_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        if run.SHA_PATTERN.fullmatch(commit_after) is None:
            raise RuntimeError("git returned an invalid full revision")
        source_provenance = "dirty-development-override"
    else:
        commit_before = run.source_revision()
        executable = run.build_harness()
        commit_after = run.source_revision()
        if commit_before != commit_after:
            raise RuntimeError("source revision changed during harness build")
        source_provenance = "clean-committed-source"
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = Path(f"{output}.protocol.json")
    summary_path = Path(f"{output}.summary.json")
    fixture_lineage_path = Path(f"{output}.fixture_lineage.jsonl")
    fixture_keys = fixture_keys_for_run(profile, tracks, variants, matrix_cases)
    projected_payload_bytes = projected_canonical_payload_bytes(profile, fixture_keys)
    projected_unique_index_bytes = (
        projected_unique_initial_index_payload_bytes_lower_bound(
            profile, fixture_keys, matrix_cases
        )
    )
    projected_logical_data_bytes, projected_logical_index_bytes = (
        projected_no_dedup_logical_payload_bytes(
            profile, tracks, variants, matrix_cases
        )
    )
    projected_scan_bytes = projected_minimum_full_scan_payload_bytes(
        profile, tracks, variants, matrix_cases
    )
    if args.resume:
        if not sidecar.is_file():
            raise FileNotFoundError("--resume requires an existing sidecar")
        sidecar_value = json.loads(sidecar.read_text(encoding="utf-8"))
        timestamp = sidecar_value["created_at_utc"]
        run_id = sidecar_value["run_id"]
        host = sidecar_value["host"]
    else:
        if output.exists() or sidecar.exists() or summary_path.exists():
            raise FileExistsError("protocol output, sidecar, or summary already exists")
        timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        run_id = f"stable-row-address-protocol-{timestamp}-{commit_after[:12]}"
        host = args.host or socket.gethostname()
        sidecar_value = {
            "schema_version": 1,
            "suite": "stable_row_address_design_protocol",
            "run_id": run_id,
            "created_at_utc": timestamp,
            "commit": commit_after,
            "source_provenance": source_provenance,
            "host": host,
            "seed": args.seed,
            "profile": args.profile,
            "tracks": tracks,
            "variants": variants,
            "matrix_case_names": matrix_case_names,
            "storage": args.storage,
            "dataset_root": dataset_root,
            "base_dataset_root": args.dataset_root,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "shard_id": shard_id,
            "shard_strategy": "schema_and_fragment_layout_fixture_locality",
            "output_jsonl": str(output),
            "executable": str(executable),
            "data_retention": "preserve",
            "storage_scope": (
                "same_region_s3_preserved_release"
                if args.profile == "release"
                else "bounded_smoke"
            ),
            "fixture_strategy": "canonical_base_per_format_schema_fragment_layout_then_shallow_clone",
            "fixture_lineage_jsonl": str(fixture_lineage_path),
            "checkpoint_json": str(Path(f"{output}.checkpoint.json")),
            "projected_canonical_payload_bytes": projected_payload_bytes,
            "projected_unique_initial_index_payload_bytes_lower_bound": projected_unique_index_bytes,
            "projected_no_dedup_logical_data_payload_bytes": projected_logical_data_bytes,
            "projected_no_dedup_logical_index_payload_bytes": projected_logical_index_bytes,
            "projected_minimum_full_scan_payload_bytes": projected_scan_bytes,
            "matrix_sha256": matrix_sha256,
            "matrix_canonical_json": matrix_canonical,
            "matrix": matrix,
            "policy_sha256": policy_sha256,
            "policy_canonical_json": policy_bytes.decode("utf-8"),
            "policy": policy,
        }
        write_json_atomic(sidecar, sidecar_value)
    resume_expected = {
        "commit": commit_after,
        "source_provenance": source_provenance,
        "seed": args.seed,
        "profile": args.profile,
        "tracks": tracks,
        "variants": variants,
        "matrix_case_names": matrix_case_names,
        "storage": args.storage,
        "dataset_root": dataset_root,
        "base_dataset_root": args.dataset_root,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "shard_id": shard_id,
        "shard_strategy": "schema_and_fragment_layout_fixture_locality",
        "output_jsonl": str(output),
        "matrix_sha256": matrix_sha256,
        "policy_sha256": policy_sha256,
        "projected_canonical_payload_bytes": projected_payload_bytes,
        "projected_unique_initial_index_payload_bytes_lower_bound": projected_unique_index_bytes,
        "projected_no_dedup_logical_data_payload_bytes": projected_logical_data_bytes,
        "projected_no_dedup_logical_index_payload_bytes": projected_logical_index_bytes,
        "projected_minimum_full_scan_payload_bytes": projected_scan_bytes,
    }
    resume_mismatches = {
        field: (expected, sidecar_value.get(field))
        for field, expected in resume_expected.items()
        if sidecar_value.get(field) != expected
    }
    if resume_mismatches:
        raise ValueError(
            f"protocol sidecar does not match invocation: {resume_mismatches}"
        )
    runner = ProtocolRunner(
        executable=executable,
        output=output,
        dataset_root=dataset_root,
        storage=args.storage,
        mode=args.profile,
        commit=commit_after,
        host=host,
        seed=args.seed,
        policy=policy,
        policy_sha256=policy_sha256,
        policy_version=policy["schema_version"],
        run_id=run_id,
        rows=profile["rows"],
        rows_per_fragment=_rows_per_fragment(
            profile["rows"], profile["logical_fragment_counts"][0]
        ),
        take_count=profile["take_counts"][-1],
        matrix_sha256=matrix_sha256,
        shard_id=shard_id,
        resume=args.resume,
    )
    execution_error: Exception | None = None
    try:
        for track in tracks:
            if track == "matrix":
                run_matrix(runner, profile, matrix_cases)
            elif track == "sustained":
                run_sustained(runner, profile, variants)
            elif track == "adversarial_natural":
                run_adversarial_natural(runner, profile, variants)
            elif track == "adversarial_aligned":
                run_adversarial_aligned(runner, profile, variants)
            else:
                raise AssertionError(track)
    except Exception as error:
        execution_error = error
        runner.failures.append(f"protocol execution aborted: {error}")
    finally:
        runner.close()
    summary = {
        "schema_version": 1,
        "suite": "stable_row_address_design_protocol_summary",
        "run_id": run_id,
        "records": runner.records,
        "sustained_boundaries": runner.boundaries,
        "placement_maintenance_required_triggers": runner.pmr_triggers,
        "failures": runner.failures,
        "verdict": "PASS" if not runner.failures else "FAIL",
    }
    if args.resume or summary_path.exists():
        replace_json_atomic(summary_path, summary)
    else:
        write_json_atomic(summary_path, summary)
    print(json.dumps(summary, sort_keys=True), file=sys.stderr)
    if execution_error is not None:
        raise execution_error
    return 0 if not runner.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
