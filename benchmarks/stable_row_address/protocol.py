#!/usr/bin/env python3
"""Execute the design-owned stable logical row-address benchmark protocol."""

from __future__ import annotations

import argparse
import copy
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
import environment_attestation  # noqa: E402


DEFAULT_MATRIX = SCRIPT_DIR / "workload_matrix.v2.json"
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
    "random_delete_reclaim_admission",
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
TRACK_ORDER = (
    "matrix",
    "sustained",
    "adversarial_natural",
    "adversarial_aligned",
)
VARIANT_ORDER = ("bare", "scalar", "vector")
RELEASE_SEED = 0x4C414E43455F3233
RELEASE_CONTRACT_FIELDS = {
    "shard_count",
    "operation_timeout_seconds",
    "fail_fast_runtime_ratio",
    "tracks",
    "variants",
    "matrix_case_names",
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


class PerformanceRegressionTimeout(RuntimeError):
    """The candidate exceeded the release fail-fast runtime budget."""


class OperationTimeout(RuntimeError):
    """A worker exceeded the absolute per-operation runtime budget."""


def execution_format_order(
    profile_name: str, repeat: int, scope: str, *, dynamic: bool = False
) -> tuple[str, ...]:
    """Return the replayable format order, with release baselines before v2.3."""

    order = (
        run.dynamic_format_order(repeat, scope)
        if dynamic
        else run.paired_format_order(repeat, scope)
    )
    if profile_name != "release":
        return order
    return tuple(
        format_name for format_name in order if format_name != "v23_logical"
    ) + ("v23_logical",)


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
        {
            "schema_version",
            "name",
            "profiles",
            "release_contract",
            "tracks",
            "measurement",
        },
        "workload matrix",
    )
    if matrix["schema_version"] != 2:
        raise ValueError("workload matrix schema_version must be 2")
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
        expected_reclaim_admission = {
            "smoke": "must_admit",
            "release": "must_not_admit",
        }[profile_name]
        if profile["random_delete_reclaim_admission"] != expected_reclaim_admission:
            raise ValueError(
                f"profile {profile_name}.random_delete_reclaim_admission must be "
                f"{expected_reclaim_admission}"
            )

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

    release_contract = _strict_object(
        matrix["release_contract"], RELEASE_CONTRACT_FIELDS, "release contract"
    )
    for field in ("shard_count", "operation_timeout_seconds"):
        _positive_int(release_contract[field], f"release contract.{field}")
    fail_fast_runtime_ratio = release_contract["fail_fast_runtime_ratio"]
    if (
        not isinstance(fail_fast_runtime_ratio, (int, float))
        or isinstance(fail_fast_runtime_ratio, bool)
        or fail_fast_runtime_ratio <= 1
    ):
        raise ValueError(
            "release contract.fail_fast_runtime_ratio must be greater than one"
        )
    release_tracks = release_contract["tracks"]
    if (
        not isinstance(release_tracks, list)
        or not release_tracks
        or len(set(release_tracks)) != len(release_tracks)
        or any(track not in TRACK_ORDER for track in release_tracks)
    ):
        raise ValueError("release contract.tracks must be unique canonical tracks")
    release_variants = release_contract["variants"]
    if (
        not isinstance(release_variants, list)
        or len(set(release_variants)) != len(release_variants)
        or any(variant not in VARIANT_ORDER for variant in release_variants)
    ):
        raise ValueError("release contract.variants must be unique canonical variants")
    repeated_tracks = set(release_tracks) - {"matrix"}
    if repeated_tracks and not release_variants:
        raise ValueError("release repeated tracks require at least one variant")
    if not repeated_tracks and release_variants:
        raise ValueError("release variants require at least one repeated track")
    release_case_names = release_contract["matrix_case_names"]
    if (
        not isinstance(release_case_names, list)
        or not release_case_names
        or len(set(release_case_names)) != len(release_case_names)
        or any(not isinstance(name, str) or not name for name in release_case_names)
    ):
        raise ValueError(
            "release contract.matrix_case_names must be unique non-empty strings"
        )
    if "matrix" not in release_tracks:
        raise ValueError("release contract must include the matrix track")

    canonical_release_cases(matrix)
    canonical_release_shard_count(matrix)

    _strict_object(matrix["measurement"], MEASUREMENT_FIELDS, "measurement")
    canonical = json.dumps(
        matrix, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return matrix, canonical, digest


def development_tiny_matrix(matrix: dict[str, Any]) -> dict[str, Any]:
    """Return the single canonical smoke-only reduction used by local plumbing tests."""

    reduced = copy.deepcopy(matrix)
    reduced["profiles"]["smoke"].update(
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
    return reduced


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
    compaction_mode: str = "standard"
    preflight_expected_admission: bool | None = None
    maintenance_target_file_size_bytes: int | None = None

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
        if self.compaction_mode not in {"standard", "fragment_reuse"}:
            raise ValueError(f"unsupported compaction_mode: {self.compaction_mode}")
        if (
            self.compaction_mode != "standard"
            and self.operation != "default_compaction"
        ):
            raise ValueError("non-standard compaction modes require default_compaction")
        if self.preflight_expected_admission is not None and self.operation not in {
            "default_compaction",
            "default_compaction_preflight",
            "random_delete_reclaim",
            "bounded_recluster",
        }:
            raise ValueError(
                "default-compaction admission requires a relocation or plan-only operation"
            )
        if (
            self.maintenance_target_file_size_bytes is not None
            and self.maintenance_target_file_size_bytes <= 0
        ):
            raise ValueError("maintenance target file size must be positive")

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
        if (
            self.operation == "default_compaction"
            and self.compaction_mode == "standard"
        ):
            return "default_compaction"
        if self.operation == "default_compaction_preflight":
            return "default_compaction_plan_only"
        if self.operation == "random_delete_reclaim":
            return "same_postcondition_repack_or_default_compaction"
        if self.operation == "bounded_recluster":
            return "same_postcondition_bounded_recluster"
        if self.operation in {"index_build", "index_take", "index_optimize"}:
            return INDEX_RECORD_NAMES[self.index_kind]
        if self.operation == "row_id_scan":
            return "full_scan_business_id_and_row_id_selection"
        return "native_dataset_api"

    def implementation_path_for_format(self, format_name: str) -> str:
        if format_name not in run.FORMATS:
            raise ValueError(f"unsupported benchmark format: {format_name}")
        if self.operation == "random_delete_reclaim":
            if format_name == "v23_logical":
                return "explicit_repack"
            if format_name == "v22_stable" and self.index_kind != "none":
                return "same_postcondition_default_compaction_full_index_rebuild"
            return "same_postcondition_default_compaction"
        if self.operation == "bounded_recluster":
            return (
                "default_bounded_recluster_fast_path"
                if format_name == "v23_logical"
                else "same_postcondition_bounded_recluster_rewrite"
            )
        if (
            self.operation == "default_compaction"
            and self.compaction_mode == "fragment_reuse"
        ):
            return {
                "v22_no_stable": "deferred_fragment_reuse_compaction",
                "v22_stable": "inline_index_remap_compaction",
                "v23_logical": "stable_logical_zero_remap_compaction",
            }[format_name]
        return self.implementation_path


@dataclasses.dataclass(frozen=True)
class MatrixCase:
    name: str
    schema_kind: str
    rows_per_fragment: int
    take_count: int
    steps: tuple[Step, ...]
    fixture_index_kind: str = "none"
    fixture_segments: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.steps or self.steps[0].operation != "create":
            raise ValueError("matrix cases must start from a create fixture")
        if not self.fixture_segments:
            return
        if any(
            rows <= 0 or rows_per_fragment <= 0
            for rows, rows_per_fragment in self.fixture_segments
        ):
            raise ValueError("fixture segments must contain positive row counts")
        if (
            sum(rows for rows, _ in self.fixture_segments)
            != self.steps[0].expected_rows
        ):
            raise ValueError("fixture segments must sum to the fixture row count")
        if self.fixture_segments[0][1] != self.rows_per_fragment:
            raise ValueError("first fixture segment must define rows_per_fragment")


FixtureKey = tuple[str, tuple[tuple[int, int], ...], str]


def fixture_segments_for_case(case: MatrixCase) -> tuple[tuple[int, int], ...]:
    return case.fixture_segments or (
        (case.steps[0].expected_rows, case.rows_per_fragment),
    )


def fixture_key_for_case(case: MatrixCase) -> FixtureKey:
    return (
        case.schema_kind,
        fixture_segments_for_case(case),
        case.fixture_index_kind,
    )


def fixture_layout_path(segments: tuple[tuple[int, int], ...]) -> str:
    rows = sum(segment_rows for segment_rows, _ in segments)
    if len(segments) == 1:
        return f"rows-{rows}/rows-per-fragment-{segments[0][1]}"
    canonical = json.dumps(segments, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return f"rows-{rows}/segmented-layout-{digest}"


def skewed_fixture_segments(rows: int, fragments: int) -> tuple[tuple[int, int], ...]:
    if rows % fragments != 0 or fragments % 2 != 0:
        raise ValueError("skewed fixture requires an even exact fragment partition")
    mean = rows // fragments
    if mean <= 1:
        raise ValueError("skewed fixture requires at least two rows per fragment")
    half = fragments // 2
    return (
        (half * (mean - 1), mean - 1),
        (half * (mean + 1), mean + 1),
    )


def _rows_per_fragment(rows: int, fragments: int) -> int:
    return max(1, math.ceil(rows / fragments))


def repeated_compaction_target_rows(
    rows: int, source_fragments: int, rounds: int, round_index: int
) -> int:
    if rounds <= 0 or source_fragments <= rounds:
        raise ValueError(
            "repeated compaction requires more source fragments than rounds"
        )
    if round_index < 0 or round_index >= rounds:
        raise ValueError("repeated compaction round index is out of range")
    rounds_remaining = rounds - round_index - 1
    target_fragments = (
        1
        if rounds_remaining == 0
        else math.ceil(source_fragments * rounds_remaining / rounds)
    )
    return math.ceil(rows / target_fragments)


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
                    inline_reclaim = delete_kind == "random" and percentage == 1
                    explicit_reclaim = delete_kind == "random" and percentage in {
                        50,
                        90,
                    }
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
                    if inline_reclaim:
                        steps.append(
                            Step(
                                "default_compaction",
                                rows - count,
                                mutation_count=count,
                                schema_kind=schema,
                                selection="random",
                                target_rows_per_fragment=max(1, rows - count),
                                preflight_expected_admission=True,
                            )
                        )
                    if explicit_reclaim:
                        steps.append(
                            Step(
                                "random_delete_reclaim",
                                rows - count,
                                mutation_count=count,
                                schema_kind=schema,
                                selection="random",
                                target_rows_per_fragment=max(1, rows - count),
                                preflight_expected_admission=(
                                    profile["random_delete_reclaim_admission"]
                                    == "must_admit"
                                ),
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

    if "delete_random" in selected:
        expected_reclaim_admission = (
            profile["random_delete_reclaim_admission"] == "must_admit"
        )
        for percentage in profile["delete_percentages"]:
            if percentage not in {50, 90}:
                continue
            count = rows * percentage // 100
            for schema, index_kind in (("narrow16", "scalar"), ("vector", "vector")):
                yield MatrixCase(
                    f"indexed-repack-random-delete-{percentage}/{index_kind}",
                    schema,
                    default_rows_per_fragment,
                    min(profile["take_counts"][-1], rows - count),
                    (
                        Step("create", rows, schema_kind=schema, index_kind=index_kind),
                        Step(
                            "delete",
                            rows - count,
                            mutation_count=count,
                            schema_kind=schema,
                            index_kind=index_kind,
                            selection="random",
                        ),
                        Step(
                            "random_delete_reclaim",
                            rows - count,
                            mutation_count=count,
                            schema_kind=schema,
                            index_kind=index_kind,
                            selection="random",
                            target_rows_per_fragment=max(1, rows - count),
                            preflight_expected_admission=expected_reclaim_admission,
                        ),
                    ),
                    index_kind,
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
        for fragments in profile["logical_fragment_counts"]:
            if fragments not in {10_000, 100_000}:
                continue
            segments = skewed_fixture_segments(rows, fragments)
            yield MatrixCase(
                f"compact-{fragments}-skew-to-1/narrow16",
                "narrow16",
                segments[0][1],
                min(profile["take_counts"][-1], rows),
                (
                    Step("create", rows, schema_kind="narrow16"),
                    Step(
                        "default_compaction",
                        rows,
                        schema_kind="narrow16",
                        target_rows_per_fragment=rows,
                    ),
                ),
                fixture_segments=segments,
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
                        target_rows_per_fragment=repeated_compaction_target_rows(
                            rows, fragments, rounds, step
                        ),
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
                if percentage == 1:
                    steps.append(
                        Step(
                            "default_compaction",
                            expected,
                            mutation_count=count,
                            schema_kind=schema,
                            selection="random",
                            target_rows_per_fragment=max(1, expected),
                            preflight_expected_admission=True,
                        )
                    )
                if percentage in {50, 90}:
                    steps.append(
                        Step(
                            "random_delete_reclaim",
                            expected,
                            mutation_count=count,
                            schema_kind=schema,
                            selection="random",
                            target_rows_per_fragment=max(1, expected),
                            preflight_expected_admission=(
                                profile["random_delete_reclaim_admission"]
                                == "must_admit"
                            ),
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
                        target_rows_per_fragment=repeated_compaction_target_rows(
                            rows, fragments, rounds, step
                        ),
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
    if "bounded_recluster" in selected:
        fragments = profile["logical_fragment_counts"][0]
        rows_per_fragment = _rows_per_fragment(rows, fragments)
        for schema, index_kind in (
            ("narrow16", "scalar"),
            ("wide128", "none"),
            ("vector", "vector"),
        ):
            yield MatrixCase(
                f"bounded-default-clustering-{fragments}/{schema}",
                schema,
                rows_per_fragment,
                min(profile["take_counts"][-1], rows),
                (
                    Step("create", rows, schema_kind=schema, index_kind=index_kind),
                    Step(
                        "bounded_recluster",
                        rows,
                        schema_kind=schema,
                        index_kind=index_kind,
                        target_rows_per_fragment=max(1, rows_per_fragment * 8),
                    ),
                ),
                index_kind,
            )
            yield MatrixCase(
                f"bounded-recluster-{fragments}/{schema}",
                schema,
                rows_per_fragment,
                min(profile["take_counts"][-1], rows),
                (
                    Step("create", rows, schema_kind=schema),
                    Step(
                        "recluster",
                        rows,
                        schema_kind=schema,
                        index_kind=index_kind,
                        target_rows_per_fragment=rows_per_fragment,
                    ),
                ),
                index_kind,
            )
    if "fragment_reuse" in selected:
        for fragments in profile["logical_fragment_counts"]:
            for schema, index_kind in (("narrow16", "scalar"), ("vector", "vector")):
                yield MatrixCase(
                    f"fragment-reuse-{fragments}-to-1/{index_kind}",
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
                            compaction_mode="fragment_reuse",
                        ),
                    ),
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


def canonical_release_cases(matrix: dict[str, Any]) -> tuple[MatrixCase, ...]:
    """Resolve the exact scale hypotheses frozen by the release contract."""

    generated = tuple(
        iter_matrix_cases(
            matrix["profiles"]["release"], set(matrix["tracks"]["matrix"]["cases"])
        )
    )
    by_name = {case.name: case for case in generated}
    if len(by_name) != len(generated):
        raise ValueError("release matrix generation produced duplicate case names")
    names = matrix["release_contract"]["matrix_case_names"]
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"release contract references unknown cases: {unknown}")
    return tuple(by_name[name] for name in names)


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
        operation_timeout_seconds: int | None = None,
        fail_fast_runtime_ratio: float | None = None,
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
        self.operation_timeout_seconds = operation_timeout_seconds
        self.fail_fast_runtime_ratio = fail_fast_runtime_ratio
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
        self._fixtures: set[FixtureKey] = set()
        self.take_ids_root = output.parent / f"{output.name}.{run_id}.take_ids"
        self.maintenance_plans_root = (
            output.parent / f"{output.name}.{run_id}.maintenance_plans"
        )

    def close(self) -> None:
        self._sink.close()
        self._lineage_sink.close()

    def _worker_timeout(
        self, pair_id: str | None, format_name: str | None
    ) -> float | None:
        fail_fast_timeout = self._candidate_fail_fast_timeout(pair_id, format_name)
        timeout = self.operation_timeout_seconds
        if fail_fast_timeout is None:
            return timeout
        return (
            fail_fast_timeout
            if timeout is None
            else min(float(timeout), fail_fast_timeout)
        )

    def _candidate_fail_fast_timeout(
        self, pair_id: str | None, format_name: str | None
    ) -> float | None:
        if (
            self.mode != "release"
            or format_name != "v23_logical"
            or pair_id is None
            or self.fail_fast_runtime_ratio is None
        ):
            return None
        baselines = [
            self._existing_records.get((pair_id, baseline))
            for baseline in ("v22_no_stable", "v22_stable")
        ]
        if any(record is None or record.get("status") != "ok" for record in baselines):
            return None
        baseline_seconds = min(record["duration_ns"] for record in baselines) / 1e9
        return max(5.0, baseline_seconds * self.fail_fast_runtime_ratio)

    def _run_worker(
        self,
        command: Sequence[str],
        *,
        pair_id: str | None = None,
        format_name: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        fail_fast_timeout = self._candidate_fail_fast_timeout(pair_id, format_name)
        timeout = self._worker_timeout(pair_id, format_name)
        try:
            return subprocess.run(
                command,
                cwd=run.REPOSITORY_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=None,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            if fail_fast_timeout is not None and fail_fast_timeout <= float(timeout):
                raise PerformanceRegressionTimeout(
                    "worker exceeded the release performance timeout: "
                    f"seconds={timeout}, pair_id={pair_id}, format={format_name}, "
                    f"fail_fast_runtime_ratio={self.fail_fast_runtime_ratio}"
                ) from error
            raise OperationTimeout(
                "worker exceeded the absolute operation timeout: "
                f"seconds={timeout}, pair_id={pair_id}, format={format_name}"
            ) from error

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
        fixture_segments: tuple[tuple[int, int], ...] | None = None,
    ) -> str:
        segments = fixture_segments or ((rows, rows_per_fragment),)
        if sum(segment_rows for segment_rows, _ in segments) != rows:
            raise ValueError("fixture URI segments do not sum to rows")
        suffix = (
            f"{self.run_id}/fixtures/{schema_kind}/{fixture_layout_path(segments)}/"
            f"index-{index_kind}/"
            f"{format_name}.lance"
        )
        if self.dataset_root.startswith("s3://"):
            return f"{self.dataset_root.rstrip('/')}/{suffix}"
        return str((Path(self.dataset_root).expanduser().resolve() / suffix).resolve())

    def _expected(self, step: Step, **identity: Any) -> dict[str, Any]:
        format_name = identity.get("format")
        if not isinstance(format_name, str):
            raise ValueError("worker identity must include format")
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
            "implementation_path": step.implementation_path_for_format(format_name),
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
            compaction_mode=step.compaction_mode,
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
        prepare_take_ids_output: Path | None = None,
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
            prepare_take_ids_output=prepare_take_ids_output,
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
        result = self._run_worker(command, pair_id=pair_id, format_name=format_name)
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
        self,
        schema_kind: str,
        rows_per_fragment: int,
        index_kind: str,
        fixture_segments: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        segments = fixture_segments or ((self.rows, rows_per_fragment),)
        total_rows = sum(segment_rows for segment_rows, _ in segments)
        if segments[0][1] != rows_per_fragment:
            raise ValueError(
                "fixture primary rows_per_fragment does not match segments"
            )
        key: FixtureKey = (schema_kind, segments, index_kind)
        if key in self._fixtures:
            return
        saved_rows = self.rows
        saved_rows_per_fragment = self.rows_per_fragment
        if index_kind != "none":
            self.ensure_fixture(
                schema_kind,
                rows_per_fragment,
                "none",
                fixture_segments=segments,
            )
            self.rows = total_rows
            self.rows_per_fragment = rows_per_fragment
            clone_step = Step("fixture_clone", total_rows, schema_kind=schema_kind)
            clone_pair_id = (
                f"{self.run_id}/fixtures/{schema_kind}/{fixture_layout_path(segments)}/"
                f"index-{index_kind}/"
                "fixture_clone"
            )
            order = execution_format_order(
                getattr(self, "mode", "smoke"), 0, clone_pair_id
            )
            clone_records = []
            for order_index, format_name in enumerate(order):
                source_uri = self.fixture_uri(
                    schema_kind,
                    total_rows,
                    rows_per_fragment,
                    "none",
                    format_name,
                    segments,
                )
                target_uri = self.fixture_uri(
                    schema_kind,
                    total_rows,
                    rows_per_fragment,
                    index_kind,
                    format_name,
                    segments,
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
                total_rows,
                schema_kind=schema_kind,
                index_kind=index_kind,
            )
            index_pair_id = (
                f"{self.run_id}/fixtures/{schema_kind}/{fixture_layout_path(segments)}/"
                f"index-{index_kind}/index_build"
            )
            order = execution_format_order(
                getattr(self, "mode", "smoke"), 0, index_pair_id
            )
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
                        total_rows,
                        rows_per_fragment,
                        index_kind,
                        format_name,
                        segments,
                    ),
                )
                for order_index, format_name in enumerate(order)
            ]
            self.require_success(
                index_records,
                f"fixture/{schema_kind}/{rows_per_fragment}/{index_kind}/index",
            )
            self._fixtures.add(key)
            self.rows = saved_rows
            self.rows_per_fragment = saved_rows_per_fragment
            return
        cumulative_rows = 0
        try:
            for segment_index, (segment_rows, segment_rows_per_fragment) in enumerate(
                segments
            ):
                self.rows = segment_rows
                self.rows_per_fragment = segment_rows_per_fragment
                operation = "create" if segment_index == 0 else "append"
                cumulative_before = cumulative_rows
                cumulative_rows += segment_rows
                step = Step(
                    operation,
                    cumulative_rows,
                    mutation_count=(segment_rows if operation == "append" else 1),
                    id_start=cumulative_before,
                    schema_kind=schema_kind,
                )
                label = (
                    "create" if segment_index == 0 else f"append-{segment_index:03d}"
                )
                pair_id = (
                    f"{self.run_id}/fixtures/{schema_kind}/{fixture_layout_path(segments)}/"
                    f"index-none/{label}"
                )
                order = execution_format_order(
                    getattr(self, "mode", "smoke"), 0, pair_id
                )
                records = [
                    self.invoke_one(
                        step,
                        track="fixtures",
                        case=f"{schema_kind}-{fixture_layout_path(segments)}",
                        repeat=0,
                        format_name=format_name,
                        pair_id=pair_id,
                        order_index=order_index,
                        dataset_uri_override=self.fixture_uri(
                            schema_kind,
                            total_rows,
                            rows_per_fragment,
                            "none",
                            format_name,
                            segments,
                        ),
                    )
                    for order_index, format_name in enumerate(order)
                ]
                self.require_success(
                    records,
                    f"fixture/{schema_kind}/{fixture_layout_path(segments)}/{label}",
                )
        finally:
            self.rows = saved_rows
            self.rows_per_fragment = saved_rows_per_fragment
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
        fixture_segments: tuple[tuple[int, int], ...] | None = None,
    ) -> dict[str, dict[str, Any]]:
        segments = fixture_segments or ((self.rows, rows_per_fragment),)
        total_rows = sum(segment_rows for segment_rows, _ in segments)
        if total_rows != self.rows:
            raise ValueError("fixture clone row count does not match protocol state")
        self.ensure_fixture(
            schema_kind,
            rows_per_fragment,
            index_kind,
            fixture_segments=segments,
        )
        step = Step(
            "fixture_clone",
            total_rows,
            schema_kind=schema_kind,
            index_kind=index_kind,
        )
        pair_id = f"{self.run_id}/{track}/{case}/repeat-{repeat:03d}/{label}"
        order = execution_format_order(getattr(self, "mode", "smoke"), repeat, pair_id)
        records: dict[str, dict[str, Any]] = {}
        for order_index, format_name in enumerate(order):
            source_uri = self.fixture_uri(
                schema_kind,
                total_rows,
                rows_per_fragment,
                index_kind,
                format_name,
                segments,
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
        order = execution_format_order(getattr(self, "mode", "smoke"), repeat, pair_id)
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

    def invoke_all_with_take_ids(
        self,
        step: Step,
        *,
        track: str,
        case: str,
        repeat: int,
        label: str,
    ) -> dict[str, dict[str, Any]]:
        pair_id = f"{self.run_id}/{track}/{case}/repeat-{repeat:03d}/{label}"
        order = execution_format_order(getattr(self, "mode", "smoke"), repeat, pair_id)
        take_artifacts = self.prepare_paired_take_ids(
            step,
            track=track,
            case=case,
            repeat=repeat,
            label=label,
            formats=order,
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
                take_ids_input=take_artifacts[format_name],
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
            result = self._run_worker(command)
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
        pair_id = (
            f"{self.run_id}/{track}/{case}/repeat-{repeat:03d}/{label}/row-id-scan"
        )
        key = (pair_id, format_name)
        if output.exists() and key not in self._existing_records:
            raise RuntimeError(
                "take-ID artifact exists without its durable row-id-scan record: "
                f"{output}; restart this shard under a fresh prefix"
            )
        scan_step = dataclasses.replace(step, operation="row_id_scan")
        record = self.invoke_one(
            scan_step,
            track=track,
            case=case,
            repeat=repeat,
            format_name=format_name,
            pair_id=pair_id,
            order_index=order_index,
            prepare_take_ids_output=output,
        )
        if record["status"] != "ok" or not output.is_file():
            raise RuntimeError(
                "row-id-scan did not produce a durable take-ID artifact: "
                f"status={record['status']}, output={output}"
            )
        return output

    def prepare_paired_take_ids(
        self,
        step: Step,
        *,
        track: str,
        case: str,
        repeat: int,
        label: str,
        formats: Sequence[str],
    ) -> dict[str, Path]:
        user_ids: dict[str, tuple[int, ...]] = {}
        artifacts: dict[str, Path] = {}
        for order_index, format_name in enumerate(formats):
            artifact_path = self.prepare_take_ids(
                step,
                track=track,
                case=case,
                repeat=repeat,
                format_name=format_name,
                label=label,
                order_index=order_index,
            )
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            user_ids[format_name] = tuple(artifact["user_ids"])
            artifacts[format_name] = artifact_path
        if len(set(user_ids.values())) != 1:
            failure = (
                f"{track}/{case}/repeat-{repeat}/{label}: "
                "paired take-ID setup selected different user rows"
            )
            self.failures.append(failure)
            raise RuntimeError(failure)
        return artifacts

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
            result = self._run_worker(command)
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
                raise ValueError(
                    "physical maintenance group values must be non-negative integers"
                )
            if (
                group["start_ordinal"] != expected_start
                or group["end_ordinal"] <= expected_start
                or group["end_ordinal"] - expected_start
                > max_source_fragments_per_group
            ):
                raise ValueError("physical maintenance plan groups are not contiguous")
            if (
                group["source_live_rows"] > group["source_physical_rows"]
                or group["source_live_data_bytes"] > group["source_physical_data_bytes"]
            ):
                raise ValueError(
                    "physical maintenance plan live source exceeds physical source"
                )
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
        expected_output_live_rows = [expected_execution_target] * (
            planned_live_rows // expected_execution_target
        )
        if planned_live_rows % expected_execution_target:
            expected_output_live_rows.append(
                planned_live_rows % expected_execution_target
            )
        if not expected_output_live_rows:
            expected_output_live_rows.append(0)
        if (
            planned_live_rows != step.expected_rows
            or plan["execution_target_rows_per_fragment"] != expected_execution_target
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
        order = execution_format_order(getattr(self, "mode", "smoke"), repeat, pair_id)
        take_artifacts = self.prepare_paired_take_ids(
            take_step,
            track=track,
            case=case,
            repeat=repeat,
            label=f"step-{step_index:03d}/cold-take",
            formats=order,
        )
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
            index_take_step = dataclasses.replace(common, operation="index_take")
            index_pair_id = (
                f"{self.run_id}/{track}/{case}/repeat-{repeat:03d}/"
                f"step-{step_index:03d}/cold-index-take"
            )
            index_order = execution_format_order(
                getattr(self, "mode", "smoke"), repeat, index_pair_id
            )
            for order_index, format_name in enumerate(index_order):
                self.invoke_one(
                    index_take_step,
                    track=track,
                    case=case,
                    repeat=repeat,
                    format_name=format_name,
                    pair_id=index_pair_id,
                    order_index=order_index,
                    take_ids_input=take_artifacts[format_name],
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
                        fixture_segments=fixture_segments_for_case(case),
                    )
                    runner.require_success(
                        records.values(),
                        f"matrix/{case.name}/step-{step_index}/fixture-clone",
                    )
                    continue
                if step.operation == "random_delete_reclaim":
                    runner.probes(
                        track="matrix",
                        case=case.name,
                        repeat=repeat,
                        expected_rows=step.expected_rows,
                        schema_kind=case.schema_kind,
                        index_kind=step.index_kind,
                        step_index=step_index,
                    )
                if step.preflight_expected_admission is not None:
                    source_version = records["v23_logical"]["dataset_version"]
                    if source_version is None:
                        raise RuntimeError(
                            "uniform-random delete did not report its committed version"
                        )
                    preflight_label = (
                        "default-reclaim-preflight"
                        if step.operation == "random_delete_reclaim"
                        else "default-compaction-preflight"
                    )
                    preflight = runner.invoke_one(
                        dataclasses.replace(
                            step, operation="default_compaction_preflight"
                        ),
                        track="matrix",
                        case=case.name,
                        repeat=repeat,
                        format_name="v23_logical",
                        pair_id=(
                            f"{runner.run_id}/matrix/{case.name}/repeat-{repeat:03d}/"
                            f"step-{step_index:03d}/{preflight_label}"
                        ),
                        order_index=0,
                    )
                    assert_default_compaction_preflight(
                        preflight,
                        source_version=source_version,
                        expected_admission=step.preflight_expected_admission,
                        context=f"{case.name} default compaction",
                    )
                maintenance_plan = None
                max_source_fragments = None
                target_file_size_bytes = None
                if step.operation in {
                    "default_compaction",
                    "random_delete_reclaim",
                    "bounded_recluster",
                    "recluster",
                }:
                    max_source_fragments = max(profile["logical_fragment_counts"]) * 2
                    source_format = (
                        "v23_logical"
                        if step.operation == "recluster"
                        else "v22_no_stable"
                    )
                    if step.maintenance_target_file_size_bytes is not None:
                        target_file_size_bytes = step.maintenance_target_file_size_bytes
                    else:
                        source = records[source_format]
                        live_rows = source["result_rows"]
                        live_bytes = source["estimated_live_data_bytes"]
                        physical_bytes = source["physical_data_bytes"]
                        if (
                            not isinstance(live_rows, int)
                            or live_rows <= 0
                            or not isinstance(live_bytes, int)
                            or live_bytes <= 0
                        ):
                            raise RuntimeError(
                                "maintenance source is missing positive live rows or bytes"
                            )
                        target_rows = min(step.target_rows_per_fragment, live_rows)
                        if target_rows == live_rows:
                            if (
                                not isinstance(physical_bytes, int)
                                or physical_bytes <= 0
                            ):
                                raise RuntimeError(
                                    "full-table maintenance target requires physical bytes"
                                )
                            target_file_size_bytes = physical_bytes
                        else:
                            target_file_size_bytes = max(
                                1,
                                (live_bytes * target_rows + live_rows - 1) // live_rows,
                            )
                    maintenance_plan = runner.prepare_maintenance_plan(
                        dataclasses.replace(step, step=step_index),
                        track="matrix",
                        case=case.name,
                        repeat=repeat,
                        label=f"step-{step_index:03d}/{step.operation}-plan",
                        source_format=source_format,
                        max_source_fragments_per_group=max_source_fragments,
                        target_file_size_bytes=target_file_size_bytes,
                    )
                measured_step = dataclasses.replace(step, step=step_index)
                if step.operation == "recluster":
                    pair_id = (
                        f"{runner.run_id}/matrix/{case.name}/repeat-{repeat:03d}/"
                        f"step-{step_index:03d}/recluster"
                    )
                    records = {
                        "v23_logical": runner.invoke_one(
                            measured_step,
                            track="matrix",
                            case=case.name,
                            repeat=repeat,
                            format_name="v23_logical",
                            pair_id=pair_id,
                            order_index=0,
                            maintenance_plan=maintenance_plan,
                            max_source_fragments_per_group=max_source_fragments,
                            target_file_size_bytes=target_file_size_bytes,
                        )
                    }
                elif step.operation == "index_take":
                    records = runner.invoke_all_with_take_ids(
                        measured_step,
                        track="matrix",
                        case=case.name,
                        repeat=repeat,
                        label=f"step-{step_index:03d}/{step.operation}",
                    )
                else:
                    records = runner.invoke_all(
                        measured_step,
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
) -> set[FixtureKey]:
    keys = (
        {fixture_key_for_case(case) for case in matrix_cases}
        if "matrix" in tracks
        else set()
    )
    if set(tracks) & {"sustained", "adversarial_natural", "adversarial_aligned"}:
        rows_per_fragment = _rows_per_fragment(
            profile["rows"], profile["logical_fragment_counts"][0]
        )
        for variant in variants:
            schema_kind, index_kind = variant_config(variant)
            keys.add((schema_kind, ((profile["rows"], rows_per_fragment),), index_kind))
    return keys


def canonical_release_shard_count(matrix: dict[str, Any]) -> int:
    """Return the fixture-local shard count frozen by the release contract."""

    profile = matrix["profiles"]["release"]
    contract = matrix["release_contract"]
    fixture_keys = fixture_keys_for_run(
        profile,
        contract["tracks"],
        contract["variants"],
        canonical_release_cases(matrix),
    )
    data_layouts = {
        (schema_kind, segments) for schema_kind, segments, _ in fixture_keys
    }
    derived = len(data_layouts)
    declared = contract["shard_count"]
    if declared != derived:
        raise ValueError(
            "release contract.shard_count must equal its distinct fixture layouts: "
            f"declared={declared}, derived={derived}"
        )
    return declared


def fixture_keys_for_shard(
    fixture_keys: Iterable[FixtureKey], shard_count: int, shard_index: int
) -> set[FixtureKey]:
    fixture_keys = set(fixture_keys)
    data_keys = sorted(
        {(schema_kind, segments) for schema_kind, segments, _ in fixture_keys}
    )
    selected_data_keys = {
        key
        for ordinal, key in enumerate(data_keys)
        if ordinal % shard_count == shard_index
    }
    return {key for key in fixture_keys if (key[0], key[1]) in selected_data_keys}


def projected_canonical_payload_bytes(
    profile: dict[str, Any], fixture_keys: Iterable[FixtureKey]
) -> int:
    bytes_per_row = {"narrow16": 16, "wide128": 128, "vector": 528}
    data_keys = {(schema_kind, segments) for schema_kind, segments, _ in fixture_keys}
    return sum(
        sum(segment_rows for segment_rows, _ in segments)
        * bytes_per_row[schema_kind]
        * len(run.FORMATS)
        for schema_kind, segments in data_keys
    )


def projected_unique_initial_index_payload_bytes_lower_bound(
    profile: dict[str, Any],
    fixture_keys: Iterable[FixtureKey],
    matrix_cases: Sequence[MatrixCase],
) -> int:
    index_bytes_per_row = {"scalar": 48, "vector": 520}
    canonical = sum(
        sum(segment_rows for segment_rows, _ in segments)
        * index_bytes_per_row[index_kind]
        * len(run.FORMATS)
        for _, segments, index_kind in set(fixture_keys)
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


def assert_default_compaction_preflight(
    record: dict[str, Any],
    *,
    source_version: int,
    expected_admission: bool,
    context: str,
) -> None:
    planned = record["compaction_groups_planned"]
    admitted = record["compaction_groups_admitted"]
    not_admitted = record["compaction_groups_not_admitted"]
    io_by_path = record["io_by_path"]
    wrote_objects = io_by_path is None or any(
        metrics["put_requests"] != 0
        or metrics["delete_requests"] != 0
        or metrics["write_bytes"] != 0
        for metrics in (io_by_path or {}).values()
    )
    reported_relocation = any(
        record[field] is not None
        for field in (
            "compacted_data_bytes",
            "index_storage_bytes_before",
            "row_addresses_remapped",
            "indices_remapped",
            "index_coverage_reuse",
            "layout_index_maintenance_ns",
        )
    )
    if (
        record["status"] != "ok"
        or record["operation"] != "default_compaction_preflight"
        or record["implementation_path"] != "default_compaction_plan_only"
        or record["placement_maintenance_required"] is True
        or record["admission"] is not expected_admission
        or planned is None
        or planned <= 0
        or admitted is None
        or not_admitted is None
        or planned != admitted + not_admitted
        or record["admission"] is not (admitted == planned)
        or record["dataset_version"] != source_version
        or record["put_requests"] != 0
        or record["delete_requests"] != 0
        or record["write_bytes"] != 0
        or (record["actual_put_attempts"] or 0) != 0
        or (record["actual_delete_attempts"] or 0) != 0
        or record["maintenance_plan_path"] is not None
        or record["maintenance_plan_sha256"] is not None
        or reported_relocation
        or wrote_objects
    ):
        raise RuntimeError(
            f"{context}: default compaction plan-only preflight did not produce the "
            "profile-required side-effect-free admission result"
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
                    maintenance_order_scope = (
                        f"adversarial_natural/{variant}/"
                        f"repeat-{repeat:03d}/round-{update_round:03d}/"
                        "pmr-maintenance"
                    )
                    maintenance_pair = f"{runner.run_id}/{maintenance_order_scope}"
                    maintenance_order = execution_format_order(
                        runner.mode, repeat, maintenance_order_scope, dynamic=True
                    )
                    candidate_order_index = maintenance_order.index("v23_logical")
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
                        pair_id=maintenance_pair,
                        order_index=candidate_order_index,
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
                        order_index=candidate_order_index,
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
                natural_maintenance_order_scope = (
                    f"adversarial_natural/{variant}/"
                    f"repeat-{repeat:03d}/round-{update_round:03d}/"
                    "natural-maintenance"
                )
                natural_maintenance_scope = (
                    f"{runner.run_id}/{natural_maintenance_order_scope}"
                )
                natural_maintenance_order = execution_format_order(
                    runner.mode,
                    repeat,
                    natural_maintenance_order_scope,
                    dynamic=True,
                )
                for order_index, format_name in enumerate(natural_maintenance_order):
                    record = scans[format_name]
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
                        pair_id=(f"{natural_maintenance_scope}/{format_name}"),
                        order_index=order_index,
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
                    baseline_maintenance = dataclasses.replace(
                        update,
                        operation="default_compaction",
                        target_rows_per_fragment=target_rows_for_policy(
                            schema_kind, runner.policy
                        ),
                    )
                    maintenance_order_scope = (
                        f"adversarial_aligned/{variant}/"
                        f"repeat-{repeat:03d}/round-{update_round:03d}/"
                        "aligned-maintenance"
                    )
                    maintenance_order = execution_format_order(
                        runner.mode, repeat, maintenance_order_scope, dynamic=True
                    )
                    maintenance_records: dict[str, dict[str, Any]] = {}
                    for order_index, format_name in enumerate(maintenance_order):
                        is_candidate = format_name == "v23_logical"
                        operation = normalize if is_candidate else baseline_maintenance
                        pair_id = (
                            f"{runner.run_id}/adversarial_aligned/{variant}/"
                            f"repeat-{repeat:03d}/round-{update_round:03d}/normalize"
                            if is_candidate
                            else (
                                f"{runner.run_id}/adversarial_aligned/{variant}/"
                                f"repeat-{repeat:03d}/round-{update_round:03d}/"
                                f"forced-baseline-maintenance/{format_name}"
                            )
                        )
                        maintenance_records[format_name] = runner.invoke_one(
                            operation,
                            track="adversarial_aligned",
                            case=variant,
                            repeat=repeat,
                            format_name=format_name,
                            pair_id=pair_id,
                            order_index=order_index,
                            maintenance_plan=maintenance_plan,
                            max_source_fragments_per_group=max_source_fragments,
                            target_file_size_bytes=runner.policy["target_topology"][
                                "target_file_size_bytes"
                            ],
                        )
                    maintained = maintenance_records["v23_logical"]
                    baseline_records = [
                        maintenance_records[format_name]
                        for format_name in ("v22_no_stable", "v22_stable")
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
                        order_index=maintenance_order.index("v23_logical"),
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
    parser.add_argument("--seed", type=int, default=RELEASE_SEED)
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
    if args.profile == "release" and args.seed != RELEASE_SEED:
        raise ValueError("release profile requires the canonical seed")
    if args.storage == "ebs" and args.dataset_root.startswith("s3://"):
        raise ValueError("--storage=ebs requires a local dataset root")
    if args.storage == "s3" and not args.dataset_root.startswith("s3://"):
        raise ValueError("--storage=s3 requires an s3:// dataset root")
    if args.profile == "release" and args.storage != "s3":
        raise ValueError("release profile requires same-region S3; EBS is smoke-only")
    matrix, matrix_canonical, matrix_sha256 = load_matrix(args.matrix)
    if args.profile == "release":
        release_shard_count = canonical_release_shard_count(matrix)
        if args.shard_count != release_shard_count:
            raise ValueError(
                "release profile requires exactly "
                f"{release_shard_count} canonical shards"
            )
        _, release_matrix_canonical, release_matrix_sha256 = load_matrix(DEFAULT_MATRIX)
        if (
            matrix_canonical != release_matrix_canonical
            or matrix_sha256 != release_matrix_sha256
        ):
            raise ValueError("release profile requires the repository default matrix")
    if args.development_tiny:
        if args.development_executable is None or args.profile != "smoke":
            raise ValueError(
                "--development-tiny requires --development-executable and --profile=smoke"
            )
        matrix = development_tiny_matrix(matrix)
        matrix_canonical = json.dumps(
            matrix, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        matrix_sha256 = hashlib.sha256(matrix_canonical.encode()).hexdigest()
    policy, policy_bytes, policy_sha256 = run.canonical_policy(args.policy)
    if args.profile == "release":
        _, release_policy_bytes, release_policy_sha256 = run.canonical_policy(
            run.DEFAULT_POLICY
        )
        if (
            policy_bytes != release_policy_bytes
            or policy_sha256 != release_policy_sha256
        ):
            raise ValueError("release profile requires the repository default policy")
    profile = matrix["profiles"][args.profile]
    if args.profile == "release":
        release_contract = matrix["release_contract"]
        release_tracks = tuple(release_contract["tracks"])
        release_variants = tuple(release_contract["variants"])
        if args.track is not None and tuple(args.track) != release_tracks:
            raise ValueError(
                "release profile requires the canonical sentinel track order"
            )
        if args.variant is not None and tuple(args.variant) != release_variants:
            raise ValueError(
                "release profile requires the canonical sentinel variant order"
            )
        if args.case is not None or args.case_filter is not None:
            raise ValueError("release profile does not allow focused matrix selection")
        tracks = list(release_tracks)
        variants = list(release_variants)
    else:
        tracks = args.track or ["matrix"]
        variants = args.variant or list(VARIANT_ORDER)
    if len(set(tracks)) != len(tracks):
        raise ValueError("--track values must be unique")
    if args.profile == "release":
        matrix_cases = list(canonical_release_cases(matrix))
    else:
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
    if "matrix" not in tracks:
        matrix_cases = []
    all_fixture_keys = sorted(
        fixture_keys_for_run(profile, tracks, variants, matrix_cases)
    )
    selected_fixture_keys = fixture_keys_for_shard(
        all_fixture_keys, args.shard_count, args.shard_index
    )
    matrix_cases = [
        case
        for case in matrix_cases
        if fixture_key_for_case(case) in selected_fixture_keys
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
            ((profile["rows"], repeated_rows_per_fragment),),
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
    storage_region_attestation = (
        environment_attestation.attest_same_region_s3(args.dataset_root)
        if args.profile == "release"
        else None
    )
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
            "development_tiny": args.development_tiny,
            "host": host,
            "seed": args.seed,
            "profile": args.profile,
            "cargo_profile": run.CARGO_PROFILE,
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
            "storage_region_attestation": storage_region_attestation,
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
        "development_tiny": args.development_tiny,
        "seed": args.seed,
        "profile": args.profile,
        "cargo_profile": run.CARGO_PROFILE,
        "tracks": tracks,
        "variants": variants,
        "matrix_case_names": matrix_case_names,
        "storage": args.storage,
        "storage_region_attestation": storage_region_attestation,
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
    operation_timeout_seconds = (
        matrix["release_contract"]["operation_timeout_seconds"]
        if args.profile == "release"
        else None
    )
    fail_fast_runtime_ratio = (
        float(matrix["release_contract"]["fail_fast_runtime_ratio"])
        if args.profile == "release"
        else None
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
        operation_timeout_seconds=operation_timeout_seconds,
        fail_fast_runtime_ratio=fail_fast_runtime_ratio,
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
        if isinstance(execution_error, PerformanceRegressionTimeout):
            return 75
        if isinstance(execution_error, OperationTimeout):
            return 74
        raise execution_error
    return 0 if not runner.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
