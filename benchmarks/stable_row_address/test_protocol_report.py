#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import protocol  # noqa: E402
import protocol_report  # noqa: E402
import run  # noqa: E402


COMMIT = "1" * 40


def make_sidecar(
    tracks: list[str],
    *,
    variants: list[str] | None = None,
    matrix_case_names: list[str] | None = None,
    profile: str = "smoke",
) -> dict[str, Any]:
    matrix, matrix_canonical, matrix_sha256 = protocol.load_matrix(
        protocol.DEFAULT_MATRIX
    )
    policy, policy_bytes, policy_sha256 = run.canonical_policy(run.DEFAULT_POLICY)
    dataset_root = "s3://release-bucket/data" if profile == "release" else "/tmp/data"
    return {
        "schema_version": 1,
        "suite": "stable_row_address_design_protocol",
        "run_id": "run-1",
        "created_at_utc": "20260712T000000.000000Z",
        "commit": COMMIT,
        "source_provenance": "clean-committed-source",
        "development_tiny": False,
        "host": "host-1",
        "seed": protocol.RELEASE_SEED if profile == "release" else 7,
        "profile": profile,
        "cargo_profile": "release-with-debug",
        "tracks": tracks,
        "variants": ["bare"] if variants is None else variants,
        "matrix_case_names": [] if matrix_case_names is None else matrix_case_names,
        "storage": "s3" if profile == "release" else "ebs",
        "dataset_root": dataset_root,
        "base_dataset_root": dataset_root,
        "shard_count": 1,
        "shard_index": 0,
        "shard_id": "shard-000-of-001",
        "shard_strategy": "schema_and_fragment_layout_fixture_locality",
        "output_jsonl": "/tmp/results.jsonl",
        "executable": "/tmp/stable_row_address_e2e",
        "data_retention": "preserve",
        "storage_scope": (
            "same_region_s3_preserved_release"
            if profile == "release"
            else "bounded_smoke"
        ),
        "storage_region_attestation": (
            {
                "schema_version": 1,
                "method": "ec2-imdsv2+aws-s3api-get-bucket-location",
                "instance_id": "i-0123456789abcdef0",
                "availability_zone": "us-east-2a",
                "compute_region": "us-east-2",
                "bucket": "release-bucket",
                "bucket_region": "us-east-2",
            }
            if profile == "release"
            else None
        ),
        "fixture_strategy": "canonical_base_per_format_schema_fragment_layout_then_shallow_clone",
        "fixture_lineage_jsonl": "/tmp/results.jsonl.fixture_lineage.jsonl",
        "checkpoint_json": "/tmp/results.jsonl.checkpoint.json",
        "projected_canonical_payload_bytes": 1,
        "projected_unique_initial_index_payload_bytes_lower_bound": 0,
        "projected_no_dedup_logical_data_payload_bytes": 1,
        "projected_no_dedup_logical_index_payload_bytes": 0,
        "projected_minimum_full_scan_payload_bytes": 1,
        "matrix_sha256": matrix_sha256,
        "matrix_canonical_json": matrix_canonical,
        "matrix": matrix,
        "policy_sha256": policy_sha256,
        "policy_canonical_json": policy_bytes.decode("utf-8"),
        "policy": policy,
    }


def operation_from_pair(pair_id: str) -> str:
    suffix = pair_id.rsplit("/", 1)[-1]
    if suffix in run.WORKER_OPERATIONS:
        return suffix
    if suffix.startswith("append-"):
        return "append"
    return {
        "create": "create",
        "fixture-clone": "fixture_clone",
        "index-build": "index_build",
        "update": "update",
        "update-attempt": "update",
        "index-catch-up": "index_optimize",
        "cold-open": "open",
        "cold-scan": "scan",
        "cold-take": "take",
        "cold-index-take": "index_take",
        "policy-maintenance": "default_compaction",
    }[suffix]


def make_record(
    sidecar: dict[str, Any],
    pair_id: str,
    format_name: str,
    *,
    operation: str | None = None,
    duration: int | None = None,
    index_kind: str = "none",
    state_digest: str | None = None,
    physical_order_digest: str | None = None,
    pmr: bool = False,
    not_admitted: bool = False,
) -> dict[str, Any]:
    operation = operation or operation_from_pair(pair_id)
    expected = protocol_report.expected_record_provenance(sidecar).get(
        (pair_id, format_name)
    )
    if expected is not None:
        operation = expected["operation"]
        index_kind = {
            "none": "none",
            "scalar_btree": "scalar",
            "vector_ivf_flat": "vector",
        }[expected["index_kind"]]
    expected_live_rows = expected["expected_rows"] if expected is not None else 65_536
    fixture_fragments = 1
    if "/fixtures/" in pair_id and "/segmented-layout-" in pair_id:
        profile = sidecar["matrix"]["profiles"][sidecar["profile"]]
        for case in protocol.iter_matrix_cases(
            profile,
            set(sidecar["matrix"]["tracks"]["matrix"]["cases"]),
        ):
            if not case.fixture_segments:
                continue
            layout = protocol.fixture_layout_path(case.fixture_segments)
            if f"/{layout}/" not in pair_id:
                continue
            label = pair_id.rsplit("/", 1)[-1]
            segment_index = 0 if label == "create" else int(label.split("-")[1])
            fixture_fragments = sum(
                rows // rows_per_fragment
                for rows, rows_per_fragment in case.fixture_segments[
                    : segment_index + 1
                ]
            )
            break
    repeat_match = re.search(r"/repeat-(\d{3})/", pair_id)
    repeat = int(repeat_match.group(1)) if repeat_match else 0
    update_match = re.search(r"/(?:round|step)-(\d{3})/", pair_id)
    step = int(update_match.group(1)) if update_match else 0
    multiplier = {"v22_no_stable": 100, "v22_stable": 110, "v23_logical": 90}[
        format_name
    ]
    duration = duration if duration is not None else multiplier
    is_preflight = operation == protocol_report.DEFAULT_COMPACTION_PREFLIGHT
    is_explicit_candidate = (
        format_name == "v23_logical"
        and operation in protocol_report.EXPLICIT_MATRIX_DIAGNOSTIC_OPERATIONS
    )
    is_fragment_reuse = (
        operation == "default_compaction" and "/fragment-reuse-" in pair_id
    )
    writes = (
        operation in protocol_report.COMMIT_OPERATIONS and not pmr and not not_admitted
    )
    reads = operation != "create"
    empty = {
        "get_requests": 0,
        "head_requests": 0,
        "list_requests": 0,
        "put_requests": 0,
        "delete_requests": 0,
        "read_bytes": 0,
        "write_bytes": 0,
    }
    data = dict(empty)
    metadata = dict(empty)
    index = dict(empty)
    if reads:
        data["get_requests"] = 1
        data["read_bytes"] = multiplier
        metadata["get_requests"] = 1
        metadata["read_bytes"] = multiplier // 2
    if writes:
        data["put_requests"] = 1
        data["write_bytes"] = multiplier
        metadata["put_requests"] = 1
        metadata["write_bytes"] = multiplier // 2
    if index_kind != "none" and operation in {"index_build", "index_optimize"}:
        index["put_requests"] = 1
        index["write_bytes"] = multiplier
    if index_kind != "none" and operation == "index_take":
        index["get_requests"] = 1
        index["read_bytes"] = multiplier
    io_by_path = {
        "data": data,
        "index": index,
        "metadata": metadata,
        "other": dict(empty),
    }
    totals = {
        field: sum(values[field] for values in io_by_path.values()) for field in empty
    }
    is_commit = operation in protocol_report.COMMIT_OPERATIONS
    record = {
        "schema_version": 1,
        "suite": run.SUITE,
        "run_id": sidecar["run_id"],
        "pair_id": pair_id,
        "commit": sidecar["commit"],
        "host": sidecar["host"],
        "seed": sidecar["seed"],
        "policy_sha256": sidecar["policy_sha256"],
        "policy_version": 1,
        "mode": sidecar["profile"],
        "format": format_name,
        "storage": sidecar["storage"],
        "operation": operation,
        "timing_scope": run.TIMING_SCOPES[operation],
        "round": repeat,
        "order_index": run.FORMATS.index(format_name),
        "dataset_uri": f"/tmp/{format_name}.lance",
        "rows": 65_536,
        "rows_per_fragment": 8192,
        "take_count": 1,
        "expected_rows": 65_536,
        "mutation_count": 1,
        "id_start": 0,
        "step": step,
        "selection_step": 0,
        "match_percent": 50,
        "schema_kind": "narrow_16b" if index_kind != "vector" else "vector_f32_128",
        "index_kind": {
            "none": "none",
            "scalar": "scalar_btree",
            "vector": "vector_ivf_flat",
        }[index_kind],
        "selection": "range",
        "implementation_path": (
            "native_update_builder"
            if operation == "update"
            else "default_compaction_plan_only"
            if is_preflight
            else "default_compaction"
            if operation == "default_compaction"
            else "native_dataset_api"
        ),
        "maintenance_plan_path": (
            "/tmp/maintenance-plan.json"
            if operation in protocol_report.RELOCATION_OPERATIONS
            and not pmr
            and not not_admitted
            else None
        ),
        "maintenance_plan_sha256": (
            "a" * 64
            if operation in protocol_report.RELOCATION_OPERATIONS
            and not pmr
            and not not_admitted
            else None
        ),
        "started_at_unix_ns": 100,
        "duration_ns": duration,
        "result_rows": expected_live_rows if operation != "open" else None,
        "dataset_version": 1,
        "fragments": fixture_fragments,
        "physical_rows": expected_live_rows,
        "physical_data_bytes": 1_000_000,
        "estimated_live_data_bytes": 1_000_000,
        "scan_byte_amplification": 1.0,
        "dataset_bytes": 2_000_000,
        "peak_rss_bytes": multiplier * 10,
        **totals,
        "actual_get_attempts": (
            totals["get_requests"] if sidecar["storage"] == "s3" else None
        ),
        "actual_head_attempts": (
            totals["head_requests"] if sidecar["storage"] == "s3" else None
        ),
        "actual_list_attempts": (
            totals["list_requests"] if sidecar["storage"] == "s3" else None
        ),
        "actual_put_attempts": (
            totals["put_requests"] if sidecar["storage"] == "s3" else None
        ),
        "actual_delete_attempts": (
            totals["delete_requests"] if sidecar["storage"] == "s3" else None
        ),
        "data_bytes": data["read_bytes"] + data["write_bytes"],
        "index_bytes": index["read_bytes"] + index["write_bytes"],
        "metadata_bytes": metadata["read_bytes"] + metadata["write_bytes"],
        "manifest_bytes": 4096,
        "placement_root_bytes": 1024 if format_name == "v23_logical" else None,
        "placement_delta_bytes": 1024 if format_name == "v23_logical" else None,
        "placement_delta_claimed_bytes": (
            1024 if format_name == "v23_logical" else None
        ),
        "w_epoch_bytes": 4096 if format_name == "v23_logical" else None,
        "coverage": 1.0 if index_kind != "none" else None,
        "recall": (
            1.0
            if operation == "index_take" and index_kind in {"scalar", "vector"}
            else None
        ),
        "admission": (
            False
            if pmr or not_admitted
            else True
            if is_commit or is_preflight
            else None
        ),
        "placement_maintenance_required": pmr,
        "pmr_reason": "projected_delta_bytes" if pmr else None,
        "pmr_projected_delta_bytes": protocol_report.B_FAST + 1 if pmr else None,
        "pmr_delta_limit_bytes": protocol_report.B_FAST if pmr else None,
        "pmr_projected_epoch_bytes": None,
        "pmr_epoch_limit_bytes": None,
        "pmr_generation_delta_bytes": None,
        "pmr_generation_epoch_bytes": None,
        "pmr_blocking_indices": None,
        "rows_inserted": None,
        "rows_updated": expected_live_rows if is_explicit_candidate else None,
        "rows_deleted": None,
        "compacted_data_bytes": 100
        if operation in protocol_report.RELOCATION_OPERATIONS
        and not pmr
        and not not_admitted
        else None,
        "index_storage_bytes_before": (
            200
            if index_kind != "none"
            and operation in protocol_report.RELOCATION_OPERATIONS
            else None
        ),
        "row_addresses_remapped": (
            0 if operation in protocol_report.RELOCATION_OPERATIONS else None
        ),
        "indices_remapped": (
            0 if operation in protocol_report.RELOCATION_OPERATIONS else None
        ),
        "index_coverage_reuse": (
            1.0
            if index_kind != "none"
            and operation in protocol_report.RELOCATION_OPERATIONS
            else None
        ),
        "layout_index_maintenance_ns": (
            multiplier if operation in protocol_report.RELOCATION_OPERATIONS else None
        ),
        "fragment_reuse_index_present": (
            format_name == "v22_no_stable"
            if is_fragment_reuse
            else False
            if operation in protocol_report.RELOCATION_OPERATIONS
            else None
        ),
        "explicit_locator_objects_written": (
            1
            if is_explicit_candidate
            else 0
            if operation == "random_delete_reclaim"
            else None
        ),
        "explicit_locator_bytes_written": (
            expected_live_rows
            if is_explicit_candidate
            else 0
            if operation == "random_delete_reclaim"
            else None
        ),
        "compaction_groups_planned": (
            1
            if operation in protocol_report.RELOCATION_OPERATIONS or is_preflight
            else None
        ),
        "compaction_groups_admitted": (
            0
            if not_admitted
            else 1
            if operation in protocol_report.RELOCATION_OPERATIONS or is_preflight
            else None
        ),
        "compaction_groups_not_admitted": (
            1
            if not_admitted
            else 0
            if operation in protocol_report.RELOCATION_OPERATIONS or is_preflight
            else None
        ),
        "state_digest": state_digest,
        "physical_order_digest": physical_order_digest,
        "io_by_path": io_by_path,
        "io_metrics_status": (
            "measured" if sidecar["storage"] == "s3" else "logical_only"
        ),
        "status": "ok",
        "error": None,
    }
    if expected is not None:
        record.update(
            {field: expected[field] for field in protocol_report.CORE_PROVENANCE_FIELDS}
        )
    run.validate_record(record)
    return record


def bind_exact_provenance(
    sidecar: dict[str, Any], records: list[dict[str, Any]]
) -> None:
    expected, issues = protocol_report.exact_record_provenance(sidecar, records)
    if issues:
        raise AssertionError(issues)
    for record in records:
        provenance = expected.get((record["pair_id"], record["format"]))
        if provenance is not None:
            record.update(provenance)
            run.validate_record(record)


def clear_index_write_metrics(record: dict[str, Any]) -> None:
    index_io = record["io_by_path"]["index"]
    write_bytes = index_io["write_bytes"]
    put_requests = index_io["put_requests"]
    index_io["write_bytes"] = 0
    index_io["put_requests"] = 0
    record["write_bytes"] -= write_bytes
    record["put_requests"] -= put_requests
    record["index_bytes"] -= write_bytes
    if record["actual_put_attempts"] is not None:
        record["actual_put_attempts"] -= put_requests
    run.validate_record(record)


def complete_records(sidecar: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pair_id in sorted(protocol_report.expected_complete_pair_ids(sidecar)):
        variant = next(
            (value for value in sidecar["variants"] if f"/{value}/" in pair_id), None
        )
        index_kind = variant if variant in {"scalar", "vector"} else "none"
        if index_kind == "none":
            index_kind = next(
                (
                    value
                    for value in ("scalar", "vector")
                    if f"/index-{value}/" in pair_id
                ),
                "none",
            )
        for format_name in run.FORMATS:
            records.append(
                make_record(
                    sidecar,
                    pair_id,
                    format_name,
                    index_kind=index_kind,
                    state_digest=("0" * 48 if pair_id.endswith("cold-scan") else None),
                    physical_order_digest=(
                        "0" * 48 if pair_id.endswith("cold-scan") else None
                    ),
                )
            )
    bind_exact_provenance(sidecar, records)
    return records


def append_reclaim_preflights(
    sidecar: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    case_name: str,
    not_admitted: bool,
) -> list[dict[str, Any]]:
    preflights: list[dict[str, Any]] = []
    profile = sidecar["matrix"]["profiles"][sidecar["profile"]]
    matrix_case = next(
        case
        for case in protocol.iter_matrix_cases(
            profile,
            set(sidecar["matrix"]["tracks"]["matrix"]["cases"]),
        )
        if case.name == case_name
    )
    step_index, step = next(
        (index, step)
        for index, step in enumerate(matrix_case.steps)
        if step.preflight_expected_admission is not None
    )
    label = (
        "default-reclaim-preflight"
        if step.operation == "random_delete_reclaim"
        else "default-compaction-preflight"
    )
    for repeat in range(profile["paired_repeats"]):
        pair_id = (
            f"{sidecar['run_id']}/matrix/{case_name}/repeat-{repeat:03d}/"
            f"step-{step_index:03d}/{label}"
        )
        preflight = make_record(
            sidecar,
            pair_id,
            "v23_logical",
            operation=protocol_report.DEFAULT_COMPACTION_PREFLIGHT,
            not_admitted=not_admitted,
        )
        records.append(preflight)
        preflights.append(preflight)
    bind_exact_provenance(sidecar, records)
    return preflights


class ProtocolReportTests(unittest.TestCase):
    def test_machine_report_is_bound_to_the_source_commit(self) -> None:
        sidecar = make_sidecar(["matrix"], matrix_case_names=[])
        result = protocol_report.analyze(sidecar, [], bootstrap_samples=101)

        self.assertEqual(result.machine["commit"], COMMIT)

    def test_sidecar_cannot_redefine_the_frozen_workload(self) -> None:
        sidecar = make_sidecar(["matrix"], matrix_case_names=["append/narrow16/take-1"])
        sidecar["matrix"]["profiles"]["smoke"]["rows"] = 1024
        canonical = json.dumps(
            sidecar["matrix"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        sidecar["matrix_canonical_json"] = canonical
        sidecar["matrix_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "frozen workload matrix"):
            protocol_report.validate_sidecar(sidecar)

    def test_release_sidecar_rejects_region_attestation_mismatch(self) -> None:
        sidecar = make_sidecar(["matrix"], profile="release")
        sidecar["storage_region_attestation"]["bucket_region"] = "us-west-2"

        with self.assertRaisesRegex(ValueError, "regions differ"):
            protocol_report.validate_sidecar(sidecar)

    def test_sidecar_dataset_root_is_derived_from_shard_identity(self) -> None:
        single = make_sidecar(["sustained"])
        single["dataset_root"] = "/tmp/data/."
        self.assertTrue(protocol_report.expected_record_provenance(single))
        single["dataset_root"] = "/tmp/arbitrary"
        with self.assertRaisesRegex(ValueError, "dataset_root"):
            protocol_report.expected_record_provenance(single)

        sharded = make_sidecar(["sustained"])
        sharded.update(
            {
                "base_dataset_root": "/tmp/data/",
                "dataset_root": "/tmp/data/shard-001-of-004",
                "shard_count": 4,
                "shard_index": 1,
                "shard_id": "shard-001-of-004",
            }
        )
        self.assertTrue(protocol_report.expected_record_provenance(sharded))
        sharded["dataset_root"] = "/tmp/data/another-shard"
        with self.assertRaisesRegex(ValueError, "dataset_root"):
            protocol_report.expected_record_provenance(sharded)

        wrong_ebs_scheme = make_sidecar(["sustained"])
        wrong_ebs_scheme.update(
            {
                "base_dataset_root": "s3://bucket/prefix",
                "dataset_root": "s3://bucket/prefix",
            }
        )
        with self.assertRaisesRegex(ValueError, "storage=ebs"):
            protocol_report.expected_record_provenance(wrong_ebs_scheme)

        smoke_s3 = make_sidecar(["sustained"])
        smoke_s3.update(
            {
                "storage": "s3",
                "base_dataset_root": "s3://bucket/prefix/",
                "dataset_root": "s3://bucket/prefix",
            }
        )
        self.assertTrue(protocol_report.expected_record_provenance(smoke_s3))

    def test_matrix_track_and_case_names_are_atomic(self) -> None:
        sidecar = make_sidecar(["sustained"])
        sidecar["matrix_case_names"] = ["append/narrow16/take-1"]
        with self.assertRaisesRegex(ValueError, "present together"):
            protocol_report.expected_record_provenance(sidecar)

    def test_release_selection_is_recomputed_from_the_frozen_shard(self) -> None:
        standalone = make_sidecar(
            ["matrix"],
            matrix_case_names=["append/narrow16/take-1"],
            profile="release",
        )
        standalone.update(
            {
                "base_dataset_root": "s3://bucket/prefix",
                "dataset_root": "s3://bucket/prefix",
            }
        )
        with self.assertRaisesRegex(ValueError, "exactly nine canonical shards"):
            protocol_report.expected_record_provenance(standalone)

        tracks, variants, cases = protocol_report.frozen_release_shard_contract(9, 4)
        self.assertEqual(
            tracks,
            (
                "matrix",
                "sustained",
                "adversarial_natural",
                "adversarial_aligned",
            ),
        )
        self.assertEqual(variants, ("bare", "scalar"))
        self.assertEqual(cases[0], "append/narrow16/take-1")
        sidecar = make_sidecar(
            list(tracks),
            variants=list(variants),
            matrix_case_names=list(cases),
            profile="release",
        )
        sidecar.update(
            {
                "base_dataset_root": "s3://bucket/prefix",
                "dataset_root": "s3://bucket/prefix/shard-004-of-009",
                "shard_count": 9,
                "shard_index": 4,
                "shard_id": "shard-004-of-009",
            }
        )
        protocol_report.validate_frozen_release_selection(sidecar)

        wrong_release_scheme = dict(sidecar)
        wrong_release_scheme.update(
            {
                "base_dataset_root": "/tmp/data",
                "dataset_root": "/tmp/data/shard-004-of-009",
            }
        )
        with self.assertRaisesRegex(ValueError, "storage=s3"):
            protocol_report.expected_record_provenance(wrong_release_scheme)

        for field in ("tracks", "variants", "matrix_case_names"):
            reordered = dict(sidecar)
            reordered[field] = list(reversed(sidecar[field]))
            with self.assertRaisesRegex(ValueError, f"release {field}"):
                protocol_report.expected_record_provenance(reordered)

        _, _, shard_zero_cases = protocol_report.frozen_release_shard_contract(9, 0)
        moved = dict(sidecar)
        moved["matrix_case_names"] = [shard_zero_cases[0], *cases[1:]]
        with self.assertRaisesRegex(ValueError, "release matrix_case_names"):
            protocol_report.expected_record_provenance(moved)

        wrong_seed = dict(sidecar)
        wrong_seed["seed"] = protocol.RELEASE_SEED + 1
        with self.assertRaisesRegex(ValueError, "canonical seed"):
            protocol_report.expected_record_provenance(wrong_seed)

        wrong_policy = dict(sidecar)
        wrong_policy["policy"] = json.loads(json.dumps(sidecar["policy"]))
        wrong_policy["policy"]["trigger"]["conditions"][0]["threshold"] = 0.1
        policy_canonical = json.dumps(
            wrong_policy["policy"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        wrong_policy["policy_canonical_json"] = policy_canonical
        wrong_policy["policy_sha256"] = hashlib.sha256(
            policy_canonical.encode()
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "repository default policy"):
            protocol_report.expected_record_provenance(wrong_policy)

    def test_maintenance_plan_hash_is_audited(self) -> None:
        sidecar = make_sidecar(["matrix"], matrix_case_names=["append/narrow16/take-1"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            plan = {
                "schema_version": 1,
                "suite": "stable_row_address_physical_maintenance_plan",
                "run_id": sidecar["run_id"],
                "commit": sidecar["commit"],
                "policy_sha256": sidecar["policy_sha256"],
                "source_format": "v22_no_stable",
                "source_dataset_uri": "/tmp/source",
                "source_dataset_version": 1,
                "schema_kind": "narrow_16b",
                "expected_rows": 10,
                "target_rows_per_fragment": 10,
                "execution_target_rows_per_fragment": 10,
                "target_file_size_bytes": 134_217_728,
                "max_source_fragments_per_group": 256,
                "fragment_count": 2,
                "groups": [
                    {
                        "start_ordinal": 0,
                        "end_ordinal": 2,
                        "source_live_rows": 10,
                        "source_physical_rows": 10,
                        "source_physical_data_bytes": 160,
                        "source_live_data_bytes": 160,
                        "expected_output_fragments": 1,
                    }
                ],
                "expected_output_live_rows": [10],
                "expected_output_fragment_count": 1,
            }
            path.write_text(json.dumps(plan) + "\n", encoding="utf-8")
            canonical = json.dumps(
                plan, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
            record = complete_records(sidecar)[0]
            record["maintenance_plan_path"] = str(path)
            record["maintenance_plan_sha256"] = hashlib.sha256(
                canonical.encode()
            ).hexdigest()
            self.assertEqual(
                protocol_report.audit_maintenance_plans(sidecar, [record]), []
            )
            plan["commit"] = "2" * 40
            path.write_text(json.dumps(plan) + "\n", encoding="utf-8")
            self.assertTrue(protocol_report.audit_maintenance_plans(sidecar, [record]))

    def test_latency_ci_uses_p95_pair_ratio(self) -> None:
        candidate = [100] * 9 + [200]
        baseline = [100] * 10
        median = protocol_report.paired_ratio_ci(
            candidate, baseline, samples=101, seed=7
        )
        p95 = protocol_report.paired_ratio_ci(
            candidate, baseline, samples=101, seed=7, statistic="p95"
        )
        self.assertEqual(median[0], 1.0)
        self.assertEqual(p95[0], 2.0)

    def test_complete_matrix_passes_and_missing_pair_is_incomplete(self) -> None:
        sidecar = make_sidecar(["matrix"], matrix_case_names=["append/narrow16/take-1"])
        records = complete_records(sidecar)
        result = protocol_report.analyze(sidecar, records, bootstrap_samples=101)
        self.assertEqual(result.verdict, "PASS")
        self.assertTrue(result.machine["gates"])
        incomplete = protocol_report.analyze(
            sidecar, records[:-1], bootstrap_samples=101
        )
        self.assertEqual(incomplete.verdict, "INCOMPLETE")

    def test_placement_delta_measurement_must_match_independent_claim(self) -> None:
        sidecar = make_sidecar(["matrix"], matrix_case_names=["append/narrow16/take-1"])
        records = complete_records(sidecar)
        candidate = next(
            record
            for record in records
            if record["format"] == "v23_logical" and record["operation"] == "append"
        )
        candidate["placement_delta_claimed_bytes"] += 1
        result = protocol_report.analyze(sidecar, records, bootstrap_samples=101)
        self.assertEqual(result.verdict, "FAIL")
        self.assertTrue(
            any(
                "does not match independent claim" in failure
                for failure in result.machine["failures"]
            )
        )

        records = complete_records(sidecar)
        legacy = next(
            record
            for record in records
            if record["format"] == "v22_no_stable" and record["operation"] == "append"
        )
        legacy["placement_delta_claimed_bytes"] = 1
        result = protocol_report.analyze(sidecar, records, bootstrap_samples=101)
        self.assertEqual(result.verdict, "FAIL")
        self.assertTrue(
            any(
                "legacy format claimed v2.3 placement Delta" in failure
                for failure in result.machine["failures"]
            )
        )

    def test_default_fast_records_enforce_delta_and_epoch_budgets(self) -> None:
        sidecar = make_sidecar(["matrix"], matrix_case_names=["append/narrow16/take-1"])
        records = complete_records(sidecar)
        candidate = next(
            record
            for record in records
            if record["format"] == "v23_logical" and record["operation"] == "append"
        )
        candidate["placement_delta_bytes"] = protocol_report.B_FAST + 1
        candidate["placement_delta_claimed_bytes"] = protocol_report.B_FAST + 1
        result = protocol_report.analyze(sidecar, records, bootstrap_samples=101)
        self.assertEqual(result.verdict, "FAIL")
        self.assertTrue(
            any(
                "default-fast placement Delta" in value
                for value in result.machine["failures"]
            )
        )

        records = complete_records(sidecar)
        candidate = next(
            record
            for record in records
            if record["format"] == "v23_logical" and record["operation"] == "append"
        )
        candidate["w_epoch_bytes"] = protocol_report.W_FAST + 1
        result = protocol_report.analyze(sidecar, records, bootstrap_samples=101)
        self.assertEqual(result.verdict, "FAIL")
        self.assertTrue(
            any("default-fast W_epoch" in value for value in result.machine["failures"])
        )

    def test_explicit_maintenance_diagnostics_are_exempt_from_fast_budgets(
        self,
    ) -> None:
        sidecar = make_sidecar(
            ["matrix"], matrix_case_names=["bounded-recluster-8/narrow16"]
        )
        records = complete_records(sidecar)
        profile = sidecar["matrix"]["profiles"]["smoke"]
        for repeat in range(profile["paired_repeats"]):
            candidate = make_record(
                sidecar,
                f"run-1/matrix/bounded-recluster-8/narrow16/"
                f"repeat-{repeat:03d}/step-001/recluster",
                "v23_logical",
                operation="recluster",
                index_kind="scalar",
            )
            candidate["placement_delta_bytes"] = protocol_report.B_FAST + 1
            candidate["placement_delta_claimed_bytes"] = protocol_report.B_FAST + 1
            candidate["w_epoch_bytes"] = protocol_report.W_FAST + 1
            records.append(candidate)
        bind_exact_provenance(sidecar, records)
        result = protocol_report.analyze(sidecar, records, bootstrap_samples=101)
        self.assertEqual(result.verdict, "PASS")

    def test_typed_pmr_diagnostics_are_relationally_validated(self) -> None:
        sidecar = make_sidecar(["matrix"], matrix_case_names=["append/narrow16/take-1"])

        projected_delta = make_record(
            sidecar,
            "run-1/typed-pmr/projected-delta",
            "v23_logical",
            operation="update",
            pmr=True,
        )
        self.assertEqual(
            protocol_report.audit_row_address_record_contract(sidecar, projected_delta),
            [],
        )
        projected_delta["pmr_projected_delta_bytes"] = protocol_report.B_FAST
        self.assertTrue(
            any(
                "requires projected > limit" in failure
                for failure in protocol_report.audit_row_address_record_contract(
                    sidecar, projected_delta
                )
            )
        )

        projected_epoch = make_record(
            sidecar,
            "run-1/typed-pmr/projected-epoch",
            "v23_logical",
            operation="update",
            pmr=True,
        )
        projected_epoch.update(
            {
                "pmr_reason": "projected_epoch_bytes",
                "pmr_projected_delta_bytes": None,
                "pmr_delta_limit_bytes": None,
                "pmr_projected_epoch_bytes": protocol_report.W_FAST + 1,
                "pmr_epoch_limit_bytes": protocol_report.W_FAST,
            }
        )
        self.assertEqual(
            protocol_report.audit_row_address_record_contract(sidecar, projected_epoch),
            [],
        )

        generation_blocked = make_record(
            sidecar,
            "run-1/typed-pmr/generation-blocked",
            "v23_logical",
            operation="update",
            pmr=True,
        )
        generation_blocked.update(
            {
                "pmr_reason": "index_generation_blocked",
                "pmr_projected_delta_bytes": protocol_report.B_FAST + 1,
                "pmr_delta_limit_bytes": protocol_report.B_FAST,
                "pmr_projected_epoch_bytes": protocol_report.W_FAST,
                "pmr_epoch_limit_bytes": protocol_report.W_FAST,
                "pmr_generation_delta_bytes": 1024,
                "pmr_generation_epoch_bytes": 4096,
                "pmr_blocking_indices": [
                    {
                        "index_id": "00000000-0000-0000-0000-000000000001",
                        "index_name": "value_idx",
                        "field_ids": [0],
                        "oldest_generation": 2,
                        "region_bytes": 1024,
                        "blocked_transaction_start": 2,
                        "blocked_transaction_end": 7,
                    }
                ],
            }
        )
        self.assertEqual(
            protocol_report.audit_row_address_record_contract(
                sidecar, generation_blocked
            ),
            [],
        )
        generation_blocked["pmr_blocking_indices"][0]["blocked_transaction_start"] = 8
        self.assertTrue(
            any(
                "invalid blocked transaction range" in failure
                for failure in protocol_report.audit_row_address_record_contract(
                    sidecar, generation_blocked
                )
            )
        )

        structural = make_record(
            sidecar,
            "run-1/typed-pmr/structural",
            "v23_logical",
            operation="update",
            pmr=True,
        )
        structural.update(
            {
                "pmr_reason": "extent_fanout",
                "pmr_projected_delta_bytes": None,
                "pmr_delta_limit_bytes": None,
            }
        )
        self.assertEqual(
            protocol_report.audit_row_address_record_contract(sidecar, structural),
            [],
        )
        structural["pmr_reason"] = "unknown"
        self.assertTrue(
            any(
                "unknown or missing PMR reason" in failure
                for failure in protocol_report.audit_row_address_record_contract(
                    sidecar, structural
                )
            )
        )

        non_pmr = make_record(
            sidecar,
            "run-1/typed-pmr/non-pmr",
            "v23_logical",
            operation="update",
        )
        non_pmr["pmr_reason"] = "extent_fanout"
        self.assertTrue(
            any(
                "non-PMR record populated PMR diagnostics" in failure
                for failure in protocol_report.audit_row_address_record_contract(
                    sidecar, non_pmr
                )
            )
        )

    def test_state_mismatch_and_latency_regression_fail(self) -> None:
        sidecar = make_sidecar(["matrix"], matrix_case_names=["append/narrow16/take-1"])
        records = complete_records(sidecar)
        scan = next(
            record
            for record in records
            if record["operation"] == "scan" and record["format"] == "v23_logical"
        )
        scan["state_digest"] = "f" * 48
        for record in records:
            if record["format"] == "v23_logical":
                record["duration_ns"] = 200
        result = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=True
        )
        self.assertEqual(result.verdict, "FAIL")
        self.assertTrue(result.machine["failures"])
        self.assertTrue(any(not gate["passed"] for gate in result.machine["gates"]))

    def test_sustained_prefixes_are_gated(self) -> None:
        sidecar = make_sidecar(["sustained"], variants=["bare"])
        records = complete_records(sidecar)
        profile = sidecar["matrix"]["profiles"]["smoke"]
        for repeat in range(profile["paired_repeats"]):
            for update_round in (2, 5, 8):
                scan_pair_id = (
                    f"run-1/sustained/bare/repeat-{repeat:03d}/"
                    f"step-{update_round:03d}/cold-scan"
                )
                next(
                    record
                    for record in records
                    if record["pair_id"] == scan_pair_id
                    and record["format"] == "v22_no_stable"
                )["fragments"] = 8
                pair_id = (
                    f"run-1/sustained/bare/repeat-{repeat:03d}/"
                    f"round-{update_round:03d}/policy-maintenance"
                )
                for format_name in run.FORMATS:
                    records.append(
                        make_record(
                            sidecar,
                            pair_id,
                            format_name,
                            operation="default_compaction",
                        )
                    )
                post_step = profile["repeated_update_rounds"] + update_round
                for label in ("cold-open", "cold-scan", "cold-take"):
                    post_pair_id = (
                        f"run-1/sustained/bare/repeat-{repeat:03d}/"
                        f"step-{post_step:03d}/{label}"
                    )
                    for format_name in run.FORMATS:
                        records.append(
                            make_record(
                                sidecar,
                                post_pair_id,
                                format_name,
                                state_digest=(
                                    "0" * 48 if label == "cold-scan" else None
                                ),
                            )
                        )
        bind_exact_provenance(sidecar, records)
        result = protocol_report.analyze(sidecar, records, bootstrap_samples=101)
        self.assertEqual(result.verdict, "PASS")
        prefix_gates = [
            gate
            for gate in result.machine["gates"]
            if gate["scope"].endswith("/prefix")
        ]
        self.assertTrue(prefix_gates)
        prefixes = result.machine["sustained_prefixes"]["variants"]["bare"]
        self.assertEqual(len(prefixes), 3)
        sample = prefixes[0]["repeats"][0]["formats"]["v23_logical"]
        self.assertEqual(
            {
                "metadata_read_bytes",
                "metadata_write_bytes",
                "row_address_resident_bytes",
                "row_address_epoch_write_bytes",
            }
            - sample.keys(),
            set(),
        )

    def test_adversarial_indexed_full_epoch_has_strict_net_gate(self) -> None:
        sidecar = make_sidecar(["adversarial_natural"], variants=["scalar"])
        records = complete_records(sidecar)
        result = protocol_report.analyze(sidecar, records, bootstrap_samples=101)
        self.assertEqual(result.verdict, "PASS")
        epoch_gates = [
            gate
            for gate in result.machine["gates"]
            if gate["scope"] == "adversarial_natural/scalar/full-epoch"
        ]
        self.assertEqual(len(epoch_gates), 20)
        self.assertTrue(all(gate["passed"] for gate in epoch_gates))
        observations = result.machine["adversarial_natural"]["variants"]["scalar"]
        self.assertEqual(
            len(observations[0]["prefixes"]),
            sidecar["matrix"]["profiles"]["smoke"]["repeated_update_rounds"],
        )
        self.assertEqual(set(observations[0]["terminal_debt"]), set(run.FORMATS))

    def test_adversarial_policy_boundary_evidence_is_fail_closed(self) -> None:
        sidecar = make_sidecar(["adversarial_natural"], variants=["bare"])
        records = complete_records(sidecar)
        pre_scan_id = "run-1/adversarial_natural/bare/repeat-000/step-000/cold-scan"
        for record in records:
            if record["pair_id"] == pre_scan_id:
                record["fragments"] = 8
        missing = protocol_report.analyze(sidecar, records, bootstrap_samples=101)
        self.assertEqual(missing.verdict, "INCOMPLETE")
        self.assertTrue(
            any(
                "frozen physical policy triggered" in issue
                for issue in missing.machine["issues"]
            )
        )

        for format_name in run.FORMATS:
            pair_id = (
                "run-1/adversarial_natural/bare/repeat-000/round-000/"
                f"natural-maintenance/{format_name}"
            )
            records.append(
                make_record(
                    sidecar,
                    pair_id,
                    format_name,
                    operation="default_compaction",
                )
            )
        bind_exact_provenance(sidecar, records)
        complete = protocol_report.analyze(sidecar, records, bootstrap_samples=101)
        self.assertEqual(complete.verdict, "PASS")
        observed = complete.machine["adversarial_natural"]["variants"]["bare"][0]
        self.assertEqual(
            observed["natural_maintenance_rounds"],
            {name: [0] for name in run.FORMATS},
        )

    def test_untriggered_dynamic_maintenance_is_rejected(self) -> None:
        sidecar = make_sidecar(["adversarial_natural"], variants=["bare"])
        records = complete_records(sidecar)
        for format_name in run.FORMATS:
            pair_id = (
                "run-1/adversarial_natural/bare/repeat-000/round-000/"
                f"natural-maintenance/{format_name}"
            )
            records.append(
                make_record(
                    sidecar,
                    pair_id,
                    format_name,
                    operation="default_compaction",
                )
            )
        result = protocol_report.analyze(sidecar, records, bootstrap_samples=101)
        self.assertEqual(result.verdict, "FAIL")
        self.assertTrue(
            any(
                "unexpected dynamic invocation" in failure
                for failure in result.machine["failures"]
            )
        )

    def test_failed_pmr_maintenance_does_not_authorize_retry(self) -> None:
        sidecar = make_sidecar(["adversarial_natural"], variants=["bare"])
        records = complete_records(sidecar)
        round_prefix = "run-1/adversarial_natural/bare/repeat-000/round-000"
        candidate = next(
            record
            for record in records
            if record["pair_id"] == f"{round_prefix}/update-attempt"
            and record["format"] == "v23_logical"
        )
        candidate["placement_maintenance_required"] = True
        candidate["admission"] = False
        maintenance = make_record(
            sidecar,
            f"{round_prefix}/pmr-maintenance",
            "v23_logical",
            operation="normalize_placement",
        )
        maintenance["status"] = "error"
        maintenance["error"] = "maintenance failed"
        retry = make_record(
            sidecar,
            f"{round_prefix}/update-retry",
            "v23_logical",
            operation="update",
        )
        records.extend((maintenance, retry))

        _, failures = protocol_report.audit_record_provenance(sidecar, records)
        self.assertTrue(
            any(
                f"{round_prefix}/update-retry/v23_logical: unexpected dynamic invocation"
                == failure
                for failure in failures
            )
        )

    def test_successful_pmr_maintenance_authorizes_retry(self) -> None:
        sidecar = make_sidecar(["adversarial_natural"], variants=["bare"])
        records = complete_records(sidecar)
        round_prefix = "run-1/adversarial_natural/bare/repeat-000/round-000"
        candidate = next(
            record
            for record in records
            if record["pair_id"] == f"{round_prefix}/update-attempt"
            and record["format"] == "v23_logical"
        )
        candidate["placement_maintenance_required"] = True
        candidate["admission"] = False
        records.extend(
            (
                make_record(
                    sidecar,
                    f"{round_prefix}/pmr-maintenance",
                    "v23_logical",
                    operation="normalize_placement",
                ),
                make_record(
                    sidecar,
                    f"{round_prefix}/update-retry",
                    "v23_logical",
                    operation="update",
                ),
            )
        )
        bind_exact_provenance(sidecar, records)

        issues, failures = protocol_report.audit_record_provenance(sidecar, records)
        self.assertEqual(issues, [])
        self.assertEqual(failures, [])

    def test_dynamic_maintenance_order_is_replayed_from_its_scope(self) -> None:
        sidecar = make_sidecar(["adversarial_natural"], variants=["bare"])
        records = complete_records(sidecar)
        round_prefix = "run-1/adversarial_natural/bare/repeat-000/round-000"
        candidate = next(
            record
            for record in records
            if record["pair_id"] == f"{round_prefix}/update-attempt"
            and record["format"] == "v23_logical"
        )
        candidate["placement_maintenance_required"] = True
        candidate["admission"] = False
        maintenance = make_record(
            sidecar,
            f"{round_prefix}/pmr-maintenance",
            "v23_logical",
            operation="normalize_placement",
        )
        retry = make_record(
            sidecar,
            f"{round_prefix}/update-retry",
            "v23_logical",
            operation="update",
        )
        records.extend((maintenance, retry))
        bind_exact_provenance(sidecar, records)

        expected_order = run.dynamic_format_order(
            0, maintenance["pair_id"].removeprefix(f"{sidecar['run_id']}/")
        )
        expected_index = expected_order.index("v23_logical")
        self.assertEqual(maintenance["order_index"], expected_index)
        self.assertEqual(retry["order_index"], expected_index)

        retry["order_index"] = (expected_index + 1) % len(run.FORMATS)
        _, failures = protocol_report.audit_record_provenance(sidecar, records)
        self.assertTrue(
            any(
                retry["pair_id"] in failure and "order_index" in failure
                for failure in failures
            )
        )

    def test_order_index_is_replayed_from_the_frozen_phase_order(self) -> None:
        sidecar = make_sidecar(["matrix"], matrix_case_names=["append/narrow16/take-1"])
        records = complete_records(sidecar)
        self.assertTrue(any(record["order_index"] != 0 for record in records))
        for record in records:
            record["order_index"] = 0
        result = protocol_report.analyze(sidecar, records, bootstrap_samples=101)
        self.assertEqual(result.verdict, "FAIL")
        self.assertTrue(
            any("order_index" in failure for failure in result.machine["failures"])
        )

    def test_no_stable_relocation_keeps_five_percent_baseline(self) -> None:
        case_name = "compact-8-to-1/narrow16"
        sidecar = make_sidecar(["matrix"], matrix_case_names=[case_name])
        result = protocol_report.analyze(
            sidecar,
            complete_records(sidecar),
            bootstrap_samples=101,
            enforce_gates=True,
        )
        gates = [
            gate
            for gate in result.machine["gates"]
            if gate["scope"].endswith("/default_compaction")
            and gate["metric"] == "data_write_bytes"
        ]
        by_baseline = {gate["baseline"]: gate for gate in gates}
        self.assertEqual(by_baseline["v22_no_stable"]["threshold"], 1.05)
        self.assertFalse(by_baseline["v22_no_stable"]["strict"])
        self.assertEqual(by_baseline["v22_stable"]["threshold"], 1.0)
        self.assertFalse(by_baseline["v22_stable"]["strict"])

    def test_relocation_latency_and_throughput_strictly_beat_stable(self) -> None:
        case_name = "compact-8-to-1/narrow16"
        sidecar = make_sidecar(["matrix"], matrix_case_names=[case_name])
        records = complete_records(sidecar)
        for record in records:
            if record["operation"] != "default_compaction":
                continue
            record["duration_ns"] = {
                "v22_no_stable": 120,
                "v22_stable": 110,
                "v23_logical": 110,
            }[record["format"]]
        result = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=True
        )
        self.assertEqual(result.verdict, "FAIL")
        stable_gates = [
            gate
            for gate in result.machine["gates"]
            if gate["scope"].endswith("/default_compaction")
            and gate["baseline"] == "v22_stable"
        ]
        strict_by_metric = {gate["metric"]: gate["strict"] for gate in stable_gates}
        self.assertTrue(strict_by_metric["latency_p95"])
        self.assertTrue(strict_by_metric["throughput"])
        self.assertFalse(strict_by_metric["data_write_bytes"])
        self.assertTrue(
            any(
                not gate["passed"]
                for gate in stable_gates
                if gate["metric"] in {"latency_p95", "throughput"}
            )
        )

    def test_vector_index_take_requires_non_regressing_recall(self) -> None:
        sidecar = make_sidecar(["adversarial_natural"], variants=["vector"])
        records = complete_records(sidecar)
        self.assertEqual(
            protocol_report.analyze(sidecar, records, bootstrap_samples=101).verdict,
            "PASS",
        )

        candidate_take = next(
            record
            for record in records
            if record["operation"] == "index_take" and record["format"] == "v23_logical"
        )
        candidate_take["recall"] = None
        self.assertEqual(
            protocol_report.analyze(sidecar, records, bootstrap_samples=101).verdict,
            "INCOMPLETE",
        )
        candidate_take["recall"] = 0.0
        self.assertEqual(
            protocol_report.analyze(sidecar, records, bootstrap_samples=101).verdict,
            "FAIL",
        )

    def test_effective_index_coverage_must_be_complete(self) -> None:
        sidecar = make_sidecar(["adversarial_natural"], variants=["scalar"])
        records = complete_records(sidecar)
        optimized = next(
            record
            for record in records
            if record["operation"] == "index_optimize"
            and record["format"] == "v23_logical"
        )
        optimized["coverage"] = 0.5
        result = protocol_report.analyze(sidecar, records, bootstrap_samples=101)
        self.assertEqual(result.verdict, "FAIL")
        self.assertTrue(
            any(
                "effective index coverage" in failure
                for failure in result.machine["failures"]
            )
        )

    def test_vector_repeated_value_update_allows_index_optimize_noop(self) -> None:
        sidecar = make_sidecar(["adversarial_natural"], variants=["vector"])
        records = complete_records(sidecar)
        optimized = [
            record for record in records if record["operation"] == "index_optimize"
        ]
        self.assertTrue(optimized)
        for record in optimized:
            clear_index_write_metrics(record)

        result = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=False
        )

        self.assertEqual(result.verdict, "PASS")
        self.assertFalse(
            any(
                "unified tracker missed index writes" in failure
                for failure in result.machine["failures"]
            )
        )

    def test_vector_append_catch_up_still_requires_index_writes(self) -> None:
        sidecar = make_sidecar(["matrix"], matrix_case_names=["vector-index"])
        records = complete_records(sidecar)
        optimized = [
            record for record in records if record["operation"] == "index_optimize"
        ]
        self.assertTrue(optimized)
        for record in optimized:
            clear_index_write_metrics(record)

        result = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=False
        )

        self.assertEqual(result.verdict, "FAIL")
        self.assertTrue(
            any(
                "unified tracker missed index writes" in failure
                for failure in result.machine["failures"]
            )
        )

    def test_adversarial_aligned_gates_are_diagnostic_only(self) -> None:
        sidecar = make_sidecar(["matrix"], matrix_case_names=["append/narrow16/take-1"])
        records = complete_records(sidecar)
        diagnostic = protocol_report.Gate(
            track="adversarial_aligned",
            scope="adversarial_aligned/scalar/trigger-0",
            metric="latency",
            baseline="v22_no_stable",
            samples=3,
            ratio=2.0,
            ci_low=2.0,
            ci_high=2.0,
            direction="upper",
            threshold=1.0,
            strict=True,
            passed=False,
        )
        with mock.patch.object(
            protocol_report,
            "add_aligned_relocation_gates",
            return_value=[diagnostic],
        ):
            result = protocol_report.analyze(
                sidecar, records, bootstrap_samples=101, enforce_gates=True
            )

        aligned = [
            gate
            for gate in result.machine["gates"]
            if gate["track"] == "adversarial_aligned"
        ]
        self.assertEqual(result.verdict, "PASS")
        self.assertEqual(len(aligned), 1)
        self.assertFalse(aligned[0]["passed"])
        self.assertFalse(aligned[0]["aggregate_release_gate"])
        self.assertIn("diagnostic-only", result.markdown)
        self.assertIn("do not affect the aggregate release verdict", result.markdown)

    def test_machine_report_is_strict_json(self) -> None:
        sidecar = make_sidecar(["matrix"], matrix_case_names=["append/narrow16/take-1"])
        result = protocol_report.analyze(
            sidecar, complete_records(sidecar), bootstrap_samples=11
        )
        encoded = json.dumps(
            result.machine, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        self.assertEqual(hashlib.sha256(encoded.encode()).digest_size, 32)

    def test_random_delete_reclaim_preflight_accepts_profile_result(self) -> None:
        case_name = "delete-random-50/narrow16/take-1"
        sidecar = make_sidecar(["matrix"], matrix_case_names=[case_name])
        records = complete_records(sidecar)
        append_reclaim_preflights(
            sidecar, records, case_name=case_name, not_admitted=False
        )
        result = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=False
        )
        self.assertEqual(result.verdict, "PASS")

    def test_one_percent_random_delete_is_preflighted_and_reclaimed(self) -> None:
        case_name = "delete-random-1/narrow16/take-1"
        sidecar = make_sidecar(["matrix"], matrix_case_names=[case_name])
        records = complete_records(sidecar)
        preflights = append_reclaim_preflights(
            sidecar, records, case_name=case_name, not_admitted=False
        )
        self.assertTrue(
            all(
                preflight["pair_id"].endswith("/step-002/default-compaction-preflight")
                for preflight in preflights
            )
        )
        self.assertTrue(
            any(record["operation"] == "default_compaction" for record in records)
        )
        result = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=True
        )
        self.assertEqual(result.verdict, "PASS")

    def test_frozen_record_provenance_rejects_rows_and_path_tampering(self) -> None:
        case_name = "delete-random-50/narrow16/take-1"
        sidecar = make_sidecar(["matrix"], matrix_case_names=[case_name])
        records = complete_records(sidecar)
        append_reclaim_preflights(
            sidecar, records, case_name=case_name, not_admitted=False
        )
        baseline = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=False
        )
        self.assertEqual(baseline.verdict, "PASS")
        reclaim = next(
            record
            for record in records
            if record["operation"] == "random_delete_reclaim"
            and record["format"] == "v23_logical"
        )
        self.assertEqual(reclaim["rows"], 65_536)
        self.assertEqual(reclaim["implementation_path"], "explicit_repack")

        reclaim["rows"] = 12345
        wrong_rows = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=False
        )
        self.assertEqual(wrong_rows.verdict, "FAIL")
        self.assertTrue(
            any("'rows':" in failure for failure in wrong_rows.machine["failures"])
        )

        reclaim["rows"] = 65_536
        reclaim["implementation_path"] = "native_dataset_api"
        wrong_path = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=False
        )
        self.assertEqual(wrong_path.verdict, "FAIL")
        self.assertTrue(
            any(
                "'implementation_path':" in failure
                for failure in wrong_path.machine["failures"]
            )
        )

    def test_frozen_provenance_covers_fixture_and_repeated_track_state(self) -> None:
        sidecar = make_sidecar(
            [
                "matrix",
                "sustained",
                "adversarial_natural",
                "adversarial_aligned",
            ],
            variants=["scalar"],
            matrix_case_names=["append/narrow16/take-1"],
        )
        expected = protocol_report.expected_record_provenance(sidecar)
        fixture_clone = expected[
            (
                "run-1/fixtures/narrow16/rows-65536/rows-per-fragment-8192/"
                "index-scalar/fixture_clone",
                "v23_logical",
            )
        ]
        self.assertEqual(fixture_clone["take_count"], 1)
        self.assertEqual(fixture_clone["index_kind"], "none")

        sustained = expected[
            (
                "run-1/sustained/scalar/repeat-000/round-003/update",
                "v23_logical",
            )
        ]
        self.assertEqual(sustained["mutation_count"], 655)
        self.assertEqual(sustained["selection_step"], 0)
        self.assertEqual(
            sustained["implementation_path"], "exact_selection_matched_merge"
        )

        natural = expected[
            (
                "run-1/adversarial_natural/scalar/repeat-000/round-003/"
                "natural-maintenance/v22_stable",
                "v22_stable",
            )
        ]
        self.assertEqual(natural["mutation_count"], 1)
        self.assertEqual(natural["selection"], "range")

        aligned = expected[
            (
                "run-1/adversarial_aligned/scalar/repeat-000/round-003/"
                "forced-baseline-maintenance/v22_no_stable",
                "v22_no_stable",
            )
        ]
        self.assertEqual(aligned["mutation_count"], 655)
        self.assertEqual(aligned["selection_step"], 3)
        self.assertEqual(aligned["selection"], "uniform_without_replacement")
        self.assertEqual(aligned["implementation_path"], "default_compaction")

    def test_random_delete_reclaim_preflight_rejects_inverse_result(self) -> None:
        case_name = "delete-random-50/narrow16/take-1"
        sidecar = make_sidecar(["matrix"], matrix_case_names=[case_name])
        records = complete_records(sidecar)
        append_reclaim_preflights(
            sidecar, records, case_name=case_name, not_admitted=True
        )
        result = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=False
        )
        self.assertEqual(result.verdict, "FAIL")

    def test_random_delete_reclaim_preflight_missing_is_incomplete(self) -> None:
        case_name = "delete-random-50/narrow16/take-1"
        sidecar = make_sidecar(["matrix"], matrix_case_names=[case_name])
        records = complete_records(sidecar)
        append_reclaim_preflights(
            sidecar, records, case_name=case_name, not_admitted=False
        )
        missing = protocol_report.analyze(sidecar, records[:-1], bootstrap_samples=101)
        self.assertEqual(missing.verdict, "INCOMPLETE")

    def test_random_delete_reclaim_preflight_version_side_effect_fails(self) -> None:
        case_name = "delete-random-50/narrow16/take-1"
        sidecar = make_sidecar(["matrix"], matrix_case_names=[case_name])
        records = complete_records(sidecar)
        preflights = append_reclaim_preflights(
            sidecar, records, case_name=case_name, not_admitted=False
        )
        preflights[0]["dataset_version"] += 1
        result = protocol_report.analyze(sidecar, records, bootstrap_samples=101)
        self.assertEqual(result.verdict, "FAIL")
        self.assertTrue(
            any(
                "changed dataset version" in failure
                for failure in result.machine["failures"]
            )
        )

    def test_random_delete_reclaim_preflight_requires_plan_only_path(self) -> None:
        case_name = "delete-random-50/narrow16/take-1"
        sidecar = make_sidecar(["matrix"], matrix_case_names=[case_name])
        records = complete_records(sidecar)
        preflights = append_reclaim_preflights(
            sidecar, records, case_name=case_name, not_admitted=False
        )
        preflights[0]["implementation_path"] = "default_compaction"
        result = protocol_report.analyze(sidecar, records, bootstrap_samples=101)
        self.assertEqual(result.verdict, "FAIL")
        self.assertTrue(
            any(
                "implementation path" in failure
                for failure in result.machine["failures"]
            )
        )

    def test_random_delete_reclaim_preflight_write_side_effect_fails(self) -> None:
        case_name = "delete-random-50/narrow16/take-1"
        sidecar = make_sidecar(["matrix"], matrix_case_names=[case_name])
        records = complete_records(sidecar)
        preflights = append_reclaim_preflights(
            sidecar, records, case_name=case_name, not_admitted=False
        )
        metadata = preflights[0]["io_by_path"]["metadata"]
        metadata["put_requests"] = 1
        metadata["write_bytes"] = 10
        preflights[0]["put_requests"] = 1
        preflights[0]["write_bytes"] = 10
        preflights[0]["metadata_bytes"] += 10
        run.validate_record(preflights[0])
        result = protocol_report.analyze(sidecar, records, bootstrap_samples=101)
        self.assertEqual(result.verdict, "FAIL")
        self.assertTrue(
            any("wrote objects" in failure for failure in result.machine["failures"])
        )

    def test_indexed_relocation_has_zero_remap_and_10x_2x_gates(self) -> None:
        case_name = "indexed-compact-8-to-1/scalar"
        sidecar = make_sidecar(["matrix"], matrix_case_names=[case_name])
        records = complete_records(sidecar)
        for record in records:
            if record["operation"] != "default_compaction":
                continue
            if record["format"] == "v23_logical":
                record["duration_ns"] = 5
                record["layout_index_maintenance_ns"] = 5
            else:
                record["duration_ns"] = 100
                record["layout_index_maintenance_ns"] = 100
        result = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=True
        )
        relocation_gates = [
            gate
            for gate in result.machine["gates"]
            if gate["scope"].endswith("/indexed-relocation")
        ]
        self.assertEqual(len(relocation_gates), 4)
        self.assertTrue(all(gate["passed"] for gate in relocation_gates))

        candidate = next(
            record
            for record in records
            if record["operation"] == "default_compaction"
            and record["format"] == "v23_logical"
        )
        candidate["index_storage_bytes_before"] = 99
        candidate["io_by_path"]["index"]["delete_requests"] = 1
        candidate["delete_requests"] += 1
        failures: list[str] = []
        protocol_report.add_indexed_relocation_contract_gates(
            sidecar,
            records,
            bootstrap_samples=101,
            issues=[],
            failures=failures,
        )
        self.assertTrue(
            any("accessed index objects" in failure for failure in failures)
        )

    def test_indexed_relocation_large_index_gates_are_conditional(self) -> None:
        case_name = "indexed-compact-8-to-1/vector"
        sidecar = make_sidecar(["matrix"], matrix_case_names=[case_name])
        records = complete_records(sidecar)
        for record in records:
            if record["operation"] != "default_compaction":
                continue
            record["index_storage_bytes_before"] = 200
            record["compacted_data_bytes"] = 100
            if record["format"] == "v23_logical":
                record["duration_ns"] = 5
                record["layout_index_maintenance_ns"] = 5
            else:
                record["duration_ns"] = 100
                record["layout_index_maintenance_ns"] = 100
            if (
                "/repeat-001/" in record["pair_id"]
                and record["format"] == "v22_stable"
            ):
                record["index_storage_bytes_before"] = 99

        result = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=True
        )

        self.assertEqual(result.verdict, "PASS")
        self.assertFalse(
            any(
                gate["scope"].endswith("/indexed-relocation")
                for gate in result.machine["gates"]
            )
        )
        self.assertFalse(
            any(
                "below compacted data bytes" in failure
                for failure in result.machine["failures"]
            )
        )

    def test_explicit_repack_and_post_probes_are_diagnostic(self) -> None:
        case_name = "delete-random-50/narrow16/take-1"
        sidecar = make_sidecar(["matrix"], matrix_case_names=[case_name])
        records = complete_records(sidecar)
        append_reclaim_preflights(
            sidecar, records, case_name=case_name, not_admitted=False
        )
        for record in records:
            if record["format"] != "v23_logical":
                continue
            if record["operation"] == "random_delete_reclaim" or (
                f"/matrix/{case_name}/" in record["pair_id"]
                and "/step-003/" in record["pair_id"]
            ):
                record["duration_ns"] = 10_000
        result = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=True
        )
        self.assertEqual(result.verdict, "PASS")
        self.assertTrue(result.machine["explicit_maintenance"]["cases"][case_name])
        self.assertIn("## Explicit maintenance public cost", result.markdown)
        self.assertIn("### Paired maintenance and lookup cost", result.markdown)
        self.assertFalse(
            any(
                "/random_delete_reclaim" in gate["scope"]
                or "/step-003/" in gate["scope"]
                for gate in result.machine["gates"]
            )
        )

        for record in records:
            if record["format"] == "v23_logical" and record["pair_id"].endswith(
                "/step-002/cold-take"
            ):
                record["duration_ns"] = 10_000
        gated = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=True
        )
        self.assertEqual(gated.verdict, "FAIL")

    def test_indexed_repack_gates_reuse_lookup_quality_and_read_io(self) -> None:
        case_name = "indexed-repack-random-delete-50/scalar"
        sidecar = make_sidecar(["matrix"], matrix_case_names=[case_name])
        records = complete_records(sidecar)
        append_reclaim_preflights(
            sidecar, records, case_name=case_name, not_admitted=False
        )
        result = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=True
        )
        self.assertEqual(result.verdict, "PASS")
        lookup_gates = [
            gate
            for gate in result.machine["gates"]
            if gate["scope"].endswith("/indexed-repack-lookup")
        ]
        self.assertTrue(lookup_gates)
        self.assertTrue(all(gate["passed"] for gate in lookup_gates))
        observation = result.machine["explicit_maintenance"]["cases"][case_name][0]
        candidate_lookup = observation["post_cold_lookup"]["cold-index-take"][
            "v23_logical"
        ]
        self.assertIn("index_read_bytes", candidate_lookup)
        self.assertIn("metadata_get_requests", candidate_lookup)

        candidate_maintenance = next(
            record
            for record in records
            if record["operation"] == "random_delete_reclaim"
            and record["format"] == "v23_logical"
        )
        candidate_maintenance["index_coverage_reuse"] = 0.5
        failed_reuse = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=False
        )
        self.assertEqual(failed_reuse.verdict, "FAIL")
        candidate_maintenance["index_coverage_reuse"] = 1.0

        candidate_lookup_record = next(
            record
            for record in records
            if record["operation"] == "index_take"
            and record["format"] == "v23_logical"
            and f"/matrix/{case_name}/" in record["pair_id"]
            and "/step-003/" in record["pair_id"]
        )
        candidate_lookup_record["io_by_path"]["index"]["read_bytes"] = 1_000
        failed_io = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=True
        )
        self.assertEqual(failed_io.verdict, "FAIL")
        self.assertTrue(
            any(
                not gate["passed"] and gate["metric"] == "index_read_bytes"
                for gate in failed_io.machine["gates"]
            )
        )

    def test_bounded_recluster_is_candidate_only_public_diagnostic(self) -> None:
        case_name = "bounded-recluster-8/narrow16"
        sidecar = make_sidecar(["matrix"], matrix_case_names=[case_name])
        records = complete_records(sidecar)
        pair_id = f"run-1/matrix/{case_name}/repeat-000/step-001/recluster"
        expected = protocol_report.expected_record_provenance(sidecar)
        self.assertIn((pair_id, "v23_logical"), expected)
        self.assertNotIn((pair_id, "v22_no_stable"), expected)
        self.assertNotIn((pair_id, "v22_stable"), expected)
        missing = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=True
        )
        self.assertEqual(missing.verdict, "INCOMPLETE")

        profile = sidecar["matrix"]["profiles"]["smoke"]
        for repeat in range(profile["paired_repeats"]):
            records.append(
                make_record(
                    sidecar,
                    f"run-1/matrix/{case_name}/repeat-{repeat:03d}/step-001/recluster",
                    "v23_logical",
                    operation="recluster",
                    index_kind="scalar",
                )
            )
        bind_exact_provenance(sidecar, records)
        result = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=True
        )
        self.assertEqual(result.verdict, "PASS")
        observations = result.machine["explicit_maintenance"]["cases"][case_name]
        self.assertEqual(len(observations), profile["paired_repeats"])
        self.assertTrue(
            all(set(item["maintenance"]) == {"v23_logical"} for item in observations)
        )

    def test_default_bounded_clustering_is_paired_and_order_equivalent(self) -> None:
        case_name = "bounded-default-clustering-8/narrow16"
        sidecar = make_sidecar(["matrix"], matrix_case_names=[case_name])
        records = complete_records(sidecar)
        result = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=True
        )
        self.assertEqual(result.verdict, "PASS")
        pair_id = f"run-1/matrix/{case_name}/repeat-000/step-001/bounded_recluster"
        expected = protocol_report.expected_record_provenance(sidecar)
        self.assertEqual(
            {
                format_name
                for current_pair, format_name in expected
                if current_pair == pair_id
            },
            set(run.FORMATS),
        )

        scan_pair = f"run-1/matrix/{case_name}/repeat-000/step-002/cold-scan"
        next(
            record
            for record in records
            if record["pair_id"] == scan_pair and record["format"] == "v23_logical"
        )["physical_order_digest"] = "f" * 48
        failed = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=False
        )
        self.assertEqual(failed.verdict, "FAIL")
        self.assertTrue(
            any(
                "physical row order differs" in failure
                for failure in failed.machine["failures"]
            )
        )

    def test_placement_bytes_are_history_independent(self) -> None:
        sidecar = make_sidecar(
            ["matrix"],
            matrix_case_names=[
                "compact-64-to-1/narrow16",
                "repeated-compaction-10/narrow16",
            ],
        )
        records = complete_records(sidecar)
        result = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=False
        )
        self.assertEqual(result.verdict, "PASS")
        self.assertTrue(result.machine["placement_history_independence"]["comparisons"])

        repeated = next(
            record
            for record in records
            if record["pair_id"].endswith(
                "/repeated-compaction-10/narrow16/repeat-000/step-010/default_compaction"
            )
            and record["format"] == "v23_logical"
        )
        repeated["placement_root_bytes"] = 2_000
        failed = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=False
        )
        self.assertEqual(failed.verdict, "FAIL")
        self.assertTrue(
            any(
                "history-dependent" in failure for failure in failed.machine["failures"]
            )
        )

    def test_skewed_fixture_audit_checks_exact_fragment_counts(self) -> None:
        tracks, variants, cases = protocol_report.frozen_release_shard_contract(9, 0)
        sidecar = make_sidecar(
            list(tracks),
            variants=list(variants),
            matrix_case_names=list(cases),
            profile="release",
        )
        sidecar.update(
            {
                "dataset_root": "s3://release-bucket/data/shard-000-of-009",
                "shard_count": 9,
                "shard_index": 0,
                "shard_id": "shard-000-of-009",
            }
        )
        case_name = "compact-100000-skew-to-1/narrow16"
        self.assertIn(case_name, sidecar["matrix_case_names"])
        records = complete_records(sidecar)
        result = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=False
        )
        self.assertEqual(result.verdict, "PASS")
        fixtures = result.machine["skewed_packed_run_fixtures"]["fixtures"]
        self.assertEqual(fixtures[0]["source_fragments"], 100_000)

        append_fixture = next(
            record
            for record in records
            if "/segmented-layout-" in record["pair_id"]
            and record["pair_id"].endswith("/append-001")
            and record["format"] == "v23_logical"
        )
        append_fixture["fragments"] -= 1
        failed = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=False
        )
        self.assertEqual(failed.verdict, "FAIL")
        self.assertTrue(
            any(
                "fragments, expected" in failure
                for failure in failed.machine["failures"]
            )
        )

    def test_take_gates_zero_additional_placement_metadata_requests(self) -> None:
        case_name = "append/narrow16/take-1"
        sidecar = make_sidecar(["matrix"], matrix_case_names=[case_name])
        records = complete_records(sidecar)
        candidate_take = next(
            record
            for record in records
            if record["operation"] == "take" and record["format"] == "v23_logical"
        )
        candidate_take["io_by_path"]["metadata"]["get_requests"] += 1
        candidate_take["get_requests"] += 1
        failed = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=True
        )
        self.assertEqual(failed.verdict, "FAIL")
        self.assertTrue(
            any(
                not gate["passed"] and gate["metric"] == "metadata_get_requests"
                for gate in failed.machine["gates"]
            )
        )

    def test_fragment_reuse_materializes_real_comparison_paths(self) -> None:
        case_name = "fragment-reuse-8-to-1/scalar"
        sidecar = make_sidecar(["matrix"], matrix_case_names=[case_name])
        records = complete_records(sidecar)
        for record in records:
            if record["operation"] != "default_compaction":
                continue
            if record["format"] == "v23_logical":
                record["duration_ns"] = 5
                record["layout_index_maintenance_ns"] = 5
            else:
                record["duration_ns"] = 100
                record["layout_index_maintenance_ns"] = 100
        result = protocol_report.analyze(
            sidecar, records, bootstrap_samples=101, enforce_gates=True
        )
        self.assertEqual(result.verdict, "PASS")
        reuse_records = [
            record for record in records if record["operation"] == "default_compaction"
        ]
        self.assertEqual(
            {
                record["format"]: record["implementation_path"]
                for record in reuse_records[:3]
            },
            {
                "v22_no_stable": "deferred_fragment_reuse_compaction",
                "v22_stable": "inline_index_remap_compaction",
                "v23_logical": "stable_logical_zero_remap_compaction",
            },
        )

        broken = complete_records(sidecar)
        next(
            record
            for record in broken
            if record["operation"] == "default_compaction"
            and record["format"] == "v22_no_stable"
        )["fragment_reuse_index_present"] = False
        failed = protocol_report.analyze(
            sidecar, broken, bootstrap_samples=101, enforce_gates=False
        )
        self.assertEqual(failed.verdict, "FAIL")
        self.assertTrue(
            any(
                "system-index state" in failure
                for failure in failed.machine["failures"]
            )
        )


if __name__ == "__main__":
    unittest.main()
