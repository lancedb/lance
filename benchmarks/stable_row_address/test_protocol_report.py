#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import unittest
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
) -> dict[str, Any]:
    matrix, matrix_canonical, matrix_sha256 = protocol.load_matrix(
        protocol.DEFAULT_MATRIX
    )
    policy, policy_bytes, policy_sha256 = run.canonical_policy(run.DEFAULT_POLICY)
    return {
        "schema_version": 1,
        "suite": "stable_row_address_design_protocol",
        "run_id": "run-1",
        "created_at_utc": "20260712T000000.000000Z",
        "commit": COMMIT,
        "source_provenance": "clean-committed-source",
        "host": "host-1",
        "seed": 7,
        "profile": "smoke",
        "tracks": tracks,
        "variants": variants or ["bare"],
        "matrix_case_names": matrix_case_names or [],
        "storage": "ebs",
        "dataset_root": "/tmp/data",
        "base_dataset_root": "/tmp/data",
        "shard_count": 1,
        "shard_index": 0,
        "shard_id": "shard-000-of-001",
        "shard_strategy": "schema_and_fragment_layout_fixture_locality",
        "output_jsonl": "/tmp/results.jsonl",
        "executable": "/tmp/stable_row_address_e2e",
        "data_retention": "preserve",
        "storage_scope": "bounded_smoke",
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
    pmr: bool = False,
    not_admitted: bool = False,
) -> dict[str, Any]:
    operation = operation or operation_from_pair(pair_id)
    repeat_match = re.search(r"/repeat-(\d{3})/", pair_id)
    repeat = int(repeat_match.group(1)) if repeat_match else 0
    update_match = re.search(r"/(?:round|step)-(\d{3})/", pair_id)
    step = int(update_match.group(1)) if update_match else 0
    multiplier = {"v22_no_stable": 100, "v22_stable": 110, "v23_logical": 90}[
        format_name
    ]
    duration = duration if duration is not None else multiplier
    writes = operation in protocol_report.COMMIT_OPERATIONS and not pmr and not not_admitted
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
        "result_rows": 65_536 if operation != "open" else None,
        "dataset_version": 1,
        "fragments": 1,
        "physical_rows": 65_536,
        "physical_data_bytes": 1_000_000,
        "estimated_live_data_bytes": 1_000_000,
        "scan_byte_amplification": 1.0,
        "dataset_bytes": 2_000_000,
        "peak_rss_bytes": multiplier * 10,
        **totals,
        "actual_get_attempts": None,
        "actual_head_attempts": None,
        "actual_list_attempts": None,
        "actual_put_attempts": None,
        "actual_delete_attempts": None,
        "data_bytes": data["read_bytes"] + data["write_bytes"],
        "index_bytes": index["read_bytes"] + index["write_bytes"],
        "metadata_bytes": metadata["read_bytes"] + metadata["write_bytes"],
        "manifest_bytes": 4096,
        "placement_root_bytes": 1024 if format_name == "v23_logical" else None,
        "placement_delta_bytes": 1024 if format_name == "v23_logical" else None,
        "w_epoch_bytes": 4096 if format_name == "v23_logical" else None,
        "coverage": 1.0 if index_kind != "none" else None,
        "recall": (
            1.0
            if operation == "index_take" and index_kind in {"scalar", "vector"}
            else None
        ),
        "admission": False if pmr or not_admitted else True if is_commit else None,
        "placement_maintenance_required": pmr,
        "rows_inserted": None,
        "rows_updated": None,
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
        "compaction_groups_planned": (
            1 if operation in protocol_report.RELOCATION_OPERATIONS else None
        ),
        "compaction_groups_admitted": (
            0
            if not_admitted
            else 1
            if operation in protocol_report.RELOCATION_OPERATIONS
            else None
        ),
        "compaction_groups_not_admitted": (
            1
            if not_admitted
            else 0
            if operation in protocol_report.RELOCATION_OPERATIONS
            else None
        ),
        "state_digest": state_digest,
        "io_by_path": io_by_path,
        "io_metrics_status": "logical_only",
        "status": "ok",
        "error": None,
    }
    run.validate_record(record)
    return record


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
                )
            )
    return records


class ProtocolReportTests(unittest.TestCase):
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
            self.assertTrue(
                protocol_report.audit_maintenance_plans(sidecar, [record])
            )

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
        result = protocol_report.analyze(sidecar, records, bootstrap_samples=101)
        self.assertEqual(result.verdict, "PASS")
        prefix_gates = [
            gate
            for gate in result.machine["gates"]
            if gate["scope"].endswith("/prefix")
        ]
        self.assertTrue(prefix_gates)

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
        self.assertEqual(
            set(observations[0]["terminal_debt"]), set(run.FORMATS)
        )

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
            any("frozen physical policy triggered" in issue for issue in missing.machine["issues"])
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
        complete = protocol_report.analyze(sidecar, records, bootstrap_samples=101)
        self.assertEqual(complete.verdict, "PASS")
        observed = complete.machine["adversarial_natural"]["variants"]["bare"][0]
        self.assertEqual(
            observed["natural_maintenance_rounds"],
            {name: [0] for name in run.FORMATS},
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
            any("effective index coverage" in failure for failure in result.machine["failures"])
        )

    def test_machine_report_is_strict_json(self) -> None:
        sidecar = make_sidecar(["matrix"], matrix_case_names=["append/narrow16/take-1"])
        result = protocol_report.analyze(
            sidecar, complete_records(sidecar), bootstrap_samples=11
        )
        encoded = json.dumps(
            result.machine, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        self.assertEqual(hashlib.sha256(encoded.encode()).digest_size, 32)

    def test_random_delete_reclaim_requires_not_admitted_preflight(self) -> None:
        case_name = "delete-random-50/narrow16/take-1"
        sidecar = make_sidecar(["matrix"], matrix_case_names=[case_name])
        records = complete_records(sidecar)
        for repeat in range(sidecar["matrix"]["profiles"]["smoke"]["paired_repeats"]):
            pair_id = (
                f"run-1/matrix/{case_name}/repeat-{repeat:03d}/"
                "step-002/default-reclaim-preflight"
            )
            records.append(
                make_record(
                    sidecar,
                    pair_id,
                    "v23_logical",
                    operation="default_compaction",
                    not_admitted=True,
                )
            )
        result = protocol_report.analyze(sidecar, records, bootstrap_samples=101)
        self.assertEqual(result.verdict, "PASS")
        missing = protocol_report.analyze(sidecar, records[:-1], bootstrap_samples=101)
        self.assertEqual(missing.verdict, "INCOMPLETE")

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


if __name__ == "__main__":
    unittest.main()
