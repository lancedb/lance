#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import report  # noqa: E402
import run  # noqa: E402


COMMIT = "1" * 40
POLICY_SHA256 = "2" * 64


def make_record(
    operation: str,
    round_index: int,
    format_name: str,
    order_index: int,
    duration_ns: int,
) -> report.SourceRecord:
    logical_requests = 12 if format_name == "v22_stable" else 10
    io_bytes = 1_200 if format_name == "v22_stable" else 1_000
    peak_rss_bytes = 9_600 if format_name == "v22_stable" else 8_000
    empty_path_metrics = {
        "get_requests": 0,
        "head_requests": 0,
        "list_requests": 0,
        "put_requests": 0,
        "delete_requests": 0,
        "read_bytes": 0,
        "write_bytes": 0,
    }
    value = {
        "schema_version": 1,
        "suite": "stable_row_address_e2e",
        "run_id": "run-1",
        "pair_id": f"run-1/round-{round_index:03d}/{operation}",
        "commit": COMMIT,
        "host": "host-1",
        "seed": 7,
        "policy_sha256": POLICY_SHA256,
        "policy_version": 1,
        "mode": "smoke",
        "format": format_name,
        "storage": "ebs",
        "operation": operation,
        "timing_scope": run.TIMING_SCOPES[operation],
        "round": round_index,
        "order_index": order_index,
        "dataset_uri": f"/tmp/run-1/round-{round_index:03d}/{format_name}.lance",
        "rows": 1024,
        "rows_per_fragment": 128,
        "take_count": 16,
        "expected_rows": 1024,
        "mutation_count": 1,
        "id_start": 0,
        "step": 0,
        "selection_step": 0,
        "match_percent": 50,
        "schema_kind": "narrow_16b",
        "index_kind": "none",
        "selection": "range",
        "implementation_path": "native_dataset_api",
        "maintenance_plan_path": None,
        "maintenance_plan_sha256": None,
        "started_at_unix_ns": 100,
        "duration_ns": duration_ns,
        "result_rows": None if operation == "open" else 1024,
        "dataset_version": 1,
        "fragments": 8,
        "physical_rows": 1024,
        "physical_data_bytes": 4096,
        "estimated_live_data_bytes": 4096,
        "scan_byte_amplification": 1.0,
        "dataset_bytes": 4096,
        "peak_rss_bytes": peak_rss_bytes,
        "get_requests": logical_requests,
        "head_requests": 0,
        "list_requests": 0,
        "put_requests": 0,
        "delete_requests": 0,
        "actual_get_attempts": None,
        "actual_head_attempts": None,
        "actual_list_attempts": None,
        "actual_put_attempts": None,
        "actual_delete_attempts": None,
        "read_bytes": io_bytes,
        "write_bytes": 0,
        "data_bytes": io_bytes,
        "index_bytes": 0,
        "metadata_bytes": 0,
        "manifest_bytes": None,
        "placement_root_bytes": None,
        "placement_delta_bytes": None,
        "placement_delta_claimed_bytes": None,
        "w_epoch_bytes": None,
        "coverage": None,
        "recall": None,
        "admission": None,
        "placement_maintenance_required": None,
        "pmr_reason": None,
        "pmr_projected_delta_bytes": None,
        "pmr_delta_limit_bytes": None,
        "pmr_projected_epoch_bytes": None,
        "pmr_epoch_limit_bytes": None,
        "pmr_generation_delta_bytes": None,
        "pmr_generation_epoch_bytes": None,
        "pmr_blocking_indices": None,
        "rows_inserted": None,
        "rows_updated": None,
        "rows_deleted": None,
        "compacted_data_bytes": None,
        "index_storage_bytes_before": None,
        "row_addresses_remapped": None,
        "indices_remapped": None,
        "index_coverage_reuse": None,
        "layout_index_maintenance_ns": None,
        "fragment_reuse_index_present": None,
        "explicit_locator_objects_written": None,
        "explicit_locator_bytes_written": None,
        "compaction_groups_planned": None,
        "compaction_groups_admitted": None,
        "compaction_groups_not_admitted": None,
        "state_digest": None,
        "physical_order_digest": None,
        "io_by_path": {
            "data": {
                **empty_path_metrics,
                "get_requests": logical_requests,
                "read_bytes": io_bytes,
            },
            "index": dict(empty_path_metrics),
            "metadata": dict(empty_path_metrics),
            "other": dict(empty_path_metrics),
        },
        "io_metrics_status": "logical_only",
        "status": "ok",
        "error": None,
    }
    run.validate_record(value)
    return report.SourceRecord("memory.jsonl", 1, value)


def complete_records(
    candidate_duration: int = 100,
    no_stable_duration: int = 100,
    stable_duration: int = 120,
) -> list[report.SourceRecord]:
    durations = {
        "v22_no_stable": no_stable_duration,
        "v22_stable": stable_duration,
        "v23_logical": candidate_duration,
    }
    records = []
    for round_index in range(3):
        for operation_index, operation in enumerate(run.OPERATIONS):
            for order_index, format_name in enumerate(
                run.format_order(round_index, operation_index)
            ):
                records.append(
                    make_record(
                        operation,
                        round_index,
                        format_name,
                        order_index,
                        durations[format_name],
                    )
                )
    return records


class ReportTests(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic(self) -> None:
        first = report.paired_bootstrap_ratio(
            [100, 200, 300], [100, 100, 100], samples=10_000, seed=9
        )
        second = report.paired_bootstrap_ratio(
            [100, 200, 300], [100, 100, 100], samples=10_000, seed=9
        )
        self.assertEqual(first, second)
        self.assertEqual(first[0], 2.0)

    def test_complete_balanced_run_passes_enforced_gates(self) -> None:
        result = report.analyze(
            complete_records(),
            mode="smoke",
            enforce_gates=True,
            bootstrap_samples=101,
        )
        self.assertEqual(result.verdict, "PASS")
        self.assertIn("101", result.markdown)
        self.assertEqual(result.markdown.count("| PASS |"), 32)

    def test_missing_format_is_incomplete(self) -> None:
        records = complete_records()
        records.pop()
        result = report.analyze(
            records, mode="smoke", enforce_gates=True, bootstrap_samples=101
        )
        self.assertEqual(result.verdict, "INCOMPLETE")
        self.assertIn("expected one record per format", result.markdown)

    def test_worker_error_fails_the_report(self) -> None:
        records = complete_records()
        failing = records[0].value
        failing["status"] = "error"
        failing["error"] = "write failed"
        result = report.analyze(records, mode="smoke", bootstrap_samples=101)
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("write failed", result.markdown)

    def test_smoke_mode_reports_but_does_not_enforce_observed_gate(self) -> None:
        records = complete_records(
            candidate_duration=120, no_stable_duration=100, stable_duration=100
        )
        smoke = report.analyze(
            records, mode="smoke", enforce_gates=False, bootstrap_samples=101
        )
        enforced = report.analyze(
            records, mode="smoke", enforce_gates=True, bootstrap_samples=101
        )
        self.assertEqual(smoke.verdict, "PASS")
        self.assertIn("diagnostic because gates were not enforced", smoke.markdown)
        self.assertEqual(enforced.verdict, "FAIL")

    def test_release_mode_rejects_three_round_smoke_evidence(self) -> None:
        result = report.analyze(
            complete_records(), mode="release", bootstrap_samples=101
        )
        self.assertEqual(result.verdict, "INCOMPLETE")
        self.assertIn(
            "record mode smoke does not match report mode release", result.markdown
        )

    def test_run_sidecar_binds_full_canonical_policy(self) -> None:
        policy, policy_bytes, policy_hash = run.canonical_policy(run.DEFAULT_POLICY)
        sidecar = {
            "schema_version": 1,
            "suite": "stable_row_address_e2e",
            "run_id": "run-1",
            "created_at_utc": "20260712T000000.000000Z",
            "commit": COMMIT,
            "host": "host-1",
            "seed": 7,
            "mode": "smoke",
            "storage": "ebs",
            "formats": list(run.FORMATS),
            "operations": list(run.OPERATIONS),
            "rounds": 3,
            "rows": 1024,
            "rows_per_fragment": 128,
            "take_count": 16,
            "dataset_root": "/tmp/data",
            "output_jsonl": "/tmp/results.jsonl",
            "executable": "/tmp/stable_row_address_e2e",
            "data_retention": "preserve",
            "take_ids_root": "/tmp/take-ids",
            "policy_version": 1,
            "policy_sha256": policy_hash,
            "policy_canonical_json": policy_bytes.decode("utf-8"),
            "policy": policy,
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.jsonl.run.json"
            self.assertIs(report.validate_run_sidecar(sidecar, source), sidecar)
            sidecar["policy_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "does not match canonical policy"):
                report.validate_run_sidecar(sidecar, source)


if __name__ == "__main__":
    unittest.main()
