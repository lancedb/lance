#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run  # noqa: E402


COMMIT = "1" * 40
POLICY_SHA256 = "2" * 64


def valid_record() -> dict[str, object]:
    return {
        "schema_version": 1,
        "suite": "stable_row_address_e2e",
        "run_id": "run-1",
        "pair_id": "run-1/round-000/open",
        "commit": COMMIT,
        "host": "host-1",
        "seed": 7,
        "policy_sha256": POLICY_SHA256,
        "policy_version": 1,
        "mode": "smoke",
        "format": "v23_logical",
        "storage": "ebs",
        "operation": "open",
        "timing_scope": "dataset_open_and_contract_validation",
        "round": 0,
        "order_index": 0,
        "dataset_uri": "/tmp/example.lance",
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
        "duration_ns": 50,
        "result_rows": None,
        "dataset_version": 1,
        "fragments": 8,
        "physical_rows": 1024,
        "physical_data_bytes": 4096,
        "estimated_live_data_bytes": 4096,
        "scan_byte_amplification": 1.0,
        "dataset_bytes": 4096,
        "peak_rss_bytes": 8192,
        "get_requests": None,
        "head_requests": None,
        "list_requests": None,
        "put_requests": None,
        "delete_requests": None,
        "actual_get_attempts": None,
        "actual_head_attempts": None,
        "actual_list_attempts": None,
        "actual_put_attempts": None,
        "actual_delete_attempts": None,
        "read_bytes": None,
        "write_bytes": None,
        "data_bytes": None,
        "index_bytes": None,
        "metadata_bytes": None,
        "manifest_bytes": None,
        "placement_root_bytes": None,
        "placement_delta_bytes": None,
        "w_epoch_bytes": None,
        "coverage": None,
        "recall": None,
        "admission": None,
        "placement_maintenance_required": None,
        "rows_inserted": None,
        "rows_updated": None,
        "rows_deleted": None,
        "compacted_data_bytes": None,
        "index_storage_bytes_before": None,
        "row_addresses_remapped": None,
        "indices_remapped": None,
        "index_coverage_reuse": None,
        "layout_index_maintenance_ns": None,
        "compaction_groups_planned": None,
        "compaction_groups_admitted": None,
        "compaction_groups_not_admitted": None,
        "state_digest": None,
        "io_by_path": None,
        "io_metrics_status": "not_instrumented",
        "status": "ok",
        "error": None,
    }


class RunTests(unittest.TestCase):
    def test_format_order_is_balanced_and_deterministic(self) -> None:
        orders = [run.format_order(round_index, 0) for round_index in range(3)]
        self.assertEqual(
            orders,
            [
                ("v22_no_stable", "v22_stable", "v23_logical"),
                ("v22_stable", "v23_logical", "v22_no_stable"),
                ("v23_logical", "v22_no_stable", "v22_stable"),
            ],
        )

    def test_policy_hash_uses_canonical_json(self) -> None:
        policy = json.loads(run.DEFAULT_POLICY.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            first.write_text(json.dumps(policy), encoding="utf-8")
            second.write_text(json.dumps(policy, indent=4), encoding="utf-8")
            _, first_bytes, first_hash = run.canonical_policy(first)
            _, second_bytes, second_hash = run.canonical_policy(second)
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first_hash, second_hash)

    def test_policy_requires_total_fragment_order_and_any_trigger(self) -> None:
        policy = json.loads(run.DEFAULT_POLICY.read_text(encoding="utf-8"))
        policy["trigger"]["semantics"] = "all"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(policy), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "semantics must be any"):
                run.canonical_policy(path)

    def test_validate_record_is_strict(self) -> None:
        record = valid_record()
        self.assertIs(run.validate_record(record), record)
        record["unknown"] = 1
        with self.assertRaisesRegex(run.RecordValidationError, "unknown"):
            run.validate_record(record)

    def test_uninstrumented_record_cannot_claim_request_metrics(self) -> None:
        record = valid_record()
        record["actual_get_attempts"] = 1
        with self.assertRaisesRegex(
            run.RecordValidationError, "must not publish request metrics"
        ):
            run.validate_record(record)

    def test_worker_output_requires_one_record(self) -> None:
        record = valid_record()
        encoded = json.dumps(record)
        parsed = run.parse_worker_stdout(encoded + "\n", {"commit": COMMIT})
        self.assertEqual(parsed, record)
        with self.assertRaisesRegex(run.RecordValidationError, "exactly one"):
            run.parse_worker_stdout(encoded + "\n" + encoded + "\n", {})

    def test_clean_checkout_validation_rejects_dirty_status(self) -> None:
        self.assertEqual(run._validate_clean_checkout(COMMIT + "\n", ""), COMMIT)
        with self.assertRaisesRegex(RuntimeError, "clean checkout"):
            run._validate_clean_checkout(COMMIT, "?? result.jsonl\n")

    def test_dataset_uri_keeps_s3_under_requested_prefix(self) -> None:
        uri = run.dataset_uri("s3://bucket/prefix/", "run-1", 2, "v23_logical")
        self.assertEqual(uri, "s3://bucket/prefix/run-1/round-002/v23_logical.lance")

    def test_build_harness_passes_command_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "stable_row_address_e2e"
            executable.touch()
            artifact = {
                "reason": "compiler-artifact",
                "target": {"name": "stable_row_address_e2e"},
                "executable": str(executable),
            }

            def fake_run(command: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
                self.assertEqual(command[0:2], ("cargo", "build"))
                self.assertIn("cwd", kwargs)
                return SimpleNamespace(returncode=0, stdout=json.dumps(artifact) + "\n")

            with mock.patch.object(run.subprocess, "run", side_effect=fake_run):
                self.assertEqual(run.build_harness(), executable.resolve())

    def test_worker_command_maps_internal_operation_to_clap_name(self) -> None:
        command = run._worker_command(
            Path("/tmp/worker"),
            uri="/tmp/data",
            format_name="v23_logical",
            storage="ebs",
            operation="merge_insert",
            mode="smoke",
            run_id="run",
            pair_id="pair",
            round_index=0,
            order_index=0,
            rows=10,
            rows_per_fragment=10,
            take_count=1,
            seed=1,
            commit=COMMIT,
            host="host",
            policy_sha256=POLICY_SHA256,
            policy_version=1,
        )
        self.assertEqual(command[command.index("--operation") + 1], "merge-insert")

    def test_run_sidecar_is_atomic_and_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.jsonl"
            payload = {"run_id": "run-1", "policy": {"schema_version": 1}}
            sidecar = run.write_run_sidecar(output, payload)
            self.assertEqual(json.loads(sidecar.read_text(encoding="utf-8")), payload)
            self.assertEqual(list(Path(directory).glob(".*.tmp-*")), [])
            with self.assertRaises(FileExistsError):
                run.write_run_sidecar(output, payload)


if __name__ == "__main__":
    unittest.main()
