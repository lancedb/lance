#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import protocol  # noqa: E402
import run  # noqa: E402


class ProtocolTests(unittest.TestCase):
    def test_release_profile_freezes_design_matrix(self) -> None:
        matrix, canonical, digest = protocol.load_matrix(protocol.DEFAULT_MATRIX)
        release = matrix["profiles"]["release"]
        self.assertEqual(release["rows"], 100_000_000)
        self.assertEqual(release["logical_fragment_counts"], [100, 10_000, 100_000])
        self.assertEqual(release["take_counts"], [1, 32, 1024, 10_000])
        self.assertEqual(release["delete_percentages"], [1, 50, 90])
        self.assertEqual(release["repeated_update_rounds"], 100)
        self.assertEqual(release["hot_set_rows"], 1_000_000)
        self.assertEqual(
            matrix["profiles"]["smoke"]["random_delete_reclaim_admission"],
            "must_admit",
        )
        self.assertEqual(
            release["random_delete_reclaim_admission"], "must_not_admit"
        )
        self.assertEqual(len(digest), 64)
        self.assertIn('"schema_version":1', canonical)

    def test_random_update_uses_exact_selection_driver(self) -> None:
        matrix, _, _ = protocol.load_matrix(protocol.DEFAULT_MATRIX)
        cases = list(
            protocol.iter_matrix_cases(matrix["profiles"]["smoke"], {"update_random"})
        )
        updates = [
            step for case in cases for step in case.steps if step.operation == "update"
        ]
        self.assertTrue(updates)
        self.assertTrue(all(step.selection == "random" for step in updates))
        self.assertTrue(
            all(step.update_driver == "exact-matched-merge" for step in updates)
        )

    def test_random_delete_reclaim_and_pack_chains_are_frozen(self) -> None:
        matrix, _, _ = protocol.load_matrix(protocol.DEFAULT_MATRIX)
        profile = matrix["profiles"]["smoke"]
        delete_cases = {
            case.name: case
            for case in protocol.iter_matrix_cases(profile, {"delete_random"})
        }
        self.assertNotIn(
            "random_delete_reclaim",
            [
                step.operation
                for step in delete_cases["delete-random-1/narrow16/take-1"].steps
            ],
        )
        self.assertEqual(
            delete_cases["delete-random-50/narrow16/take-1"].steps[-1].operation,
            "random_delete_reclaim",
        )
        chain_cases = {
            case.name: case
            for case in protocol.iter_matrix_cases(
                profile, {"pack_random_mutation_chain"}
            )
        }
        self.assertEqual(
            [
                step.operation
                for step in chain_cases["pack-random-update-1/narrow16"].steps
            ],
            ["create", "default_compaction", "update"],
        )
        self.assertEqual(
            chain_cases["pack-random-delete-90/vector"].steps[-1].operation,
            "random_delete_reclaim",
        )

    def test_random_delete_reclaim_provenance_is_format_specific(self) -> None:
        reclaim = protocol.Step("random_delete_reclaim", expected_rows=50)
        runner = object.__new__(protocol.ProtocolRunner)
        runner.run_id = "run-1"
        runner.commit = "1" * 40
        runner.host = "host"
        runner.seed = 7
        runner.policy_sha256 = "2" * 64
        runner.policy_version = 1
        runner.mode = "smoke"
        runner.storage = "ebs"
        runner.rows = 100
        runner.rows_per_fragment = 10
        runner.take_count = 1
        expected_paths = {
            "v22_no_stable": "same_postcondition_default_compaction",
            "v22_stable": "same_postcondition_default_compaction",
            "v23_logical": "explicit_repack",
        }
        for order_index, (format_name, implementation_path) in enumerate(
            expected_paths.items()
        ):
            expected = runner._expected(
                reclaim,
                pair_id="run-1/reclaim",
                round=0,
                order_index=order_index,
                dataset_uri=f"/tmp/{format_name}.lance",
                format=format_name,
                maintenance_plan_path=None,
                maintenance_plan_sha256=None,
            )
            self.assertEqual(
                expected["implementation_path"], implementation_path
            )
            record = dict.fromkeys(run.RECORD_FIELDS)
            record.update(expected)
            record.update(
                {
                    "started_at_unix_ns": 1,
                    "duration_ns": 1,
                    "result_rows": 50,
                    "dataset_version": 2,
                    "fragments": 1,
                    "physical_rows": 100,
                    "physical_data_bytes": 1600,
                    "estimated_live_data_bytes": 800,
                    "scan_byte_amplification": 2.0,
                    "dataset_bytes": 1600,
                    "peak_rss_bytes": 4096,
                    "io_metrics_status": "not_instrumented",
                    "status": "ok",
                    "error": None,
                }
            )
            self.assertIs(run.validate_record(record, expected), record)
            record["implementation_path"] = (
                "same_postcondition_repack_or_default_compaction"
            )
            with self.assertRaisesRegex(
                run.RecordValidationError, "worker record provenance mismatch"
            ):
                run.validate_record(record, expected)

    def test_clustered_delete_reclaim_is_a_default_fast_path(self) -> None:
        matrix, _, _ = protocol.load_matrix(protocol.DEFAULT_MATRIX)
        cases = {
            case.name: case
            for case in protocol.iter_matrix_cases(
                matrix["profiles"]["smoke"], {"delete_clustered"}
            )
        }
        for percentage in (1, 50, 90):
            case = cases[f"delete-clustered-{percentage}/narrow16/take-1"]
            self.assertEqual(
                [step.operation for step in case.steps],
                ["create", "delete", "default_compaction"],
            )
            self.assertEqual(case.steps[-1].selection, "range")
    def test_indexed_relocation_cases_cover_scalar_and_vector(self) -> None:
        matrix, _, _ = protocol.load_matrix(protocol.DEFAULT_MATRIX)
        cases = list(
            protocol.iter_matrix_cases(
                matrix["profiles"]["smoke"],
                {"indexed_n_to_one_compaction", "indexed_repeated_compaction"},
            )
        )
        self.assertTrue(any(case.name.endswith("/scalar") for case in cases))
        self.assertTrue(any(case.name.endswith("/vector") for case in cases))
        relocation_steps = [
            step
            for case in cases
            for step in case.steps
            if step.operation == "default_compaction"
        ]
        self.assertTrue(relocation_steps)
        self.assertEqual(
            {step.index_kind for step in relocation_steps}, {"scalar", "vector"}
        )
        self.assertEqual(
            {case.fixture_index_kind for case in cases}, {"scalar", "vector"}
        )

    def test_release_fixture_dedup_and_cost_projection_are_frozen(self) -> None:
        matrix, _, _ = protocol.load_matrix(protocol.DEFAULT_MATRIX)
        profile = matrix["profiles"]["release"]
        tracks = list(matrix["tracks"])
        variants = ["bare", "scalar", "vector"]
        cases = list(
            protocol.iter_matrix_cases(
                profile, set(matrix["tracks"]["matrix"]["cases"])
            )
        )
        keys = protocol.fixture_keys_for_run(profile, tracks, variants, cases)
        self.assertEqual(len(keys), 15)
        self.assertEqual(
            protocol.projected_canonical_payload_bytes(profile, keys),
            604_800_000_000,
        )
        self.assertEqual(
            protocol.projected_unique_initial_index_payload_bytes_lower_bound(
                profile, keys, cases
            ),
            511_200_000_000,
        )
        self.assertEqual(
            protocol.projected_minimum_full_scan_payload_bytes(
                profile, tracks, variants, cases
            ),
            1_669_685_760_000_000,
        )
        shards = [protocol.fixture_keys_for_shard(keys, 4, index) for index in range(4)]
        self.assertEqual(set().union(*shards), keys)
        self.assertTrue(
            all(
                left.isdisjoint(right)
                for i, left in enumerate(shards)
                for right in shards[i + 1 :]
            )
        )

    def test_checkpoint_resume_is_exact_and_ambiguous_safe(self) -> None:
        policy, _, policy_sha = run.canonical_policy(run.DEFAULT_POLICY)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            kwargs = {
                "executable": Path("/tmp/worker"),
                "output": output,
                "dataset_root": str(Path(directory) / "data"),
                "storage": "ebs",
                "mode": "smoke",
                "commit": "1" * 40,
                "host": "host",
                "seed": 7,
                "policy": policy,
                "policy_sha256": policy_sha,
                "policy_version": 1,
                "run_id": "run-1",
                "rows": 10,
                "rows_per_fragment": 5,
                "take_count": 1,
                "matrix_sha256": "2" * 64,
                "shard_id": "shard-000-of-001",
            }
            runner = protocol.ProtocolRunner(**kwargs, resume=False)
            runner.close()
            resumed = protocol.ProtocolRunner(**kwargs, resume=True)
            resumed.close()
            checkpoint_path = Path(f"{output}.checkpoint.json")
            checkpoint = json.loads(checkpoint_path.read_text())
            checkpoint["inflight"] = {"pair_id": "missing", "format": "v23_logical"}
            protocol.replace_json_atomic(checkpoint_path, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "ambiguous worker outcome"):
                protocol.ProtocolRunner(**kwargs, resume=True)

    def test_resume_bootstraps_after_sidecar_only_crash(self) -> None:
        policy, _, policy_sha = run.canonical_policy(run.DEFAULT_POLICY)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            runner = protocol.ProtocolRunner(
                executable=Path("/tmp/worker"),
                output=output,
                dataset_root=str(Path(directory) / "data"),
                storage="ebs",
                mode="smoke",
                commit="1" * 40,
                host="host",
                seed=7,
                policy=policy,
                policy_sha256=policy_sha,
                policy_version=1,
                run_id="run-1",
                rows=10,
                rows_per_fragment=5,
                take_count=1,
                matrix_sha256="2" * 64,
                shard_id="shard-000-of-001",
                resume=True,
            )
            runner.close()
            self.assertEqual(output.read_text(encoding="utf-8"), "")
            self.assertTrue(Path(f"{output}.checkpoint.json").is_file())
            self.assertTrue(Path(f"{output}.fixture_lineage.jsonl").is_file())

    def test_maintenance_plan_is_hashed_and_reused(self) -> None:
        policy, _, policy_sha = run.canonical_policy(run.DEFAULT_POLICY)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            runner = protocol.ProtocolRunner(
                executable=Path("/tmp/worker"),
                output=output,
                dataset_root=str(Path(directory) / "data"),
                storage="ebs",
                mode="smoke",
                commit="1" * 40,
                host="host",
                seed=7,
                policy=policy,
                policy_sha256=policy_sha,
                policy_version=1,
                run_id="run-1",
                rows=10,
                rows_per_fragment=5,
                take_count=1,
                matrix_sha256="2" * 64,
                shard_id="shard-000-of-001",
                resume=False,
            )
            step = protocol.Step(
                "default_compaction",
                10,
                target_rows_per_fragment=10,
            )

            def write_plan(command: tuple[str, ...], **_: object) -> object:
                plan_path = Path(
                    command[command.index("--prepare-maintenance-plan-output") + 1]
                )
                value = {
                    "schema_version": 1,
                    "suite": "stable_row_address_physical_maintenance_plan",
                    "run_id": "run-1",
                    "commit": "1" * 40,
                    "policy_sha256": policy_sha,
                    "source_format": "v22_no_stable",
                    "source_dataset_uri": runner.dataset_uri(
                        "sustained", "bare", 0, "v22_no_stable"
                    ),
                    "source_dataset_version": 2,
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
                plan_path.parent.mkdir(parents=True, exist_ok=True)
                plan_path.write_text(json.dumps(value) + "\n", encoding="utf-8")
                return type("Result", (), {"returncode": 0, "stdout": ""})()

            with mock.patch.object(
                protocol.subprocess, "run", side_effect=write_plan
            ) as run_mock:
                first = runner.prepare_maintenance_plan(
                    step,
                    track="sustained",
                    case="bare",
                    repeat=0,
                    label="round-000/policy-maintenance-plan",
                    source_format="v22_no_stable",
                    max_source_fragments_per_group=256,
                    target_file_size_bytes=134_217_728,
                )
                second = runner.prepare_maintenance_plan(
                    step,
                    track="sustained",
                    case="bare",
                    repeat=0,
                    label="round-000/policy-maintenance-plan",
                    source_format="v22_no_stable",
                    max_source_fragments_per_group=256,
                    target_file_size_bytes=134_217_728,
                )
            runner.close()
            self.assertEqual(first, second)
            self.assertRegex(first[1], r"^[0-9a-f]{64}$")
            self.assertEqual(run_mock.call_count, 1)

    def test_sustained_and_adversarial_sampling_nonces_are_distinct(self) -> None:
        sustained = protocol.Step(
            "update",
            100,
            mutation_count=1,
            step=7,
            selection_step=0,
            update_driver="exact-matched-merge",
            selection="random",
        )
        adversarial = dataclasses_replace(sustained, selection_step=7)
        self.assertEqual(sustained.selection_step, 0)
        self.assertEqual(adversarial.selection_step, adversarial.step)
        self.assertEqual(sustained.implementation_path, "exact_selection_matched_merge")

    def test_policy_uses_strict_frozen_thresholds(self) -> None:
        policy, _, _ = run.canonical_policy(run.DEFAULT_POLICY)
        record = {
            "physical_data_bytes": 1000,
            "estimated_live_data_bytes": 800,
            "fragments": 1,
            "scan_byte_amplification": 1.02,
        }
        triggered, metrics = protocol.policy_triggers(record, policy)
        self.assertFalse(triggered)
        self.assertEqual(metrics["live_byte_ratio"], 0.8)
        record["estimated_live_data_bytes"] = 799
        triggered, _ = protocol.policy_triggers(record, policy)
        self.assertTrue(triggered)

    def test_matrix_rejects_unknown_profile_field(self) -> None:
        matrix, _, _ = protocol.load_matrix(protocol.DEFAULT_MATRIX)
        invalid = copy.deepcopy(matrix)
        invalid["profiles"]["smoke"]["unknown"] = 1
        with self.assertRaisesRegex(ValueError, "unknown"):
            protocol._strict_object(
                invalid["profiles"]["smoke"],
                protocol.PROFILE_FIELDS,
                "profile smoke",
            )

    def test_matrix_rejects_changed_reclaim_admission_contract(self) -> None:
        matrix = json.loads(protocol.DEFAULT_MATRIX.read_text(encoding="utf-8"))
        matrix["profiles"]["smoke"][
            "random_delete_reclaim_admission"
        ] = "must_not_admit"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.json"
            path.write_text(json.dumps(matrix), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be must_admit"):
                protocol.load_matrix(path)


def dataclasses_replace(value: protocol.Step, **changes: object) -> protocol.Step:
    # Keep the test independent of protocol's orchestration methods.
    import dataclasses

    return dataclasses.replace(value, **changes)


if __name__ == "__main__":
    unittest.main()
