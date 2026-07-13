#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import protocol  # noqa: E402
import protocol_report  # noqa: E402
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
        self.assertEqual(release["random_delete_reclaim_admission"], "must_not_admit")
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
        one_percent_reclaim = delete_cases["delete-random-1/narrow16/take-1"].steps[-1]
        self.assertEqual(one_percent_reclaim.operation, "default_compaction")
        self.assertIs(one_percent_reclaim.preflight_expected_admission, True)
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
        self.assertEqual(
            chain_cases["pack-random-delete-1/vector"].steps[-1].operation,
            "default_compaction",
        )
        self.assertIs(
            chain_cases["pack-random-delete-1/vector"]
            .steps[-1]
            .preflight_expected_admission,
            True,
        )

    def test_default_compaction_preflight_preserves_frozen_admission(self) -> None:
        relocation = protocol.Step(
            "default_compaction",
            expected_rows=99,
            preflight_expected_admission=True,
        )
        preflight = dataclasses_replace(
            relocation,
            operation="default_compaction_preflight",
        )

        self.assertEqual(preflight.operation, "default_compaction_preflight")
        self.assertIs(preflight.preflight_expected_admission, True)
        self.assertEqual(preflight.implementation_path, "default_compaction_plan_only")

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
            self.assertEqual(expected["implementation_path"], implementation_path)
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

    def test_indexed_random_delete_repack_is_prebuilt_and_paired(self) -> None:
        matrix, _, _ = protocol.load_matrix(protocol.DEFAULT_MATRIX)
        cases = {
            case.name: case
            for case in protocol.iter_matrix_cases(
                matrix["profiles"]["release"], {"delete_random"}
            )
            if case.name.startswith("indexed-repack-random-delete-")
        }
        self.assertEqual(
            set(cases),
            {
                "indexed-repack-random-delete-50/scalar",
                "indexed-repack-random-delete-50/vector",
                "indexed-repack-random-delete-90/scalar",
                "indexed-repack-random-delete-90/vector",
            },
        )
        for case in cases.values():
            self.assertIn(case.fixture_index_kind, {"scalar", "vector"})
            self.assertEqual(case.steps[-1].index_kind, case.fixture_index_kind)
            self.assertEqual(case.steps[-1].operation, "random_delete_reclaim")
            self.assertIs(case.steps[-1].preflight_expected_admission, False)

    def test_index_probe_reuses_prepared_live_user_ids(self) -> None:
        runner = object.__new__(protocol.ProtocolRunner)
        runner.run_id = "run-1"
        runner.phase_index = 0
        runner.failures = []
        invocations: list[tuple[str, str, Path | None]] = []

        with tempfile.TemporaryDirectory() as directory:
            artifacts: dict[str, Path] = {}
            for format_name in run.FORMATS:
                path = Path(directory) / f"{format_name}.json"
                path.write_text(json.dumps({"user_ids": [7, 41]}), encoding="utf-8")
                artifacts[format_name] = path

            runner.invoke_all = mock.Mock(
                side_effect=lambda step, **_: {
                    name: {"status": "ok", "state_digest": "same"}
                    for name in run.FORMATS
                }
            )
            runner.prepare_take_ids = mock.Mock(
                side_effect=lambda _step, *, format_name, **__: artifacts[format_name]
            )

            def invoke_one(
                step: protocol.Step,
                *,
                format_name: str,
                take_ids_input: Path | None = None,
                **_: object,
            ) -> dict[str, object]:
                invocations.append((step.operation, format_name, take_ids_input))
                return {"status": "ok"}

            runner.invoke_one = invoke_one
            runner.probes(
                track="matrix",
                case="indexed-repack-random-delete-90/vector",
                repeat=0,
                expected_rows=10,
                schema_kind="vector",
                index_kind="vector",
                step_index=2,
            )

        indexed = [
            invocation for invocation in invocations if invocation[0] == "index_take"
        ]
        self.assertEqual(len(indexed), len(run.FORMATS))
        self.assertEqual(
            {format_name: artifact for _, format_name, artifact in indexed},
            artifacts,
        )

    def test_explicit_recluster_and_fragment_reuse_cases_are_frozen(self) -> None:
        matrix, _, _ = protocol.load_matrix(protocol.DEFAULT_MATRIX)
        profile = matrix["profiles"]["smoke"]
        cases = {
            case.name: case
            for case in protocol.iter_matrix_cases(
                profile, {"bounded_recluster", "fragment_reuse"}
            )
        }
        bounded_fragments = profile["logical_fragment_counts"][0]
        self.assertEqual(
            {name for name in cases if name.startswith("bounded-default-clustering-")},
            {
                f"bounded-default-clustering-{bounded_fragments}/narrow16",
                f"bounded-default-clustering-{bounded_fragments}/wide128",
                f"bounded-default-clustering-{bounded_fragments}/vector",
            },
        )
        for name, case in cases.items():
            if not name.startswith("bounded-default-clustering-"):
                continue
            step = case.steps[-1]
            self.assertEqual(step.operation, "bounded_recluster")
            self.assertEqual(
                {
                    format_name: step.implementation_path_for_format(format_name)
                    for format_name in run.FORMATS
                },
                {
                    "v22_no_stable": "same_postcondition_bounded_recluster_rewrite",
                    "v22_stable": "same_postcondition_bounded_recluster_rewrite",
                    "v23_logical": "default_bounded_recluster_fast_path",
                },
            )
        self.assertEqual(
            {name for name in cases if name.startswith("bounded-recluster-")},
            {
                f"bounded-recluster-{bounded_fragments}/narrow16",
                f"bounded-recluster-{bounded_fragments}/wide128",
                f"bounded-recluster-{bounded_fragments}/vector",
            },
        )
        self.assertTrue(
            all(
                case.steps[-1].operation == "recluster"
                for name, case in cases.items()
                if name.startswith("bounded-recluster-")
            )
        )

        reuse_cases = {
            name: case
            for name, case in cases.items()
            if name.startswith("fragment-reuse-")
        }
        self.assertEqual(len(reuse_cases), 2 * len(profile["logical_fragment_counts"]))
        for case in reuse_cases.values():
            step = case.steps[-1]
            self.assertEqual(step.compaction_mode, "fragment_reuse")
            self.assertEqual(
                {
                    format_name: step.implementation_path_for_format(format_name)
                    for format_name in run.FORMATS
                },
                {
                    "v22_no_stable": "deferred_fragment_reuse_compaction",
                    "v22_stable": "inline_index_remap_compaction",
                    "v23_logical": "stable_logical_zero_remap_compaction",
                },
            )

    def test_release_skewed_packed_run_fixtures_are_exact(self) -> None:
        matrix, _, _ = protocol.load_matrix(protocol.DEFAULT_MATRIX)
        profile = matrix["profiles"]["release"]
        cases = {
            case.name: case
            for case in protocol.iter_matrix_cases(profile, {"n_to_one_compaction"})
            if case.fixture_segments
        }
        for fragments in (10_000, 100_000):
            case = cases[f"compact-{fragments}-skew-to-1/narrow16"]
            segments = case.fixture_segments
            self.assertEqual(sum(rows for rows, _ in segments), profile["rows"])
            self.assertEqual(
                sum(rows // rows_per_fragment for rows, rows_per_fragment in segments),
                fragments,
            )
            self.assertEqual(
                len({rows_per_fragment for _, rows_per_fragment in segments}), 2
            )

    def test_repeated_compaction_has_strictly_progressive_topology(self) -> None:
        rows = 65_536
        source_fragments = 64
        rounds = 10
        targets = [
            protocol.repeated_compaction_target_rows(
                rows, source_fragments, rounds, round_index
            )
            for round_index in range(rounds)
        ]
        output_fragments = [math.ceil(rows / target) for target in targets]
        self.assertTrue(
            all(
                later < earlier
                for earlier, later in zip(output_fragments, output_fragments[1:])
            )
        )
        self.assertEqual(output_fragments[-1], 1)

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
        self.assertEqual(len(keys), 17)
        self.assertEqual(
            protocol.projected_canonical_payload_bytes(profile, keys),
            614_400_000_000,
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
            1_691_650_560_000_000,
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

    def test_focused_sustained_development_tiny_sidecar_is_verifiable(self) -> None:
        class FakeRunner:
            records = 0
            boundaries = 0
            pmr_triggers = 0
            failures: list[str] = []

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "stable_row_address_e2e"
            executable.touch()
            output = root / "results.jsonl"
            with (
                mock.patch.object(
                    protocol.subprocess,
                    "run",
                    return_value=type(
                        "Result", (), {"stdout": "1" * 40, "returncode": 0}
                    )(),
                ),
                mock.patch.object(
                    protocol, "ProtocolRunner", return_value=FakeRunner()
                ),
                mock.patch.object(protocol, "run_sustained") as sustained,
            ):
                self.assertEqual(
                    protocol.main(
                        [
                            "--dataset-root",
                            str(root / "data"),
                            "--output",
                            str(output),
                            "--track",
                            "sustained",
                            "--variant",
                            "bare",
                            "--development-executable",
                            str(executable),
                            "--development-tiny",
                        ]
                    ),
                    0,
                )
            sustained.assert_called_once()
            sidecar = json.loads(
                Path(f"{output}.protocol.json").read_text(encoding="utf-8")
            )
            self.assertEqual(sidecar["tracks"], ["sustained"])
            self.assertEqual(sidecar["matrix_case_names"], [])
            self.assertTrue(sidecar["development_tiny"])
            self.assertEqual(sidecar["matrix"]["profiles"]["smoke"]["rows"], 4096)
            self.assertEqual(protocol_report.validate_sidecar(sidecar), sidecar)

    def test_release_requires_canonical_shards_and_selection(self) -> None:
        base = [
            "--dataset-root",
            "s3://bucket/prefix",
            "--storage",
            "s3",
            "--output",
            "/tmp/release.jsonl",
            "--profile",
            "release",
        ]
        with self.assertRaisesRegex(ValueError, "exactly nine canonical shards"):
            protocol.main(base)
        with self.assertRaisesRegex(ValueError, "canonical complete track order"):
            protocol.main([*base, "--shard-count", "9", "--track", "sustained"])
        with self.assertRaisesRegex(ValueError, "canonical seed"):
            protocol.main(
                [
                    *base,
                    "--shard-count",
                    "9",
                    "--seed",
                    str(protocol.RELEASE_SEED + 1),
                ]
            )
        with tempfile.TemporaryDirectory() as directory:
            matrix = json.loads(protocol.DEFAULT_MATRIX.read_text(encoding="utf-8"))
            matrix["name"] = "custom_release_matrix"
            matrix_path = Path(directory) / "matrix.json"
            matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "repository default matrix"):
                protocol.main(
                    [
                        *base,
                        "--shard-count",
                        "9",
                        "--matrix",
                        str(matrix_path),
                    ]
                )

            policy = json.loads(run.DEFAULT_POLICY.read_text(encoding="utf-8"))
            policy["trigger"]["conditions"][0]["threshold"] = 0.1
            policy_path = Path(directory) / "policy.json"
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "repository default policy"):
                protocol.main(
                    [
                        *base,
                        "--shard-count",
                        "9",
                        "--policy",
                        str(policy_path),
                    ]
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

    def test_sustained_index_consolidation_stays_in_timed_maintenance(self) -> None:
        policy, _, _ = run.canonical_policy(run.DEFAULT_POLICY)

        class RecordingRunner:
            def __init__(self) -> None:
                self.policy = policy
                self.events: list[tuple[object, ...]] = []
                self.boundaries = 0
                self.failures: list[str] = []

            def clone_fixture_all(self, **_: object) -> dict[str, dict[str, object]]:
                self.events.append(("fixture",))
                return {name: {"status": "ok"} for name in run.FORMATS}

            def invoke_all(
                self, step: protocol.Step, **kwargs: object
            ) -> dict[str, dict[str, object]]:
                self.events.append(
                    ("invoke", step.operation, kwargs.get("maintenance_plan"))
                )
                return {name: {"status": "ok"} for name in run.FORMATS}

            def require_success(self, records: object, context: str) -> None:
                self.events.append(("require", context))

            def probes(self, **_: object) -> dict[str, dict[str, object]]:
                self.events.append(("probes",))
                triggered = {
                    "physical_data_bytes": 1_000,
                    "estimated_live_data_bytes": 500,
                    "fragments": 2,
                    "scan_byte_amplification": 2.0,
                }
                return {name: dict(triggered) for name in run.FORMATS}

            def prepare_maintenance_plan(
                self, step: protocol.Step, **_: object
            ) -> tuple[Path, str]:
                self.events.append(("prepare", step.operation))
                return Path("/tmp/frozen-plan.json"), "a" * 64

            def complete_unit(self, unit: str) -> None:
                self.events.append(("complete", unit))

        runner = RecordingRunner()
        protocol.run_sustained(
            runner,  # type: ignore[arg-type]
            {
                "rows": 100,
                "hot_set_rows": 10,
                "repeated_update_rounds": 1,
                "logical_fragment_counts": [2],
                "paired_repeats": 1,
                "minimum_sustained_boundaries": 1,
            },
            ["vector"],
        )
        prepare_index = runner.events.index(("prepare", "default_compaction"))
        maintenance_event = (
            "invoke",
            "default_compaction",
            (Path("/tmp/frozen-plan.json"), "a" * 64),
        )
        maintenance_index = runner.events.index(maintenance_event)
        self.assertLess(prepare_index, maintenance_index)
        self.assertNotIn(
            ("invoke", "index_optimize", None),
            runner.events[prepare_index + 1 : maintenance_index],
        )
        self.assertEqual(
            [event[1] for event in runner.events if event[0] == "invoke"],
            ["update", "index_optimize", "default_compaction"],
        )

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
        matrix["profiles"]["smoke"]["random_delete_reclaim_admission"] = (
            "must_not_admit"
        )
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
