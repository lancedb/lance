#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import protocol_aggregate  # noqa: E402
import protocol_report  # noqa: E402


def make_shard(index: int, *, dataset_root: str | None = None) -> dict[str, object]:
    shard_id = f"shard-{index:03d}-of-002"
    return {
        "run_id": f"run-{index}",
        "commit": "1" * 40,
        "source_provenance": "clean-committed-source",
        "development_tiny": False,
        "profile": "smoke",
        "cargo_profile": "release-with-debug",
        "storage": "ebs",
        "host": "host-1",
        "storage_region_attestation": None,
        "base_dataset_root": "/tmp/data",
        "dataset_root": dataset_root or f"/tmp/data/{shard_id}",
        "seed": 7,
        "matrix_sha256": "2" * 64,
        "policy_sha256": "3" * 64,
        "shard_count": 2,
        "shard_index": index,
        "shard_id": shard_id,
        "shard_strategy": "schema_and_fragment_layout_fixture_locality",
        "matrix_case_names": [f"case-{index}"],
        "tracks": ["matrix"],
        "variants": [],
        "projected_canonical_payload_bytes": 10,
        "projected_unique_initial_index_payload_bytes_lower_bound": 20,
        "projected_no_dedup_logical_data_payload_bytes": 30,
        "projected_no_dedup_logical_index_payload_bytes": 40,
        "projected_minimum_full_scan_payload_bytes": 50,
    }


def passing_report() -> protocol_report.ReportResult:
    return protocol_report.ReportResult(
        verdict="PASS",
        markdown="",
        machine={"records": 3, "complete_pairs": 1},
    )


class ProtocolAggregateTests(unittest.TestCase):
    def test_release_aggregate_requires_exact_execution_commit_marker(self) -> None:
        shards = [make_shard(0), make_shard(1)]
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "execution-complete"
            marker.write_text("1" * 40 + "\n", encoding="utf-8")
            with (
                mock.patch.object(
                    protocol_report,
                    "load_evidence",
                    side_effect=[(shard, [], []) for shard in shards],
                ),
                mock.patch.object(
                    protocol_report,
                    "analyze",
                    side_effect=[passing_report(), passing_report()],
                ),
            ):
                exit_code, _, machine = protocol_aggregate.aggregate(
                    [Path("a.jsonl"), Path("b.jsonl")],
                    bootstrap_samples=101,
                    expected_commit="1" * 40,
                    execution_marker=marker,
                )
            self.assertEqual(exit_code, 0)
            self.assertEqual(machine["verdict"], "PASS")

            marker.write_text("2" * 40 + "\n", encoding="utf-8")
            with (
                mock.patch.object(
                    protocol_report,
                    "load_evidence",
                    side_effect=[(shard, [], []) for shard in shards],
                ),
                mock.patch.object(
                    protocol_report,
                    "analyze",
                    side_effect=[passing_report(), passing_report()],
                ),
            ):
                exit_code, _, machine = protocol_aggregate.aggregate(
                    [Path("a.jsonl"), Path("b.jsonl")],
                    bootstrap_samples=101,
                    expected_commit="1" * 40,
                    execution_marker=marker,
                )
            self.assertEqual(exit_code, 2)
            self.assertTrue(
                any("execution-complete marker" in issue for issue in machine["issues"])
            )

    def test_complete_independent_shards_pass_and_sum_projections(self) -> None:
        shards = [make_shard(0), make_shard(1)]
        with (
            mock.patch.object(
                protocol_report,
                "load_evidence",
                side_effect=[(shard, [], []) for shard in shards],
            ),
            mock.patch.object(
                protocol_report,
                "analyze",
                side_effect=[passing_report(), passing_report()],
            ),
        ):
            exit_code, _, machine = protocol_aggregate.aggregate(
                [Path("a.jsonl"), Path("b.jsonl")], bootstrap_samples=101
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(machine["verdict"], "PASS")
        self.assertEqual(machine["commit"], "1" * 40)
        self.assertEqual(
            machine["storage_projections"]["projected_minimum_full_scan_payload_bytes"],
            100,
        )

    def test_missing_or_shared_shard_is_incomplete(self) -> None:
        shard = make_shard(0)
        duplicate_root = make_shard(0)
        with (
            mock.patch.object(
                protocol_report,
                "load_evidence",
                side_effect=[(shard, [], []), (duplicate_root, [], [])],
            ),
            mock.patch.object(
                protocol_report,
                "analyze",
                side_effect=[passing_report(), passing_report()],
            ),
        ):
            exit_code, _, machine = protocol_aggregate.aggregate(
                [Path("a.jsonl"), Path("b.jsonl")], bootstrap_samples=101
            )
        self.assertEqual(exit_code, 2)
        self.assertTrue(
            any("independent dataset prefixes" in issue for issue in machine["issues"])
        )

        with (
            mock.patch.object(
                protocol_report, "load_evidence", return_value=(shard, [], [])
            ),
            mock.patch.object(
                protocol_report, "analyze", return_value=passing_report()
            ),
        ):
            exit_code, _, machine = protocol_aggregate.aggregate(
                [Path("a.jsonl")], bootstrap_samples=101
            )
        self.assertEqual(exit_code, 2)
        self.assertTrue(
            any("expected shard indices" in issue for issue in machine["issues"])
        )

    def test_unrelated_shard_dataset_root_is_incomplete(self) -> None:
        shards = [
            make_shard(0, dataset_root="/tmp/unrelated/shard-000-of-002"),
            make_shard(1),
        ]
        with (
            mock.patch.object(
                protocol_report,
                "load_evidence",
                side_effect=[(shard, [], []) for shard in shards],
            ),
            mock.patch.object(
                protocol_report,
                "analyze",
                side_effect=[passing_report(), passing_report()],
            ),
        ):
            exit_code, _, machine = protocol_aggregate.aggregate(
                [Path("a.jsonl"), Path("b.jsonl")], bootstrap_samples=101
            )

        self.assertEqual(exit_code, 2)
        self.assertTrue(
            any("dataset_root is inconsistent" in issue for issue in machine["issues"])
        )

    def test_region_attestation_disagreement_is_incomplete(self) -> None:
        shards = [make_shard(0), make_shard(1)]
        shards[0]["storage_region_attestation"] = {"bucket_region": "us-east-2"}
        shards[1]["storage_region_attestation"] = {"bucket_region": "us-west-2"}
        with (
            mock.patch.object(
                protocol_report,
                "load_evidence",
                side_effect=[(shard, [], []) for shard in shards],
            ),
            mock.patch.object(
                protocol_report,
                "analyze",
                side_effect=[passing_report(), passing_report()],
            ),
        ):
            exit_code, _, machine = protocol_aggregate.aggregate(
                [Path("a.jsonl"), Path("b.jsonl")], bootstrap_samples=101
            )

        self.assertEqual(exit_code, 2)
        self.assertIn(
            "shards disagree on storage_region_attestation", machine["issues"]
        )


if __name__ == "__main__":
    unittest.main()
