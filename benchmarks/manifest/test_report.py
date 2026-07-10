"""Tests for the standalone manifest benchmark report generator."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


REPORT_PATH = Path(__file__).with_name("report.py")
SPEC = importlib.util.spec_from_file_location("manifest_benchmark_report", REPORT_PATH)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)

FRAGMENT_SIZES = (1_000, 100_000, 1_000_000, 10_000_000)
COMMIT = "0123456789abcdef0123456789abcdef01234567"
SEED = 8675309
HOST = "benchmark-host"


def _base_record(
    *,
    suite: str,
    scenario: str,
    fragments: int,
    format_name: str,
    storage: str,
    operation: str,
    round_number: int,
) -> dict[str, Any]:
    scenario_numerator = 5 if scenario == "S2" else 4
    scale = fragments * scenario_numerator // 4
    return {
        "schema_version": 2,
        "suite": suite,
        "scenario": scenario,
        "fragments": fragments,
        "format": format_name,
        "storage": storage,
        "operation": operation,
        "round": round_number,
        "wall_ns": scale * 100,
        "bytes": scale * 80,
        "peak_rss_bytes": scale * 60,
        "get_requests": 1,
        "put_requests": 1,
        "read_bytes": scale * 80,
        "write_bytes": scale * 200,
        "status": "success",
        "error": None,
        "commit": COMMIT,
        "seed": SEED,
        "host": HOST,
    }


def _codec_records() -> list[dict[str, Any]]:
    records = []
    for scenario in ("S1", "S2"):
        for fragments in FRAGMENT_SIZES:
            for operation in ("encode", "decode", "size", "decode_rss"):
                for format_name in ("protobuf", "lance"):
                    for round_number in range(5):
                        record = _base_record(
                            suite="codec",
                            scenario=scenario,
                            fragments=fragments,
                            format_name=format_name,
                            storage="memory",
                            operation=operation,
                            round_number=round_number,
                        )
                        record["get_requests"] = 0
                        record["put_requests"] = 0
                        if operation == "encode":
                            record["read_bytes"] = 0
                            record["write_bytes"] = record["bytes"]
                        elif operation == "size":
                            record["read_bytes"] = 0
                            record["write_bytes"] = 0
                        else:
                            record["read_bytes"] = record["bytes"]
                            record["write_bytes"] = 0
                        if format_name == "lance":
                            if operation == "encode":
                                record["wall_ns"] = record["wall_ns"] * 11 // 10
                            elif operation == "decode":
                                record["wall_ns"] = record["wall_ns"] * 4 // 5
                            elif operation == "size":
                                record["bytes"] = record["bytes"] * 4 // 5
                            elif operation == "decode_rss":
                                record["peak_rss_bytes"] = (
                                    record["peak_rss_bytes"] * 4 // 5
                                )
                        records.append(record)
    return records


def _e2e_records() -> list[dict[str, Any]]:
    records = []
    for scenario in ("S1", "S2"):
        for storage in ("ebs", "s3"):
            for operation in ("open", "commit", "conflict_retry", "time_travel"):
                fragments_set = (
                    FRAGMENT_SIZES
                    if operation in ("open", "time_travel")
                    else (1_000_000, 10_000_000)
                )
                for fragments in fragments_set:
                    for format_name in ("protobuf", "lance"):
                        for round_number in range(5):
                            record = _base_record(
                                suite="e2e",
                                scenario=scenario,
                                fragments=fragments,
                                format_name=format_name,
                                storage=storage,
                                operation=operation,
                                round_number=round_number,
                            )
                            if operation == "conflict_retry":
                                record["get_requests"] = 3
                                record["put_requests"] = 2
                            if format_name == "lance":
                                if operation == "open":
                                    factor_numerator = (
                                        5
                                        if storage == "s3" and fragments >= 1_000_000
                                        else 9
                                    )
                                    factor_denominator = 10
                                elif operation == "commit":
                                    factor_numerator = 21
                                    factor_denominator = 20
                                else:
                                    factor_numerator = 9
                                    factor_denominator = 10
                                record["wall_ns"] = (
                                    record["wall_ns"]
                                    * factor_numerator
                                    // factor_denominator
                                )
                                record["write_bytes"] = record["write_bytes"] * 4 // 5
                            record["bytes"] = (
                                record["read_bytes"] + record["write_bytes"]
                            )
                            records.append(record)
    return records


def _complete_records() -> list[dict[str, Any]]:
    return _codec_records() + _e2e_records()


def _case_selector(
    *,
    suite: str,
    scenario: str,
    fragments: int,
    storage: str,
    operation: str,
    metric: str,
) -> dict[str, Any]:
    return {
        "gate": "case",
        "schema_version": 2,
        "suite": suite,
        "commit": COMMIT,
        "seed": SEED,
        "host": HOST,
        "scenario": scenario,
        "fragments": fragments,
        "storage": storage,
        "operation": operation,
        "metric": metric,
    }


def _scaling_selector(
    *, scenario: str, format_name: str, operation: str, metric: str
) -> dict[str, Any]:
    return {
        "gate": "scaling",
        "schema_version": 2,
        "suite": "codec",
        "commit": COMMIT,
        "seed": SEED,
        "host": HOST,
        "scenario": scenario,
        "format": format_name,
        "operation": operation,
        "metric": metric,
    }


def _sidecar(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 2, "explanations": list(entries)}


class ReportTest(unittest.TestCase):
    def _generate(
        self,
        records: list[dict[str, Any]],
        explanations: dict[str, Any] | None = None,
    ):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "manifest-results.jsonl"
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        explanations_path = None
        if explanations is not None:
            explanations_path = Path(temporary_directory.name) / "explanations.json"
            explanations_path.write_text(json.dumps(explanations), encoding="utf-8")
        return REPORT.generate_report([path], COMMIT, explanations_path), path

    def test_complete_passing_matrix_is_pass(self) -> None:
        outcome, path = self._generate(_complete_records())

        self.assertEqual("PASS", outcome.verdict)
        self.assertEqual(0, outcome.exit_code)
        self.assertIn("## Codec Gates", outcome.markdown)
        self.assertIn("## End-to-End Gates", outcome.markdown)
        self.assertIn("read I/O operations", outcome.markdown)
        self.assertIn("GET/PUT requests", outcome.markdown)
        self.assertIn("### Benchmark Commands", outcome.markdown)
        self.assertIn("python3 benchmarks/manifest/run_codec.py", outcome.markdown)
        self.assertIn("python3 benchmarks/manifest/run_e2e.py", outcome.markdown)
        self.assertIn(f"--expected-commit {COMMIT}", outcome.markdown)
        self.assertIn("--dataset-prefix /mnt/lance-manifest-bench", outcome.markdown)
        self.assertIn("s3://<bucket>/<prefix>/lance-manifest-bench", outcome.markdown)
        self.assertIn(f"--output {path.resolve()}", outcome.markdown)
        self.assertIn("## 100M Linear Extrapolation (Not Measured)", outcome.markdown)
        self.assertIn("exactly 10 times the measured 10M median", outcome.markdown)
        self.assertIn(str(path.resolve()), outcome.markdown)

    def test_schema_v1_records_are_rejected(self) -> None:
        records = _complete_records()
        records[0]["schema_version"] = 1

        outcome, _ = self._generate(records)

        self.assertEqual("INCOMPLETE", outcome.verdict)
        self.assertIn("unsupported schema_version 1; expected 2", outcome.markdown)

    def test_record_commit_must_be_expected_full_sha(self) -> None:
        for commit, message in (
            ("not-a-git-sha", "full lowercase 40- or 64-character Git SHA"),
            ("f" * 40, "does not match expected commit"),
        ):
            records = _complete_records()
            records[0]["commit"] = commit

            outcome, _ = self._generate(records)

            with self.subTest(commit=commit):
                self.assertEqual("INCOMPLETE", outcome.verdict)
                self.assertIn(message, outcome.markdown)

    def test_invalid_expected_commit_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected_commit"):
            REPORT.generate_report([], "deadbeef")

    def test_e2e_total_bytes_must_match_components(self) -> None:
        records = _complete_records()
        target = next(record for record in records if record["suite"] == "e2e")
        target["bytes"] += 1

        outcome, _ = self._generate(records)

        self.assertEqual("INCOMPLETE", outcome.verdict)
        self.assertIn("e2e bytes must equal read_bytes + write_bytes", outcome.markdown)

    def test_codec_request_counts_must_be_zero(self) -> None:
        records = _complete_records()
        target = next(record for record in records if record["suite"] == "codec")
        target["get_requests"] = 1

        outcome, _ = self._generate(records)

        self.assertEqual("INCOMPLETE", outcome.verdict)
        self.assertIn(
            "codec get_requests and put_requests must both be zero", outcome.markdown
        )

    def test_codec_operation_byte_contracts_are_enforced(self) -> None:
        cases = (
            ("encode", "read_bytes", 1, "codec encode requires"),
            ("size", "write_bytes", 1, "codec size requires"),
            ("decode", "write_bytes", 1, "codec decode requires"),
            ("decode_rss", "bytes", 1, "codec decode_rss requires"),
        )
        for operation, field, delta, message in cases:
            records = _complete_records()
            target = next(
                record
                for record in records
                if record["suite"] == "codec" and record["operation"] == operation
            )
            target[field] += delta

            outcome, _ = self._generate(records)

            with self.subTest(operation=operation):
                self.assertEqual("INCOMPLETE", outcome.verdict)
                self.assertIn(message, outcome.markdown)

    def test_complete_threshold_regression_is_fail(self) -> None:
        records = _complete_records()
        for record in records:
            if (
                record["suite"] == "codec"
                and record["scenario"] == "S1"
                and record["fragments"] == 1_000
                and record["operation"] == "encode"
                and record["format"] == "lance"
            ):
                record["wall_ns"] = 130_000

        outcome, _ = self._generate(records)

        self.assertEqual("FAIL", outcome.verdict)
        self.assertEqual(1, outcome.exit_code)
        self.assertIn(
            "gate failed, so a mechanism explanation is required", outcome.markdown
        )
        self.assertIn("UNEXPLAINED — mechanism and fix are required", outcome.markdown)

    def test_explanation_is_rendered_without_changing_fail_verdict(self) -> None:
        records = _complete_records()
        for record in records:
            if (
                record["suite"] == "codec"
                and record["scenario"] == "S1"
                and record["fragments"] == 1_000
                and record["operation"] == "encode"
                and record["format"] == "lance"
            ):
                record["wall_ns"] = 130_000
        explanations = _sidecar(
            {
                "selector": _case_selector(
                    suite="codec",
                    scenario="S1",
                    fragments=1_000,
                    storage="memory",
                    operation="encode",
                    metric="wall_ns",
                ),
                "mechanism": "Encoder setup dominates this small manifest.",
                "fix": "Reuse initialized encoder state before the measured path.",
            }
        )

        outcome, _ = self._generate(records, explanations)

        self.assertEqual("FAIL", outcome.verdict)
        self.assertEqual(1, outcome.exit_code)
        self.assertIn(
            "EXPLAINED — Mechanism: Encoder setup dominates this small manifest.",
            outcome.markdown,
        )
        self.assertIn(
            "Fix: Reuse initialized encoder state before the measured path.",
            outcome.markdown,
        )
        self.assertNotIn("UNEXPLAINED", outcome.markdown)
        self.assertIn("--explanations", outcome.markdown)

    def test_conflict_request_medians_must_match_exactly(self) -> None:
        records = _complete_records()
        for record in records:
            if (
                record["suite"] == "e2e"
                and record["scenario"] == "S1"
                and record["storage"] == "ebs"
                and record["fragments"] == 1_000_000
                and record["operation"] == "conflict_retry"
            ):
                record["get_requests"] = (
                    1_000_000_001 if record["format"] == "lance" else 1_000_000_000
                )

        outcome, _ = self._generate(records)

        self.assertEqual("FAIL", outcome.verdict)
        self.assertIn("retry behavior differs", outcome.markdown)

    def test_e2e_request_gate_accepts_stable_explanation_selector(self) -> None:
        records = _complete_records()
        for record in records:
            if (
                record["suite"] == "e2e"
                and record["scenario"] == "S1"
                and record["storage"] == "ebs"
                and record["fragments"] == 1_000_000
                and record["operation"] == "conflict_retry"
            ):
                record["get_requests"] = (
                    1_000_000_001 if record["format"] == "lance" else 1_000_000_000
                )
        explanations = _sidecar(
            {
                "selector": _case_selector(
                    suite="e2e",
                    scenario="S1",
                    fragments=1_000_000,
                    storage="ebs",
                    operation="conflict_retry",
                    metric="request_counts",
                ),
                "mechanism": "The retry performs one extra metadata read.",
                "fix": "Carry the observed version into the retry path.",
            }
        )

        outcome, _ = self._generate(records, explanations)

        self.assertEqual("FAIL", outcome.verdict)
        self.assertIn("EXPLAINED — Mechanism: The retry performs", outcome.markdown)
        self.assertNotIn("UNEXPLAINED", outcome.markdown)

    def test_scaling_gate_accepts_stable_explanation_selector(self) -> None:
        records = _complete_records()
        for record in records:
            if (
                record["suite"] == "codec"
                and record["scenario"] == "S1"
                and record["fragments"] == 10_000_000
                and record["operation"] == "encode"
                and record["format"] == "lance"
            ):
                record["wall_ns"] = 600_000_000
        explanations = _sidecar(
            {
                "selector": _scaling_selector(
                    scenario="S1",
                    format_name="lance",
                    operation="encode",
                    metric="wall_ns",
                ),
                "mechanism": "Fixed work dominates the 1K endpoint.",
                "fix": "Measure and remove repeated fixed initialization.",
            }
        )

        outcome, _ = self._generate(records, explanations)

        self.assertEqual("FAIL", outcome.verdict)
        self.assertIn("EXPLAINED — Mechanism: Fixed work dominates", outcome.markdown)
        self.assertNotIn("UNEXPLAINED", outcome.markdown)

    def test_malformed_explanation_is_rejected(self) -> None:
        malformed = _sidecar(
            {
                "selector": _case_selector(
                    suite="codec",
                    scenario="S1",
                    fragments=1_000,
                    storage="memory",
                    operation="encode",
                    metric="wall_ns",
                ),
                "mechanism": "Missing the required fix field.",
            }
        )

        with self.assertRaisesRegex(REPORT.ExplanationError, "missing field.*fix"):
            self._generate(_complete_records(), malformed)

    def test_duplicate_explanation_selector_is_rejected(self) -> None:
        selector = _case_selector(
            suite="codec",
            scenario="S1",
            fragments=1_000,
            storage="memory",
            operation="encode",
            metric="wall_ns",
        )
        duplicate = _sidecar(
            {"selector": selector, "mechanism": "First.", "fix": "First fix."},
            {"selector": selector, "mechanism": "Second.", "fix": "Second fix."},
        )

        with self.assertRaisesRegex(REPORT.ExplanationError, "duplicate"):
            self._generate(_complete_records(), duplicate)

    def test_unknown_explanation_selector_is_rejected(self) -> None:
        unknown = _sidecar(
            {
                "selector": _case_selector(
                    suite="codec",
                    scenario="S1",
                    fragments=42,
                    storage="memory",
                    operation="encode",
                    metric="wall_ns",
                ),
                "mechanism": "This case is not in the matrix.",
                "fix": "Select an emitted gate row.",
            }
        )

        with self.assertRaisesRegex(REPORT.ExplanationError, "unknown"):
            self._generate(_complete_records(), unknown)

    def test_missing_field_and_round_are_incomplete(self) -> None:
        records = _complete_records()
        target = next(
            record
            for record in records
            if record["suite"] == "codec"
            and record["scenario"] == "S1"
            and record["fragments"] == 1_000
            and record["operation"] == "encode"
            and record["format"] == "lance"
            and record["round"] == 0
        )
        del target["wall_ns"]

        outcome, _ = self._generate(records)

        self.assertEqual("INCOMPLETE", outcome.verdict)
        self.assertEqual(2, outcome.exit_code)
        self.assertIn("missing required field(s): wall_ns", outcome.markdown)
        self.assertIn("has 4 unique rounds; at least 5 required", outcome.markdown)

    def test_wrong_integer_type_is_incomplete_instead_of_crashing(self) -> None:
        records = _complete_records()
        records[0]["fragments"] = "1000"

        outcome, _ = self._generate(records)

        self.assertEqual("INCOMPLETE", outcome.verdict)
        self.assertEqual(2, outcome.exit_code)
        self.assertIn("fragments must be a non-negative integer", outcome.markdown)

    def test_unpaired_round_sets_are_incomplete(self) -> None:
        records = _complete_records()
        target = next(
            record
            for record in records
            if record["suite"] == "codec"
            and record["scenario"] == "S1"
            and record["fragments"] == 1_000
            and record["operation"] == "decode"
            and record["format"] == "lance"
            and record["round"] == 0
        )
        target["round"] = 5

        outcome, _ = self._generate(records)

        self.assertEqual("INCOMPLETE", outcome.verdict)
        self.assertIn("unpaired format rounds", outcome.markdown)

    def test_invalid_utf8_is_incomplete_instead_of_crashing(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "invalid-utf8.jsonl"
        path.write_bytes(b"\xff\xfe\n")

        outcome = REPORT.generate_report([path], COMMIT)

        self.assertEqual("INCOMPLETE", outcome.verdict)
        self.assertEqual(2, outcome.exit_code)
        self.assertIn("invalid UTF-8", outcome.markdown)


if __name__ == "__main__":
    unittest.main()
