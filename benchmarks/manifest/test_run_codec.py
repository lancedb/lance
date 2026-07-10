"""Tests for the process-isolated manifest codec benchmark runner."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


RUNNER_PATH = Path(__file__).with_name("run_codec.py")
SPEC = importlib.util.spec_from_file_location("manifest_run_codec", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)
COMMIT = "0123456789abcdef0123456789abcdef01234567"


class SourceRevisionTest(unittest.TestCase):
    def test_clean_revision_is_head(self) -> None:
        with (
            mock.patch.object(RUNNER, "_run_git", return_value=COMMIT) as run_git,
            mock.patch.object(RUNNER, "_git_bytes", return_value=b"") as git_bytes,
        ):
            self.assertEqual(COMMIT, RUNNER._source_revision())
        run_git.assert_called_once_with("rev-parse", "--verify", "HEAD^{commit}")
        git_bytes.assert_called_once_with(
            "status", "--porcelain=v1", "-z", "--untracked-files=all"
        )

    def test_dirty_revision_is_rejected(self) -> None:
        with (
            mock.patch.object(RUNNER, "_run_git", return_value=COMMIT),
            mock.patch.object(RUNNER, "_git_bytes", return_value=b" M source.rs\0"),
            self.assertRaisesRegex(RuntimeError, "clean Git worktree"),
        ):
            RUNNER._source_revision()

    def test_non_full_head_is_rejected(self) -> None:
        with (
            mock.patch.object(RUNNER, "_run_git", return_value="deadbeef"),
            self.assertRaisesRegex(RuntimeError, "full lowercase SHA"),
        ):
            RUNNER._source_revision()

    def test_provenance_bypass_arguments_are_rejected(self) -> None:
        for flag in ("--commit", "--executable"):
            with (
                self.subTest(flag=flag),
                mock.patch("sys.stderr"),
                self.assertRaises(SystemExit),
            ):
                RUNNER._parser().parse_args(["--output", "result.jsonl", flag, "value"])


class RecordValidationTest(unittest.TestCase):
    def base_record(self, operation: str, warmup: str) -> dict[str, object]:
        return {
            "schema_version": 2,
            "suite": "codec",
            "scenario": "S1",
            "fragments": 1_000,
            "format": "lance",
            "storage": "memory",
            "operation": operation,
            "round": 0,
            "wall_ns": 1,
            "bytes": 2,
            "peak_rss_bytes": 3,
            "get_requests": 0,
            "put_requests": 0,
            "read_bytes": 2,
            "write_bytes": 0,
            "status": "success",
            "error": None,
            "commit": COMMIT,
            "seed": 7,
            "host": "host",
            "warmup": warmup,
        }

    def test_split_worker_records_are_accepted(self) -> None:
        for operation, warmup in (
            ("encode", "tiny"),
            ("decode", "tiny"),
            ("decode_rss", "cold"),
        ):
            RUNNER._validate_records(
                [self.base_record(operation, warmup)],
                expected_operations=(operation,),
                warmup=warmup,
                scenario="S1",
                fragments=1_000,
                format_name="lance",
                round_number=0,
                seed=7,
                commit=COMMIT,
                host="host",
            )

    def test_wrong_worker_warmup_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "warmup"):
            RUNNER._validate_records(
                [self.base_record("decode_rss", "tiny")],
                expected_operations=("decode_rss",),
                warmup="cold",
                scenario="S1",
                fragments=1_000,
                format_name="lance",
                round_number=0,
                seed=7,
                commit=COMMIT,
                host="host",
            )

    def test_worker_provenance_must_match_job(self) -> None:
        for field, value in (
            ("commit", "f" * 40),
            ("seed", 8),
            ("host", "other-host"),
        ):
            record = self.base_record("decode", "tiny")
            record[field] = value
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(RuntimeError, "dimensions do not match"),
            ):
                RUNNER._validate_records(
                    [record],
                    expected_operations=("decode",),
                    warmup="tiny",
                    scenario="S1",
                    fragments=1_000,
                    format_name="lance",
                    round_number=0,
                    seed=7,
                    commit=COMMIT,
                    host="host",
                )


if __name__ == "__main__":
    unittest.main()
