"""Tests for the process-isolated manifest E2E benchmark runner."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run_e2e as RUNNER


COMMIT = "0123456789abcdef0123456789abcdef01234567"


class E2ERunnerTest(unittest.TestCase):
    def test_explicit_commit_detection(self) -> None:
        self.assertTrue(RUNNER._has_commit_argument(("--commit", COMMIT)))
        self.assertTrue(RUNNER._has_commit_argument((f"--commit={COMMIT}",)))
        self.assertFalse(RUNNER._has_commit_argument(("--output", "results.jsonl")))

    def test_explicit_commit_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "cannot be supplied"):
            RUNNER.main(("--commit", COMMIT))

    def test_clean_head_is_always_injected(self) -> None:
        completed = SimpleNamespace(returncode=0)
        with (
            mock.patch.object(RUNNER, "_source_revision", side_effect=[COMMIT, COMMIT]),
            mock.patch.object(RUNNER, "_build_harness", return_value=Path("/tmp/e2e")),
            mock.patch.object(RUNNER.subprocess, "run", return_value=completed) as run,
        ):
            self.assertEqual(0, RUNNER.main(("--output", "/tmp/result.jsonl")))

        command = run.call_args.args[0]
        self.assertEqual(("--commit", COMMIT), command[-2:])


if __name__ == "__main__":
    unittest.main()
