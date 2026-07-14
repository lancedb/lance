#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SMOKE_SCRIPT = SCRIPT_DIR / "smoke_remote.sh"
SYSTEMD_SCRIPT = SCRIPT_DIR / "smoke_remote_systemd.sh"


class SmokeRemoteTests(unittest.TestCase):
    def run_failed_smoke(
        self, report_verdict: str
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        repository = root / "repository"
        dataset_root = root / "dataset"
        result_root = root / "results"
        bin_dir = root / "bin"
        repository.mkdir()
        bin_dir.mkdir()

        subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Benchmark Test",
                "-c",
                "user.email=benchmark@example.com",
                "commit",
                "--quiet",
                "--allow-empty",
                "-m",
                "initial",
            ],
            cwd=repository,
            check=True,
        )
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        fake_python = bin_dir / "python"
        fake_python.write_text(
            f"#!{sys.executable}\n"
            + textwrap.dedent(
                """
                import json
                import os
                import pathlib
                import sys

                arguments = sys.argv[1:]
                if arguments[0].endswith("protocol.py"):
                    raise SystemExit(1)
                if arguments[0].endswith("protocol_report.py"):
                    output = pathlib.Path(arguments[arguments.index("--json") + 1])
                    verdict = os.environ["FAKE_REPORT_VERDICT"]
                    output.write_text(
                        json.dumps(
                            {
                                "commit": os.environ["EXPECTED_COMMIT"],
                                "verdict": verdict,
                            }
                        ),
                        encoding="utf-8",
                    )
                    raise SystemExit(0 if verdict == "PASS" else 1)
                if arguments[0] == "-c":
                    if '["commit"]' in arguments[1]:
                        print(os.environ["EXPECTED_COMMIT"])
                    elif '["verdict"]' in arguments[1]:
                        print(os.environ["FAKE_REPORT_VERDICT"])
                    else:
                        print("0" * 64)
                    raise SystemExit(0)
                raise SystemExit(2)
                """
            ),
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        environment = os.environ.copy()
        environment.update(
            {
                "DATASET_ROOT": str(dataset_root),
                "EXPECTED_COMMIT": commit,
                "FAKE_REPORT_VERDICT": report_verdict,
                "PYTHON": str(fake_python),
                "RESULT_ROOT": str(result_root),
            }
        )
        result = subprocess.run(
            [str(SMOKE_SCRIPT), "smoke-all"],
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
        )
        return result, result_root

    def test_correctness_failure_stops_without_pass_marker(self) -> None:
        result, result_root = self.run_failed_smoke("FAIL")

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse((result_root / "stable-row-address-smoke.pass").exists())
        self.assertFalse(
            (result_root / "stable-row-address-smoke.execution-complete").exists()
        )

    def test_stale_pass_report_requires_execution_marker(self) -> None:
        result, result_root = self.run_failed_smoke("PASS")

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("execution-complete marker is missing", result.stderr)
        self.assertFalse((result_root / "stable-row-address-smoke.pass").exists())

    def test_systemd_service_waits_for_dataset_mount_and_does_not_restart_failures(
        self,
    ) -> None:
        script = SYSTEMD_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('"${repository_root}" "${DATASET_ROOT}" "${RESULT_ROOT}"', script)
        self.assertNotIn("Restart=", script)


if __name__ == "__main__":
    unittest.main()
