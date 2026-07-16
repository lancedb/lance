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
RELEASE_SCRIPT = SCRIPT_DIR / "release_remote.sh"
RELEASE_SYSTEMD_SCRIPT = SCRIPT_DIR / "release_remote_systemd.sh"


class ReleaseRemoteTests(unittest.TestCase):
    def run_release(
        self,
        *,
        protocol_status: int,
        report_status: int,
        aggregate_status: int,
        aggregate_verdict: str,
    ) -> tuple[subprocess.CompletedProcess[str], Path, str]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        repository = root / "repository"
        result_root = root / "results"
        bin_dir = root / "bin"
        repository.mkdir()
        bin_dir.mkdir()
        matrix = repository / "benchmarks/stable_row_address/workload_matrix.v2.json"
        matrix.parent.mkdir(parents=True)
        matrix.write_text('{"release_contract":{"shard_count":5}}', encoding="utf-8")

        subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
        subprocess.run(["git", "add", "."], cwd=repository, check=True)
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

        for name, body in {
            "aws": "#!/bin/sh\nprintf 'us-east-2\\n'\n",
            "flock": "#!/bin/sh\nexit 0\n",
        }.items():
            executable = bin_dir / name
            executable.write_text(body, encoding="utf-8")
            executable.chmod(0o755)

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
                    raise SystemExit(int(os.environ["FAKE_PROTOCOL_STATUS"]))
                if arguments[0].endswith("protocol_report.py"):
                    raise SystemExit(int(os.environ["FAKE_REPORT_STATUS"]))
                if arguments[0].endswith("protocol_aggregate.py"):
                    output = pathlib.Path(arguments[arguments.index("--json") + 1])
                    output.write_text(
                        json.dumps(
                            {
                                "commit": os.environ["EXPECTED_COMMIT"],
                                "verdict": os.environ["FAKE_AGGREGATE_VERDICT"],
                            }
                        ),
                        encoding="utf-8",
                    )
                    raise SystemExit(int(os.environ["FAKE_AGGREGATE_STATUS"]))
                if arguments[0] == "-c":
                    if 'release_contract' in arguments[1] and 'shard_count' in arguments[1]:
                        print("5")
                    elif '["commit"]' in arguments[1]:
                        print(os.environ["EXPECTED_COMMIT"])
                    elif '["verdict"]' in arguments[1]:
                        print(os.environ["FAKE_AGGREGATE_VERDICT"])
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
                "AWS_REGION": "us-east-2",
                "DATASET_ROOT": "s3://benchmark-bucket/release-test",
                "EXPECTED_COMMIT": commit,
                "FAKE_AGGREGATE_STATUS": str(aggregate_status),
                "FAKE_AGGREGATE_VERDICT": aggregate_verdict,
                "FAKE_PROTOCOL_STATUS": str(protocol_status),
                "FAKE_REPORT_STATUS": str(report_status),
                "PATH": f"{bin_dir}:{environment['PATH']}",
                "PYTHON": str(fake_python),
                "RESULT_ROOT": str(result_root),
            }
        )
        result = subprocess.run(
            [str(RELEASE_SCRIPT), "release-all"],
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
        )
        return result, result_root, commit

    def test_systemd_service_does_not_restart_failed_release(self) -> None:
        script = RELEASE_SYSTEMD_SCRIPT.read_text(encoding="utf-8")

        self.assertNotIn("Restart=", script)

    def test_complete_correctness_failure_stops_restart_loop(self) -> None:
        result, result_root, commit = self.run_release(
            protocol_status=1,
            report_status=1,
            aggregate_status=1,
            aggregate_verdict="FAIL",
        )

        self.assertEqual(result.returncode, 65, result.stderr)
        self.assertEqual(
            (result_root / "stable-row-address-release.execution-complete").read_text(
                encoding="utf-8"
            ),
            f"{commit}\n",
        )
        self.assertFalse((result_root / "stable-row-address-release.pass").exists())

    def test_failed_execution_cannot_leave_pass_from_false_positive_report(
        self,
    ) -> None:
        result, result_root, _ = self.run_release(
            protocol_status=1,
            report_status=0,
            aggregate_status=0,
            aggregate_verdict="PASS",
        )

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse((result_root / "stable-row-address-release.pass").exists())

    def test_v23_runtime_regression_stops_remaining_shards(self) -> None:
        result, result_root, _ = self.run_release(
            protocol_status=75,
            report_status=0,
            aggregate_status=0,
            aggregate_verdict="PASS",
        )

        self.assertEqual(result.returncode, 75, result.stderr)
        self.assertIn("starting shard-000-of-005", result.stderr)
        self.assertIn("exceeded the v2.3 fail-fast runtime budget", result.stderr)
        self.assertNotIn("starting shard-001-of-005", result.stderr)
        self.assertFalse(
            (result_root / "stable-row-address-release.execution-complete").exists()
        )
        self.assertFalse((result_root / "stable-row-address-release.pass").exists())

    def test_absolute_runtime_timeout_stops_remaining_shards(self) -> None:
        result, result_root, _ = self.run_release(
            protocol_status=74,
            report_status=0,
            aggregate_status=0,
            aggregate_verdict="PASS",
        )

        self.assertEqual(result.returncode, 74, result.stderr)
        self.assertIn("starting shard-000-of-005", result.stderr)
        self.assertIn("exceeded the absolute operation runtime budget", result.stderr)
        self.assertNotIn("starting shard-001-of-005", result.stderr)
        self.assertFalse(
            (result_root / "stable-row-address-release.execution-complete").exists()
        )
        self.assertFalse((result_root / "stable-row-address-release.pass").exists())

    def test_successful_execution_owns_pass_marker_creation(self) -> None:
        result, result_root, commit = self.run_release(
            protocol_status=0,
            report_status=0,
            aggregate_status=0,
            aggregate_verdict="PASS",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("starting shard-004-of-005", result.stderr)
        self.assertNotIn("starting shard-005", result.stderr)
        expected_marker = f"commit={commit}\naggregate_sha256={'0' * 64}\n"
        self.assertEqual(
            (result_root / "stable-row-address-release.pass").read_text(
                encoding="utf-8"
            ),
            expected_marker,
        )


if __name__ == "__main__":
    unittest.main()
