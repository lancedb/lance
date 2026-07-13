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


class ReleaseRemoteTests(unittest.TestCase):
    def test_complete_correctness_failure_stops_restart_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
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
                        raise SystemExit(1)
                    if arguments[0].endswith("protocol_report.py"):
                        raise SystemExit(1)
                    if arguments[0].endswith("protocol_aggregate.py"):
                        output = pathlib.Path(arguments[arguments.index("--json") + 1])
                        output.write_text(
                            json.dumps(
                                {
                                    "commit": os.environ["EXPECTED_COMMIT"],
                                    "verdict": "FAIL",
                                }
                            ),
                            encoding="utf-8",
                        )
                        raise SystemExit(1)
                    if arguments[0] == "-c":
                        if '["commit"]' in arguments[1]:
                            print(os.environ["EXPECTED_COMMIT"])
                        elif '["verdict"]' in arguments[1]:
                            print("FAIL")
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

            self.assertEqual(result.returncode, 65, result.stderr)
            self.assertEqual(
                (
                    result_root / "stable-row-address-release.execution-complete"
                ).read_text(encoding="utf-8"),
                f"{commit}\n",
            )
            self.assertFalse((result_root / "stable-row-address-release.pass").exists())


if __name__ == "__main__":
    unittest.main()
