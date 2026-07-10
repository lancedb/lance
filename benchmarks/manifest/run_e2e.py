#!/usr/bin/env python3
"""Build and run the manifest E2E benchmark with verified provenance."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from run_codec import REPOSITORY_ROOT, _source_revision


def _build_harness() -> Path:
    command = (
        "cargo",
        "build",
        "--profile",
        "release-with-debug",
        "--package",
        "lance",
        "--features",
        "metrics",
        "--bench",
        "manifest_e2e",
        "--message-format=json-render-diagnostics",
    )
    print("Building manifest_e2e with release-with-debug", file=sys.stderr)
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
    )
    executable: Path | None = None
    rendered_diagnostics: list[str] = []
    for line in result.stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("reason") == "compiler-message":
            rendered = message.get("message", {}).get("rendered")
            if rendered:
                rendered_diagnostics.append(rendered)
        target = message.get("target", {})
        if (
            message.get("reason") == "compiler-artifact"
            and target.get("name") == "manifest_e2e"
            and message.get("executable")
        ):
            executable = Path(message["executable"]).resolve()
    if result.returncode != 0:
        for diagnostic in rendered_diagnostics:
            print(diagnostic, file=sys.stderr, end="")
        raise RuntimeError(f"cargo build failed with exit status {result.returncode}")
    if executable is None or not executable.is_file():
        raise RuntimeError("cargo did not report the manifest_e2e executable")
    return executable


def _has_commit_argument(arguments: Sequence[str]) -> bool:
    return any(
        argument == "--commit" or argument.startswith("--commit=")
        for argument in arguments
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if _has_commit_argument(arguments):
        raise RuntimeError(
            "--commit is managed by run_e2e.py and cannot be supplied by the caller"
        )
    source_revision_before_build = _source_revision()
    executable = _build_harness()
    source_revision_after_build = _source_revision()
    if source_revision_after_build != source_revision_before_build:
        raise RuntimeError(
            "source revision changed while building manifest_e2e: "
            f"before={source_revision_before_build}, "
            f"after={source_revision_after_build}"
        )
    arguments.extend(("--commit", source_revision_after_build))
    return subprocess.run((str(executable), *arguments), cwd=REPOSITORY_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
