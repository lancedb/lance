#!/usr/bin/env python3
"""Build and run the process-isolated manifest codec benchmark matrix."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FRAGMENT_SIZES = (1_000, 100_000, 1_000_000, 10_000_000)
SCENARIOS = ("S1", "S2")
FORMATS = ("protobuf", "lance")
OPERATIONS = ("encode", "decode", "size", "decode_rss")
REQUIRED_FIELDS = frozenset(
    (
        "schema_version",
        "suite",
        "scenario",
        "fragments",
        "format",
        "storage",
        "operation",
        "round",
        "wall_ns",
        "bytes",
        "peak_rss_bytes",
        "get_requests",
        "put_requests",
        "read_bytes",
        "write_bytes",
        "status",
        "error",
        "commit",
        "seed",
        "host",
    )
)
DEFAULT_SEED = 0x4C414E43455F4D46
SCHEMA_VERSION = 2
GIT_SHA_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def _positive_integer(value: str) -> int:
    parsed = int(value, 0)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _non_negative_integer(value: str) -> int:
    parsed = int(value, 0)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _run_git(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def _git_bytes(*arguments: str) -> bytes:
    result = subprocess.run(
        ("git", *arguments),
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout


def _source_revision() -> str:
    head = _run_git("rev-parse", "--verify", "HEAD^{commit}")
    if GIT_SHA_PATTERN.fullmatch(head) is None:
        raise RuntimeError(f"git HEAD is not a full lowercase SHA: {head!r}")
    status = _git_bytes("status", "--porcelain=v1", "-z", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "manifest gate benchmarks require a clean Git worktree; "
            "commit or remove tracked and untracked changes before running"
        )
    return head


def _build_harness() -> Path:
    command = (
        "cargo",
        "build",
        "--profile",
        "release-with-debug",
        "--package",
        "lance-table",
        "--bench",
        "manifest_codec",
        "--message-format=json-render-diagnostics",
    )
    print("Building manifest_codec with release-with-debug", file=sys.stderr)
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
            and target.get("name") == "manifest_codec"
            and message.get("executable")
        ):
            executable = Path(message["executable"]).resolve()
    if result.returncode != 0:
        for diagnostic in rendered_diagnostics:
            print(diagnostic, file=sys.stderr, end="")
        raise RuntimeError(f"cargo build failed with exit status {result.returncode}")
    if executable is None or not executable.is_file():
        raise RuntimeError("cargo did not report the manifest_codec executable")
    return executable


def _validate_records(
    records: Sequence[dict[str, Any]],
    *,
    expected_operations: tuple[str, ...],
    warmup: str,
    scenario: str,
    fragments: int,
    format_name: str,
    round_number: int,
    seed: int,
    commit: str,
    host: str,
) -> None:
    operations = [record.get("operation") for record in records]
    if tuple(operations) != expected_operations:
        raise RuntimeError(
            f"worker returned operations {operations!r}; expected {expected_operations!r}"
        )
    for record in records:
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise RuntimeError(
                f"worker record is missing fields: {', '.join(sorted(missing))}"
            )
        expected = {
            "schema_version": SCHEMA_VERSION,
            "suite": "codec",
            "scenario": scenario,
            "fragments": fragments,
            "format": format_name,
            "storage": "memory",
            "round": round_number,
            "seed": seed,
            "commit": commit,
            "host": host,
            "status": "success",
            "error": None,
            "warmup": warmup,
        }
        mismatches = {
            field: (expected_value, record.get(field))
            for field, expected_value in expected.items()
            if record.get(field) != expected_value
        }
        if mismatches:
            raise RuntimeError(f"worker record dimensions do not match: {mismatches}")


def _run_worker(
    executable: Path,
    *,
    mode: str,
    warmup: str,
    fixture: Path,
    scenario: str,
    fragments: int,
    format_name: str,
    round_number: int,
    seed: int,
    commit: str,
    host: str,
) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    command = (
        str(executable),
        "--mode",
        mode,
        "--warmup",
        warmup,
        "--fixture",
        str(fixture),
        "--scenario",
        scenario,
        "--fragments",
        str(fragments),
        "--format",
        format_name,
        "--round",
        str(round_number),
        "--seed",
        str(seed),
        "--commit",
        commit,
        "--host",
        host,
    )
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        raise RuntimeError(
            f"manifest_codec exited with status {result.returncode} for "
            f"{scenario}/{fragments}/{format_name}/round-{round_number}"
        )
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(result.stdout.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"worker emitted invalid JSON on stdout line {line_number}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise RuntimeError(f"worker stdout line {line_number} is not a JSON object")
        records.append(record)
    _validate_records(
        records,
        expected_operations={
            "encode": ("encode", "size"),
            "decode": ("decode",),
            "rss": ("decode_rss",),
        }[mode],
        warmup=warmup,
        scenario=scenario,
        fragments=fragments,
        format_name=format_name,
        round_number=round_number,
        seed=seed,
        commit=commit,
        host=host,
    )
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sizes", nargs="+", type=_positive_integer, default=list(FRAGMENT_SIZES)
    )
    parser.add_argument(
        "--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS)
    )
    parser.add_argument("--rounds", type=_positive_integer, default=5)
    parser.add_argument("--seed", type=_non_negative_integer, default=DEFAULT_SEED)
    parser.add_argument("--host", help="host identity; defaults to socket hostname")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run S1/S2 at 1K for one round; output is intentionally incomplete",
    )
    parser.add_argument(
        "--cold",
        action="store_true",
        help="disable tiny warm-up for diagnostic cold/warm comparisons",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.smoke:
        args.sizes = [1_000]
        args.scenarios = list(SCENARIOS)
        args.rounds = 1
    elif args.rounds < 5:
        raise SystemExit("gate runs require --rounds >= 5 (use --smoke for one round)")
    unsupported_sizes = sorted(set(args.sizes) - set(FRAGMENT_SIZES))
    if unsupported_sizes and not args.smoke:
        raise SystemExit(
            "gate runs only support sizes "
            f"{', '.join(map(str, FRAGMENT_SIZES))}; found {unsupported_sizes}"
        )

    source_revision_before_build = _source_revision()
    executable = _build_harness()
    if not executable.is_file():
        raise SystemExit(f"benchmark executable does not exist: {executable}")
    commit = source_revision_before_build
    source_revision_after_build = _source_revision()
    if source_revision_after_build != source_revision_before_build:
        raise SystemExit(
            "source changed while the benchmark harness was being built; rerun "
            "to produce traceable results"
        )
    host = args.host or socket.gethostname()
    if not commit or not host:
        raise SystemExit("commit and host must not be empty")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.partial")
    if partial.exists():
        partial.unlink()

    total_samples = len(args.scenarios) * len(args.sizes) * len(FORMATS) * args.rounds
    warmup = "cold" if args.cold else "tiny"
    completed = 0
    try:
        with partial.open("x", encoding="utf-8") as handle:
            for scenario in args.scenarios:
                for fragments in args.sizes:
                    for format_name in FORMATS:
                        for round_number in range(args.rounds):
                            completed += 1
                            print(
                                f"[{completed}/{total_samples}] {scenario} "
                                f"{fragments} {format_name} round {round_number}",
                                file=sys.stderr,
                            )
                            with tempfile.TemporaryDirectory(
                                prefix="lance-manifest-codec-"
                            ) as temporary_directory:
                                fixture = Path(temporary_directory) / "manifest.bin"
                                encode_records = _run_worker(
                                    executable,
                                    mode="encode",
                                    warmup=warmup,
                                    fixture=fixture,
                                    scenario=scenario,
                                    fragments=fragments,
                                    format_name=format_name,
                                    round_number=round_number,
                                    seed=args.seed,
                                    commit=commit,
                                    host=host,
                                )
                                decode_records = _run_worker(
                                    executable,
                                    mode="decode",
                                    warmup=warmup,
                                    fixture=fixture,
                                    scenario=scenario,
                                    fragments=fragments,
                                    format_name=format_name,
                                    round_number=round_number,
                                    seed=args.seed,
                                    commit=commit,
                                    host=host,
                                )
                                rss_records = _run_worker(
                                    executable,
                                    mode="rss",
                                    warmup="cold",
                                    fixture=fixture,
                                    scenario=scenario,
                                    fragments=fragments,
                                    format_name=format_name,
                                    round_number=round_number,
                                    seed=args.seed,
                                    commit=commit,
                                    host=host,
                                )
                            records_by_operation = {
                                record["operation"]: record
                                for record in encode_records
                                + decode_records
                                + rss_records
                            }
                            records = [
                                records_by_operation[operation]
                                for operation in OPERATIONS
                            ]
                            for record in records:
                                handle.write(
                                    json.dumps(
                                        record, sort_keys=True, separators=(",", ":")
                                    )
                                    + "\n"
                                )
                            handle.flush()
                            os.fsync(handle.fileno())
    except Exception:
        print(f"Partial results retained at {partial}", file=sys.stderr)
        raise
    partial.replace(output)
    print(f"Wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
