#!/usr/bin/env python3
"""Aggregate independently resumable stable-row-address protocol shards."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import protocol  # noqa: E402
import protocol_report  # noqa: E402


def aggregate(
    inputs: Sequence[Path],
    *,
    bootstrap_samples: int,
    expected_commit: str | None = None,
    execution_marker: Path | None = None,
) -> tuple[int, str, dict[str, Any]]:
    if not inputs:
        raise ValueError("at least one shard input is required")
    if (expected_commit is None) != (execution_marker is None):
        raise ValueError(
            "expected_commit and execution_marker must be provided together"
        )
    if (
        expected_commit is not None
        and re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None
    ):
        raise ValueError("expected_commit must be a lowercase full Git SHA")
    shard_results = []
    issues: list[str] = []
    if execution_marker is not None:
        try:
            marker_lines = execution_marker.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            issues.append(f"execution-complete marker is unavailable: {error}")
        else:
            if marker_lines != [expected_commit]:
                issues.append(
                    "execution-complete marker does not contain exactly the expected commit"
                )
    for input_path in inputs:
        sidecar, records, load_issues = protocol_report.load_evidence(input_path)
        result = protocol_report.analyze(
            sidecar,
            records,
            bootstrap_samples=bootstrap_samples,
            initial_issues=load_issues,
            enforce_gates=sidecar.get("profile") == "release",
        )
        shard_results.append((input_path, sidecar, result))

    sidecars = [sidecar for _, sidecar, _ in shard_results if sidecar]
    if len(sidecars) != len(shard_results):
        issues.append("one or more shard sidecars are unavailable")
    if sidecars:
        common_fields = (
            "commit",
            "source_provenance",
            "development_tiny",
            "host",
            "profile",
            "cargo_profile",
            "storage",
            "seed",
            "matrix_sha256",
            "policy_sha256",
            "shard_count",
            "shard_strategy",
        )
        for field in common_fields:
            values = {sidecar[field] for sidecar in sidecars}
            if len(values) != 1:
                issues.append(f"shards disagree on {field}: {sorted(values)}")
        if expected_commit is not None and any(
            sidecar["commit"] != expected_commit for sidecar in sidecars
        ):
            issues.append(
                "one or more shard sidecars do not match the execution-complete commit"
            )
        attestation_values = {
            json.dumps(
                sidecar["storage_region_attestation"],
                sort_keys=True,
                separators=(",", ":"),
            )
            for sidecar in sidecars
        }
        if len(attestation_values) != 1:
            issues.append("shards disagree on storage_region_attestation")
        base_dataset_roots: list[str] = []
        dataset_roots: list[str] = []
        for sidecar in sidecars:
            try:
                base_dataset_roots.append(
                    protocol_report.canonical_dataset_root(
                        sidecar["base_dataset_root"], sidecar["storage"]
                    )
                )
                dataset_roots.append(
                    protocol_report.validate_dataset_root_binding(sidecar)
                )
            except (KeyError, TypeError, ValueError) as error:
                issues.append(f"{sidecar.get('shard_id', '<unknown shard>')}: {error}")
        if len(set(base_dataset_roots)) != 1:
            issues.append(
                "shards disagree on base_dataset_root: "
                f"{sorted(set(base_dataset_roots))}"
            )
        expected_count = sidecars[0]["shard_count"]
        indices = [sidecar["shard_index"] for sidecar in sidecars]
        if Counter(indices) != Counter(range(expected_count)):
            issues.append(
                f"expected shard indices 0..{expected_count - 1}, found {sorted(indices)}"
            )
        if len(set(dataset_roots)) != len(dataset_roots):
            issues.append("shards do not use independent dataset prefixes")

        matrix_cases = [
            case for sidecar in sidecars for case in sidecar["matrix_case_names"]
        ]
        duplicates = sorted(
            case for case, count in Counter(matrix_cases).items() if count != 1
        )
        if duplicates:
            issues.append(f"matrix cases appear in multiple shards: {duplicates}")

        if sidecars[0]["profile"] == "release":
            matrix = sidecars[0]["matrix"]
            profile = matrix["profiles"]["release"]
            expected_cases = {
                case.name
                for case in protocol.iter_matrix_cases(
                    profile, set(matrix["tracks"]["matrix"]["cases"])
                )
            }
            actual_cases = set(matrix_cases)
            missing_cases = sorted(expected_cases - actual_cases)
            extra_cases = sorted(actual_cases - expected_cases)
            if missing_cases or extra_cases:
                issues.append(
                    "release matrix shard union mismatch: "
                    f"missing={missing_cases}, extra={extra_cases}"
                )
            repeated_units = {
                (track, variant)
                for sidecar in sidecars
                for track in sidecar["tracks"]
                if track
                in {
                    "sustained",
                    "adversarial_natural",
                    "adversarial_aligned",
                }
                for variant in sidecar["variants"]
            }
            expected_units = {
                (track, variant)
                for track in (
                    "sustained",
                    "adversarial_natural",
                    "adversarial_aligned",
                )
                for variant in ("bare", "scalar", "vector")
            }
            if repeated_units != expected_units:
                issues.append(
                    "release repeated-track shard union mismatch: "
                    f"missing={sorted(expected_units - repeated_units)}, "
                    f"extra={sorted(repeated_units - expected_units)}"
                )

    shard_verdicts = [result.verdict for _, _, result in shard_results]
    if issues or "INCOMPLETE" in shard_verdicts:
        verdict = "INCOMPLETE"
        exit_code = 2
    elif "FAIL" in shard_verdicts:
        verdict = "FAIL"
        exit_code = 1
    else:
        verdict = "PASS"
        exit_code = 0
    projections = {
        field: sum(sidecar[field] for sidecar in sidecars)
        for field in (
            "projected_canonical_payload_bytes",
            "projected_unique_initial_index_payload_bytes_lower_bound",
            "projected_no_dedup_logical_data_payload_bytes",
            "projected_no_dedup_logical_index_payload_bytes",
            "projected_minimum_full_scan_payload_bytes",
        )
    }
    machine = {
        "schema_version": 1,
        "suite": "stable_row_address_design_protocol_aggregate_report",
        "commit": sidecars[0]["commit"] if sidecars else None,
        "verdict": verdict,
        "bootstrap_samples": bootstrap_samples,
        "issues": issues,
        "storage_projections": projections,
        "shards": [
            {
                "input": str(input_path),
                "run_id": sidecar.get("run_id") if sidecar else None,
                "shard_id": sidecar.get("shard_id") if sidecar else None,
                "report": result.machine,
            }
            for input_path, sidecar, result in shard_results
        ],
    }
    lines = [
        "# Stable Logical Row Address Aggregate Protocol Report",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"Shards: {len(shard_results)}; bootstrap resamples per shard: {bootstrap_samples}",
        "",
        "| Shard | Run | Verdict | Records | Complete pairs |",
        "|---|---|---|---:|---:|",
    ]
    for input_path, sidecar, result in shard_results:
        lines.append(
            f"| {sidecar.get('shard_id', input_path.name) if sidecar else input_path.name} | "
            f"{sidecar.get('run_id', '—') if sidecar else '—'} | {result.verdict} | "
            f"{result.machine['records']} | {result.machine['complete_pairs']} |"
        )
    lines.extend(("", "## Aggregate projections", ""))
    lines.extend(f"- {field}: {value} bytes" for field, value in projections.items())
    if issues:
        lines.extend(("", "## Incomplete shard evidence", ""))
        lines.extend(f"- {issue}" for issue in issues)
    failed = [
        (sidecar.get("shard_id", str(path)), result.verdict)
        for path, sidecar, result in shard_results
        if result.verdict != "PASS"
    ]
    if failed:
        lines.extend(("", "## Non-passing shards", ""))
        lines.extend(f"- {shard}: {shard_verdict}" for shard, shard_verdict in failed)
    return exit_code, "\n".join(lines) + "\n", machine


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=protocol_report.BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--expected-commit")
    parser.add_argument("--execution-marker", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.bootstrap_samples < protocol_report.BOOTSTRAP_SAMPLES:
        raise ValueError(
            f"aggregate reports require at least {protocol_report.BOOTSTRAP_SAMPLES} bootstrap samples"
        )
    exit_code, markdown, machine = aggregate(
        args.inputs,
        bootstrap_samples=args.bootstrap_samples,
        expected_commit=args.expected_commit,
        execution_marker=args.execution_marker,
    )
    protocol.replace_text_atomic(args.markdown, markdown)
    protocol.replace_text_atomic(
        args.json,
        json.dumps(machine, sort_keys=True, separators=(",", ":")) + "\n",
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
