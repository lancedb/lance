# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import argparse
import json

import lance
from common import (
    DEFAULT_BASELINE_BRANCH,
    DEFAULT_DATASET_URI,
    DEFAULT_OPTIMIZED_BRANCH,
    storage_options_for,
)


def prepare_branches(args: argparse.Namespace) -> None:
    storage_options = storage_options_for(args.dataset_uri)
    dataset = lance.dataset(args.dataset_uri, storage_options=storage_options)
    existing = dataset.branches.list()
    requested = (args.baseline_branch, args.optimized_branch)
    if len(set(requested)) != len(requested):
        raise ValueError("Baseline and optimized branch names must be different")
    collisions = [branch for branch in requested if branch in existing]
    if collisions:
        raise ValueError(
            "Refusing to reuse existing Lance branches: " + ", ".join(collisions)
        )

    requested_reference = (args.source_branch, args.source_version)
    source = dataset.checkout_version(requested_reference)
    source_reference = (args.source_branch, source.version)
    if source.describe_indices() and not args.allow_source_indices:
        raise ValueError(
            "The source reference already has indices. Use an unindexed source "
            "reference, or pass --allow-source-indices after verifying isolation."
        )

    created = []
    for branch in requested:
        branch_dataset = dataset.create_branch(branch, reference=source_reference)
        created.append(
            {
                "branch": branch,
                "version": branch_dataset.version,
                "rows": branch_dataset.count_rows(),
            }
        )
    print(
        json.dumps({"source_reference": source_reference, "created": created}, indent=2)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the two prerequisite Lance branches for the benchmark."
    )
    parser.add_argument("--dataset-uri", default=DEFAULT_DATASET_URI)
    parser.add_argument("--baseline-branch", default=DEFAULT_BASELINE_BRANCH)
    parser.add_argument("--optimized-branch", default=DEFAULT_OPTIMIZED_BRANCH)
    parser.add_argument("--source-branch", default="main")
    parser.add_argument(
        "--source-version",
        type=int,
        help="Source branch version; defaults to its latest version.",
    )
    parser.add_argument("--allow-source-indices", action="store_true")
    return parser


if __name__ == "__main__":
    prepare_branches(build_parser().parse_args())
