#!/usr/bin/env python3

"""Validate resolved page layouts emitted by the Rust metadata diagnostic."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


CASES = (
    "v2_0_default",
    "v2_1_default",
    "v2_3_default",
    "v2_3_miniblock",
    "v2_3_sparse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attrs", type=int, required=True)
    return parser.parse_args()


def write_result(path: Path, result: dict[str, object]) -> str:
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    assert json.loads(temporary.read_text()) == result
    temporary.replace(path)
    return payload


def main() -> None:
    args = parse_args()
    summary = {}
    for name in CASES:
        resolved = json.loads((args.root / f"layout-{name}.json").read_text())
        counts: Counter[str] = Counter()
        for column in resolved["columns"]:
            counts.update(column["layout_counts"])
        summary[name] = {
            "file_version": resolved["file_version"],
            "num_rows": resolved["num_rows"],
            "num_data_bytes": resolved["num_data_bytes"],
            "num_pages": sum(column["num_pages"] for column in resolved["columns"]),
            "layout_counts": dict(counts),
        }

    for name in ("v2_0_default", "v2_1_default", "v2_3_miniblock"):
        assert summary[name]["layout_counts"].get("sparse", 0) == 0
    for name in ("v2_3_default", "v2_3_sparse"):
        assert summary[name]["layout_counts"].get("sparse", 0) == args.attrs

    print(write_result(args.root / "layout-summary.json", summary), end="")


if __name__ == "__main__":
    main()
