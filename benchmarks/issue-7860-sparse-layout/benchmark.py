#!/usr/bin/env python3

"""Reproduce lance-format/lance#7860 against Lance 2.3 sparse layout."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path

import lance
import pyarrow as pa


STRUCTURAL_ENCODING_KEY = "lance-encoding:structural-encoding"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--attrs", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-sha", required=True)
    return parser.parse_args()


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def make_table(num_rows: int, num_attrs: int, seed: int) -> pa.Table:
    rng = random.Random(seed)
    value_pool = [f"value_{index}" for index in range(1_000)]
    arrays = [
        pa.array([rng.randint(0, num_rows) for _ in range(num_rows)]),
        pa.array(["a"] * num_rows),
    ]
    fields = [pa.field("id", arrays[0].type), pa.field("kind", arrays[1].type)]
    list_type = pa.list_(pa.utf8())
    for column_index in range(num_attrs):
        values = [
            [rng.choice(value_pool)] if rng.random() < 0.1 else None
            for _ in range(num_rows)
        ]
        arrays.append(pa.array(values, type=list_type))
        fields.append(pa.field(f"attr_{column_index:03}", list_type))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def with_structural_mode(table: pa.Table, mode: str | None) -> pa.Table:
    fields = []
    for field in table.schema:
        if mode is not None and field.name.startswith("attr_"):
            field = field.with_metadata({STRUCTURAL_ENCODING_KEY: mode})
        fields.append(field)
    return pa.Table.from_arrays(table.columns, schema=pa.schema(fields))


def dataset_size(path: Path) -> int:
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def layout_summary(path: Path) -> dict[str, object]:
    from lance.file import LanceFileReader

    data_files = sorted((path / "data").glob("*.lance"))
    layout_counts: Counter[str] = Counter()
    page_count = 0
    file_versions: Counter[str] = Counter()
    for data_file in data_files:
        metadata = LanceFileReader(str(data_file)).metadata()
        file_versions[f"{metadata.major_version}.{metadata.minor_version}"] += 1
        for column in metadata.columns:
            if column is None:
                continue
            for page in column.pages:
                page_count += 1
                encoding = page.encoding
                if "SparseLayout" in encoding:
                    layout = "sparse"
                elif "MiniBlockLayout" in encoding:
                    layout = "miniblock"
                elif "FullZipLayout" in encoding:
                    layout = "fullzip"
                elif "ConstantLayout" in encoding:
                    layout = "constant"
                elif "Flat" in encoding or "List" in encoding:
                    layout = "legacy"
                else:
                    layout = "other"
                layout_counts[layout] += 1
    return {
        "data_files": len(data_files),
        "file_versions": dict(file_versions),
        "page_count": page_count,
        "layout_counts": dict(layout_counts),
        "dataset_bytes": dataset_size(path),
    }


def scan_once(path: Path, late_materialization: bool | None) -> dict[str, float | int]:
    dataset = lance.dataset(path)
    scanner = dataset.scanner(
        order_by=["id"],
        filter="kind = 'a'",
        batch_size=8_192,
        late_materialization=late_materialization,
    )
    started = previous = time.perf_counter()
    batch_times = []
    row_count = 0
    for batch in scanner.to_batches():
        now = time.perf_counter()
        batch_times.append(now - previous)
        previous = now
        row_count += batch.num_rows
    total = time.perf_counter() - started
    return {
        "total_seconds": total,
        "first_batch_seconds": batch_times[0],
        "rest_batch_median_seconds": statistics.median(batch_times[1:]),
        "batches": len(batch_times),
        "rows": row_count,
    }


def validate_dataset(path: Path, expected: pa.Table) -> None:
    actual = lance.dataset(path).to_table()
    assert actual.equals(expected)

    sorted_ids = (
        lance.dataset(path)
        .scanner(
            columns=["id"],
            order_by=["id"],
            filter="kind = 'a'",
            batch_size=8_192,
        )
        .to_table()
        .column("id")
        .combine_chunks()
        .to_pylist()
    )
    assert len(sorted_ids) == len(expected)
    assert all(left <= right for left, right in zip(sorted_ids, sorted_ids[1:]))


def aggregate(samples: list[dict[str, float | int]]) -> dict[str, object]:
    def summarize(key: str) -> dict[str, float]:
        values = sorted(float(sample[key]) for sample in samples)
        return {
            "median": statistics.median(values),
            "min": values[0],
            "max": values[-1],
        }

    return {
        "total_seconds": summarize("total_seconds"),
        "first_batch_seconds": summarize("first_batch_seconds"),
        "rest_batch_median_seconds": summarize("rest_batch_median_seconds"),
        "batches": samples[0]["batches"],
        "rows": samples[0]["rows"],
        "samples": samples,
    }


def write_result(path: Path, result: dict[str, object]) -> str:
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(payload)
    assert json.loads(temp_path.read_text()) == result
    temp_path.replace(path)
    return payload


def main() -> None:
    args = parse_args()
    actual_sha = git_sha()
    if actual_sha != args.expected_sha:
        raise RuntimeError(f"expected git SHA {args.expected_sha}, found {actual_sha}")
    args.root.mkdir(parents=True, exist_ok=False)

    table = make_table(args.rows, args.attrs, args.seed)
    cases = {
        "v2_0_default": ("2.0", None),
        "v2_1_default": ("2.1", None),
        "v2_3_default": ("2.3", None),
        "v2_3_miniblock": ("2.3", "miniblock"),
        "v2_3_sparse": ("2.3", "sparse"),
    }
    paths = {}
    layouts = {}
    write_seconds = {}
    for name, (version, mode) in cases.items():
        path = args.root / name
        expected = with_structural_mode(table, mode)
        started = time.perf_counter()
        lance.write_dataset(
            expected,
            path,
            data_storage_version=version,
            max_rows_per_file=args.rows,
        )
        write_seconds[name] = time.perf_counter() - started
        paths[name] = path
        layouts[name] = layout_summary(path)
        validate_dataset(path, expected)

    for name in ("v2_0_default", "v2_1_default", "v2_3_miniblock"):
        assert layouts[name]["layout_counts"].get("sparse", 0) == 0
    for name in ("v2_3_default", "v2_3_sparse"):
        assert layouts[name]["layout_counts"].get("sparse", 0) == args.attrs

    measurements = {}
    for late_materialization in (None, False):
        late_name = "default" if late_materialization is None else "early"
        samples = {name: [] for name in cases}
        order = list(cases)
        for name in order:
            scan_once(paths[name], late_materialization)
        for repeat in range(args.repeats):
            round_order = order[repeat % len(order) :] + order[: repeat % len(order)]
            if repeat % 2:
                round_order.reverse()
            for name in round_order:
                samples[name].append(scan_once(paths[name], late_materialization))
        measurements[late_name] = {
            name: aggregate(case_samples) for name, case_samples in samples.items()
        }

    result = {
        "benchmark_git_sha": actual_sha,
        "pylance_version": lance.__version__,
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "rows": args.rows,
        "columns": len(table.schema),
        "sparse_probability": 0.1,
        "seed": args.seed,
        "repeats": args.repeats,
        "write_seconds": write_seconds,
        "layouts": layouts,
        "measurements": measurements,
    }
    payload = write_result(args.root / "results.json", result)
    print(payload, end="")


if __name__ == "__main__":
    main()
