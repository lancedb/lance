# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Exercise concurrent BlobFile.read_range calls through the Python binding.

The Rust ``blob_range_concurrency`` benchmark is authoritative for physical S3
request metrics. This companion records the Python/PyO3 end-to-end throughput
and latency while reusing the same datasets, offsets, and caller concurrency.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import lance
from lance import BlobFile

MASK_U64 = (1 << 64) - 1
MIX_MULTIPLIER = 0x9E3779B97F4A7C15
MIX_INCREMENT = 0xD1B54A32D192ED03


@dataclass(frozen=True)
class Workload:
    name: str
    handles: Sequence[BlobFile]


@dataclass(frozen=True)
class BenchmarkResult:
    workload: str
    backend: str
    concurrency: int
    samples: int
    elapsed_seconds: float
    samples_per_second: float
    latency_p50_ms: float
    latency_p95_ms: float
    logical_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark concurrent BlobFile.read_range calls through Python"
    )
    parser.add_argument("--base-uri", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--instance-type", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--concurrencies", default="1,4,16,32,64")
    parser.add_argument("--samples-per-worker", type=int, default=16)
    parser.add_argument("--window-bytes", type=int, default=100 * 1024)
    parser.add_argument("--stagger-micros", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def dataset_uri(base_uri: str, name: str) -> str:
    return f"{base_uri.rstrip('/')}/{name}"


def load_workloads(base_uri: str) -> list[Workload]:
    dedicated_dataset = lance.dataset(dataset_uri(base_uri, "dedicated.lance"))
    dedicated = dedicated_dataset.take_blobs("blob", indices=range(4))
    packed_dataset = lance.dataset(dataset_uri(base_uri, "packed.lance"))
    packed = packed_dataset.take_blobs("blob", indices=range(4))
    if len(dedicated) != 4 or len(packed) != 4:
        raise ValueError("benchmark datasets must each contain four blob values")

    # Preparation writes four dedicated objects and two values per packed object.
    return [
        Workload("dedicated_same_source", [dedicated[0]]),
        Workload("dedicated_multiple_sources", dedicated),
        Workload("packed_same_source", packed[:2]),
        Workload("packed_multiple_sources", [packed[0], packed[2]]),
    ]


def sample_offset(sample_index: int, blob_size: int, window_bytes: int) -> int:
    max_start = blob_size - window_bytes
    if max_start < 0:
        raise ValueError(f"window size {window_bytes} exceeds blob size {blob_size}")
    if max_start == 0:
        return 0
    mixed = (sample_index * MIX_MULTIPLIER + MIX_INCREMENT) & MASK_U64
    return mixed % (max_start + 1)


def read_sample(handle: BlobFile, sample_index: int, window_bytes: int) -> float:
    offset = sample_offset(sample_index, handle.size(), window_bytes)
    started = time.perf_counter()
    data = handle.read_range(offset, window_bytes)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if len(data) != window_bytes:
        raise EOFError(f"range returned {len(data)} bytes, expected {window_bytes}")
    return elapsed_ms


def percentile(sorted_values: Sequence[float], quantile: float) -> float:
    rank = math.ceil(quantile * len(sorted_values))
    return sorted_values[max(rank - 1, 0)]


def prewarm(workload: Workload) -> None:
    for handle in workload.handles:
        if handle.read_range(0, 1) == b"":
            raise EOFError("prewarm range returned no bytes")


def run_one(
    workload: Workload,
    concurrency: int,
    samples_per_worker: int,
    window_bytes: int,
    stagger_micros: int,
) -> BenchmarkResult:
    sample_count = concurrency * samples_per_worker
    futures = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for sample_index in range(sample_count):
            if sample_index and stagger_micros:
                time.sleep(stagger_micros / 1_000_000)
            handle = workload.handles[sample_index % len(workload.handles)]
            futures.append(
                executor.submit(read_sample, handle, sample_index, window_bytes)
            )
        latencies_ms = [future.result() for future in futures]
    elapsed_seconds = time.perf_counter() - started
    latencies_ms.sort()
    return BenchmarkResult(
        workload=workload.name,
        backend="python_pyo3_lance",
        concurrency=concurrency,
        samples=sample_count,
        elapsed_seconds=elapsed_seconds,
        samples_per_second=sample_count / elapsed_seconds,
        latency_p50_ms=percentile(latencies_ms, 0.50),
        latency_p95_ms=percentile(latencies_ms, 0.95),
        logical_bytes=sample_count * window_bytes,
    )


def write_json_atomically(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(report, indent=2, sort_keys=True)
    json.loads(payload)
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    concurrencies = [int(value) for value in args.concurrencies.split(",")]
    if (
        not concurrencies
        or any(value <= 0 for value in concurrencies)
        or args.samples_per_worker <= 0
        or args.window_bytes <= 0
        or args.stagger_micros < 0
    ):
        raise ValueError("benchmark counts and sizes must be positive")

    results = []
    for workload in load_workloads(args.base_uri):
        for concurrency in concurrencies:
            prewarm(workload)
            result = run_one(
                workload,
                concurrency,
                args.samples_per_worker,
                args.window_bytes,
                args.stagger_micros,
            )
            print(
                f"workload={result.workload} concurrency={result.concurrency} "
                f"samples_per_second={result.samples_per_second:.2f} "
                f"p50_ms={result.latency_p50_ms:.3f} "
                f"p95_ms={result.latency_p95_ms:.3f}"
            )
            results.append(asdict(result))

    write_json_atomically(
        args.output,
        {
            "schema_version": 1,
            "generated_at_unix_seconds": time.time(),
            "measurement_surface": "Python BlobFile.read_range through PyO3",
            "physical_metrics_source": "Rust blob_range_concurrency benchmark",
            "label": args.label,
            "revision": args.revision,
            "base_uri": args.base_uri,
            "region": args.region,
            "instance_id": args.instance_id,
            "instance_type": args.instance_type,
            "window_bytes": args.window_bytes,
            "samples_per_worker": args.samples_per_worker,
            "stagger_micros": args.stagger_micros,
            "concurrencies": concurrencies,
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
