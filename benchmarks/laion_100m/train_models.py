# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

import psutil
import pyarrow as pa
import pyarrow.ipc as ipc
from common import (
    DEFAULT_BASELINE_BRANCH,
    DEFAULT_DATASET_URI,
    DEFAULT_DIMENSION,
    DEFAULT_NUM_PARTITIONS,
    DEFAULT_VECTOR_COLUMN,
    ResourceSampler,
    open_branch,
    process_resource_snapshot,
    sha256_file,
    write_json,
)
from lance.indices.builder import IndicesBuilder
from lance.lance import indices

IVF_CENTROIDS_FILE = "ivf_centroids.arrow"
RQ_MODEL_FILE = "rq_model.json"
PQ_CODEBOOK_FILE = "pq_codebook.arrow"
MODEL_METADATA_FILE = "model.json"
METRICS_FILE = "training_metrics.json"
METRICS_SCHEMA_VERSION = 1

T = TypeVar("T")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise ValueError(f"Output directory must be empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True)


def _write_array(
    path: Path,
    array: pa.Array,
    field_name: str,
    *,
    metadata: dict[bytes, bytes] | None = None,
) -> None:
    batch = pa.RecordBatch.from_arrays([array], [field_name])
    schema = batch.schema.with_metadata(metadata)
    with path.open("wb") as sink:
        with ipc.new_file(sink, schema) as writer:
            writer.write_batch(batch)


def _phase_seconds(metrics: dict[str, Any], name: str) -> float | None:
    phase = metrics["phases"].get(name)
    if phase is None or phase["status"] != "completed":
        return None
    return phase["wall_seconds"]


def _sum_completed_phases(
    metrics: dict[str, Any], names: tuple[str, ...]
) -> float | None:
    values = [_phase_seconds(metrics, name) for name in names]
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _refresh_summaries(metrics: dict[str, Any]) -> None:
    metrics["ivf_training_seconds"] = _phase_seconds(metrics, "train_ivf")
    metrics["rq_model_build_seconds"] = _phase_seconds(metrics, "build_rq_model")
    metrics["pq_training_seconds"] = _phase_seconds(metrics, "train_pq")
    metrics["model_persist_seconds"] = _sum_completed_phases(
        metrics, ("persist_ivf", "persist_rq", "persist_pq")
    )
    metrics["ivf_rq_model_prepare_seconds"] = _sum_completed_phases(
        metrics, ("train_ivf", "build_rq_model")
    )
    metrics["ivf_pq_model_prepare_seconds"] = _sum_completed_phases(
        metrics, ("train_ivf", "train_pq")
    )
    metrics["ivf_rq_model_ready_seconds"] = _sum_completed_phases(
        metrics, ("train_ivf", "persist_ivf", "build_rq_model", "persist_rq")
    )
    metrics["ivf_pq_model_ready_seconds"] = _sum_completed_phases(
        metrics, ("train_ivf", "persist_ivf", "train_pq", "persist_pq")
    )
    completed_peaks = [
        phase["peak_rss_gib"]
        for phase in metrics["phases"].values()
        if phase.get("peak_rss_gib") is not None
    ]
    metrics["peak_rss_gib"] = max(completed_peaks, default=None)


def _run_phase(
    metrics: dict[str, Any],
    metrics_path: Path,
    name: str,
    action: Callable[[], T],
) -> T:
    process = psutil.Process()
    phase = {
        "status": "running",
        "started_at": _utc_now(),
        "completed_at": None,
        "wall_seconds": None,
        "process_cpu_seconds": None,
        "peak_rss_gib": None,
        "error_type": None,
        "error": None,
    }
    metrics["phases"][name] = phase
    _refresh_summaries(metrics)
    write_json(metrics_path, metrics)

    sampler = ResourceSampler(process)
    sampler.start()
    cpu_before, _ = process_resource_snapshot(process)
    started = time.perf_counter()
    try:
        result = action()
    except BaseException as error:
        phase["status"] = "failed"
        phase["error_type"] = type(error).__name__
        phase["error"] = str(error)
        raise
    else:
        phase["status"] = "completed"
        return result
    finally:
        wall_seconds = time.perf_counter() - started
        cpu_after, _ = process_resource_snapshot(process)
        peak_rss = sampler.stop()
        phase["completed_at"] = _utc_now()
        phase["wall_seconds"] = wall_seconds
        phase["process_cpu_seconds"] = cpu_after - cpu_before
        phase["peak_rss_gib"] = peak_rss / 2**30
        _refresh_summaries(metrics)
        write_json(metrics_path, metrics)


def _record_artifact(
    metrics: dict[str, Any], metrics_path: Path, name: str, path: Path
) -> None:
    metrics["artifacts"][name] = {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    write_json(metrics_path, metrics)


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "expected_rows": args.expected_rows,
        "dimension": args.dimension,
        "num_partitions": args.num_partitions,
        "max_iters": args.max_iters,
        "rq_num_bits": args.rq_num_bits,
        "pq_num_subvectors": args.pq_num_subvectors,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    for name in ("ivf_sample_rate", "pq_sample_rate"):
        value = getattr(args, name)
        if value < 2:
            raise ValueError(f"{name} must be at least 2, got {value}")
    if args.dimension % args.pq_num_subvectors != 0:
        raise ValueError(
            f"dimension ({args.dimension}) must be divisible by pq_num_subvectors "
            f"({args.pq_num_subvectors})"
        )
    if args.pq_num_bits not in {4, 8}:
        raise ValueError(f"pq_num_bits must be 4 or 8, got {args.pq_num_bits}")


def train_models(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    _prepare_output_dir(args.output_dir)
    metrics_path = args.output_dir / METRICS_FILE
    command_started = time.perf_counter()
    process = psutil.Process()
    cpu_before, _ = process_resource_snapshot(process)

    dataset = open_branch(args.dataset_uri, args.branch)
    row_count = dataset.count_rows()
    if row_count != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} rows, got {row_count}")
    vector_type = dataset.schema.field(args.column).type
    if not pa.types.is_fixed_size_list(vector_type):
        raise ValueError(f"Expected a fixed-size vector column, got {vector_type}")
    if vector_type.list_size != args.dimension:
        raise ValueError(
            f"Expected dimension {args.dimension}, got {vector_type.list_size}"
        )

    metrics: dict[str, Any] = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "status": "running",
        "started_at": _utc_now(),
        "completed_at": None,
        "dataset_uri": args.dataset_uri,
        "branch": args.branch,
        "dataset_version": dataset.version,
        "row_count": row_count,
        "column": args.column,
        "dimension": args.dimension,
        "config": {
            "metric": "L2",
            "num_partitions": args.num_partitions,
            "ivf_sample_rate": args.ivf_sample_rate,
            "pq_sample_rate": args.pq_sample_rate,
            "max_iters": args.max_iters,
            "rq_num_bits": args.rq_num_bits,
            "pq_num_subvectors": args.pq_num_subvectors,
            "pq_num_bits": args.pq_num_bits,
        },
        "phases": {},
        "artifacts": {},
        "error_type": None,
        "error": None,
        "total_wall_seconds": None,
        "process_cpu_seconds": None,
    }
    _refresh_summaries(metrics)
    write_json(metrics_path, metrics)

    builder = IndicesBuilder(dataset, args.column)
    try:
        ivf_model = _run_phase(
            metrics,
            metrics_path,
            "train_ivf",
            lambda: builder.train_ivf(
                num_partitions=args.num_partitions,
                distance_type="l2",
                sample_rate=args.ivf_sample_rate,
                max_iters=args.max_iters,
            ),
        )

        ivf_path = args.output_dir / IVF_CENTROIDS_FILE
        _run_phase(
            metrics,
            metrics_path,
            "persist_ivf",
            lambda: _write_array(ivf_path, ivf_model.centroids, "ivf_centroids"),
        )
        _record_artifact(metrics, metrics_path, "ivf_centroids", ivf_path)

        rq_model = _run_phase(
            metrics,
            metrics_path,
            "build_rq_model",
            lambda: indices.build_rq_model(
                dimension=args.dimension, num_bits=args.rq_num_bits
            ),
        )
        rq_path = args.output_dir / RQ_MODEL_FILE
        _run_phase(
            metrics,
            metrics_path,
            "persist_rq",
            lambda: rq_path.write_text(rq_model, encoding="utf-8"),
        )
        _record_artifact(metrics, metrics_path, "rq_model", rq_path)

        pq_model = _run_phase(
            metrics,
            metrics_path,
            "train_pq",
            lambda: builder.train_pq(
                ivf_model,
                args.pq_num_subvectors,
                sample_rate=args.pq_sample_rate,
                max_iters=args.max_iters,
                num_bits=args.pq_num_bits,
            ),
        )
        pq_path = args.output_dir / PQ_CODEBOOK_FILE
        _run_phase(
            metrics,
            metrics_path,
            "persist_pq",
            lambda: _write_array(
                pq_path,
                pq_model.codebook,
                "_pq_codebook",
                metadata={
                    b"num_sub_vectors": str(args.pq_num_subvectors).encode(),
                    b"num_bits": str(args.pq_num_bits).encode(),
                },
            ),
        )
        _record_artifact(metrics, metrics_path, "pq_codebook", pq_path)

        model_metadata = {
            "column": args.column,
            "dimension": args.dimension,
            "metric": "L2",
            "num_partitions": args.num_partitions,
            "num_bits": args.rq_num_bits,
            "sample_rate": args.ivf_sample_rate,
            "max_iters": args.max_iters,
            "pq_num_subvectors": args.pq_num_subvectors,
            "pq_num_bits": args.pq_num_bits,
            "pq_sample_rate": args.pq_sample_rate,
            "centroids_sha256": metrics["artifacts"]["ivf_centroids"]["sha256"],
            "rq_model_sha256": metrics["artifacts"]["rq_model"]["sha256"],
            "pq_codebook_sha256": metrics["artifacts"]["pq_codebook"]["sha256"],
        }
        metadata_path = args.output_dir / MODEL_METADATA_FILE
        write_json(metadata_path, model_metadata)
        _record_artifact(metrics, metrics_path, "model_metadata", metadata_path)
        metrics["status"] = "completed"
    except BaseException as error:
        metrics["status"] = "failed"
        metrics["error_type"] = type(error).__name__
        metrics["error"] = str(error)
        raise
    finally:
        cpu_after, _ = process_resource_snapshot(process)
        metrics["completed_at"] = _utc_now()
        metrics["total_wall_seconds"] = time.perf_counter() - command_started
        metrics["process_cpu_seconds"] = cpu_after - cpu_before
        _refresh_summaries(metrics)
        write_json(metrics_path, metrics)

    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure IVF, RaBitQ, and PQ model preparation independently"
    )
    parser.add_argument("--dataset-uri", default=DEFAULT_DATASET_URI)
    parser.add_argument("--branch", default=DEFAULT_BASELINE_BRANCH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--column", default=DEFAULT_VECTOR_COLUMN)
    parser.add_argument("--expected-rows", type=int, default=100_000_000)
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--num-partitions", type=int, default=DEFAULT_NUM_PARTITIONS)
    parser.add_argument("--ivf-sample-rate", type=int, default=256)
    parser.add_argument("--pq-sample-rate", type=int, default=256)
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--rq-num-bits", type=int, default=5)
    parser.add_argument("--pq-num-subvectors", type=int, default=48)
    parser.add_argument("--pq-num-bits", type=int, default=8)
    return parser


if __name__ == "__main__":
    train_models(build_parser().parse_args())
