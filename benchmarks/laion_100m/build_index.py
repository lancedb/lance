# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil
import pyarrow as pa
import pyarrow.ipc as ipc
from common import (
    DEFAULT_DATASET_URI,
    DEFAULT_DIMENSION,
    DEFAULT_NUM_PARTITIONS,
    DEFAULT_SEGMENT_ROWS,
    DEFAULT_TARGET_PARTITION_SIZE,
    DEFAULT_VECTOR_COLUMN,
    ResourceSampler,
    find_index,
    open_branch,
    parse_bool,
    process_resource_snapshot,
    sha256_file,
    write_json,
)
from lance.dataset import Index, IndexFile
from lance.indices.builder import IndicesBuilder
from lance.lance import indices

if TYPE_CHECKING:
    import lance

CENTROIDS_FILE = "ivf_centroids.arrow"
RQ_MODEL_FILE = "rq_model.json"
MODEL_METADATA_FILE = "model.json"
CHECKPOINT_FILE = "segments.json"
METRICS_FILE = "build_metrics.json"
METRICS_SCHEMA_VERSION = 1
DEFAULT_MAX_IOP_SIZE_BYTES = 16 * 1024 * 1024


def group_fragments_by_rows(
    fragments: list[lance.LanceFragment], target_rows: int
) -> list[list[int]]:
    if target_rows <= 0:
        raise ValueError(f"target_rows must be positive, got {target_rows}")
    groups: list[list[int]] = []
    current: list[int] = []
    current_rows = 0
    for fragment in fragments:
        rows = fragment.physical_rows - fragment.num_deletions
        if current and current_rows + rows > target_rows:
            groups.append(current)
            current = []
            current_rows = 0
        current.append(fragment.fragment_id)
        current_rows += rows
    if current:
        groups.append(current)
    return groups


def _write_centroids(path: Path, centroids: pa.FixedSizeListArray) -> None:
    batch = pa.RecordBatch.from_arrays([centroids], ["ivf_centroids"])
    with path.open("wb") as sink:
        with ipc.new_file(sink, batch.schema) as writer:
            writer.write_batch(batch)


def _read_centroids(path: Path) -> pa.FixedSizeListArray:
    with path.open("rb") as source:
        array = ipc.open_file(source).read_all()["ivf_centroids"].combine_chunks()
    if not pa.types.is_fixed_size_list(array.type):
        raise ValueError(f"Invalid IVF centroid type in {path}: {array.type}")
    return array


def load_or_train_model(
    dataset: lance.LanceDataset,
    model_dir: Path,
    *,
    column: str,
    dimension: int,
    num_partitions: int,
    num_bits: int,
    sample_rate: int,
    max_iters: int,
) -> tuple[pa.FixedSizeListArray, str, dict[str, Any], str]:
    model_dir.mkdir(parents=True, exist_ok=True)
    centroids_path = model_dir / CENTROIDS_FILE
    rq_model_path = model_dir / RQ_MODEL_FILE
    metadata_path = model_dir / MODEL_METADATA_FILE
    expected = {
        "column": column,
        "dimension": dimension,
        "metric": "L2",
        "num_partitions": num_partitions,
        "num_bits": num_bits,
        "sample_rate": sample_rate,
        "max_iters": max_iters,
    }

    if centroids_path.exists() or rq_model_path.exists() or metadata_path.exists():
        if not all(
            path.exists() for path in (centroids_path, rq_model_path, metadata_path)
        ):
            raise ValueError(f"Incomplete shared model artifacts in {model_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"Shared model mismatch for {key}: "
                    f"{metadata.get(key)!r} != {value!r}"
                )
        if metadata.get("centroids_sha256") != sha256_file(centroids_path):
            raise ValueError("IVF centroid checksum mismatch")
        if metadata.get("rq_model_sha256") != sha256_file(rq_model_path):
            raise ValueError("RQ model checksum mismatch")
        return (
            _read_centroids(centroids_path),
            rq_model_path.read_text(encoding="utf-8"),
            metadata,
            "loaded",
        )

    builder = IndicesBuilder(dataset, column)
    ivf_model = builder.train_ivf(
        num_partitions=num_partitions,
        distance_type="l2",
        sample_rate=sample_rate,
        max_iters=max_iters,
    )
    rq_model = indices.build_rq_model(dimension=dimension, num_bits=num_bits)
    _write_centroids(centroids_path, ivf_model.centroids)
    rq_model_path.write_text(rq_model, encoding="utf-8")
    metadata = {
        **expected,
        "centroids_sha256": sha256_file(centroids_path),
        "rq_model_sha256": sha256_file(rq_model_path),
    }
    write_json(metadata_path, metadata)
    return ivf_model.centroids, rq_model, metadata, "trained"


def _index_to_json(index: Index) -> dict[str, Any]:
    details = None
    if index.index_details is not None:
        details = [
            index.index_details[0],
            base64.b64encode(index.index_details[1]).decode("ascii"),
        ]
    return {
        "uuid": index.uuid,
        "name": index.name,
        "fields": index.fields,
        "dataset_version": index.dataset_version,
        "fragment_ids": sorted(index.fragment_ids),
        "index_version": index.index_version,
        "created_at": index.created_at.isoformat() if index.created_at else None,
        "base_id": index.base_id,
        "files": [dataclasses.asdict(file) for file in index.files or []],
        "index_details": details,
    }


def _index_from_json(value: dict[str, Any]) -> Index:
    details = value["index_details"]
    return Index(
        uuid=value["uuid"],
        name=value["name"],
        fields=value["fields"],
        dataset_version=value["dataset_version"],
        fragment_ids=set(value["fragment_ids"]),
        index_version=value["index_version"],
        created_at=(
            datetime.fromisoformat(value["created_at"]) if value["created_at"] else None
        ),
        base_id=value["base_id"],
        files=[IndexFile(**file) for file in value["files"]],
        index_details=(
            (details[0], base64.b64decode(details[1])) if details is not None else None
        ),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metrics_identity(
    args: argparse.Namespace, dataset: lance.LanceDataset
) -> dict[str, Any]:
    max_iop_size_bytes = int(
        os.environ.get("LANCE_MAX_IOP_SIZE", DEFAULT_MAX_IOP_SIZE_BYTES)
    )
    if max_iop_size_bytes <= 0:
        raise ValueError(
            f"LANCE_MAX_IOP_SIZE must be a positive integer, got {max_iop_size_bytes}"
        )
    return {
        "dataset_uri": args.dataset_uri,
        "branch": args.branch,
        "dataset_version": dataset.version,
        "index_name": args.index_name,
        "shared_coarse_quantizer": args.shared_coarse_quantizer,
        "column": args.column,
        "dimension": args.dimension,
        "num_partitions": args.num_partitions,
        "target_partition_size": args.target_partition_size,
        "segment_target_rows": args.segment_rows,
        "num_bits": args.num_bits,
        "max_iop_size_bytes": max_iop_size_bytes,
    }


def _load_or_create_metrics(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if metrics.get("schema_version") != METRICS_SCHEMA_VERSION:
            raise ValueError(
                "Build metrics schema mismatch: "
                f"{metrics.get('schema_version')!r} != {METRICS_SCHEMA_VERSION}"
            )
        for key, value in identity.items():
            if metrics.get(key) != value:
                raise ValueError(f"Build metrics mismatch for {key}")
        return metrics

    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        **identity,
        "status": "running",
        "started_at": _utc_now(),
        "completed_at": None,
        "last_error_type": None,
        "model_preparations": [],
        "segments": [],
        "commit_seconds": None,
        "runs": [],
    }


def _refresh_metric_summaries(metrics: dict[str, Any]) -> None:
    segment_times = [segment.get("wall_seconds") for segment in metrics["segments"]]
    measured_segment_times = [value for value in segment_times if value is not None]
    metrics["model_source"] = (
        metrics["model_preparations"][-1]["source"]
        if metrics["model_preparations"]
        else None
    )
    metrics["model_prepare_seconds"] = sum(
        phase["wall_seconds"] for phase in metrics["model_preparations"]
    )
    metrics["segment_build_seconds"] = segment_times
    metrics["segment_rows"] = [segment["rows"] for segment in metrics["segments"]]
    metrics["segments_total_seconds"] = sum(measured_segment_times)
    metrics["unmeasured_resumed_segments"] = len(segment_times) - len(
        measured_segment_times
    )
    commit_seconds = metrics.get("commit_seconds") or 0.0
    metrics["index_build_seconds"] = metrics["segments_total_seconds"] + commit_seconds
    metrics["end_to_end_phase_seconds"] = (
        metrics["model_prepare_seconds"] + metrics["index_build_seconds"]
    )
    metrics["total_wall_seconds"] = sum(run["wall_seconds"] for run in metrics["runs"])
    metrics["process_cpu_seconds"] = sum(
        run["process_cpu_seconds"] for run in metrics["runs"]
    )
    metrics["peak_rss_gib"] = max(
        (run["peak_rss_gib"] for run in metrics["runs"]), default=0.0
    )


def _reconcile_segment_metrics(
    metrics: dict[str, Any],
    checkpoint: dict[str, Any],
    groups: list[list[int]],
    fragment_rows: dict[int, int],
) -> None:
    if len(metrics["segments"]) > len(checkpoint["segments"]):
        raise ValueError("Build metrics contain more segments than the checkpoint")
    for ordinal in range(len(metrics["segments"]), len(checkpoint["segments"])):
        segment = checkpoint["segments"][ordinal]
        group = groups[ordinal]
        metrics["segments"].append(
            {
                "ordinal": ordinal + 1,
                "index_uuid": segment["uuid"],
                "fragment_count": len(group),
                "rows": sum(fragment_rows[fragment_id] for fragment_id in group),
                "started_at": None,
                "completed_at": segment["created_at"],
                "wall_seconds": None,
                "measurement_status": "unavailable_from_existing_checkpoint",
            }
        )
    _refresh_metric_summaries(metrics)


def build_index(args: argparse.Namespace) -> None:
    process = psutil.Process()
    run_started_at = _utc_now()
    wall_started = time.perf_counter()
    cpu_started, _ = process_resource_snapshot(process)
    resources = ResourceSampler(process)
    resources.start()
    metrics: dict[str, Any] | None = None
    metrics_path: Path | None = None
    resumed_segments = 0
    new_segments = 0
    run_status = "failed"
    error_type: str | None = None
    result: dict[str, Any] | None = None

    try:
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
        if any(index.name == args.index_name for index in dataset.describe_indices()):
            raise ValueError(
                f"Index {args.index_name!r} already exists on branch {args.branch!r}"
            )

        checkpoint_path = args.checkpoint_dir / args.branch / CHECKPOINT_FILE
        metrics_path = getattr(args, "metrics_file", None) or (
            args.checkpoint_dir / args.branch / METRICS_FILE
        )
        metrics = _load_or_create_metrics(
            metrics_path, _metrics_identity(args, dataset)
        )
        metrics["status"] = "running"
        metrics["completed_at"] = None
        metrics["last_error_type"] = None
        write_json(metrics_path, metrics)

        model_started = time.perf_counter()
        centroids, rq_model, model_metadata, model_source = load_or_train_model(
            dataset,
            args.model_dir,
            column=args.column,
            dimension=args.dimension,
            num_partitions=args.num_partitions,
            num_bits=args.num_bits,
            sample_rate=args.sample_rate,
            max_iters=args.max_iters,
        )
        metrics["model_preparations"].append(
            {
                "source": model_source,
                "wall_seconds": time.perf_counter() - model_started,
                "completed_at": _utc_now(),
            }
        )
        _refresh_metric_summaries(metrics)
        write_json(metrics_path, metrics)

        fragments = dataset.get_fragments()
        groups = group_fragments_by_rows(fragments, args.segment_rows)
        if len(groups) != args.expected_segments:
            raise ValueError(
                f"Expected {args.expected_segments} fragment groups, got {len(groups)}"
            )
        fragment_rows = {
            fragment.fragment_id: fragment.physical_rows - fragment.num_deletions
            for fragment in fragments
        }

        checkpoint = {
            "dataset_uri": args.dataset_uri,
            "branch": args.branch,
            "dataset_version": dataset.version,
            "index_name": args.index_name,
            "shared_coarse_quantizer": args.shared_coarse_quantizer,
            "model": model_metadata,
            "groups": groups,
            "segments": [],
        }
        if checkpoint_path.exists():
            existing = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            for key in (
                "dataset_uri",
                "branch",
                "dataset_version",
                "index_name",
                "shared_coarse_quantizer",
                "model",
                "groups",
            ):
                if existing.get(key) != checkpoint[key]:
                    raise ValueError(f"Checkpoint mismatch for {key}")
            checkpoint = existing

        _reconcile_segment_metrics(metrics, checkpoint, groups, fragment_rows)
        segments = [_index_from_json(segment) for segment in checkpoint["segments"]]
        resumed_segments = len(segments)
        metrics["resumed_segments"] = resumed_segments
        write_json(metrics_path, metrics)

        for group in groups[len(segments) :]:
            ordinal = len(segments) + 1
            segment_started_at = _utc_now()
            segment_started = time.perf_counter()
            segment = dataset.create_index_uncommitted(
                column=args.column,
                index_type="IVF_RQ",
                name=args.index_name,
                metric="L2",
                num_partitions=args.num_partitions,
                target_partition_size=args.target_partition_size,
                num_bits=args.num_bits,
                ivf_centroids=centroids,
                rabitq_model=rq_model,
                fragment_ids=group,
                shared_coarse_quantizer=args.shared_coarse_quantizer,
                shuffle_partition_batches=args.shuffle_partition_batches,
                shuffle_partition_concurrency=args.shuffle_partition_concurrency,
            )
            segment_seconds = time.perf_counter() - segment_started
            segments.append(segment)
            checkpoint["segments"] = [_index_to_json(value) for value in segments]
            write_json(checkpoint_path, checkpoint)
            metrics["segments"].append(
                {
                    "ordinal": ordinal,
                    "index_uuid": segment.uuid,
                    "fragment_count": len(group),
                    "rows": sum(fragment_rows[fragment_id] for fragment_id in group),
                    "started_at": segment_started_at,
                    "completed_at": _utc_now(),
                    "wall_seconds": segment_seconds,
                    "measurement_status": "measured",
                }
            )
            new_segments += 1
            _refresh_metric_summaries(metrics)
            write_json(metrics_path, metrics)

        covered = [
            fragment_id for segment in segments for fragment_id in segment.fragment_ids
        ]
        expected = [fragment.fragment_id for fragment in fragments]
        if sorted(covered) != sorted(expected) or len(covered) != len(set(covered)):
            raise ValueError(
                "Built index segments do not cover every fragment exactly once"
            )

        commit_started = time.perf_counter()
        dataset.commit_existing_index_segments(args.index_name, args.column, segments)
        metrics["commit_seconds"] = time.perf_counter() - commit_started
        _refresh_metric_summaries(metrics)
        write_json(metrics_path, metrics)

        description = find_index(dataset, args.index_name)
        if len(description.segments) != args.expected_segments:
            raise RuntimeError(
                f"Committed index has {len(description.segments)} segments, "
                f"expected {args.expected_segments}"
            )
        details = description.details
        has_fingerprint = "coarse_quantizer_fingerprint" in details
        if has_fingerprint != args.shared_coarse_quantizer:
            raise RuntimeError(
                "Committed coarse-quantizer fingerprint does not match "
                "the requested mode"
            )
        result = {
            "branch": args.branch,
            "segments": len(segments),
            "shared_coarse_quantizer": args.shared_coarse_quantizer,
            "coarse_quantizer_fingerprint": details.get("coarse_quantizer_fingerprint"),
        }
        run_status = "completed"
    except BaseException as error:
        error_type = type(error).__name__
        raise
    finally:
        peak_rss = resources.stop()
        cpu_finished, _ = process_resource_snapshot(process)
        if metrics is not None and metrics_path is not None:
            completed_at = _utc_now()
            metrics["runs"].append(
                {
                    "started_at": run_started_at,
                    "completed_at": completed_at,
                    "status": run_status,
                    "wall_seconds": time.perf_counter() - wall_started,
                    "process_cpu_seconds": cpu_finished - cpu_started,
                    "peak_rss_gib": peak_rss / (1024**3),
                    "resumed_segments": resumed_segments,
                    "new_segments": new_segments,
                }
            )
            metrics["status"] = run_status
            metrics["completed_at"] = (
                completed_at if run_status == "completed" else None
            )
            metrics["last_error_type"] = error_type
            metrics["resumed_segments"] = resumed_segments
            _refresh_metric_summaries(metrics)
            write_json(metrics_path, metrics)

    if result is not None and metrics is not None and metrics_path is not None:
        result["metrics_file"] = str(metrics_path)
        result["metrics"] = {
            key: metrics[key]
            for key in (
                "model_source",
                "model_prepare_seconds",
                "segment_build_seconds",
                "segment_rows",
                "segments_total_seconds",
                "commit_seconds",
                "index_build_seconds",
                "total_wall_seconds",
                "process_cpu_seconds",
                "peak_rss_gib",
                "resumed_segments",
            )
        }
        print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build one six-segment LAION 100M IVF_RQ index on an existing branch."
        )
    )
    parser.add_argument("--dataset-uri", default=DEFAULT_DATASET_URI)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--shared-coarse-quantizer", type=parse_bool, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--metrics-file",
        type=Path,
        help=(
            "Build metrics output; defaults to "
            "<checkpoint-dir>/<branch>/build_metrics.json."
        ),
    )
    parser.add_argument("--column", default=DEFAULT_VECTOR_COLUMN)
    parser.add_argument("--index-name", default="emb_ivf_rq")
    parser.add_argument("--expected-rows", type=int, default=100_000_000)
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--num-partitions", type=int, default=DEFAULT_NUM_PARTITIONS)
    parser.add_argument(
        "--target-partition-size", type=int, default=DEFAULT_TARGET_PARTITION_SIZE
    )
    parser.add_argument("--segment-rows", type=int, default=DEFAULT_SEGMENT_ROWS)
    parser.add_argument("--expected-segments", type=int, default=6)
    parser.add_argument("--num-bits", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=256)
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--shuffle-partition-batches", type=int, default=10240)
    parser.add_argument("--shuffle-partition-concurrency", type=int, default=2)
    return parser


if __name__ == "__main__":
    build_index(build_parser().parse_args())
