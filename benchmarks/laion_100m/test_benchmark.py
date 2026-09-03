# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from types import SimpleNamespace

import lance
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import train_models as train_models_module
from benchmark import (
    RequestResult,
    _configured_index_cache_size_bytes,
    _prewarm_index,
    _search,
    build_parser,
    find_stable_nprobes,
    parse_duration_ms,
    parse_stage_plan,
    recall_at_k,
    recall_from_timed_results,
    summarize_comparison,
    summarize_latencies,
    summarize_single_mode,
    summarize_stage_profile,
    validate_ann_only_plan,
    validate_mode_plan_metrics,
    validate_plan_metrics,
)
from build_index import build_index, group_fragments_by_rows
from common import find_index, parse_bool
from prepare_branches import prepare_branches


@dataclass
class FakeFragment:
    fragment_id: int
    physical_rows: int
    num_deletions: int = 0


class FakeTrainingDataset:
    version = 7

    def __init__(self, rows: int, dimension: int):
        self._rows = rows
        self.schema = pa.schema(
            [pa.field("emb", pa.list_(pa.float32(), list_size=dimension))]
        )

    def count_rows(self):
        return self._rows


def _training_args(tmp_path, **overrides):
    values = {
        "dataset_uri": "memory://training",
        "branch": "ivf-rq-reuse-off",
        "output_dir": tmp_path / "models",
        "column": "emb",
        "expected_rows": 512,
        "dimension": 8,
        "num_partitions": 2,
        "ivf_sample_rate": 8,
        "pq_sample_rate": 8,
        "max_iters": 2,
        "rq_num_bits": 5,
        "pq_num_subvectors": 2,
        "pq_num_bits": 4,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _install_training_fakes(monkeypatch, *, fail_pq=False):
    dimension = 8
    dataset = FakeTrainingDataset(rows=512, dimension=dimension)
    centroids = pa.FixedSizeListArray.from_arrays(
        pa.array(np.arange(2 * dimension, dtype=np.float32)), dimension
    )
    codebook = pa.FixedSizeListArray.from_arrays(
        pa.array(np.arange(4 * dimension, dtype=np.float32)), dimension
    )
    ivf_model = SimpleNamespace(centroids=centroids)
    calls = []

    class FakeBuilder:
        def __init__(self, actual_dataset, column):
            assert actual_dataset is dataset
            assert column == "emb"

        def train_ivf(self, **kwargs):
            calls.append(("ivf", kwargs))
            return ivf_model

        def train_pq(self, actual_ivf_model, num_subvectors, **kwargs):
            assert actual_ivf_model is ivf_model
            calls.append(("pq", num_subvectors, kwargs))
            if fail_pq:
                raise RuntimeError("simulated PQ training failure")
            return SimpleNamespace(codebook=codebook)

    monkeypatch.setattr(train_models_module, "open_branch", lambda *_: dataset)
    monkeypatch.setattr(train_models_module, "IndicesBuilder", FakeBuilder)
    monkeypatch.setattr(
        train_models_module.indices,
        "build_rq_model",
        lambda **kwargs: json.dumps({"rotation": "fast", **kwargs}),
    )
    return calls


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("ON", True), ("0", False), ("no", False)],
)
def test_parse_bool(value, expected):
    assert parse_bool(value) is expected


def test_prewarm_requires_explicit_index_cache_capacity():
    with pytest.raises(ValueError, match="--index-cache-size-gib is required"):
        _configured_index_cache_size_bytes(
            argparse.Namespace(prewarm_index=True, index_cache_size_gib=None)
        )


def test_index_cache_capacity_converts_gib_to_bytes():
    assert (
        _configured_index_cache_size_bytes(
            argparse.Namespace(prewarm_index=True, index_cache_size_gib=96.0)
        )
        == 96 * 2**30
    )


def test_benchmark_defaults_to_128_gib_index_cache(tmp_path):
    args = build_parser().parse_args(
        [
            "baseline",
            "--branch",
            "off",
            "--queries",
            "queries.parquet",
            "--ground-truth",
            "truth.parquet",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert args.index_cache_size_gib == 128.0
    assert args.prewarm_index is True
    assert args.nprobes == [16, 64, 256, 1024]
    assert args.concurrency == [1, 8, 16]
    assert args.mode == "off"
    assert args.duration_seconds == 10.0
    assert args.recall_sample_queries == 100


def test_benchmark_can_explicitly_disable_full_index_prewarm(tmp_path):
    args = build_parser().parse_args(
        [
            "baseline",
            "--branch",
            "off",
            "--queries",
            "queries.parquet",
            "--ground-truth",
            "truth.parquet",
            "--output-dir",
            str(tmp_path),
            "--no-prewarm-index",
        ]
    )

    assert args.prewarm_index is False


def test_prewarm_records_cache_and_resource_metrics(tmp_path, monkeypatch):
    class FakeSession:
        cache_size = 1024

        def index_cache_size_bytes(self):
            return self.cache_size

    class FakeDataset:
        def __init__(self):
            self.fake_session = FakeSession()
            self.prewarmed = []

        def session(self):
            return self.fake_session

        def prewarm_index(self, name):
            self.prewarmed.append(name)
            self.fake_session.cache_size = 4096

    dataset = FakeDataset()
    index = SimpleNamespace(total_size_bytes=2048, segments=["a", "b"])
    monkeypatch.setattr("benchmark.find_index", lambda *_: index)
    args = argparse.Namespace(
        prewarm_index=True,
        index_cache_size_gib=4 / 2**20,
        index_name="emb_ivf_rq",
        output_dir=tmp_path,
    )

    metrics = _prewarm_index(dataset, args, output_name="prewarm.json")

    assert dataset.prewarmed == ["emb_ivf_rq"]
    assert metrics["segments"] == 2
    assert metrics["cache_size_bytes_before"] == 1024
    assert metrics["cache_size_bytes_after"] == 4096
    assert metrics["cache_size_bytes_delta"] == 3072
    assert metrics["fully_resident_by_size"] is True
    assert json.loads((tmp_path / "prewarm.json").read_text())["segments"] == 2


def test_train_models_reuses_one_ivf_model_and_records_timings(tmp_path, monkeypatch):
    calls = _install_training_fakes(monkeypatch)
    args = _training_args(tmp_path)

    metrics = train_models_module.train_models(args)

    assert [call[0] for call in calls] == ["ivf", "pq"]
    assert metrics["status"] == "completed"
    assert metrics["config"] == {
        "metric": "L2",
        "num_partitions": 2,
        "ivf_sample_rate": 8,
        "pq_sample_rate": 8,
        "max_iters": 2,
        "rq_num_bits": 5,
        "pq_num_subvectors": 2,
        "pq_num_bits": 4,
    }
    assert set(metrics["phases"]) == {
        "train_ivf",
        "persist_ivf",
        "build_rq_model",
        "persist_rq",
        "train_pq",
        "persist_pq",
    }
    assert all(phase["status"] == "completed" for phase in metrics["phases"].values())
    assert metrics["ivf_training_seconds"] >= 0
    assert metrics["rq_model_build_seconds"] >= 0
    assert metrics["pq_training_seconds"] >= 0
    assert metrics["ivf_rq_model_prepare_seconds"] == pytest.approx(
        metrics["ivf_training_seconds"] + metrics["rq_model_build_seconds"]
    )
    assert metrics["ivf_pq_model_prepare_seconds"] == pytest.approx(
        metrics["ivf_training_seconds"] + metrics["pq_training_seconds"]
    )
    assert metrics["model_persist_seconds"] == pytest.approx(
        sum(
            metrics["phases"][name]["wall_seconds"]
            for name in ("persist_ivf", "persist_rq", "persist_pq")
        )
    )
    assert metrics["peak_rss_gib"] > 0

    output_dir = args.output_dir
    assert {
        "ivf_centroids.arrow",
        "rq_model.json",
        "pq_codebook.arrow",
        "model.json",
        "training_metrics.json",
    } == {path.name for path in output_dir.iterdir()}
    assert set(metrics["artifacts"]) == {
        "ivf_centroids",
        "rq_model",
        "pq_codebook",
        "model_metadata",
    }
    assert all(
        len(artifact["sha256"]) == 64 for artifact in metrics["artifacts"].values()
    )
    model_metadata = json.loads((output_dir / "model.json").read_text())
    assert (
        model_metadata["centroids_sha256"]
        == metrics["artifacts"]["ivf_centroids"]["sha256"]
    )
    assert (
        model_metadata["rq_model_sha256"] == metrics["artifacts"]["rq_model"]["sha256"]
    )
    assert (
        model_metadata["pq_codebook_sha256"]
        == metrics["artifacts"]["pq_codebook"]["sha256"]
    )
    with (output_dir / "pq_codebook.arrow").open("rb") as source:
        pq_schema = ipc.open_file(source).schema
    assert pq_schema.field(0).name == "_pq_codebook"
    assert pq_schema.metadata[b"num_sub_vectors"] == b"2"
    assert pq_schema.metadata[b"num_bits"] == b"4"


def test_train_models_preserves_partial_metrics_on_failure(tmp_path, monkeypatch):
    _install_training_fakes(monkeypatch, fail_pq=True)
    args = _training_args(tmp_path)

    with pytest.raises(RuntimeError, match="simulated PQ training failure"):
        train_models_module.train_models(args)

    metrics = json.loads((args.output_dir / "training_metrics.json").read_text())
    assert metrics["status"] == "failed"
    assert metrics["error_type"] == "RuntimeError"
    assert metrics["phases"]["train_ivf"]["status"] == "completed"
    assert metrics["phases"]["build_rq_model"]["status"] == "completed"
    assert metrics["phases"]["train_pq"]["status"] == "failed"
    assert metrics["ivf_rq_model_prepare_seconds"] is not None
    assert metrics["ivf_pq_model_prepare_seconds"] is None
    assert metrics["model_persist_seconds"] is None
    assert set(metrics["artifacts"]) == {"ivf_centroids", "rq_model"}

    with pytest.raises(ValueError, match="Output directory must be empty"):
        train_models_module.train_models(args)


def test_group_fragments_by_rows_keeps_fragments_whole():
    fragments = [FakeFragment(fragment_id=i, physical_rows=4) for i in range(5)]
    assert group_fragments_by_rows(fragments, target_rows=10) == [
        [0, 1],
        [2, 3],
        [4],
    ]


def test_prepare_branches_is_a_separate_idempotency_guard(tmp_path):
    dataset_uri = str(tmp_path / "branches.lance")
    lance.write_dataset(pa.table({"id": [1, 2]}), dataset_uri)
    args = argparse.Namespace(
        dataset_uri=dataset_uri,
        baseline_branch="off",
        optimized_branch="on",
        source_branch="main",
        source_version=None,
        allow_source_indices=False,
    )

    prepare_branches(args)
    branches = lance.dataset(dataset_uri).branches.list()
    assert {"off", "on"}.issubset(branches)
    with pytest.raises(ValueError, match="Refusing to reuse"):
        prepare_branches(args)


def test_unified_index_build_flow_uses_shared_model_with_toggle(tmp_path, monkeypatch):
    dimension = 32
    dataset_uri = str(tmp_path / "index-build.lance")
    rng = np.random.default_rng(42)
    for batch_index in range(2):
        vectors = rng.random((256, dimension), dtype=np.float32)
        vector_array = pa.FixedSizeListArray.from_arrays(
            pa.array(vectors.reshape(-1)), dimension
        )
        table = pa.table(
            {
                "id": pa.array(np.arange(batch_index * 256, (batch_index + 1) * 256)),
                "emb": vector_array,
            }
        )
        lance.write_dataset(
            table,
            dataset_uri,
            mode="create" if batch_index == 0 else "append",
        )

    prepare_branches(
        argparse.Namespace(
            dataset_uri=dataset_uri,
            baseline_branch="off",
            optimized_branch="on",
            source_branch="main",
            source_version=None,
            allow_source_indices=False,
        )
    )
    common_args = {
        "dataset_uri": dataset_uri,
        "model_dir": tmp_path / "model",
        "checkpoint_dir": tmp_path / "checkpoints",
        "column": "emb",
        "index_name": "emb_ivf_rq",
        "expected_rows": 512,
        "dimension": dimension,
        "num_partitions": 2,
        "target_partition_size": 256,
        "segment_rows": 256,
        "expected_segments": 2,
        "num_bits": 1,
        "sample_rate": 8,
        "max_iters": 2,
        "shuffle_partition_batches": 16,
        "shuffle_partition_concurrency": 1,
    }
    original_create_index = lance.LanceDataset.create_index_uncommitted
    baseline_calls = 0

    def interrupt_second_segment(self, *args, **kwargs):
        nonlocal baseline_calls
        baseline_calls += 1
        if baseline_calls == 2:
            raise RuntimeError("simulated interruption")
        return original_create_index(self, *args, **kwargs)

    monkeypatch.setattr(
        lance.LanceDataset,
        "create_index_uncommitted",
        interrupt_second_segment,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        build_index(
            argparse.Namespace(
                **common_args, branch="off", shared_coarse_quantizer=False
            )
        )
    monkeypatch.setattr(
        lance.LanceDataset,
        "create_index_uncommitted",
        original_create_index,
    )
    build_index(
        argparse.Namespace(**common_args, branch="off", shared_coarse_quantizer=False)
    )
    build_index(
        argparse.Namespace(**common_args, branch="on", shared_coarse_quantizer=True)
    )

    baseline = find_index(
        lance.dataset(dataset_uri).checkout_version(("off", None)), "emb_ivf_rq"
    )
    optimized = find_index(
        lance.dataset(dataset_uri).checkout_version(("on", None)), "emb_ivf_rq"
    )
    assert len(baseline.segments) == len(optimized.segments) == 2
    assert "coarse_quantizer_fingerprint" not in baseline.details
    assert "coarse_quantizer_fingerprint" in optimized.details

    baseline_metrics = json.loads(
        (tmp_path / "checkpoints" / "off" / "build_metrics.json").read_text()
    )
    optimized_metrics = json.loads(
        (tmp_path / "checkpoints" / "on" / "build_metrics.json").read_text()
    )
    assert baseline_metrics["status"] == optimized_metrics["status"] == "completed"
    assert [phase["source"] for phase in baseline_metrics["model_preparations"]] == [
        "trained",
        "loaded",
    ]
    assert optimized_metrics["model_source"] == "loaded"
    assert baseline_metrics["resumed_segments"] == 1
    assert [run["status"] for run in baseline_metrics["runs"]] == [
        "failed",
        "completed",
    ]
    assert baseline_metrics["segment_rows"] == [256, 256]
    assert len(baseline_metrics["segment_build_seconds"]) == 2
    assert all(value >= 0 for value in baseline_metrics["segment_build_seconds"])
    assert baseline_metrics["segments_total_seconds"] == pytest.approx(
        sum(baseline_metrics["segment_build_seconds"])
    )
    assert baseline_metrics["index_build_seconds"] == pytest.approx(
        baseline_metrics["segments_total_seconds"] + baseline_metrics["commit_seconds"]
    )
    assert baseline_metrics["total_wall_seconds"] > 0
    assert baseline_metrics["process_cpu_seconds"] >= 0
    assert baseline_metrics["peak_rss_gib"] > 0
    assert baseline_metrics["max_iop_size_bytes"] == 16 * 1024 * 1024
    assert baseline_metrics["download_retry_count"] == 3


def test_recall_at_k_uses_id_sets():
    result = np.array([1, 3, 8, 9])
    truth = np.array([1, 2, 3, 4])
    assert recall_at_k(result, truth, 4) == 0.5


def test_search_projects_rowid_without_base_columns():
    class FakeDataset:
        def __init__(self):
            self.columns = None

        def to_table(self, *, columns, nearest):
            self.columns = columns
            assert nearest["column"] == "emb"
            return pa.table(
                {
                    "_rowid": pa.array([11, 22], type=pa.uint64()),
                    "_distance": pa.array([0.1, 0.2], type=pa.float32()),
                }
            )

    dataset = FakeDataset()
    result = _search(
        dataset,
        np.zeros(4, dtype=np.float32),
        query_index=7,
        vector_column="emb",
        k=2,
        nprobes=16,
        query_parallelism=1,
        approx_mode="normal",
    )

    assert dataset.columns == ["_rowid", "_distance"]
    assert result.error is None
    assert result.query_index == 7
    assert result.row_ids.tolist() == [11, 22]
    assert result.hits == 0


def test_recall_from_timed_results_backfills_after_search():
    class FakeDataset:
        def _take_rows(self, row_ids, *, columns):
            assert columns == ["id"]
            mapping = {11: 1, 22: 2, 33: 3, 44: 4}
            return pa.table({"id": [mapping[row_id] for row_id in row_ids]})

    results = [
        RequestResult(0.01, 0, np.array([11, 22], dtype=np.uint64)),
        RequestResult(0.01, 1, np.array([33, 44], dtype=np.uint64)),
    ]
    recall, queries, backfill_seconds = recall_from_timed_results(
        FakeDataset(),
        results,
        [np.array([1, 3]), np.array([3, 4])],
        id_column="id",
        k=2,
        max_queries=100,
    )

    assert recall == pytest.approx(0.75)
    assert queries == 2
    assert backfill_seconds >= 0


def test_validate_ann_only_plan_rejects_base_materialization():
    validate_ann_only_plan("ANNSubIndex: metrics=[]\n  ANNIvfPartition: metrics=[]")
    with pytest.raises(ValueError, match="contains LanceRead"):
        validate_ann_only_plan(
            "ProjectionExec: metrics=[]\n  LanceRead: projection=[id]\n    ANNSubIndex:"
        )


def test_find_stable_nprobes_selects_first_plateau():
    recalls = {128: 0.96, 256: 0.9780, 512: 0.9788, 1024: 0.9790}
    assert find_stable_nprobes(recalls) == 256


def test_find_stable_nprobes_rejects_unfinished_curve():
    recalls = {128: 0.90, 256: 0.94, 512: 0.96}
    assert find_stable_nprobes(recalls) is None


def test_summarize_latencies_reports_percentiles_in_milliseconds():
    summary = summarize_latencies([0.001, 0.002, 0.003, 0.004])
    assert summary["latency_mean_ms"] == pytest.approx(2.5)
    assert summary["latency_p50_ms"] == pytest.approx(2.5)
    assert summary["latency_p95_ms"] == pytest.approx(3.85)


def test_validate_plan_metrics_accepts_expected_ab_shape():
    metrics = {
        "off": {
            "find_partitions_calls": ["6"],
            "shared_coarse_quantizer_fast_path": ["0"],
            "coarse_quantizer_reused_segments": ["0"],
        },
        "on": {
            "find_partitions_calls": ["1"],
            "shared_coarse_quantizer_fast_path": ["1"],
            "coarse_quantizer_reused_segments": ["5"],
        },
    }
    validate_plan_metrics(metrics, expected_segments=6)


def test_validate_plan_metrics_rejects_disabled_fast_path():
    metrics = {
        "off": {
            "find_partitions_calls": ["6"],
            "shared_coarse_quantizer_fast_path": ["0"],
            "coarse_quantizer_reused_segments": ["0"],
        },
        "on": {
            "find_partitions_calls": ["6"],
            "shared_coarse_quantizer_fast_path": ["0"],
            "coarse_quantizer_reused_segments": ["0"],
        },
    }
    with pytest.raises(ValueError, match="find_partitions_calls"):
        validate_plan_metrics(metrics, expected_segments=6)


def test_summarize_comparison_reports_relative_improvement():
    rows = []
    for mode, qps, p99, recall in (
        ("off", 100.0, 20.0, 0.95),
        ("on", 125.0, 16.0, 0.95),
    ):
        rows.append(
            {
                "mode": mode,
                "k": 10,
                "nprobes": 512,
                "concurrency": 8,
                "qps": qps,
                "recall": recall,
                "error_rate": 0.0,
                "average_cpu_cores": 8.0,
                "rss_peak_gib": 1.0,
                "latency_mean_ms": p99,
                "latency_p50_ms": p99,
                "latency_p95_ms": p99,
                "latency_p99_ms": p99,
            }
        )
    summary = summarize_comparison(rows)[0]
    assert summary["qps_gain_percent"] == pytest.approx(25.0)
    assert summary["latency_p99_ms_reduction_percent"] == pytest.approx(20.0)
    assert summary["recall_delta"] == 0.0


def test_summarize_single_mode_reports_medians():
    rows = []
    for repeat, qps in enumerate((90.0, 100.0, 120.0)):
        rows.append(
            {
                "repeat": repeat,
                "k": 10,
                "nprobes": 512,
                "concurrency": 8,
                "qps": qps,
                "recall": 0.95,
                "error_rate": 0.0,
                "average_cpu_cores": 8.0,
                "rss_peak_gib": 1.0,
                "latency_mean_ms": 10.0,
                "latency_p50_ms": 9.0,
                "latency_p95_ms": 12.0,
                "latency_p99_ms": 15.0,
            }
        )

    summary = summarize_single_mode(rows)[0]
    assert summary["repeats"] == 3
    assert summary["qps"] == 100.0
    assert summary["latency_p99_ms"] == 15.0


def test_baseline_parser_uses_the_same_timed_defaults():
    args = build_parser().parse_args(
        [
            "baseline",
            "--branch",
            "ivf-rq-reuse-off",
            "--queries",
            "test.parquet",
            "--ground-truth",
            "neighbors.parquet",
            "--output-dir",
            "results",
        ]
    )
    assert args.nprobes == [16, 64, 256, 1024]
    assert args.concurrency == [1, 8, 16]
    assert args.duration_seconds == 10.0
    assert args.repeats == 3
    assert args.expected_segments == 6


def test_baseline_parser_accepts_reuse_on_mode():
    args = build_parser().parse_args(
        [
            "baseline",
            "--branch",
            "ivf-rq-reuse-on",
            "--mode",
            "on",
            "--queries",
            "test.parquet",
            "--ground-truth",
            "neighbors.parquet",
            "--output-dir",
            "results",
        ]
    )
    assert args.mode == "on"


def test_validate_mode_plan_metrics_checks_baseline_only():
    validate_mode_plan_metrics(
        {
            "find_partitions_calls": ["6"],
            "shared_coarse_quantizer_fast_path": ["0"],
            "coarse_quantizer_reused_segments": ["0"],
        },
        "off",
        expected_segments=6,
    )


@pytest.mark.parametrize(
    ("value", "expected_ms"),
    [
        ("1000000ns", 1.0),
        ("1000us", 1.0),
        ("1000µs", 1.0),
        ("1000μs", 1.0),
        ("1.5ms", 1.5),
        ("0.25s", 250.0),
    ],
)
def test_parse_duration_ms(value, expected_ms):
    assert parse_duration_ms(value) == pytest.approx(expected_ms)


def test_parse_stage_plan_attributes_metrics_to_ann_nodes():
    plan = "\n".join(
        [
            "ANNSubIndex: name=emb, metrics=["
            "search_partitions_elapsed=12.5ms, search_partitions_calls=6, "
            "partitions_searched=768, bytes_read=200, iops=20, requests=30]",
            "  ANNIvfPartition: uuid=abc, metrics=["
            "find_partitions_elapsed=750us, find_partitions_calls=1, "
            "partitions_ranked=24414, bytes_read=10, iops=2, requests=3]",
        ]
    )
    metrics = parse_stage_plan(plan)
    assert metrics["coarse_task_ms"] == pytest.approx(0.75)
    assert metrics["bucket_task_ms"] == pytest.approx(12.5)
    assert metrics["coarse_calls"] == 1
    assert metrics["bucket_calls"] == 6
    assert metrics["partitions_ranked"] == 24414
    assert metrics["partitions_searched"] == 768
    assert metrics["coarse_bytes_read"] == 10
    assert metrics["bucket_bytes_read"] == 200
    assert metrics["coarse_task_share"] == pytest.approx(0.75 / 13.25)
    assert metrics["bucket_task_share"] == pytest.approx(12.5 / 13.25)


def test_parse_stage_plan_accepts_compact_large_metric_values():
    plan = "\n".join(
        [
            "ANNSubIndex: name=emb, metrics=["
            "search_partitions_elapsed=11.71s, search_partitions_calls=6, "
            "partitions_searched=1.54 K, bytes_read=625.3 M, iops=57.77 K, "
            "requests=57.77 K]",
            "  ANNIvfPartition: uuid=abc, metrics=["
            "find_partitions_elapsed=187.07ms, find_partitions_calls=6, "
            "partitions_ranked=146.5 K, bytes_read=452.2 M, iops=60, requests=36]",
        ]
    )

    metrics = parse_stage_plan(plan)

    assert metrics["coarse_task_ms"] == pytest.approx(187.07)
    assert metrics["bucket_task_ms"] == pytest.approx(11_710)
    assert metrics["partitions_ranked"] == 146_500
    assert metrics["partitions_searched"] == 1_540
    assert metrics["coarse_bytes_read"] == 452_200_000
    assert metrics["bucket_bytes_read"] == 625_300_000
    assert metrics["bucket_iops"] == 57_770


def test_parse_stage_plan_rejects_missing_bucket_metric():
    plan = "\n".join(
        [
            "ANNSubIndex: name=emb, metrics=[search_partitions_calls=1, "
            "partitions_searched=1, bytes_read=0, iops=0, requests=0]",
            "ANNIvfPartition: uuid=abc, metrics=[find_partitions_elapsed=1ms, "
            "find_partitions_calls=1, partitions_ranked=2, bytes_read=0, "
            "iops=0, requests=0]",
        ]
    )
    with pytest.raises(ValueError, match="search_partitions_elapsed"):
        parse_stage_plan(plan)


def test_summarize_stage_profile_reports_paired_work_changes():
    rows = []
    for query_id in range(2):
        for mode, coarse, bucket, wall in (
            ("off", 6.0, 10.0, 20.0),
            ("on", 1.0, 11.0, 18.0),
        ):
            rows.append(
                {
                    "mode": mode,
                    "query_id": query_id,
                    "k": 10,
                    "nprobes": 256,
                    "profile_wall_ms": wall,
                    "coarse_task_ms": coarse,
                    "bucket_task_ms": bucket,
                    "coarse_task_share": coarse / (coarse + bucket),
                    "bucket_task_share": bucket / (coarse + bucket),
                    "coarse_calls": 6 if mode == "off" else 1,
                    "bucket_calls": 6,
                    "partitions_ranked": 24414 * (6 if mode == "off" else 1),
                    "partitions_searched": 1536,
                    "coarse_bytes_read": 0,
                    "coarse_iops": 0,
                    "coarse_requests": 0,
                    "bucket_bytes_read": 100,
                    "bucket_iops": 10,
                    "bucket_requests": 10,
                }
            )
    summary = summarize_stage_profile(rows)[0]
    assert summary["off_coarse_task_ms_median"] == pytest.approx(6.0)
    assert summary["on_coarse_task_ms_median"] == pytest.approx(1.0)
    assert summary["coarse_work_reduction_percent"] == pytest.approx(100 * 5 / 6)
    assert summary["bucket_work_delta_percent"] == pytest.approx(10.0)
    assert summary["profile_wall_reduction_percent"] == pytest.approx(10.0)


def test_summarize_stage_profile_handles_zero_baseline():
    common = {
        "query_id": 0,
        "k": 10,
        "nprobes": 128,
        "coarse_task_ms": 0.0,
        "bucket_task_ms": 0.0,
        "coarse_task_share": float("nan"),
        "bucket_task_share": float("nan"),
        "coarse_calls": 0,
        "bucket_calls": 0,
        "partitions_ranked": 0,
        "partitions_searched": 0,
        "coarse_bytes_read": 0,
        "coarse_iops": 0,
        "coarse_requests": 0,
        "bucket_bytes_read": 0,
        "bucket_iops": 0,
        "bucket_requests": 0,
    }
    rows = [
        {**common, "mode": "off", "profile_wall_ms": 0.0},
        {**common, "mode": "on", "profile_wall_ms": 0.0},
    ]
    summary = summarize_stage_profile(rows)[0]
    assert math.isnan(summary["coarse_work_reduction_percent"])
    assert math.isnan(summary["bucket_work_delta_percent"])
    assert math.isnan(summary["profile_wall_reduction_percent"])
