# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import lance
import numpy as np
import pyarrow as pa
import pytest
from benchmark import (
    find_stable_nprobes,
    recall_at_k,
    summarize_comparison,
    summarize_latencies,
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


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("ON", True), ("0", False), ("no", False)],
)
def test_parse_bool(value, expected):
    assert parse_bool(value) is expected


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


def test_recall_at_k_uses_id_sets():
    result = np.array([1, 3, 8, 9])
    truth = np.array([1, 2, 3, 4])
    assert recall_at_k(result, truth, 4) == 0.5


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
