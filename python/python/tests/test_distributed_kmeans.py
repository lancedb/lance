# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Tests for :mod:`lance.indices.distributed_kmeans`."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import lance
import lance.indices.distributed_kmeans as dk
import numpy as np
import pyarrow as pa
import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def vector_dataset(tmp_path: Path):
    rng = np.random.default_rng(0)
    data = rng.normal(size=(2_000, 8)).astype(np.float32)
    schema = pa.schema([("vec", pa.list_(pa.float32(), 8))])
    arr = pa.FixedSizeListArray.from_arrays(pa.array(data.reshape(-1)), 8)
    table = pa.table([arr], schema=schema)
    uri = str(tmp_path / "vec.lance")
    return lance.write_dataset(table, uri, max_rows_per_file=500)


def test_round_trip_end_to_end(vector_dataset):
    ds = vector_dataset
    samples = dk.sample_round_0(ds, "vec", target=512, distance_type="l2", rng_seed=1)
    assert samples.num_rows == 512

    centroids = dk.bootstrap_centroids([samples], k=16, distance_type="l2", rng_seed=2)
    assert centroids.shape == (16, 8)

    partial = dk.compute_partial_stats(ds, "vec", centroids, distance_type="l2")
    assert partial.num_rows == 16
    assert partial.schema.field("count").type == pa.uint64()

    merged = dk.reduce_partial_stats([partial])
    new_centroids = dk.finalize_centroids(merged, centroids)
    assert new_centroids.shape == centroids.shape
    assert np.all(np.isfinite(new_centroids))


def test_partial_stats_arrow_ipc_round_trip(vector_dataset):
    ds = vector_dataset
    centroids = np.random.RandomState(3).normal(size=(8, 8)).astype(np.float32)
    partial = dk.compute_partial_stats(ds, "vec", centroids)

    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, partial.schema) as writer:
        writer.write_batch(partial)
    sink.seek(0)
    reader = pa.ipc.open_stream(sink)
    restored = next(reader)

    merged = dk.merge_partial_stats(partial, restored)
    assert merged.column("count").to_pylist() == [
        2 * c for c in partial.column("count").to_pylist()
    ]


def test_select_initial_centroids_picks_k(vector_dataset):
    ds = vector_dataset
    samples = dk.sample_round_0(ds, "vec", target=256, rng_seed=4)
    centroids = dk.select_initial_centroids([samples], k=32, rng_seed=5)
    assert centroids.shape == (32, 8)
