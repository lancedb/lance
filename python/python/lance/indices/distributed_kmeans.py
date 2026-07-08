# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Distributed IVF centroid training primitives.

Mirrors :mod:`lance::index::vector::ivf::distributed`. The caller (Spark / Ray /
custom RPC) is responsible for fragment partitioning, broadcast, treeReduce,
and convergence. Lance only provides the math: one E-step, one merge, one
M-step, plus a Round-0 reservoir-sample initializer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Union

import numpy as np
import pyarrow as pa

from .. import lance as _lance

if TYPE_CHECKING:
    from ..dataset import LanceDataset

_indices = _lance.indices


def _to_array_data(
    centroids: Union[np.ndarray, pa.FixedSizeListArray],
) -> pa.FixedSizeListArray:
    """Coerce centroids to a ``pa.FixedSizeListArray`` for the FFI boundary."""
    if isinstance(centroids, np.ndarray):
        if centroids.ndim != 2:
            raise ValueError(f"expected 2-D centroids, got shape {centroids.shape}")
        flat = pa.array(centroids.reshape(-1))
        return pa.FixedSizeListArray.from_arrays(flat, centroids.shape[1])
    return centroids


def _fsl_to_ndarray(arr: pa.FixedSizeListArray) -> np.ndarray:
    """Reshape a 1-D ``FixedSizeListArray`` flat values buffer into ``(k, dim)``."""
    return np.asarray(arr.values).reshape(-1, arr.type.list_size)


def sample_round_0(
    dataset: "LanceDataset",
    column: str,
    target: int,
    *,
    fragment_ids: Optional[Sequence[int]] = None,
    distance_type: str = "l2",
    rng_seed: int = 0,
) -> pa.RecordBatch:
    """Round-0 reservoir-sample on the worker's fragment slice."""
    return _indices.distributed_sample_round_0(
        dataset._ds,
        column,
        target,
        distance_type,
        rng_seed,
        list(fragment_ids) if fragment_ids is not None else None,
    )


def compute_partial_stats(
    dataset: "LanceDataset",
    column: str,
    centroids: Union[np.ndarray, pa.FixedSizeListArray],
    *,
    distance_type: str = "l2",
    fragment_ids: Optional[Sequence[int]] = None,
) -> pa.RecordBatch:
    """Round-r E-step on the worker's fragment slice."""
    return _indices.distributed_compute_partial_stats(
        dataset._ds,
        column,
        _to_array_data(centroids),
        distance_type,
        list(fragment_ids) if fragment_ids is not None else None,
    )


def merge_partial_stats(a: pa.RecordBatch, b: pa.RecordBatch) -> pa.RecordBatch:
    """Combine two partial stats produced against the same centroids."""
    return _indices.distributed_merge_partial_stats(a, b)


def reduce_partial_stats(
    stats: Iterable[pa.RecordBatch],
) -> pa.RecordBatch:
    """Fold an iterable of partial stats."""
    return _indices.distributed_reduce_partial_stats(list(stats))


def finalize_centroids(
    stats: pa.RecordBatch,
    prev_centroids: Union[np.ndarray, pa.FixedSizeListArray],
) -> np.ndarray:
    """Compute the new centroids as a ``(k, dim)`` ndarray."""
    arr = _indices.distributed_finalize_centroids(stats, _to_array_data(prev_centroids))
    return _fsl_to_ndarray(arr)


def select_initial_centroids(
    samples: Sequence[pa.RecordBatch],
    k: int,
    *,
    rng_seed: int = 0,
) -> np.ndarray:
    """Driver-side: pick ``k`` rows uniformly at random from worker samples."""
    arr = _indices.distributed_select_initial_centroids(list(samples), k, rng_seed)
    return _fsl_to_ndarray(arr)


def bootstrap_centroids(
    samples: Sequence[pa.RecordBatch],
    k: int,
    *,
    distance_type: str = "l2",
    rng_seed: int = 0,
) -> np.ndarray:
    """Driver-side: run single-machine kmeans over the union of worker samples."""
    arr = _indices.distributed_bootstrap_centroids(
        list(samples), k, distance_type, rng_seed
    )
    return _fsl_to_ndarray(arr)
