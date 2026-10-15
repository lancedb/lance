# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from dataclasses import dataclass


@dataclass(frozen=True)
class PcaParams:
    """Configuration for a PCA vector pre-transform.

    **Experimental.** This pre-transform may change incompatibly, in which
    case the index must be rebuilt. The source data is not rewritten.

    Parameters
    ----------
    dimension : int
        Number of dimensions produced by the PCA transform.
    random_rotation : bool, default False
        Whether to apply a deterministic orthogonal rotation after PCA.
    whitening : bool, default False
        Whether to scale retained components by their regularized variance.
    epsilon : float, default 1e-5
        Regularizer used when whitening the retained components.
    sample_size : int, default 65536
        Target number of valid vectors sampled for PCA training.
    max_training_memory_bytes : int, default 1073741824
        Maximum estimated peak memory, in bytes, permitted for PCA training.
    sampling_seed : int, default 42
        Seed used to sample vectors for PCA training.
    rotation_seed : int, default 5
        Seed used to construct the deterministic random rotation.
    """

    dimension: int
    random_rotation: bool = False
    whitening: bool = False
    epsilon: float = 1e-5
    sample_size: int = 65536
    max_training_memory_bytes: int = 1073741824
    sampling_seed: int = 42
    rotation_seed: int = 5
