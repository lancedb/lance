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
    sample_size : int, default 65536
        Target number of valid rows used to train PCA. Null rows and rows
        containing NaN or infinity are skipped and do not count toward this
        target. Training stops early when the dataset has fewer valid rows.
        The output dimension plus one is only the rank required to estimate
        that many components, not a recommended sample size. 65536 is the
        default target for a stable covariance on typical embedding widths.
        This sample is separate from IVF centroid sampling.
    max_training_memory_bytes : int, default 1073741824
        Maximum estimated peak memory, in bytes, permitted for PCA training.
    sampling_seed : int, default 42
        Seed used to sample vectors for PCA training.
    rotation_seed : int, default 5
        Seed used to construct the deterministic random rotation.
    """

    dimension: int
    random_rotation: bool = False
    sample_size: int = 65536
    max_training_memory_bytes: int = 1073741824
    sampling_seed: int = 42
    rotation_seed: int = 5
