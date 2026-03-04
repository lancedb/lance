# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from enum import Enum

from lance.indices.builder import IndexConfig, IndicesBuilder
from lance.indices.ivf import IvfModel
from lance.indices.pq import PqModel

__all__ = ["IndicesBuilder", "IndexConfig", "PqModel", "IvfModel", "IndexFileVersion"]


class IndexFileVersion(str, Enum):
    LEGACY = "Legacy"
    V3 = "V3"


class SupportedDistributedIndices(str, Enum):
    # Scalar index types
    BTREE = "BTREE"
    INVERTED = "INVERTED"
    # Precise vector index types supported by distributed merge
    IVF_FLAT = "IVF_FLAT"
    IVF_PQ = "IVF_PQ"
    IVF_SQ = "IVF_SQ"
    # Deprecated generic placeholder (kept for backward compatibility)
    VECTOR = "VECTOR"
