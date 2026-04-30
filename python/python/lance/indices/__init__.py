# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from enum import Enum

from .. import lance as _lance
from .builder import IndexConfig, IndicesBuilder
from .ivf import IvfModel
from .pq import PqModel

IndexSegmentDescription = _lance.indices.IndexSegmentDescription
IndexSegment = _lance.indices.IndexSegment
IndexSegmentPlan = _lance.indices.IndexSegmentPlan

__all__ = [
    "IndicesBuilder",
    "IndexConfig",
    "PqModel",
    "IvfModel",
    "IndexFileVersion",
    "IndexSegment",
    "IndexSegmentPlan",
    "IndexSegmentDescription",
]


class IndexFileVersion(str, Enum):
    LEGACY = "Legacy"
    V3 = "V3"


class SupportedDistributedIndices(str, Enum):
    # Scalar index types
    BTREE = "BTREE"
    BITMAP = "BITMAP"
    INVERTED = "INVERTED"

    # Precise vector index types supported by distributed merge
    IVF_FLAT = "IVF_FLAT"
    IVF_PQ = "IVF_PQ"
    IVF_SQ = "IVF_SQ"

    # Deprecated generic placeholder (kept for backward compatibility)
    VECTOR = "VECTOR"
