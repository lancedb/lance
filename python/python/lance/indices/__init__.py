# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from enum import Enum

from .. import lance as _lance
from .builder import IndexConfig, IndicesBuilder
from .ivf import IvfModel
from .pq import PqModel

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
]


class IndexFileVersion(str, Enum):
    LEGACY = "Legacy"
    V3 = "V3"


class SupportedScalarDistributedIndices(str, Enum):
    # Scalar index types
    BTREE = "BTREE"
    INVERTED = "INVERTED"
