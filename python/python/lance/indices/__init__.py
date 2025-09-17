# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from enum import Enum

from lance.indices.builder import IndicesBuilder, IndexConfig
from lance.indices.pq import PqModel
from lance.indices.ivf import IvfModel

__all__ = ["IndicesBuilder", "IndexConfig", "PqModel", "IndexFileVersion"]


class IndexFileVersion(str, Enum):
    LEGACY = "Legacy"
    V3 = "V3"
