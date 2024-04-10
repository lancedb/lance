# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from typing import Optional, TypedDict

# Re-exported from native module. See src/dataset/optimize.rs for implementation.
from .lance import Compaction as Compaction
from .lance import CompactionMetrics as CompactionMetrics
from .lance import CompactionPlan as CompactionPlan
from .lance import CompactionTask as CompactionTask
from .lance import RewriteResult as RewriteResult

# from .lance import CompactionPlan as CompactionPlan


class CompactionOptions(TypedDict):
    """Options for compaction."""

    target_rows_per_fragment: Optional[int]
    """
    The target number of rows per fragment. This is the number of rows
    that will be in each fragment after compaction. (default: 1024*1024)
    """
    max_rows_per_group: Optional[int]
    """
    Max number of rows per group. This does not affect which fragments
    need compaction, but does affect how they are re-written if selected.
    (default: 1024)
    """
    materialize_deletions: Optional[bool]
    """
    Whether to compact fragments with soft deleted rows so they are no
    longer present in the file. (default: True)
    """
    materialize_deletions_threadhold: Optional[float]
    """
    The fraction of original rows that are soft deleted in a fragment
    before the fragment is a candidate for compaction.
    (default: 0.1 = 10%)
    """
    num_threads: Optional[int]
    """
    The number of threads to use when performing compaction. If not
    specified, defaults to the number of cores on the machine.
    """
