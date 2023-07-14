from typing import List

from lance import Dataset
from lance.fragment import FragmentMetadata
from lance.optimize import CompactionOptions

class CompactionMetrics:
    fragments_removed: int
    fragments_added: int
    files_removed: int
    files_added: int

class RewriteResult:
    read_version: int
    metrics: CompactionMetrics
    old_fragments: List["FragmentMetadata"]
    new_fragments: List["FragmentMetadata"]

class CompactionTask:
    read_version: int
    fragments: List["FragmentMetadata"]

    def execute(self, dataset: "Dataset") -> RewriteResult: ...

class CompactionPlan:
    read_version: int
    tasks: List[CompactionTask]

    def num_tasks(self) -> int: ...

class Compaction:
    @staticmethod
    def execute(
        dataset: "Dataset", options: CompactionOptions
    ) -> CompactionMetrics: ...
    @staticmethod
    def plan(dataset: "Dataset", options: CompactionOptions) -> CompactionPlan: ...
    @staticmethod
    def commit(
        dataset: "Dataset", rewrites: List[RewriteResult]
    ) -> CompactionMetrics: ...
