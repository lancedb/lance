from typing import List

from ... import _Dataset

class AlignFragmentsPlan:
    tasks: List["AlignFragmentsTask"]
    @staticmethod
    def create(
        source: _Dataset, target: _Dataset, join_key: str
    ) -> AlignFragmentsPlan: ...
    def commit(
        self,
        results: List[AlignFragmentsTaskResult],
        source: _Dataset,
        target: _Dataset,
    ) -> None: ...
    def __repr__(self) -> str: ...

class AlignFragmentsTask:
    def execute(
        self, source: _Dataset, target: _Dataset
    ) -> AlignFragmentsTaskResult: ...
    def __repr__(self) -> str: ...

class AlignFragmentsTaskResult:
    def __repr__(self) -> str: ...
