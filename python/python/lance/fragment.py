#  Copyright (c) 2023. Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""Dataset Fragment"""

from __future__ import annotations

from typing import Callable, Iterator, Optional, Union

import pyarrow as pa

class LanceFragment(pa.dataset.Fragment):
    def __init__(self, dataset: "LanceDataset", fragment_id):
        self._ds = dataset
        self._fragment = dataset.get_fragment(fragment_id)
        if self._fragment is None:
            raise ValueError(f"Fragment id does not exist: {fragment_id}")

    def __reduce__(self):
        from .dataset import LanceDataset

        ds = LanceDataset(self._ds.uri, self._ds.version)
        return LanceFragment, (ds, self.fragment_id)

    @property
    def fragment_id(self):
        return self._fragment.id()

    def count_rows(
        self, filter: Optional[Union[pa.compute.Expression, str]] = None
    ) -> int:
        if filter is not None:
            raise ValueError("Does not support filter at the moment")
        return self._fragment.count_rows()

    def head(self, num_rows: int) -> pa.Table:
        return self.scanner(limit=num_rows).to_table()

    def scanner(
        self,
        columns: Optional[list[str]] = None,
        filter: Optional[Union[str, pa.compute.Expression]] = None,
        limit: int = 0,
        offset: Optional[int] = None,
    ) -> "LanceScanner":
        """See Dataset::scanner for details"""
        filter_str = str(filter) if filter is not None else None
        s = self._fragment.scanner(
            columns=columns, filter=filter_str, limit=limit, offset=offset
        )

        from .dataset import LanceScanner

        return LanceScanner(s, self._ds)

    def take(self, indices, columns: Optional[list[str]] = None) -> pa.Table:
        return pa.Table.from_batches([self._fragment.take(indices, columns=columns)])

    def to_batches(
        self,
        columns: Optional[list[str]] = None,
        filter: Optional[Union[str, pa.compute.Expression]] = None,
        limit: int = 0,
        offset: Optional[int] = None,
    ) -> Iterator[pa.RecordBatch]:
        return self.scanner(
            columns=columns, filter=filter, limit=limit, offset=offset
        ).to_batches()

    def to_table(
        self,
        columns: Optional[list[str]] = None,
        filter: Optional[Union[str, pa.compute.Expression]] = None,
        limit: int = 0,
        offset: Optional[int] = None,
    ) -> pa.Table:
        return self.scanner(
            columns=columns, filter=filter, limit=limit, offset=offset
        ).to_table()

    def add_columns(
        self,
        value_func: Callable[[pa.RecordBatch], pa.RecordBatch],
        columns: Optional[list[str]] = None,
    ) -> LanceFragment:
        """Add columns to this Fragment.

        Parameters
        ----------
        value_func: Callable.
            A function that takes a RecordBatch as input and returns a RecordBatch.
        columns: Optional[list[str]].
            If specified, only the columns in this list will be passed to the value_func.
            Otherwise, all columns will be passed to the value_func.

        Returns
        -------
            A new fragment with the added column(s).
        """
        updater = self._fragment.updater(columns)

        while True:
            batch = updater.next()
            if batch is None:
                break
            new_value = value_func(batch)
            if not isinstance(new_value, pa.RecordBatch):
                raise ValueError(
                    f"value_func must return a Pyarrow RecordBatch, got {type(new_value)}"
                )

            updater.update(new_value)
        return updater.finish()


