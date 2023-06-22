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

"""Dataset Fragment"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Optional, Union

import pandas as pd
import pyarrow as pa

from .lance import _Fragment
from .lance import _FragmentMetadata as _FragmentMetadata

if TYPE_CHECKING:
    from .dataset import LanceDataset, LanceScanner


class LanceFragment(pa.dataset.Fragment):
    def __init__(self, dataset: "LanceDataset", fragment_id: int):
        self._ds = dataset
        self._fragment = dataset.get_fragment(fragment_id)
        if self._fragment is None:
            raise ValueError(f"Fragment id does not exist: {fragment_id}")

    def __repr__(self):
        return self._fragment.__repr__()

    def __reduce__(self):
        from .dataset import LanceDataset

        ds = LanceDataset(self._ds.uri, self._ds.version)
        return LanceFragment, (ds, self.fragment_id)

    @staticmethod
    def create(
        dataset_uri: Union[str, Path],
        fragment_id: int,
        data: pa.Table,
        schema: Optional[pa.Schema] = None,
        max_rows_per_group: int = 1024,
    ) -> LanceFragment:
        """Create a new fragment from the given data.

        This can be used if the dataset is not yet created.

        Parameters
        ----------
        dataset_uri: str
            The URI of the dataset.
        fragment_id: int
            The ID of the fragment.
        data: pa.Table
            The data to write to this fragment.
        schema: pa.Schema, optional
            The schema of the data. If not specified, the schema will be inferred
            from the data.
        """
        if isinstance(data, pd.DataFrame):
            reader = pa.Table.from_pandas(data, schema=schema).to_reader()
        elif isinstance(data, pa.Table):
            reader = data.to_reader()
        elif isinstance(data, pa.dataset.Scanner):
            reader = data.to_reader()
        elif isinstance(data, pa.RecordBatchReader):
            reader = data
        else:
            raise TypeError(f"Unknown data_obj type {type(data)}")

        if isinstance(dataset_uri, Path):
            dataset_uri = str(dataset_uri)
        return _Fragment.create(
            dataset_uri, fragment_id, reader, max_rows_per_group=max_rows_per_group
        )

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
            If specified, only the columns in this list will be passed to the
            value_func. Otherwise, all columns will be passed to the value_func.

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
                    f"value_func must return a Pyarrow RecordBatch, "
                    f"got {type(new_value)}"
                )

            updater.update(new_value)
        return updater.finish()

    def delete(self, predicate: str) -> LanceFragment | None:
        """Delete rows from this Fragment.

        This will add or update the deletion file of this fragment. It does not
        modify or delete the data files of this fragment. If no rows are left after
        the deletion, this method will return None.

        Parameters
        ----------
        predicate: str
            A SQL predicate that specifies the rows to delete.

        Returns
        -------
        LanceFragment or None
            A new fragment containing the new deletion file, or None if no rows left.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> tab = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> dataset = lance.write_dataset(tab, "dataset")
        >>> frag = dataset.get_fragment(0)
        >>> frag.delete("a > 1")
        LanceFileFragment(id=0, data_files=['....lance'], \
deletion_file='_deletions/0-1-....arrow')
        >>> frag.delete("a > 0") is None
        True
        """
        self._fragment.delete(predicate)

    @property
    def schema(self) -> pa.Schema:
        """Return the schema of this fragment."""

        return self._fragment.schema()

    def data_files(self):
        """Return the data files of this fragment."""

        return self._fragment.data_files()

    def deletion_file(self):
        """Return the deletion file, if any"""
        return self._fragment.deletion_file()

    @property
    def metadata(self) -> _FragmentMetadata:
        """Return the metadata of this fragment."""

        return self._fragment.metadata()
