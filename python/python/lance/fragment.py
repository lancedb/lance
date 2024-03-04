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

import json
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Union,
)

import pyarrow as pa

from .dependencies import _check_for_pandas
from .dependencies import pandas as pd
from .lance import _Fragment, _write_fragments
from .lance import _FragmentMetadata as _FragmentMetadata
from .progress import FragmentWriteProgress, NoopFragmentWriteProgress

if TYPE_CHECKING:
    from .dataset import LanceDataset, LanceScanner, ReaderLike


class FragmentMetadata:
    """Metadata of a Fragment in the dataset."""

    def __init__(self, metadata: str):
        """Construct a FragmentMetadata from a JSON representation of the metadata.

        Internal use only.
        """
        self._metadata = _FragmentMetadata.from_json(metadata)

    def __repr__(self):
        return self._metadata.__repr__()

    def __reduce__(self):
        return (FragmentMetadata, (self._metadata.json(),))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FragmentMetadata):
            return False
        return self._metadata.__eq__(other._metadata)

    def to_json(self) -> str:
        """Serialize :class:`FragmentMetadata` to a JSON blob"""
        return json.loads(self._metadata.json())

    @staticmethod
    def from_json(json_data: str) -> FragmentMetadata:
        """Reconstruct :class:`FragmentMetadata` from a JSON blob"""
        return FragmentMetadata(json_data)

    def data_files(self) -> Iterable[str]:
        """Return the data files of the fragment"""
        return self._metadata.data_files()

    def deletion_file(self):
        """Return the deletion file, if any"""
        return self._metadata.deletion_file()

    @property
    def id(self) -> int:
        return self._metadata.id


class LanceFragment(pa.dataset.Fragment):
    def __init__(
        self,
        dataset: "LanceDataset",
        fragment_id: Optional[int],
        *,
        fragment: Optional[_Fragment] = None,
    ):
        self._ds = dataset
        if fragment is None:
            if fragment_id is None:
                raise ValueError("Either fragment or fragment_id must be specified")
            fragment = dataset.get_fragment(fragment_id)._fragment
        self._fragment = fragment
        if self._fragment is None:
            raise ValueError(f"Fragment id does not exist: {fragment_id}")

    def __repr__(self):
        return self._fragment.__repr__()

    def __reduce__(self):
        from .dataset import LanceDataset

        ds = LanceDataset(self._ds.uri, self._ds.version)
        return LanceFragment, (ds, self.fragment_id)

    @staticmethod
    def create_from_file(
        filename: Union[str, Path],
        schema: pa.Schema,
        fragment_id: int,
    ) -> LanceFragment:
        """Create a fragment from the given datafile uri.

        This can be used if the datafile is loss from dataset.

        .. warning::

            Internal API. This method is not intended to be used by end users.

        Parameters
        ----------
        filename: str
            The filename of the datafile.
        scheme: pa.Schema
            The schema for the new datafile.
        fragment_id: int
            The ID of the fragment.
        """
        return _Fragment.create_from_file(filename, schema, fragment_id)

    @staticmethod
    def create(
        dataset_uri: Union[str, Path],
        data: Union[pa.Table, pa.RecordBatchReader],
        fragment_id: Optional[int] = None,
        schema: Optional[pa.Schema] = None,
        max_rows_per_group: int = 1024,
        progress: Optional[FragmentWriteProgress] = None,
    ) -> FragmentMetadata:
        """Create a :class:`FragmentMetadata` from the given data.

        This can be used if the dataset is not yet created.

        .. warning::

            Internal API. This method is not intended to be used by end users.

        Parameters
        ----------
        dataset_uri: str
            The URI of the dataset.
        fragment_id: int
            The ID of the fragment.
        data: pa.Table or pa.RecordBatchReader
            The data to be written to the fragment.
        schema: pa.Schema, optional
            The schema of the data. If not specified, the schema will be inferred
            from the data.
        max_rows_per_group: int, default 1024
            The maximum number of rows per group in the data file.
        progress: FragmentWriteProgress, optional
            *Experimental API*. Progress tracking for writing the fragment. Pass
            a custom class that defines hooks to be called when each fragment is
            starting to write and finishing writing.

        See Also
        --------
        lance.dataset.LanceOperation.Overwrite :
            The operation used to create a new dataset or overwrite one using
            fragments created with this API. See the doc page for an example of
            using this API.
        lance.dataset.LanceOperation.Append :
            The operation used to append fragments created with this API to an
            existing dataset. See the doc page for an example of using this API.

        Returns
        -------
        FragmentMetadata
        """
        if _check_for_pandas(data) and isinstance(data, pd.DataFrame):
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
        if progress is None:
            progress = NoopFragmentWriteProgress()

        inner_meta = _Fragment.create(
            dataset_uri,
            fragment_id,
            reader,
            max_rows_per_group=max_rows_per_group,
            progress=progress,
        )
        return FragmentMetadata(inner_meta.json())

    @property
    def fragment_id(self):
        return self._fragment.id()

    def count_rows(
        self, filter: Optional[Union[pa.compute.Expression, str]] = None
    ) -> int:
        if filter is not None:
            raise ValueError("Does not support filter at the moment")
        return self._fragment.count_rows()

    @property
    def num_deletions(self) -> int:
        """Return the number of deleted rows in this fragment."""
        return self._fragment.num_deletions

    @property
    def physical_rows(self) -> int:
        """
        Return the number of rows originally in this fragment.

        To get the number of rows after deletions, use
        :meth:`count_rows` instead.
        """
        return self._fragment.physical_rows

    @property
    def physical_schema(self) -> pa.Schema:
        # override the pyarrow super class method otherwise causes segfault
        raise NotImplementedError("Not implemented yet for LanceFragment")

    @property
    def partition_expression(self) -> pa.Schema:
        # override the pyarrow super class method otherwise causes segfault
        raise NotImplementedError("Not implemented yet for LanceFragment")

    def head(self, num_rows: int) -> pa.Table:
        return self.scanner(limit=num_rows).to_table()

    def scanner(
        self,
        *,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        batch_size: Optional[int] = None,
        filter: Optional[Union[str, pa.compute.Expression]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        with_row_id: bool = False,
        batch_readahead: int = 16,
    ) -> "LanceScanner":
        """See Dataset::scanner for details"""
        filter_str = str(filter) if filter is not None else None
        s = self._fragment.scanner(
            columns=columns,
            batch_size=batch_size,
            filter=filter_str,
            limit=limit,
            offset=offset,
            with_row_id=with_row_id,
            batch_readahead=batch_readahead,
        )
        from .dataset import LanceScanner

        return LanceScanner(s, self._ds)

    def take(
        self,
        indices,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
    ) -> pa.Table:
        return pa.Table.from_batches([self._fragment.take(indices, columns=columns)])

    def to_batches(
        self,
        *,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        batch_size: Optional[int] = None,
        filter: Optional[Union[str, pa.compute.Expression]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        with_row_id: bool = False,
        batch_readahead: int = 16,
    ) -> Iterator[pa.RecordBatch]:
        return self.scanner(
            columns=columns,
            batch_size=batch_size,
            filter=filter,
            limit=limit,
            offset=offset,
            with_row_id=with_row_id,
            batch_readahead=batch_readahead,
        ).to_batches()

    def to_table(
        self,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        filter: Optional[Union[str, pa.compute.Expression]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        with_row_id: bool = False,
    ) -> pa.Table:
        return self.scanner(
            columns=columns,
            filter=filter,
            limit=limit,
            offset=offset,
            with_row_id=with_row_id,
        ).to_table()

    def add_columns(
        self,
        value_func: Callable[[pa.RecordBatch], pa.RecordBatch],
        columns: Optional[list[str]] = None,
    ) -> FragmentMetadata:
        """Add columns to this Fragment.

        .. warning::

            Internal API. This method is not intended to be used by end users.

        Parameters
        ----------
        value_func: Callable.
            A function that takes a RecordBatch as input and returns a RecordBatch.
        columns: Optional[list[str]].
            If specified, only the columns in this list will be passed to the
            value_func. Otherwise, all columns will be passed to the value_func.

        See Also
        --------
        lance.dataset.LanceOperation.Merge :
            The operation used to commit these changes to the dataset. See the
            doc page for an example of using this API.

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
        metadata = updater.finish()
        return FragmentMetadata(metadata.json())

    def delete(self, predicate: str) -> FragmentMetadata | None:
        """Delete rows from this Fragment.

        This will add or update the deletion file of this fragment. It does not
        modify or delete the data files of this fragment. If no rows are left after
        the deletion, this method will return None.

        .. warning::

            Internal API. This method is not intended to be used by end users.

        Parameters
        ----------
        predicate: str
            A SQL predicate that specifies the rows to delete.

        Returns
        -------
        FragmentMetadata or None
            A new fragment containing the new deletion file, or None if no rows left.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> tab = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> dataset = lance.write_dataset(tab, "dataset")
        >>> frag = dataset.get_fragment(0)
        >>> frag.delete("a > 1")
        Fragment { id: 0, files: ..., deletion_file: Some(...), physical_rows: Some(3) }
        >>> frag.delete("a > 0") is None
        True

        See Also
        --------
        lance.dataset.LanceOperation.Delete :
            The operation used to commit these changes to a dataset. See the
            doc page for an example of using this API.
        """
        raw_fragment = self._fragment.delete(predicate)
        if raw_fragment is None:
            return None
        return FragmentMetadata(raw_fragment.metadata().json())

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
    def metadata(self) -> FragmentMetadata:
        """Return the metadata of this fragment.

        Returns
        -------
        FragmentMetadata
        """
        return FragmentMetadata(self._fragment.metadata().json())


def write_fragments(
    data: ReaderLike,
    dataset_uri: Union[str, Path],
    schema: Optional[pa.Schema] = None,
    *,
    max_rows_per_file: int = 1024 * 1024,
    max_rows_per_group: int = 1024,
    max_bytes_per_file: int = 90 * 1024 * 1024 * 1024,
    progress: Optional[FragmentWriteProgress] = None,
) -> List[FragmentMetadata]:
    """
    Write data into one or more fragments.

    .. warning::

        This is a low-level API intended for manually implementing distributed
        writes. For most users, :func:`lance.write_dataset` is the recommended API.

    Parameters
    ----------
    data : pa.Table or pa.RecordBatchReader
        The data to be written to the fragment.
    dataset_uri : str
        The URI of the dataset.
    schema : pa.Schema, optional
        The schema of the data. If not specified, the schema will be inferred
        from the data.
    max_rows_per_file : int, default 1024 * 1024
        The maximum number of rows per data file.
    max_rows_per_group : int, default 1024
        The maximum number of rows per group in the data file.
    max_bytes_per_file : int, default 90 * 1024 * 1024 * 1024
        The max number of bytes to write before starting a new file. This is a
        soft limit. This limit is checked after each group is written, which
        means larger groups may cause this to be overshot meaningfully. This
        defaults to 90 GB, since we have a hard limit of 100 GB per file on
        object stores.
    progress : FragmentWriteProgress, optional
        *Experimental API*. Progress tracking for writing the fragment. Pass
        a custom class that defines hooks to be called when each fragment is
        starting to write and finishing writing.

    Returns
    -------
    List[FragmentMetadata]
        A list of :class:`FragmentMetadata` for the fragments written. The
        fragment ids are left as zero meaning they are not yet specified. They
        will be assigned when the fragments are committed to a dataset.
    """
    if _check_for_pandas(data) and isinstance(data, pd.DataFrame):
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

    fragments = _write_fragments(
        dataset_uri,
        reader,
        max_rows_per_file=max_rows_per_file,
        max_rows_per_group=max_rows_per_group,
        max_bytes_per_file=max_bytes_per_file,
        progress=progress,
    )
    return [FragmentMetadata(frag.json()) for frag in fragments]
