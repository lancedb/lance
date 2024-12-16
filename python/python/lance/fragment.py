# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Dataset Fragment"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import pyarrow as pa

from .dependencies import _check_for_pandas
from .dependencies import pandas as pd
from .lance import _Fragment, _write_fragments
from .lance import _FragmentMetadata as _FragmentMetadata
from .progress import FragmentWriteProgress, NoopFragmentWriteProgress
from .types import _coerce_reader
from .udf import BatchUDF, normalize_transform

if TYPE_CHECKING:
    from .dataset import LanceDataset, LanceScanner, ReaderLike
    from .schema import LanceSchema


DEFAULT_MAX_BYTES_PER_FILE = 90 * 1024 * 1024 * 1024


class FragmentMetadata:
    """Metadata of a Fragment in the dataset."""

    def __init__(self, metadata: str):
        """Construct a FragmentMetadata from a JSON representation of the metadata.

        Internal use only.
        """
        self._metadata = _FragmentMetadata.from_json(metadata)

    @classmethod
    def from_metadata(cls, metadata: _FragmentMetadata):
        instance = cls.__new__(cls)
        instance._metadata = metadata
        return instance

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
        dataset: LanceDataset,
        fragment_id: int,
    ) -> FragmentMetadata:
        """Create a fragment from the given datafile uri.

        This can be used if the datafile is loss from dataset.

        .. warning::

            Internal API. This method is not intended to be used by end users.

        Parameters
        ----------
        filename: str
            The filename of the datafile.
        dataset: LanceDataset
            The dataset that the fragment belongs to.
        fragment_id: int
            The ID of the fragment.
        """
        fragment = _Fragment.create_from_file(filename, dataset._ds, fragment_id)
        return FragmentMetadata(fragment.json())

    @staticmethod
    def create(
        dataset_uri: Union[str, Path],
        data: Union[pa.Table, pa.RecordBatchReader],
        fragment_id: Optional[int] = None,
        schema: Optional[pa.Schema] = None,
        max_rows_per_group: int = 1024,
        progress: Optional[FragmentWriteProgress] = None,
        mode: str = "append",
        *,
        data_storage_version: Optional[str] = None,
        use_legacy_format: Optional[bool] = None,
        storage_options: Optional[Dict[str, str]] = None,
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
        mode: str, default "append"
            The write mode. If "append" is specified, the data will be checked
            against the existing dataset's schema. Otherwise, pass "create" or
            "overwrite" to assign new field ids to the schema.
        data_storage_version: optional, str, default None
            The version of the data storage format to use. Newer versions are more
            efficient but require newer versions of lance to read.  The default (None)
            will use the latest stable version.  See the user guide for more details.
        use_legacy_format: bool, default None
            Deprecated parameter.  Use data_storage_version instead.
        storage_options : optional, dict
            Extra options that make sense for a particular storage connection. This is
            used to store connection parameters like credentials, endpoint, etc.

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
        if use_legacy_format is not None:
            warnings.warn(
                "use_legacy_format is deprecated, use data_storage_version instead",
                DeprecationWarning,
            )
            if use_legacy_format:
                data_storage_version = "legacy"
            else:
                data_storage_version = "stable"

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
            mode=mode,
            data_storage_version=data_storage_version,
            storage_options=storage_options,
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
        with_row_address: bool = False,
        batch_readahead: int = 16,
    ) -> "LanceScanner":
        """See Dataset::scanner for details"""
        filter_str = str(filter) if filter is not None else None

        columns_arg = {}
        if isinstance(columns, dict):
            # convert to list of tuples
            columns_arg = {"columns_with_transform": list(columns.items())}
        elif isinstance(columns, list):
            columns_arg = {"columns": columns}

        s = self._fragment.scanner(
            batch_size=batch_size,
            filter=filter_str,
            limit=limit,
            offset=offset,
            with_row_id=with_row_id,
            with_row_address=with_row_address,
            batch_readahead=batch_readahead,
            **columns_arg,
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

    def merge(
        self,
        data_obj: ReaderLike,
        left_on: str,
        right_on: Optional[str] = None,
        schema=None,
    ) -> Tuple[FragmentMetadata, LanceSchema]:
        """
        Merge another dataset into this fragment.

        Performs a left join, where the fragment is the left side and data_obj
        is the right side. Rows existing in the dataset but not on the left will
        be filled with null values, unless Lance doesn't support null values for
        some types, in which case an error will be raised.

        Parameters
        ----------
        data_obj: Reader-like
            The data to be merged. Acceptable types are:
            - Pandas DataFrame, Pyarrow Table, Dataset, Scanner,
            Iterator[RecordBatch], or RecordBatchReader
        left_on: str
            The name of the column in the dataset to join on.
        right_on: str or None
            The name of the column in data_obj to join on. If None, defaults to
            left_on.

        Examples
        --------

        >>> import lance
        >>> import pyarrow as pa
        >>> df = pa.table({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})
        >>> dataset = lance.write_dataset(df, "dataset")
        >>> dataset.to_table().to_pandas()
           x  y
        0  1  a
        1  2  b
        2  3  c
        >>> fragments = dataset.get_fragments()
        >>> new_df = pa.table({'x': [1, 2, 3], 'z': ['d', 'e', 'f']})
        >>> merged = []
        >>> schema = None
        >>> for f in fragments:
        >>>    f, schema = f.merge(new_df, 'x')
        >>>    merged.append(f)
        >>> merge = lance.LanceOperation.Merge(merged, schema)
        >>> dataset = lance.LanceDataset.commit(base_dir, merge, read_version=1)
        >>> dataset.to_table().to_pandas()
           x  y  z
        0  1  a  d
        1  2  b  e
        2  3  c  f

        See Also
        --------
        LanceDataset.merge_columns :
            Add columns to this Fragment.

        Returns
        -------
        Tuple[FragmentMetadata, LanceSchema]
            A new fragment with the merged column(s) and the final schema.
        """
        if right_on is None:
            right_on = left_on

        reader = _coerce_reader(data_obj, schema)
        manifest_max_field_id = self._ds.manifest_max_field_id
        metadata, schema = self._fragment.merge(
            reader, left_on, right_on, manifest_max_field_id
        )
        return FragmentMetadata.from_metadata(metadata), schema

    def merge_columns(
        self,
        value_func: Dict[str, str]
        | BatchUDF
        | ReaderLike
        | Callable[[pa.RecordBatch], pa.RecordBatch],
        columns: Optional[list[str]] = None,
        batch_size: Optional[int] = None,
        reader_schema: Optional[pa.Schema] = None,
    ) -> Tuple[FragmentMetadata, LanceSchema]:
        """Add columns to this Fragment.

        .. warning::

            Internal API. This method is not intended to be used by end users.

        The parameters and their interpretation are the same as in the
        :meth:`lance.dataset.LanceDataset.add_columns` operation.

        The only difference is that, instead of modifying the dataset, a new
        fragment is created.  The new schema of the fragment is returned as well.
        These can be used in a later operation to commit the changes to the dataset.

        See Also
        --------
        lance.dataset.LanceOperation.Merge :
            The operation used to commit these changes to the dataset. See the
            doc page for an example of using this API.

        Returns
        -------
        Tuple[FragmentMetadata, LanceSchema]
            A new fragment with the added column(s) and the final schema.
        """
        transforms = normalize_transform(value_func, self, columns, reader_schema)

        if isinstance(transforms, BatchUDF):
            if transforms.cache is not None:
                raise ValueError(
                    "A checkpoint file cannot be used when applying a UDF with "
                    "LanceFragment.merge_columns.  You must apply your own "
                    "checkpointing for fragment-level operations."
                )

        if isinstance(transforms, pa.RecordBatchReader):
            metadata, schema = self._fragment.add_columns_from_reader(
                transforms, batch_size
            )
        else:
            metadata, schema = self._fragment.add_columns(
                transforms, columns, batch_size
            )

        return FragmentMetadata.from_metadata(metadata), schema

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
        Fragment { id: 0, files: ..., deletion_file: Some(...), ...}
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
        return FragmentMetadata.from_metadata(raw_fragment.metadata())

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
        return FragmentMetadata.from_metadata(self._fragment.metadata())


def write_fragments(
    data: ReaderLike,
    dataset_uri: Union[str, Path, LanceDataset],
    schema: Optional[pa.Schema] = None,
    *,
    mode: str = "append",
    max_rows_per_file: int = 1024 * 1024,
    max_rows_per_group: int = 1024,
    max_bytes_per_file: int = DEFAULT_MAX_BYTES_PER_FILE,
    progress: Optional[FragmentWriteProgress] = None,
    data_storage_version: Optional[str] = None,
    use_legacy_format: Optional[bool] = None,
    storage_options: Optional[Dict[str, str]] = None,
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
    dataset_uri : str, Path, or LanceDataset
        The URI of the dataset or the dataset object.
    schema : pa.Schema, optional
        The schema of the data. If not specified, the schema will be inferred
        from the data.
    mode : str, default "append"
        The write mode. If "append" is specified, the data will be checked
        against the existing dataset's schema. Otherwise, pass "create" or
        "overwrite" to assign new field ids to the schema.
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
    data_storage_version: optional, str, default None
        The version of the data storage format to use. Newer versions are more
        efficient but require newer versions of lance to read.  The default (None)
        will use the 2.0 version.  See the user guide for more details.
    use_legacy_format : optional, bool, default None
        Deprecated method for setting the data storage version. Use the
        `data_storage_version` parameter instead.
    storage_options : Optional[Dict[str, str]]
        Extra options that make sense for a particular storage connection. This is
        used to store connection parameters like credentials, endpoint, etc.

    Returns
    -------
    List[FragmentMetadata]
        A list of :class:`FragmentMetadata` for the fragments written. The
        fragment ids are left as zero meaning they are not yet specified. They
        will be assigned when the fragments are committed to a dataset.
    """
    from .dataset import LanceDataset

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
    elif isinstance(dataset_uri, LanceDataset):
        dataset_uri = dataset_uri._ds
    elif not isinstance(dataset_uri, str):
        raise TypeError(f"Unknown dataset_uri type {type(dataset_uri)}")

    if use_legacy_format is not None:
        warnings.warn(
            "use_legacy_format is deprecated, use data_storage_version instead",
            DeprecationWarning,
        )
        if use_legacy_format:
            data_storage_version = "legacy"
        else:
            data_storage_version = "stable"

    fragments = _write_fragments(
        dataset_uri,
        reader,
        mode=mode,
        max_rows_per_file=max_rows_per_file,
        max_rows_per_group=max_rows_per_group,
        max_bytes_per_file=max_bytes_per_file,
        progress=progress,
        data_storage_version=data_storage_version,
        storage_options=storage_options,
    )
    return [FragmentMetadata.from_metadata(frag) for frag in fragments]
