# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Dataset Fragment"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    overload,
)

import pyarrow as pa

from .dependencies import _check_for_pandas
from .dependencies import pandas as pd
from .lance import (
    DeletionFile as DeletionFile,
)
from .lance import (
    RowIdMeta as RowIdMeta,
)
from .lance import _Fragment, _write_fragments, _write_fragments_transaction
from .progress import FragmentWriteProgress, NoopFragmentWriteProgress
from .types import _coerce_reader
from .udf import BatchUDF, normalize_transform

if TYPE_CHECKING:
    from .dataset import LanceDataset, LanceScanner, ReaderLike, Transaction
    from .lance import LanceSchema


DEFAULT_MAX_BYTES_PER_FILE = 90 * 1024 * 1024 * 1024


@dataclass
class FragmentMetadata:
    """Metadata for a fragment.

    Attributes
    ----------
    id : int
        The ID of the fragment.
    files : List[DataFile]
        The data files of the fragment. Each data file must have the same number
        of rows. Each file stores a different subset of the columns.
    physical_rows : int
        The number of rows originally in this fragment. This is the number of rows
        in the data files before deletions.
    deletion_file : Optional[DeletionFile]
        The deletion file, if any.
    row_id_meta : Optional[RowIdMeta]
        The row id metadata, if any.
    """

    id: int
    files: List[DataFile]
    physical_rows: int
    deletion_file: Optional[DeletionFile] = None
    row_id_meta: Optional[RowIdMeta] = None

    @property
    def num_deletions(self) -> int:
        """The number of rows that have been deleted from this fragment."""
        if self.deletion_file is None:
            return 0
        else:
            return self.deletion_file.num_deleted_rows

    @property
    def num_rows(self) -> int:
        """The number of rows in this fragment after deletions."""
        return self.physical_rows - self.num_deletions

    def data_files(self) -> List[DataFile]:
        warnings.warn(
            "FragmentMetadata.data_files is deprecated. Use .files instead.",
            DeprecationWarning,
        )
        return self.files

    def to_json(self) -> dict:
        """Get this as a simple JSON-serializable dictionary."""
        files = [asdict(f) for f in self.files]
        for f in files:
            f["path"] = f.pop("_path")
        return dict(
            id=self.id,
            files=files,
            physical_rows=self.physical_rows,
            deletion_file=(
                self.deletion_file.asdict() if self.deletion_file is not None else None
            ),
            row_id_meta=(
                self.row_id_meta.asdict() if self.row_id_meta is not None else None
            ),
        )

    @staticmethod
    def from_json(json_data: str) -> FragmentMetadata:
        json_data = json.loads(json_data)

        deletion_file = json_data.get("deletion_file")
        if deletion_file is not None:
            deletion_file = DeletionFile(**deletion_file)

        row_id_meta = json_data.get("row_id_meta")
        if row_id_meta is not None:
            row_id_meta = RowIdMeta(**row_id_meta)

        return FragmentMetadata(
            id=json_data["id"],
            files=[DataFile(**f) for f in json_data["files"]],
            physical_rows=json_data["physical_rows"],
            deletion_file=deletion_file,
            row_id_meta=row_id_meta,
        )


@dataclass
class DataFile:
    """
    A data file in a fragment.

    Attributes
    ----------
    path : str
        The path to the data file.
    fields : List[int]
        The field ids of the columns in this file.
    column_indices : List[int]
        The column indices where the fields are stored in the file.  Will  have
        the same length as `fields`.
    file_major_version : int
        The major version of the data storage format.
    file_minor_version : int
        The minor version of the data storage format.
    """

    _path: str
    fields: List[int]
    column_indices: List[int] = field(default_factory=list)
    file_major_version: int = 0
    file_minor_version: int = 0

    def __init__(
        self,
        path: str,
        fields: List[int],
        column_indices: List[int] = None,
        file_major_version: int = 0,
        file_minor_version: int = 0,
    ):
        # TODO: only we eliminate the path method, we can remove this
        self._path = path
        self.fields = fields
        self.column_indices = column_indices or []
        self.file_major_version = file_major_version
        self.file_minor_version = file_minor_version

    def __repr__(self):
        # pretend we have a 'path' attribute
        return (
            f"DataFile(path='{self._path}', fields={self.fields}, "
            f"column_indices={self.column_indices}, "
            f"file_major_version={self.file_major_version}, "
            f"file_minor_version={self.file_minor_version})"
        )

    @property
    def path(self) -> str:
        # path used to be a method. This is for backwards compatibility.
        class CallableStr(str):
            def __call__(self):
                warnings.warn(
                    "DataFile.path() is deprecated, use DataFile.path instead",
                    DeprecationWarning,
                )
                return self

            def __reduce__(self):
                return (str, (str(self),))

        return CallableStr(self._path)

    def field_ids(self) -> List[int]:
        warnings.warn(
            "DataFile.field_ids is deprecated, use DataFile.fields instead",
            DeprecationWarning,
        )
        return self.fields


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
        self._fragment: _Fragment = fragment
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
        filename: str,
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
        return _Fragment.create_from_file(filename, dataset._ds, fragment_id)

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

        return _Fragment.create(
            dataset_uri,
            fragment_id,
            reader,
            max_rows_per_group=max_rows_per_group,
            progress=progress,
            mode=mode,
            data_storage_version=data_storage_version,
            storage_options=storage_options,
        )

    @property
    def fragment_id(self):
        return self._fragment.id()

    def count_rows(
        self, filter: Optional[Union[pa.compute.Expression, str]] = None
    ) -> int:
        if filter is not None:
            return self.scanner(filter=filter).count_rows()
        return self._fragment.count_rows(filter)

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
        ...     f, schema = f.merge(new_df, 'x')
        ...     merged.append(f)
        >>> merge = lance.LanceOperation.Merge(merged, schema)
        >>> dataset = lance.LanceDataset.commit("dataset", merge, read_version=1)
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
        max_field_id = self._ds.max_field_id
        metadata, schema = self._fragment.merge(reader, left_on, right_on, max_field_id)
        return metadata, schema

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

        return metadata, schema

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
        FragmentMetadata(id=0, files=[DataFile(path='...', fields=[0, 1], ...), ...)
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
        return raw_fragment.metadata()

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
        return self._fragment.metadata()


if TYPE_CHECKING:

    @overload
    def write_fragments(
        data: ReaderLike,
        dataset_uri: Union[str, Path, LanceDataset],
        schema: Optional[pa.Schema] = None,
        *,
        return_transaction: Literal[True],
        mode: str = "append",
        max_rows_per_file: int = 1024 * 1024,
        max_rows_per_group: int = 1024,
        max_bytes_per_file: int = DEFAULT_MAX_BYTES_PER_FILE,
        progress: Optional[FragmentWriteProgress] = None,
        data_storage_version: Optional[str] = None,
        use_legacy_format: Optional[bool] = None,
        storage_options: Optional[Dict[str, str]] = None,
        enable_move_stable_row_ids: bool = False,
    ) -> Transaction: ...

    @overload
    def write_fragments(
        data: ReaderLike,
        dataset_uri: Union[str, Path, LanceDataset],
        schema: Optional[pa.Schema] = None,
        *,
        return_transaction: Literal[False] = False,
        mode: str = "append",
        max_rows_per_file: int = 1024 * 1024,
        max_rows_per_group: int = 1024,
        max_bytes_per_file: int = DEFAULT_MAX_BYTES_PER_FILE,
        progress: Optional[FragmentWriteProgress] = None,
        data_storage_version: Optional[str] = None,
        use_legacy_format: Optional[bool] = None,
        storage_options: Optional[Dict[str, str]] = None,
        enable_move_stable_row_ids: bool = False,
    ) -> List[FragmentMetadata]: ...


def write_fragments(
    data: ReaderLike,
    dataset_uri: Union[str, Path, LanceDataset],
    schema: Optional[pa.Schema] = None,
    *,
    return_transaction: bool = False,
    mode: str = "append",
    max_rows_per_file: int = 1024 * 1024,
    max_rows_per_group: int = 1024,
    max_bytes_per_file: int = DEFAULT_MAX_BYTES_PER_FILE,
    progress: Optional[FragmentWriteProgress] = None,
    data_storage_version: Optional[str] = None,
    use_legacy_format: Optional[bool] = None,
    storage_options: Optional[Dict[str, str]] = None,
    enable_move_stable_row_ids: bool = False,
) -> List[FragmentMetadata] | Transaction:
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
    return_transaction: bool, default False
        If it's true, the transaction will be returned.
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
    enable_move_stable_row_ids: bool
        Experimental: if set to true, the writer will use move-stable row ids.
        These row ids are stable after compaction operations, but not after updates.
        This makes compaction more efficient, since with stable row ids no
        secondary indices need to be updated to point to new row ids.
    Returns
    -------
    List[FragmentMetadata] | Transaction
        If return_transaction is False:
            a list of :class:`FragmentMetadata` for the fragments written. The
            fragment ids are left as zero meaning they are not yet specified. They
            will be assigned when the fragments are committed to a dataset.

        If return_transaction is True:
            The write transaction. The type of transaction will correspond to
            the mode parameter specified. This transaction can be passed to
            :meth:`LanceDataset.commit`.

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

    function = _write_fragments_transaction if return_transaction else _write_fragments

    return function(
        dataset_uri,
        reader,
        mode=mode,
        max_rows_per_file=max_rows_per_file,
        max_rows_per_group=max_rows_per_group,
        max_bytes_per_file=max_bytes_per_file,
        progress=progress,
        data_storage_version=data_storage_version,
        storage_options=storage_options,
        enable_move_stable_row_ids=enable_move_stable_row_ids,
    )
