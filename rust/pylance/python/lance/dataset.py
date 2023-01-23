from __future__ import annotations
from pathlib import Path
from typing import Optional, Union

import pyarrow as pa
import pyarrow.dataset

from .lance import _Dataset, _Scanner, _write_dataset


class LanceDataset:
    """
    A dataset in Lance format where the data is stored at the given uri
    """

    def __init__(self, uri: Union[str, Path]):
        if isinstance(uri, Path):
            uri = str(uri.absolute())
        self._uri = uri
        self._ds = _Dataset(uri)

    @property
    def uri(self) -> str:
        """
        The location of the data
        """
        return self._uri

    def scanner(self, columns: Optional[list[str]] = None,
                limit: int = 0, offset: Optional[int] = None) -> LanceScanner:
        """
        Create a new dataset scanner to read the data
        (but doesn't actually read the data yet).
        Optionally with project and limit/offset pushdown

        Parameters
        ----------
        columns: list of str, default None
            List of column names to be fetched.
            All columns if None or unspecified.
        limit: int, default 0
            Fetch up to this many rows. All rows if 0 or unspecified.
        offset: int, default None
            Fetch starting with this row. 0 if None or unspecified.
        """
        return LanceScanner(self._ds.scanner(columns, limit, offset))

    @property
    def schema(self) -> pa.Schema:
        """
        The pyarrow Schema for this dataset
        """
        return self._ds.schema

    def to_table(self, columns: Optional[list[str]] = None,
                 limit: int = 0, offset: Optional[int] = None) -> pa.Table:
        """
        Read the data into memory and return a pyarrow Table.

        Parameters
        ----------
        columns: list of str, default None
            List of column names to be fetched.
            All columns if None or unspecified.
        limit: int, default 0
            Fetch up to this many rows. All rows if 0 or unspecified.
        offset: int, default None
            Fetch starting with this row. 0 if None or unspecified.
        """
        return self.scanner(
            columns=columns, limit=limit, offset=offset
        ).to_table()


class LanceScanner:

    def __init__(self, scanner: _Scanner):
        self._scanner = scanner

    def to_table(self) -> pa.Table:
        """
        Read the data into memory and return a pyarrow Table.
        """
        return self.to_reader().read_all()

    def to_reader(self) -> pa.RecordBatchReader:
        return self._scanner.to_pyarrow()


ReaderLike = Union[pa.Table, pa.dataset.Dataset, pa.dataset.Scanner,
                   pa.RecordBatchReader, LanceDataset, LanceScanner]


def write_dataset(data_obj: ReaderLike, uri: Union[str, Path],
                  mode: str = "create",
                  max_rows_per_file: int = 1024*1024,
                  max_rows_per_group: int = 1024) -> bool:
    """
    Write a given data_obj to the given uri

    Parameters
    ----------
    data_obj: Reader-like
        The data to be written. Acceptable types are:
        - Pyarrow Table, Dataset, Scanner, or RecordBatchReader
        - LanceDataset or LanceScanner
    uri: str or Path
        Where to write the dataset to (directory)
    mode: str
        create - create a new dataset (raises if uri already exists)
        overwrite - create a new snapshot version
        append - create a new version that is the concat of the input the
                 latest version (raises if uri does not exist)
    max_rows_per_file: int, default 1024 * 1024
        The max number of rows to write before starting a new file
    max_rows_per_group: int, default 1024
        The max number of rows before starting a new group (in the same file)
    """
    if isinstance(data_obj, pa.Table):
        reader = data_obj.to_reader()
    elif isinstance(data_obj, pa.dataset.Dataset):
        reader = pa.dataset.Scanner.from_dataset(data_obj).to_reader()
    elif isinstance(data_obj, pa.dataset.Scanner):
        reader = data_obj.to_reader()
    elif isinstance(data_obj, pa.RecordBatchReader):
        reader = data_obj
    elif isinstance(data_obj, LanceDataset):
        reader = data_obj.scanner()
    elif isinstance(data_obj, LanceScanner):
        reader = data_obj
    else:
        raise TypeError(f"Unknown data_obj type {type(data_obj)}")

    params = {
        "mode": mode,
        "max_rows_per_file": max_rows_per_file,
        "max_rows_per_group": max_rows_per_group
    }
    if isinstance(uri, Path):
        uri = str(uri.absolute())
    return _write_dataset(reader, str(uri), params)
