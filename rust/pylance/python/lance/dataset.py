from __future__ import annotations
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
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
                limit: int = 0, offset: Optional[int] = None,
                nearest: Optional[dict] = None) -> LanceScanner:
        """
        Return a Scanner that can support various pushdowns

        Parameters
        ----------
        columns: list of str, default None
            List of column names to be fetched.
            All columns if None or unspecified.
        limit: int, default 0
            Fetch up to this many rows. All rows if 0 or unspecified.
        offset: int, default None
            Fetch starting with this row. 0 if None or unspecified.
        nearest: dict, default None
            Get the rows corresponding to the K most similar vectors
            nearest should look like {
              "columns": <embedding col name>,
              "q": <query vector as pa.Float32Array>,
              "k": 10
            }
        """
        return (ScannerBuilder(self)
                .columns(columns)
                .limit(limit)
                .offset(offset)
                .nearest(**(nearest or {}))
                .to_scanner())

    @property
    def schema(self) -> pa.Schema:
        """
        The pyarrow Schema for this dataset
        """
        return self._ds.schema

    def to_table(self, columns: Optional[list[str]] = None,
                 limit: int = 0, offset: Optional[int] = None,
                 nearest: Optional[dict] = None) -> pa.Table:
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
        nearest: dict, default None
            Get the rows corresponding to the K most similar vectors
            nearest should look like {
              "columns": <embedding col name>,
              "q": <query vector as pa.Float32Array>,
              "k": 10
            }
        """
        return self.scanner(
            columns=columns, limit=limit, offset=offset, nearest=nearest
        ).to_table()


class ScannerBuilder:

    def __init__(self, ds: LanceDataset):
        self.ds = ds
        self._limit = 0
        self._offset = None
        self._columns = None
        self._nearest = None

    def limit(self, n: int = 0) -> ScannerBuilder:
        if int(n) < 0:
            raise ValueError("Limit must be non-negative")
        self._limit = n
        return self

    def offset(self, n: Optional[int] = None) -> ScannerBuilder:
        if n is not None and int(n) < 0:
            raise ValueError("Offset must be non-negative")
        self._offset = n
        return self

    def columns(self, cols: Optional[list[str]] = None) -> ScannerBuilder:
        if cols is not None and len(cols) == 0:
            cols = None
        self._columns = cols
        return self

    def nearest(self, column: Optional[str] = None,
                q: Optional[pa.Float32Array] = None,
                k: Optional[int] = None) -> ScannerBuilder:
        if column is None or q is None:
            self._nearest = None
            return self

        if self.ds.schema.get_field_index(column) < 0:
            raise ValueError(f"Embedding column {column} not in dataset")
        if isinstance(q, (np.ndarray, list, tuple)):
            q = pa.Float32Array.from_pandas(q)
        if k is not None and int(k) <= 0:
            raise ValueError(f"Nearest-K must be > 0 but got {k}")
        self._nearest = {
            "column": column,
            "q": q,
            "k": k
        }
        return self

    def to_scanner(self) -> LanceScanner:
        scanner = self.ds._ds.scanner(self._columns, self._limit, self._offset, self._nearest)
        return LanceScanner(scanner)


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
    else:
        raise TypeError(f"Unknown data_obj type {type(data_obj)}")
    # TODO add support for passing in LanceDataset and LanceScanner here

    params = {
        "mode": mode,
        "max_rows_per_file": max_rows_per_file,
        "max_rows_per_group": max_rows_per_group
    }
    if isinstance(uri, Path):
        uri = str(uri.absolute())
    return _write_dataset(reader, str(uri), params)
