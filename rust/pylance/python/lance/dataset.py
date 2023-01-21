from __future__ import annotations

from typing import Optional

import pyarrow as pa

from .lance import _Dataset, _Scanner


class LanceDataset:
    """
    A dataset in Lance format where the data is stored at the given uri
    """

    def __init__(self, uri: str):
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
        return pa.Table.from_batches(self._scanner.to_reader())
