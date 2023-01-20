from __future__ import annotations
import pyarrow as pa
import pyarrow.dataset

from .lance import _Dataset, _Scanner


class FileSystemDataset(pa.dataset.Dataset):

    def __init__(self, uri: str):
        self._uri = uri
        self._ds = _Dataset(uri)

    @property
    def uri(self) -> str:
        return self._uri

    def scanner(self, **kwargs) -> Scanner:
        return Scanner(self._ds.scanner())

    @property
    def schema(self) -> pa.Schema:
        return self._ds.schema


class Scanner(pa.dataset.Scanner):

    def __init__(self, scanner: _Scanner):
        self._scanner = scanner

    def to_table(self) -> pa.Table:
        return pa.Table.from_batches(self._scanner.to_reader())
