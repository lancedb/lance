from typing import IO, Any, Iterator, List, Optional, Union

import pyarrow as pa

from lance.sampler import SampleMetrics, SampleParams

def infer_tfrecord_schema(
    uri: str,
    tensor_features: Optional[List[str]] = None,
    string_features: Optional[List[str]] = None,
) -> pa.Schema: ...
def read_tfrecord(uri: str, schema: pa.Schema) -> pa.RecordBatchReader: ...

class CleanupStats:
    bytes_removed: int
    old_versions: int

class CompactionMetrics:
    fragments_removed: int
    fragments_added: int
    files_removed: int
    files_added: int

class DatasetSample:
    params: SampleParams
    metrics: SampleMetrics

    @property
    def num_rows(self) -> int: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[pa.Array]: ...
    def __getitem__(self, item: Any) -> Union[pa.Array, DatasetSample]: ...
    def serialize_into(self, path_or_file: Union[str, IO[bytes]]) -> None: ...
    @staticmethod
    def deserialize_from(path_or_file: Union[str, IO[bytes]]) -> DatasetSample: ...
