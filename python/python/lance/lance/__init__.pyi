from typing import List, Optional

import pyarrow as pa

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
