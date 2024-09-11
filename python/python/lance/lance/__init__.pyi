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

from typing import Dict, List, Optional

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

class LanceFileWriter:
    def __init__(
        self,
        path: str,
        schema: Optional[pa.Schema],
        data_cache_bytes: Optional[int],
        version: Optional[str],
        storage_options: Optional[Dict[str, str]],
        keep_original_array: Optional[bool],
    ): ...
    def write_batch(self, batch: pa.RecordBatch) -> None: ...
    def finish(self) -> int: ...
    def add_schema_metadata(self, key: str, value: str) -> None: ...
    def add_global_buffer(self, data: bytes) -> int: ...

class LanceFileReader:
    def __init__(self, path: str, storage_options: Optional[Dict[str, str]]): ...
    def read_all(
        self, batch_size: int, batch_readahead: int
    ) -> pa.RecordBatchReader: ...
    def read_range(
        self, start: int, num_rows: int, batch_size: int, batch_readahead: int
    ) -> pa.RecordBatchReader: ...
    def take_rows(
        self, indices: List[int], batch_size: int, batch_readahead: int
    ) -> pa.RecordBatchReader: ...
    def read_global_buffer(self, index: int) -> bytes: ...

class LanceBufferDescriptor:
    position: int
    size: int

class LancePageMetadata:
    buffers: List[LanceBufferDescriptor]
    encoding: str

class LanceColumnMetadata:
    column_buffers: List[LanceBufferDescriptor]
    pages: List[LancePageMetadata]

class LanceFileMetadata:
    schema: pa.Schema
    num_rows: int
    num_data_bytes: int
    num_column_metadata_bytes: int
    num_global_buffer_bytes: int
    global_buffers: List[LanceBufferDescriptor]
    columns: List[LanceColumnMetadata]

class _Session:
    def size_bytes(self) -> int: ...
