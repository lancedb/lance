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

from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Self,
    Sequence,
    Tuple,
    Union,
)

import pyarrow as pa

from .._arrow.bf16 import BFloat16Array
from ..commit import CommitLock
from ..dataset import (
    AlterColumn,
    ExecuteResult,
    Index,
    LanceOperation,
    Tag,
    Transaction,
    UpdateResult,
    Version,
)
from ..fragment import (
    DataFile,
    FragmentMetadata,
)
from ..progress import FragmentWriteProgress as FragmentWriteProgress
from ..types import ReaderLike as ReaderLike
from ..udf import BatchUDF as BatchUDF
from .debug import format_fragment as format_fragment
from .debug import format_manifest as format_manifest
from .debug import format_schema as format_schema
from .debug import list_transactions as list_transactions
from .fragment import (
    DeletionFile as DeletionFile,
)
from .fragment import (
    RowIdMeta as RowIdMeta,
)
from .optimize import (
    Compaction as Compaction,
)
from .optimize import (
    CompactionMetrics as CompactionMetrics,
)
from .optimize import (
    CompactionPlan as CompactionPlan,
)
from .optimize import (
    CompactionTask as CompactionTask,
)
from .optimize import (
    RewriteResult as RewriteResult,
)
from .schema import LanceSchema as LanceSchema
from .trace import trace_to_chrome as trace_to_chrome

def infer_tfrecord_schema(
    uri: str,
    tensor_features: Optional[List[str]] = None,
    string_features: Optional[List[str]] = None,
) -> pa.Schema: ...
def read_tfrecord(uri: str, schema: pa.Schema) -> pa.RecordBatchReader: ...

class CleanupStats:
    bytes_removed: int
    old_versions: int

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
    def metadata(self) -> LanceFileMetadata: ...
    def file_statistics(self) -> LanceFileStatistics: ...

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

class LanceFileStatistics:
    columns: List[LanceColumnStatistics]

class LanceColumnStatistics:
    num_pages: int
    size_bytes: int

class _Session:
    def size_bytes(self) -> int: ...

class LanceBlobFile:
    def close(self): ...
    def is_closed(self) -> bool: ...
    def seek(self, offset: int): ...
    def tell(self) -> int: ...
    def size(self) -> int: ...
    def readall(self) -> bytes: ...
    def read_into(self, b: bytearray) -> int: ...

class _Dataset:
    @property
    def uri(self) -> str: ...
    def __init__(
        self,
        uri: str,
        version: Optional[int | str] = None,
        block_size: Optional[int] = None,
        index_cache_size: Optional[int] = None,
        metadata_cache_size: Optional[int] = None,
        commit_handler: Optional[CommitLock] = None,
        storage_options: Optional[Dict[str, str]] = None,
        manifest: Optional[bytes] = None,
        **kwargs,
    ): ...
    @property
    def schema(self) -> pa.Schema: ...
    @property
    def lance_schema(self) -> LanceSchema: ...
    def replace_schema_metadata(self, metadata: Dict[str, str]): ...
    def replace_field_metadata(self, field_name: str, metadata: Dict[str, str]): ...
    @property
    def data_storage_version(self) -> str: ...
    def index_statistics(self, index_name: str) -> str: ...
    def serialized_manifest(self) -> bytes: ...
    def load_indices(self) -> List[Index]: ...
    def scanner(
        self,
        columns: Optional[List[str]] = None,
        columns_with_transform: Optional[List[Tuple[str, str]]] = None,
        filter: Optional[str] = None,
        prefilter: Optional[bool] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        nearest: Optional[Dict] = None,
        batch_size: Optional[int] = None,
        io_buffer_size: Optional[int] = None,
        batch_readahead: Optional[int] = None,
        fragment_readahead: Optional[int] = None,
        scan_in_order: Optional[bool] = None,
        fragments: Optional[List[_Fragment]] = None,
        with_row_id: Optional[bool] = None,
        with_row_address: Optional[bool] = None,
        use_stats: Optional[bool] = None,
        substrait_filter: Optional[bytes] = None,
        fast_search: Optional[bool] = None,
        full_text_query: Optional[dict] = None,
        late_materialization: Optional[bool | List[str]] = None,
        use_scalar_index: Optional[bool] = None,
    ) -> _Scanner: ...
    def count_rows(self, filter: Optional[str] = None) -> int: ...
    def take(
        self,
        row_indices: List[int],
        columns: Optional[List[str]] = None,
        columns_with_transform: Optional[List[Tuple[str, str]]] = None,
    ) -> pa.RecordBatch: ...
    def take_rows(
        self,
        row_indices: List[int],
        columns: Optional[List[str]] = None,
        columns_with_transform: Optional[List[Tuple[str, str]]] = None,
    ) -> pa.RecordBatch: ...
    def take_blobs(
        self,
        row_indices: List[int],
        blob_column: str,
    ) -> List[LanceBlobFile]: ...
    def take_scan(
        self,
        row_slices: Iterable[Tuple[int, int]],
        columns: Optional[List[str]] = None,
        batch_readahead: int = 10,
    ) -> pa.RecordBatchReader: ...
    def alter_columns(self, alterations: List[AlterColumn]): ...
    def merge(self, reader: pa.RecordBatchReader, left_on: str, right_on: str): ...
    def delete(self, predicate: str): ...
    def update(
        self,
        updates: Dict[str, str],
        predicate: Optional[str] = None,
    ) -> UpdateResult: ...
    def count_deleted_rows(self) -> int: ...
    def versions(self) -> List[Version]: ...
    def version(self) -> int: ...
    def latest_version(self) -> int: ...
    def checkout_version(self, version: int | str) -> _Dataset: ...
    def restore(self): ...
    def cleanup_old_versions(
        self,
        older_than_micros: int,
        delete_unverified: Optional[bool] = None,
        error_if_tagged_old_versions: Optional[bool] = None,
    ) -> CleanupStats: ...
    def tags(self) -> Dict[str, Tag]: ...
    def create_tag(self, tag: str, version: int): ...
    def delete_tag(self, tag: str): ...
    def update_tag(self, tag: str, version: int): ...
    def optimize_indices(self, **kwargs): ...
    def create_index(
        self,
        columns: List[str],
        index_type: str,
        name: Optional[str] = None,
        replace: Optional[bool] = None,
        storage_options: Optional[Dict[str, str]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ): ...
    def drop_index(self, name: str): ...
    def count_fragments(self) -> int: ...
    def num_small_files(self, max_rows_per_group: int) -> int: ...
    def get_fragments(self) -> List[_Fragment]: ...
    def get_fragment(self, fragment_id: int) -> Optional[_Fragment]: ...
    def index_cache_entry_count(self) -> int: ...
    def index_cache_hit_rate(self) -> float: ...
    def session(self) -> _Session: ...
    @staticmethod
    def drop(dest: str, storage_options: Optional[Dict[str, str]] = None): ...
    @staticmethod
    def commit(
        dest: str | _Dataset,
        operation: LanceOperation.BaseOperation,
        blobs_op: Optional[LanceOperation.BaseOperation] = None,
        read_version: Optional[int] = None,
        commit_lock: Optional[CommitLock] = None,
        storage_options: Optional[Dict[str, str]] = None,
        enable_v2_manifest_paths: Optional[bool] = None,
        detached: Optional[bool] = None,
        max_retries: Optional[int] = None,
        **kwargs,
    ) -> _Dataset: ...
    @staticmethod
    def commit_batch(
        dest: str | _Dataset,
        transactions: Sequence[Transaction],
        commit_lock: Optional[CommitLock] = None,
        storage_options: Optional[Dict[str, str]] = None,
        enable_v2_manifest_paths: Optional[bool] = None,
        detached: Optional[bool] = None,
        max_retries: Optional[int] = None,
    ) -> Tuple[_Dataset, Transaction]: ...
    def validate(self): ...
    def migrate_manifest_paths_v2(self): ...
    def drop_columns(self, columns: List[str]): ...
    def add_columns_from_reader(
        self, reader: pa.RecordBatchReader, batch_size: Optional[int] = None
    ): ...
    def add_columns(
        self,
        transforms: Dict[str, str] | BatchUDF | ReaderLike,
        read_columns: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
    ): ...

class _MergeInsertBuilder:
    def __init__(self, dataset: _Dataset, on: str | Iterable[str]): ...
    def when_matched_update_all(self, condition: Optional[str] = None) -> Self: ...
    def when_not_matched_insert_all(self) -> Self: ...
    def when_not_matched_by_source_delete(self, expr: Optional[str] = None) -> Self: ...
    def execute(self, new_data: pa.RecordBatchReader) -> ExecuteResult: ...

class _Scanner:
    @property
    def schema(self) -> pa.Schema: ...
    def explain_plan(self, verbose: bool) -> str: ...
    def analyze_plan(self) -> str: ...
    def count_rows(self) -> int: ...
    def to_pyarrow(self) -> pa.RecordBatchReader: ...

class _Fragment:
    @staticmethod
    def create_from_file(
        filename: str,
        dataset: _Dataset,
        fragment_id: int,
    ) -> FragmentMetadata: ...
    @staticmethod
    def create(
        dataset_uri: str,
        fragment_id: Optional[int],
        reader: ReaderLike,
        **kwargs,
    ): ...
    def id(self) -> int: ...
    def metadata(self) -> FragmentMetadata: ...
    def count_rows(self, _filter: Optional[str] = None) -> int: ...
    def take(
        self,
        row_indices: List[int],
        columns: Optional[Union[List[str], Dict[str, str]]],
    ) -> pa.RecordBatch: ...
    def scanner(
        self,
        columns: Optional[List[str]],
        columns_with_transform: Optional[List[Tuple[str, str]]],
        batch_size: Optional[int],
        filter: Optional[str],
        limit: Optional[int],
        offset: Optional[int],
        with_row_id: Optional[bool],
        batch_readahead: Optional[int],
        **kwargs,
    ) -> _Scanner: ...
    def add_columns_from_reader(
        self,
        reader: ReaderLike,
        batch_size: Optional[int],
    ) -> Tuple[FragmentMetadata, LanceSchema]: ...
    def add_columns(
        self,
        transforms: Dict[str, str] | BatchUDF | ReaderLike,
        read_columns: Optional[List[str]],
        batch_size: Optional[int],
    ) -> Tuple[FragmentMetadata, LanceSchema]: ...
    def delete(self, predicate: str) -> Optional[_Fragment]: ...
    def schema(self) -> pa.Schema: ...
    def data_files(self) -> List[DataFile]: ...
    def deletion_file(self) -> Optional[str]: ...
    @property
    def physical_rows(self) -> int: ...
    @property
    def num_deletions(self) -> int: ...

def iops_counter() -> int: ...
def bytes_read_counter() -> int: ...
def _write_dataset(
    reader: pa.RecordBatchReader, uri: str | Path | _Dataset, params: Dict[str, Any]
): ...
def _write_fragments(
    dataset_uri: str | Path | _Dataset,
    reader: ReaderLike,
    mode: str,
    max_rows_per_file: int,
    max_rows_per_group: int,
    max_bytes_per_file: int,
    progress: Optional[FragmentWriteProgress],
    data_storage_version: Optional[str],
    storage_options: Optional[Dict[str, str]],
    enable_move_stable_row_ids: bool,
): ...
def _write_fragments_transaction(
    dataset_uri: str | Path | _Dataset,
    reader: ReaderLike,
    mode: str,
    max_rows_per_file: int,
    max_rows_per_group: int,
    max_bytes_per_file: int,
    progress: Optional[FragmentWriteProgress],
    data_storage_version: Optional[str],
    storage_options: Optional[Dict[str, str]],
    enable_move_stable_row_ids: bool,
) -> Transaction: ...
def _json_to_schema(schema_json: str) -> pa.Schema: ...
def _schema_to_json(schema: pa.Schema) -> str: ...

class _Hnsw:
    @staticmethod
    def build(
        vectors_array: Iterator[pa.Array],
        max_level: int,
        m: int,
        ef_construction: int,
    ): ...
    def to_lance_file(self, file_path: str): ...
    def vectors(self) -> pa.Array: ...

class _KMeans:
    def __init__(
        self,
        k: int,
        metric_type: str,
        max_iters: int,
        centroids_arr: Optional[pa.FixedSizeListArray] = None,
    ): ...
    def fit(self, data: pa.FixedSizeListArray): ...
    def predict(self, data: pa.FixedSizeListArray) -> pa.UInt32Array: ...
    def centroids(
        self,
    ) -> Union[pa.FixedShapeTensorType, pa.FixedSizeListType | None]: ...

class BFloat16:
    def __init__(self, value: float) -> None: ...
    @classmethod
    def from_bytes(cls, bytes: bytes) -> BFloat16: ...
    def as_float(self) -> float: ...
    def __lt__(self, other: BFloat16) -> bool: ...
    def __le__(self, other: BFloat16) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __gt__(self, other: BFloat16) -> bool: ...
    def __ge__(self, other: BFloat16) -> bool: ...

def bfloat16_array(values: List[str | None]) -> BFloat16Array: ...

__version__: str
language_model_home: Callable[[], str]
