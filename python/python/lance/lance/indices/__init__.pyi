#  Copyright (c) 2024. Lance Developers
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

from datetime import datetime
from typing import Optional

import pyarrow as pa

from ...dataset import Index

class IndexConfig:
    index_type: str
    config: str

class IndexSegment:
    uuid: str
    fragment_ids: set[int]
    index_version: int

    def __repr__(self) -> str: ...

class IndexSegmentPlan:
    staging_index_uuid: str
    segment: IndexSegment
    partial_indices: list[Index]
    estimated_bytes: int

    def __repr__(self) -> str: ...

def train_ivf_model(
    dataset,
    column: str,
    dimension: int,
    num_partitions: int,
    distance_type: str,
    sample_rate: int,
    max_iters: int,
    fragment_ids: Optional[list[int]] = None,
) -> pa.Array: ...
def train_pq_model(
    dataset,
    column: str,
    dimension: int,
    num_subvectors: int,
    distance_type: str,
    sample_rate: int,
    max_iters: int,
    ivf_model: pa.Array,
    fragment_ids: Optional[list[int]] = None,
) -> pa.Array: ...
def transform_vectors(
    dataset,
    column: str,
    dimension: int,
    num_subvectors: int,
    distance_type: str,
    ivf_centroids: pa.Array,
    pq_codebook: pa.Array,
    dst_uri: str,
): ...

class IndexSegmentDescription:
    uuid: str
    dataset_version_at_last_update: int
    fragment_ids: set[int]
    index_version: int
    created_at: Optional[datetime]
    size_bytes: Optional[int]

    def __repr__(self) -> str: ...

class IndexDescription:
    name: str
    type_url: str
    index_type: str
    num_rows_indexed: int
    fields: list[int]
    field_names: list[str]
    segments: list[IndexSegmentDescription]
    details: dict
    total_size_bytes: Optional[int]

    def __repr__(self) -> str: ...
