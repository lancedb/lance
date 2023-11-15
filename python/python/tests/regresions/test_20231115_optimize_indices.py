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

import random
import uuid
from pathlib import Path

import lance
import pyarrow as pa
import pytest


def random_table(n):
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 128)),
        ]
    )
    return pa.Table.from_pylist(
        [
            {
                "id": uuid.uuid4().hex,
                "vector": [random.random() for _ in range(128)],
            }
            for _ in range(n)
        ],
        schema=schema,
    )


@pytest.mark.xfail
def test_optimize_indices_corrupting_indices_when_fragments_are_dropped(tmp_path: Path):
    lance.write_dataset(random_table(0), tmp_path)
    ds = lance.write_dataset(random_table(512), tmp_path, mode="append")

    ds.create_index(
        "vector", "IVF_PQ", metric="L2", num_partitions=1, num_sub_vectors=16
    )

    second_append = random_table(2)
    ds = lance.write_dataset(second_append, tmp_path, mode="append")

    ds.optimize.optimize_indices()

    ids_to_delete = second_append["id"].to_pylist()
    quoted_ids = ", ".join([f'"{id}"' for id in ids_to_delete])
    ds = lance.dataset(tmp_path)
    ds.delete(f"`id` IN ({quoted_ids})")

    second_append = random_table(2)
    ds = lance.write_dataset(second_append, tmp_path, mode="append")

    ds.optimize.optimize_indices()

    for _ in range(1000):
        query = [random.random() for _ in range(128)]
        ds.to_table(nearest={"column": "vector", "q": query, "use_index": True})
