#  Copyright 2023 Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import random
import string
import time

import lance
import numpy as np
import pyarrow as pa
import pytest
from lance.vector import vec_to_table


def create_table(nvec=10000, ndim=768):
    mat = np.random.randn(nvec, ndim)
    price = (np.random.rand(nvec) + 1) * 100

    def gen_str(n):
        return "".join(random.choices(string.ascii_letters + string.digits, k=n))

    meta = np.array([gen_str(1000) for _ in range(nvec)])
    tbl = (
        vec_to_table(data=mat)
        .append_column("price", pa.array(price))
        .append_column("meta", pa.array(meta))
    )
    return tbl


@pytest.fixture()
def dataset(tmp_path):
    tbl = create_table()
    yield lance.write_dataset(tbl, tmp_path)


@pytest.fixture()
def indexed_dataset(tmp_path):
    tbl = create_table()
    dataset = lance.write_dataset(tbl, tmp_path)
    yield dataset.create_index(
        "vector", index_type="IVF_PQ", num_partitions=32, num_sub_vectors=16
    )


def run(ds):
    q = np.random.randn(768)
    project = [None, ["price"], ["vector"], ["vector", "meta"]]
    refine = [None, 1, 2]
    # filters = [None, pc.field("price") > 50.0]
    times = []

    for columns in project:

        expected_columns = []
        if columns is None:
            expected_columns.extend(ds.schema.names)
        else:
            expected_columns.extend(columns)
        for c in ["vector", "score"]:
            if c not in expected_columns:
                expected_columns.append(c)

        for rf in refine:
            # for filter_ in filters:
            start = time.time()
            rs = ds.to_table(
                columns=columns,
                # filter=filter_,
                nearest={
                    "column": "vector",
                    "q": q,
                    "k": 10,
                    "nprobes": 1,
                    "refine_factor": rf,
                },
            )
            end = time.time()
            times.append(end - start)
            assert rs.column_names == expected_columns
            assert len(rs) == 10
            scores = rs["score"].to_numpy()
            assert (scores.max() - scores.min()) > 1e-6
    return times


def test_flat(dataset):
    print(run(dataset))


@pytest.mark.skipif(
    (os.uname().sysname == "Darwin") and (os.uname().machine != "arm64"),
    reason="no neon on GHA",
)
def test_ann(indexed_dataset):
    print(run(indexed_dataset))
