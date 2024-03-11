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

import random
import string
from datetime import date, datetime, timedelta

import lance
import numpy as np
import pyarrow as pa
import pytest
from lance.vector import vec_to_table


def create_table(nvec=1000, ndim=128):
    mat = np.random.randn(nvec, ndim)
    price = np.random.rand(nvec) * 100

    def gen_str(n):
        return "".join(random.choices(string.ascii_letters + string.digits, k=n))

    meta = np.array([gen_str(100) for _ in range(nvec)])
    tbl = (
        vec_to_table(data=mat)
        .append_column("price", pa.array(price))
        .append_column("meta", pa.array(meta))
        .append_column("id", pa.array(range(nvec)))
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
    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=2,
    )
    dataset.create_scalar_index("meta", index_type="BTREE")
    return dataset


@pytest.fixture()
def data_table(indexed_dataset: lance.LanceDataset):
    return indexed_dataset.scanner().to_table()


def test_load_indices(indexed_dataset: lance.LanceDataset):
    indices = indexed_dataset.list_indices()
    vec_idx = next(idx for idx in indices if idx["type"] == "Vector")
    scalar_idx = next(idx for idx in indices if idx["type"] == "Scalar")
    assert vec_idx is not None
    assert scalar_idx is not None


def test_indexed_scalar_scan(indexed_dataset: lance.LanceDataset, data_table: pa.Table):
    sample_meta = data_table["meta"][50]
    expected_price = data_table["price"][50]

    for filter in [f"meta='{sample_meta}'", f"price >= 0 AND meta='{sample_meta}'"]:
        scanner = indexed_dataset.scanner(
            columns=["price"], filter=filter, prefilter=True
        )

        assert "MaterializeIndex" in scanner.explain_plan()

        actual_data = scanner.to_table()
        assert actual_data.num_rows == 1
        assert actual_data.num_columns == 1

        actual_price = actual_data["price"][0]
        assert actual_price == expected_price


def test_temporal_index(tmp_path):
    # Timestamps
    now = datetime.now()
    today = date.today()
    table = pa.Table.from_pydict({
        "ts": [now - timedelta(days=i) for i in range(100)],
        "date": [today - timedelta(days=i) for i in range(100)],
        "time": pa.array([i for i in range(100)], type=pa.time32("s")),
        "id": [i for i in range(100)],
    })
    dataset = lance.write_dataset(table, tmp_path)
    dataset.create_scalar_index("ts", index_type="BTREE")
    dataset.create_scalar_index("date", index_type="BTREE")
    dataset.create_scalar_index("time", index_type="BTREE")

    # Timestamp
    half_now = now - timedelta(days=50)
    scanner = dataset.scanner(filter=f"ts > timestamp '{half_now}'", scan_in_order=True)
    assert "MaterializeIndex" in scanner.explain_plan(True)
    assert scanner.to_table() == table.slice(0, 50)

    # Date
    half_toady = today - timedelta(days=50)
    scanner = dataset.scanner(filter=f"date > date '{half_toady}'", scan_in_order=True)
    assert "MaterializeIndex" in scanner.explain_plan(True)
    assert scanner.to_table() == table.slice(0, 50)


def test_indexed_vector_scan(indexed_dataset: lance.LanceDataset, data_table: pa.Table):
    # We query for row 25 but our prefilter only
    # considers row 50 so we should get row 50
    query_vec = data_table["vector"][25].as_py()
    sample_meta = data_table["meta"][50]
    expected_price = data_table["price"][50]

    def check_result(table: pa.Table):
        assert table.num_rows == 1
        assert table.num_columns == 2

        actual_price = table["price"][0]
        assert actual_price == expected_price

    scanner = indexed_dataset.scanner(
        nearest={"column": "vector", "q": query_vec, "k": 5, "nprobes": 4},
        columns=["price"],
        prefilter=True,
        filter=f"meta='{sample_meta}'",
    )

    assert "ScalarIndexQuery" in scanner.explain_plan()

    check_result(scanner.to_table())

    scanner = indexed_dataset.scanner(
        nearest={"column": "vector", "q": query_vec, "k": 5, "nprobes": 4},
        columns=["price"],
        prefilter=True,
        filter=f"price >= 0 AND meta='{sample_meta}'",
    )

    assert "MaterializeIndex" in scanner.explain_plan()

    check_result(scanner.to_table())


# Post filtering does not use scalar indices.  This test merely confirms
# that post filtering is still being applied even if the query could be
# satisfied with scalar indices
def test_indexed_vector_scan_postfilter(
    indexed_dataset: lance.LanceDataset, data_table: pa.Table
):
    query_vec = data_table["vector"][25].as_py()
    sample_meta = data_table["meta"][50]

    scanner = indexed_dataset.scanner(
        nearest={"column": "vector", "q": query_vec, "k": 1, "nprobes": 4},
        columns=["price"],
        prefilter=False,
        filter=f"meta='{sample_meta}'",
    )

    assert scanner.to_table().num_rows == 0
