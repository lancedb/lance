# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import os
import random
import string
from datetime import date, datetime, timedelta
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
from lance.vector import vec_to_table


def create_table(nvec=1000, ndim=128):
    mat = np.random.randn(nvec, ndim)
    price = np.random.rand(nvec) * 100

    def gen_str(n, split="", char_set=string.ascii_letters + string.digits):
        return "".join(random.choices(char_set, k=n))

    meta = np.array([gen_str(100) for _ in range(nvec)])
    doc = [gen_str(10, " ", string.ascii_letters) for _ in range(nvec)]
    tbl = (
        vec_to_table(data=mat)
        .append_column("price", pa.array(price))
        .append_column("meta", pa.array(meta))
        .append_column("doc", pa.array(doc, pa.large_string()))
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
    vec_idx = next(idx for idx in indices if idx["type"] == "IVF_PQ")
    scalar_idx = next(idx for idx in indices if idx["type"] == "BTree")
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
    table = pa.Table.from_pydict(
        {
            "ts": [now - timedelta(days=i) for i in range(100)],
            "date": [today - timedelta(days=i) for i in range(100)],
            "time": pa.array([i for i in range(100)], type=pa.time32("s")),
            "id": [i for i in range(100)],
        }
    )
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


def test_all_null_chunk(tmp_path):
    def gen_string(idx: int):
        if idx % 2 == 0:
            return None
        return f"string-{idx}"

    strings = pa.array([gen_string(i) for i in range(100 * 1024)], pa.string())
    table = pa.Table.from_arrays([strings], ["str"])
    dataset = lance.write_dataset(table, tmp_path)
    dataset.create_scalar_index("str", index_type="BTREE")
    scanner = dataset.scanner(prefilter=True, filter="str='string-501'")

    assert scanner.to_table().num_rows == 1


# We currently allow the memory pool default size to be configured with an
# environment variable.  This test ensures that the environment variable
# is respected.
def test_lance_mem_pool_env_var(tmp_path):
    strings = pa.array([f"string-{i}" * 10 for i in range(100 * 1024)])
    table = pa.Table.from_arrays([strings], ["str"])
    dataset = lance.write_dataset(table, tmp_path)

    # Should succeed
    dataset.create_scalar_index("str", index_type="BTREE")

    try:
        # Should fail if we intentionally use a very small memory pool
        os.environ["LANCE_MEM_POOL_SIZE"] = "1024"
        with pytest.raises(Exception):
            dataset.create_scalar_index("str", index_type="BTREE", replace=True)

        # Should succeed again since bypassing spilling takes precedence
        os.environ["LANCE_BYPASS_SPILLING"] = "1"
        dataset.create_scalar_index("str", index_type="BTREE", replace=True)
    finally:
        del os.environ["LANCE_MEM_POOL_SIZE"]
        del os.environ["LANCE_BYPASS_SPILLING"]


@pytest.mark.parametrize("with_position", [True, False])
def test_full_text_search(dataset, with_position):
    dataset.create_scalar_index(
        "doc", index_type="INVERTED", with_position=with_position
    )
    row = dataset.take(indices=[0], columns=["doc"])
    query = row.column(0)[0].as_py()
    query = query.split(" ")[0]
    results = dataset.scanner(
        columns=["doc"],
        full_text_query=query,
    ).to_table()
    results = results.column(0)
    for row in results:
        assert query in row.as_py()


def test_bitmap_index(tmp_path: Path):
    """Test create bitmap index"""
    tbl = pa.Table.from_arrays(
        [pa.array([["a", "b", "c"][i % 3] for i in range(100)])], names=["a"]
    )
    dataset = lance.write_dataset(tbl, tmp_path / "dataset")
    dataset.create_scalar_index("a", index_type="BITMAP")
    indices = dataset.list_indices()
    assert len(indices) == 1
    assert indices[0]["type"] == "Bitmap"


def test_label_list_index(tmp_path: Path):
    tags = pa.array(["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7"])
    tag_list = pa.ListArray.from_arrays([0, 2, 4], tags)
    tbl = pa.Table.from_arrays([tag_list], names=["tags"])
    dataset = lance.write_dataset(tbl, tmp_path / "dataset")
    dataset.create_scalar_index("tags", index_type="LABEL_LIST")
    indices = dataset.list_indices()
    assert len(indices) == 1
    assert indices[0]["type"] == "LabelList"
