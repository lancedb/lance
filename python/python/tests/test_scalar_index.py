# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import os
import random
import shutil
import string
import zipfile
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
        .append_column("doc2", pa.array(doc, pa.string()))
        .append_column("id", pa.array(range(nvec)))
    )
    return tbl


def set_language_model_path():
    os.environ["LANCE_LANGUAGE_MODEL_HOME"] = os.path.join(
        os.path.dirname(__file__), "models"
    )


@pytest.fixture()
def lindera_ipadic():
    set_language_model_path()
    model_path = os.path.join(os.path.dirname(__file__), "models", "lindera", "ipadic")
    cwd = os.getcwd()
    try:
        os.chdir(model_path)
        with zipfile.ZipFile("main.zip", "r") as zip_ref:
            zip_ref.extractall()
        os.chdir(cwd)
        yield
    finally:
        shutil.rmtree(os.path.join(model_path, "main"))


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


def test_indexed_between(tmp_path):
    dataset = lance.write_dataset(pa.table({"val": range(100)}), tmp_path)
    dataset.create_scalar_index("val", index_type="BTREE")

    scanner = dataset.scanner(filter="val BETWEEN 10 AND 20", prefilter=True)

    assert "MaterializeIndex" in scanner.explain_plan()

    actual_data = scanner.to_table()
    assert actual_data.num_rows == 11

    scanner = dataset.scanner(filter="val >= 10 AND val <= 20", prefilter=True)

    assert "MaterializeIndex" in scanner.explain_plan()

    actual_data = scanner.to_table()
    assert actual_data.num_rows == 11


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
    ints = pa.array([i * 10 for i in range(100 * 1024)])
    table = pa.Table.from_arrays([ints], ["int"])
    dataset = lance.write_dataset(table, tmp_path)

    # Should succeed
    dataset.create_scalar_index("int", index_type="BTREE")

    try:
        # Should fail if we intentionally use a very small memory pool
        os.environ["LANCE_MEM_POOL_SIZE"] = "1024"
        with pytest.raises(Exception):
            dataset.create_scalar_index("int", index_type="BTREE", replace=True)

        # Should succeed again since bypassing spilling takes precedence
        os.environ["LANCE_BYPASS_SPILLING"] = "1"
        dataset.create_scalar_index("int", index_type="BTREE", replace=True)
    finally:
        del os.environ["LANCE_MEM_POOL_SIZE"]
        if "LANCE_BYPASS_SPILLING" in os.environ:
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
    assert results.num_rows > 0
    results = results.column(0)
    for row in results:
        assert query in row.as_py()


def test_filter_with_fts_index(dataset):
    dataset.create_scalar_index("doc", index_type="INVERTED", with_position=False)
    row = dataset.take(indices=[0], columns=["doc"])
    query = row.column(0)[0].as_py()
    query = query.split(" ")[0]
    results = dataset.scanner(
        filter=f"doc = '{query}'",
        prefilter=True,
    ).to_table()
    assert results.num_rows > 0
    results = results["doc"]
    for row in results:
        assert query == row.as_py()


def test_indexed_filter_with_fts_index(tmp_path):
    data = pa.table(
        {
            "text": [
                "Frodo was a puppy",
                "There were several kittens playing",
                "Frodo was a happy puppy",
                "Frodo was a very happy puppy",
            ],
            "sentiment": ["neutral", "neutral", "positive", "positive"],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    ds.create_scalar_index("text", "INVERTED")
    ds.create_scalar_index("sentiment", "BITMAP")

    # append more data to test flat FTS
    data = pa.table(
        {
            "text": ["flat", "search"],
            "sentiment": ["positive", "positive"],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="append")

    results = ds.to_table(
        full_text_query="puppy",
        filter="sentiment='positive'",
        prefilter=True,
        with_row_id=True,
    )
    assert results["_rowid"].to_pylist() == [2, 3]


def test_fts_with_postfilter(tmp_path):
    tab = pa.table({"text": ["Frodo the puppy"] * 100, "id": range(100)})
    dataset = lance.write_dataset(tab, tmp_path)
    dataset.create_scalar_index("text", index_type="INVERTED", with_position=False)

    results = dataset.to_table(
        full_text_query="Frodo", filter="id = 7", prefilter=False
    )
    assert results.num_rows == 1

    dataset.create_scalar_index("id", index_type="BTREE")

    results = dataset.to_table(
        full_text_query="Frodo", filter="id = 7", prefilter=False
    )

    assert results.num_rows == 1


def test_fts_with_other_str_scalar_index(dataset):
    dataset.create_scalar_index("doc", index_type="INVERTED", with_position=False)
    dataset.create_scalar_index("doc2", index_type="BTREE")

    row = dataset.take(indices=[0], columns=["doc"])
    query = row.column(0)[0].as_py()
    query = query.split(" ")[0]

    assert dataset.to_table(full_text_query=query).num_rows > 0


def test_fts_all_deleted(dataset):
    dataset.create_scalar_index("doc", index_type="INVERTED", with_position=False)
    first_row_doc = dataset.take(indices=[0], columns=["doc"]).column(0)[0].as_py()
    dataset.delete(f"doc = '{first_row_doc}'")
    dataset.to_table(full_text_query=first_row_doc)


def test_indexed_filter_with_fts_index_with_lindera_ipadic_jp_tokenizer(
    tmp_path, lindera_ipadic
):
    os.environ["LANCE_LANGUAGE_MODEL_HOME"] = os.path.join(
        os.path.dirname(__file__), "models"
    )
    data = pa.table(
        {
            "text": [
                "成田国際空港",
                "東京国際空港",
                "羽田空港",
            ],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    ds.create_scalar_index("text", "INVERTED", base_tokenizer="lindera/ipadic")

    results = ds.to_table(
        full_text_query="成田",
        prefilter=True,
        with_row_id=True,
    )
    assert results["_rowid"].to_pylist() == [0]


def test_lindera_ipadic_jp_tokenizer_invalid_user_dict_path(tmp_path, lindera_ipadic):
    data = pa.table(
        {
            "text": [
                "成田国際空港",
            ],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    with pytest.raises(OSError):
        ds.create_scalar_index(
            "text", "INVERTED", base_tokenizer="lindera/invalid_dict"
        )


def test_lindera_ipadic_jp_tokenizer_csv_user_dict_without_type(
    tmp_path, lindera_ipadic
):
    data = pa.table(
        {
            "text": [
                "成田国際空港",
            ],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    with pytest.raises(OSError):
        ds.create_scalar_index(
            "text", "INVERTED", base_tokenizer="lindera/invalid_dict2"
        )


def test_lindera_ipadic_jp_tokenizer_csv_user_dict(tmp_path, lindera_ipadic):
    data = pa.table(
        {
            "text": [
                "成田国際空港",
                "東京国際空港",
                "羽田空港",
            ],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    ds.create_scalar_index("text", "INVERTED", base_tokenizer="lindera/user_dict")
    results = ds.to_table(
        full_text_query="成田",
        prefilter=True,
        with_row_id=True,
    )
    assert len(results) == 0
    results = ds.to_table(
        full_text_query="成田国際空港",
        prefilter=True,
        with_row_id=True,
    )
    assert results["_rowid"].to_pylist() == [0]


def test_lindera_ipadic_jp_tokenizer_bin_user_dict(tmp_path, lindera_ipadic):
    data = pa.table(
        {
            "text": [
                "成田国際空港",
            ],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    ds.create_scalar_index("text", "INVERTED", base_tokenizer="lindera/user_dict2")


def test_jieba_tokenizer(tmp_path):
    set_language_model_path()
    data = pa.table(
        {
            "text": ["我们都有光明的前途", "光明的前途"],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    ds.create_scalar_index("text", "INVERTED", base_tokenizer="jieba/default")
    results = ds.to_table(
        full_text_query="我们",
        prefilter=True,
        with_row_id=True,
    )
    assert results["_rowid"].to_pylist() == [0]


def test_jieba_invalid_user_dict_tokenizer(tmp_path):
    set_language_model_path()
    data = pa.table(
        {
            "text": [
                "我们都有光明的前途",
            ],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    with pytest.raises(OSError):
        ds.create_scalar_index("text", "INVERTED", base_tokenizer="jieba/invalid_dict")


def test_jieba_invalid_main_dict_tokenizer(tmp_path):
    set_language_model_path()
    data = pa.table(
        {
            "text": [
                "我们都有光明的前途",
            ],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    with pytest.raises(OSError):
        ds.create_scalar_index("text", "INVERTED", base_tokenizer="jieba/invalid_dict2")


def test_jieba_user_dict_tokenizer(tmp_path):
    set_language_model_path()
    data = pa.table(
        {
            "text": ["我们都有光明的前途", "光明的前途"],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    ds.create_scalar_index("text", "INVERTED", base_tokenizer="jieba/user_dict")
    results = ds.to_table(
        full_text_query="的前",
        prefilter=True,
        with_row_id=True,
    )
    assert len(results) == 0
    results = ds.to_table(
        full_text_query="光明的前途",
        prefilter=True,
        with_row_id=True,
    )
    assert results["_rowid"].to_pylist() == [1, 0]


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


def test_ngram_index(tmp_path: Path):
    """Test create ngram index"""
    tbl = pa.Table.from_arrays(
        [
            pa.array(
                [["apple", "apples", "banana", "coconut"][i % 4] for i in range(100)]
            )
        ],
        names=["words"],
    )
    dataset = lance.write_dataset(tbl, tmp_path / "dataset")
    dataset.create_scalar_index("words", index_type="NGRAM")
    indices = dataset.list_indices()
    assert len(indices) == 1
    assert indices[0]["type"] == "NGram"

    scan_plan = dataset.scanner(filter="contains(words, 'apple')").explain_plan(True)
    assert "MaterializeIndex" in scan_plan

    assert dataset.to_table(filter="contains(words, 'apple')").num_rows == 50
    assert dataset.to_table(filter="contains(words, 'banana')").num_rows == 25
    assert dataset.to_table(filter="contains(words, 'coconut')").num_rows == 25
    assert dataset.to_table(filter="contains(words, 'apples')").num_rows == 25
    assert (
        dataset.to_table(
            filter="contains(words, 'apple') AND contains(words, 'banana')"
        ).num_rows
        == 0
    )
    assert (
        dataset.to_table(
            filter="contains(words, 'apple') OR contains(words, 'banana')"
        ).num_rows
        == 75
    )


def test_null_handling(tmp_path: Path):
    tbl = pa.table(
        {
            "x": [1, 2, None, 3],
        }
    )
    dataset = lance.write_dataset(tbl, tmp_path / "dataset")

    def check(has_index: bool):
        assert dataset.to_table(filter="x IS NULL").num_rows == 1
        assert dataset.to_table(filter="x IS NOT NULL").num_rows == 3
        assert dataset.to_table(filter="x > 0").num_rows == 3
        assert dataset.to_table(filter="x < 5").num_rows == 3
        assert dataset.to_table(filter="x IN (1, 2)").num_rows == 2
        # Note: there is a bit of discrepancy here.  Datafusion does not consider
        # NULL==NULL when doing an IN operation due to classic SQL shenanigans.
        # We should decide at some point which behavior we want and make this
        # consistent.
        if has_index:
            assert dataset.to_table(filter="x IN (1, 2, NULL)").num_rows == 3
        else:
            assert dataset.to_table(filter="x IN (1, 2, NULL)").num_rows == 2

    check(False)
    dataset.create_scalar_index("x", index_type="BITMAP")
    check(True)
    dataset.create_scalar_index("x", index_type="BTREE")
    check(True)


def test_scalar_index_with_nulls(tmp_path):
    # Create a test dataframe with 50% null values.
    test_table_size = 10_000
    test_table = pa.table(
        {
            "item_id": list(range(test_table_size)),
            "inner_id": list(range(test_table_size)),
            "category": ["a", None] * (test_table_size // 2),
            "numeric_int": [1, None] * (test_table_size // 2),
            "numeric_float": [0.1, None] * (test_table_size // 2),
            "boolean_col": [True, None] * (test_table_size // 2),
            "timestamp_col": [datetime(2023, 1, 1), None] * (test_table_size // 2),
            "ngram_col": ["apple", None] * (test_table_size // 2),
        }
    )
    ds = lance.write_dataset(test_table, tmp_path)
    ds.create_scalar_index("inner_id", index_type="BTREE")
    ds.create_scalar_index("category", index_type="BTREE")
    ds.create_scalar_index("boolean_col", index_type="BTREE")
    ds.create_scalar_index("timestamp_col", index_type="BTREE")
    ds.create_scalar_index("ngram_col", index_type="NGRAM")
    # Test querying with filters on columns with nulls.
    k = test_table_size // 2
    result = ds.to_table(filter="category = 'a'", limit=k)
    assert len(result) == k
    # Booleans should be stored as strings in the table for backwards compatibility.
    result = ds.to_table(filter="boolean_col IS TRUE", limit=k)
    assert len(result) == k
    result = ds.to_table(filter="timestamp_col IS NOT NULL", limit=k)
    assert len(result) == k

    # Ensure ngram index works with nulls
    result = ds.to_table(filter="ngram_col = 'apple'")
    assert len(result) == k
    result = ds.to_table(filter="ngram_col IS NULL")
    assert len(result) == k
    result = ds.to_table(filter="contains(ngram_col, 'appl')")
    assert len(result) == k


def test_label_list_index(tmp_path: Path):
    tags = pa.array(["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7"])
    tag_list = pa.ListArray.from_arrays([0, 2, 4], tags)
    tbl = pa.Table.from_arrays([tag_list], names=["tags"])
    dataset = lance.write_dataset(tbl, tmp_path / "dataset")
    dataset.create_scalar_index("tags", index_type="LABEL_LIST")
    indices = dataset.list_indices()
    assert len(indices) == 1
    assert indices[0]["type"] == "LabelList"


def test_create_index_empty_dataset(tmp_path: Path):
    # Creating an index on an empty dataset is (currently) not terribly useful but
    # we shouldn't return strange errors.
    schema = pa.schema(
        [
            pa.field("btree", pa.int32()),
            pa.field("bitmap", pa.int32()),
            pa.field("label_list", pa.list_(pa.string())),
            pa.field("inverted", pa.string()),
            pa.field("ngram", pa.string()),
        ]
    )
    ds = lance.write_dataset([], tmp_path, schema=schema)

    for index_type in ["BTREE", "BITMAP", "LABEL_LIST", "INVERTED", "NGRAM"]:
        ds.create_scalar_index(index_type.lower(), index_type=index_type)

    # Make sure the empty index doesn't cause searches to fail
    ds.insert(
        pa.table(
            {
                "btree": pa.array([1], pa.int32()),
                "bitmap": pa.array([1], pa.int32()),
                "label_list": [["foo", "bar"]],
                "inverted": ["blah"],
                "ngram": ["apple"],
            }
        )
    )

    def test_searches():
        assert ds.to_table(filter="btree = 1").num_rows == 1
        assert ds.to_table(filter="btree = 0").num_rows == 0
        assert ds.to_table(filter="bitmap = 1").num_rows == 1
        assert ds.to_table(filter="bitmap = 0").num_rows == 0
        assert ds.to_table(filter="array_has_any(label_list, ['foo'])").num_rows == 1
        assert ds.to_table(filter="array_has_any(label_list, ['oof'])").num_rows == 0
        assert ds.to_table(filter="inverted = 'blah'").num_rows == 1
        assert ds.to_table(filter="inverted = 'halb'").num_rows == 0
        assert ds.to_table(filter="contains(ngram, 'apple')").num_rows == 1
        assert ds.to_table(filter="contains(ngram, 'banana')").num_rows == 0
        assert ds.to_table(filter="ngram = 'apple'").num_rows == 1

    test_searches()

    # Make sure fetching index stats on empty index is ok
    for idx in ds.list_indices():
        ds.stats.index_stats(idx["name"])

    # Make sure updating empty indices is ok
    ds.optimize.optimize_indices()

    # Finally, make sure we can still search after updating
    test_searches()


def test_optimize_no_new_data(tmp_path: Path):
    tbl = pa.table(
        {
            "btree": pa.array([None], pa.int64()),
            "bitmap": pa.array([None], pa.int64()),
            "ngram": pa.array([None], pa.string()),
        }
    )
    dataset = lance.write_dataset(tbl, tmp_path)
    dataset.create_scalar_index("btree", index_type="BTREE")
    dataset.create_scalar_index("bitmap", index_type="BITMAP")
    dataset.create_scalar_index("ngram", index_type="NGRAM")

    assert dataset.to_table(filter="btree IS NULL").num_rows == 1
    assert dataset.to_table(filter="bitmap IS NULL").num_rows == 1
    assert dataset.to_table(filter="ngram IS NULL").num_rows == 1

    dataset.insert([], schema=tbl.schema)
    dataset.optimize.optimize_indices()

    assert dataset.to_table(filter="btree IS NULL").num_rows == 1
    assert dataset.to_table(filter="bitmap IS NULL").num_rows == 1
    assert dataset.to_table(filter="ngram IS NULL").num_rows == 1

    dataset.insert(pa.table({"btree": [2]}))
    dataset.optimize.optimize_indices()

    assert dataset.to_table(filter="btree IS NULL").num_rows == 1
    assert dataset.to_table(filter="bitmap IS NULL").num_rows == 2
    assert dataset.to_table(filter="ngram IS NULL").num_rows == 2

    dataset.insert(pa.table({"bitmap": [2]}))
    dataset.optimize.optimize_indices()

    assert dataset.to_table(filter="btree IS NULL").num_rows == 2
    assert dataset.to_table(filter="bitmap IS NULL").num_rows == 2
    assert dataset.to_table(filter="ngram IS NULL").num_rows == 3

    dataset.insert(pa.table({"ngram": ["apple"]}))

    assert dataset.to_table(filter="btree IS NULL").num_rows == 3
    assert dataset.to_table(filter="bitmap IS NULL").num_rows == 3
    assert dataset.to_table(filter="ngram IS NULL").num_rows == 3


def test_drop_index(tmp_path):
    test_table_size = 100
    test_table = pa.table(
        {
            "btree": list(range(test_table_size)),
            "bitmap": list(range(test_table_size)),
            "fts": ["a" for _ in range(test_table_size)],
            "ngram": ["a" for _ in range(test_table_size)],
        }
    )
    ds = lance.write_dataset(test_table, tmp_path)
    ds.create_scalar_index("btree", index_type="BTREE")
    ds.create_scalar_index("bitmap", index_type="BITMAP")
    ds.create_scalar_index("fts", index_type="INVERTED")
    ds.create_scalar_index("ngram", index_type="NGRAM")

    assert len(ds.list_indices()) == 4

    # Attempt to drop index (name does not exist)
    with pytest.raises(RuntimeError, match="index not found"):
        ds.drop_index("nonexistent_name")

    for idx in ds.list_indices():
        idx_name = idx["name"]
        ds.drop_index(idx_name)

    assert len(ds.list_indices()) == 0

    # Ensure we can still search columns
    assert ds.to_table(filter="btree = 1").num_rows == 1
    assert ds.to_table(filter="bitmap = 1").num_rows == 1
    assert ds.to_table(filter="fts = 'a'").num_rows == test_table_size
    assert ds.to_table(filter="contains(ngram, 'a')").num_rows == test_table_size
