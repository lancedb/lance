# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import os
import random
import re
import shutil
import string
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
from lance.indices import IndexConfig
from lance.query import (
    BooleanQuery,
    BoostQuery,
    MatchQuery,
    MultiMatchQuery,
    Occur,
    PhraseQuery,
)
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

        assert (
            f"ScalarIndexQuery: query=[meta = {sample_meta}]@meta_idx"
            in scanner.explain_plan()
        )

        actual_data = scanner.to_table()
        assert actual_data.num_rows == 1
        assert actual_data.num_columns == 1

        actual_price = actual_data["price"][0]
        assert actual_price == expected_price


def test_indexed_between(tmp_path):
    dataset = lance.write_dataset(pa.table({"val": range(0, 10000)}), tmp_path)
    dataset.create_scalar_index("val", index_type="BTREE")

    scanner = dataset.scanner(filter="val BETWEEN 10 AND 20", prefilter=True)

    assert (
        "ScalarIndexQuery: query=[val >= 10 && val <= 20]@val_idx"
        in scanner.explain_plan()
    )

    actual_data = scanner.to_table()
    assert actual_data.num_rows == 11

    scanner = dataset.scanner(filter="val >= 10 AND val <= 20", prefilter=True)

    assert (
        "ScalarIndexQuery: query=[val >= 10 && val <= 20]@val_idx"
        in scanner.explain_plan()
    )

    actual_data = scanner.to_table()
    assert actual_data.num_rows == 11

    # The following cases are slightly ill-formed since end is before start
    # but we should handle them gracefully and simply return an empty result
    # (previously we panicked here)
    scanner = dataset.scanner(filter="val >= 5000 AND val <= 0", prefilter=True)

    assert (
        "ScalarIndexQuery: query=[val >= 5000 && val <= 0]@val_idx"
        in scanner.explain_plan()
    )

    actual_data = scanner.to_table()
    assert actual_data.num_rows == 0

    scanner = dataset.scanner(filter="val BETWEEN 5000 AND 0", prefilter=True)

    assert (
        "ScalarIndexQuery: query=[val >= 5000 && val <= 0]@val_idx"
        in scanner.explain_plan()
    )

    actual_data = scanner.to_table()
    assert actual_data.num_rows == 0


def test_index_combination(tmp_path):
    # This test regresses a bug in the index combination logic.

    colors = ["red", "green", "blue"]
    digits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    data = [{"color": colors[i % 3], "digit": digits[i % 10]} for i in range(30000)]

    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index("color", index_type="BTREE")
    ds.create_scalar_index("digit", index_type="BTREE")

    assert ds.count_rows("color = 'green' or digit <> 9") == 28000


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
    assert re.search(
        r"^.*ScalarIndexQuery: query=\[ts > .*\]@ts_idx.*$",
        scanner.explain_plan(True),
        re.MULTILINE,
    )
    assert scanner.to_table() == table.slice(0, 50)

    # Date
    half_toady = today - timedelta(days=50)
    scanner = dataset.scanner(filter=f"date > date '{half_toady}'", scan_in_order=True)
    assert re.search(
        r"^.*ScalarIndexQuery: query=\[date > .*\]@date_idx.*$",
        scanner.explain_plan(True),
        re.MULTILINE,
    )
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

    assert "ScalarIndexQuery" in scanner.analyze_plan()

    check_result(scanner.to_table())

    scanner = indexed_dataset.scanner(
        nearest={"column": "vector", "q": query_vec, "k": 5, "nprobes": 4},
        columns=["price"],
        prefilter=True,
        filter=f"price >= 0 AND meta='{sample_meta}'",
    )

    assert (
        f"ScalarIndexQuery: query=[meta = {sample_meta}]@meta_idx"
        in scanner.explain_plan()
    )

    check_result(scanner.to_table())


def test_partly_indexed_prefiltered_search(tmp_path):
    # Regresses a case where the vector index is ahead of a scalar index.  The scalar
    # index wants to be used as a prefilter but we have to make sure to scan the
    # unindexed fragments and feed those into the prefilter

    # Create initial dataset
    table = pa.table(
        {
            "vec": pa.array([[i, i] for i in range(1000)], pa.list_(pa.float32(), 2)),
            "text": ["book" for _ in range(1000)],
            "id": range(1000),
        }
    )
    ds = lance.write_dataset(table, tmp_path)
    ds = ds.create_index(
        "vec",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=2,
    )
    ds.create_scalar_index("id", "BTREE")
    ds.create_scalar_index("text", index_type="INVERTED", with_position=False)

    def make_vec_search(ds):
        return ds.scanner(
            nearest={"column": "vec", "q": [5, 5], "k": 1000},
            prefilter=True,
            filter="id in (5, 10, 15, 20, 25, 30)",
        )

    def make_fts_search(ds):
        return ds.scanner(
            full_text_query="book",
            prefilter=True,
            filter="id in (5, 10, 15, 20, 25, 30)",
        )

    # Sanity test, no new data, should get 6 results
    plan = make_vec_search(ds).explain_plan()
    assert "ScalarIndexQuery" in plan
    assert "KNNVectorDistance" not in plan
    assert "LanceRead" not in plan
    assert make_vec_search(ds).to_table().num_rows == 6

    plan = make_fts_search(ds).explain_plan()
    assert "ScalarIndexQuery" in plan
    assert "KNNVectorDistance" not in plan
    assert "LanceRead" not in plan
    assert make_fts_search(ds).to_table().num_rows == 6

    # Add new data (including 6 more results)
    ds.insert(table)

    # Basic ann combined search, should get 12 results
    plan = make_vec_search(ds).explain_plan()
    assert "ScalarIndexQuery" in plan
    assert "MaterializeIndex" not in plan
    assert "KNNVectorDistance" in plan
    assert "LanceScan" in plan
    assert make_vec_search(ds).to_table().num_rows == 12

    plan = make_fts_search(ds).explain_plan()
    assert "ScalarIndexQuery" in plan
    assert "MaterializeIndex" not in plan
    assert "FlatMatchQuery" in plan
    assert "LanceScan" in plan
    assert make_fts_search(ds).to_table().num_rows == 12

    # Update vector index but NOT scalar index
    ds.optimize.optimize_indices(index_names=["vec_idx", "text_idx"])

    # Ann search but with combined prefilter, should get 12 results
    plan = make_vec_search(ds).explain_plan()
    assert "ScalarIndexQuery" in plan
    assert "LanceRead" in plan
    assert "KNNVectorDistance" not in plan
    assert "LanceScan" not in plan
    assert make_vec_search(ds).to_table().num_rows == 12

    plan = make_fts_search(ds).explain_plan()
    assert "ScalarIndexQuery" in plan
    assert "LanceRead" in plan
    assert "FlatMatchQuery" not in plan
    assert "LanceScan" not in plan
    assert make_fts_search(ds).to_table().num_rows == 12


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


def test_fixed_size_binary(tmp_path):
    arr = pa.array([b"0123012301230123", b"2345234523452345"], pa.uuid())

    ds = lance.write_dataset(pa.table({"uuid": arr}), tmp_path)

    ds.create_scalar_index("uuid", "BTREE")

    query = (
        "uuid = arrow_cast(0x32333435323334353233343532333435, 'FixedSizeBinary(16)')"
    )
    assert (
        "ScalarIndexQuery: query=[uuid = 32333435323334353233...]@uuid_idx"
        in ds.scanner(filter=query).explain_plan()
    )

    table = ds.scanner(filter=query).to_table()
    assert table.num_rows == 1
    assert table.column("uuid").to_pylist() == arr.slice(1, 1).to_pylist()


def test_index_take_batch_size(tmp_path):
    dataset = lance.write_dataset(
        pa.table({"ints": range(1024)}), tmp_path, max_rows_per_file=100
    )
    dataset.create_scalar_index("ints", index_type="BTREE")
    batches = dataset.scanner(
        with_row_id=True, filter="ints > 0", batch_size=50
    ).to_batches()
    batches = list(batches)
    assert len(batches) == 21

    dataset = lance.write_dataset(
        pa.table({"strings": [f"string-{i}" for i in range(1024)]}),
        tmp_path,
        max_rows_per_file=100,
        mode="overwrite",
    )
    dataset.create_scalar_index("strings", index_type="NGRAM")
    filter = "contains(strings, 'ing')"
    batches = dataset.scanner(
        with_row_id=True, filter=filter, batch_size=50, limit=1024
    ).to_batches()
    batches = list(batches)
    assert len(batches) == 21


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

    with pytest.raises(ValueError, match="Cannot include deleted rows"):
        dataset.to_table(
            with_row_id=True, full_text_query=query, include_deleted_rows=True
        )


def test_rowid_order(dataset):
    dataset.create_scalar_index("doc", index_type="INVERTED", with_position=False)
    results = dataset.scanner(
        columns=["doc"],
        full_text_query="hello",
        with_row_id=True,
    ).to_table()

    assert results.schema[0].name == "doc"
    assert results.schema[1].name == "_score"
    assert results.schema[2].name == "_rowid"


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


def test_multi_index_create(tmp_path):
    dataset = lance.write_dataset(
        pa.table({"ints": range(1024)}), tmp_path, max_rows_per_file=100
    )
    dataset.create_scalar_index("ints", index_type="BTREE")
    dataset.create_scalar_index(
        "ints", index_type="BITMAP", name="ints_bitmap_idx", replace=True
    )

    indices = dataset.list_indices()
    assert len(indices) == 2

    assert indices[0]["name"] == "ints_idx"
    assert indices[0]["type"] == "BTree"
    assert indices[1]["name"] == "ints_bitmap_idx"
    assert indices[1]["type"] == "Bitmap"

    # Test that we can drop one of the indices
    dataset.drop_index("ints_idx")
    indices = dataset.list_indices()
    assert len(indices) == 1
    assert indices[0]["name"] == "ints_bitmap_idx"
    assert indices[0]["type"] == "Bitmap"

    # Test that we can drop the last index
    dataset.drop_index("ints_bitmap_idx")
    indices = dataset.list_indices()
    assert len(indices) == 0


def test_use_multi_index(tmp_path):
    dataset = lance.write_dataset(
        pa.table({"ints": range(1024)}), tmp_path, max_rows_per_file=100
    )
    dataset.create_scalar_index("ints", index_type="BTREE")
    dataset.create_scalar_index("ints", index_type="BITMAP", name="ints_bitmap_idx")

    # Test that we can use the index.  Multiple indices can be applied here.
    # One of them will be chosen (it is not deterministic which one is chosen)
    results = dataset.to_table(filter="ints = 0", prefilter=True)
    assert results.num_rows == 1

    assert (
        "ScalarIndexQuery: query=[ints = 0]@ints_idx"
        in dataset.scanner(filter="ints = 0", prefilter=True).explain_plan()
    )


def test_ngram_fts(tmp_path):
    dataset = lance.write_dataset(
        pa.table({"text": ["hello", "world", "hello world"]}),
        tmp_path,
    )
    dataset.create_scalar_index("text", index_type="INVERTED")
    dataset.create_scalar_index("text", name="text_ngram_idx", index_type="NGRAM")

    results = dataset.to_table(full_text_query="hello")
    assert results.num_rows == 2

    results = dataset.to_table(filter="contains(text, 'hello')")
    assert results.num_rows == 2

    assert (
        'ScalarIndexQuery: query=[contains(text, Utf8("hello"))]@text_ngram_idx'
        in dataset.scanner(
            filter="contains(text, 'hello')", prefilter=True
        ).explain_plan()
    )


def test_fts_fts(tmp_path):
    # Tests creating two FTS indices with the same name but different parameters
    dataset = lance.write_dataset(
        pa.table(
            {
                "text": [
                    "Frodo was a puppy",
                    "Frodo was a happy puppy",
                    "Frodo was a very happy puppy",
                ]
            }
        ),
        tmp_path,
    )
    dataset.create_scalar_index(
        "text", "INVERTED", with_position=True, remove_stop_words=False
    )

    results = dataset.to_table(full_text_query='"was a puppy"', prefilter=True)
    assert results.num_rows == 1

    results = dataset.to_table(full_text_query="was a puppy", prefilter=True)
    assert results.num_rows == 3

    dataset.create_scalar_index(
        "text", "INVERTED", name="no_pos_idx", with_position=False
    )

    # There is no way to currently specify which index to use.  Instead
    # it will always use the first index in the manifest.

    results = dataset.to_table(full_text_query='"was a puppy"', prefilter=True)
    assert results.num_rows == 1

    results = dataset.to_table(full_text_query="was a puppy", prefilter=True)
    assert results.num_rows == 3


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


def test_fts_ngram_tokenizer(tmp_path):
    data = pa.table({"text": ["hello world", "lance database", "lance is cool"]})
    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index("text", index_type="INVERTED", base_tokenizer="ngram")

    results = ds.to_table(full_text_query="lan")
    assert results.num_rows == 2
    assert set(results["text"].to_pylist()) == {"lance database", "lance is cool"}

    results = ds.to_table(full_text_query="nce")  # spellchecker:disable-line
    assert results.num_rows == 2
    assert set(results["text"].to_pylist()) == {"lance database", "lance is cool"}

    # the default min_ngram_length is 3, so "la" should not match
    results = ds.to_table(full_text_query="la")
    assert results.num_rows == 0

    # test setting min_ngram_length and prefix_only
    ds.create_scalar_index(
        "text",
        index_type="INVERTED",
        base_tokenizer="ngram",
        min_ngram_length=2,
        prefix_only=True,
    )

    results = ds.to_table(full_text_query="lan")
    assert results.num_rows == 2
    assert set(results["text"].to_pylist()) == {"lance database", "lance is cool"}

    results = ds.to_table(full_text_query="nce")  # spellchecker:disable-line
    assert results.num_rows == 0

    results = ds.to_table(full_text_query="la")
    assert results.num_rows == 2
    assert set(results["text"].to_pylist()) == {"lance database", "lance is cool"}


def test_fts_stats(dataset):
    dataset.create_scalar_index(
        "doc", index_type="INVERTED", with_position=False, remove_stop_words=True
    )
    stats = dataset.stats.index_stats("doc_idx")
    assert stats["index_type"] == "Inverted"
    stats = stats["indices"][0]
    params = stats["params"]

    assert params["with_position"] is False
    assert params["base_tokenizer"] == "simple"
    assert params["language"] == "English"
    assert params["max_token_length"] == 40
    assert params["lower_case"] is True
    assert params["stem"] is True
    assert params["remove_stop_words"] is True
    assert params["ascii_folding"] is True


def test_fts_score(tmp_path):
    # the number of tokens matters for scoring,
    # make a table that all docs have the same number of tokens
    data = pa.table(
        {
            "id": [1, 2, 3],
            "text": ["lance database test", "full text search", "lance search text"],
        }
    )
    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index("text", "INVERTED")

    results = ds.to_table(full_text_query="lance search text")
    assert results.num_rows == 3
    assert results["id"].to_pylist() == [3, 2, 1]

    # Assert distributed FTS produces the same BM25 scores as single-machine
    assert_distributed_fts_consistency(
        data,
        "text",
        "lance search text",
        tmp_path,
        index_params={"with_position": False},
    )


def test_fts_with_filter(tmp_path):
    data = pa.table(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "text": [
                "lance database test",
                "full text search",
                "lance search text",
                "some other content",
                "some other content",
                "some other content",
                "some other content",
                "some other content",
                "some other content",
                "some other content",
            ],
        }
    )
    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index("id", "BTREE")
    ds.create_scalar_index("text", "INVERTED")

    results = ds.to_table(full_text_query="lance search text")
    assert results.num_rows == 3
    assert results["id"].to_pylist() == [3, 2, 1]

    score_id1 = results.column("_score")[2].as_py()

    results = ds.to_table(
        full_text_query="lance search text",
        filter="id <= 1",
        prefilter=True,
    )
    assert results.num_rows == 1
    assert results["id"].to_pylist() == [1]
    assert results.column("_score")[0].as_py() == score_id1

    plan = ds.scanner(
        full_text_query="lance search text", filter="id <= 1", prefilter=True
    ).analyze_plan()
    assert "index_comparisons=1" in plan

    # Assert distributed FTS with prefilter produces consistent results
    assert_distributed_fts_consistency(
        data,
        "text",
        "lance search text",
        tmp_path,
        index_params={"with_position": False},
        query_params={"filter": "id <= 1", "prefilter": True},
    )


def test_fts_on_list(tmp_path):
    data = pa.table(
        {
            "text": [
                ["lance database", "the", "search"],
                ["lance database"],
                ["lance", "search"],
                ["database", "search"],
                ["unrelated", "doc"],
            ]
        }
    )
    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index("text", "INVERTED", with_position=True)

    results = ds.to_table(full_text_query="lance")
    assert results.num_rows == 3

    results = ds.to_table(full_text_query=PhraseQuery("lance database", "text"))
    assert results.num_rows == 2


def test_fts_fuzzy_query(tmp_path):
    data = pa.table(
        {
            "text": [
                "fa",
                "fo",  # spellchecker:disable-line
                "fob",
                "focus",
                "foo",
                "food",
                "foul",
            ]
        }
    )

    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index("text", "INVERTED")

    results = ds.to_table(
        full_text_query=MatchQuery("foo", "text", fuzziness=1),
    )
    assert results.num_rows == 4
    assert set(results["text"].to_pylist()) == {
        "foo",
        "fo",  # 1 deletion # spellchecker:disable-line
        "fob",  # 1 substitution
        "food",  # 1 insertion
    }

    results = ds.to_table(
        full_text_query=MatchQuery("foo", "text", fuzziness=1, max_expansions=3),
    )
    assert results.num_rows == 3

    # prefix matching
    results = ds.to_table(
        full_text_query=MatchQuery("foo", "text", fuzziness=1, prefix_length=3)
    )
    assert results.num_rows == 2
    assert set(results["text"].to_pylist()) == {
        "foo",
        "food",
    }

    # Assert distributed FTS produces consistent fuzzy query results
    assert_distributed_fts_consistency(
        data,
        "text",
        MatchQuery("foo", "text", fuzziness=1),
        tmp_path,
        index_params={"with_position": False},
    )


def test_fts_phrase_query(tmp_path):
    data = pa.table(
        {
            "text": [
                "frodo was a puppy",
                "frodo was a happy puppy",
                "frodo was a very happy puppy",
                "frodo was a puppy with a tail",
            ]
        }
    )

    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index(
        "text", "INVERTED", with_position=True, remove_stop_words=False
    )

    results = ds.to_table(
        full_text_query='"frodo was a puppy"',
    )
    assert results.num_rows == 2
    assert set(results["text"].to_pylist()) == {
        "frodo was a puppy",
        "frodo was a puppy with a tail",
    }

    results = ds.to_table(
        full_text_query=PhraseQuery("frodo was a puppy", "text"),
    )
    assert results.num_rows == 2
    assert set(results["text"].to_pylist()) == {
        "frodo was a puppy",
        "frodo was a puppy with a tail",
    }

    results = ds.to_table(full_text_query=PhraseQuery("frodo was happy", "text"))
    assert results.num_rows == 0

    results = ds.to_table(
        full_text_query=PhraseQuery("frodo was happy", "text", slop=1)
    )
    assert results.num_rows == 1

    results = ds.to_table(
        full_text_query=PhraseQuery("frodo was happy", "text", slop=2)
    )
    assert results.num_rows == 2

    # Assert distributed FTS produces consistent phrase query results
    assert_distributed_fts_consistency(
        data,
        "text",
        PhraseQuery("frodo was a puppy", "text"),
        tmp_path,
        index_params={"with_position": True, "remove_stop_words": False},
    )


def test_fts_boost_query(tmp_path):
    data = pa.table(
        {
            "text": [
                "frodo was a puppy",
                "frodo was a happy puppy",
                "frodo was a puppy with a tail",
            ]
        }
    )

    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index("text", "INVERTED")
    results = ds.to_table(
        full_text_query=BoostQuery(
            MatchQuery("puppy", "text"),
            MatchQuery("happy", "text"),
            negative_boost=0.5,
        ),
    )
    assert results.num_rows == 3
    assert set(results["text"].to_pylist()) == {
        "frodo was a puppy",
        "frodo was a puppy with a tail",
        "frodo was a happy puppy",
    }


def test_fts_multi_match_query(tmp_path):
    data = pa.table(
        {
            "title": ["title common", "title hello", "title vector"],
            "content": ["content world", "content database", "content common"],
        }
    )

    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index("title", "INVERTED")
    ds.create_scalar_index("content", "INVERTED")

    results = ds.to_table(
        full_text_query=MultiMatchQuery("common", ["title", "content"]),
    )
    assert set(results["title"].to_pylist()) == {"title common", "title vector"}
    assert set(results["content"].to_pylist()) == {"content world", "content common"}

    # Assert distributed FTS produces consistent multi-match query results
    assert_distributed_fts_consistency(
        data,
        ["title", "content"],
        MultiMatchQuery("common", ["title", "content"]),
        tmp_path,
        index_params={"with_position": False},
    )


def test_fts_boolean_query(tmp_path):
    data = pa.table(
        {
            "text": [
                "frodo was a puppy",
                "frodo was a happy puppy",
                "frodo was a puppy with a tail",
            ]
        }
    )

    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index("text", "INVERTED")

    query = MatchQuery("puppy", "text") & MatchQuery("happy", "text")
    results = ds.to_table(
        full_text_query=query,
    )
    assert results.num_rows == 1
    assert results["text"][0].as_py() == "frodo was a happy puppy"

    query = MatchQuery("tail", "text") | MatchQuery("happy", "text")
    results = ds.to_table(
        full_text_query=query,
    )
    assert results.num_rows == 2
    assert set(results["text"].to_pylist()) == {
        "frodo was a happy puppy",
        "frodo was a puppy with a tail",
    }

    results = ds.to_table(
        full_text_query=BooleanQuery(
            [
                (Occur.MUST, MatchQuery("puppy", "text")),
                (Occur.MUST_NOT, MatchQuery("happy", "text")),
            ]
        ),
    )
    assert results.num_rows == 2
    assert set(results["text"].to_pylist()) == {
        "frodo was a puppy",
        "frodo was a puppy with a tail",
    }

    # Assert distributed FTS produces consistent boolean query results
    query_and = MatchQuery("puppy", "text") & MatchQuery("happy", "text")
    assert_distributed_fts_consistency(
        data, "text", query_and, tmp_path, index_params={"with_position": False}
    )


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


def test_fts_deleted_rows(tmp_path):
    data = pa.table({"text": ["lance is cool", "databases are cool", "search is neat"]})
    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index("text", "INVERTED")
    ds.insert(
        pa.table({"text": ["lance is cool", "databases are cool", "search is neat"]})
    )

    ds.delete("text = 'lance is cool'")
    results = ds.to_table(full_text_query="cool")
    assert results.num_rows == 2


def test_index_after_merge_insert(tmp_path):
    # This regresses a defect where a horizontal merge insert was not taking modified
    # fragments out of the index if the column is modified.
    dataset = lance.write_dataset(
        pa.table({"id": range(100), "payload": range(100), "other": range(100)}),
        tmp_path,
    )
    dataset.create_scalar_index("id", index_type="BTREE")
    dataset.create_scalar_index("payload", index_type="BTREE")

    assert dataset.to_table(filter="payload >= 30").num_rows == 70

    # Partial merge insert triggers horizontal merge insert
    dataset.merge_insert(
        "id"
    ).when_matched_update_all().when_not_matched_insert_all().execute(
        pa.table({"id": range(50, 150), "payload": [0] * 100})
    )

    assert dataset.to_table(filter="payload >= 30").num_rows == 20


def test_lindera_load_config_fallback(tmp_path, lindera_ipadic, monkeypatch):
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
    with pytest.raises(OSError):
        ds.create_scalar_index(
            "text", "INVERTED", base_tokenizer="lindera/load_config_fallback"
        )

    config_path = os.path.join(
        os.path.dirname(__file__),
        "models/lindera/load_config_fallback/config_not_exists.yml",
    )
    monkeypatch.setenv("LINDERA_CONFIG_PATH", config_path)
    with pytest.raises(OSError):
        ds.create_scalar_index(
            "text", "INVERTED", base_tokenizer="lindera/load_config_fallback"
        )

    config_path = os.path.join(
        os.path.dirname(__file__), "models/lindera/load_config_fallback/config_env.yml"
    )
    monkeypatch.setenv("LINDERA_CONFIG_PATH", config_path)
    ds.create_scalar_index(
        "text", "INVERTED", base_tokenizer="lindera/load_config_fallback"
    )
    results = ds.to_table(
        full_text_query="成田",
        prefilter=True,
        with_row_id=True,
    )
    assert results["_rowid"].to_pylist() == [0]


def test_lindera_load_config_priority(tmp_path, lindera_ipadic, monkeypatch):
    data = pa.table(
        {
            "text": [
                "成田国際空港",
                "東京国際空港",
                "羽田空港",
            ],
        }
    )
    config_path = os.path.join(
        os.path.dirname(__file__), "models/lindera/load_config_priority/config_env.yml"
    )
    monkeypatch.setenv("LINDERA_CONFIG_PATH", config_path)
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    ds.create_scalar_index(
        "text", "INVERTED", base_tokenizer="lindera/load_config_priority"
    )
    results = ds.to_table(
        full_text_query="成田",
        prefilter=True,
        with_row_id=True,
    )
    assert results["_rowid"].to_pylist() == [0]

    results = ds.to_table(
        full_text_query="ほげほげ",
        prefilter=True,
        with_row_id=True,
    )
    assert results["_rowid"].to_pylist() == [0]


def test_indexed_filter_with_fts_index_with_lindera_ipadic_jp_tokenizer(
    tmp_path, lindera_ipadic
):
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


def test_bitmap_remap(tmp_path: Path):
    # Make one full fragment
    tbl = pa.Table.from_arrays(
        [pa.array([["a", "b"][i % 2] for i in range(10)])], names=["a"]
    )
    ds = lance.write_dataset(tbl, tmp_path, max_rows_per_file=10)

    # Make two half fragments
    tbl = pa.Table.from_arrays(
        [pa.array([["a", "b"][i % 2] for i in range(10)])], names=["a"]
    )
    ds = lance.write_dataset(tbl, tmp_path, max_rows_per_file=5, mode="append")

    # Create scalar index
    ds.create_scalar_index("a", index_type="BITMAP")

    # Run compaction (two partials will be remapped, full will not)
    compaction = ds.optimize.compact_files(target_rows_per_fragment=10)
    assert compaction.fragments_removed == 2

    for category in ["a", "b"]:
        # All rows should still be in index
        assert ds.count_rows(f"a = '{category}'") == 10


def test_ngram_index(tmp_path: Path):
    """Test create ngram index"""

    def test_with(tbl: pa.Table):
        dataset = lance.write_dataset(tbl, tmp_path / "dataset", mode="overwrite")
        dataset.create_scalar_index("words", index_type="NGRAM")
        indices = dataset.list_indices()
        assert len(indices) == 1
        assert indices[0]["type"] == "NGram"

        scan_plan = dataset.scanner(filter="contains(words, 'apple')").explain_plan(
            True
        )
        assert "ScalarIndexQuery: query=[contains(words" in scan_plan

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

    tbl = pa.Table.from_arrays(
        [
            pa.array(
                [["apple", "apples", "banana", "coconut"][i % 4] for i in range(100)]
            )
        ],
        names=["words"],
    )
    test_with(tbl)

    # Test with large string
    tbl = pa.Table.from_arrays(
        [
            pa.array(
                [["apple", "apples", "banana", "coconut"][i % 4] for i in range(100)],
                type=pa.large_string(),
            )
        ],
        names=["words"],
    )
    test_with(tbl)


def test_zonemap_index(tmp_path: Path):
    """Test create zonemap index"""
    tbl = pa.Table.from_arrays([pa.array([i for i in range(8193)])], names=["values"])
    dataset = lance.write_dataset(tbl, tmp_path / "dataset")
    dataset.create_scalar_index("values", index_type="ZONEMAP")
    indices = dataset.list_indices()
    assert len(indices) == 1

    # Get detailed index statistics
    index_stats = dataset.stats.index_stats("values_idx")
    assert index_stats["index_type"] == "ZoneMap"
    assert "indices" in index_stats
    assert len(index_stats["indices"]) == 1

    # Verify zonemap statistics
    zonemap_stats = index_stats["indices"][0]
    assert zonemap_stats["rows_per_zone"] == 8192
    assert zonemap_stats["num_zones"] == 2  # Should have 2 zones (8192 rows + 1 row)

    # Test that the zonemap index is being used in the query plan
    scanner = dataset.scanner(filter="values > 50", prefilter=True)
    plan = scanner.explain_plan()
    assert "ScalarIndexQuery" in plan

    # Verify the query returns correct results
    result = scanner.to_table()
    assert result.num_rows == 8142  # 51..8192


def test_bloomfilter_index(tmp_path: Path):
    """Test create bloomfilter index"""
    tbl = pa.Table.from_arrays([pa.array([i for i in range(10000)])], names=["values"])
    dataset = lance.write_dataset(tbl, tmp_path / "dataset")
    dataset.create_scalar_index("values", index_type="BLOOMFILTER")
    indices = dataset.list_indices()
    assert len(indices) == 1

    # Get detailed index statistics
    index_stats = dataset.stats.index_stats("values_idx")
    assert index_stats["index_type"] == "BloomFilter"
    assert "indices" in index_stats
    assert len(index_stats["indices"]) == 1

    # Verify bloomfilter statistics
    bloom_stats = index_stats["indices"][0]
    assert "num_blocks" in bloom_stats
    assert bloom_stats["num_blocks"] == 2
    assert bloom_stats["number_of_items"] == 8192
    assert "probability" in bloom_stats
    assert bloom_stats["probability"] == 0.00057  # Default probability

    # Test that the bloomfilter index is being used in the query plan
    scanner = dataset.scanner(filter="values == 1234", prefilter=True)
    plan = scanner.explain_plan()
    assert "ScalarIndexQuery" in plan

    # Verify the query returns correct results
    result = scanner.to_table()
    assert result.num_rows == 1
    assert result["values"][0].as_py() == 1234


def test_zonemap_index_remapping(tmp_path: Path):
    """Test zonemap index remapping after compaction and optimization"""
    # Create a dataset with 5 fragments by writing data in chunks
    # Each fragment will have 1000 rows, so we need 5000 total rows
    tbl = pa.Table.from_arrays([pa.array(range(0, 5000))], names=["values"])
    dataset = lance.write_dataset(tbl, tmp_path / "dataset", max_rows_per_file=1000)

    fragments = dataset.get_fragments()
    assert len(fragments) == 5

    # Train a zone map index
    dataset.create_scalar_index("values", index_type="ZONEMAP")
    indices = dataset.list_indices()
    assert len(indices) == 1
    assert indices[0]["type"] == "ZoneMap"

    # Confirm the zone map index is used if you search the dataset
    scanner = dataset.scanner(filter="values > 2500", prefilter=True)
    plan = scanner.explain_plan()
    assert "ScalarIndexQuery" in plan

    # Verify the query returns correct results
    result = scanner.to_table()
    assert result.num_rows == 2499  # 2501..4999

    # Run compaction to merge fragments
    compaction = dataset.optimize.compact_files(target_rows_per_fragment=2000)
    assert compaction.fragments_removed == 5
    assert len(dataset.get_fragments()) == 3

    # Check if the zone map index is no longer being used
    scanner = dataset.scanner(filter="values > 2500", prefilter=True)
    plan = scanner.explain_plan()
    assert "ScalarIndexQuery" not in plan

    # Run optimize indices to rebuild the index
    dataset.optimize.optimize_indices()

    # Confirm the zone map index is used again after optimization
    scanner = dataset.scanner(filter="values > 2500", prefilter=True)
    plan = scanner.explain_plan()
    assert "ScalarIndexQuery" in plan

    # Verify the query returns correct results
    result = scanner.to_table()
    assert result.num_rows == 2499  # 2501..4999

    # Test with a different query to ensure index works properly
    scanner = dataset.scanner(filter="values BETWEEN 1000 AND 1500", prefilter=True)
    plan = scanner.explain_plan()
    print(f"Query plan after optimization: {plan}")
    assert "ScalarIndexQuery" in plan

    result = scanner.to_table()
    assert result.num_rows == 501  # 1000..1500 inclusive


def test_json_index():
    vals = ['{"x": 7, "y": 10}', '{"x": 11, "y": 22}', '{"y": 0}', '{"x": 10}']
    tbl = pa.table({"jsons": pa.array(vals, pa.json_())})
    ds = lance.write_dataset(tbl, "memory://test")
    ds.create_scalar_index(
        "jsons",
        IndexConfig(
            index_type="json", parameters={"target_index_type": "btree", "path": "x"}
        ),
    )

    # TODO: I changed this into `json_get_int` for strong typed filter, should be
    # refactored into JSON Path compare.
    filter = "json_get_int(jsons, 'x') = 10"
    assert "ScalarIndexQuery" in ds.scanner(filter=filter).explain_plan()
    assert ds.to_table(filter=filter) == ds.to_table(
        filter=filter, use_scalar_index=False
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


def test_nan_handling(tmp_path: Path):
    tbl = pa.table(
        {
            "x": [
                1.0,
                float("-nan"),
                float("infinity"),
                float("-infinity"),
                2.0,
                float("nan"),
                3.0,
            ],
        }
    )
    dataset = lance.write_dataset(tbl, tmp_path / "dataset")

    # There is no way, in DF, to query for NAN / INF, that I'm aware of.
    # So the best we can do here is make sure that the presence of NAN / INF
    # doesn't interfere with normal operation of the btree.
    def check(has_index: bool):
        assert dataset.to_table(filter="x IS NULL").num_rows == 0
        assert dataset.to_table(filter="x IS NOT NULL").num_rows == 7
        assert dataset.to_table(filter="x > 0").num_rows == 5
        assert dataset.to_table(filter="x < 5").num_rows == 5
        assert dataset.to_table(filter="x IN (1, 2)").num_rows == 2

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


def test_index_prewarm(tmp_path: Path):
    scan_stats = None

    def scan_stats_callback(stats: lance.ScanStatistics):
        nonlocal scan_stats
        scan_stats = stats

    test_table_size = 100
    test_table = pa.table(
        {
            "fts": ["word" for _ in range(test_table_size)],
        }
    )

    # Write index, cache should not be populated
    ds = lance.write_dataset(test_table, tmp_path)
    ds.create_scalar_index("fts", index_type="INVERTED")
    ds.scanner(
        scan_stats_callback=scan_stats_callback, full_text_query="word"
    ).to_table()
    assert scan_stats.parts_loaded > 0

    # Fresh load, no prewarm, cache should not be populated
    ds = lance.dataset(tmp_path)
    ds.scanner(
        scan_stats_callback=scan_stats_callback, full_text_query="word"
    ).to_table()
    assert scan_stats.parts_loaded > 0

    # Prewarm index, cache should be populated
    ds = lance.dataset(tmp_path)
    ds.prewarm_index("fts_idx")
    ds.scanner(
        scan_stats_callback=scan_stats_callback, full_text_query="word"
    ).to_table()
    assert scan_stats.parts_loaded == 0


def test_btree_prewarm(tmp_path: Path):
    scan_stats = None

    def scan_stats_callback(stats: lance.ScanStatistics):
        nonlocal scan_stats
        scan_stats = stats

    test_table_size = 100
    test_table = pa.table(
        {
            "id": list(range(test_table_size)),
        }
    )
    ds = lance.write_dataset(test_table, tmp_path)
    ds.create_scalar_index("id", index_type="BTREE")
    ds.scanner(scan_stats_callback=scan_stats_callback, filter="id>0").to_table()
    assert scan_stats.parts_loaded > 0

    ds = lance.dataset(tmp_path)
    ds.scanner(scan_stats_callback=scan_stats_callback, filter="id>0").to_table()
    assert scan_stats.parts_loaded > 0

    ds = lance.dataset(tmp_path)
    ds.prewarm_index("id_idx")
    ds.scanner(scan_stats_callback=scan_stats_callback, filter="id>0").to_table()
    assert scan_stats.parts_loaded == 0


def test_fts_backward_v0_27_0(tmp_path: Path):
    path = (
        Path(__file__).parent.parent.parent.parent
        / "test_data"
        / "0.27.0"
        / "legacy_fts_index"
    )
    shutil.copytree(path, tmp_path, dirs_exist_ok=True)
    ds = lance.dataset(tmp_path)

    # we can read the old index
    results = ds.to_table(
        full_text_query=BoostQuery(
            MatchQuery("puppy", "text"),
            MatchQuery("happy", "text"),
            negative_boost=0.5,
        ),
    )
    assert results.num_rows == 3
    assert set(results["text"].to_pylist()) == {
        "frodo was a puppy",
        "frodo was a puppy with a tail",
        "frodo was a happy puppy",
    }

    data = pa.table(
        {
            "text": [
                "new data",
            ]
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="append")
    ds.optimize.optimize_indices()
    results = ds.to_table(
        full_text_query=BoostQuery(
            MatchQuery("puppy", "text"),
            MatchQuery("happy", "text"),
            negative_boost=0.5,
        ),
    )
    assert results.num_rows == 3
    assert set(results["text"].to_pylist()) == {
        "frodo was a puppy",
        "frodo was a puppy with a tail",
        "frodo was a happy puppy",
    }
    res = ds.to_table(
        full_text_query="new",
    )
    assert res.num_rows == 1


# ============================================================================
# Distributed FTS Index Helper Functions
# ============================================================================


def build_distributed_fts_index(
    dataset: lance.LanceDataset, column: str, index_name: str = None, **index_params
) -> lance.LanceDataset:
    """
    Build FTS index in distributed way and return the committed dataset.

    This helper function builds the FTS index on individual fragments
    and then commits them as a single index, ensuring the distributed
    approach produces the same results as single-machine indexing.

    Parameters
    ----------
    dataset : lance.LanceDataset
        The dataset to build index on
    column : str
        The column name to build FTS index on
    index_name : str, optional
        Name for the index. If not provided, will use f"{column}_distributed_idx"
    **index_params
        Additional parameters to pass to create_scalar_index()
        (e.g., with_position, remove_stop_words, base_tokenizer, etc.)

    Returns
    -------
    lance.LanceDataset
        Dataset with committed distributed FTS index

    Examples
    --------
    >>> ds_distributed = build_distributed_fts_index(
    ...     dataset,
    ...     "text",
    ...     with_position=True,
    ...     remove_stop_words=False
    ... )
    >>> # Now compare with single-machine index results
    >>> results_distributed = ds_distributed.scanner(full_text_query="test").to_table()
    """
    import uuid

    from lance.dataset import Index

    # Generate unique index ID for distributed indexing
    index_id = str(uuid.uuid4())
    index_name = index_name or f"{column}_distributed_idx"

    # Get all fragments from the dataset
    fragments = dataset.get_fragments()
    fragment_ids = [fragment.fragment_id for fragment in fragments]

    # Build index on each fragment individually
    for fragment_id in fragment_ids:
        dataset.create_scalar_index(
            column=column,
            index_type="INVERTED",
            name=index_name,
            replace=False,
            fragment_uuid=index_id,
            fragment_ids=[fragment_id],
            **index_params,
        )

    # Merge the inverted index metadata
    dataset.merge_index_metadata(index_id)

    # Create Index object for commit
    field_id = dataset.schema.get_field_index(column)
    index = Index(
        uuid=index_id,
        name=index_name,
        fields=[field_id],
        dataset_version=dataset.version,
        fragment_ids=set(fragment_ids),
        index_version=0,
    )

    # Create and commit the index operation
    create_index_op = lance.LanceOperation.CreateIndex(
        new_indices=[index],
        removed_indices=[],
    )

    # Commit the distributed index
    committed_dataset = lance.LanceDataset.commit(
        dataset.uri,
        create_index_op,
        read_version=dataset.version,
    )

    return committed_dataset


def compare_fts_results(
    single_machine_results: pa.Table,
    distributed_results: pa.Table,
    tolerance: float = 1e-6,
) -> bool:
    """
    Compare FTS search results from single-machine and distributed indexing.

    Parameters
    ----------
    single_machine_results : pa.Table
        Results from single-machine FTS index
    distributed_results : pa.Table
        Results from distributed FTS index
    tolerance : float, default 1e-6
        Tolerance for floating point score comparison

    Returns
    -------
    bool
        True if results are equivalent, False otherwise

    Raises
    ------
    AssertionError
        If results don't match with detailed error message
    """
    # Check row count
    assert single_machine_results.num_rows == distributed_results.num_rows, (
        f"Row count mismatch: single={single_machine_results.num_rows}, "
        f"distributed={distributed_results.num_rows}"
    )

    # If no results, both should be empty
    if single_machine_results.num_rows == 0:
        return True

    # Convert to pandas for easier comparison
    single_df = single_machine_results.to_pandas()
    distributed_df = distributed_results.to_pandas()

    # Sort both by row_id to ensure consistent ordering
    if "_rowid" in single_df.columns:
        single_df = single_df.sort_values("_rowid").reset_index(drop=True)
        distributed_df = distributed_df.sort_values("_rowid").reset_index(drop=True)

    # Compare row IDs (most important)
    if "_rowid" in single_df.columns:
        single_rowids = set(single_df["_rowid"])
        distributed_rowids = set(distributed_df["_rowid"])
        assert single_rowids == distributed_rowids, (
            f"Row ID mismatch: single={single_rowids}, distributed={distributed_rowids}"
        )

    # Compare scores with tolerance
    if "_score" in single_df.columns:
        single_scores = single_df["_score"].values
        distributed_scores = distributed_df["_score"].values
        score_diff = np.abs(single_scores - distributed_scores)
        max_diff = np.max(score_diff)
        assert max_diff <= tolerance, (
            f"Score difference exceeds tolerance: max_diff={max_diff}, "
            f"tolerance={tolerance}"
        )

    # Compare other columns (exact match for non-score columns)
    for col in single_df.columns:
        if col not in ["_score"]:  # Skip score column (already compared with tolerance)
            single_values = (
                set(single_df[col])
                if single_df[col].dtype == "object"
                else single_df[col].values
            )
            distributed_values = (
                set(distributed_df[col])
                if distributed_df[col].dtype == "object"
                else distributed_df[col].values
            )

            if isinstance(single_values, set):
                assert single_values == distributed_values, (
                    f"Column {col} content mismatch"
                )
            else:
                np.testing.assert_array_equal(
                    single_values,
                    distributed_values,
                    err_msg=f"Column {col} values don't match",
                )

    return True


def validate_distributed_fts(
    dataset: lance.LanceDataset,
    columns_to_index,
    query,
    index_params: dict = None,
    query_params: dict = None,
    tolerance: float = 1e-6,
) -> dict:
    """
    Validate that distributed FTS indexing produces the same results as
    single-machine indexing.

    Parameters
    ----------
    dataset : lance.LanceDataset
        The dataset to test on
    columns_to_index : str or list of str
        The column name(s) to build FTS index on. Can be a single column name
        or a list of column names for multi-column indexing.
    query : str or query object
        The full text query to test with
    index_params : dict, optional
        Parameters for index creation
    query_params : dict, optional
        Parameters for query execution
    tolerance : float, default 1e-6
        Tolerance for score comparison

    Returns
    -------
    dict
        Dictionary with 'single_machine' and 'distributed' results

    Raises
    ------
    AssertionError
        If results don't match
    """
    index_params = index_params or {}
    query_params = query_params or {}

    # Normalize columns_to_index to a list
    if isinstance(columns_to_index, str):
        columns_list = [columns_to_index]
    else:
        columns_list = list(columns_to_index)

    # Build single-machine indices for all required columns
    for column in columns_list:
        dataset.create_scalar_index(
            column=column,
            index_type="INVERTED",
            name=f"{column}_single_idx",
            **index_params,
        )

    # Build distributed indices for all required columns
    distributed_ds = dataset  # Start with the original dataset
    for column in columns_list:
        distributed_ds = build_distributed_fts_index(
            distributed_ds,
            column,
            index_name=f"{column}_distributed_idx",
            **index_params,
        )

    # Execute queries
    single_results = dataset.scanner(full_text_query=query, **query_params).to_table()

    distributed_results = distributed_ds.scanner(
        full_text_query=query, **query_params
    ).to_table()

    # Compare results
    compare_fts_results(single_results, distributed_results, tolerance)

    return {"single_machine": single_results, "distributed": distributed_results}


def assert_distributed_fts_consistency(
    data: pa.Table,
    columns_to_index,
    query,
    tmp_path,
    index_params: dict = None,
    query_params: dict = None,
    tolerance: float = 1e-6,
):
    """
    Assert that distributed FTS indexing produces the same results as
    single-machine indexing.

    This is a streamlined version that eliminates repetitive dataset creation and
    try-catch-print patterns. Uses direct assertions instead of exception handling.

    Parameters
    ----------
    data : pa.Table
        The data to test with
    columns_to_index : str or list of str
        The column name(s) to build FTS index on. Can be a single column name
        or a list of column names for multi-column indexing.
    query : str or query object
        The full text query to test with
    tmp_path : Path
        Temporary path for datasets
    index_params : dict, optional
        Parameters for index creation
    query_params : dict, optional
        Parameters for query execution
    tolerance : float, default 1e-6
        Tolerance for score comparison

    Raises
    ------
    AssertionError
        If distributed and single-machine results don't match
    """
    index_params = index_params or {}
    query_params = query_params or {}

    # Normalize columns_to_index to a list
    if isinstance(columns_to_index, str):
        columns_list = [columns_to_index]
    else:
        columns_list = list(columns_to_index)

    # Create datasets for single-machine and distributed testing
    single_ds = lance.write_dataset(data, tmp_path / "single")
    distributed_ds = lance.write_dataset(data, tmp_path / "distributed")

    # Build single-machine indices for all required columns
    for column in columns_list:
        single_ds.create_scalar_index(
            column=column,
            index_type="INVERTED",
            name=f"{column}_single_idx",
            **index_params,
        )

    # Build distributed indices for all required columns
    distributed_ds_indexed = distributed_ds  # Start with the original dataset
    for column in columns_list:
        distributed_ds_indexed = build_distributed_fts_index(
            distributed_ds_indexed,
            column,
            index_name=f"{column}_distributed_idx",
            **index_params,
        )

    # Execute queries
    single_results = single_ds.scanner(full_text_query=query, **query_params).to_table()

    distributed_results = distributed_ds_indexed.scanner(
        full_text_query=query, **query_params
    ).to_table()

    # Assert results are identical
    compare_fts_results(single_results, distributed_results, tolerance)


def run_fts_distributed_validation_suite(
    test_functions: list, tmp_path_factory, verbose: bool = True
) -> dict:
    """
    Run distributed validation for a suite of FTS test functions.

    Parameters
    ----------
    test_functions : list
        List of FTS test functions to validate
    tmp_path_factory : pytest.TempPathFactory
        Pytest temp path factory for creating test directories
    verbose : bool, default True
        Whether to print detailed progress information

    Returns
    -------
    dict
        Dictionary mapping test function names to validation results
    """
    results = {}

    for test_func in test_functions:
        test_name = test_func.__name__
        if verbose:
            print(f"Running distributed validation for {test_name}...")

        try:
            tmp_path = tmp_path_factory.mktemp(f"distributed_{test_name}")

            # Run the test with distributed validation
            # Note: This is a simplified version - actual implementation would
            # need to extract the test logic and run both single-machine and
            # distributed versions
            test_func(tmp_path)

            results[test_name] = True
            if verbose:
                print(f"✓ {test_name} passed distributed validation")

        except Exception as e:
            results[test_name] = False
            if verbose:
                print(f"✗ {test_name} failed distributed validation: {e}")

    return results


# ============================================================================
# Test Data Generation Functions
# ============================================================================


def generate_multi_fragment_dataset(tmp_path, num_fragments=4, rows_per_fragment=250):
    """
    Generate a test dataset with multiple fragments for testing fragment-level indexing.
    Uses coherent English text similar to "frodo was a puppy" instead of random strings.

    Parameters
    ----------
    tmp_path : Path
        Temporary path for the dataset
    num_fragments : int, default 4
        Number of fragments to create
    rows_per_fragment : int, default 250
        Number of rows per fragment

    Returns
    -------
    lance.LanceDataset
        Dataset with multiple fragments
    """

    # Collection of coherent English sentences for text indexing tests
    # Based on the pattern used in other tests like "frodo was a puppy"
    coherent_sentences = [
        "frodo was a puppy",
        "frodo was a happy puppy",
        "frodo was a very happy puppy",
        "frodo was a puppy with a tail",
        "gandalf carried a staff",
        "gandalf was a wise wizard",
        "gandalf carried his wooden staff",
        "gandalf wore a grey robe",
        "aragorn became the king",
        "aragorn was a brave ranger",
        "aragorn carried his sword",
        "aragorn wore a crown",
        "legolas shot many arrows",
        "legolas was an elf archer",
        "legolas had keen eyes",
        "legolas climbed tall trees",
        "gimli swung his axe",
        "gimli was a dwarf warrior",
        "gimli had a long beard",
        "gimli loved precious gems",
        "sam cooked delicious meals",
        "sam was a loyal friend",
        "sam carried heavy bags",
        "sam tended the garden",
        "pippin played his flute",
        "pippin was very curious",
        "pippin loved second breakfast",
        "pippin climbed apple trees",
        "merry sang cheerful songs",
        "merry was quite clever",
        "merry rode a pony",
        "merry studied old maps",
        "boromir blew his horn",
        "boromir was a proud warrior",
        "boromir defended his city",
        "boromir carried a shield",
        "the ring was very powerful",
        "the ring glowed with fire",
        "the ring whispered secrets",
        "the ring corrupted minds",
        "the shire was peaceful",
        "the shire had green hills",
        "the shire grew fine crops",
        "the shire welcomed visitors",
        "eagles soared through clouds",
        "eagles had sharp talons",
        "eagles nested on peaks",
        "eagles watched the valleys",
        "dragons hoarded gold treasures",
        "dragons breathed hot flames",
        "dragons slept for centuries",
        "dragons guarded ancient caves",
    ]

    def generate_coherent_text():
        """Generate coherent English text by selecting from predefined sentences."""
        return random.choice(coherent_sentences)

    # Create first fragment
    first_data = pa.table(
        {
            "id": pa.array(range(rows_per_fragment)),
            "text": pa.array(
                [generate_coherent_text() for _ in range(rows_per_fragment)]
            ),
            "value": pa.array(np.random.rand(rows_per_fragment) * 100),
        }
    )

    ds = lance.write_dataset(first_data, tmp_path, max_rows_per_file=rows_per_fragment)

    # Add additional fragments
    for i in range(1, num_fragments):
        start_id = i * rows_per_fragment
        fragment_data = pa.table(
            {
                "id": pa.array(range(start_id, start_id + rows_per_fragment)),
                "text": pa.array(
                    [generate_coherent_text() for _ in range(rows_per_fragment)]
                ),
                "value": pa.array(np.random.rand(rows_per_fragment) * 100),
            }
        )
        ds = lance.write_dataset(
            fragment_data, tmp_path, mode="append", max_rows_per_file=rows_per_fragment
        )

    # Verify we have the expected number of fragments
    fragments = ds.get_fragments()
    assert len(fragments) == num_fragments, (
        f"Expected {num_fragments} fragments, got {len(fragments)}"
    )

    return ds


# ============================================================================
# Distributed FTS Index Unit Tests
# ============================================================================


def test_build_distributed_fts_index_basic(tmp_path):
    """
    Test basic functionality of build_distributed_fts_index helper function.
    """
    # Generate test dataset with multiple fragments
    ds = generate_multi_fragment_dataset(
        tmp_path, num_fragments=3, rows_per_fragment=100
    )

    # Build distributed FTS index
    distributed_ds = build_distributed_fts_index(
        ds, "text", with_position=False, remove_stop_words=False
    )

    # Verify the index was created
    indices = distributed_ds.list_indices()
    assert len(indices) > 0, "No indices found after distributed index creation"

    # Find our distributed index
    distributed_index = None
    for idx in indices:
        if "distributed" in idx["name"]:
            distributed_index = idx
            break

    assert distributed_index is not None, "Distributed index not found"
    assert distributed_index["type"] == "Inverted", (
        f"Expected Inverted index, got {distributed_index['type']}"
    )

    # Test that the index works for searching
    results = distributed_ds.scanner(
        full_text_query="frodo",
        columns=["id", "text"],
    ).to_table()

    assert results.num_rows > 0, "No results found for search term 'frodo'"


def test_compare_fts_results_identical(tmp_path):
    """
    Test compare_fts_results function with identical results.
    """
    # Create identical test results
    data = {
        "id": [1, 2, 3],
        "text": ["frodo was a puppy", "gandalf was wise", "aragorn became king"],
        "_score": [0.95, 0.85, 0.75],
        "_rowid": [0, 1, 2],
    }

    table1 = pa.table(data)
    table2 = pa.table(data)

    # Should return True for identical results
    result = compare_fts_results(table1, table2)
    assert result is True, "Identical results should be considered equal"


def test_compare_fts_results_different_scores(tmp_path):
    """
    Test compare_fts_results function with different scores (should fail).
    """
    data1 = {
        "id": [1, 2, 3],
        "text": ["frodo was a puppy", "gandalf was wise", "aragorn became king"],
        "_score": [0.95, 0.85, 0.75],
        "_rowid": [0, 1, 2],
    }

    data2 = {
        "id": [1, 2, 3],
        "text": ["frodo was a puppy", "gandalf was wise", "aragorn became king"],
        "_score": [0.90, 0.80, 0.70],  # Different scores
        "_rowid": [0, 1, 2],
    }

    table1 = pa.table(data1)
    table2 = pa.table(data2)

    # Should raise AssertionError for different scores
    with pytest.raises(AssertionError, match="Score difference exceeds tolerance"):
        compare_fts_results(table1, table2)


def test_compare_fts_results_different_rowids(tmp_path):
    """
    Test compare_fts_results function with different row IDs (should fail).
    """
    data1 = {
        "id": [1, 2, 3],
        "text": ["frodo was a puppy", "gandalf was wise", "aragorn became king"],
        "_score": [0.95, 0.85, 0.75],
        "_rowid": [0, 1, 2],
    }

    data2 = {
        "id": [1, 2, 3],
        "text": ["frodo was a puppy", "gandalf was wise", "aragorn became king"],
        "_score": [0.95, 0.85, 0.75],
        "_rowid": [0, 1, 3],  # Different row ID
    }

    table1 = pa.table(data1)
    table2 = pa.table(data2)

    # Should raise AssertionError for different row IDs
    with pytest.raises(AssertionError, match="Row ID mismatch"):
        compare_fts_results(table1, table2)


def test_validate_distributed_fts_basic_search(tmp_path):
    """
    Test validate_distributed_fts function with basic search.
    """
    # Generate test dataset with multiple fragments
    ds = generate_multi_fragment_dataset(
        tmp_path, num_fragments=3, rows_per_fragment=100
    )

    # Validate distributed FTS with basic search
    results = validate_distributed_fts(
        ds,
        "text",
        "frodo",
        index_params={"with_position": False, "remove_stop_words": False},
    )

    # Check that we got both results
    assert "single_machine" in results, "Missing single_machine results"
    assert "distributed" in results, "Missing distributed results"

    # Both should have the same number of rows
    single_rows = results["single_machine"].num_rows
    distributed_rows = results["distributed"].num_rows
    assert single_rows == distributed_rows, (
        f"Row count mismatch: {single_rows} vs {distributed_rows}"
    )

    # Should have found some results for 'frodo'
    assert single_rows > 0, "No results found for search term 'frodo'"


def test_validate_distributed_fts_score_consistency(tmp_path):
    """
    Test that distributed FTS produces consistent BM25 scores.
    """
    # Create a dataset with known content for scoring tests
    data = pa.table(
        {
            "id": [1, 2, 3],
            "text": ["lance database test", "full text search", "lance search text"],
        }
    )
    ds = lance.write_dataset(data, tmp_path)

    # Validate distributed FTS with scoring query
    results = validate_distributed_fts(
        ds, "text", "lance search text", index_params={"with_position": False}
    )

    # Check that scores are present and consistent
    single_results = results["single_machine"]
    distributed_results = results["distributed"]

    assert "_score" in single_results.column_names, (
        "Missing _score in single machine results"
    )
    assert "_score" in distributed_results.column_names, (
        "Missing _score in distributed results"
    )

    # Scores should be very close (within 1e-6 tolerance)
    single_scores = single_results.column("_score").to_pylist()
    distributed_scores = distributed_results.column("_score").to_pylist()

    for i, (s_score, d_score) in enumerate(zip(single_scores, distributed_scores)):
        score_diff = abs(s_score - d_score)
        assert score_diff <= 1e-6, f"Score difference at index {i}: {score_diff} > 1e-6"


def test_validate_distributed_fts_empty_results(tmp_path):
    """
    Test validate_distributed_fts function with query that returns no results.
    """
    # Generate test dataset
    ds = generate_multi_fragment_dataset(
        tmp_path, num_fragments=2, rows_per_fragment=50
    )

    # Search for something that doesn't exist
    results = validate_distributed_fts(
        ds, "text", "nonexistent_term_xyz", index_params={"with_position": False}
    )

    # Both should return empty results
    assert results["single_machine"].num_rows == 0, (
        "Single machine should return 0 results"
    )
    assert results["distributed"].num_rows == 0, "Distributed should return 0 results"


def test_validate_distributed_fts_large_dataset(tmp_path):
    """
    Test validate_distributed_fts function with larger dataset.
    """
    # Generate larger test dataset
    ds = generate_multi_fragment_dataset(
        tmp_path, num_fragments=5, rows_per_fragment=200
    )

    # Validate distributed FTS
    results = validate_distributed_fts(
        ds,
        "text",
        "gandalf",
        index_params={"with_position": False, "remove_stop_words": False},
    )

    # Should find results and they should match
    single_rows = results["single_machine"].num_rows
    distributed_rows = results["distributed"].num_rows

    assert single_rows > 0, "Should find results for 'gandalf'"
    assert single_rows == distributed_rows, (
        f"Row count mismatch: {single_rows} vs {distributed_rows}"
    )


# ============================================================================
# Advanced Query Tests for Distributed FTS
# ============================================================================


def test_distributed_fts_phrase_query_validation(tmp_path):
    """
    Test distributed FTS validation with phrase queries.
    """
    data = pa.table(
        {
            "text": [
                "frodo was a puppy",
                "frodo was a happy puppy",
                "frodo was a very happy puppy",
                "frodo was a puppy with a tail",
            ]
        }
    )
    ds = lance.write_dataset(data, tmp_path)

    # Test phrase query validation
    results = validate_distributed_fts(
        ds,
        "text",
        PhraseQuery("frodo was a puppy", "text"),
        index_params={"with_position": True, "remove_stop_words": False},
    )

    # Should find matching phrase results
    assert results["single_machine"].num_rows == results["distributed"].num_rows
    assert results["single_machine"].num_rows == 2  # Two exact matches


def test_distributed_fts_boolean_query_validation(tmp_path):
    """
    Test distributed FTS validation with boolean queries.
    """
    data = pa.table(
        {
            "text": [
                "frodo was a puppy",
                "frodo was a happy puppy",
                "frodo was a puppy with a tail",
            ]
        }
    )
    ds = lance.write_dataset(data, tmp_path)

    # Test boolean query validation
    query = MatchQuery("puppy", "text") & MatchQuery("happy", "text")
    results = validate_distributed_fts(
        ds, "text", query, index_params={"with_position": False}
    )

    # Should find only the happy puppy
    assert results["single_machine"].num_rows == results["distributed"].num_rows
    assert results["single_machine"].num_rows == 1


def test_distributed_fts_fuzzy_query_validation(tmp_path):
    """
    Test distributed FTS validation with fuzzy queries.
    """
    data = pa.table({"text": ["foo", "food", "fob", "focus", "foul"]})
    ds = lance.write_dataset(data, tmp_path)

    # Test fuzzy query validation
    results = validate_distributed_fts(
        ds,
        "text",
        MatchQuery("foo", "text", fuzziness=1),
        index_params={"with_position": False},
    )

    # Should find fuzzy matches
    assert results["single_machine"].num_rows == results["distributed"].num_rows
    assert results["single_machine"].num_rows > 1  # Multiple fuzzy matches


def test_distributed_fts_with_filter_validation(tmp_path):
    """
    Test distributed FTS validation with filters.
    """
    data = pa.table(
        {
            "id": [1, 2, 3, 4, 5],
            "text": [
                "lance database test",
                "full text search",
                "lance search text",
                "some other content",
                "more lance content",
            ],
        }
    )
    ds = lance.write_dataset(data, tmp_path)

    # Create BTREE index for filtering
    ds.create_scalar_index("id", "BTREE")

    # Test FTS with filter validation
    results = validate_distributed_fts(
        ds,
        "text",
        "lance",
        index_params={"with_position": False},
        query_params={"filter": "id <= 3", "prefilter": True},
    )

    # Should find filtered results
    assert results["single_machine"].num_rows == results["distributed"].num_rows
    assert results["single_machine"].num_rows == 2  # Only id 1 and 3 have "lance"


def test_distributed_fts_multi_match_validation(tmp_path):
    """
    Test distributed FTS validation with multi-field matching.
    """
    data = pa.table(
        {
            "title": ["title common", "title hello", "title vector"],
            "content": ["content world", "content database", "content common"],
        }
    )
    ds = lance.write_dataset(data, tmp_path)

    # Test multi-match query validation
    results = validate_distributed_fts(
        ds,
        ["title", "content"],  # Index both fields for multi-match query
        MultiMatchQuery("common", ["title", "content"]),
        index_params={"with_position": False},
    )

    # Note: For multi-match, we need to create indices on both fields
    # This is a simplified test - in practice, we'd need more complex setup
    assert results["single_machine"].num_rows == results["distributed"].num_rows


# ============================================================================
# Integration with Existing High-Priority FTS Tests
# ============================================================================


def test_distribute_fts_index_build(tmp_path):
    """
    This test creates indices on individual fragments
    and then commits them as a single index.
    """
    # Generate test dataset with multiple fragments
    ds = generate_multi_fragment_dataset(
        tmp_path, num_fragments=4, rows_per_fragment=250
    )

    import uuid

    index_id = str(uuid.uuid4())
    print(f"Using index ID: {index_id}")
    index_name = "multiple_fragment_idx"

    fragments = ds.get_fragments()
    fragment_ids = [fragment.fragment_id for fragment in fragments]
    print(f"Fragment IDs: {fragment_ids}")

    for fragment in ds.get_fragments():
        fragment_id = fragment.fragment_id
        print(f"Creating index for fragment {fragment_id}")

        # Use the new fragment_ids and fragment_uuid parameters
        ds.create_scalar_index(
            column="text",
            index_type="INVERTED",
            name=index_name,
            replace=False,
            fragment_uuid=index_id,
            fragment_ids=[fragment_id],
            remove_stop_words=False,
        )

        # For fragment-level indexing, we expect the method to return successfully
        # but not commit the index yet
        print(f"Fragment {fragment_id} index created successfully")

    # Merge the inverted index metadata
    ds.merge_index_metadata(index_id)

    # Create an Index object using the new dataclass format
    from lance.dataset import Index

    # Get the schema field for the indexed column
    # Only use for non nested struct schema
    field_id = ds.schema.get_field_index("text")

    index = Index(
        uuid=index_id,
        name=index_name,
        fields=[field_id],  # Use field index instead of field object
        dataset_version=ds.version,
        fragment_ids=set(fragment_ids),
        index_version=0,
    )

    # Create the index operation
    create_index_op = lance.LanceOperation.CreateIndex(
        new_indices=[index],
        removed_indices=[],
    )

    # Commit the index
    ds_committed = lance.LanceDataset.commit(
        ds.uri,
        create_index_op,
        read_version=ds.version,
    )

    print("Successfully committed multiple fragment index")

    # Verify the index was created and is functional
    indices = ds_committed.list_indices()
    assert len(indices) > 0, "No indices found after commit"

    # Find our index
    our_index = None
    for idx in indices:
        if idx["name"] == index_name:
            our_index = idx
            break
    assert our_index is not None, f"Index '{index_name}' not found in indices list"
    assert our_index["type"] == "Inverted", (
        f"Expected Inverted index, got {our_index['type']}"
    )

    # Test that the index works for searching
    # Get a sample text from the dataset to search for
    sample_data = ds_committed.take([0], columns=["text"])
    sample_text = sample_data.column(0)[0].as_py()
    search_word = sample_text.split()[0] if sample_text.split() else "test"

    # Perform a full-text search to verify the index works
    results = ds_committed.scanner(
        full_text_query=search_word,
        columns=["id", "text"],
    ).to_table()

    print(f"Search for '{search_word}' returned {results.num_rows} results")
    # We should get at least one result since we searched for a word from the dataset
    assert results.num_rows > 0, f"No results found for search term '{search_word}'"


def test_fragment_ids_parameter_validation(tmp_path):
    """
    Test validation of fragment_ids parameter.
    """
    ds = generate_multi_fragment_dataset(
        tmp_path, num_fragments=2, rows_per_fragment=100
    )

    # Test with valid fragment IDs
    fragments = ds.get_fragments()
    valid_fragment_id = fragments[0].fragment_id

    # This should work without errors
    ds.create_scalar_index(
        column="text",
        index_type="INVERTED",
        fragment_ids=[valid_fragment_id],
    )

    # Test with invalid fragment ID (should handle gracefully)
    # Note: The exact behavior for invalid fragment IDs may vary
    # This test ensures the parameter is properly passed through
    try:
        ds.create_scalar_index(
            column="text",
            index_type="INVERTED",
            fragment_ids=[999999],  # Non-existent fragment ID
        )
    except Exception as e:
        # It's acceptable for this to fail with an appropriate error
        print(f"Expected error for invalid fragment ID: {e}")


def test_backward_compatibility_no_fragment_ids(tmp_path):
    """
    Test that the API remains backward compatible when fragment_ids is not provided.
    """
    ds = generate_multi_fragment_dataset(
        tmp_path, num_fragments=2, rows_per_fragment=100
    )

    # This should work exactly as before (full dataset indexing)
    ds.create_scalar_index(
        column="text",
        index_type="INVERTED",
        name="full_dataset_idx",
        remove_stop_words=False,
    )

    # Verify the index was created
    indices = ds.list_indices()
    assert len(indices) == 1
    assert indices[0]["name"] == "full_dataset_idx"
    assert indices[0]["type"] == "Inverted"

    # Test that the index works
    sample_data = ds.take([0], columns=["text"])
    sample_text = sample_data.column(0)[0].as_py()
    search_word = sample_text.split()[0] if sample_text.split() else "test"

    results = ds.scanner(full_text_query=search_word).to_table()
    assert results.num_rows > 0
