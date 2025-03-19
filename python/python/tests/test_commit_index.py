# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import random
import shutil
import string
from pathlib import Path
from typing import List

import lance
import numpy as np
import pyarrow as pa
import pytest


@pytest.fixture()
def test_table():
    num_rows = 1000
    price = np.random.rand(num_rows) * 100

    def gen_str(n, split="", char_set=string.ascii_letters + string.digits):
        return "".join(random.choices(char_set, k=n))

    meta = np.array([gen_str(100) for _ in range(num_rows)])
    doc = [gen_str(10, " ", string.ascii_letters) for _ in range(num_rows)]
    tbl = pa.Table.from_arrays(
        [
            pa.array(price),
            pa.array(meta),
            pa.array(doc, pa.large_string()),
            pa.array(range(num_rows)),
        ],
        names=["price", "meta", "doc", "id"],
    )
    return tbl


@pytest.fixture()
def dataset_with_index(test_table, tmp_path):
    dataset = lance.write_dataset(test_table, tmp_path)
    dataset.create_scalar_index("meta", index_type="BTREE")
    return dataset


def test_commit_index(dataset_with_index, test_table, tmp_path):
    index_id = dataset_with_index.list_indices()[0]["uuid"]

    # Create a new dataset without index
    dataset_without_index = lance.write_dataset(
        test_table, tmp_path / "dataset_without_index"
    )

    # Copy the index from dataset_with_index to dataset_without_index
    src_index_dir = Path(dataset_with_index.uri) / "_indices" / index_id
    dest_index_dir = Path(dataset_without_index.uri) / "_indices" / index_id
    shutil.copytree(src_index_dir, dest_index_dir)

    # Commit the index to dataset_without_index
    field_idx = dataset_without_index.schema.get_field_index("meta")
    create_index_op = lance.LanceOperation.CreateIndex(
        new_indices=[
            lance.IndexInfo(
                uuid=index_id,
                name="meta_idx",
                fields=[field_idx],
                version=dataset_without_index.version,
                fragment_ids=set(
                    [f.fragment_id for f in dataset_without_index.get_fragments()]
                ),
            )
        ],
        removed_indices=[],
    )
    dataset_without_index = lance.LanceDataset.commit(
        dataset_without_index.uri,
        create_index_op,
        read_version=dataset_without_index.version,
    )

    # Verify that both datasets have the index
    assert len(dataset_with_index.list_indices()) == 1
    assert len(dataset_without_index.list_indices()) == 1

    assert (
        dataset_without_index.list_indices()[0] == dataset_with_index.list_indices()[0]
    )

    # Check if the index is used in scans
    for dataset in [dataset_with_index, dataset_without_index]:
        scanner = dataset.scanner(
            fast_search=True, prefilter=True, filter="meta = 'hello'"
        )
        plan = scanner.explain_plan()
        assert "MaterializeIndex" in plan


@pytest.fixture()
def tmp_tables() -> List[pa.Table]:
    tables = [
        {
            "text": [
                "Frodo was a puppy",
                "There were several kittens playing",
            ],
            "sentiment": ["neutral", "neutral"],
        },
        {
            "text": [
                "Frodo was a happy puppy",
                "Frodo was a very happy puppy",
            ],
            "sentiment": ["positive", "positive"],
        },
        {
            "text": [
                "Frodo was a sad puppy",
                "Frodo was a very sad puppy",
            ],
            "sentiment": ["negative", "negative"],
        },
    ]
    for tb in tables:
        tb["text2"] = tb["text"]
        tb["text3"] = tb["text"]
    return [pa.table(tb) for tb in tables]


def test_indexed_unindexed_fragments(tmp_tables, tmp_path):
    ds = lance.write_dataset(tmp_tables[0], tmp_path, mode="overwrite")
    ds = lance.write_dataset(tmp_tables[1], tmp_path, mode="append")
    ds = lance.write_dataset(tmp_tables[2], tmp_path, mode="append")
    frags = [f for f in ds.get_fragments()]
    index = ds.create_scalar_index(
        "text", "INVERTED", fragment_ids=[frags[0].fragment_id]
    )
    assert isinstance(index, dict)

    indices = [index]
    create_index_op = lance.LanceOperation.CreateIndex(
        new_indices=indices,
        removed_indices=[],
    )
    ds = lance.LanceDataset.commit(
        ds.uri,
        create_index_op,
        read_version=ds.version,
    )

    unindexed_fragments = ds.unindexed_fragments("text_idx")
    assert len(unindexed_fragments) == 2
    assert unindexed_fragments[0].id == frags[1].fragment_id
    assert unindexed_fragments[1].id == frags[2].fragment_id

    indexed_fragments = [f for fs in ds.indexed_fragments("text_idx") for f in fs]
    assert len(indexed_fragments) == 1
    assert indexed_fragments[0].id == frags[0].fragment_id


def test_dfs_query_then_fetch(tmp_tables, tmp_path):
    ds = lance.write_dataset(tmp_tables[0], tmp_path, mode="overwrite")
    ds = lance.write_dataset(tmp_tables[1], tmp_path, mode="append")
    ds = lance.write_dataset(tmp_tables[2], tmp_path, mode="append")
    indices = []
    frags = list(ds.get_fragments())
    for f in frags[:2]:
        # we can create an inverted index distributely
        index = ds.create_scalar_index("text", "INVERTED", fragment_ids=[f.fragment_id])
        assert isinstance(index, dict)
        indices.append(index)

    index = ds.create_scalar_index(
        "text2", "INVERTED", fragment_ids=[frags[0].fragment_id, frags[1].fragment_id]
    )
    indices.append(index)
    index = ds.create_scalar_index(
        "text3", "INVERTED", fragment_ids=[frags[0].fragment_id]
    )
    indices.append(index)
    create_index_op = lance.LanceOperation.CreateIndex(
        new_indices=indices,
        removed_indices=[],
    )

    ds = lance.LanceDataset.commit(
        ds.uri,
        create_index_op,
        read_version=ds.version,
    )

    # test query then fetch
    text_query_fetch = ds.to_table(
        full_text_query={"columns": ["text"], "query": "puppy"},
        prefilter=True,
        with_row_id=True,
    )
    assert sorted(text_query_fetch["_rowid"].to_pylist()) == [
        0,
        1 << 32,
        (1 << 32) + 1,
        2 << 32,
        (2 << 32) + 1,
    ]

    # test dfs query then fetch
    text_dfs_query_fetch = ds.to_table(
        full_text_query={
            "columns": ["text"],
            "query": "puppy",
            "search_type": "DfsQueryThenFetch",
        },
        prefilter=True,
        with_row_id=True,
    )
    assert sorted(text_dfs_query_fetch["_rowid"].to_pylist()) == [
        0,
        1 << 32,
        (1 << 32) + 1,
        2 << 32,
        (2 << 32) + 1,
    ]

    def table_to_tuple(tb):
        return list(zip(tb["_rowid"].to_pylist(), tb["_score"].to_pylist()))

    # it should be the same as dfs query then fetch for column text
    text2_query_fetch = ds.to_table(
        full_text_query={"columns": ["text2"], "query": "puppy"},
        prefilter=True,
        with_row_id=True,
    )
    assert sorted(table_to_tuple(text2_query_fetch)) == sorted(
        table_to_tuple(text_dfs_query_fetch)
    )

    # for column text2, it should be the same as query then fetch
    text2_dfs_query_fetch = ds.to_table(
        full_text_query={
            "columns": ["text2"],
            "query": "puppy",
            "search_type": "DfsQueryThenFetch",
        },
        prefilter=True,
        with_row_id=True,
    )
    assert sorted(table_to_tuple(text2_query_fetch)) == sorted(
        table_to_tuple(text2_dfs_query_fetch)
    )

    text3_dfs_neutral = ds.to_table(
        full_text_query={
            "columns": ["text3"],
            "query": "puppy",
            "search_type": "DfsQueryThenFetch",
        },
        filter="sentiment='neutral'",
        prefilter=True,
        with_row_id=True,
    )
    assert (
        sorted(table_to_tuple(text3_dfs_neutral))
        == sorted(table_to_tuple(text_query_fetch))[:1]
    )

    text3_neutral = ds.to_table(
        full_text_query={"columns": ["text3"], "query": "puppy"},
        filter="sentiment='neutral'",
        prefilter=True,
        with_row_id=True,
    )
    assert sorted(table_to_tuple(text3_neutral)) == sorted(
        table_to_tuple(text3_dfs_neutral)
    )

    text_neutral = ds.to_table(
        full_text_query={"columns": ["text"], "query": "puppy"},
        filter="sentiment='neutral'",
        prefilter=True,
        with_row_id=True,
    )
    assert sorted(table_to_tuple(text_neutral)) == sorted(table_to_tuple(text3_neutral))


def test_fragment_fts(tmp_tables, tmp_path):
    ds = lance.write_dataset(tmp_tables[0], tmp_path, mode="overwrite")
    ds = lance.write_dataset(tmp_tables[1], tmp_path, mode="append")
    ds = lance.write_dataset(tmp_tables[2], tmp_path, mode="append")
    indices = []
    frags = list(ds.get_fragments())
    for f in frags:
        # we can create an inverted index distributely
        index = ds.create_scalar_index("text", "INVERTED", fragment_ids=[f.fragment_id])
        assert isinstance(index, dict)
        indices.append(index)

    create_index_op = lance.LanceOperation.CreateIndex(
        new_indices=indices,
        removed_indices=[],
    )
    ds = lance.LanceDataset.commit(
        ds.uri,
        create_index_op,
        read_version=ds.version,
    )

    # test fragment level full text search
    scanner = ds.scanner(
        full_text_query={"columns": ["text"], "query": "several"},
        prefilter=True,
        with_row_id=True,
        fragments=[frags[0]],
    )
    assert scanner.to_table()["_rowid"].to_pylist() == [1]

    # test fragment level full text search with filter
    scanner = ds.scanner(
        full_text_query={"columns": ["text"], "query": "several"},
        prefilter=True,
        with_row_id=True,
        fragments=[frags[0]],
        filter="sentiment='neutral'",
    )
    assert scanner.to_table()["_rowid"].to_pylist() == [1]

    # test fragment level full text search with filter
    scanner = ds.scanner(
        full_text_query={"columns": ["text"], "query": "puppy"},
        prefilter=True,
        with_row_id=True,
        fragments=[frags[0]],
        filter="sentiment='positive'",
    )
    assert scanner.to_table()["_rowid"].to_pylist() == []

    # test second fragment
    # test fragment level full text search
    scanner = ds.scanner(
        full_text_query={"columns": ["text"], "query": "very"},
        prefilter=True,
        with_row_id=True,
        fragments=[frags[1]],
    )
    assert scanner.to_table()["_rowid"].to_pylist() == [(1 << 32) + 1]

    # test fragment level full text search with filter
    scanner = ds.scanner(
        full_text_query={"columns": ["text"], "query": "very"},
        prefilter=True,
        with_row_id=True,
        fragments=[frags[1]],
        filter="sentiment='neutral'",
    )
    assert scanner.to_table()["_rowid"].to_pylist() == []

    # test fragment level full text search with filter
    scanner = ds.scanner(
        full_text_query={"columns": ["text"], "query": "very"},
        prefilter=True,
        with_row_id=True,
        fragments=[frags[1]],
        filter="sentiment='positive'",
    )
    assert scanner.to_table()["_rowid"].to_pylist() == [(1 << 32) + 1]

    with pytest.raises(ValueError):
        # DfsQueryThenFetch is not supported for fragment level full text search,
        # because it requires a new api to expose distributed frequency information.
        scanner = ds.scanner(
            full_text_query={
                "columns": ["text"],
                "query": "very",
                "search_type": "DfsQueryThenFetch",
            },
            prefilter=True,
            with_row_id=True,
            fragments=[frags[1]],
        )
        assert scanner.to_table()["_rowid"].to_pylist() == [(1 << 32) + 1]
