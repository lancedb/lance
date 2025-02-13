# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import random
import shutil
import string
from pathlib import Path

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
        new_indices=[lance.Index(
            uuid=index_id,
            name="meta_idx",
            fields=[field_idx],
            version=dataset_without_index.version,
            fragment_ids=set([f.fragment_id for f in dataset_without_index.get_fragments()]),
        )],
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


def create_multi_fragments_table(tmp_path) -> lance.LanceDataset:
    data = pa.table(
        {
            "text": [
                "Frodo was a puppy",
                "There were several kittens playing",
            ],
            "sentiment": ["neutral", "neutral"],
        }
    )
    data2 = pa.table(
        {
            "text": [
                "Frodo was a happy puppy",
                "Frodo was a very happy puppy",
            ],
            "sentiment": ["positive", "positive"],
        }
    )

    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    ds = lance.write_dataset(data2, tmp_path, mode="append")
    return ds


def test_indexed_unindexed_fragments(tmp_path):
    ds = create_multi_fragments_table(tmp_path)
    frags = [f for f in ds.get_fragments()]
    index = ds.create_scalar_index("text", "INVERTED", fragment_ids=[frags[0].fragment_id])
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
    assert len(unindexed_fragments) == 1
    assert unindexed_fragments[0].id == frags[1].fragment_id

    indexed_fragments = [f for fs in ds.indexed_fragments("text_idx") for f in fs]
    assert len(indexed_fragments) == 1
    assert indexed_fragments[0].id == frags[0].fragment_id


def test_commit_index2(tmp_path):
    ds = create_multi_fragments_table(tmp_path)

    indices = []
    for f in ds.get_fragments():
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

    indices = []
    frags = [f for f in ds.get_fragments()]

    for f in frags:
        index = ds.create_scalar_index("sentiment", "BITMAP", fragment_ids=[f.fragment_id])
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

    unindexed_fragments = ds.unindexed_fragments("text_idx")
    assert len(unindexed_fragments) == 0

    indexed_fragments = [f for fs in ds.indexed_fragments("text_idx") for f in fs]
    assert len(indexed_fragments) == 2
    assert indexed_fragments[0].id == frags[0].fragment_id
    assert indexed_fragments[1].id == frags[1].fragment_id

    results = ds.to_table(
        full_text_query="puppy",
        filter="sentiment='positive'",
        prefilter=True,
        with_row_id=True,
    )
    assert results["_rowid"].to_pylist() == [2, 3]
