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
        index_id,
        "meta_idx",
        [field_idx],
        dataset_without_index.version,
        set([f.fragment_id for f in dataset_without_index.get_fragments()]),
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
