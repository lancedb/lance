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


def _get_field_id_by_name(lance_schema, field_name):
    fields = lance_schema.fields()
    for field in fields:
        if field.name() == field_name:
            return field.id()
    return None


def test_commit_index(dataset_with_index, test_table, tmp_path):
    from lance.dataset import Index

    index_id = dataset_with_index.describe_indices()[0].segments[0].uuid

    # Create a new dataset without index
    dataset_without_index = lance.write_dataset(
        test_table, tmp_path / "dataset_without_index"
    )

    # Copy the index from dataset_with_index to dataset_without_index
    src_index_dir = Path(dataset_with_index.uri) / "_indices" / index_id
    dest_index_dir = Path(dataset_without_index.uri) / "_indices" / index_id
    shutil.copytree(src_index_dir, dest_index_dir)

    # Get the field id instead of field index
    # as they are different in nested data
    field_id = _get_field_id_by_name(dataset_without_index.lance_schema, "meta")

    # Create an Index object
    index = Index(
        uuid=index_id,
        name="meta_idx",
        fields=[field_id],
        dataset_version=dataset_without_index.version,
        fragment_ids=set(
            [f.fragment_id for f in dataset_without_index.get_fragments()]
        ),
        index_version=0,
    )

    create_index_op = lance.LanceOperation.CreateIndex(
        new_indices=[index],
        removed_indices=[],
    )
    dataset_without_index = lance.LanceDataset.commit(
        dataset_without_index.uri,
        create_index_op,
        read_version=dataset_without_index.version,
    )

    # Verify the manually committed index matches the original index stats
    stats_with = dataset_with_index.stats.index_stats("meta_idx")
    stats_without = dataset_without_index.stats.index_stats("meta_idx")

    assert stats_without["name"] == stats_with["name"]
    assert stats_without["index_type"] == stats_with["index_type"]
    assert stats_without["num_indexed_rows"] == stats_with["num_indexed_rows"]

    # Check if the index is used in scans
    for dataset in [dataset_with_index, dataset_without_index]:
        scanner = dataset.scanner(
            fast_search=True, prefilter=True, filter="meta = 'hello'"
        )
        plan = scanner.explain_plan()
        assert "ScalarIndexQuery: query=[meta = hello]@meta_idx" in plan


def test_commit_index_with_files(dataset_with_index, test_table, tmp_path):
    """Test that the files field on Index round-trips through commit."""
    from lance.dataset import Index, IndexFile

    # Get info about the existing index created by the fixture
    original_desc = dataset_with_index.describe_indices()[0]
    index_id = original_desc.segments[0].uuid

    # Verify the original index has file sizes
    original_size = original_desc.total_size_bytes
    assert original_size is not None and original_size > 0

    # Create a new dataset without index
    dataset_without_index = lance.write_dataset(
        test_table, tmp_path / "dataset_without_index"
    )

    # Copy the index files from dataset_with_index to dataset_without_index
    src_index_dir = Path(dataset_with_index.uri) / "_indices" / index_id
    dest_index_dir = Path(dataset_without_index.uri) / "_indices" / index_id
    shutil.copytree(src_index_dir, dest_index_dir)

    # Get the field id
    field_id = _get_field_id_by_name(dataset_without_index.lance_schema, "meta")

    # Create IndexFile objects with custom sizes to verify they round-trip
    index_files = [
        IndexFile(path="index.idx", size_bytes=1024),
        IndexFile(path="auxiliary.bin", size_bytes=2048),
    ]

    # Create an Index object with the files field
    index = Index(
        uuid=index_id,
        name="meta_idx",
        fields=[field_id],
        dataset_version=dataset_without_index.version,
        fragment_ids=set(
            [f.fragment_id for f in dataset_without_index.get_fragments()]
        ),
        index_version=0,
        files=index_files,
    )

    create_index_op = lance.LanceOperation.CreateIndex(
        new_indices=[index],
        removed_indices=[],
    )
    dataset_without_index = lance.LanceDataset.commit(
        dataset_without_index.uri,
        create_index_op,
        read_version=dataset_without_index.version,
    )

    # Read back the transaction to verify the files were stored
    transactions = dataset_without_index.get_transactions(1)
    assert len(transactions) == 1
    transaction = transactions[0]
    assert transaction is not None
    assert transaction.operation is not None

    # The operation should be a CreateIndex with our index that has files
    op = transaction.operation
    assert len(op.new_indices) == 1
    committed_index = op.new_indices[0]
    assert committed_index.files is not None
    assert len(committed_index.files) == 2

    # Verify the file sizes match what we set
    files_by_path = {f.path: f.size_bytes for f in committed_index.files}
    assert files_by_path["index.idx"] == 1024
    assert files_by_path["auxiliary.bin"] == 2048


def test_commit_index_with_included_fields(dataset_with_index, test_table, tmp_path):
    """Test that included_fields (covering columns) round-trip through the Python
    transaction bindings."""
    from lance.dataset import Index

    original_desc = dataset_with_index.describe_indices()[0]
    index_id = original_desc.segments[0].uuid

    dataset_without_index = lance.write_dataset(
        test_table, tmp_path / "dataset_without_index"
    )
    src_index_dir = Path(dataset_with_index.uri) / "_indices" / index_id
    dest_index_dir = Path(dataset_without_index.uri) / "_indices" / index_id
    shutil.copytree(src_index_dir, dest_index_dir)

    meta_id = _get_field_id_by_name(dataset_without_index.lance_schema, "meta")
    price_id = _get_field_id_by_name(dataset_without_index.lance_schema, "price")

    # Declare "price" as a covering (included) column on the index.
    index = Index(
        uuid=index_id,
        name="meta_idx",
        fields=[meta_id],
        dataset_version=dataset_without_index.version,
        fragment_ids=set(
            [f.fragment_id for f in dataset_without_index.get_fragments()]
        ),
        index_version=0,
        included_fields=[price_id],
    )

    create_index_op = lance.LanceOperation.CreateIndex(
        new_indices=[index],
        removed_indices=[],
    )
    dataset_without_index = lance.LanceDataset.commit(
        dataset_without_index.uri,
        create_index_op,
        read_version=dataset_without_index.version,
    )

    # Read back the transaction to verify the covering columns were stored.
    transaction = dataset_without_index.get_transactions(1)[0]
    committed_index = transaction.operation.new_indices[0]
    assert committed_index.included_fields == [price_id]


def test_commit_index_rejects_malformed_included_fields(
    dataset_with_index, test_table, tmp_path
):
    """A malformed included_fields value must propagate an error, not be silently
    coerced to an empty covering set -- otherwise the covered declaration and its
    writer fence are dropped while the index files still carry payload columns."""
    from lance.dataset import Index

    original_desc = dataset_with_index.describe_indices()[0]
    index_id = original_desc.segments[0].uuid

    dataset_without_index = lance.write_dataset(
        test_table, tmp_path / "dataset_without_index"
    )
    src_index_dir = Path(dataset_with_index.uri) / "_indices" / index_id
    dest_index_dir = Path(dataset_without_index.uri) / "_indices" / index_id
    shutil.copytree(src_index_dir, dest_index_dir)

    meta_id = _get_field_id_by_name(dataset_without_index.lance_schema, "meta")

    # 2**40 is a valid Python int but overflows the i32 field-id type.
    index = Index(
        uuid=index_id,
        name="meta_idx",
        fields=[meta_id],
        dataset_version=dataset_without_index.version,
        fragment_ids=set(
            [f.fragment_id for f in dataset_without_index.get_fragments()]
        ),
        index_version=0,
        included_fields=[2**40],
    )
    create_index_op = lance.LanceOperation.CreateIndex(
        new_indices=[index],
        removed_indices=[],
    )
    with pytest.raises((OverflowError, ValueError, TypeError)):
        lance.LanceDataset.commit(
            dataset_without_index.uri,
            create_index_op,
            read_version=dataset_without_index.version,
        )


def test_commit_index_propagates_raising_included_fields(
    dataset_with_index, test_table, tmp_path
):
    """An `included_fields` access that raises (e.g. a lazy proxy whose deferred
    load fails) must propagate the error. Swallowing it would commit an empty
    covering set while the index files still carry the payload columns."""

    class RaisingIndex:
        def __init__(self, uuid, name, fields, dataset_version, fragment_ids):
            self.uuid = uuid
            self.name = name
            self.fields = fields
            self.dataset_version = dataset_version
            self.fragment_ids = fragment_ids
            self.index_version = 0
            self.created_at = None
            self.base_id = None
            self.files = None

        @property
        def included_fields(self):
            raise RuntimeError("deferred covering-column load failed")

    original_desc = dataset_with_index.describe_indices()[0]
    index_id = original_desc.segments[0].uuid

    dataset_without_index = lance.write_dataset(
        test_table, tmp_path / "dataset_without_index"
    )
    src_index_dir = Path(dataset_with_index.uri) / "_indices" / index_id
    dest_index_dir = Path(dataset_without_index.uri) / "_indices" / index_id
    shutil.copytree(src_index_dir, dest_index_dir)

    meta_id = _get_field_id_by_name(dataset_without_index.lance_schema, "meta")
    index = RaisingIndex(
        uuid=index_id,
        name="meta_idx",
        fields=[meta_id],
        dataset_version=dataset_without_index.version,
        fragment_ids=set(
            [f.fragment_id for f in dataset_without_index.get_fragments()]
        ),
    )
    create_index_op = lance.LanceOperation.CreateIndex(
        new_indices=[index],
        removed_indices=[],
    )
    with pytest.raises(RuntimeError, match="deferred covering-column load failed"):
        lance.LanceDataset.commit(
            dataset_without_index.uri,
            create_index_op,
            read_version=dataset_without_index.version,
        )


def test_commit_index_propagates_getter_attribute_error(
    dataset_with_index, test_table, tmp_path
):
    """An AttributeError raised INSIDE an `included_fields` property getter (e.g. a
    lazy proxy whose backing attribute is unset) must propagate -- it is not the same
    as the attribute being absent, and swallowing it would commit an empty covering
    set while the index files keep the payload columns."""

    class LazyIndex:
        def __init__(self, uuid, name, fields, dataset_version, fragment_ids):
            self.uuid = uuid
            self.name = name
            self.fields = fields
            self.dataset_version = dataset_version
            self.fragment_ids = fragment_ids
            self.index_version = 0
            self.created_at = None
            self.base_id = None
            self.files = None

        @property
        def included_fields(self):
            # `_inner` is never set: the getter itself raises AttributeError.
            return self._inner.included_fields

    original_desc = dataset_with_index.describe_indices()[0]
    index_id = original_desc.segments[0].uuid

    dataset_without_index = lance.write_dataset(
        test_table, tmp_path / "dataset_without_index"
    )
    src_index_dir = Path(dataset_with_index.uri) / "_indices" / index_id
    dest_index_dir = Path(dataset_without_index.uri) / "_indices" / index_id
    shutil.copytree(src_index_dir, dest_index_dir)

    meta_id = _get_field_id_by_name(dataset_without_index.lance_schema, "meta")
    index = LazyIndex(
        uuid=index_id,
        name="meta_idx",
        fields=[meta_id],
        dataset_version=dataset_without_index.version,
        fragment_ids=set(
            [f.fragment_id for f in dataset_without_index.get_fragments()]
        ),
    )
    create_index_op = lance.LanceOperation.CreateIndex(
        new_indices=[index],
        removed_indices=[],
    )
    with pytest.raises(AttributeError, match="_inner"):
        lance.LanceDataset.commit(
            dataset_without_index.uri,
            create_index_op,
            read_version=dataset_without_index.version,
        )


def test_commit_index_with_index_details(dataset_with_index, test_table, tmp_path):
    """Test that index_details round-trip through Python transaction bindings."""
    from lance.dataset import Index

    original_txn = dataset_with_index.get_transactions(1)[0]
    original_index = original_txn.operation.new_indices[0]
    assert original_index.index_details is not None

    index_id = original_index.uuid
    dataset_without_index = lance.write_dataset(
        test_table, tmp_path / "dataset_without_index"
    )

    src_index_dir = Path(dataset_with_index.uri) / "_indices" / index_id
    dest_index_dir = Path(dataset_without_index.uri) / "_indices" / index_id
    shutil.copytree(src_index_dir, dest_index_dir)

    field_id = _get_field_id_by_name(dataset_without_index.lance_schema, "meta")
    index = Index(
        uuid=index_id,
        name="meta_idx",
        fields=[field_id],
        dataset_version=dataset_without_index.version,
        fragment_ids=set(
            [f.fragment_id for f in dataset_without_index.get_fragments()]
        ),
        index_version=0,
        index_details=original_index.index_details,
    )

    create_index_op = lance.LanceOperation.CreateIndex(
        new_indices=[index],
        removed_indices=[],
    )
    dataset_without_index = lance.LanceDataset.commit(
        dataset_without_index.uri,
        create_index_op,
        read_version=dataset_without_index.version,
    )

    committed_txn = dataset_without_index.get_transactions(1)[0]
    committed_index = committed_txn.operation.new_indices[0]
    assert committed_index.index_details == original_index.index_details
