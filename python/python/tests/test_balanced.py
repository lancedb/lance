# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import lance
import pyarrow as pa
import pytest


@pytest.fixture(scope="module")
def big_val():
    # 1 MiB per value
    return b"0" * 1024 * 1024


# 16 batches of 8 rows = 128 rows
def balanced_datagen(big_val):
    for batch_idx in range(16):
        start = batch_idx * 8
        end = start + 8
        values = pa.array([big_val for _ in range(start, end)], pa.large_binary())
        idx = pa.array(range(start, end), pa.uint64())
        table = pa.record_batch(
            [values, idx],
            schema=pa.schema(
                [
                    pa.field(
                        "blobs",
                        pa.large_binary(),
                        metadata={
                            "lance-schema:storage-class": "blob",
                        },
                    ),
                    pa.field("idx", pa.uint64()),
                ]
            ),
        )
        yield table


@pytest.fixture
def balanced_dataset(tmp_path, big_val):
    # 16 MiB per file, 128 total MiB, so we should have 8 blob files
    #
    # In addition, max_rows_per_file=64 means we should get 2 regular files
    schema = next(iter(balanced_datagen(big_val))).schema
    return lance.write_dataset(
        balanced_datagen(big_val),
        tmp_path / "test_ds",
        max_bytes_per_file=16 * 1024 * 1024,
        max_rows_per_file=64,
        schema=schema,
    )


def test_append_then_take(balanced_dataset, tmp_path, big_val):
    blob_dir = tmp_path / "test_ds" / "_blobs" / "data"
    assert len(list(blob_dir.iterdir())) == 8

    # A read will only return non-blob columns
    assert balanced_dataset.to_table() == pa.table(
        {
            "idx": pa.array(range(128), pa.uint64()),
        }
    )

    # Now verify we can append some data
    ds = lance.write_dataset(
        balanced_datagen(big_val),
        tmp_path / "test_ds",
        max_bytes_per_file=32 * 1024 * 1024,
        schema=balanced_dataset.schema,
        mode="append",
    )

    assert len(list(blob_dir.iterdir())) == 12

    assert ds.to_table() == pa.table(
        {
            "idx": pa.array(list(range(128)) + list(range(128)), pa.uint64()),
        }
    )

    # Verify we can take blob values
    row_ids = ds.to_table(columns=[], with_row_id=True).column("_rowid")

    take_tbl = ds._take_rows(row_ids.to_pylist(), columns=["idx", "blobs"])

    blobs = take_tbl.column("blobs")
    for val in blobs:
        assert val.as_py() == big_val


def test_delete(balanced_dataset):
    # This will delete some of the first fragment (deletion vector) and
    # the entire second fragment
    balanced_dataset.delete("idx >= 40")

    row_ids = balanced_dataset.to_table(columns=[], with_row_id=True).column("_rowid")

    assert len(row_ids) == 40

    assert balanced_dataset._take_rows(
        row_ids.to_pylist(), columns=["idx"]
    ) == pa.table(
        {
            "idx": pa.array(list(range(40)), pa.uint64()),
        }
    )

    assert (
        len(balanced_dataset._take_rows(row_ids.to_pylist(), columns=["blobs"])) == 40
    )

    assert len(balanced_dataset._take_rows([100], columns=["idx"])) == 0
    assert len(balanced_dataset._take_rows([100], columns=["blobs"])) == 0

    assert len(balanced_dataset._take_rows(range(20, 80), columns=["idx"])) == 20
    assert len(balanced_dataset._take_rows(range(20, 80), columns=["blobs"])) == 20


# TODO: Once https://github.com/lancedb/lance/pull/3041 merges we will
#       want to test partial appends.  We need to make sure an append of
#       non-blob data is supported.  In order to do this we need to make
#       sure a blob tx is created that marks the row ids as used so that
#       the two row id sequences stay in sync.
#
# def test_one_sided_append(balanced_dataset, tmp_path):
#     # Write new data, but only to the idx column
#     ds = lance.write_dataset(
#         pa.table({"idx": pa.array(range(128, 256), pa.uint64())}),
#         tmp_path / "test_ds",
#         max_bytes_per_file=32 * 1024 * 1024,
#         mode="append",
#     )

#     print(ds.to_table())
