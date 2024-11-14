# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import lance
import pyarrow as pa
import pytest
from lance import BlobColumn


def test_blob_read_from_binary():
    values = [b"foo", b"bar", b"baz"]
    data = pa.table(
        {
            "bin": pa.array(values, type=pa.binary()),
            "largebin": pa.array(values, type=pa.large_binary()),
        }
    )

    for col_name in ["bin", "largebin"]:
        blobs = BlobColumn(data.column(col_name))
        for i, f in enumerate(blobs):
            assert f.read() in values[i]


def test_blob_reject_invalid_col():
    values = pa.array([1, 2, 3])
    with pytest.raises(ValueError, match="Expected a binary array"):
        BlobColumn(values)


def test_blob_descriptions(tmp_path):
    values = pa.array([b"foo", b"bar", b"baz"], pa.large_binary())
    table = pa.table(
        [values],
        schema=pa.schema(
            [
                pa.field(
                    "blobs", pa.large_binary(), metadata={"lance-encoding:blob": "true"}
                )
            ]
        ),
    )
    ds = lance.write_dataset(table, tmp_path / "test_ds")
    # These positions may be surprising but lance pads buffers to 64-byte boundaries
    expected_positions = pa.array([0, 64, 128], pa.uint64())
    expected_sizes = pa.array([3, 3, 3], pa.uint64())
    descriptions = ds.to_table().column("blobs").chunk(0)

    assert descriptions.field(0) == expected_positions
    assert descriptions.field(1) == expected_sizes


@pytest.fixture
def dataset_with_blobs(tmp_path):
    values = pa.array([b"foo", b"bar", b"baz"], pa.large_binary())
    idx = pa.array([0, 1, 2], pa.uint64())
    table = pa.table(
        [values, idx],
        schema=pa.schema(
            [
                pa.field(
                    "blobs", pa.large_binary(), metadata={"lance-encoding:blob": "true"}
                ),
                pa.field("idx", pa.uint64()),
            ]
        ),
    )
    ds = lance.write_dataset(table, tmp_path / "test_ds")
    return ds


def test_blob_files(tmp_path, dataset_with_blobs):
    row_ids = (
        dataset_with_blobs.to_table(columns=[], with_row_id=True)
        .column("_rowid")
        .to_pylist()
    )
    blobs = dataset_with_blobs.take_blobs(row_ids, "blobs")

    for expected in [b"foo", b"bar", b"baz"]:
        with blobs.pop(0) as f:
            assert f.read() == expected


def test_blob_file_seek(tmp_path, dataset_with_blobs):
    row_ids = (
        dataset_with_blobs.to_table(columns=[], with_row_id=True)
        .column("_rowid")
        .to_pylist()
    )
    blobs = dataset_with_blobs.take_blobs(row_ids, "blobs")
    with blobs[1] as f:
        assert f.seek(1) == 1
        assert f.read(1) == b"a"


def test_blob_file_read_middle(tmp_path, dataset_with_blobs):
    # This regresses an issue where we were not setting the cursor
    # correctly after a call to `read` when the blob was not the
    # first thing in the file.
    row_ids = (
        dataset_with_blobs.to_table(columns=[], with_row_id=True)
        .column("_rowid")
        .to_pylist()
    )
    blobs = dataset_with_blobs.take_blobs(row_ids, "blobs")
    with blobs[1] as f:
        assert f.read(1) == b"b"
        assert f.read(1) == b"a"
        assert f.read(1) == b"r"


def test_take_deleted_blob(tmp_path, dataset_with_blobs):
    row_ids = (
        dataset_with_blobs.to_table(columns=[], with_row_id=True)
        .column("_rowid")
        .to_pylist()
    )
    dataset_with_blobs.delete("idx = 1")

    with pytest.raises(
        NotImplementedError,
        match="A take operation that includes row addresses must not target deleted",
    ):
        dataset_with_blobs.take_blobs(row_ids, "blobs")
