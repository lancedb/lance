# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import io
import subprocess
import sys
import tarfile
import textwrap

import lance
import pyarrow as pa
import pytest
from lance import Blob, BlobColumn


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


def test_scan_blob_as_binary(tmp_path):
    values = [b"foo", b"bar", b"baz"]
    arr = pa.array(values, pa.large_binary())
    table = pa.table(
        [arr],
        schema=pa.schema(
            [
                pa.field(
                    "blobs", pa.large_binary(), metadata={"lance-encoding:blob": "true"}
                )
            ]
        ),
    )
    ds = lance.write_dataset(table, tmp_path / "test_ds")

    tbl = ds.scanner(columns=["blobs"], blob_handling="all_binary").to_table()
    assert tbl.column("blobs").to_pylist() == values


def test_fragment_scan_blob_as_binary(tmp_path):
    values = [b"foo", b"bar", b"baz"]
    arr = pa.array(values, pa.large_binary())
    table = pa.table(
        [arr],
        schema=pa.schema(
            [
                pa.field(
                    "blobs", pa.large_binary(), metadata={"lance-encoding:blob": "true"}
                )
            ]
        ),
    )
    ds = lance.write_dataset(table, tmp_path / "test_ds")

    fragment = ds.get_fragments()[0]

    tbl = fragment.scanner(columns=["blobs"], blob_handling="all_binary").to_table()
    assert tbl.column("blobs").to_pylist() == values

    tbl = fragment.to_table(columns=["blobs"], blob_handling="all_binary")
    assert tbl.column("blobs").to_pylist() == values


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

    values = pa.array([b"qux", b"quux", b"corge"], pa.large_binary())
    idx = pa.array([3, 4, 5], pa.uint64())
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
    ds.insert(table)
    return ds


def test_blob_files(dataset_with_blobs):
    row_ids = (
        dataset_with_blobs.to_table(columns=[], with_row_id=True)
        .column("_rowid")
        .to_pylist()
    )
    blobs = dataset_with_blobs.take_blobs("blobs", ids=row_ids)

    for expected in [b"foo", b"bar", b"baz"]:
        with blobs.pop(0) as f:
            assert f.read() == expected


def test_blob_files_close_no_shutdown_panic(tmp_path):
    script = textwrap.dedent(
        f"""
        import pyarrow as pa
        import lance

        table = pa.table(
            [pa.array([b"foo", b"bar"], pa.large_binary())],
            schema=pa.schema(
                [
                    pa.field(
                        "blob",
                        pa.large_binary(),
                        metadata={{"lance-encoding:blob": "true"}},
                    )
                ]
            ),
        )
        ds = lance.write_dataset(table, {str(tmp_path / "ds")!r})
        row_ids = ds.to_table(columns=[], with_row_id=True).column("_rowid").to_pylist()
        blobs = ds.take_blobs("blob", ids=row_ids)
        for blob in blobs:
            blob.close()
        print("done")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "interpreter_lifecycle.rs" not in result.stderr
    assert "The Python interpreter is not initialized" not in result.stderr


def test_blob_files_by_address(dataset_with_blobs):
    addresses = (
        dataset_with_blobs.to_table(columns=[], with_row_address=True)
        .column("_rowaddr")
        .to_pylist()
    )
    blobs = dataset_with_blobs.take_blobs("blobs", addresses=addresses)

    for expected in [b"foo", b"bar", b"baz"]:
        with blobs.pop(0) as f:
            assert f.read() == expected


def test_blob_files_by_address_with_stable_row_ids(tmp_path):
    table = pa.table(
        {
            "blobs": pa.array([b"foo"], pa.large_binary()),
            "idx": pa.array([0], pa.uint64()),
        },
        schema=pa.schema(
            [
                pa.field(
                    "blobs", pa.large_binary(), metadata={"lance-encoding:blob": "true"}
                ),
                pa.field("idx", pa.uint64()),
            ]
        ),
    )
    ds = lance.write_dataset(
        table,
        tmp_path / "test_ds",
        enable_stable_row_ids=True,
    )

    ds.insert(
        pa.table(
            {
                "blobs": pa.array([b"bar"], pa.large_binary()),
                "idx": pa.array([1], pa.uint64()),
            },
            schema=table.schema,
        )
    )

    t = ds.to_table(columns=["idx"], with_row_address=True)
    row_idx = t.column("idx").to_pylist().index(1)
    addr = t.column("_rowaddr").to_pylist()[row_idx]

    blobs = ds.take_blobs("blobs", addresses=[addr])
    assert len(blobs) == 1
    with blobs[0] as f:
        assert f.read() == b"bar"


def test_blob_by_indices(tmp_path, dataset_with_blobs):
    indices = [0, 4]
    blobs = dataset_with_blobs.take_blobs("blobs", indices=indices)

    blobs2 = dataset_with_blobs.take_blobs("blobs", ids=[0, (1 << 32) + 1])
    assert len(blobs) == len(blobs2)
    for b1, b2 in zip(blobs, blobs2):
        with b1 as f1, b2 as f2:
            assert f1.read() == f2.read()


def test_blob_file_seek(tmp_path, dataset_with_blobs):
    row_ids = (
        dataset_with_blobs.to_table(columns=[], with_row_id=True)
        .column("_rowid")
        .to_pylist()
    )
    blobs = dataset_with_blobs.take_blobs("blobs", ids=row_ids)
    with blobs[1] as f:
        assert f.seek(1) == 1
        assert f.read(1) == b"a"


def test_null_blobs(tmp_path):
    table = pa.table(
        {
            "id": range(100),
            "blob": pa.array([None] * 100, pa.large_binary()),
        },
        schema=pa.schema(
            [
                pa.field("id", pa.uint64()),
                pa.field(
                    "blob", pa.large_binary(), metadata={"lance-encoding:blob": "true"}
                ),
            ]
        ),
    )
    ds = lance.write_dataset(table, tmp_path / "test_ds")

    blobs = ds.take_blobs("blob", ids=range(100))
    for blob in blobs:
        assert blob.size() == 0

    ds.insert(pa.table({"id": pa.array(range(100, 200), pa.uint64())}))

    ds.add_columns(
        pa.field(
            "more_blob",
            pa.large_binary(),
            metadata={"lance-encoding:blob": "true"},
        )
    )

    for blob_col in ["blob", "more_blob"]:
        blobs = ds.take_blobs(blob_col, indices=range(100, 200))
        for blob in blobs:
            assert blob.size() == 0

        blobs = ds.to_table(columns=[blob_col])
        for blob in blobs.column(blob_col):
            py_blob = blob.as_py()
            # When we write blobs to a file we store the position as 1 and size as 0
            # to avoid needing a validity buffer.
            assert py_blob is None or py_blob == {
                "position": 1,
                "size": 0,
            }


def test_blob_file_read_middle(tmp_path, dataset_with_blobs):
    # This regresses an issue where we were not setting the cursor
    # correctly after a call to `read` when the blob was not the
    # first thing in the file.
    row_ids = (
        dataset_with_blobs.to_table(columns=[], with_row_id=True)
        .column("_rowid")
        .to_pylist()
    )
    blobs = dataset_with_blobs.take_blobs("blobs", ids=row_ids)
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
        dataset_with_blobs.take_blobs("blobs", ids=row_ids)


def test_scan_blob(tmp_path, dataset_with_blobs):
    ds = dataset_with_blobs.scanner(filter="idx = 2").to_table()
    assert ds.num_rows == 1


def test_blob_extension_write_inline(tmp_path):
    table = pa.table({"blob": lance.blob_array([b"foo", b"bar"])})
    ds = lance.write_dataset(
        table,
        tmp_path / "test_ds_v2",
        data_storage_version="2.2",
    )

    desc = ds.to_table(columns=["blob"]).column("blob").chunk(0)
    assert pa.types.is_struct(desc.type)

    blobs = ds.take_blobs("blob", indices=[0, 1])
    with blobs[0] as f:
        assert f.read() == b"foo"


def test_blob_extension_write_external(tmp_path):
    blob_path = tmp_path / "external_blob.bin"
    blob_path.write_bytes(b"hello")
    uri = blob_path.as_uri()

    table = pa.table({"blob": lance.blob_array([uri])})
    ds = lance.write_dataset(
        table,
        tmp_path / "test_ds_v2_external",
        data_storage_version="2.2",
    )

    blob = ds.take_blobs("blob", indices=[0])[0]
    assert blob.size() == 5
    with blob as f:
        assert f.read() == b"hello"


def test_blob_extension_write_external_slice(tmp_path):
    tar_path = tmp_path / "container.tar"
    names = ["a.bin", "b.bin", "c.bin"]
    payloads = [b"alpha", b"bravo", b"charlie"]

    # Build a tar container with three distinct binary entries.
    with tarfile.open(tar_path, "w") as tf:
        for name, data in zip(names, payloads):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

    # Re-open the tar to obtain offsets and sizes for each member.
    positions: list[int] = []
    sizes: list[int] = []
    with tarfile.open(tar_path, "r") as tf:
        for name in names:
            member = tf.getmember(name)
            positions.append(member.offset_data)
            sizes.append(member.size)

    uri = tar_path.as_uri()

    blob_values = [
        Blob.from_uri(uri, position, size) for position, size in zip(positions, sizes)
    ]

    table = pa.table({"blob": lance.blob_array(blob_values)})

    ds = lance.write_dataset(
        table,
        tmp_path / "ds",
        data_storage_version="2.2",
    )

    blobs = ds.take_blobs("blob", indices=[0, 1, 2])
    assert len(blobs) == len(payloads)

    for expected, blob_file in zip(payloads, blobs):
        assert blob_file.size() == len(expected)
        with blob_file as f:
            assert f.read() == expected
