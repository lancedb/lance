# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

# This script generates Lance files that are read by test_forward_compat.py

from pathlib import Path

import pyarrow as pa
from lance.file import LanceFileWriter


def get_path(name: str):
    dataset_dir = (
        Path(__file__).parent.parent.parent.parent.parent
        / "test_data"
        / "forward_compat"
        / name
    )
    return dataset_dir


def build_basic_types():
    schema = pa.schema(
        [
            pa.field("int", pa.int64()),
            pa.field("float", pa.float32()),
            pa.field("str", pa.string()),
            pa.field("list_int", pa.list_(pa.int64())),
            pa.field("list_str", pa.list_(pa.string())),
            pa.field("struct", pa.struct([pa.field("a", pa.int64())])),
            pa.field("dict", pa.dictionary(pa.int16(), pa.string())),
            pa.field("str_as_dict", pa.string()),
        ]
    )

    return pa.table(
        [
            pa.array(range(1000)),
            pa.array(range(1000), pa.float32()),
            pa.array([str(i) for i in range(1000)]),
            pa.array([list(range(i)) for i in range(1000)]),
            pa.array([[str(i)] for i in range(1000)]),
            pa.array([{"a": i} for i in range(1000)]),
            pa.array(
                [str(i % 10) for i in range(1000)],
                pa.dictionary(pa.int16(), pa.string()),
            ),
            pa.array(["a"] * 500 + ["b"] * 500),
        ],
        schema=schema,
    )


def write_basic_types():
    path = get_path("basic_types.lance")
    with LanceFileWriter(str(path)) as writer:
        writer.write_batch(build_basic_types())


def build_large():
    # ~40MB of vector embedding data (10K 1024-float32)
    fsl_data = pa.array(range(1024 * 1000 * 10), pa.float32())
    fsls = pa.FixedSizeListArray.from_arrays(fsl_data, 1024)
    # ~40 MiB of binary data (10k 4KiB chunks)
    bindata = pa.allocate_buffer(1024 * 1000 * 40)
    offsets = pa.array(
        range(0, (1024 * 1000 * 40) + 4 * 1024, 4 * 1024), pa.int32()
    ).buffers()[1]
    bins = pa.BinaryArray.from_buffers(pa.binary(), 10000, [None, offsets, bindata])

    schema = pa.schema(
        [
            pa.field("int", pa.int32()),
            pa.field("fsl", pa.list_(pa.float32())),
            pa.field("bin", pa.binary()),
        ]
    )

    return pa.table(
        [
            pa.array(range(10000), pa.int32()),
            fsls,
            bins,
        ],
        schema=schema,
    )


def write_large():
    path = get_path("large.lance")
    with LanceFileWriter(str(path)) as writer:
        writer.write_batch(build_large())


if __name__ == "__main__":
    write_basic_types()
    write_large()
