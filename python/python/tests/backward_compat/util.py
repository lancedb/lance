# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

# Utilities shared by datagen.py and test_compat.py
#
# Everything here must be runnable by older versions of Lance.

from pathlib import Path

import lance
import pyarrow as pa
from packaging.version import Version


def get_path(name: str):
    path = (
        Path(__file__).parent.parent.parent.parent.parent
        / "test_data"
        / "backward_compat"
        / name
    )
    return path


def get_old_version_path():
    return get_path("old_version.txt")


def write_version_file():
    path = get_old_version_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(lance.__version__)


def old_version():
    if get_old_version_path().exists():
        with open(get_old_version_path(), "r") as f:
            return Version(f.read().strip())
    else:
        return None


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
