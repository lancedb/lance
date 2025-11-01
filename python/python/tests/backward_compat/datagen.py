# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

# Data generation for backward compatibility tests
#
# This file will be run on the older versions of Lance to generate
# test data that will be read by up-to-date version of Lance in test_compat.py

import shutil

import lance
import pyarrow as pa
import pyarrow.compute as pc
from lance.file import LanceFileWriter
from packaging.version import Version

from backward_compat.util import (
    build_basic_types,
    build_large,
    get_path,
    write_version_file,
)


def write_basic_types():
    path = get_path("basic_types.lance")
    with LanceFileWriter(str(path)) as writer:
        writer.write_batch(build_basic_types())


def write_large():
    path = get_path("large.lance")
    with LanceFileWriter(str(path)) as writer:
        writer.write_batch(build_large())


def write_dataset_pq_buffer():
    # In https://github.com/lancedb/lance/pull/3829, we started storing the PQ
    # codebook in a global buffer instead of the schema metadata as JSON.

    shutil.rmtree(get_path("pq_in_schema"), ignore_errors=True)

    ndims = 32
    nvecs = 512

    data = pa.table(
        {
            "id": pa.array(range(nvecs)),
            "vec": pa.FixedSizeListArray.from_arrays(
                pc.random(ndims * nvecs).cast(pa.float32()), ndims
            ),
        }
    )

    dataset = lance.write_dataset(data, get_path("pq_in_schema"))
    dataset.create_index(
        "vec",
        "IVF_PQ",
        num_partitions=1,
        num_sub_vectors=4,
    )


def write_dataset_json():
    try:
        from lance.indices.builder import IndexConfig
    except ImportError:
        # for versions <= 0.36
        from lance.indices import IndexConfig

    shutil.rmtree(get_path("json"), ignore_errors=True)

    for frag in range(10):
        r = range(frag * 100, (frag + 1) * 100)
        data = pa.table(
            {
                "idx": pa.array(r),
                "json": pa.array([f'{{"val": {i}}}' for i in r], pa.json_()),
            }
        )
        dataset = lance.write_dataset(
            data, get_path("json"), mode="create" if frag == 0 else "append"
        )

    dataset.create_scalar_index(
        "json",
        IndexConfig(
            index_type="json", parameters={"target_index_type": "btree", "path": "val"}
        ),
    )


def write_dataset_scalar_index():
    shutil.rmtree(get_path("scalar_index"), ignore_errors=True)

    for frag in range(10):
        r = range(frag * 100, (frag + 1) * 100)
        data = pa.table(
            {
                "idx": pa.array(r),
                "btree": pa.array(r),
                "bitmap": pa.array(r),
                "label_list": pa.array([[f"label{i}"] for i in r]),
                "ngram": pa.array([f"word{i}" for i in r]),
                "zonemap": pa.array(r),
                "bloomfilter": pa.array(r),
            }
        )
        dataset = lance.write_dataset(
            data, get_path("scalar_index"), mode="create" if frag == 0 else "append"
        )

    dataset.create_scalar_index("btree", "BTREE")
    dataset.create_scalar_index("bitmap", "BITMAP")
    dataset.create_scalar_index("label_list", "LABEL_LIST")

    if Version(lance.__version__) >= Version("0.36.0"):
        dataset.create_scalar_index("ngram", "NGRAM")
        dataset.create_scalar_index("zonemap", "ZONEMAP")
        dataset.create_scalar_index("bloomfilter", "BLOOMFILTER")


def write_dataset_fts_index():
    shutil.rmtree(get_path("fts_index"), ignore_errors=True)

    for frag in range(10):
        r = range(frag * 100, (frag + 1) * 100)
        data = pa.table(
            {
                "idx": pa.array(r),
                "text": pa.array([f"document with words {i} and more text" for i in r]),
            }
        )
        dataset = lance.write_dataset(
            data, get_path("fts_index"), mode="create" if frag == 0 else "append"
        )

    dataset.create_scalar_index("text", "INVERTED")


if __name__ == "__main__":
    write_version_file()
    write_basic_types()
    write_large()

    if Version(lance.__version__) >= Version("0.29.1.beta2"):
        write_dataset_pq_buffer()

    if Version(lance.__version__) >= Version("0.20.0"):
        write_dataset_scalar_index()

    if Version(lance.__version__) >= Version("0.36.0"):
        write_dataset_json()
        write_dataset_fts_index()
