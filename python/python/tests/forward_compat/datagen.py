# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

# Data generation for forward compatibility tests
#
# This file will be run on the up-to-date version of Lance to generate
# test data that will be read by older versions of Lance in test_compat.py

import shutil

import lance
import pyarrow as pa
import pyarrow.compute as pc
from lance.file import LanceFileWriter
from lance.indices.builder import IndexConfig

from forward_compat.util import build_basic_types, build_large, get_path


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
    shutil.rmtree(get_path("json"), ignore_errors=True)

    data = pa.table(
        {
            "idx": pa.array(range(1000)),
            "json": pa.array([f'{{"val": {i}}}' for i in range(1000)], pa.json_()),
        }
    )

    dataset = lance.write_dataset(data, get_path("json"), max_rows_per_file=100)
    dataset.create_scalar_index(
        "json",
        IndexConfig(
            index_type="json", parameters={"target_index_type": "btree", "path": "val"}
        ),
    )


def write_dataset_btree_index():
    shutil.rmtree(get_path("btree_index"), ignore_errors=True)

    data = pa.table(
        {
            "idx": pa.array(range(1000)),
            "btree": pa.array(range(1000)),
        }
    )

    dataset = lance.write_dataset(data, get_path("btree_index"), max_rows_per_file=100)
    dataset.create_scalar_index("btree", "BTREE")


def write_dataset_bitmap_labellist_index():
    shutil.rmtree(get_path("bitmap_labellist_index"), ignore_errors=True)

    data = pa.table(
        {
            "idx": pa.array(range(1000)),
            "bitmap": pa.array(range(1000)),
            "label_list": pa.array([[f"label{i}"] for i in range(1000)]),
        }
    )

    dataset = lance.write_dataset(
        data, get_path("bitmap_labellist_index"), max_rows_per_file=100
    )
    dataset.create_scalar_index("bitmap", "BITMAP")
    dataset.create_scalar_index("label_list", "LABEL_LIST")


def write_dataset_ngram_zonemap_bloomfilter_index():
    shutil.rmtree(get_path("ngram_zonemap_bloomfilter_index"), ignore_errors=True)

    data = pa.table(
        {
            "idx": pa.array(range(1000)),
            "ngram": pa.array([f"word{i}" for i in range(1000)]),
            "zonemap": pa.array(range(1000)),
            "bloomfilter": pa.array(range(1000)),
        }
    )

    dataset = lance.write_dataset(
        data, get_path("ngram_zonemap_bloomfilter_index"), max_rows_per_file=100
    )
    dataset.create_scalar_index("ngram", "NGRAM")
    dataset.create_scalar_index("zonemap", "ZONEMAP")
    dataset.create_scalar_index("bloomfilter", "BLOOMFILTER")


def write_dataset_fts_index():
    shutil.rmtree(get_path("fts_index"), ignore_errors=True)

    data = pa.table(
        {
            "idx": pa.array(range(1000)),
            "text": pa.array(
                [f"document with words {i} and more text" for i in range(1000)]
            ),
        }
    )

    dataset = lance.write_dataset(data, get_path("fts_index"), max_rows_per_file=100)
    dataset.create_scalar_index("text", "INVERTED")


if __name__ == "__main__":
    write_basic_types()
    write_large()
    write_dataset_pq_buffer()
    write_dataset_btree_index()
    write_dataset_bitmap_labellist_index()
    write_dataset_ngram_zonemap_bloomfilter_index()
    write_dataset_json()
    write_dataset_fts_index()
