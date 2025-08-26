# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
import os
import tempfile
import uuid

import lance
import numpy as np
import pyarrow as pa
import pytest
from lance import LanceDataset

BATCH_SIZE = 1024
TEST_LANCE_DATASET_NAME = "test.lance"
ENV_OBJECT_STORAGE_TEST_DATASET_URI_PREFIX = (
    "LANCE_OBJECT_STORAGE_TEST_DATASET_URI_PREFIX"
)


def get_path_prefixes():
    temp_dir = tempfile.mkdtemp()
    prefixes = ["memory://", f"file://{temp_dir}"]
    object_store_path_prefix = os.getenv(ENV_OBJECT_STORAGE_TEST_DATASET_URI_PREFIX, "")
    if object_store_path_prefix:
        prefixes.append(object_store_path_prefix)
    return prefixes


def create_dataset(
    path: str,
    data_storage_version,
    num_batches: int,
    file_size: int,
    batch_size: int = BATCH_SIZE,
) -> LanceDataset:
    schema = pa.schema(
        [
            pa.field("i", pa.int32(), nullable=False),
            pa.field("f", pa.float32(), nullable=False),
            pa.field("s", pa.binary(), nullable=False),
            pa.field("fsl", pa.list_(pa.float32(), 2), nullable=False),
            pa.field("blob", pa.binary(), nullable=False),
        ]
    )

    batches = []
    for i in range(num_batches):
        start = i * batch_size
        stop = (i + 1) * batch_size
        arr_i = pa.array(np.arange(start, stop, dtype=np.int32))
        arr_f = pa.array(np.arange(start, stop, dtype=np.float32))
        arr_s = pa.array([f"blob-{x}".encode() for x in range(start, stop)])
        fsl_values = np.array(
            [
                float(batch_size + (x - batch_size) // 2)
                for x in range(start, stop + batch_size)
            ],
            dtype=np.float32,
        )
        arr_fsl = pa.FixedSizeListArray.from_arrays(pa.array(fsl_values), 2)
        arr_blob = pa.array([f"blob-{x}".encode() for x in range(start, stop)])

        batch = pa.RecordBatch.from_arrays(
            [arr_i, arr_f, arr_s, arr_fsl, arr_blob], schema=schema
        )
        batches.append(batch)

    table = pa.Table.from_batches(batches, schema=schema)

    storage_options = None
    if not path.startswith(("memory://", "file://")):
        storage_options = {
            "access_key_id": os.getenv("ACCESS_KEY_ID", ""),
            "secret_access_key": os.getenv("SECRET_ACCESS_KEY", ""),
            "session_token": os.getenv("SESSION_TOKEN", ""),
            "endpoint": os.getenv("ENDPOINT", ""),
            "virtual_hosted_style_request": os.getenv(
                "VIRTUAL_HOSTED_STYLE_REQUEST", "true"
            ),
        }

    return lance.write_dataset(
        table,
        path,
        max_rows_per_file=file_size,
        max_rows_per_group=batch_size,
        data_storage_version=data_storage_version,
        storage_options=storage_options,
    )


def gen_ranges(total_rows, num_rows):
    return np.random.choice(total_rows, num_rows, replace=False)


@pytest.mark.benchmark()
@pytest.mark.parametrize("file_size", (1024 * 1024, 1024))
@pytest.mark.parametrize("lance_format_version", [("2.0", "V2_0"), ("2.1", "V2_1")])
@pytest.mark.parametrize("num_rows", [1, 10, 100, 1000])
@pytest.mark.parametrize("batch_size", [512, 1024, 2048])
@pytest.mark.parametrize("path_prefix", get_path_prefixes())
def test_dataset_take(
    benchmark,
    tmp_path,
    file_size,
    lance_format_version,
    num_rows,
    batch_size,
    path_prefix,
):
    data_storage_version, version_name = lance_format_version

    random_uuid = str(uuid.uuid4())
    path = f"{path_prefix.rstrip('/')}/{random_uuid}.lance/"

    num_batches = 1024
    ds = create_dataset(path, data_storage_version, num_batches, file_size, batch_size)
    total_rows = ds.count_rows()
    rows = gen_ranges(total_rows, num_rows)

    def dataset_take_rows_bench():
        batch = ds.take(rows)
        assert batch.num_rows == num_rows

    benchmark.group = (
        f"{version_name} Random Take Dataset({file_size} file size, "
        f"{num_batches} batches, {num_rows} rows per take, "
        f"{batch_size} batch size, {path} path)"
    )
    benchmark(dataset_take_rows_bench)
