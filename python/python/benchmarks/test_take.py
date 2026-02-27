# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
import os
import platform
import subprocess
import tempfile
import uuid

import lance
import numpy as np
import pyarrow as pa
import pytest
from lance import LanceDataset

DEFAULT_BATCH_SIZE = 1024
ENV_OBJECT_STORAGE_TEST_DATASET_URI_PREFIX = (
    "LANCE_OBJECT_STORAGE_TEST_DATASET_URI_PREFIX"
)


def clear_page_cache():
    """Clear the OS page cache to minimize caching effects between benchmark runs."""
    try:
        system = platform.system()
        if system == "Darwin":  # macOS
            subprocess.run(["purge"], check=True)
        elif system == "Linux":
            subprocess.run(["sync"], check=True)
            subprocess.run(
                ["sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True
            )
        else:
            # other OSes are not supported currently
            pass
    except subprocess.CalledProcessError:
        pass


def get_path_prefixes():
    temp_dir = tempfile.mkdtemp()
    prefixes = ["memory://", f"file://{temp_dir}"]
    object_store_path_prefix = os.getenv(ENV_OBJECT_STORAGE_TEST_DATASET_URI_PREFIX, "")
    if object_store_path_prefix:
        prefixes.append(object_store_path_prefix)
    return prefixes


def get_scheme_from_path(path: str) -> str:
    if "://" in path:
        return path.split("://")[0]
    return "file"


def create_dataset(
    path: str,
    data_storage_version,
    num_batches: int,
    file_size: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    compression: str = None,
) -> LanceDataset:
    metadata = {}
    if compression:
        metadata["lance-encoding:compression"] = compression

    schema = pa.schema(
        [
            pa.field("i", pa.int32(), nullable=False, metadata=metadata),
            pa.field("f", pa.float32(), nullable=False, metadata=metadata),
            pa.field("s", pa.binary(), nullable=False, metadata=metadata),
            pa.field(
                "fsl", pa.list_(pa.float32(), 2), nullable=False, metadata=metadata
            ),
            pa.field("blob", pa.binary(), nullable=False, metadata=metadata),
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
@pytest.mark.parametrize("file_size", [1024 * 1024], ids=["1MB"])
@pytest.mark.parametrize(
    "lance_format_version",
    [("2.0", "V2_0"), ("2.1", "V2_1"), ("2.2", "V2_2")],
    ids=["V2_0", "V2_1", "V2_2"],
)
@pytest.mark.parametrize("num_rows", [100, 1000], ids=["100rows", "1000rows"])
@pytest.mark.parametrize(
    "batch_size", [512, 1024, 2048], ids=["batch512", "batch1024", "batch2048"]
)
@pytest.mark.parametrize("compression", [None, "zstd"], ids=["no_compression", "zstd"])
@pytest.mark.parametrize(
    "path_prefix", get_path_prefixes(), ids=lambda x: get_scheme_from_path(x)
)
def test_dataset_take(
    benchmark,
    tmp_path,
    file_size,
    lance_format_version,
    num_rows,
    batch_size,
    compression,
    path_prefix,
):
    data_storage_version, version_name = lance_format_version

    random_uuid = str(uuid.uuid4())
    path = f"{path_prefix.rstrip('/')}/{random_uuid}.lance/"

    num_batches = 1024
    ds = create_dataset(
        path, data_storage_version, num_batches, file_size, batch_size, compression
    )
    total_rows = ds.count_rows()
    rows = gen_ranges(total_rows, num_rows)

    def dataset_take_rows_bench():
        batch = ds.take(rows)
        assert batch.num_rows == num_rows

    benchmark.group = (
        f"Random Take Dataset({file_size} file size, "
        f"{num_batches} batches, {num_rows} rows per take, "
        f"{batch_size} batch size, {get_scheme_from_path(path_prefix)} scheme)"
    )
    benchmark.pedantic(
        dataset_take_rows_bench, setup=clear_page_cache, rounds=5, iterations=1
    )
