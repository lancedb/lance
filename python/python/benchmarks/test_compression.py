# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
import os
import random
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pytest
from lance.file import LanceFileReader, LanceFileWriter

NUM_ROWS = 1000000
NUM_INDICES = 100
PREFIX = """
SOME VERY LONG PREFIX THAT IS COMPRESSIBLE PROBABLY BUT WE WILL ADD A
NUMBER TO THE END OF IT TO MAKE IT NOT DICTIONARY COMPRESSIBLE.  THIS
IS A PRETTY IDEAL CASE FOR COMPRESSION
"""


def generate_test_data(compression_scheme: Optional[str]):
    strings = pa.array([f"{PREFIX}-{i}" for i in range(NUM_ROWS)], type=pa.string())

    if compression_scheme is None:
        metadata = None
    else:
        metadata = {"lance-encoding:compression": compression_scheme}

    schema = pa.schema(
        [
            pa.field("strings", pa.string(), metadata=metadata),
        ]
    )
    return pa.table([strings], schema=schema)


# generate NUM_INDICES random indices between 0 and NUM_ROWS for scanning
@pytest.fixture(scope="module")
def random_indices():
    random.seed(42)
    random_indices = sorted([random.randint(0, NUM_ROWS) for _ in range(NUM_INDICES)])
    return random_indices


def drop_page_cache():
    # Note: this will prompt the user for password, not ideal but simple
    os.system('sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"')


@pytest.mark.benchmark
@pytest.mark.parametrize("compression", [None, "zstd"])
def test_random_access(tmp_path: Path, benchmark, random_indices, compression):
    benchmark.group = f"random-access-{compression}"
    test_data = generate_test_data(compression)
    lance_path = str(tmp_path / "random_access.lance")

    with LanceFileWriter(lance_path) as writer:
        writer.write_batch(test_data)

    def read_lance_file_random(lance_path, random_indices):
        drop_page_cache()
        reader = LanceFileReader(lance_path)
        reader.take_rows(random_indices).to_table()

    benchmark(read_lance_file_random, lance_path, random_indices)


@pytest.mark.benchmark
@pytest.mark.parametrize("compression", [None, "zstd"])
def test_full_scan(tmp_path: Path, benchmark, compression):
    benchmark.group = f"full-scan-{compression}"
    test_data = generate_test_data(compression)
    lance_path = str(tmp_path / "full_scan.lance")

    with LanceFileWriter(lance_path) as writer:
        writer.write_batch(test_data)

    def read_lance_file_full(lance_path):
        drop_page_cache()
        reader = LanceFileReader(lance_path)
        reader.read_all().to_table()

    benchmark(read_lance_file_full, lance_path)
