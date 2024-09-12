# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
import os
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from lance.file import LanceFileReader, LanceFileWriter

# This file adds a benchmark to read fixed size binary data using
# the lance v2 format. It can be used to test performance of the
# fixed size binary encoding in lance.
# We can test both random access and full scan access performance by
# setting RANDOM_ACCESS to "indices" or "full", respectively

NUM_ROWS = 1000000
RANDOM_ACCESS = "full"
NUM_ROUNDS = 10
STRING_LENGTH = 8
NUM_INDICES = 10


def generate_test_data():
    np.random.seed(40)
    # Generate random strings of fixed length
    random_strings = np.array(
        [
            "".join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), STRING_LENGTH))
            for _ in range(NUM_ROWS)
        ]
    )

    # Create a string array directly from the list of strings
    fixed_size_string = pa.array(random_strings, type=pa.string())

    # Create the table
    table = pa.table([fixed_size_string], names=["fixed_string"])
    print(table)

    return table


@pytest.fixture(scope="session")
def test_data():
    return generate_test_data()


# generate NUM_INDICES random indices between 0 and NUM_ROWS for scanning
@pytest.fixture(scope="module")
def random_indices():
    random.seed(42)
    random_indices = sorted([random.randint(0, NUM_ROWS) for _ in range(NUM_INDICES)])
    print(random_indices)
    return random_indices


def read_lance_file_random(lance_path, random_indices):
    for batch in (
        LanceFileReader(lance_path).take_rows(indices=random_indices).to_batches()
    ):
        pass


def read_lance_file_full(lance_path):
    for batch in LanceFileReader(lance_path).read_all(batch_size=1000).to_batches():
        pass


@pytest.mark.benchmark(group="read")
def test_lance_read_fixed_size_encoding(
    tmp_path: Path, benchmark, test_data, random_indices
):
    lance_path = str(tmp_path) + "/lance_data1"
    print(lance_path)
    print(test_data)

    with LanceFileWriter(lance_path, test_data.schema) as writer:
        for batch in test_data.to_batches():
            writer.write_batch(batch)

    print("Size: ", os.path.getsize(lance_path))

    if RANDOM_ACCESS == "indices":
        benchmark.pedantic(
            read_lance_file_random, args=(lance_path, random_indices), rounds=NUM_ROUNDS
        )
    elif RANDOM_ACCESS == "full":
        benchmark.pedantic(read_lance_file_full, args=(lance_path,), rounds=NUM_ROUNDS)
