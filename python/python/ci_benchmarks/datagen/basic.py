# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

# Creates a dataset containing some basic patterns of synthetic data.
#
# Unlike the image EDA dataset (which has similar patterns) this dataset
# is much smaller and easier to run locally.

import lance
import pyarrow as pa
from lance.log import LOGGER

from ci_benchmarks.datasets import get_dataset_uri

NUM_ROWS = 10_000_000
NUM_BATCHES = 100
ROWS_PER_BATCH = NUM_ROWS // NUM_BATCHES

SCHEMA = pa.schema(
    {
        "row_number": pa.uint64(),
        "row_number_bitmap": pa.uint64(),
        "integers": pa.int64(),
        "small_strings": pa.string(),
    }
)


def _gen_data():
    LOGGER.info("Generating %d rows of data", NUM_ROWS)
    for batch_idx in range(NUM_BATCHES):
        batch = pa.record_batch(
            [
                pa.array(
                    [batch_idx * ROWS_PER_BATCH + i for i in range(ROWS_PER_BATCH)]
                ),
                pa.array(
                    [batch_idx * ROWS_PER_BATCH + i for i in range(ROWS_PER_BATCH)]
                ),
                pa.array(
                    [batch_idx * ROWS_PER_BATCH + i for i in range(ROWS_PER_BATCH)]
                ),
                pa.array([f"payload_{i}" for i in range(ROWS_PER_BATCH)]),
            ],
            names=["row_number", "row_number_bitmap", "integers", "small_strings"],
        )
        yield batch


def _create(dataset_uri: str):
    try:
        ds = lance.dataset(dataset_uri)
        print(ds.count_rows())
        if ds.count_rows() != NUM_ROWS:
            if ds.count_rows() == 0 and ds.schema == SCHEMA:
                ds = lance.write_dataset(
                    _gen_data(),
                    dataset_uri,
                    schema=SCHEMA,
                    mode="append",
                    use_legacy_format=False,
                )
            else:
                raise Exception(
                    "Cannot generate basic dataset because a dataset with the URI "
                    f"{dataset_uri} already exists and doesn't appear to be the "
                    "same dataset"
                )
    except ValueError:
        ds = lance.write_dataset(
            _gen_data(),
            dataset_uri,
            schema=SCHEMA,
            mode="create",
            use_legacy_format=False,
        )
    if ds.list_indices() == []:
        ds.create_scalar_index("row_number", "BTREE")
        ds.create_scalar_index("row_number_bitmap", "BITMAP")


def gen_basic():
    dataset_uri = get_dataset_uri("basic")
    _create(dataset_uri)
