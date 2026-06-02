# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Generate the count_rows benchmark dataset.

Creates a 10-million-row Lance dataset with five int32 columns that all hold the
same values and the same ~1% null mask.  Each column carries a different scalar
index so the benchmark can compare no-index, BITMAP, BTREE, ZONEMAP, and
BLOOMFILTER side-by-side on the same underlying data.

Column layout
-------------
value_none        — no index (full-scan baseline)
value_bitmap      — BITMAP index
value_btree       — BTREE index
value_zonemap     — ZONEMAP index
value_bloomfilter — BLOOMFILTER index

Null pattern: row i is null when i % 100 == 0 (~1% of rows exactly).
"""

import lance
import numpy as np
import pyarrow as pa
from lance.log import LOGGER

from ci_benchmarks.datasets import get_dataset_uri

NUM_ROWS = 10_000_000
BATCH_SIZE = 1_000_000  # 1 M rows per batch → 10 batches total

COLUMNS = [
    "value_none",
    "value_bitmap",
    "value_btree",
    "value_zonemap",
    "value_bloomfilter",
]

SCHEMA = pa.schema([(col, pa.int32()) for col in COLUMNS])


def _gen_data():
    num_batches = NUM_ROWS // BATCH_SIZE
    for batch_idx in range(num_batches):
        offset = batch_idx * BATCH_SIZE
        values = np.arange(offset, offset + BATCH_SIZE, dtype=np.int32)
        # Null mask: True where the value should be null (~1% of rows)
        null_mask = (np.arange(BATCH_SIZE) + offset) % 100 == 0
        col = pa.array(values, type=pa.int32(), mask=null_mask)
        yield pa.record_batch([col] * len(COLUMNS), schema=SCHEMA)


def gen_count_rows() -> lance.LanceDataset:
    dataset_uri = get_dataset_uri("count_rows")

    try:
        ds = lance.dataset(dataset_uri)
        if ds.count_rows() == NUM_ROWS:
            LOGGER.info(
                "count_rows dataset already exists at %s (%d rows)",
                dataset_uri,
                NUM_ROWS,
            )
            return ds
        LOGGER.warning(
            "count_rows dataset at %s has unexpected row count %d; regenerating",
            dataset_uri,
            ds.count_rows(),
        )
    except Exception:
        pass

    LOGGER.info(
        "Writing count_rows dataset (%d rows, %d columns) to %s",
        NUM_ROWS,
        len(COLUMNS),
        dataset_uri,
    )
    ds = lance.write_dataset(
        _gen_data(),
        dataset_uri,
        schema=SCHEMA,
        mode="overwrite",
    )
    LOGGER.info("Dataset written; building scalar indexes …")

    for index_type, column in [
        ("BITMAP", "value_bitmap"),
        ("BTREE", "value_btree"),
        ("ZONEMAP", "value_zonemap"),
        ("BLOOMFILTER", "value_bloomfilter"),
    ]:
        LOGGER.info("  Creating %s index on %s …", index_type, column)
        ds.create_scalar_index(column, index_type)

    LOGGER.info("count_rows dataset ready.")
    return ds
