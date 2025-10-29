# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import pyarrow as pa

from lance import write_dataset, LanceDataset


def test_delta_get_inserted_rows_basic():
    # Create initial dataset (version 1)
    table1 = pa.table({
        "id": pa.array([1, 2, 3], type=pa.int32()),
        "val": pa.array(["a", "b", "c"], type=pa.string()),
    })
    ds1 = write_dataset(table1, "memory://delta_api_test")

    # Append more rows to create version 2
    table2 = pa.table({
        "id": pa.array([4, 5], type=pa.int32()),
        "val": pa.array(["d", "e"], type=pa.string()),
    })
    ds2 = write_dataset(table2, "memory://delta_api_test", mode="append")

    # Build delta compared to v1 and fetch inserted rows
    delta = ds2.delta().compared_against_version(1).build()
    reader = delta.get_inserted_rows()

    # Sum rows from all batches
    total_rows = 0
    for batch in reader:
        total_rows += batch.num_rows

    assert total_rows == 2
