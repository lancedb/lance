# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import pyarrow as pa
from lance import write_dataset


def test_delta_get_inserted_rows():
    # Create initial dataset (version 1)
    table1 = pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int32()),
            "val": pa.array(["a", "b", "c"], type=pa.string()),
        }
    )
    ds = write_dataset(table1, "memory://delta_api_test", enable_stable_row_ids=True)

    # Append more rows to create version 2
    table2 = pa.table(
        {
            "id": pa.array([4, 5], type=pa.int32()),
            "val": pa.array(["d", "e"], type=pa.string()),
        }
    )
    ds.insert(table2)

    # Build delta compared to v1 and fetch inserted rows
    delta = ds.delta().compared_against_version(1).build()
    print(delta.list_transactions())
    reader = delta.get_inserted_rows()

    # Sum rows from all batches
    total_rows = 0
    for batch in reader:
        total_rows += batch.num_rows

    assert total_rows == 2


def test_delta_get_updated_rows():
    # Create initial dataset (version 1)
    table1 = pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int32()),
            "val": pa.array(["a", "b", "c"], type=pa.string()),
        }
    )
    ds = write_dataset(
        table1, "memory://delta_api_test_update", enable_stable_row_ids=True
    )

    # Update an existing row to create version 2
    update_stats = ds.update({"val": "'b_updated'"}, where="id = 2")
    assert update_stats["num_rows_updated"] == 1

    # Build delta compared to v1 and fetch updated rows
    delta = ds.delta().compared_against_version(1).build()

    # Ensure the transaction is an Update (not an Append/Delete)
    txs = delta.list_transactions()
    assert len(txs) == 1
    assert type(txs[0].operation).__name__ == "Update"

    reader = delta.get_updated_rows()

    # Collect updated rows and validate contents
    total_rows = 0
    for batch in reader:
        total_rows += batch.num_rows

    assert total_rows == 1

    # Ensure no inserted rows are present in this diff
    inserted_reader = delta.get_inserted_rows()
    total_inserted = 0
    for batch in inserted_reader:
        total_inserted += batch.num_rows
    assert total_inserted == 0
