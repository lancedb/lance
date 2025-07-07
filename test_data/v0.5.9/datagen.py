# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import lance
import pyarrow as pa

# To generate the test file, we should be running this version of lance
assert lance.__version__ == "0.5.9"


def create_empty_dataset():
    """Create an empty dataset with no fragments"""
    schema = pa.schema({"id": pa.int64()})
    table = pa.Table.from_pylist([], schema=schema)
    ds = lance.write_dataset(table, "no_fragments")
    assert len(ds.get_fragments()) == 0, "Expected no fragments in the dataset"


def create_dataset_with_fragments():
    """Create a dataset with some fragments"""
    for i in range(3):
        table = pa.table({"id": [i]})
        # max_rows_per_file didn't work back then.
        ds = lance.write_dataset(table, "dataset_with_fragments", mode="append")
    assert len(ds.get_fragments()) == 3, "Expected 3 fragments in the dataset"


create_empty_dataset()
create_dataset_with_fragments()
