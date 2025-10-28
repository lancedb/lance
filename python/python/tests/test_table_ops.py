# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""
Tests for table operations such as conflict handling and raw commit operations.
"""

import uuid

import lance
import pyarrow as pa
import pytest
from lance.file import LanceFileWriter, stable_version
from lance.fragment import DataFile


def test_can_replace_and_index_diff_columns(tmp_path: str):
    """
    It should be possible to create an index on column X while replacing column Y with
    new data.
    """

    # Create a dataset with columns a and b in separate data files
    table = pa.Table.from_pydict({"a": range(100)})

    ds = lance.write_dataset(table, tmp_path)
    ds.add_columns({"b": "a + 1"})

    # Make some copies of the dataset before the change
    ds2 = lance.dataset(tmp_path)
    ds3 = lance.dataset(tmp_path)

    # Replace column b with new data
    new_file_name = f"{uuid.uuid4()}.lance"
    new_file_path = f"{ds.uri}/data/{new_file_name}"
    with LanceFileWriter(new_file_path) as writer:
        writer.write_batch(pa.table({"b": range(100, 200)}))

    new_data_file = DataFile(
        path=new_file_name,
        fields=[1],
        column_indices=[0],
        file_major_version=int(stable_version().split(".")[0]),
        file_minor_version=int(stable_version().split(".")[1]),
    )

    ds.commit(
        ds.uri,
        lance.LanceOperation.DataReplacement(
            [lance.LanceOperation.DataReplacementGroup(0, new_data_file)]
        ),
        read_version=ds.version,
    )

    # Should be ok to create an index on column a
    ds2.create_scalar_index("a", "BTREE")

    # Creating an index on column b should conflict
    with pytest.raises(Exception, match="Retryable commit conflict for version 3"):
        ds3.create_scalar_index("b", "BTREE")

    # Should be ok to create an index when read version is higher than replacement
    lance.dataset(tmp_path).create_scalar_index("b", "BTREE")
