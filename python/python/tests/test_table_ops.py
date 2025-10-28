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


def make_data_file(
    ds: lance.LanceDataset, fields: list[int], data: pa.Table
) -> DataFile:
    new_file_name = f"{uuid.uuid4()}.lance"
    new_file_path = f"{ds.uri}/data/{new_file_name}"
    with LanceFileWriter(new_file_path) as writer:
        writer.write_batch(data)

    return DataFile(
        path=new_file_name,
        fields=fields,
        column_indices=[i for i in range(len(fields))],
        file_major_version=int(stable_version().split(".")[0]),
        file_minor_version=int(stable_version().split(".")[1]),
    )


def test_index_after_replacement(tmp_path: str):
    """
    It should be possible to create an index on column X after a data replacement
    only if that replacement does not modify the column being indexed.
    """

    # Create a dataset with columns a and b in separate data files
    table = pa.Table.from_pydict({"a": range(100)})

    ds = lance.write_dataset(table, tmp_path)
    ds.add_columns({"b": "a + 1"})

    ds2 = lance.dataset(tmp_path)  # copies of the dataset
    ds3 = lance.dataset(tmp_path)  # from before the replacement

    # Replace column b with new data
    new_data_file = make_data_file(ds, [1], pa.table({"b": range(100, 200)}))

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


def test_replacement_after_index(tmp_path: str):
    """
    It should be possible to replace data after an index has been created on the column
    only if the index was not covering the column being replaced.
    """
    table = pa.Table.from_pydict({"a": range(100)})

    ds = lance.write_dataset(table, tmp_path)
    ds.add_columns({"b": "a + 1"})

    ds2 = lance.dataset(tmp_path)  # copies of the dataset
    ds3 = lance.dataset(tmp_path)  # from before the index

    # Create an index on column a
    ds.create_scalar_index("a", "BTREE")

    # Replace column b with new data
    new_data_file = make_data_file(ds, [1], pa.table({"b": range(100, 200)}))

    # Should be ok (index was on column a, new data is on column b)
    ds2.commit(
        ds.uri,
        lance.LanceOperation.DataReplacement(
            [lance.LanceOperation.DataReplacementGroup(0, new_data_file)]
        ),
        read_version=ds2.version,
    )

    new_data_file = make_data_file(ds, [0], pa.table({"a": range(100, 200)}))

    # Should fail since index was on column a and new data is on column a
    with pytest.raises(Exception, match="Retryable commit conflict for version 3"):
        ds3.commit(
            ds.uri,
            lance.LanceOperation.DataReplacement(
                [lance.LanceOperation.DataReplacementGroup(0, new_data_file)]
            ),
            read_version=ds3.version,
        )
