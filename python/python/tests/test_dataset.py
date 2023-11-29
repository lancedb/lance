#  Copyright (c) 2023. Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import base64
import contextlib
import os
import pickle
import platform
import re
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest import mock

import lance
import lance.fragment
import pandas as pd
import pandas.testing as tm
import polars as pl
import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
import pytest
from helper import ProgressForTest
from lance.commit import CommitConflictError

# Various valid inputs for write_dataset
input_schema = pa.schema([pa.field("a", pa.float64()), pa.field("b", pa.int64())])
input_data = [
    # (schema, data)
    (None, pa.table({"a": [1.0, 2.0], "b": [20, 30]})),
    (None, pa.record_batch([[1.0, 2.0], [20, 30]], names=["a", "b"])),
    (None, pd.DataFrame({"a": [1.0, 2.0], "b": [20, 30]})),
    (None, pl.DataFrame({"a": [1.0, 2.0], "b": [20, 30]})),
    (
        input_schema,
        [pa.record_batch([pa.array([1.0, 2.0]), pa.array([20, 30])], names=["a", "b"])],
    ),
    # Can provide an iterator with a schema that is different but cast-able
    (
        input_schema,
        iter(
            pa.table(
                {"a": [1.0, 2.0], "b": pa.array([20, 30], pa.int32())}
            ).to_batches()
        ),
    ),
]


@pytest.mark.parametrize("schema,data", input_data, ids=type)
def test_input_data(tmp_path: Path, schema, data):
    base_dir = tmp_path / "test"
    dataset = lance.write_dataset(data, base_dir, schema=schema)
    assert dataset.to_table() == input_data[0][1]


def test_dataset_overwrite(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    table2 = pa.Table.from_pylist([{"s": "one"}, {"s": "two"}])
    lance.write_dataset(table2, base_dir, mode="overwrite")

    dataset = lance.dataset(base_dir)
    assert dataset.to_table() == table2

    ds_v1 = lance.dataset(base_dir, version=1)
    assert ds_v1.to_table() == table1


def test_dataset_append(tmp_path: Path):
    table = pa.Table.from_pydict({"colA": [1, 2, 3], "colB": [4, 5, 6]})
    base_dir = tmp_path / "test"

    # verify append works even if no dataset existed at the uri
    lance.write_dataset(table, base_dir, mode="append")
    dataset = lance.dataset(base_dir)
    assert dataset.to_table() == table

    # verify appending batches with a different schema doesn't work
    table2 = pa.Table.from_pydict({"COLUMN-C": [1, 2, 3], "colB": [4, 5, 6]})
    with pytest.raises(OSError):
        lance.write_dataset(table2, base_dir, mode="append")


def test_dataset_from_record_batch_iterable(tmp_path: Path):
    base_dir = tmp_path / "test"

    test_pylist = [{"colA": "Alice", "colB": 20}, {"colA": "Blob", "colB": 30}]

    # split into two batches
    batches = [
        pa.RecordBatch.from_pylist([test_pylist[0]]),
        pa.RecordBatch.from_pylist([test_pylist[1]]),
    ]

    # define schema
    schema = pa.schema(
        [
            pa.field("colA", pa.string()),
            pa.field("colB", pa.int64()),
        ]
    )

    # write dataset with iterator
    lance.write_dataset(iter(batches), base_dir, schema)
    dataset = lance.dataset(base_dir)

    # After combined into one batch, make sure it is the same as original pylist
    assert list(dataset.to_batches())[0].to_pylist() == test_pylist

    # write dataset with list
    lance.write_dataset(batches, base_dir, schema, mode="overwrite")

    # After combined into one batch, make sure it is the same as original pylist
    assert list(dataset.to_batches())[0].to_pylist() == test_pylist


def test_versions(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    assert len(lance.dataset(base_dir).versions()) == 1

    table2 = pa.Table.from_pylist([{"s": "one"}, {"s": "two"}])
    time.sleep(1)
    lance.write_dataset(table2, base_dir, mode="overwrite")

    assert len(lance.dataset(base_dir).versions()) == 2

    v1, v2 = lance.dataset(base_dir).versions()
    assert v1["version"] == 1
    assert v2["version"] == 2
    assert isinstance(v1["timestamp"], datetime)
    assert isinstance(v2["timestamp"], datetime)
    assert v1["timestamp"] < v2["timestamp"]
    assert isinstance(v1["metadata"], dict)
    assert isinstance(v2["metadata"], dict)


def test_version_id(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    original_ds = lance.write_dataset(table1, base_dir)

    assert original_ds.version == 1
    assert original_ds.latest_version == 1

    table2 = pa.Table.from_pylist([{"s": "one"}, {"s": "two"}])
    time.sleep(1)
    updated_ds = lance.write_dataset(table2, base_dir, mode="overwrite")

    assert original_ds.version == 1
    assert original_ds.latest_version == 2

    assert updated_ds.version == 2
    assert updated_ds.latest_version == 2


def test_asof_checkout(tmp_path: Path):
    table = pa.Table.from_pydict({"colA": [1, 2, 3], "colB": [4, 5, 6]})
    base_dir = tmp_path / "test"

    lance.write_dataset(table, base_dir)
    assert len(lance.dataset(base_dir).versions()) == 1
    time.sleep(0.1)
    ts_1 = datetime.now()
    time.sleep(0.1)

    lance.write_dataset(table, base_dir, mode="append")
    assert len(lance.dataset(base_dir).versions()) == 2
    time.sleep(0.1)
    ts_2 = datetime.now()
    time.sleep(0.1)

    lance.write_dataset(table, base_dir, mode="append")
    assert len(lance.dataset(base_dir).versions()) == 3
    time.sleep(0.1)
    ts_3 = datetime.now()

    # check that only the first batch is present
    ds = lance.dataset(base_dir, asof=ts_1)
    assert ds.version == 1
    assert len(ds.to_table()) == 3

    # check that the first and second batch are present
    ds = lance.dataset(base_dir, asof=ts_2)
    assert ds.version == 2
    assert len(ds.to_table()) == 6

    # check that all batches are present
    ds = lance.dataset(base_dir, asof=ts_3)
    assert ds.version == 3
    assert len(ds.to_table()) == 9


def test_sample(tmp_path: Path):
    table1 = pa.Table.from_pydict({"x": [0, 10, 20, 30, 40, 50], "y": range(6)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    dataset = lance.dataset(base_dir)
    sampled = dataset.sample(3)

    assert sampled.num_rows == 3
    assert sampled.num_columns == 2

    sampled = dataset.sample(4, columns=["x"])

    assert sampled.num_rows == 4
    assert sampled.num_columns == 1

    for row in sampled.column(0).chunk(0):
        assert row.as_py() % 10 == 0


def test_take(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    dataset = lance.dataset(base_dir)
    table2 = dataset.take([0, 1])

    assert isinstance(table2, pa.Table)
    assert table2 == table1


def test_take_with_columns(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    dataset = lance.dataset(base_dir)
    table2 = dataset.take([0], columns=["b"])

    assert table2 == pa.Table.from_pylist([{"b": 2}])


def test_filter(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    actual_tab = dataset.to_table(columns=["a"], filter=(pa.compute.field("b") > 50))
    assert actual_tab == pa.Table.from_pydict({"a": range(51, 100)})


def test_limit_offset(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)
    dataset = lance.dataset(base_dir)

    # test just limit
    assert dataset.to_table(limit=10) == table.slice(0, 10)

    # test just offset
    assert dataset.to_table(offset=10) == table.slice(10, 100)

    # test both
    assert dataset.to_table(offset=10, limit=10) == table.slice(10, 10)


def test_relative_paths(tmp_path: Path):
    # relative paths get coerced to the full absolute path
    current_dir = os.getcwd()
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    rel_uri = "test.lance"
    try:
        os.chdir(tmp_path)
        lance.write_dataset(table, rel_uri)

        # relative path works in the current dir
        ds = lance.dataset(rel_uri)
        assert ds.to_table() == table
    finally:
        os.chdir(current_dir)

    # relative path doesn't work in the context of a different dir
    with pytest.raises(ValueError):
        ds = lance.dataset(rel_uri)

    # relative path gets resolved to the right absolute path
    ds = lance.dataset(tmp_path / rel_uri)
    assert ds.to_table() == table


@pytest.mark.skipif(
    (platform.system() == "Windows"),
    reason="mocking user's home folder it not working",
)
def test_tilde_paths(tmp_path: Path):
    # tilde paths get resolved to the right absolute path
    tilde_uri = "~/test.lance"
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})

    with mock.patch.dict(
        os.environ, {"HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}
    ):
        # NOTE: the resolution logic is a bit finicky
        # link 1 - https://docs.rs/dirs/4.0.0/dirs/fn.home_dir.html
        # link 2 - https://docs.python.org/3/library/os.path.html#os.path.expanduser
        expected_abs_path = os.path.expanduser(tilde_uri)
        assert expected_abs_path == os.fspath(tmp_path / "test.lance")

        lance.write_dataset(table, tilde_uri)
        # works in the current context
        ds = lance.dataset(tilde_uri)
        assert ds.to_table() == table

    # works with the resolved absolute path
    ds = lance.dataset(expected_abs_path)
    assert ds.to_table() == table


def test_to_batches(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    batches = dataset.to_batches(batch_readahead=20)
    assert pa.Table.from_batches(batches) == table

    unordered_batches = dataset.to_batches(batch_readahead=20, scan_in_order=False)
    sorted_batches = pa.Table.from_batches(unordered_batches).sort_by("a")
    assert sorted_batches == table


def test_pickle(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    pickled = pickle.dumps(dataset)
    unpickled = pickle.loads(pickled)
    assert dataset.to_table() == unpickled.to_table()


def test_polar_scan(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    polars_df = pl.scan_pyarrow_dataset(dataset)
    df = dataset.to_table().to_pandas()
    tm.assert_frame_equal(polars_df.collect().to_pandas(), df)


def test_count_fragments(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    count_fragments = dataset.stats.dataset_stats()["num_fragments"]
    assert count_fragments == 1


def test_get_fragments(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    fragment = dataset.get_fragments()[0]
    assert fragment.count_rows() == 100
    assert fragment.physical_rows == 100
    assert fragment.num_deletions == 0

    assert fragment.metadata.id == 0

    head = fragment.head(10)
    tm.assert_frame_equal(head.to_pandas(), table.to_pandas()[0:10])

    assert fragment.to_table() == table

    taken = fragment.take([18, 20, 33, 53])
    assert taken == pa.Table.from_pydict({"a": [18, 20, 33, 53], "b": [18, 20, 33, 53]})


def test_pickle_fragment(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    fragment = dataset.get_fragments()[0]
    pickled = pickle.dumps(fragment)
    unpickled = pickle.loads(pickled)

    assert fragment.to_table() == unpickled.to_table()


def test_add_columns(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    fragments = dataset.get_fragments()

    fragment = fragments[0]

    def adder(batch: pa.RecordBatch) -> pa.RecordBatch:
        c_array = pa.compute.multiply(batch.column(0), 2)
        return pa.RecordBatch.from_arrays([c_array], names=["c"])

    fragment_metadata = fragment.add_columns(adder, columns=["a"])
    schema = dataset.schema.append(pa.field("c", pa.int64()))

    operation = lance.LanceOperation.Overwrite(schema, [fragment_metadata])
    dataset = lance.LanceDataset.commit(base_dir, operation)
    assert dataset.schema == schema

    tbl = dataset.to_table()
    assert tbl == pa.Table.from_pydict(
        {"a": range(100), "b": range(100), "c": pa.array(range(0, 200, 2), pa.int64())}
    )


def test_cleanup_old_versions(tmp_path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)
    moment = datetime.now()
    lance.write_dataset(table, base_dir, mode="overwrite")

    dataset = lance.dataset(base_dir)

    # These calls do nothing, but make sure we can call the method ok

    # Ok, defaults to two weeks ago
    stats = dataset.cleanup_old_versions()
    assert stats.bytes_removed == 0
    assert stats.old_versions == 0

    # Ok, can accept timedelta
    dataset.cleanup_old_versions(older_than=timedelta(days=14))

    print(tmp_path)
    for root, dirnames, filenames in os.walk(tmp_path):
        for filename in filenames:
            print(root + "/" + filename)

    # Now this call will actually delete the old version
    stats = dataset.cleanup_old_versions(older_than=(datetime.now() - moment))
    print(stats)
    assert stats.bytes_removed > 0
    assert stats.old_versions == 1


def test_create_from_commit(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    fragment = lance.fragment.LanceFragment.create(base_dir, table)

    operation = lance.LanceOperation.Overwrite(table.schema, [fragment])
    dataset = lance.LanceDataset.commit(base_dir, operation)
    tbl = dataset.to_table()
    assert tbl == table


def test_append_with_commit(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"

    lance.write_dataset(table, base_dir)

    fragment = lance.fragment.LanceFragment.create(base_dir, table)
    append = lance.LanceOperation.Append([fragment])

    with pytest.raises(OSError):
        # Must specify read version
        dataset = lance.LanceDataset.commit(base_dir, append)

    dataset = lance.LanceDataset.commit(base_dir, append, read_version=1)

    tbl = dataset.to_table()

    expected = pa.Table.from_pydict(
        {
            "a": list(range(100)) + list(range(100)),
            "b": list(range(100)) + list(range(100)),
        }
    )
    assert tbl == expected


def test_delete_with_commit(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"

    lance.write_dataset(table, base_dir)
    lance.write_dataset(table, base_dir, mode="append")

    half_table = pa.Table.from_pydict({"a": range(50), "b": range(50)})

    fragments = lance.dataset(base_dir).get_fragments()

    updated_fragment = fragments[0].delete("a >= 50")
    delete = lance.LanceOperation.Delete(
        [updated_fragment], [fragments[1].fragment_id], "hello"
    )

    dataset = lance.LanceDataset.commit(base_dir, delete, read_version=2)

    tbl = dataset.to_table()
    assert tbl == half_table


def test_count_deleted_rows(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"

    dataset = lance.write_dataset(table, base_dir)
    dataset.delete("a >= 50")

    assert dataset.stats.dataset_stats()["num_deleted_rows"] == 50


def test_restore_with_commit(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"

    lance.write_dataset(table, base_dir)
    lance.write_dataset(table, base_dir, mode="append")

    restore = lance.LanceOperation.Restore(1)
    dataset = lance.LanceDataset.commit(base_dir, restore)

    tbl = dataset.to_table()
    assert tbl == table


def test_merge_with_commit(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"

    lance.write_dataset(table, base_dir)

    fragment = lance.dataset(base_dir).get_fragments()[0]
    merged = fragment.add_columns(
        lambda _: pa.RecordBatch.from_pydict({"c": range(100)})
    )

    expected = pa.Table.from_pydict({"a": range(100), "b": range(100), "c": range(100)})

    merge = lance.LanceOperation.Merge([merged], expected.schema)
    dataset = lance.LanceDataset.commit(base_dir, merge, read_version=1)

    tbl = dataset.to_table()
    assert tbl == expected


def test_data_files(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    fragment = lance.fragment.LanceFragment.create(base_dir, table)

    data_files = fragment.data_files()
    assert len(data_files) == 1
    # it is a valid uuid
    uuid.UUID(os.path.splitext(data_files[0].path())[0])

    assert fragment.deletion_file() is None


def test_deletion_file(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    dataset = lance.write_dataset(table, base_dir)
    fragment = dataset.get_fragment(0)

    assert fragment.deletion_file() is None

    new_fragment = fragment.delete("a < 10")

    # Old fragment unchanged
    assert fragment.deletion_file() is None

    # New fragment has deletion file
    assert new_fragment.deletion_file() is not None
    assert re.match("_deletions/0-1-[0-9]{1,32}.arrow", new_fragment.deletion_file())
    print(type(new_fragment), new_fragment)
    operation = lance.LanceOperation.Overwrite(table.schema, [new_fragment])
    dataset = lance.LanceDataset.commit(base_dir, operation)
    assert dataset.count_rows() == 90


def test_commit_fragments_via_scanner(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    parquet_dir = tmp_path / "parquet.parquet"
    pq.write_to_dataset(table, parquet_dir)

    base_dir = tmp_path / "test"
    scanner = pa.dataset.dataset(parquet_dir).scanner()
    fragment_metadata = lance.fragment.LanceFragment.create(base_dir, scanner)

    # Pickle-able
    pickled = pickle.dumps(fragment_metadata)
    unpickled = pickle.loads(pickled)
    assert fragment_metadata == unpickled

    operation = lance.LanceOperation.Overwrite(table.schema, [fragment_metadata])
    dataset = lance.LanceDataset.commit(base_dir, operation)
    assert dataset.schema == table.schema

    tbl = dataset.to_table()
    assert tbl == table


def test_load_scanner_from_fragments(tmp_path: Path):
    tab = pa.table({"a": range(100), "b": range(100)})
    for _ in range(3):
        lance.write_dataset(tab, tmp_path / "dataset", mode="append")

    dataset = lance.dataset(tmp_path / "dataset")
    fragments = list(dataset.get_fragments())
    assert len(fragments) == 3

    scanner = dataset.scanner(fragments=fragments[0:2])
    assert scanner.to_table().num_rows == 2 * 100

    # Accepts an iterator
    scanner = dataset.scanner(fragments=iter(fragments[0:2]), scan_in_order=False)
    assert scanner.to_table().num_rows == 2 * 100


def test_merge_data(tmp_path: Path):
    tab = pa.table({"a": range(100), "b": range(100)})
    lance.write_dataset(tab, tmp_path / "dataset", mode="append")

    dataset = lance.dataset(tmp_path / "dataset")

    # rejects partial data for non-nullable types
    new_tab = pa.table({"a": range(40), "c": range(40)})
    # TODO: this should be ValueError
    with pytest.raises(
        OSError, match=".+Lance does not yet support nulls for type Int64."
    ):
        dataset.merge(new_tab, "a")

    # accepts a full merge
    new_tab = pa.table({"a": range(100), "c": range(100)})
    dataset.merge(new_tab, "a")
    assert dataset.version == 2
    assert dataset.to_table() == pa.table(
        {"a": range(100), "b": range(100), "c": range(100)}
    )

    # accepts a partial for string
    new_tab = pa.table({"a2": range(5), "d": ["a", "b", "c", "d", "e"]})
    dataset.merge(new_tab, left_on="a", right_on="a2")
    assert dataset.version == 3
    expected = pa.table(
        {
            "a": range(100),
            "b": range(100),
            "c": range(100),
            "d": ["a", "b", "c", "d", "e"] + [None] * 95,
        }
    )
    assert dataset.to_table() == expected
    # Verify we can also load from fresh instance
    dataset = lance.dataset(tmp_path / "dataset")
    assert dataset.to_table() == expected


def test_delete_data(tmp_path: Path):
    # We pass schema explicitly since we want b to be non-nullable.
    schema = pa.schema(
        [
            pa.field("a", pa.int64()),
            pa.field("b", pa.int64(), nullable=False),
        ]
    )
    tab = pa.table({"a": range(100), "b": range(100)}, schema=schema)
    lance.write_dataset(tab, tmp_path / "dataset", mode="append")

    dataset = lance.dataset(tmp_path / "dataset")

    dataset.delete("a < 10")
    dataset.delete("b in (98, 99)")
    assert dataset.version == 3
    assert dataset.to_table() == pa.table(
        {"a": range(10, 98), "b": range(10, 98)}, schema=schema
    )

    dataset.delete(pa_ds.field("a") < 20)
    assert dataset.version == 4
    assert dataset.to_table() == pa.table(
        {"a": range(20, 98), "b": range(20, 98)}, schema=schema
    )

    # These sorts of filters were previously used as a work-around for not
    # supporting "WHERE true". But now we need to make sure they still work
    # even with presence of expression simplification passes.
    old_version = dataset.version
    dataset.delete("b IS NOT NULL")
    assert dataset.count_rows() == 0

    dataset = lance.dataset(tmp_path / "dataset", version=old_version)
    dataset.restore()
    assert dataset.count_rows() > 0
    dataset.delete("true")
    assert dataset.count_rows() == 0


def test_update_dataset(tmp_path: Path):
    nrows = 100
    vecs = pa.FixedSizeListArray.from_arrays(
        pa.array(range(2 * nrows), type=pa.float32()), 2
    )
    tab = pa.table({"a": range(nrows), "b": range(nrows), "vec": vecs})
    lance.write_dataset(tab, tmp_path / "dataset", mode="append")

    dataset = lance.dataset(tmp_path / "dataset")

    dataset.update(dict(b="b + 1"))
    expected = pa.table({"a": range(100), "b": range(1, 101)})
    assert dataset.to_table(columns=["a", "b"]) == expected

    dataset.update(dict(a="a * 2"), where="a < 50")
    expected = pa.table(
        {"a": [x * 2 if x < 50 else x for x in range(100)], "b": range(1, 101)}
    )
    assert dataset.to_table(columns=["a", "b"]).sort_by("b") == expected

    dataset.update(dict(vec="[42.0, 43.0]"))
    expected = pa.table(
        {
            "b": range(1, 101),
            "vec": pa.array(
                [[42.0, 43.0] for _ in range(100)], pa.list_(pa.float32(), 2)
            ),
        }
    )
    assert dataset.to_table(columns=["b", "vec"]).sort_by("b") == expected


def test_update_dataset_all_types(tmp_path: Path):
    table = pa.table(
        {
            "int32": pa.array([1], pa.int32()),
            "int64": pa.array([1], pa.int64()),
            "uint32": pa.array([1], pa.uint32()),
            "string": pa.array(["foo"], pa.string()),
            "large_string": pa.array(["foo"], pa.large_string()),
            "float32": pa.array([1.0], pa.float32()),
            "float64": pa.array([1.0], pa.float64()),
            "bool": pa.array([True], pa.bool_()),
            "date32": pa.array([date(2021, 1, 1)], pa.date32()),
            "timestamp_ns": pa.array([datetime(2021, 1, 1)], pa.timestamp("ns")),
            "timestamp_ms": pa.array([datetime(2021, 1, 1)], pa.timestamp("ms")),
            "vec_f32": pa.array([[1.0, 2.0]], pa.list_(pa.float32(), 2)),
            "vec_f64": pa.array([[1.0, 2.0]], pa.list_(pa.float64(), 2)),
        }
    )

    dataset = lance.write_dataset(table, tmp_path)

    # One update with all matching types
    dataset.update(
        dict(
            int32="2",
            int64="2",
            uint32="2",
            string="'bar'",
            large_string="'bar'",
            float32="2.0",
            float64="2.0",
            bool="false",
            date32="DATE '2021-01-02'",
            timestamp_ns='TIMESTAMP "2021-01-02 00:00:00"',
            timestamp_ms='TIMESTAMP "2021-01-02 00:00:00"',
            vec_f32="[3.0, 4.0]",
            vec_f64="[3.0, 4.0]",
        )
    )
    expected = pa.table(
        {
            "int32": pa.array([2], pa.int32()),
            "int64": pa.array([2], pa.int64()),
            "uint32": pa.array([2], pa.uint32()),
            "string": pa.array(["bar"], pa.string()),
            "large_string": pa.array(["bar"], pa.large_string()),
            "float32": pa.array([2.0], pa.float32()),
            "float64": pa.array([2.0], pa.float64()),
            "bool": pa.array([False], pa.bool_()),
            "date32": pa.array([date(2021, 1, 2)], pa.date32()),
            "timestamp_ns": pa.array([datetime(2021, 1, 2)], pa.timestamp("ns")),
            "timestamp_ms": pa.array([datetime(2021, 1, 2)], pa.timestamp("ms")),
            "vec_f32": pa.array([[3.0, 4.0]], pa.list_(pa.float32(), 2)),
            "vec_f64": pa.array([[3.0, 4.0]], pa.list_(pa.float64(), 2)),
        }
    )
    assert dataset.to_table() == expected


def test_create_update_empty_dataset(tmp_path: Path, provide_pandas: bool):
    base_dir = tmp_path / "dataset"

    fields = [
        ("a", pa.string()),
        ("b", pa.int64()),
        ("c", pa.float64()),
    ]
    df = pd.DataFrame(columns=[name for name, _ in fields])
    tab = pa.Table.from_pandas(df, schema=pa.schema(fields))
    dataset = lance.write_dataset(tab, base_dir)

    assert dataset.count_rows() == 0
    expected_schema = pa.schema(
        [
            pa.field("a", pa.string()),
            pa.field("b", pa.int64()),
            pa.field("c", pa.float64()),
        ]
    )
    assert dataset.schema == expected_schema
    assert dataset.to_table() == pa.table(
        {"a": [], "b": [], "c": []}, schema=expected_schema
    )

    tab2 = pa.table({"a": ["foo"], "b": [1], "c": [2.0]})
    dataset = lance.write_dataset(tab2, base_dir, mode="append")

    assert dataset.count_rows() == 1
    assert dataset.to_table() == pa.table(
        {"a": ["foo"], "b": [1], "c": [2.0]}, schema=expected_schema
    )


def test_scan_with_batch_size(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    df = pd.DataFrame({"a": range(10000), "b": range(10000)})
    dataset = lance.write_dataset(df, base_dir)

    batches = dataset.scanner(batch_size=16, scan_in_order=True).to_batches()

    for idx, batch in enumerate(batches):
        assert batch.num_rows == 16
        df = batch.to_pandas()
        assert df["a"].iloc[0] == idx * 16


def test_scan_prefilter(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    vecs = pa.array(
        [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], type=pa.list_(pa.float32(), 2)
    )
    df = pa.Table.from_pydict(
        {
            "index": [1, 2, 3, 4, 5, 6],
            "type": ["a", "a", "a", "b", "b", "b"],
            "vecs": vecs,
        }
    )
    dataset = lance.write_dataset(df, base_dir)
    query = pa.array([1, 1], pa.float32())
    args = {
        "columns": ["index"],
        "filter": "type = 'b'",
        "nearest": {"column": "vecs", "q": pa.array(query), "k": 2, "use_index": False},
    }

    # With post-filter no results are returned because all
    # the closest results don't match the filter
    assert dataset.scanner(**args).to_table().num_rows == 0

    args["prefilter"] = True
    table = dataset.scanner(**args).to_table()

    expected = pa.Table.from_pydict({"index": [4, 5]})

    assert table.column("index") == expected.column("index")

    table = dataset.to_table(**args)
    assert table.column("index") == expected.column("index")

    table = pa.Table.from_batches(dataset.to_batches(**args))
    assert table.column("index") == expected.column("index")


def test_scan_count_rows(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    df = pd.DataFrame({"a": range(42), "b": range(42)})
    dataset = lance.write_dataset(df, base_dir)

    assert dataset.scanner().count_rows() == 42
    assert dataset.count_rows(filter="a < 10") == 10
    assert dataset.count_rows(filter=pa_ds.field("a") < 20) == 20


def test_scanner_schemas(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    df = pd.DataFrame({"a": range(50), "s": [f"s-{i}" for i in range(50)]})

    dataset = lance.write_dataset(df, base_dir)

    scanner = dataset.scanner(columns=["a"])
    assert scanner.dataset_schema == dataset.schema
    assert scanner.projected_schema == pa.schema([pa.field("a", pa.int64())])


def test_custom_commit_lock(tmp_path: Path):
    called_lock = False
    called_release = False

    @contextlib.contextmanager
    def commit_lock(version: int):
        nonlocal called_lock
        nonlocal called_release
        called_lock = True
        assert version == 1
        yield
        called_release = True

    lance.write_dataset(
        pa.table({"a": range(100)}), tmp_path / "test1", commit_lock=commit_lock
    )
    assert called_lock
    assert called_release

    @contextlib.contextmanager
    def commit_lock(_version: int):
        try:
            yield
        finally:
            raise Exception("hello world!")

    with pytest.raises(Exception, match="hello world!"):
        lance.write_dataset(
            pa.table({"a": range(100)}), tmp_path / "test2", commit_lock=commit_lock
        )

    @contextlib.contextmanager
    def commit_lock(_version: int):
        raise CommitConflictError()

    with pytest.raises(Exception, match="CommitConflictError"):
        lance.write_dataset(
            pa.table({"a": range(100)}), tmp_path / "test3", commit_lock=commit_lock
        )

    dataset = lance.dataset(tmp_path / "test1", commit_lock=commit_lock)
    with pytest.raises(Exception, match="CommitConflictError"):
        dataset.delete("a < 10")


def test_dataset_restore(tmp_path: Path):
    data = pa.table({"a": range(100)})
    dataset = lance.write_dataset(data, tmp_path)
    assert dataset.version == 1
    assert dataset.count_rows() == 100

    dataset.delete("a >= 50")
    assert dataset.count_rows() == 50
    assert dataset.version == 2

    dataset = lance.dataset(tmp_path, version=1)
    assert dataset.version == 1

    dataset.restore()
    assert dataset.version == 3
    assert dataset.count_rows() == 100


def test_roundtrip_reader(tmp_path: Path):
    # Can roundtrip a reader
    data = pa.table({"a": range(100)})
    dataset = lance.write_dataset(data, tmp_path / "test1")
    reader = dataset.to_batches()
    lance.write_dataset(reader, tmp_path / "test2", schema=dataset.schema)

    # Can roundtrip a whole dataset
    lance.write_dataset(dataset, tmp_path / "test3")


def test_metadata(tmp_path: Path):
    data = pa.table({"a": range(100)})
    data = data.replace_schema_metadata({"foo": pickle.dumps("foo")})
    with pytest.raises(ValueError):
        lance.write_dataset(data, tmp_path)

    data = data.replace_schema_metadata({"foo": base64.b64encode(pickle.dumps("foo"))})
    lance.write_dataset(data, tmp_path)


def test_scan_with_row_ids(tmp_path: Path):
    """Test expose physical row ids in the scanner."""
    data = pa.table({"a": range(1000)})
    ds = lance.write_dataset(data, tmp_path, max_rows_per_file=250)

    tbl = ds.scanner(filter="a % 10 == 0 AND a < 500", with_row_id=True).to_table()
    assert "_rowid" in tbl.column_names
    row_ids = tbl["_rowid"].to_pylist()
    assert row_ids == list(range(0, 250, 10)) + list(range(2**32, 2**32 + 250, 10))

    tbl2 = ds._take_rows(row_ids)
    assert tbl2["a"] == tbl["a"]


def test_count_index_rows(tmp_path: Path):
    schema = pa.schema([pa.field("a", pa.list_(pa.float32(), 32), False)])
    table = pa.Table.from_pydict(
        {"a": [[float(i) for i in range(32)] for _ in range(512)]}, schema=schema
    )

    base_dir = tmp_path / "test"

    dataset = lance.write_dataset(table, base_dir)

    # assert we return None for index name that doesn't exist
    index_name = "a_idx"
    with pytest.raises(KeyError):
        dataset.stats.index_stats(index_name)["num_unindexed_rows"]
    with pytest.raises(KeyError):
        dataset.stats.index_stats(index_name)["num_indexed_rows"]

    # create index and assert no rows are uncounted
    dataset.create_index(
        "a", "IVF_PQ", name=index_name, num_partitions=2, num_sub_vectors=1
    )
    assert dataset.stats.index_stats(index_name)["num_unindexed_rows"] == 0
    assert dataset.stats.index_stats(index_name)["num_indexed_rows"] == 512

    # append some data
    new_table = pa.Table.from_pydict(
        {"a": [[float(i) for i in range(32)] for _ in range(512)]}, schema=schema
    )
    dataset = lance.write_dataset(new_table, base_dir, mode="append")

    # assert rows added since index was created are uncounted
    assert dataset.stats.index_stats(index_name)["num_unindexed_rows"] == 512
    assert dataset.stats.index_stats(index_name)["num_indexed_rows"] == 512


def test_dataset_progress(tmp_path: Path):
    data = pa.table({"a": range(10)})
    progress = ProgressForTest()
    ds = lance.write_dataset(data, tmp_path, max_rows_per_file=5, progress=progress)
    assert len(ds.get_fragments()) == 2
    assert progress.begin_called == 2
    assert progress.complete_called == 2
