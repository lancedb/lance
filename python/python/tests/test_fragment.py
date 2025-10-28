# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import json
import multiprocessing
import pickle
import uuid
from pathlib import Path

import lance
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from helper import ProgressForTest
from lance import (
    FragmentMetadata,
    LanceDataset,
    LanceFragment,
    LanceOperation,
    write_dataset,
)
from lance.debug import format_fragment
from lance.file import LanceFileWriter
from lance.fragment import write_fragments
from lance.progress import FileSystemFragmentWriteProgress


def test_write_fragment(tmp_path: Path):
    with pytest.raises(OSError):
        LanceFragment.create(tmp_path, pd.DataFrame([]))
    with pytest.raises(OSError):
        LanceFragment.create(tmp_path, pd.DataFrame([{}]))

    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    frag = LanceFragment.create(tmp_path, df)

    assert len(frag.files) == 1
    assert frag.files[0].fields == [0]
    assert frag.physical_rows == 5
    assert frag.row_id_meta is None
    assert frag.deletion_file is None

    meta = frag.to_json()
    assert "id" in meta
    assert "files" in meta
    assert meta["files"][0]["fields"] == [0]


def test_write_fragment_two_phases(tmp_path: Path):
    num_files = 10
    json_array = []
    for i in range(num_files):
        df = pd.DataFrame({"a": [i * 10]})
        frag = LanceFragment.create(tmp_path, df)
        json_array.append(json.dumps(frag.to_json()))

    fragments = [FragmentMetadata.from_json(j) for j in json_array]

    schema = pa.schema([pa.field("a", pa.int64())])

    operation = LanceOperation.Overwrite(schema, fragments)
    dataset = LanceDataset.commit(tmp_path, operation)

    df = dataset.to_table().to_pandas()
    pd.testing.assert_frame_equal(
        df, pd.DataFrame({"a": [i * 10 for i in range(num_files)]})
    )


def test_write_legacy_fragment(tmp_path: Path):
    tab = pa.table({"a": range(1024)})
    frag = LanceFragment.create(tmp_path, tab, data_storage_version="legacy")
    assert "file_major_version=2" not in str(frag)

    tab = pa.table({"a": range(1024)})
    frag = LanceFragment.create(tmp_path, tab, data_storage_version="stable")
    assert "file_major_version=2" in str(frag)


def test_scan_fragment(tmp_path: Path):
    tab = pa.table({"a": range(100), "b": range(100, 200)})
    ds = write_dataset(tab, tmp_path)
    frag = ds.get_fragments()[0]

    actual = frag.to_table(
        columns=["b"],
        filter="a >= 2",
        offset=20,
    )
    expected = pa.table({"b": range(122, 200)})
    assert actual == expected


def test_scan_fragment_with_dynamic_projection(tmp_path: Path):
    tab = pa.table({"a": range(100), "b": range(100, 200)})
    ds = write_dataset(tab, tmp_path)
    frag = ds.get_fragments()[0]

    actual = frag.to_table(
        columns={"b_proj": "b"},
        filter="a >= 2",
        offset=20,
    )
    expected = pa.table({"b_proj": range(122, 200)})
    assert actual == expected


def test_fragment_session(tmp_path: Path):
    tab = pa.table({"a": range(100), "b": range(100, 200)})
    ds = write_dataset(tab, tmp_path)
    frag = ds.get_fragments()[0]

    session = frag.open_session(columns=["a", "b"], with_row_address=False)
    expected = frag.take(indices=range(1, 50), columns=["a", "b"])
    actual = session.take(range(1, 50))
    assert actual == expected

    session = frag.open_session(columns=["a", "b"], with_row_address=True)
    assert session.take(range(1, 5)).schema.names == ["a", "b", "_rowaddr"]


def test_write_fragments(tmp_path: Path):
    # Should result in two files since each batch is 8MB and max_bytes_per_file is small
    batches = pa.RecordBatchReader.from_batches(
        pa.schema([pa.field("a", pa.string())]),
        [
            pa.record_batch([pa.array(["0" * 1024] * 1024 * 8)], names=["a"]),
            pa.record_batch([pa.array(["0" * 1024] * 1024 * 8)], names=["a"]),
        ],
    )

    progress = ProgressForTest()
    fragments = write_fragments(
        batches,
        tmp_path,
        max_rows_per_group=512,
        max_bytes_per_file=1024,
        progress=progress,
    )
    assert len(fragments) == 2
    assert all(isinstance(f, FragmentMetadata) for f in fragments)
    # progress hook was called for each fragment
    assert progress.begin_called == 2
    assert progress.complete_called == 2


def test_write_fragments_schema_holes(tmp_path: Path):
    # Create table with 3 cols
    data = pa.table({"a": range(3)})
    dataset = write_dataset(data, tmp_path)
    dataset.add_columns({"b": "a + 1"})
    dataset.add_columns({"c": "a + 2"})
    # Delete the middle column to create a hole in the field ids
    dataset.drop_columns(["b"])

    def get_field_ids(fragment):
        return [id for f in fragment.files for id in f.fields]

    field_ids = get_field_ids(dataset.get_fragments()[0].metadata)

    data = pa.table({"a": range(3, 6), "c": range(5, 8)})
    fragment = LanceFragment.create(tmp_path, data)
    assert get_field_ids(fragment) == field_ids

    data = pa.table({"a": range(6, 9), "c": range(8, 11)})
    fragments = write_fragments(data, tmp_path)
    assert len(fragments) == 1
    assert get_field_ids(fragments[0]) == field_ids

    operation = LanceOperation.Append([fragment, *fragments])
    dataset = LanceDataset.commit(tmp_path, operation, read_version=dataset.version)

    assert dataset.to_table().equals(pa.table({"a": range(9), "c": range(2, 11)}))


def test_write_fragment_with_progress(tmp_path: Path):
    df = pd.DataFrame({"a": [10 * 10]})
    progress = ProgressForTest()
    LanceFragment.create(tmp_path, df, progress=progress)
    assert progress.begin_called == 1
    assert progress.complete_called == 1


def failing_write(progress_uri: str, dataset_uri: str):
    # re-create progress so we don't have to pickle it
    progress = FileSystemFragmentWriteProgress(
        progress_uri, metadata={"test_key": "test_value"}
    )
    arr = pa.array(range(100))
    batch = pa.record_batch([arr], names=["a"])

    def data():
        yield batch
        raise Exception("Something went wrong!")

    reader = pa.RecordBatchReader.from_batches(batch.schema, data())
    with pytest.raises(Exception):
        LanceFragment.create(
            dataset_uri,
            reader,
            fragment_id=1,
            progress=progress,
        )


def test_dataset_progress(tmp_path: Path):
    dataset_uri = tmp_path / "dataset"
    progress_uri = tmp_path / "progress"
    data = pa.table({"a": range(100)})
    progress = FileSystemFragmentWriteProgress(progress_uri)
    fragment = LanceFragment.create(
        dataset_uri,
        data,
        progress=progress,
    )

    # In-progress file should be deleted
    assert not (progress_uri / "fragment_0.in_progress").exists()

    # Metadata should be written
    with open(progress_uri / "fragment_0.json") as f:
        metadata = json.load(f)

    assert metadata["id"] == 0
    assert len(metadata["files"]) == 1
    # Fragments aren't exactly equal, because the file was written before
    # physical_rows was known.  However, the paths should be the same.
    assert len(fragment.files) == 1
    deserialized = FragmentMetadata.from_json(json.dumps(metadata))
    assert len(deserialized.files) == 1
    assert fragment.files[0].path == deserialized.files[0].path

    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=failing_write, args=(progress_uri, dataset_uri))
    p.start()
    try:
        p.join()
    except Exception:
        # Allow a crash to happen
        pass

    # In-progress file should be present
    with open(progress_uri / "fragment_1.in_progress") as f:
        progress_data = json.load(f)
    assert progress_data["fragment_id"] == 1
    # progress contains custom metadata
    assert progress_data["metadata"]["test_key"] == "test_value"

    # Metadata should be written
    with open(progress_uri / "fragment_1.json") as f:
        metadata = json.load(f)
    assert metadata["id"] == 1


def test_fragment_meta():
    # Intentionally leaving off column_indices / version fields to make sure
    # we can handle backwards compatibility (though not clear we need to)
    data = {
        "id": 0,
        "files": [
            {"path": "0.lance", "fields": [0], "file_size_bytes": 100},
            {"path": "1.lance", "fields": [1]},
        ],
        "deletion_file": None,
        "physical_rows": 100,
    }
    meta = FragmentMetadata.from_json(json.dumps(data))

    assert meta.id == 0
    assert len(meta.files) == 2
    with pytest.warns(DeprecationWarning):
        assert meta.files[0].path() == "0.lance"
    assert meta.files[1].path == "1.lance"

    assert repr(meta) == (
        "FragmentMetadata(id=0, files=[DataFile(path='0.lance', fields=[0], "
        "column_indices=[], file_major_version=0, file_minor_version=0, "
        "file_size_bytes=100), DataFile(path='1.lance', fields=[1], column_indices=[], "
        "file_major_version=0, file_minor_version=0, file_size_bytes=None)], "
        "physical_rows=100, deletion_file=None, row_id_meta=None, "
        "created_at_version_meta=None, last_updated_at_version_meta=None)"
    )


def test_fragment_v2(tmp_path):
    dataset_uri = tmp_path / "dataset"
    tab = pa.table(
        {
            "a": pa.array(range(1024)),
        }
    )
    lance.write_dataset([], dataset_uri, schema=tab.schema)
    fragments = write_fragments(
        tab,
        tmp_path,
        data_storage_version="stable",
    )
    assert len(fragments) == 1
    ds = lance.dataset(dataset_uri)
    assert "major_version: 2" in format_fragment(fragments[0], ds)


def test_mixed_fragment_versions(tmp_path):
    data = pa.table({"a": range(800), "b": range(800)})

    # Create empty v2 dataset
    ds = lance.write_dataset(
        data_obj=[],
        uri=tmp_path / "dataset2",
        schema=data.schema,
        data_storage_version="stable",
    )

    # Add one v1 file and one v2 file
    fragments = []
    fragments.append(
        lance.LanceFragment.create(ds.uri, data, data_storage_version="legacy")
    )
    fragments.append(
        lance.LanceFragment.create(ds.uri, data, data_storage_version="stable")
    )

    # Attempt to commit
    operation = lance.LanceOperation.Overwrite(ds.schema, fragments)
    with pytest.raises(OSError, match="All data files must have the same version"):
        lance.LanceDataset.commit(ds.uri, operation)


def test_create_from_file(tmp_path):
    data = pa.table({"a": range(800), "b": range(800)})
    dataset = lance.write_dataset(
        [], tmp_path, schema=data.schema, data_storage_version="stable"
    )

    # Append first file
    fragment_name = f"{uuid.uuid4()}.lance"
    with LanceFileWriter(str(tmp_path / "data" / fragment_name)) as writer:
        writer.write_batch(data)

    frag = LanceFragment.create_from_file(fragment_name, dataset, 0)
    op = LanceOperation.Append([frag])

    dataset = lance.LanceDataset.commit(dataset.uri, op, read_version=dataset.version)
    frag = dataset.get_fragments()[0]
    assert frag.fragment_id == 0

    assert dataset.count_rows() == 800

    # Append second file (fragment id shouldn't be 0 even though we pass in 0)
    fragment_name = f"{uuid.uuid4()}.lance"
    with LanceFileWriter(str(tmp_path / "data" / fragment_name)) as writer:
        writer.write_batch(data)

    frag = LanceFragment.create_from_file(fragment_name, dataset, 0)
    op = LanceOperation.Append([frag])

    dataset = lance.LanceDataset.commit(dataset.uri, op, read_version=dataset.version)
    frag = dataset.get_fragments()[1]
    assert frag.fragment_id == 1

    assert dataset.count_rows() == 1600

    # Simulate compaction
    compacted_name = f"{uuid.uuid4()}.lance"
    with LanceFileWriter(str(tmp_path / "data" / compacted_name)) as writer:
        for batch in dataset.to_batches():
            writer.write_batch(batch)

    frag = LanceFragment.create_from_file(compacted_name, dataset, 0)
    group = LanceOperation.RewriteGroup(
        old_fragments=[frag.metadata for frag in dataset.get_fragments()],
        new_fragments=[frag],
    )
    op = LanceOperation.Rewrite(groups=[group], rewritten_indices=[])
    dataset = lance.LanceDataset.commit(dataset.uri, op, read_version=dataset.version)

    assert dataset.count_rows() == 1600
    assert len(dataset.get_fragments()) == 1
    assert dataset.get_fragments()[0].fragment_id == 2


def test_fragment_merge(tmp_path):
    schema = pa.schema([pa.field("a", pa.string())])
    batches = pa.RecordBatchReader.from_batches(
        schema,
        [
            pa.record_batch([pa.array(["0" * 1024] * 1024 * 8)], names=["a"]),
            pa.record_batch([pa.array(["0" * 1024] * 1024 * 8)], names=["a"]),
        ],
    )

    progress = ProgressForTest()
    fragments = write_fragments(
        batches,
        tmp_path,
        max_rows_per_group=512,
        max_bytes_per_file=1024,
        progress=progress,
    )

    operation = lance.LanceOperation.Overwrite(schema, fragments)
    dataset = lance.LanceDataset.commit(tmp_path, operation)
    merged = []
    schema = None
    for fragment in dataset.get_fragments():
        table = fragment.scanner(with_row_id=True, columns=[]).to_table()
        table = table.add_column(0, "b", [[i for i in range(len(table))]])
        fragment, schema = fragment.merge(table, "_rowid")
        merged.append(fragment)

    merge = lance.LanceOperation.Merge(merged, schema)
    dataset = lance.LanceDataset.commit(
        tmp_path, merge, read_version=dataset.latest_version
    )

    merged = []
    schema = None
    for fragment in dataset.get_fragments():
        table = fragment.scanner(with_row_address=True, columns=[]).to_table()
        table = table.add_column(0, "c", [[i + 1 for i in range(len(table))]])
        fragment, schema = fragment.merge(table, "_rowaddr")
        merged.append(fragment)

    merge = lance.LanceOperation.Merge(merged, schema)
    dataset = lance.LanceDataset.commit(
        tmp_path, merge, read_version=dataset.latest_version
    )

    merged = []
    for fragment in dataset.get_fragments():
        table = fragment.scanner(columns=["b"]).to_table()
        table = table.add_column(0, "d", [[i + 2 for i in range(len(table))]])
        fragment, schema = fragment.merge(table, "b")
        merged.append(fragment)

    merge = lance.LanceOperation.Merge(merged, schema)
    dataset = lance.LanceDataset.commit(
        tmp_path, merge, read_version=dataset.latest_version
    )
    assert [f.name for f in dataset.schema] == ["a", "b", "c", "d"]


def test_fragment_count_rows(tmp_path: Path):
    data = pa.table({"a": range(800), "b": range(800)})
    ds = write_dataset(data, tmp_path)

    fragments = ds.get_fragments()
    assert len(fragments) == 1

    assert fragments[0].count_rows() == 800
    assert fragments[0].count_rows("a < 200") == 200
    assert fragments[0].count_rows(pc.field("a") < 200) == 200


@pytest.mark.parametrize("enable_stable_row_ids", [False, True])
def test_fragment_metadata_pickle(tmp_path: Path, enable_stable_row_ids: bool):
    ds = write_dataset(
        pa.table({"a": range(100)}),
        tmp_path,
        enable_stable_row_ids=enable_stable_row_ids,
    )
    # Create a deletion file
    ds.delete("a < 50")
    fragment = ds.get_fragments()[0]

    frag_meta = fragment.metadata

    assert frag_meta.deletion_file is not None
    if enable_stable_row_ids:
        assert frag_meta.row_id_meta is not None

    # Pickle and unpickle the fragment metadata
    round_trip = pickle.loads(pickle.dumps(frag_meta))

    assert frag_meta == round_trip


def test_deletion_file_with_base_id_serialization():
    """Test that DeletionFile with base_id serializes correctly."""
    from lance.fragment import DeletionFile, FragmentMetadata

    # Create a DeletionFile with base_id
    deletion_file = DeletionFile(
        read_version=1, id=123, file_type="array", num_deleted_rows=10, base_id=456
    )

    # Verify the base_id is set
    assert deletion_file.base_id == 456

    # Test asdict includes base_id
    deletion_dict = deletion_file.asdict()
    assert "base_id" in deletion_dict
    assert deletion_dict["base_id"] == 456

    # Create a FragmentMetadata with the deletion file
    metadata = FragmentMetadata(
        id=1, files=[], physical_rows=1000, deletion_file=deletion_file
    )

    # Test pickle serialization/deserialization
    pickled = pickle.dumps(metadata)
    unpickled = pickle.loads(pickled)

    # Verify the deletion file was correctly deserialized
    assert unpickled.deletion_file is not None
    assert unpickled.deletion_file.base_id == 456
    assert unpickled == metadata

    # Test JSON serialization/deserialization
    json_data = metadata.to_json()
    assert json_data["deletion_file"]["base_id"] == 456

    deserialized = FragmentMetadata.from_json(json.dumps(json_data))
    assert deserialized.deletion_file is not None
    assert deserialized.deletion_file.base_id == 456


def test_fragment_update_columns_basic(tmp_path):
    """Test basic fragment update columns functionality."""
    # Create initial dataset
    data = pa.table(
        {
            "id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "David"],
            "value": [10, 20, 30, 40],
        }
    )
    dataset_uri = tmp_path / "test_dataset_update_columns_basic"
    dataset = lance.write_dataset(data, dataset_uri)

    # Prepare update data with _rowid (must be UInt64 to match Lance's internal type)
    update_data = pa.table(
        {
            "_rowid": pa.array([0, 2], type=pa.uint64()),
            "name": ["Alice_Updated", "Charlie_Updated"],
            "value": [100, 300],
        }
    )

    # Get the fragment and update columns
    fragment = dataset.get_fragment(0)
    updated_fragment, fields_modified = fragment.update_columns(update_data)

    # Verify fields_modified is returned
    assert isinstance(fields_modified, list)
    assert len(fields_modified) > 0

    # Commit the changes using Update operation

    op = LanceOperation.Update(
        updated_fragments=[updated_fragment],
        fields_modified=fields_modified,
    )
    updated_dataset = lance.LanceDataset.commit(
        str(dataset_uri), op, read_version=dataset.version
    )

    # Verify the update
    result = updated_dataset.to_table().to_pydict()
    assert result["name"] == ["Alice_Updated", "Bob", "Charlie_Updated", "David"]
    assert result["value"] == [100, 20, 300, 40]
    assert result["id"] == [1, 2, 3, 4]  # id column should remain unchanged


def test_fragment_update_columns_with_custom_join_key(tmp_path):
    """Test fragment update columns with custom join key."""
    # Create initial dataset
    data = pa.table(
        {
            "id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "David"],
            "score": [85, 90, 75, 80],
        }
    )
    dataset_uri = tmp_path / "test_dataset_update_columns_custom_join_key"
    dataset = lance.write_dataset(data, dataset_uri)

    # Prepare update data using 'id' as join key
    # Note: We only update 'score', not 'id' itself
    update_data = pa.table(
        {
            "id": [1, 3],
            "name": ["Alan", "Chase"],
            "score": [95, 85],
        }
    )

    # Get the fragment and update columns
    fragment = dataset.get_fragment(0)
    updated_fragment, fields_modified = fragment.update_columns(
        update_data, left_on="id", right_on="id"
    )

    # Commit the changes

    op = LanceOperation.Update(
        updated_fragments=[updated_fragment],
        fields_modified=fields_modified,
    )
    updated_dataset = lance.LanceDataset.commit(
        str(dataset_uri), op, read_version=dataset.version
    )

    # Verify the update
    result = updated_dataset.to_table().to_pydict()
    assert result["score"][0] == 95  # id=1 should have score 95
    assert result["score"][2] == 85  # id=3 should have score 85
    assert result["name"][0] == "Alan"  # id=1 should have name Alan
    assert result["name"][2] == "Chase"  # id=3 should have name Chase


def test_fragment_update_columns_with_nulls(tmp_path):
    """Test fragment update columns with null values."""
    # Create initial dataset
    data = pa.table(
        {
            "id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "David"],
            "optional_field": ["A", "B", "C", "D"],
        }
    )
    dataset_uri = tmp_path / "test_dataset_update_columns_nulls"
    dataset = lance.write_dataset(data, dataset_uri)

    # Prepare update data with null values
    update_data = pa.table(
        {
            "_rowid": pa.array([1, 3], type=pa.uint64()),
            "optional_field": [None, "D_Updated"],
        }
    )

    # Get the fragment and update columns
    fragment = dataset.get_fragment(0)
    updated_fragment, fields_modified = fragment.update_columns(update_data)

    # Commit the changes

    op = LanceOperation.Update(
        updated_fragments=[updated_fragment],
        fields_modified=fields_modified,
    )
    updated_dataset = lance.LanceDataset.commit(
        str(dataset_uri), op, read_version=dataset.version
    )

    # Verify the update
    result = updated_dataset.to_table().to_pydict()
    assert result["optional_field"] == ["A", None, "C", "D_Updated"]


def test_fragment_update_columns_partial_update(tmp_path):
    """Test updating only some columns."""
    # Create initial dataset with multiple columns
    data = pa.table(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["NYC", "LA", "SF"],
        }
    )
    dataset_uri = tmp_path / "test_dataset_update_columns_partial_update"
    dataset = lance.write_dataset(data, dataset_uri)

    # Update only 'age' column, leaving 'name' and 'city' unchanged
    update_data = pa.table(
        {
            "_rowid": pa.array([0, 2], type=pa.uint64()),
            "age": [26, 36],
        }
    )

    # Get the fragment and update columns
    fragment = dataset.get_fragment(0)
    updated_fragment, fields_modified = fragment.update_columns(update_data)

    # Commit the changes

    op = LanceOperation.Update(
        updated_fragments=[updated_fragment],
        fields_modified=fields_modified,
    )
    updated_dataset = lance.LanceDataset.commit(
        str(dataset_uri), op, read_version=dataset.version
    )

    # Verify only age was updated
    result = updated_dataset.to_table().to_pydict()
    assert result["age"] == [26, 30, 36]
    assert result["name"] == ["Alice", "Bob", "Charlie"]  # Unchanged
    assert result["city"] == ["NYC", "LA", "SF"]  # Unchanged


def test_fragment_update_columns_no_match(tmp_path):
    """Test update when no rows match the join condition."""
    # Create initial dataset
    data = pa.table(
        {
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        }
    )
    dataset_uri = tmp_path / "test_dataset_update_columns_no_match"
    dataset = lance.write_dataset(data, dataset_uri)

    # Update data with non-existent _rowid
    update_data = pa.table(
        {
            "_rowid": pa.array(
                [100, 200], type=pa.uint64()
            ),  # These rowids don't exist
            "value": [999, 888],
        }
    )

    # Get the fragment and update columns
    fragment = dataset.get_fragment(0)
    updated_fragment, fields_modified = fragment.update_columns(update_data)

    # Commit the changes

    op = LanceOperation.Update(
        updated_fragments=[updated_fragment],
        fields_modified=fields_modified,
    )
    updated_dataset = lance.LanceDataset.commit(
        str(dataset_uri), op, read_version=dataset.version
    )

    # Verify nothing was updated (fallback to original values)
    result = updated_dataset.to_table().to_pydict()
    assert result["value"] == [10, 20, 30]  # Unchanged


def test_fragment_update_columns_error_on_nonexistent_column(tmp_path):
    """Test that updating a non-existent column raises an error."""
    # Create initial dataset
    data = pa.table(
        {
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        }
    )
    dataset_uri = tmp_path / "test_dataset_update_columns_error_on_nonexistent_column"
    dataset = lance.write_dataset(data, dataset_uri)

    # Try to update a column that doesn't exist
    update_data = pa.table(
        {
            "_rowid": pa.array([0, 1], type=pa.uint64()),
            "nonexistent_column": [100, 200],
        }
    )

    fragment = dataset.get_fragment(0)

    # Should raise an error
    with pytest.raises(Exception) as exc_info:
        fragment.update_columns(update_data)

    assert "does not exist" in str(exc_info.value).lower()


def test_fragment_update_columns_error_on_metadata_column(tmp_path):
    """Test that updating metadata columns raises an error."""
    # Create initial dataset
    data = pa.table(
        {
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        }
    )
    dataset_uri = tmp_path / "test_dataset_update_columns_error_on_metadata_column"
    dataset = lance.write_dataset(data, dataset_uri)

    # Try to update _rowid column (metadata column)
    update_data = pa.table(
        {
            "_rowid": pa.array([0, 1], type=pa.uint64()),
            "_rowaddr": pa.array([999, 888], type=pa.uint64()),  # This should fail
        }
    )

    fragment = dataset.get_fragment(0)

    # Should raise an error
    with pytest.raises(Exception) as exc_info:
        fragment.update_columns(update_data)

    assert (
        "metadata column" in str(exc_info.value).lower()
        or "cannot be updated" in str(exc_info.value).lower()
    )
