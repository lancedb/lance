# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import json
import multiprocessing
import uuid
from pathlib import Path

import lance
import pandas as pd
import pyarrow as pa
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
    assert "file_major_version: 2" not in str(frag)

    tab = pa.table({"a": range(1024)})
    frag = LanceFragment.create(tmp_path, tab, data_storage_version="stable")
    assert "file_major_version: 2" in str(frag)


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
        return [id for f in fragment.data_files() for id in f.field_ids()]

    field_ids = get_field_ids(dataset.get_fragments()[0])

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
    assert len(fragment.data_files()) == 1
    deserialized = FragmentMetadata.from_json(json.dumps(metadata))
    assert len(deserialized.data_files()) == 1
    assert fragment.data_files()[0].path() == deserialized.data_files()[0].path()

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
            {"path": "0.lance", "fields": [0]},
            {"path": "1.lance", "fields": [1]},
        ],
        "deletion_file": None,
        "physical_rows": 100,
    }
    meta = FragmentMetadata.from_json(json.dumps(data))

    assert meta.id == 0
    assert len(meta.data_files()) == 2
    assert meta.data_files()[0].path() == "0.lance"
    assert meta.data_files()[1].path() == "1.lance"

    assert repr(meta) == (
        'Fragment { id: 0, files: [DataFile { path: "0.lance", fields: [0], '
        "column_indices: [], file_major_version: 0, file_minor_version: 0 }, "
        'DataFile { path: "1.lance", fields: [1], column_indices: [], '
        "file_major_version: 0, file_minor_version: 0 }], deletion_file: None, "
        "row_id_meta: None, physical_rows: Some(100) }"
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

    dataset = lance.LanceDataset.commit(dataset.uri, op, dataset.version)
    frag = dataset.get_fragments()[0]
    assert frag.fragment_id == 0

    assert dataset.count_rows() == 800

    # Append second file (fragment id shouldn't be 0 even though we pass in 0)
    fragment_name = f"{uuid.uuid4()}.lance"
    with LanceFileWriter(str(tmp_path / "data" / fragment_name)) as writer:
        writer.write_batch(data)

    frag = LanceFragment.create_from_file(fragment_name, dataset, 0)
    op = LanceOperation.Append([frag])

    dataset = lance.LanceDataset.commit(dataset.uri, op, dataset.version)
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
    dataset = lance.LanceDataset.commit(dataset.uri, op, dataset.version)

    assert dataset.count_rows() == 1600
    assert len(dataset.get_fragments()) == 1
    assert dataset.get_fragments()[0].fragment_id == 2
