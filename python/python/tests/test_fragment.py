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

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
from lance import FragmentMetadata, LanceDataset, LanceFragment
from lance.progress import FileSystemFragmentWriteProgress, FragmentWriteProgress


def test_write_fragment(tmp_path: Path):
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
    dataset = LanceDataset._commit(tmp_path, schema, fragments)

    df = dataset.to_table().to_pandas()
    pd.testing.assert_frame_equal(
        df, pd.DataFrame({"a": [i * 10 for i in range(num_files)]})
    )


class ProgressForTest(FragmentWriteProgress):
    def __init__(self):
        super().__init__()
        self.begin_called = 0
        self.complete_called = 0

    def begin(
        self, fragment: FragmentMetadata, multipart_id: Optional[str] = None, **kwargs
    ):
        self.begin_called += 1

    def complete(self, fragment: FragmentMetadata):
        self.complete_called += 1


def test_write_fragment_with_progress(tmp_path: Path):
    df = pd.DataFrame({"a": [10 * 10]})
    progress = ProgressForTest()
    LanceFragment.create(tmp_path, df, progress=progress)
    assert progress.begin_called == 1
    assert progress.complete_called == 1


def test_dataset_progress(tmp_path: Path):
    data = pa.table({"a": range(100)})
    fragment = LanceFragment.create(
        tmp_path, data, progress=FileSystemFragmentWriteProgress(tmp_path)
    )

    # In-progress file should be deleted
    assert not (tmp_path / "fragment_0.in_progress").exists()

    # Metadata should be written
    with open(tmp_path / "fragment_0.json") as f:
        metadata = json.load(f)

    assert metadata["id"] == 0
    assert len(metadata["files"]) == 1

    assert fragment == FragmentMetadata.from_json(json.dumps(metadata))
