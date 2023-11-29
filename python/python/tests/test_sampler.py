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

import dataclasses
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pytest

import lance
from lance.dataset import LanceDataset
from lance.lance import DatasetSample
from lance.sampler import SampleParams, build_shuffle_sample, maybe_sample


def test_sample_dataset(tmp_path: Path):
    # 10240 of 32-d vectors.
    data = np.random.random(10240 * 32).astype("f")

    fsl = pa.FixedSizeListArray.from_arrays(data, 32)
    tbl = pa.Table.from_arrays([fsl], ["vec"])

    ds = lance.write_dataset(tbl, tmp_path / "data.lance")

    # Simple path
    simple_scan = list(maybe_sample(ds, 128, ["vec"]))
    assert len(simple_scan) == 1
    assert isinstance(simple_scan[0], pa.RecordBatch)
    assert simple_scan[0].schema == pa.schema([pa.field("vec", fsl.type)])
    assert simple_scan[0].num_rows == 128

    # Random path.
    large_scan = list(maybe_sample(ds, 128, ["vec"], max_takes=32))
    assert len(large_scan) == 1
    assert isinstance(large_scan[0], pa.RecordBatch)
    assert large_scan[0].schema == pa.schema([pa.field("vec", fsl.type)])
    assert large_scan[0].num_rows == 128


@pytest.fixture(scope="module")
def readonly_dataset(tmpdir_factory):
    tmp_path = Path(tmpdir_factory.mktemp("data"))

    nrows = 10240
    ndims = 32

    ids = pa.array(np.arange(nrows))
    data = np.random.random(nrows * ndims).astype("f")

    fsl = pa.FixedSizeListArray.from_arrays(data, ndims)
    tbl = pa.Table.from_arrays([ids, fsl], ["id", "vec"])

    return lance.write_dataset(tbl, tmp_path / "data.lance")


def test_sample_params(readonly_dataset: LanceDataset):
    params = SampleParams(
        "id > 20",
        batch_size=1024,
        shuffle=True,
        sample_rate=0.5,
        seed=42,
    )

    sample = build_shuffle_sample(readonly_dataset, params)
    assert sample.params == params

    sample = build_shuffle_sample(readonly_dataset, **dataclasses.asdict(params))
    assert sample.params == params

    assert repr(sample.params) in repr(sample)


def test_sample_num_rows(readonly_dataset: LanceDataset):
    sample = build_shuffle_sample(readonly_dataset)
    assert sample.num_rows == len(readonly_dataset)
    assert sample.metrics.dataset_size == len(readonly_dataset)
    assert sample.metrics.matched_rows == len(readonly_dataset)
    assert sample.metrics.sampled_rows == len(readonly_dataset)

    sample = build_shuffle_sample(
        readonly_dataset, predicate="id >= 20", batch_size=1024
    )
    assert sample.num_rows == len(readonly_dataset) - 20
    assert sample.metrics.dataset_size == len(readonly_dataset)
    assert sample.metrics.matched_rows == len(readonly_dataset) - 20
    assert sample.metrics.sampled_rows == len(readonly_dataset) - 20

    for i in range(len(sample)):
        indices = sample[i]
        assert len(indices) <= 1024
        assert all(20 <= idx.as_py() < len(readonly_dataset) for idx in indices)

    sample_sliced = sample[1:]
    assert len(sample_sliced) == len(sample) - 1
    assert sample_sliced.num_rows == len(readonly_dataset) - 20 - len(sample[0])


@pytest.mark.parametrize("batch_size", [32, 1024])
@pytest.mark.parametrize("sample_rate", [None, 0.5, 1.0])
@pytest.mark.parametrize("shuffle", [True, False])
def test_shuffle_sample_slice(
    batch_size: int,
    sample_rate: Optional[float],
    shuffle: bool,
    readonly_dataset: LanceDataset,
):
    params = SampleParams(
        "id > 20",
        batch_size=batch_size,
        shuffle=shuffle,
        sample_rate=sample_rate,
        seed=42,
    )
    sample = build_shuffle_sample(readonly_dataset, params)

    materialized = list(iter(sample))
    assert len(materialized) == len(sample)
    assert sum(len(arr) for arr in materialized) == sample.num_rows

    # Check that materializing a slice of a sample gives the save result as
    # slicing the full materialized sample.
    starts = range(len(sample))
    lengths = [1, 4, len(sample)]
    step = [None, 1, 2]
    for start, length, step in product(starts, lengths, step):
        stop = start + length
        sample_sliced = sample[start:stop:step]
        materialized_sliced = list(iter(sample_sliced))

        assert materialized_sliced == materialized[start:stop:step]


def test_shuffle_sample_serialize(readonly_dataset: LanceDataset, tmp_path: Path):
    params = SampleParams(
        "id > 20",
        batch_size=1024,
        shuffle=True,
        sample_rate=0.5,
        seed=42,
    )
    sample = build_shuffle_sample(readonly_dataset, params)

    # Roundtrip with a file path.
    path = str(tmp_path / "sample.tar.gz")
    sample.serialize_into(path)
    sample_deserialized = DatasetSample.deserialize_from(path)
    assert sample == sample
    assert sample == sample_deserialized

    # Read with a file object
    with open(path, "rb") as f:
        sample_deserialized = DatasetSample.deserialize_from(f)
    assert sample == sample_deserialized

    # Roundtrip with a file object.
    path = str(tmp_path / "sample2.tar.gz")
    with open(path, "wb") as f:
        sample.serialize_into(f)

    with open(path, "rb") as f:
        sample_deserialized = DatasetSample.deserialize_from(f)

    assert sample == sample_deserialized
