# Copyright 2022 Lance Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

torch = pytest.importorskip("torch")

from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import pyarrow as pa
import pyarrow.compute as pc

import lance
from lance.pytorch.data import LanceDataset
from lance.types import ImageArray, ImageBinary


def test_data_loader(tmp_path: Path):
    torch.Tensor([1, 2, 3])
    ids = pa.array(range(10))
    values = pa.array(range(10, 20))
    tab = pa.Table.from_arrays([ids, values], names=["id", "value"])

    lance.write_dataset(tab, tmp_path / "lance")
    dataset = LanceDataset(tmp_path / "lance", batch_size=4, mode="batch")
    id_batch, value_batch = next(iter(dataset))
    assert id_batch.shape == torch.Size([4])
    assert torch.is_tensor(id_batch)
    assert torch.equal(id_batch, torch.tensor([0, 1, 2, 3]))
    assert torch.is_tensor(value_batch)
    assert torch.equal(value_batch, torch.tensor([10, 11, 12, 13]))


def test_dataset_with_ext_types(tmp_path: Path):
    images = []
    labels = []
    for i in range(10):
        images.append(
            ImageBinary.from_numpy(
                np.random.randint(0, 256, size=(32, 32), dtype=np.uint8)
            )
        )
        labels.append(["cat", "dog", "goat"][i % 3])

    image_arr = ImageArray.from_images(images)
    labels_arr = pa.DictionaryArray.from_pandas(pd.Series(labels, dtype="category"))

    tab = pa.Table.from_arrays([image_arr, labels_arr], names=["image", "label"])
    lance.write_dataset(tab, tmp_path / "lance")

    dataset = LanceDataset(tmp_path / "lance", batch_size=4, mode="batch")
    batch = next(iter(dataset))
    assert len(batch) == 2 and all([len(col) == 4 for col in batch])
    images, labels = batch
    assert all([isinstance(p, PIL.Image.Image) for p in images])
    assert torch.equal(labels, torch.tensor([0, 1, 2, 0], dtype=torch.int8))


def test_data_loader_with_filter(tmp_path: Path):
    torch.Tensor([1, 2, 3])
    ids = pa.array(range(10))
    values = pa.array(range(10, 20))
    split = pa.array(["train", "val"] * 5)
    tab = pa.Table.from_arrays([ids, values, split], names=["id", "value", "split"])

    lance.write_dataset(tab, tmp_path / "lance")

    dataset = LanceDataset(tmp_path / "lance", filter=pc.field("split") == "train")
    for id, value, split in dataset:
        assert split == "train"
        assert id % 2 == 0
        assert torch.is_tensor(id)
        assert (value - 10) % 2 == 0
        assert torch.is_tensor(value)


def test_data_loader_projection(tmp_path: Path):
    ids = pa.array(range(10))
    values = pa.array([f"num-{i}" for i in ids])
    tab = pa.Table.from_arrays([ids, values], names=["id", "value"])
    lance.write_dataset(tab, tmp_path / "lance")

    dataset = LanceDataset(
        tmp_path / "lance", columns=["value"], filter=pc.field("id") >= 5
    )
    for elem, expected_id in zip(dataset, range(5, 10)):
        assert elem == f"num-{expected_id}"


def test_filter_resulted_empty_return(tmp_path: Path):
    ids = pa.array(range(10))
    values = pa.array([i.as_py() > 5 for i in ids])
    table = pa.Table.from_arrays([ids, values], names=["id", "bignum"])
    lance.write_dataset(table, tmp_path / "lance")

    dataset = LanceDataset(
        tmp_path / "lance",
        columns=["id"],
        filter=pc.field("bignum") == True,
        mode="batch",
        batch_size=2,
    )
    actual_ids = torch.stack(list(dataset))
    assert torch.equal(
        actual_ids, torch.stack([torch.tensor([6, 7]), torch.tensor([8, 9])])
    )


def test_multiversioned_data_loader(tmp_path: Path):
    ids = pa.array(range(10))
    values = pa.array([f"first-{i}" for i in ids])
    table = pa.Table.from_arrays([ids, values], names=["id", "value"])

    data_uri = tmp_path / "lance"
    lance.write_dataset(table, data_uri)

    ids = pa.array(range(10, 20))
    values = pa.array([f"second-{i}" for i in ids])
    table = pa.Table.from_arrays([ids, values], names=["id", "value"])
    lance.write_dataset(table, data_uri, mode="append")

    dataset = LanceDataset(
        data_uri,
        columns=["id"],
        version=1,
    )
    assert len(list(iter(dataset))) == 10

    dataset = LanceDataset(
        data_uri,
        columns=["id"],
        version=2,
    )
    assert len(list(iter(dataset))) == 20
