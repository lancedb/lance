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

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

torch = pytest.importorskip("torch")

import PIL

import lance
from lance.pytorch.data import LanceDataset
from lance.types import ImageArray, ImageBinary


def test_data_loader(tmp_path: Path):
    torch.Tensor([1, 2, 3])
    ids = pa.array(range(10))
    values = pa.array(range(10, 20))
    tab = pa.Table.from_arrays([ids, values], names=["id", "value"])

    lance.write_table(tab, tmp_path / "lance")

    dataset = LanceDataset(tmp_path / "lance", batch_size=4)
    id_batch, value_batch = next(iter(dataset))
    assert id_batch.shape == torch.Size([4])


def test_dataset_with_image(tmp_path: Path):
    images = []
    for _ in range(10):
        images.append(
            ImageBinary.from_numpy(
                np.random.randint(0, 256, size=(32, 32), dtype=np.uint8)
            )
        )
    image_arr = ImageArray.from_images(images)

    tab = pa.Table.from_arrays([image_arr], names=["image"])
    lance.write_table(tab, tmp_path / "lance")

    dataset = LanceDataset(tmp_path / "lance", batch_size=4)
    batch = next(iter(dataset))
    print(batch)
    assert len(batch) == 4
    assert all([isinstance(p, PIL.Image.Image) for p in batch])

