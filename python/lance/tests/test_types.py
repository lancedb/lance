#  Copyright 2022 Lance Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import platform

import numpy as np
import pyarrow as pa
import pytest

import lance
from lance.types import *

if platform.system() != "Linux":
    pytest.skip(allow_module_level=True)


def test_image(tmp_path):
    data = [f"s3://bucket/{x}.jpg" for x in ["a", "b", "c"]]
    storage = pa.StringArray.from_pandas(data)
    image_type = ImageType.from_storage(storage.type)
    _test_extension_rt(tmp_path, image_type, storage)


def test_image_binary(tmp_path):
    data = [b"<imagebytes>" for x in ["a", "b", "c"]]
    storage = pa.StringArray.from_pandas(data)
    image_type = ImageType.from_storage(storage.type)
    _test_extension_rt(tmp_path, image_type, storage)


def test_point(tmp_path):
    point_type = Point2dType()
    data = [(float(x), float(x)) for x in range(100)]
    storage = pa.array(data, pa.list_(pa.float64()))
    _test_extension_rt(tmp_path, point_type, storage)


def test_box2d(tmp_path):
    box_type = Box2dType()
    data = [(float(x), float(x), float(x), float(x)) for x in range(100)]
    storage = pa.array(data, pa.list_(pa.float64()))
    _test_extension_rt(tmp_path, box_type, storage)


def test_label(tmp_path):
    label_type = LabelType()
    values = ["cat", "dog", "horse", "chicken", "donkey", "pig"]
    indices = np.random.randint(0, len(values), 100)
    storage = pa.DictionaryArray.from_arrays(
        pa.array(indices, type=pa.int8()), pa.array(values, type=pa.string())
    )
    _test_extension_rt(tmp_path, label_type, storage)


def _test_extension_rt(tmp_path, ext_type, storage_arr):
    arr = pa.ExtensionArray.from_storage(ext_type, storage_arr)
    table = pa.Table.from_arrays([arr], names=["ext"])
    lance.write_table(table, str(tmp_path / "test.lance"))
    table = lance.dataset(str(tmp_path / "test.lance")).to_table()
    assert table["ext"].type == ext_type
    assert table["ext"].to_pylist() == storage_arr.to_pylist()
