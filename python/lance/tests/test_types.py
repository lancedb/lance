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
import pickle
import platform

import numpy as np
import pyarrow as pa
import pytest

import lance
from lance.types import (
    Box2dArray,
    Box2dType,
    Image,
    ImageBinary,
    ImageType,
    ImageUri,
    LabelArray,
    LabelType,
    Point2dType,
)

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
    points = np.random.random(100 * 2)
    storage = pa.FixedSizeListArray.from_arrays(points, 2)
    _test_extension_rt(tmp_path, point_type, storage)


def test_box2d(tmp_path):
    box_type = Box2dType()
    data = np.random.random(100 * 4)
    storage = pa.FixedSizeListArray.from_arrays(data, 4)
    ext_arr = _test_extension_rt(tmp_path, box_type, storage)
    assert len(ext_arr.chunks) == 1
    ext_arr = ext_arr.chunks[0]
    assert isinstance(ext_arr, Box2dArray)
    reshaped = data.reshape((100, 4))
    xmin, ymin, xmax, ymax = (
        reshaped[:, 0],
        reshaped[:, 1],
        reshaped[:, 2],
        reshaped[:, 3],
    )
    assert np.all(ext_arr.xmin == xmin)
    assert np.all(ext_arr.xmax == xmax)
    assert np.all(ext_arr.ymin == ymin)
    assert np.all(ext_arr.ymax == ymax)
    actual_areas = ext_arr.area()
    expected_areas = (xmax - xmin + 1) * (ymax - ymin + 1)
    assert np.all(actual_areas == expected_areas)
    actual_iou = ext_arr[:20].iou(ext_arr[90:])
    expected_iou = _naive_iou(ext_arr[:20], ext_arr[90:])
    assert np.all(actual_iou == expected_iou)


def _naive_iou(box_a, box_b):
    ious = np.zeros((len(box_a), len(box_b)))
    for i in range(len(box_a)):
        for j in range(len(box_b)):
            xmin = max(box_a.xmin[i], box_b.xmin[j])
            ymin = max(box_a.ymin[i], box_b.ymin[j])
            xmax = min(box_a.xmax[i], box_b.xmax[j])
            ymax = min(box_a.ymax[i], box_b.ymax[j])
            # compute the area of intersection rectangle
            inter = max(0, xmax - xmin + 1) * max(0, ymax - ymin + 1)
            # compute the area of both the prediction and ground-truth
            # rectangles
            area_i = (box_a.xmax[i] - box_a.xmin[i] + 1) * (
                box_a.ymax[i] - box_a.ymin[i] + 1
            )
            area_j = (box_b.xmax[j] - box_b.xmin[j] + 1) * (
                box_b.ymax[j] - box_b.ymin[j] + 1
            )
            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            ious[i, j] = inter / float(area_i + area_j - inter)
    return ious


def test_label(tmp_path):
    label_type = LabelType()
    values = ["cat", "dog", "horse", "chicken", "donkey", "pig"]
    indices = np.random.randint(0, len(values), 100)
    storage = pa.DictionaryArray.from_arrays(
        pa.array(indices, type=pa.int8()), pa.array(values, type=pa.string())
    )
    ext_arr = _test_extension_rt(tmp_path, label_type, storage)
    assert len(ext_arr.chunks) == 1
    ext_arr = ext_arr.chunks[0]
    assert isinstance(ext_arr, LabelArray)
    expected_arr = LabelArray.from_values(storage.to_numpy(False), values)
    assert ext_arr == expected_arr


def _test_extension_rt(tmp_path, ext_type, storage_arr):
    arr = pa.ExtensionArray.from_storage(ext_type, storage_arr)
    table = pa.Table.from_arrays([arr], names=["ext"])
    lance.write_table(table, str(tmp_path / "test.lance"))
    table = lance.dataset(str(tmp_path / "test.lance")).to_table()
    assert table["ext"].type == ext_type
    assert table["ext"].to_pylist() == arr.to_pylist()
    return table["ext"]


def test_pickle(tmp_path):
    img = Image.create("uri")
    assert isinstance(img, ImageUri)
    with (tmp_path / "image").open("wb") as fh:
        pickle.dump(img, fh)
    with (tmp_path / "image").open("rb") as fh:
        assert img == pickle.load(fh)

    img = Image.create(b"bytes")
    assert isinstance(img, ImageBinary)
    with (tmp_path / "image").open("wb") as fh:
        pickle.dump(img, fh)
    with (tmp_path / "image").open("rb") as fh:
        assert img == pickle.load(fh)
