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

import os
import pickle
import platform

import numpy as np
import pyarrow as pa
import pytest

import lance
from lance.types import (
    Box2dArray,
    Box2dType,
    Box3dArray,
    Box3dType,
    Image,
    ImageArray,
    ImageBinary,
    ImageType,
    ImageUri,
    LabelArray,
    LabelType,
    Point2dType,
    Point3dType,
    is_image_type,
)

if platform.system() != "Linux":
    pytest.skip(allow_module_level=True)


def test_image(tmp_path):
    data = [f"s3://bucket/{x}.jpg" for x in ["a", "b", "c"]]
    storage = pa.StringArray.from_pandas(data)
    _test_image(tmp_path, storage)


def test_image_binary(tmp_path):
    data = [b"<imagebytes>" for x in ["a", "b", "c"]]
    storage = pa.BinaryArray.from_pandas(data)
    _test_image(tmp_path, storage)


def test_image_array():
    images = [Image.create(x) for x in ["uri1", "uri2", None, "uri4"]]
    from_images = ImageArray.from_images(images)
    from_pandas = ImageArray.from_pandas(images)
    from_pandas_storage = ImageArray.from_pandas(
        [None if x is None else x.uri for x in images]
    )
    assert from_images == from_pandas
    assert from_pandas == from_pandas_storage
    assert isinstance(from_images, ImageArray)
    assert isinstance(from_images.to_pylist()[0], Image)


def test_image_array_chunks():
    images = [pa.array(["uri1", "uri2"]), pa.array(["uri3", "uri4"])]
    chunks = pa.chunked_array(images, pa.string())
    arr = ImageArray.from_pandas(chunks)
    assert isinstance(arr, pa.ChunkedArray)
    assert isinstance(arr.chunks[0], ImageArray)


def _test_image(tmp_path, storage):
    image_type = ImageType.from_storage(storage.type)
    ext_arr = _test_extension_rt(tmp_path, image_type, storage)
    assert len(ext_arr.chunks) == 1
    assert is_image_type(ext_arr.type)
    ext_arr = ext_arr.chunks[0]
    assert isinstance(ext_arr, ImageArray)
    expected_arr = ImageArray.from_pandas(storage.to_pylist())
    assert ext_arr == expected_arr


def test_point(tmp_path):
    for point_type in [Point2dType(), Point3dType()]:
        ndims = point_type.ndims
        points = np.random.random(100 * ndims)
        storage = pa.FixedSizeListArray.from_arrays(points, ndims)
        storage_path = tmp_path / str(ndims)
        os.makedirs(storage_path, exist_ok=True)
        _test_extension_rt(storage_path, point_type, storage)


def test_box(tmp_path):
    for box_type in [Box2dType(), Box3dType()]:
        ndims = box_type.ndims
        data = np.random.random(100 * 2 * ndims)
        storage = pa.FixedSizeListArray.from_arrays(data, 2 * ndims)
        storage_path = tmp_path / str(ndims)
        os.makedirs(storage_path, exist_ok=True)
        ext_arr = _test_extension_rt(storage_path, box_type, storage)
        assert len(ext_arr.chunks) == 1
        ext_arr = ext_arr.chunks[0]
        assert ext_arr.ndims == ndims
        # check points
        reshaped = data.reshape((100, 2 * ext_arr.ndims))
        _check_points(ext_arr, reshaped, ndims)
        _check_size(ext_arr, reshaped, ndims)
        _check_iou(ext_arr[:20], ext_arr[90:])


def _check_points(ext_arr, reshaped, ndims):
    assert np.all(ext_arr.xmin == reshaped[:, 0])
    assert np.all(ext_arr.ymin == reshaped[:, 1])
    assert np.all(ext_arr.xmax == reshaped[:, ndims])
    assert np.all(ext_arr.ymax == reshaped[:, ndims + 1])
    if ndims == 3:
        assert np.all(ext_arr.zmin == reshaped[:, 2])
        assert np.all(ext_arr.zmax == reshaped[:, ndims + 2])


def _check_size(ext_arr, reshaped, ndims):
    actual_sizes = ext_arr._box_sizes()
    expected_size = (reshaped[:, ndims] - reshaped[:, 0] + 1) * (
        reshaped[:, ndims + 1] - reshaped[:, 1] + 1
    )
    if ndims == 3:
        expected_size *= reshaped[:, ndims + 2] - reshaped[:, 2] + 1
    assert np.all(actual_sizes == expected_size)

    if ndims == 2:
        assert np.all(ext_arr.area() == ext_arr._box_sizes())
    elif ndims == 3:
        assert np.all(ext_arr.volume() == ext_arr._box_sizes())


def _check_iou(box_arr1, box_arr2):
    actual_iou = box_arr1.iou(box_arr2)
    expected_iou = _naive_iou(box_arr1, box_arr2)
    assert np.all(actual_iou == expected_iou)


def _naive_iou(box_a_arr, box_b_arr):
    assert box_a_arr.ndims == box_b_arr.ndims
    ndims = box_a_arr.ndims
    ious = np.zeros((len(box_a_arr), len(box_b_arr)))
    for i, box_i in enumerate(box_a_arr.to_numpy()):
        for j, box_j in enumerate(box_b_arr.to_numpy()):
            area_i = _box_size(box_i, ndims)
            area_j = _box_size(box_j, ndims)
            inter_ij = _intersection(box_i, box_j, ndims)
            ious[i, j] = inter_ij / float(area_i + area_j - inter_ij)
    return ious


def _box_size(box, ndims):
    size = _length(box, 0, ndims) * _length(box, 1, ndims)
    if ndims == 3:
        size *= _length(box, 2, ndims)
    return size


def _length(box, axis, ndims):
    return max(0, box[axis + ndims] - box[axis] + 1)


def _intersection(box_a, box_b, ndims):
    inter_box = [None] * ndims * 2
    for i in range(ndims):
        inter_min_i = max(box_a[i], box_b[i])
        inter_max_i = min(box_a[i + ndims], box_b[i + ndims])
        inter_box[i] = inter_min_i
        inter_box[i + ndims] = inter_max_i
    return _box_size(inter_box, ndims)


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

    img = Image.create(bytearray(b"bytes"))
    assert isinstance(img, ImageBinary)
    with (tmp_path / "image").open("wb") as fh:
        pickle.dump(img, fh)
    with (tmp_path / "image").open("rb") as fh:
        assert img == pickle.load(fh)
