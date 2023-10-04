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

import re
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from helper import requires_pyarrow_12
from lance.arrow import (
    BFloat16,
    BFloat16Array,
    ImageArray,
    ImageURIArray,
    PandasBFloat16Array,
    bfloat16_array,
)


def test_bf16_value():
    val = BFloat16(1.124565)
    assert val == BFloat16(1.124565)
    assert str(val) == "1.125"

    should_be_true = [
        BFloat16(1.1) > BFloat16(1.0),
        BFloat16(1.1) >= BFloat16(1.0),
        BFloat16(1.1) != BFloat16(1.0),
        BFloat16(1.0) < BFloat16(1.1),
        BFloat16(1.0) <= BFloat16(1.1),
    ]
    assert all(comparison for comparison in should_be_true)

    should_be_false = [
        BFloat16(1.1) < BFloat16(1.0),
        BFloat16(1.1) <= BFloat16(1.0),
        BFloat16(1.1) == BFloat16(1.0),
        BFloat16(1.0) >= BFloat16(1.1),
        BFloat16(1.0) >= BFloat16(1.1),
    ]
    assert not any(comparison for comparison in should_be_false)


def test_bf16_repr():
    data = [1.1, None, 3.4]
    arr = bfloat16_array(data)
    assert isinstance(arr, BFloat16Array)

    expected = [BFloat16(1.1), None, BFloat16(3.4)]
    assert arr[0].as_py() == expected[0]
    assert arr[1].as_py() == expected[1]
    assert arr[2].as_py() == expected[2]

    assert arr.to_pylist() == expected

    expected_re = r"""<lance.arrow.BFloat16Array object at 0x[\w\d]+>
\[1.1015625, None, 3.40625\]"""
    assert re.match(expected_re, repr(arr))

    # TODO: uncomment tests once fixed upstream.
    # https://github.com/apache/arrow/issues/36648


#     tab = pa.table({"x": arr})
#     expected_re = r"""pyarrow.Table
# x: extension<lance.bfloat16<BFloat16Type>>
# ---
# x: \[\[1.1015625, None, 3.40625\]\]"""
#     assert re.match(expected_re, repr(tab))


def test_bf16_pandas(provide_pandas):
    data = [1.1, None, 3.4]
    arr = bfloat16_array(data)
    arr_pandas = arr.to_pandas()
    assert arr_pandas[0] == BFloat16(1.1)
    assert arr_pandas[1] is None
    assert arr_pandas[2] == BFloat16(3.4)

    # Can instantiate with dtype string
    series = pd.Series(arr_pandas, dtype="lance.bfloat16")
    pd.testing.assert_series_equal(arr_pandas, series)

    # Can roundtrip to Arrow
    arr_arrow = pa.array(arr_pandas)
    assert isinstance(arr_arrow, BFloat16Array)
    assert arr == arr_arrow

    pd.testing.assert_series_equal(arr_arrow.to_pandas(), arr_pandas)


def test_bf16_numpy():
    import numpy as np
    from ml_dtypes import bfloat16

    data = [1.1, 2.1, 3.4]
    arr = bfloat16_array(data)
    arr_numpy = arr.to_numpy()

    expected = np.array(data, dtype=bfloat16)

    np.testing.assert_array_equal(arr_numpy, expected)

    # Can roundtrip to Pandas
    arr_pandas = PandasBFloat16Array.from_numpy(arr_numpy)
    np.testing.assert_array_equal(arr_pandas.to_numpy(), expected)

    # Can roundtrip to Arrow
    arr_arrow = BFloat16Array.from_numpy(arr_numpy)
    assert arr == arr_arrow
    np.testing.assert_array_equal(arr_arrow.to_numpy(), expected)


@requires_pyarrow_12
def test_roundtrip_take_ext_types(tmp_path: Path):
    tensor_type = pa.fixed_shape_tensor(pa.float32(), [2, 3])
    inner = pa.array([float(x) for x in range(0, 18)], pa.float32())
    storage = pa.FixedSizeListArray.from_arrays(inner, 6)
    tensor_arr = pa.ExtensionArray.from_storage(tensor_type, storage)

    tbl = pa.Table.from_arrays([tensor_arr], ["tensor"])
    lance.write_dataset(tbl, tmp_path)

    tbl2 = lance.dataset(tmp_path)
    rows = tbl2.take([0, 2])
    assert rows["tensor"].to_pylist() == [
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
    ]


def test_image_arrays(tmp_path: Path):
    import os
    from pathlib import Path

    import tensorflow as tf

    n = 10
    png_uris = [
        "file://" + os.path.join(os.path.dirname(__file__), "images/1.png"),
        os.path.join(os.path.dirname(__file__), "images/1.png"),
    ] * int(n / 2)

    if os.name == "nt":
        png_uris = [str(Path(x)) for x in png_uris]

    assert ImageArray.from_array(png_uris).to_pylist() == png_uris
    assert (
        ImageArray.from_array(pa.array(png_uris, pa.string())).to_pylist() == png_uris
    )
    image_array = ImageURIArray.from_uris(png_uris)
    assert ImageArray.from_array(image_array) == image_array

    encoded_image_array = image_array.read_uris()
    assert len(ImageArray.from_array(encoded_image_array.storage)) == n
    assert ImageArray.from_array(encoded_image_array) == encoded_image_array

    tensor_image_array = encoded_image_array.to_tensor()
    fixed_shape_tensor_array = pa.ExtensionArray.from_storage(
        pa.fixed_shape_tensor(
            tensor_image_array.storage.type.value_type, tensor_image_array.type.shape
        ),
        tensor_image_array.storage,
    )
    assert (
        ImageArray.from_array(fixed_shape_tensor_array).storage
        == fixed_shape_tensor_array.storage
    )
    assert ImageArray.from_array(tensor_image_array) == tensor_image_array

    assert len(ImageArray.from_array(tensor_image_array)) == n
    assert len(tensor_image_array) == n
    assert tensor_image_array.storage.type == pa.list_(pa.uint8(), 4)
    assert tensor_image_array[2].as_py() == [42, 42, 42, 255]

    test_tensor = tf.constant(
        np.array([42, 42, 42, 255] * n, dtype=np.uint8).reshape((n, 1, 1, 4))
    )

    assert test_tensor.shape == (n, 1, 1, 4)
    assert tf.math.reduce_all(
        tf.convert_to_tensor(tensor_image_array.to_numpy()) == test_tensor
    )
    assert tensor_image_array.to_encoded().to_tensor() == tensor_image_array

    def png_encoder(images):
        import tensorflow as tf

        encoded_images = (
            tf.io.encode_png(x).numpy() for x in tf.convert_to_tensor(images)
        )
        return pa.array(encoded_images, type=pa.binary())

    assert tensor_image_array.to_encoded(png_encoder).to_tensor() == tensor_image_array
    uris = [
        os.path.join(os.path.dirname(__file__), "images/1.png"),
        os.path.join(os.path.dirname(__file__), "images/2.jpeg"),
    ]
    encoded_image_array = ImageArray.from_array(uris).read_uris()
    with pytest.raises(
        tf.errors.InvalidArgumentError, match="Shapes of all inputs must match"
    ):
        encoded_image_array.to_tensor()

    pattern = r"(object at) 0x[\w\d]+(:?>)"
    repl = r"\1 0x..\2"
    assert re.sub(pattern, repl, encoded_image_array.__repr__()) == (
        "<lance.arrow.EncodedImageArray object at 0x..>\n"
        "[<tf.Tensor: shape=(1, 1, 4), dtype=uint8, numpy=array([[[ 42,  42,  42, "
        "255]]], dtype=uint8)>, ..]\n"
        "b'\\x89PNG\\r\\n\\x1a\\n\\x00\\x00\\x00\\rIHDR\\x00\\x00\\x00\\x01\\x00\\x00"
        "\\x00\\x01\\x08\\x06\\x00\\x00\\x00\\x1f'"
    )


@requires_pyarrow_12
def test_roundtrip_image_tensor(tmp_path: Path):
    import os

    png_uris = [
        os.path.join(os.path.dirname(__file__), "images/1.png"),
    ] * 5
    uri_array = ImageURIArray.from_uris(png_uris)
    encoded_image_array = uri_array.read_uris()
    tensor_image_array = encoded_image_array.to_tensor()

    tbl = pa.Table.from_arrays(
        [uri_array, encoded_image_array, tensor_image_array],
        ["uri_array", "encoded_image_array", "tensor_image_array"],
    )
    lance.write_dataset(tbl, tmp_path)
    tbl2 = lance.dataset(tmp_path)
    indices = list(range(len(png_uris)))

    assert tbl.take(indices).to_pylist() == tbl2.take(indices).to_pylist()
    tensor_image_array_2 = tbl2.take(indices).column(2)

    assert tensor_image_array_2.type == tensor_image_array.type
