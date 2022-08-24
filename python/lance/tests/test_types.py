import platform
import pytest

import lance
import pyarrow as pa

from lance.types import *


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="Extension types only work on Linux right now"
)
def test_image(tmp_path):
    data = [f"s3://bucket/{x}.jpg" for x in ["a", "b", "c"]]
    storage = pa.StringArray.from_pandas(data)
    image_type = ImageType.from_storage(storage.type)
    _test_extension_rt(tmp_path, image_type, storage)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="Extension types only work on Linux right now"
)
def test_image_binary(tmp_path):
    data = [b"<imagebytes>" for x in ["a", "b", "c"]]
    storage = pa.StringArray.from_pandas(data)
    image_type = ImageType.from_storage(storage.type)
    _test_extension_rt(tmp_path, image_type, storage)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="Extension types only work on Linux right now"
)
def test_point(tmp_path):
    point_type = Point2dType()
    data = [(float(x), float(x)) for x in range(100)]
    storage = pa.array(data, pa.list_(pa.float64()))
    _test_extension_rt(tmp_path, point_type, storage)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="Extension types only work on Linux right now"
)
def test_box2d(tmp_path):
    box_type = Box2dType()
    data = [(float(x), float(x), float(x), float(x)) for x in range(100)]
    storage = pa.array(data, pa.list_(pa.float64()))
    _test_extension_rt(tmp_path, box_type, storage)


def _test_extension_rt(tmp_path, ext_type, storage_arr):
    arr = pa.ExtensionArray.from_storage(ext_type, storage_arr)
    table = pa.Table.from_arrays([arr], names=["ext"])
    lance.write_table(table, str(tmp_path / "test.lance"))
    table = lance.dataset(str(tmp_path / "test.lance")).to_table()
    assert table["ext"].type == ext_type
    assert table["ext"].to_pylist() == storage_arr.to_pylist()
