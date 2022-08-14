import lance
import pyarrow as pa

from lance.types import *


def test_image(tmp_path):
    image_type = ImageType('uri')
    data = [f's3://bucket/{x}.jpg' for x in ['a', 'b', 'c']]
    storage = pa.StringArray.from_pandas(data)
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


def _test_extension_rt(tmp_path, ext_type, storage_arr):
    arr = pa.ExtensionArray.from_storage(ext_type, storage_arr)
    table = pa.Table.from_arrays([arr], names=['ext'])
    lance.write_table(table, str(tmp_path/'test.lance'))
    table = lance.dataset(str(tmp_path/'test.lance')).to_table()
    assert table['ext'].type == ext_type
    assert table['ext'].to_pylist() == storage_arr.to_pylist()
