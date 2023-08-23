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

import pandas as pd
import numpy as np
import pyarrow as pa
import pytest

try:
    import tensorflow as tf  # noqa: F401
except ImportError:
    pytest.skip(
        "Tensorflow is not installed. Please install tensorflow to "
        + "test lance.tf module.",
        allow_module_level=True,
    )

import lance
from lance.fragment import LanceFragment
from lance.tf.data import from_lance, lance_fragments
from lance.tf.tfrecord import infer_tfrecord_schema, read_tfrecord


@pytest.fixture
def tf_dataset(tmp_path):
    df = pd.DataFrame(
        {
            "a": range(10000),
            "s": [f"val-{i}" for i in range(10000)],
            "vec": [[i * 0.2] * 128 for i in range(10000)],
        }
    )

    schema = pa.schema(
        [
            pa.field("a", pa.int64()),
            pa.field("s", pa.string()),
            pa.field("vec", pa.list_(pa.float32(), 128)),
        ]
    )
    tbl = pa.Table.from_pandas(df, schema=schema)
    uri = tmp_path / "dataset.lance"
    lance.write_dataset(
        tbl,
        uri,
        schema=tbl.schema,
        max_rows_per_group=100,
        max_rows_per_file=1000,
    )
    return uri


def test_fragment_dataset(tf_dataset):
    ds = from_lance(tf_dataset, batch_size=100)
    for idx, batch in enumerate(ds):
        assert batch["a"].numpy()[0] == idx * 100
        assert batch["s"].numpy()[0] == f"val-{idx * 100}".encode("utf-8")
        assert batch["a"].shape == (100,)
        assert batch["vec"].shape == (
            100,
            128,
        )  # Fixed size list


def test_projection(tf_dataset):
    ds = from_lance(tf_dataset, batch_size=100, columns=["a"])

    for idx, batch in enumerate(ds):
        assert list(batch.keys()) == ["a"]
        assert batch["a"].numpy()[0] == idx * 100
        assert batch["a"].shape == (100,)


def test_scan_use_tf_data(tf_dataset):
    ds = tf.data.Dataset.from_lance(tf_dataset)
    for idx, batch in enumerate(ds):
        assert batch["a"].numpy()[0] == idx * 100
        assert batch["s"].numpy()[0] == f"val-{idx * 100}".encode("utf-8")
        assert batch["a"].shape == (100,)
        assert batch["vec"].shape == (
            100,
            128,
        )  # Fixed size list


def test_shuffle(tf_dataset):
    fragments = lance_fragments(tf_dataset).shuffle(4, seed=20).take(3)

    ds = from_lance(tf_dataset, fragments=fragments, batch_size=100)
    raw_ds = lance.dataset(tf_dataset)
    scanner = raw_ds.scanner(
        fragments=[LanceFragment(raw_ds, fid) for fid in [0, 3, 1]], batch_size=100
    )

    for batch, raw_batch in zip(ds, scanner.to_batches()):
        assert batch["a"].numpy()[0] == raw_batch.to_pydict()["a"][0]
        assert batch["a"].numpy().shape == (100,)
        assert batch["vec"].shape == (
            100,
            128,
        )  # Fixed size list


def test_var_length_list(tmp_path):
    """Treat var length list as RaggedTensor."""
    df = pd.DataFrame(
        {
            "a": range(200),
            "l": [[i] * (i % 5 + 1) for i in range(200)],
        }
    )

    schema = pa.schema(
        [
            pa.field("a", pa.int64()),
            pa.field("l", pa.list_(pa.int32())),
        ]
    )
    tbl = pa.Table.from_pandas(df, schema=schema)

    uri = tmp_path / "dataset.lance"
    lance.write_dataset(
        tbl,
        uri,
        schema=tbl.schema,
    )

    output_signature = {
        "a": tf.TensorSpec(shape=(None,), dtype=tf.int64),
        "l": tf.RaggedTensorSpec(dtype=tf.dtypes.int32, shape=(8, None), ragged_rank=1),
    }

    ds = tf.data.Dataset.from_lance(
        uri,
        batch_size=8,
        output_signature=output_signature,
    )
    for idx, batch in enumerate(ds):
        assert batch["a"].numpy()[0] == idx * 8
        assert batch["l"].shape == (8, None)
        assert isinstance(batch["l"], tf.RaggedTensor)


def test_tfrecord_parsing(tmp_path):
    # Create a TFRecord with a string, float, int, and a tensor
    tensor = tf.constant(np.array([list(range(3)), list(range(3, 6))]))

    feature = {
        'int': tf.train.Feature(int64_list = tf.train.Int64List(value=[1])),
        'int_list': tf.train.Feature(int64_list = tf.train.Int64List(value=[1, 2, 3])),
        'float': tf.train.Feature(float_list = tf.train.FloatList(value=[1.0])),
        'float_list': tf.train.Feature(float_list = tf.train.FloatList(value=[1.0, 2.0, 3.0])),
        'bytes': tf.train.Feature(bytes_list = tf.train.BytesList(value=[b'Hello, TensorFlow!'])),
        'bytes_list': tf.train.Feature(bytes_list = tf.train.BytesList(value=[b'Hello, TensorFlow!', b'Hello, Lance!'])),
        'string': tf.train.Feature(bytes_list = tf.train.BytesList(value=[b'Hello, TensorFlow!'])),
        'tensor': tf.train.Feature(bytes_list = tf.train.BytesList(value=[tf.io.serialize_tensor(tensor).numpy()])),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized = example_proto.SerializeToString()

    path = tmp_path / 'test.tfrecord'
    with tf.io.TFRecordWriter(str(path)) as writer:
        writer.write(serialized)

    inferred_schema = infer_tfrecord_schema(str(path))
    
    assert inferred_schema == pa.schema({
        'int': pa.int64(),
        'int_list': pa.list_(pa.int64()),
        'float': pa.float32(),
        'float_list': pa.list_(pa.float32()),
        'bytes': pa.binary(),
        'bytes_list': pa.list_(pa.binary()),
        # Will all be binary since we didn't specify which ones are tensors or strings
        'string': pa.binary(),
        'tensor': pa.binary()
    })

    inferred_schema = infer_tfrecord_schema(
        str(path),
        tensor_features=['tensor'],
        string_features=['string'],
    )
    assert inferred_schema == pa.schema({
        'int': pa.int64(),
        'int_list': pa.list_(pa.int64()),
        'float': pa.float32(),
        'float_list': pa.list_(pa.float32()),
        'bytes': pa.binary(),
        'bytes_list': pa.list_(pa.binary()),
        'string': pa.string(),
        'tensor': pa.fixed_shape_tensor(pa.int32(), [3, 2]),
    })

    batch = read_tfrecord(str(path), inferred_schema)
    assert batch.num_rows == 1
    assert batch.num_columns == 3
    assert batch.column_names == ['bytes', 'string', 'tensor']
    assert batch['bytes'].to_pylist() == [b'Hello, TensorFlow!']
    assert batch['string'].to_pylist() == ['Hello, TensorFlow!']
    assert batch['tensor'].to_pylist() == [tensor.numpy().tolist()]
