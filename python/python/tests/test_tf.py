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

import ml_dtypes
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from lance.arrow import BFloat16Type, bfloat16_array

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
    tensor = tf.constant(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
    tensor_bf16 = tf.constant(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=ml_dtypes.bfloat16)
    )

    feature = {
        "1_int": tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
        "2_int_list": tf.train.Feature(int64_list=tf.train.Int64List(value=[1, 2, 3])),
        "3_float": tf.train.Feature(float_list=tf.train.FloatList(value=[1.0])),
        "4_float_list": tf.train.Feature(
            float_list=tf.train.FloatList(value=[1.0, 2.0, 3.0])
        ),
        "5_bytes": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[b"Hello, TensorFlow!"])
        ),
        "6_bytes_list": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[b"Hello, TensorFlow!", b"Hello, Lance!"]
            )
        ),
        "7_string": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[b"Hello, TensorFlow!"])
        ),
        "8_tensor": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[tf.io.serialize_tensor(tensor).numpy()]
            )
        ),
        "9_tensor_bf16": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[tf.io.serialize_tensor(tensor_bf16).numpy()]
            )
        ),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized = example_proto.SerializeToString()

    path = tmp_path / "test.tfrecord"
    with tf.io.TFRecordWriter(str(path)) as writer:
        writer.write(serialized)

    inferred_schema = infer_tfrecord_schema(str(path))

    assert inferred_schema == pa.schema(
        {
            "1_int": pa.int64(),
            "2_int_list": pa.list_(pa.int64()),
            "3_float": pa.float32(),
            "4_float_list": pa.list_(pa.float32()),
            "5_bytes": pa.binary(),
            "6_bytes_list": pa.list_(pa.binary()),
            # tensors and strings assumed binary
            "7_string": pa.binary(),
            "8_tensor": pa.binary(),
            "9_tensor_bf16": pa.binary(),
        }
    )

    inferred_schema = infer_tfrecord_schema(
        str(path),
        tensor_features=["8_tensor", "9_tensor_bf16"],
        string_features=["7_string"],
    )
    assert inferred_schema == pa.schema(
        {
            "1_int": pa.int64(),
            "2_int_list": pa.list_(pa.int64()),
            "3_float": pa.float32(),
            "4_float_list": pa.list_(pa.float32()),
            "5_bytes": pa.binary(),
            "6_bytes_list": pa.list_(pa.binary()),
            "7_string": pa.string(),
            "8_tensor": pa.fixed_shape_tensor(pa.float32(), [2, 3]),
            "9_tensor_bf16": pa.fixed_shape_tensor(BFloat16Type(), [2, 3]),
        }
    )

    reader = read_tfrecord(str(path), inferred_schema)
    assert reader.schema == inferred_schema
    table = reader.read_all()

    assert table.schema == inferred_schema

    tensor_type = pa.fixed_shape_tensor(pa.float32(), [2, 3])
    inner = pa.array([float(x) for x in range(1, 7)], pa.float32())
    storage = pa.FixedSizeListArray.from_arrays(inner, 6)
    f32_array = pa.ExtensionArray.from_storage(tensor_type, storage)

    tensor_type = pa.fixed_shape_tensor(BFloat16Type(), [2, 3])
    bf16_array = bfloat16_array([float(x) for x in range(1, 7)])
    storage = pa.FixedSizeListArray.from_arrays(bf16_array, 6)
    bf16_array = pa.ExtensionArray.from_storage(tensor_type, storage)

    expected_data = pa.table(
        {
            "1_int": pa.array([1]),
            "2_int_list": pa.array([[1, 2, 3]]),
            "3_float": pa.array([1.0], pa.float32()),
            "4_float_list": pa.array([[1.0, 2.0, 3.0]], pa.list_(pa.float32())),
            "5_bytes": pa.array([b"Hello, TensorFlow!"]),
            "6_bytes_list": pa.array([[b"Hello, TensorFlow!", b"Hello, Lance!"]]),
            "7_string": pa.array(["Hello, TensorFlow!"]),
            "8_tensor": f32_array,
            "9_tensor_bf16": bf16_array,
        }
    )

    assert table == expected_data
