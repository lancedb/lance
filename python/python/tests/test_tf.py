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
