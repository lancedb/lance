# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import os
import warnings

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from lance.arrow import ImageArray
from lance.fragment import LanceFragment

pytest.skip("Skip tensorflow tests", allow_module_level=True)

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import tensorflow as tf  # noqa: F401
except ImportError:
    pytest.skip(
        "Tensorflow is not installed. Please install tensorflow to "
        + "test lance.tf module.",
        allow_module_level=True,
    )

from lance.tf.data import (  # noqa: E402
    from_lance,
    from_lance_batches,
    lance_fragments,
    lance_take_batches,
)


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


def test_filter(tf_dataset):
    ds = from_lance(tf_dataset, batch_size=100, filter="a >= 5000")

    for idx, batch in enumerate(ds):
        assert batch["a"].numpy()[0] == idx * 100 + 5000
        assert batch["a"].shape == (100,)


def test_namespace_table_id(monkeypatch):
    calls = {}

    class DummyScanner:
        def __init__(self):
            self._batch = pa.record_batch([pa.array([1, 2])], names=["a"])
            self.projected_schema = self._batch.schema

        def to_batches(self):
            yield self._batch

    class DummyDataset:
        def scanner(self, **kwargs):
            return DummyScanner()

    def fake_dataset(uri=None, **kwargs):
        calls["uri"] = uri
        calls["kwargs"] = kwargs
        return DummyDataset()

    monkeypatch.setattr(lance, "dataset", fake_dataset)

    ns = object()
    ds = from_lance(
        None,
        namespace=ns,
        table_id=["tbl"],
        ignore_namespace_table_storage_options=True,
    )

    assert calls["kwargs"]["namespace"] is ns
    assert calls["kwargs"]["table_id"] == ["tbl"]
    assert calls["kwargs"]["ignore_namespace_table_storage_options"] is True

    batches = list(ds)
    assert [b["a"].numpy().tolist() for b in batches] == [[1, 2]]


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


def test_pass_fragments(tf_dataset):
    # Can pass fragments directly to from_lance
    dataset = lance.dataset(tf_dataset)
    ds = from_lance(tf_dataset, fragments=dataset.get_fragments(), batch_size=100)
    ds_default = from_lance(tf_dataset, batch_size=100)
    for batch, batch_default in zip(ds, ds_default):
        assert batch["a"].numpy()[0] == batch_default["a"].numpy()[0]
        assert batch["a"].numpy().shape == (100,)
        assert batch["vec"].shape == (
            100,
            128,
        )

    # Can pass ids directly to from_lance
    ds = from_lance(tf_dataset, fragments=[0, 1, 2], batch_size=100)
    for idx, batch in enumerate(ds):
        assert batch["a"].numpy()[0] == idx * 100
        assert batch["a"].numpy().shape == (100,)


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


def test_dataset_batches(tf_dataset):
    tf_dataset = lance.dataset(tf_dataset)
    batch_size = 300
    batches = list(
        from_lance_batches(tf_dataset, batch_size=batch_size).as_numpy_iterator()
    )
    assert tf_dataset.count_rows() // batch_size + 1 == len(batches)
    assert all(end - start == batch_size for start, end in batches[:-2])
    assert batches[-1][1] - batches[-1][0] == tf_dataset.count_rows() % batch_size

    skip = 5
    batches_skipped = list(
        from_lance_batches(
            tf_dataset, batch_size=batch_size, skip=skip
        ).as_numpy_iterator()
    )
    assert batches_skipped == batches[skip:]

    batches_shuffled = list(
        from_lance_batches(
            tf_dataset, batch_size=batch_size, shuffle=True, seed=42
        ).as_numpy_iterator()
    )
    # make sure it does a shuffle
    assert batches_shuffled != batches
    batches_shuffled2 = list(
        from_lance_batches(
            tf_dataset, batch_size=batch_size, shuffle=True, seed=42
        ).as_numpy_iterator()
    )
    # make sure the shuffle can be deterministic
    assert batches_shuffled == batches_shuffled2


def test_take_dataset(tf_dataset):
    tf_dataset = lance.dataset(tf_dataset)
    batch_ds = from_lance_batches(
        tf_dataset, batch_size=100, shuffle=True, seed=42
    ).as_numpy_iterator()
    lance_ds = lance_take_batches(tf_dataset, batch_ds)
    lance_ds = lance_ds.unbatch().shuffle(400, seed=42).batch(100)

    for batch in lance_ds:
        assert batch["a"].numpy().shape == (100,)

    batches = [(0, 200), (100, 200)]
    lance_ds = lance_take_batches(tf_dataset, batches, columns=["a"])
    for (start, end), batch in zip(batches, lance_ds):
        assert batch["a"].numpy().tolist() == np.arange(start, end).tolist()
        assert batch.keys() == {"a"}


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


def test_nested_struct(tmp_path):
    table = pa.table(
        {
            "x": pa.array(
                [
                    {
                        "a": 1,
                        "json": {"b": "hello", "x": b"abc"},
                    },
                    {
                        "a": 24,
                        "json": {"b": "world", "x": b"def"},
                    },
                ]
            )
        }
    )
    uri = tmp_path / "dataset.lance"
    dataset = lance.write_dataset(table, uri)

    ds = tf.data.Dataset.from_lance(
        dataset,
        batch_size=8,
    )

    for batch in ds:
        tf.debugging.assert_equal(batch["x"]["a"], tf.constant([1, 24], dtype=tf.int64))
        tf.debugging.assert_equal(
            batch["x"]["json"]["b"], tf.constant(["hello", "world"])
        )
        tf.debugging.assert_equal(
            batch["x"]["json"]["x"], tf.constant([b"abc", b"def"])
        )


def test_tensor(tmp_path):
    arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
    table = pa.table({"x": pa.FixedShapeTensorArray.from_numpy_ndarray(arr)})

    uri = tmp_path / "dataset.lance"
    dataset = lance.write_dataset(table, uri)
    ds = tf.data.Dataset.from_lance(dataset)

    for batch in ds:
        assert batch["x"].shape == (2, 2, 3)
        assert batch["x"].dtype == tf.float32
        assert batch["x"].numpy().tolist() == arr.tolist()


def test_image_types(tmp_path):
    path = [os.path.join(os.path.dirname(__file__), "images/1.png")]
    uris = ImageArray.from_array(path * 3)
    encoded_images = uris.read_uris()
    tensors = encoded_images.to_tensor()
    table = pa.table(
        {
            "uris": uris,
            "encoded_images": encoded_images,
            "tensor_images": tensors,
        }
    )

    uri = tmp_path / "dataset.lance"
    dataset = lance.write_dataset(table, uri)
    ds = tf.data.Dataset.from_lance(dataset)

    for batch in ds:
        assert batch["uris"].shape == (3,)
        assert batch["uris"].dtype == tf.string
        assert batch["uris"].numpy().astype("str").tolist() == uris.tolist()

        assert batch["encoded_images"].shape == (3,)
        assert batch["encoded_images"].dtype == tf.string
        assert batch["encoded_images"].numpy().tolist() == encoded_images.tolist()

        assert batch["tensor_images"].shape == (3, 1, 1, 4)
        assert batch["tensor_images"].dtype == tf.uint8
        assert batch["tensor_images"].numpy().tolist() == tensors.to_numpy().tolist()
