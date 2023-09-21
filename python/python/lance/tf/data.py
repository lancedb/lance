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


"""Tensorflow Dataset (`tf.data <https://www.tensorflow.org/guide/data>`_)
implementation for Lance.

.. warning::

    Experimental feature. API stability is not guaranteed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import lance
import numpy as np
import pyarrow as pa
import tensorflow as tf
from lance import LanceDataset
from lance.fragment import FragmentMetadata, LanceFragment


def arrow_data_type_to_tf(dt: pa.DataType) -> tf.DType:
    """Convert Pyarrow DataType to Tensorflow."""
    if pa.types.is_boolean(dt):
        return tf.bool
    elif pa.types.is_int8(dt):
        return tf.int8
    elif pa.types.is_int16(dt):
        return tf.int16
    elif pa.types.is_int32(dt):
        return tf.int32
    elif pa.types.is_int64(dt):
        return tf.int64
    elif pa.types.is_uint8(dt):
        return tf.uint8
    elif pa.types.is_uint16(dt):
        return tf.uint16
    elif pa.types.is_uint32(dt):
        return tf.uint32
    elif pa.types.is_uint64(dt):
        return tf.uint64
    elif pa.types.is_float16(dt):
        return tf.float16
    elif pa.types.is_float32(dt):
        return tf.float32
    elif pa.types.is_float64(dt):
        return tf.float64
    elif (
        pa.types.is_string(dt)
        or pa.types.is_large_string(dt)
        or pa.types.is_binary(dt)
        or pa.types.is_large_binary(dt)
    ):
        return tf.string

    raise TypeError(f"Arrow/Tf conversion: Unsupported arrow data type: {dt}")


def data_type_to_tensor_spec(dt: pa.DataType) -> tf.TensorSpec:
    """Convert PyArrow DataType to Tensorflow TensorSpec."""
    if (
        pa.types.is_boolean(dt)
        or pa.types.is_integer(dt)
        or pa.types.is_floating(dt)
        or pa.types.is_string(dt)
        or pa.types.is_binary(dt)
    ):
        return tf.TensorSpec(shape=(None,), dtype=arrow_data_type_to_tf(dt))
    elif isinstance(dt, pa.FixedShapeTensorType):
        return tf.TensorSpec(
            shape=(None, *dt.shape), dtype=arrow_data_type_to_tf(dt.value_type)
        )
    elif pa.types.is_fixed_size_list(dt):
        return tf.TensorSpec(
            shape=(None, dt.list_size), dtype=arrow_data_type_to_tf(dt.value_type)
        )
    elif pa.types.is_list(dt) or pa.types.is_large_list(dt):
        return tf.TensorSpec(
            shape=(
                None,
                None,
            ),
            dtype=arrow_data_type_to_tf(dt.value_type),
        )
    elif pa.types.is_struct(dt):
        return {field.name: data_type_to_tensor_spec(field.type) for field in dt}

    raise TypeError("Unsupported data type: ", dt)


def schema_to_spec(schema: pa.Schema) -> tf.TypeSpec:
    """Convert PyArrow Schema to Tensorflow output signature."""
    signature = {}
    for name in schema.names:
        field = schema.field(name)
        signature[name] = data_type_to_tensor_spec(field.type)
    return signature


def column_to_tensor(array: pa.Array, tensor_spec: tf.TensorSpec) -> tf.Tensor:
    """Convert a PyArrow array into a TensorFlow tensor."""
    if isinstance(tensor_spec, tf.RaggedTensorSpec):
        return tf.ragged.constant(array.to_pylist(), dtype=tensor_spec.dtype)
    elif isinstance(array.type, pa.FixedShapeTensorType):
        return tf.constant(array.to_numpy_ndarray(), dtype=tensor_spec.dtype)
    elif isinstance(array.type, pa.StructType):
        return {
            field.name: column_to_tensor(array.field(i), tensor_spec[field.name])
            for (i, field) in enumerate(array.type)
        }
    else:
        return tf.constant(array.to_pylist(), dtype=tensor_spec.dtype)


def from_lance(
    dataset: Union[str, Path, LanceDataset],
    *,
    columns: Optional[List[str]] = None,
    batch_size: int = 256,
    filter: Optional[str] = None,
    fragments: Union[Iterable[int], Iterable[LanceFragment], tf.data.Dataset] = None,
    output_signature: Optional[Dict[str, tf.TypeSpec]] = None,
) -> tf.data.Dataset:
    """Create a ``tf.data.Dataset`` from a Lance dataset.

    Parameters
    ----------
    dataset : Union[str, Path, LanceDataset]
        Lance dataset or dataset URI/path.
    columns : Optional[List[str]], optional
        List of columns to include in the output dataset.
        If not set, all columns will be read.
    batch_size : int, optional
        Batch size, by default 256
    filter : Optional[str], optional
        SQL filter expression, by default None.
    fragments : Union[List[LanceFragment], tf.data.Dataset], optional
        If provided, only the fragments are read. It can be used to feed
        for distributed training.
    output_signature : Optional[tf.TypeSpec], optional
        Override output signature of the returned tensors. If not provided,
        the output signature is inferred from the projection Schema.

    Examples
    --------

    .. code-block:: python

        import tensorflow as tf
        import lance.tf.data

        ds = lance.tf.data.from_lance(
            "s3://bucket/path",
            columns=["image", "id"],
            filter="catalog = 'train' AND split = 'train'",
            batch_size=100)

        for batch in ds.repeat(10).shuffle(128).map(io_decode):
            print(batch["image"].shape)

    ``from_lance`` can take an iterator or ``tf.data.Dataset`` of
    Fragments. So that it can be used to feed for distributed training.

    .. code-block:: python

        import tensorflow as tf
        import lance.tf.data

        seed = 200  # seed to shuffle the fragments in distributed machines.
        fragments = lance.tf.data.lance_fragments("s3://bucket/path")
            repeat(10).shuffle(4, seed=seed)
        ds = lance.tf.data.from_lance(
            "s3://bucket/path",
            columns=["image", "id"],
            filter="catalog = 'train' AND split = 'train'",
            fragments=fragments,
            batch_size=100)
        for batch in ds.shuffle(128).map(io_decode):
            print(batch["image"].shape)

    """
    if not isinstance(dataset, LanceDataset):
        dataset = lance.dataset(dataset)

    if isinstance(fragments, tf.data.Dataset):
        fragments = list(fragments.as_numpy_iterator())
    elif isinstance(fragments, np.ndarray):
        fragments = list(fragments)

    if fragments is not None:

        def gen_fragments(fragments):
            for f in fragments:
                if isinstance(f, (int, np.integer)):
                    yield LanceFragment(dataset, int(f))
                elif isinstance(f, FragmentMetadata):
                    yield LanceFragment(dataset, f.fragment_id)
                elif isinstance(f, LanceFragment):
                    yield f
                else:
                    raise TypeError(f"Invalid type passed to `fragments`: {type(f)}")

        # A Generator of Fragments
        fragments = gen_fragments(fragments)

    scanner = dataset.scanner(
        filter=filter, columns=columns, batch_size=batch_size, fragments=fragments
    )

    if output_signature is None:
        schema = scanner.projected_schema
        output_signature = schema_to_spec(schema)
    logging.debug("Output signature: %s", output_signature)

    def generator():
        for batch in scanner.to_batches():
            yield {
                name: column_to_tensor(batch[name], output_signature[name])
                for name in batch.schema.names
            }

    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)


def lance_fragments(dataset: Union[str, Path, LanceDataset]) -> tf.data.Dataset:
    """Create a ``tf.data.Dataset`` of Lance Fragments in the dataset.

    Parameters
    ----------
    dataset : Union[str, Path, LanceDataset]
        A Lance Dataset or dataset URI/path.
    """
    if not isinstance(dataset, LanceDataset):
        dataset = lance.dataset(dataset)
    return tf.data.Dataset.from_tensor_slices(
        [f.fragment_id for f in dataset.get_fragments()]
    )


# Register `from_lance` to ``tf.data.Dataset``.
tf.data.Dataset.from_lance = from_lance
