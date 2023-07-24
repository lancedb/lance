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

from pathlib import Path
from typing import List, Optional, Union
import logging

import lance
import pyarrow as pa
import tensorflow as tf
from lance import LanceDataset
from lance.fragment import LanceFragment


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
    elif pa.types.is_string(dt) or pa.types.is_binary(dt):
        return tf.string

    raise TypeError(f"Arrow/Tf conversion: Unsupported arrow data type: {dt}")


def data_type_to_tensor_spec(dt: pa.DataType) -> tf.TensorSpec:
    """Convert PyArrow DataType to Tensorflow TensorSpec."""
    if (
        pa.types.is_boolean(dt)
        or pa.types.is_integers(dt)
        or pa.types.is_floating(dt)
        or pa.types.is_string(dt)
    ):
        return tf.TensorSpec(shape=(None,), dtype=arrow_data_type_to_tf(dt))
    elif pa.types.is_list(dt):
        return tf.TensorSpec(
            shape=(
                None,
                None,
            ),
            dtype=arrow_data_type_to_tf(dt.value_type),
        )

    raise TypeError("Unsupported data type: ", dt)


def schema_to_spec(schema: pa.Schema, batch_size: int) -> tf.TypeSpec:
    """Convert PyArrow Schema to Tensorflow output signature."""
    signature = {}
    for name in schema.names:
        field = schema.field_by_name(name)
        signature[name] = data_type_to_tensor_spec(field.type, batch_size=batch_size)
    return signature


def from_lance(
    dataset: Union[str, Path, LanceDataset],
    *,
    columns: Optional[List[str]] = None,
    batch_size: int = 256,
    filter: Optional[str] = None,
    fragments: Union[List[LanceFragment], tf.data.Dataset] = None,
) -> tf.data.Dataset:
    """Create a `tf.data.Dataset` from a Lance dataset."""
    if not isinstance(dataset, LanceDataset):
        dataset = lance.dataset(dataset)
    if isinstance(fragments, tf.data.Dataset):
        fragments = list(fragments.as_numpy_iterator())
    scanner = dataset.scanner(
        filter=filter, columns=columns, batch_size=batch_size, fragments=fragments
    )

    schema = scanner.projected_schema
    signature = schema_to_spec(schema, batch_size=batch_size)
    logging.debug("Output signature: %s", signature)

    def generator():
        for batch in scanner.to_batches():
            data = batch.to_pydict()
            yield data

    return tf.data.Dataset.from_generator(generator, output_signature=signature)


def lance_fragments(data: Union[str, Path, LanceDataset]) -> tf.data.Dataset:
    """Create a `tf.data.Dataset` from a Lance fragments."""
    if not isinstance(data, LanceDataset):
        data = lance.dataset(data)
    return tf.data.Dataset.from_tensor_slices(
        [f.fragment_id for f in data.get_fragments()]
    )
