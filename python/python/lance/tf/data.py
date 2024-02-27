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
from functools import partial
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union

import pyarrow as pa

import lance
from lance import LanceDataset
from lance.arrow import EncodedImageType, FixedShapeImageTensorType, ImageURIType
from lance.dependencies import _check_for_numpy
from lance.dependencies import numpy as np
from lance.dependencies import tensorflow as tf
from lance.fragment import FragmentMetadata, LanceFragment

if TYPE_CHECKING:
    from pathlib import Path


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
    elif isinstance(dt, (EncodedImageType, ImageURIType)):
        return tf.TensorSpec(shape=(None,), dtype=tf.string)
    elif isinstance(dt, FixedShapeImageTensorType):
        return tf.TensorSpec(
            shape=(None, *dt.shape), dtype=arrow_data_type_to_tf(dt.arrow_type)
        )

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
    elif isinstance(array.type, FixedShapeImageTensorType):
        return tf.constant(array.to_numpy(), dtype=tensor_spec.dtype)
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
    columns: Optional[Union[List[str], Dict[str, str]]] = None,
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
    elif _check_for_numpy(fragments) and isinstance(fragments, np.ndarray):
        fragments = list(fragments)

    if fragments is not None:

        def gen_fragments(fragments):
            for f in fragments:
                if isinstance(f, int) or (
                    _check_for_numpy(f) and isinstance(f, np.integer)
                ):
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
    return tf.data.Dataset.from_tensor_slices([
        f.fragment_id for f in dataset.get_fragments()
    ])


def _ith_batch(i: int, batch_size: int, total_size: int) -> Tuple[int, int]:
    """
    Get the start and end index of the ith batch.

    This takes into account the total_size, the total number of rows in the dataset.
    """
    start = i * batch_size
    end = tf.math.minimum(start + batch_size, total_size)
    return (start, end)


def from_lance_batches(
    dataset: Union[str, Path, LanceDataset],
    *,
    shuffle: bool = False,
    seed: Optional[int] = None,
    batch_size: int = 1024,
    skip: int = 0,
) -> tf.data.Dataset:
    """
    Create a ``tf.data.Dataset`` of batch indices for a Lance dataset.

    Parameters
    ----------
    dataset : Union[str, Path, LanceDataset]
        A Lance Dataset or dataset URI/path.
    shuffle : bool, optional
        Shuffle the batches, by default False
    seed : Optional[int], optional
        Random seed for shuffling, by default None
    batch_size : int, optional
        Batch size, by default 1024
    skip : int, optional
        Number of batches to skip.

    Returns
    -------
    tf.data.Dataset
        A tensorflow dataset of batch slice ranges. These can be passed to
        :func:`lance_take_batches` to create a Tensorflow dataset of batches.
    """
    if not isinstance(dataset, LanceDataset):
        dataset = lance.dataset(dataset)
    num_rows = dataset.count_rows()
    num_batches = (num_rows + batch_size - 1) // batch_size
    indices = tf.data.Dataset.range(num_batches, dtype=tf.int64)
    if shuffle:
        indices = indices.shuffle(num_batches, seed=seed)
    if skip > 0:
        indices = indices.skip(skip)
    return indices.map(partial(_ith_batch, batch_size=batch_size, total_size=num_rows))


def lance_take_batches(
    dataset: Union[str, Path, LanceDataset],
    batch_ranges: Iterable[Tuple[int, int]],
    *,
    columns: Optional[Union[List[str], Dict[str, str]]] = None,
    output_signature: Optional[Dict[str, tf.TypeSpec]] = None,
    batch_readahead: int = 10,
) -> tf.data.Dataset:
    """
    Create a ``tf.data.Dataset`` of batches from a Lance dataset.

    Parameters
    ----------
    dataset : Union[str, Path, LanceDataset]
        A Lance Dataset or dataset URI/path.
    batch_ranges : Iterable[Tuple[int, int]]
        Iterable of batch indices.
    columns : Optional[List[str]], optional
        List of columns to include in the output dataset.
        If not set, all columns will be read.
    output_signature : Optional[tf.TypeSpec], optional
        Override output signature of the returned tensors. If not provided,
        the output signature is inferred from the projection Schema.
    batch_readahead : int, default 10
        The number of batches to read ahead in parallel.

    Examples
    --------
    You can compose this with ``from_lance_batches`` to create a randomized Tensorflow
    dataset. With ``from_lance_batches``, you can deterministically randomized the
    batches by setting ``seed``.

    .. code-block:: python

        batch_iter = from_lance_batches(dataset, batch_size=100, shuffle=True, seed=200)
        batch_iter = batch_iter.as_numpy_iterator()
        lance_ds = lance_take_batches(dataset, batch_iter)
        lance_ds = lance_ds.unbatch().shuffle(500, seed=42).batch(100)
    """
    if not isinstance(dataset, LanceDataset):
        dataset = lance.dataset(dataset)

    if output_signature is None:
        schema = dataset.scanner(columns=columns).projected_schema
        output_signature = schema_to_spec(schema)
    logging.debug("Output signature: %s", output_signature)

    def gen_ranges():
        for start, end in batch_ranges:
            yield (start, end)

    def gen_batches():
        batches = dataset._ds.take_scan(
            gen_ranges(),
            columns=columns,
            batch_readahead=batch_readahead,
        )
        for batch in batches:
            yield {
                name: column_to_tensor(batch[name], output_signature[name])
                for name in batch.schema.names
            }

    return tf.data.Dataset.from_generator(
        gen_batches, output_signature=output_signature
    )


# Register `from_lance` to ``tf.data.Dataset``.
tf.data.Dataset.from_lance = from_lance
tf.data.Dataset.from_lance_batches = from_lance_batches
