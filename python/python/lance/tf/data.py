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
from typing import Callable, List, Optional, Union

import lance
import tensorflow as tf
from lance import LanceDataset as _LanceDataset
from lance.fragment import LanceFragment
from tensorflow.python.data.ops import dataset_ops, readers
from tensorflow.python.framework import dtypes, tensor_spec
from tensorflow.python.ops import gen_dataset_ops


def from_lance(uri: Union[str, Path]) -> tf.data.Dataset:
    """Create a `tf.data.Dataset` from a Lance."""
    return LanceDataset(uri)


def from_fragments(
    dataset: _LanceDataset, fragments: tf.data.Dataset, name: str = None
) -> tf.data.Dataset:
    """Create a `tf.data.Dataset` from a Lance."""
    ids = fragments.as_numpy_iterator()
    print(list(ids))
    # return fragments.map(read_lance)
    return LanceDataset(dataset.uri, fragments=ids)


class LanceFragmentDataset(dataset_ops.DatasetSource):
    def __init__(self, dataset: _LanceDataset):
        self._dataset = dataset
        self._impl = tf.data.Dataset.from_tensor_slices(
            [f.fragment_id for f in dataset.get_fragment()]
        )
        super().__init__(self._impl.element_spec)

    def __repr__(self):
        return "LanceFragmentDataset"

    def __iter__(self):
        return iter(self._impl)


class LanceDataset(dataset_ops.DatasetSource):
    """Lance Tensorflow Dataset Source.

    .. warning::

        Experimental feature. API stability is not guaranteed.

    .. code:: python

        import tensorflow as tf
        import lance.tf.data

        class DistributedSelector:
            def __init__(self, world_size, rank):
                self.world_size = world_size
                self.rank = rank

            def __call__(self, fragments):
                return fragments[self.rank::self.world_size]

        ds = lance.tf.data.from_lance("s3://bucket/path/dataset.lance")
            .select(["image", "label"])
            .filter("val > 10.5")
            .fragments(DistributedSelector(world_size=10, rank=5))
            .map(lambda x: x["a"] + x["b"])

    """

    def __init__(self, uri: str, fragments: Optional[List[int]] = None):
        self._dataset: _LanceDataset = lance.dataset(uri)
        self._fragments: List[int] = None
        self._select: Optional[List[str]] = None

        var_tensor = tf.constant(self, shape=(), dtype=tf.variant, name="lance_dataset")
        print("Dataset var tensor: ", var_tensor)
        super().__init__(variant_tensor=var_tensor)

    @staticmethod
    def from_fragments(fragments: tf.data.Dataset) -> tf.data.Dataset:
        return LanceDataset()

    def fragments(
        self,
        selector: Optional[Callable[[List[LanceFragment]], List[LanceFragment]]] = None,
    ) -> tf.data.Dataset:
        """Only read the specified fragments."""
        fragments = self._dataset.get_fragments()
        return tf.data.Dataset.from_tensor_slices([str(f.fragment_id) for f in fragments])

    def select(self, columns: List[str]) -> LanceDataset:
        """Select only columns to read, if not called, all columns are read."""
        pass

    def filter(self, predicate: Union[str, Callable], name: Optional[str] = None):
        """Filter this dataset according to predicate.

        If the predicate is string type, it is interpreted as a SQL expression, and
        be pushed down into the Lance storage.

        See `SQL filter push-down <https://lancedb.github.io/lance/read_and_write.html#filter-push-down>`_
        in more details.

        Parameters
        ----------
        predicate : str or callable
            If str, it is interpreted as a SQL expression. If callable, it is
            pass to regular `tf.data` pipeline.
        """
        if isinstance(predicate, str):
            self._dataset = self._dataset.filter(predicate)
            return self

        return super().filter(predicate, name)

    @property
    def element_spec(self):
        return tf.TensorSpec(shape=(), dtype=tf.int32)
