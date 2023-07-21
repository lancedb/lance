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

import tensorflow as tf
from lance import LanceDataset as _LanceDataset
from lance.fragment import LanceFragment
from tensorflow.python.data.ops import dataset_ops


def from_lance(uri: Union[str, Path]) -> tf.data.Dataset:
    """Create a `tf.data.Dataset` from a Lance."""
    return LanceDataset(uri)


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

    def __init__(self, uri: str):
        self._dataset: _LanceDataset = None

    def fragments(
        self, selector: Callable[[List[LanceFragment]], List[LanceFragment]]
    ) -> LanceDataset:
        """Only read the specified fragments."""
        pass

    def select(self, columns: List[str]) -> LanceDataset:
        """Select only columns to read, if not called, all columns are read."""
        pass

    def filter(self, predicate: Union[str, Callable], name=None):
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
