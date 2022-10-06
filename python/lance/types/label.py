#  Copyright (c) 2022. Lance Developers
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
from typing import Sequence

import pandas as pd
import pyarrow as pa

from lance.types.base import LanceType


class LabelType(LanceType):
    """
    A label used for classification. This is backed by a dictionary type
    to make it easier for translating between human-readable strings and
    integer classes used in the models
    """

    def __init__(self):
        super(LabelType, self).__init__(pa.dictionary(pa.int8(), pa.string()), "label")

    def __arrow_ext_class__(self):
        return LabelArray

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return LabelType()


class LabelArray(pa.ExtensionArray):
    @property
    def names(self):
        """Return all possible class names as a numpy array"""
        return self.storage.dictionary.to_numpy()

    @property
    def ids(self):
        """Return as Label IDs (int)."""
        return self.storage.indices.to_numpy()

    @staticmethod
    def from_values(values: Sequence[str], dictionary: Sequence[str]) -> "LabelArray":
        """
        Create a `LabelArray` using the given values

        Examples
        --------

        .. code-block:: python

            In [3]: LabelArray.from_values(["dog", "cat"], dictionary=["horse", "cat", "dog"])
            Out[3]:
            <pyarrow.lib.ExtensionArray object at 0x7fc018cd9780>

            -- dictionary:
            [
                "horse",
                "cat",
                "dog"
            ]
            -- indices:
            [
                2,
                1
            ]
        """
        if isinstance(values, pa.Array):
            values = values.to_numpy(False)
        cat = pd.Categorical(values, categories=dictionary)
        storage = pa.DictionaryArray.from_arrays(
            cat.codes, dictionary, from_pandas=True
        )
        return LabelArray.from_storage(LabelType(), storage)
