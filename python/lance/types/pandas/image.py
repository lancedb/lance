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
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas._libs.arrays import NDArrayBacked
from pandas.api.extensions import ExtensionDtype
from pandas.core.arrays import ExtensionArray, PandasArray
from pandas.core.construction import extract_array
from pandas.core.dtypes.common import pandas_dtype

from lance.types import Image
from lance.types.image import ImageUri


@pd.api.extensions.register_extension_dtype
class ImageUriDtype(ExtensionDtype):
    name = 'image[uri]'
    type = Image
    kind = 'O'
    na_value = pd.NA

    @classmethod
    def construct_from_string(cls, string):
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )

        if string == cls.name:
            return cls()
        else:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")

    @classmethod
    def construct_array_type(cls):
        return ImageArray

    def __from_arrow__(self, array: Union[pa.Array, pa.ChunkedArray]) -> ExtensionArray:
        return ImageArray(array.to_numpy())


def _convert_value(x):
    if pd.isna(x):
        return pd.NA
    if isinstance(x, ImageUri):
        return x.uri
    if isinstance(x, str):
        return x
    if isinstance(x, Path):
        return str(x)
    raise TypeError("ImageArray only understands NA, Image, and str scalars")


class ImageArray(PandasArray):
    _typ = "extension"
    __array_priority__ = 1000
    _dtype = ImageUriDtype()
    ndim = 1
    can_hold_na = True

    def __arrow_array__(self, type=None):
        from lance.types.image import ImageArray as ArrowImageArray
        return ArrowImageArray.from_images(self)

    def __init__(self, values, copy=False):
        values = extract_array(values)
        if copy:
            values = values.copy()
        super().__init__(values, copy=copy)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        if dtype and not (isinstance(dtype, str) and dtype == ImageUriDtype.name):
            dtype = pandas_dtype(dtype)
            assert isinstance(dtype, ImageUriDtype)

        from pandas.core.arrays.masked import BaseMaskedArray

        if isinstance(scalars, BaseMaskedArray):
            # avoid costly conversion to object dtype
            values = scalars._data
            values = pd._libs.lib.ensure_string_array(values, copy=copy, convert_na_value=False)
            values[scalars._mask] = pd.NA
        else:
            # convert non-na-likes to str, and nan-likes to StringDtype().na_value
            scalars = [_convert_value(x) for x in scalars]
            values = pd._libs.lib.ensure_string_array(scalars, na_value=pd.NA, copy=copy)

        # Manually creating new array avoids the validation step in the __init__, so is
        # faster. Refactor need for validation?
        return cls(values, copy=copy)

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype=None, copy=False
    ):
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    def _putmask(self, mask, value) -> None:
        # the super() method NDArrayBackedExtensionArray._putmask uses
        # np.putmask which doesn't properly handle None/pd.NA, so using the
        # base class implementation that uses __setitem__
        ExtensionArray._putmask(self, mask, value)

    def _where(self, mask, value):
        is_scalar = self._is_scalar_value(value)
        if is_scalar:
            value = _convert_value(value)
        return super()._where(mask, value)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def na_value(self):
        return self.dtype.na_value

    @property
    def nbytes(self):
        return self._ndarray.nbytes

    # -------------------------------------------------------------------------
    # Interfaces
    # -------------------------------------------------------------------------

    def __setitem__(self, key, value):
        # convert from Image and convert iterable of image
        value = extract_array(value, extract_numpy=True)
        if isinstance(value, type(self)):
            # extract_array doesn't extract PandasArray subclasses
            value = value._ndarray

        key = pd.core.indexers.check_array_indexer(self, key)
        scalar_key = pd._libs.lib.is_scalar(key)
        scalar_value = self._is_scalar_value(value)
        if scalar_key and not scalar_value:
            raise ValueError("setting an array element with a sequence.")

        # validate new items
        if scalar_value:
            value = _convert_value(value)
        else:
            value = [_convert_value(x) for x in value]
            if not pd.core.dtypes.api.is_array_like(value):
                value = np.asarray(value, dtype=object)
            value[pd.isna(value)] = pd.NA

        super().__setitem__(key, value)

    @staticmethod
    def _is_scalar_value(value):
        return (pd.core.dtypes.api.is_scalar(value) or
                isinstance(value, Image))

    def __getitem__(self, key):
        if pd.core.dtypes.api.is_integer(key):
            # fast-path
            result = self._ndarray[key]
            if self.ndim == 1:
                return self._box_func(result)
            return self._from_backing_data(result)

        # error: Incompatible types in assignment (expression has type "ExtensionArray",
        # variable has type "Union[int, slice, ndarray]")
        key = extract_array(key, extract_numpy=True)  # type: ignore[assignment]
        key = pd.core.indexers.check_array_indexer(self, key)
        result = self._ndarray[key]
        if self._is_scalar_value(result):
            return self._box_func(result)

        result = self._from_backing_data(result)
        return result

    def __iter__(self):
        return iter(self.to_images())

    def __contains__(self, item: object) -> bool | np.bool_:
        """
        Return for `item in self`.
        """
        # GH37867
        # comparisons of any item to pd.NA always return pd.NA, so e.g. "a" in [pd.NA]
        # would raise a TypeError. The implementation below works around that.
        is_scalar = pd._libs.lib.is_scalar(item) or isinstance(item, Image)
        if is_scalar:
            if pd.isna(item):
                if not self._can_hold_na:
                    return False
                elif item is self.dtype.na_value or isinstance(item, self.dtype.type):
                    return self._hasna
                else:
                    return False
            else:
                return (self == item).any()
        else:
            # error: Item "ExtensionArray" of "Union[ExtensionArray, ndarray]" has no
            # attribute "any"
            return (item == self).any()

    # ------------------------------------------------------------------------
    # Serializaiton / Export
    # ------------------------------------------------------------------------

    def to_images(self):
        return [self._box_scalar(v) for v in self._ndarray]

    @staticmethod
    def _box_scalar(scalar):
        return pd.NA if pd.isna(scalar) else Image.create(scalar)

    def _box_func(self, scalar):
        return self._box_scalar(scalar)

    def _cmp_method(self, other, op):
        from pandas.arrays import BooleanArray

        is_other_scalar = pd._libs.lib.is_scalar(other) or isinstance(other, Image)

        if is_other_scalar:
            other = _convert_value(other)
        elif isinstance(other, ImageArray):
            other = other._ndarray

        mask = pd.isna(self) | pd.isna(other)
        valid = ~mask

        if not is_other_scalar:
            if len(other) != len(self):
                # prevent improper broadcasting when other is 2D
                raise ValueError(
                    f"Lengths of operands do not match: {len(self)} != {len(other)}"
                )

            other = np.asarray([_convert_value(x) for x in other])
            other = other[valid]

        if op.__name__ in pd.core.ops.ARITHMETIC_BINOPS:
            result = np.empty_like(self._ndarray, dtype="object")
            result[mask] = self.na_value
            result[valid] = op(self._ndarray[valid], other)
            return ImageArray(result)
        else:
            # logical
            result = np.zeros(len(self._ndarray), dtype="bool")
            result[valid] = op(self._ndarray[valid], other)
            return BooleanArray(result, mask)

    _arith_method = _cmp_method

    def take(self, indices, allow_fill=False,
             fill_value=None, axis=0):
        if allow_fill:
            fill_value = _convert_value(fill_value)
        return super().take(indices, allow_fill=allow_fill,
                            fill_value=fill_value, axis=axis)

    def _reduce(
        self, name, *, skipna=True, axis=0, **kwargs
    ):
        if name in ["min", "max"]:
            return getattr(self, name)(skipna=skipna, axis=axis)

        raise TypeError(f"Cannot perform reduction '{name}' with string dtype")

    def _validate_scalar(self, value):
        return _convert_value(value)

    def _validate_searchsorted_value(self, value):
        if self._is_scalar_value(value):
            return self._validate_scalar(value)
        if isinstance(value, ImageArray):
            return value._ndarray
        elif isinstance(value, ExtensionArray):
            return value.to_numpy().astype(str)
        elif isinstance(value, np.ndarray):
            return value.astype(str)
        else:
            return [self._validate_scalar(x) for x in value]
