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

from abc import ABC, abstractmethod
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
from lance.types.image import ImageUri, ImageBinary


class ImageDtype(ExtensionDtype, ABC):
    kind = "O"
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


@pd.api.extensions.register_extension_dtype
class ImageUriDtype(ImageDtype):
    name = "image[uri]"
    type = ImageUri

    @classmethod
    def construct_array_type(cls):
        return ImageUriArray

    def __from_arrow__(self, array: Union[pa.Array, pa.ChunkedArray]) -> ExtensionArray:
        return ImageUriArray(array.to_numpy())


@pd.api.extensions.register_extension_dtype
class ImageBinaryDtype(ImageDtype):
    name = "image[binary]"
    type = ImageBinary

    @classmethod
    def construct_array_type(cls):
        return ImageBinaryArray

    def __from_arrow__(self, array: Union[pa.Array, pa.ChunkedArray]) -> ExtensionArray:
        return ImageBinaryArray(array.to_numpy())


class ImageArray(PandasArray, ABC):
    _typ = "extension"
    __array_priority__ = 1000
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

    def _putmask(self, mask, value) -> None:
        # the super() method NDArrayBackedExtensionArray._putmask uses
        # np.putmask which doesn't properly handle None/pd.NA, so using the
        # base class implementation that uses __setitem__
        ExtensionArray._putmask(self, mask, value)

    def _where(self, mask, value):
        is_scalar = self._is_scalar_value(value)
        if is_scalar:
            value = self._convert_value(value)
        return super()._where(mask, value)

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
            value = self._convert_value(value)
        else:
            value = [self._convert_value(x) for x in value]
            if not pd.core.dtypes.api.is_array_like(value):
                value = np.asarray(value, dtype=object)
            value[pd.isna(value)] = self.na_value

        super().__setitem__(key, value)

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

    def __contains__(self, item: object) -> bool | np.bool_:
        """
        Return for `item in self`.
        """
        # GH37867
        # comparisons of any item to pd.NA always return pd.NA, so e.g. "a" in [pd.NA]
        # would raise a TypeError. The implementation below works around that.
        is_scalar = self._is_scalar_value(item)
        if is_scalar:
            if pd.isna(item):
                if not self._can_hold_na:
                    return False
                elif item is self.na_value or self._check_scalar_type(item):
                    return self._hasna
                else:
                    return False
            else:
                return (self == item).any()
        else:
            # error: Item "ExtensionArray" of "Union[ExtensionArray, ndarray]" has no
            # attribute "any"
            return (item == self).any()

    @property
    def na_value(self):
        return self.dtype.na_value

    @property
    def nbytes(self):
        return self._ndarray.nbytes

    def _is_scalar_value(self, value):
        return pd.core.dtypes.api.is_scalar(value) or self._check_scalar_type(value)

    def _check_scalar_type(self, value):
        return isinstance(value, self.dtype.type)

    def __iter__(self):
        return iter(self.to_images())

    # ------------------------------------------------------------------------
    # Serializaiton / Export
    # ------------------------------------------------------------------------

    def to_images(self):
        return [self._box_scalar(v) for v in self._ndarray]

    def _box_scalar(self, scalar):
        return self.na_value if pd.isna(scalar) else Image.create(scalar)

    def _box_func(self, scalar):
        return self._box_scalar(scalar)

    def _cmp_method(self, other, op):
        from pandas.arrays import BooleanArray

        is_other_scalar = pd._libs.lib.is_scalar(other) or isinstance(other, Image)

        if is_other_scalar:
            other = self._convert_value(other)
        elif isinstance(other, type(self)):
            other = other._ndarray

        mask = pd.isna(self) | pd.isna(other)
        valid = ~mask

        if not is_other_scalar:
            if len(other) != len(self):
                # prevent improper broadcasting when other is 2D
                raise ValueError(
                    f"Lengths of operands do not match: {len(self)} != {len(other)}"
                )

            other = np.asarray([self._convert_value(x) for x in other])
            other = other[valid]

        if op.__name__ in pd.core.ops.ARITHMETIC_BINOPS:
            result = np.empty_like(self._ndarray, dtype="object")
            result[mask] = self.na_value
            result[valid] = op(self._ndarray[valid], other)
            klass = type(self)
            return klass(result)
        else:
            # logical
            result = np.zeros(len(self._ndarray), dtype="bool")
            result[valid] = op(self._ndarray[valid], other)
            return BooleanArray(result, mask)

    _arith_method = _cmp_method

    def take(self, indices, allow_fill=False, fill_value=None, axis=0):
        if allow_fill:
            fill_value = self._convert_value(fill_value)
        return super().take(
            indices, allow_fill=allow_fill, fill_value=fill_value, axis=axis
        )

    def _reduce(self, name, *, skipna=True, axis=0, **kwargs):
        if name in ["min", "max"]:
            return getattr(self, name)(skipna=skipna, axis=axis)

        raise TypeError(f"Cannot perform reduction '{name}' with string/bytes dtype")

    def _validate_scalar(self, value):
        return self._convert_value(value)

    @abstractmethod
    def _ensure_ndarray_type(self, arr):
        pass

    @classmethod
    @abstractmethod
    def _ensure_dtype(cls, dtype):
        pass

    @classmethod
    @abstractmethod
    def _convert_value(cls, value):
        pass


class ImageUriArray(ImageArray):
    _dtype = ImageUriDtype()

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        dtype = cls._ensure_dtype(dtype)
        from pandas.core.arrays.masked import BaseMaskedArray

        if isinstance(scalars, BaseMaskedArray):
            # avoid costly conversion to object dtype
            values = scalars._data
            values = pd._libs.lib.ensure_string_array(
                values, copy=copy, convert_na_value=False
            )
            values[scalars._mask] = pd.NA
        else:
            # convert non-na-likes to str, and nan-likes to StringDtype().na_value
            scalars = [cls._convert_value(x) for x in scalars]
            values = pd._libs.lib.ensure_string_array(
                scalars, na_value=pd.NA, copy=copy
            )

        # Manually creating new array avoids the validation step in the __init__, so is
        # faster. Refactor need for validation?
        return cls(values, copy=copy)

    @classmethod
    def _ensure_dtype(cls, dtype):
        if dtype and not (isinstance(dtype, str) and dtype == ImageUriDtype.name):
            dtype = pandas_dtype(dtype)
            assert isinstance(dtype, ImageUriDtype)
        return dtype

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype=None, copy=False):
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    def _ensure_ndarray_type(self, arr):
        return arr.astype(str)

    @classmethod
    def _convert_value(cls, x):
        if pd.isna(x):
            return cls._dtype.na_value
        if isinstance(x, ImageUri):
            return x.uri
        if isinstance(x, str):
            return x
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, ImageBinary):
            raise TypeError("Call `save` before adding inlined Image to ImageUriArray")
        raise TypeError("ImageArray only understands NA, Image, and str scalars")

    def _validate_searchsorted_value(self, value):
        if self._is_scalar_value(value):
            return self._validate_scalar(value)
        if isinstance(value, type(self)):
            return value._ndarray
        elif isinstance(value, ExtensionArray):
            return value.to_numpy().astype(str)
        elif isinstance(value, np.ndarray):
            return self._ensure_ndarray_type(value)
        else:
            return [self._validate_scalar(x) for x in value]


class ImageBinaryArray(ImageArray):
    _dtype = ImageBinaryDtype()

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        dtype = cls._ensure_dtype(dtype)
        from pandas.core.arrays.masked import BaseMaskedArray

        if isinstance(scalars, BaseMaskedArray):
            # avoid costly conversion to object dtype
            values = scalars._data.astype("O")
            values[scalars._mask] = cls._dtype.na_value
        else:
            # convert non-na-likes to str, and nan-likes to StringDtype().na_value
            scalars = [cls._convert_value(x) for x in scalars]
            values = np.array(scalars, dtype="O")

        # Manually creating new array avoids the validation step in the __init__, so is
        # faster. Refactor need for validation?
        return cls(values, copy=copy)

    @classmethod
    def _ensure_dtype(cls, dtype):
        if dtype and not (isinstance(dtype, bytes) and dtype == ImageBinaryDtype.name):
            dtype = pandas_dtype(dtype)
            assert isinstance(dtype, ImageBinaryDtype)
        return dtype

    def _ensure_ndarray_type(self, arr):
        return arr.astype("O")

    @classmethod
    def _convert_value(cls, x):
        if pd.isna(x):
            return cls._dtype.na_value
        if isinstance(x, ImageUri):
            raise TypeError(f"Call `.bytes` to make {x} compatible with inlined Image")
        if isinstance(x, bytes):
            return x
        if isinstance(x, ImageBinary):
            return x.bytes
        raise TypeError("ImageArray only understands NA, Image, and bytes scalars")

    def _validate_searchsorted_value(self, value):
        if self._is_scalar_value(value):
            return self._validate_scalar(value)
        if isinstance(value, type(self)):
            return value._ndarray
        elif isinstance(value, ExtensionArray):
            return value.to_numpy().astype("O")
        elif isinstance(value, np.ndarray):
            return self._ensure_ndarray_type(value)
        else:
            return [self._validate_scalar(x) for x in value]
