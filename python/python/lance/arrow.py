from typing import Union

import pyarrow as pa

from .lance import BFloat16
from .lance import bfloat16_array as bfloat16_array


class BFloat16Array(pa.ExtensionArray):
    def __repr__(self):
        return "<lance.arrow.BFloat16Array object at 0x%016x>\n%s" % (
            id(self),
            repr(self.to_pylist()),
        )

    def to_numpy(self, zero_copy_only=False):
        if self.null_count > 0:
            raise ValueError("Cannot convert null values to numpy")

        import numpy as np
        from ml_dtypes import bfloat16

        buffer = self.storage.buffers()[1].to_pybytes()
        array = np.frombuffer(buffer, dtype=bfloat16)

        return array


class BFloat16Scalar(pa.ExtensionScalar):
    def as_py(self) -> BFloat16 | None:
        if self.value is None:
            return None
        else:
            return BFloat16.from_bytes(self.value.as_py())


class BFloat16Type(pa.ExtensionType):
    def __init__(self):
        pa.ExtensionType.__init__(self, pa.binary(2), "lance.bfloat16")

    def __arrow_ext_serialize__(self):
        # TODO: encode endianess
        return b""

    @classmethod
    def __arrow_ext_deserialize__(self, storage_type, serialized):
        # TODO: decode endianess
        return BFloat16Type()

    def __arrow_ext_class__(self):
        return BFloat16Array

    def __arrow_ext_scalar_class__(self):
        return BFloat16Scalar

    def to_pandas_dtype(self):
        return PandasBFloat16Type()


pa.register_extension_type(BFloat16Type())

try:
    from pandas.api.extensions import (
        ExtensionArray,
        ExtensionDtype,
        register_extension_dtype,
    )
except ImportError:
    pass
else:
    # Define Pandas and register Pandas extensions
    @register_extension_dtype
    class PandasBFloat16Type(ExtensionDtype):
        kind = "f"
        na_value = None
        name = "lance.bfloat16"
        names = None
        type = BFloat16
        _is_numeric = True

        def __from_arrow__(
            self, array: Union[pa.Array, pa.ChunkedArray]
        ) -> ExtensionArray:
            return PandasBFloat16Array(array)

        def construct_array_type(self):
            return PandasBFloat16Array

        @classmethod
        def construct_from_string(cls, string):
            if string == "lance.bfloat16":
                return cls()
            else:
                raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")

    class PandasBFloat16Array(ExtensionArray):
        dtype = PandasBFloat16Type()

        def __init__(self, data):
            self.data = data

        @classmethod
        def _from_sequence(
            cls, scalars, *, dtype: PandasBFloat16Type, copy: bool = False
        ):
            return PandasBFloat16Array(bfloat16_array(scalars))

        def __getitem__(self, item):
            import numpy as np

            if isinstance(item, int):
                return self.data[item].as_py()
            elif isinstance(item, slice):
                return PandasBFloat16Array(self.data[item])
            elif isinstance(item, np.ndarray) and item.dtype == bool:
                return PandasBFloat16Array(self.data.filter(pa.array(item)))
            else:
                raise NotImplementedError()

        def __len__(self):
            return len(self.data)

        def isna(self):
            return self.data.is_null().to_numpy(zero_copy_only=False)

        def to_numpy(self, *args, **kwargs):
            return self.data.to_numpy()

        def __arrow_array__(self, type=None):
            return self.data

        @classmethod
        def from_numpy(cls, array):
            data = pa.py_buffer(array.tobytes())
            inner = pa.Array.from_buffers(BFloat16Type(), len(array), [None, data])
            return cls(inner)
