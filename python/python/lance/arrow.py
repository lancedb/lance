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
from typing import List, Optional, Union

import pyarrow as pa

from .lance import BFloat16
from .lance import bfloat16_array as bfloat16_array


class ImageURIType(pa.ExtensionType):
    def __init__(self):
        pa.ExtensionType.__init__(self, pa.string(), "lance.arrow.image_uri")

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, self, storage_type, serialized):
        return ImageURIType()

    def __arrow_ext_class__(self):
        return ImageURIArray

    def __arrow_ext_scalar_class__(self):
        return ImageURIScalar


class EncodedImageType(pa.ExtensionType):
    def __init__(self):
        # TODO: use pa.BinaryView once available?
        pa.ExtensionType.__init__(self, pa.binary(), "lance.arrow.encoded_image")

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, self, storage_type, serialized):
        return EncodedImageType()

    def __arrow_ext_class__(self):
        return EncodedImageArray

    def __arrow_ext_scalar_class__(self):
        return EncodedImageScalar


class ImageTensorType(pa.ExtensionType):
    def __init__(self, arrow_type, shape):
        # TODO: use pa.VariableShapeTensorArray once available?
        pa.ExtensionType.__init__(
            self, pa.fixed_shape_tensor(arrow_type, shape), "lance.arrow.image_tensor"
        )

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, self, storage_type, serialized):
        return ImageTensorType()

    def __arrow_ext_class__(self):
        return ImageTensorArray

    def __arrow_ext_scalar_class__(self):
        return ImageTensorScalar


class ImageURIArray(pa.ExtensionArray):
    def __repr__(self):
        return "<lance.arrow.ImageURIArray object at 0x%016x>\n%s" % (
            id(self),
            repr(self.to_pylist()),
        )

    @classmethod
    def from_uris(cls, uris: Union[pa.StringArray, List[str]]):
        if isinstance(uris, list):
            return cls.from_storage(ImageURIType(), pa.array(uris, type=pa.string()))
        else:
            return cls.from_storage(ImageURIType(), uris)

    def materialize(self):
        # Assume first URI's filesystem is the same for all URIs
        from pyarrow import fs

        first_uri = self.storage[0].as_py()
        filesystem, _ = fs.FileSystem.from_uri(first_uri)

        # Read all URIs and return their content as lance.arrow.EncodedImageArray
        images = []
        for uri in self.storage:
            with filesystem.open_input_stream(uri.as_py()) as f:
                images.append(f.read())

        return EncodedImageArray.from_binary_array(pa.array(images, type=pa.binary()))


class EncodedImageArray(pa.ExtensionArray):
    def __repr__(self):
        return "<lance.arrow.EncodedImageArray object at 0x%016x>\n%s" % (
            id(self),
            repr(self.to_pylist()),
        )

    @classmethod
    def from_binary_array(cls, array: pa.binary()):
        return cls.from_storage(EncodedImageType(), array)

    def materialize(self):
        # TODO: try other decoders (pillow/torchvision/cv2/scipy/..) to decode images
        # or rather just pick one and make it a requirement?
        import numpy as np

        try:
            import tensorflow as tf

            image_arrays = (tf.io.decode_image(img.as_py()) for img in self.storage)
        except ImportError:
            import io

            from PIL import Image

            image_arrays = (
                np.array(Image.open(io.BytesIO(img.as_py()))) for img in self.storage
            )
        except ImportError:
            raise "No image decoder available"

        image_array = np.stack(image_arrays)
        return ImageTensorArray.from_numpy_ndarray(image_array)


class ImageTensorArray(pa.ExtensionArray):
    import numpy as np

    def __repr__(self):
        return "<lance.arrow.ImageTensorArray object at 0x%016x>\n%s" % (
            id(self),
            repr(self.to_pylist()),
        )

    @classmethod
    def from_numpy_ndarray(cls, array: np.ndarray):
        arrow_type = pa.from_numpy_dtype(array.dtype)
        shape = array.shape[1:]
        size = array.size / array.shape[0]

        tensor_array = pa.FixedShapeTensorArray.from_storage(
            pa.fixed_shape_tensor(arrow_type, shape),
            pa.FixedSizeListArray.from_arrays(array.ravel(), size),
        )
        return cls.from_storage(ImageTensorType(arrow_type, shape), tensor_array)

    def to_tf(self):
        import tensorflow as tf

        return tf.convert_to_tensor(self.storage.to_numpy_ndarray())

    def to_encoded(self):
        import tensorflow as tf

        full_tensor = tf.convert_to_tensor(self.storage.to_numpy_ndarray())
        # a = tf.io.encode_png(full_tensor[0])
        encoded_images = (tf.io.encode_png(x).numpy() for x in full_tensor)

        return EncodedImageArray.from_binary_array(
            pa.array(encoded_images, type=pa.binary())
        )

    def to_tfrecords(self):
        # TODO: write to tfrecords
        pass


class ImageURIScalar(pa.ExtensionScalar):
    def as_py(self):
        return self.value.as_py()


class EncodedImageScalar(pa.ExtensionScalar):
    def as_py(self):
        return self.value.as_py()


class ImageTensorScalar(pa.ExtensionScalar):
    def as_py(self):
        return self.value.as_py()


class BFloat16Array(pa.ExtensionArray):
    def __repr__(self):
        return "<lance.arrow.BFloat16Array object at 0x%016x>\n%s" % (
            id(self),
            repr(self.to_pylist()),
        )

    def to_numpy(self, zero_copy_only=False):
        """Convert to a NumPy array.

        This will do a zero-copy conversion.

        The conversion will fail if the array contains null values."""
        if self.null_count > 0:
            raise ValueError("Cannot convert null values to numpy")

        import numpy as np
        from ml_dtypes import bfloat16

        buffer = self.storage.buffers()[1]
        array = np.frombuffer(buffer, dtype=bfloat16)

        return array

    @classmethod
    def from_numpy(cls, array):
        """Create a BFloat16Array from a NumPy array.

        Can only convert from a NumPy array of dtype bfloat16 from the ml_dtypes
        module."""
        from ml_dtypes import bfloat16

        if array.dtype != bfloat16:
            raise ValueError("Cannot convert non-bfloat16 values to BFloat16Array")
        data = pa.py_buffer(array.tobytes())
        return pa.Array.from_buffers(BFloat16Type(), len(array), [None, data])


class BFloat16Scalar(pa.ExtensionScalar):
    def as_py(self) -> Optional[BFloat16]:
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
            inner = BFloat16Array.from_numpy(array)
            return cls(inner)
