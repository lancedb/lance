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

import json
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

import numpy as np
import pyarrow as pa

from .lance import BFloat16
from .lance import bfloat16_array as bfloat16_array


class ImageURIType(pa.ExtensionType):
    def __init__(self):
        pa.ExtensionType.__init__(self, pa.string(), "lance.arrow.image_uri")

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
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
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return EncodedImageType()

    def __arrow_ext_class__(self):
        return EncodedImageArray

    def __arrow_ext_scalar_class__(self):
        return EncodedImageScalar


class FixedShapeImageTensorType(pa.ExtensionType):
    def __init__(self, arrow_type, shape):
        self.shape = shape
        self.arrow_type = arrow_type
        assert len(shape) > 0

        length = 1
        for dim in shape:
            length *= dim

        pa.ExtensionType.__init__(
            self,
            pa.list_(arrow_type, length),
            "lance.arrow.fixed_shape_image_tensor",
        )

    def __arrow_ext_serialize__(self):
        serialized = json.dumps({"shape": self.shape}).encode()
        return serialized

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        deserialized = json.loads(serialized.decode())
        return FixedShapeImageTensorType(storage_type.value_type, deserialized["shape"])

    def __arrow_ext_class__(self):
        return FixedShapeImageTensorArray

    def __arrow_ext_scalar_class__(self):
        return FixedShapeImageTensorScalar


pa.register_extension_type(ImageURIType())
pa.register_extension_type(EncodedImageType())
pa.register_extension_type(FixedShapeImageTensorType(pa.uint8(), (0,)))


class ImageArray(pa.ExtensionArray):
    def __repr__(self):
        return "<lance.arrow.%s object at 0x%016x>\n%s" % (
            type(self).__name__,
            id(self),
            repr(self.to_pylist()),
        )

    @classmethod
    def from_array(cls, images):
        """
        Create an one of subclasses of ImageArray from input data.

        Parameters
        ----------
        images : Union[pa.StringArray, pa.BinaryArray, pa.FixedShapeTensorArray,
            Iterable]

        Returns
        -------
        Union[ImageURIArray, EncodedImageArray, FixedShapeImageTensorArray]
        """

        if isinstance(images, pa.StringArray):
            return pa.ExtensionArray.from_storage(ImageURIType(), images)
        elif isinstance(images, pa.BinaryArray):
            return pa.ExtensionArray.from_storage(EncodedImageType(), images)
        elif isinstance(images, pa.FixedShapeTensorArray):
            shape = images.type.shape
            value_type = images.type.value_type
            typ = FixedShapeImageTensorType(value_type, shape)
            return pa.ExtensionArray.from_storage(typ, images.storage)
        elif isinstance(
            images, (ImageURIArray, EncodedImageArray, FixedShapeImageTensorArray)
        ):
            return images
        elif isinstance(images, (list, tuple, Iterable)):
            return pa.ExtensionArray.from_storage(
                ImageURIType(), pa.array(images, type=pa.string())
            )

        else:
            raise TypeError("Cannot build a ImageArray from {}".format(type(images)))


class ImageURIArray(ImageArray):
    """
    Array of image URIs. URIs may represent local files or remote files on the web,
    S3 or GCS if they are accessible by the machine executing this.
    """

    @classmethod
    def from_uris(cls, uris: Union[pa.StringArray, Iterable[Union[str, Path]]]):
        """
        Create an ImageURIArray from a pa.StringArray or an iterable.

        Parameters
        ----------
        uris : Union[pa.StringArray, Iterable[Union[str, Path]]]

        Returns
        -------
        ImageURIArray
            Array of image URIs

        Examples
        --------
        >>> uris = ["file::///tmp/1.png"]
        >>> ImageURIArray.from_uris(uris)
        <lance.arrow.ImageURIArray object at 0x...>
        ['file::///tmp/1.png']
        """

        if not isinstance(uris, pa.StringArray):
            uris = pa.array((str(uri) for uri in uris), type=pa.string())
        else:
            raise TypeError("Cannot build a ImageURIArray from {}".format(type(uris)))

        return cls.from_storage(ImageURIType(), uris)

    def read_uris(self):
        """
        Read the images from the URIs into memory and return an EncodedImageArray

        Returns
        -------
        EncodedImageArray
            Array of encoded images

        Examples
        --------
        >>> import os
        >>> uris = [os.path.join(os.path.dirname(__file__), "../tests/images/1.png")]
        >>> uri_array = ImageURIArray.from_uris(uris)
        >>> uri_array.read_uris()
        <lance.arrow.EncodedImageArray object at 0x...>
        ...
        """
        from urllib.error import URLError
        from urllib.parse import urlparse
        from urllib.request import Request, urlopen

        from pyarrow import fs

        def download(url):
            req = Request(url)
            try:
                return urlopen(req).read()
            except URLError as e:
                if hasattr(e, "reason"):
                    print("Failed to reach the server: ", e.reason)
                elif hasattr(e, "code"):
                    print(
                        "The server could not fulfill the request. Error code: ", e.code
                    )

        images = []
        for uri in self.storage:
            parsed_uri = urlparse(uri.as_py())
            if parsed_uri.scheme in ("http", "https"):
                images.append(download(uri))
            else:
                filesystem, path = fs.FileSystem.from_uri(uri.as_py())
                with filesystem.open_input_stream(path) as f:
                    images.append(f.read())

        return EncodedImageArray.from_storage(
            EncodedImageType(), pa.array(images, type=pa.binary())
        )


class EncodedImageArray(ImageArray):
    """
    Array of encoded images. Images may be encoded in any format. Format is not stored
    by the array but can typically be inferred from the image bytes. Alternatively a
    separate record of encoding can be kept in a separate array outside this library.
    """

    import numpy as np

    def __repr__(self):
        def pillow_metadata_decoder(images):
            import io

            from PIL import Image

            img = Image.open(io.BytesIO(images[0].as_py()))
            return img

        def tensorflow_metadata_decoder(images):
            import tensorflow as tf

            img = tf.io.decode_image(images[0].as_py())
            return img

        decoders = (
            ("tensorflow", tensorflow_metadata_decoder),
            ("PIL", pillow_metadata_decoder),
        )
        decoder = None

        for libname, metadata_decoder in decoders:
            try:
                __import__(libname)
                decoder = metadata_decoder
                break
            except ImportError:
                pass

        return "<lance.arrow.%s object at 0x%016x>%s\n%s" % (
            type(self).__name__,
            id(self),
            "\n[" + repr(decoder(self)) + ", ..]" if decoder else None,
            repr(self.to_pylist()[0][:30]),
        )

    def to_tensor(
        self, decoder: Optional[Callable[[pa.BinaryArray], np.ndarray]] = None
    ):
        """
        Decode encoded images and return a FixedShapeImageTensorArray

        Parameters
        ----------
        decoder : Callable[pa.binary()], optional
            A function that takes a pa.binary() and returns a numpy.ndarray
            or pa.fixed_shape_tensor. If not provided, will attempt to use
            tensorflow and then pillow decoder in that order.

        Returns
        -------
        FixedShapeImageTensorArray
            Array of images as tensors

        Examples
        --------
        >>> import os
        >>> uris = [os.path.join(os.path.dirname(__file__), "../tests/images/1.png")]
        >>> encoded_image_array = ImageURIArray.from_uris(uris).read_uris()
        >>> encoded_image_array.to_tensor()
        <lance.arrow.FixedShapeImageTensorArray object at 0x...>
        [[42, 42, 42, 255]]
        """
        import numpy as np

        if not hasattr(pa, "FixedShapeTensorType"):
            raise NotImplementedError("This function requires PyArrow >= 12.0.0")

        if not decoder:

            def pillow_decoder(images):
                import io

                from PIL import Image

                return np.stack(
                    Image.open(io.BytesIO(img)) for img in images.to_pylist()
                )

            def tensorflow_decoder(images):
                import tensorflow as tf

                decoded_to_tensor = tuple(
                    tf.io.decode_image(img) for img in images.to_pylist()
                )
                return tf.stack(decoded_to_tensor, axis=0).numpy()

            decoders = [
                ("tensorflow", tensorflow_decoder),
                ("PIL", pillow_decoder),
            ]
            for libname, decoder_function in decoders:
                try:
                    __import__(libname)
                    decoder = decoder_function
                    break
                except ImportError:
                    pass
            else:
                raise ValueError(
                    "No image decoder available. Please either install one of "
                    "tensorflow, pillow, or pass a decoder argument."
                )

        image_array = decoder(self.storage)
        if isinstance(image_array, pa.FixedShapeTensorType):
            shape = image_array.shape
            arrow_type = image_array.storage_type
            tensor_array = image_array
        else:
            shape = image_array.shape[1:]
            arrow_type = pa.from_numpy_dtype(image_array.dtype)
            tensor_array = pa.FixedShapeTensorArray.from_numpy_ndarray(image_array)

        return pa.ExtensionArray.from_storage(
            FixedShapeImageTensorType(arrow_type, shape), tensor_array.storage
        )


# TODO: add VariableShapeImageTensorType once pa.VariableShapeTensorArray is available
class FixedShapeImageTensorArray(ImageArray):
    """
    Array of fixed shape tensors representing image pixels.
    """

    def to_numpy(self):
        """
        Convert FixedShapeImageTensorArray to a numpy.ndarray.

        Returns
        -------
        numpy.ndarray
            Array of images

        Examples
        --------

        >>> import os
        >>> uris = [os.path.join(os.path.dirname(__file__), "../tests/images/1.png")]
        >>> tensor_image_array = ImageURIArray.from_uris(uris).read_uris().to_tensor()
        >>> tensor_image_array.to_numpy()
        array([[[[ 42,  42,  42, 255]]]], dtype=uint8)
        """
        ext_type = pa.fixed_shape_tensor(self.storage.type.value_type, self.type.shape)
        tensor_array = pa.ExtensionArray.from_storage(ext_type, self.storage)
        return tensor_array.to_numpy_ndarray()

    def to_encoded(self, encoder=None):
        """
        Encode FixedShapeImageTensorArray to PNG bytes and return an EncodedImageArray.

        Parameters
        ----------
        encoder : Callable[np.ndarray], optional
            An encoder function that takes a numpy.ndarray and returns an encoded image.

        Returns
        -------
        EncodedImageArray
            Array of encoded images

        Examples
        --------
        >>> import numpy as np
        >>> arr = np.array([[[[42, 42, 42, 255]]]], dtype=np.uint8)
        >>> arrow_type = pa.from_numpy_dtype(arr.dtype)
        >>> shape = arr.shape[1:]
        >>> tensor_array = pa.FixedShapeTensorArray.from_numpy_ndarray(arr)
        >>> tensor_image_array = FixedShapeImageTensorArray.from_storage(
        ... FixedShapeImageTensorType(arrow_type, shape), tensor_array.storage)
        >>> tensor_image_array.to_encoded()
        <lance.arrow.EncodedImageArray object at 0x...>
        ...
        """

        def pillow_encoder(x):
            import io

            from PIL import Image

            encoded_images = []
            for y in x:
                with io.BytesIO() as buf:
                    Image.fromarray(y).save(buf, format="PNG")
                    encoded_images.append(buf.getvalue())
            return pa.array(encoded_images, type=pa.binary())

        def tensorflow_encoder(x):
            import tensorflow as tf

            encoded_images = (
                tf.io.encode_png(y).numpy() for y in tf.convert_to_tensor(x)
            )
            return pa.array(encoded_images, type=pa.binary())

        if not encoder:
            encoders = (
                ("PIL", pillow_encoder),
                ("tensorflow", tensorflow_encoder),
            )
            for libname, encoder_function in encoders:
                try:
                    __import__(libname)
                    encoder = encoder_function
                    break
                except ImportError:
                    pass
            else:
                raise ValueError(
                    "No image encoder available. Please either install one of "
                    "tensorflow, pillow, or pass an encoder argument."
                )

        return EncodedImageArray.from_storage(
            EncodedImageType(), encoder(self.to_numpy())
        )


class ImageScalar(pa.ExtensionScalar):
    def as_py(self):
        return self.value.as_py()


class ImageURIScalar(ImageScalar):
    pass


class EncodedImageScalar(ImageScalar):
    pass


class FixedShapeImageTensorScalar(ImageScalar):
    pass
