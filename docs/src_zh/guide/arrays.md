# 扩展数组（Extension Arrays）

Lance 为 Arrow 数组和 Pandas Series 提供扩展，以表示机器学习应用的数据类型。

## BFloat16

[BFloat16](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus) 是一种 16 位浮点数，专为机器学习场景设计。直观地说，它只有 2-3 位精度，但与 32 位浮点数具有相同的范围：~1e-38 到 ~1e38。相比之下，16 位浮点数（Float16）的范围为 ~5.96e-8 到 65504。

Lance 为 BFloat16 提供了 Arrow 扩展数组（`lance.arrow.BFloat16Array`）和 Pandas 扩展数组（`lance._arrow.PandasBFloat16Type`）。这些与 [ml_dtypes](https://github.com/jax-ml/ml_dtypes) 的 bfloat16 NumPy 扩展数组兼容。

如果你使用 Pandas，可以使用 `lance.bfloat16` dtype 字符串来创建数组：

```python
import lance.arrow

pd.Series([1.1, 2.1, 3.4], dtype="lance.bfloat16")
# 0    1.1015625
# 1      2.09375
# 2      3.40625
# dtype: lance.bfloat16
```

要创建 Arrow 数组，使用 `lance.arrow.bfloat16_array` 函数：

```python
from lance.arrow import bfloat16_array

bfloat16_array([1.1, 2.1, 3.4])
# <lance.arrow.BFloat16Array object at 0x000000016feb94e0>
# [
#   1.1015625,
#   2.09375,
#   3.40625
# ]
```

最后，如果你已有 NumPy 数组，可以将其转换为以下任一格式：

```python
import numpy as np
from ml_dtypes import bfloat16
from lance.arrow import PandasBFloat16Array, BFloat16Array

np_array = np.array([1.1, 2.1, 3.4], dtype=bfloat16)
PandasBFloat16Array.from_numpy(np_array)
# <PandasBFloat16Array>
# [1.1015625, 2.09375, 3.40625]
# Length: 3, dtype: lance.bfloat16
BFloat16Array.from_numpy(np_array)
# <lance.arrow.BFloat16Array object at 0x...>
# [
#   1.1015625,
#   2.09375,
#   3.40625
# ]
```

读取时，可以使用每个数组类的 `to_numpy` 方法将其转换回 NumPy bfloat16 dtype。

## ImageURI

`lance.arrow.ImageURIArray` 是一个存储图像在其他存储系统中 URI 位置的数组。例如，`file:///path/to/image.png` 用于本地文件系统，`s3://bucket/path/image.jpeg` 用于 AWS S3 上的图像。当你想从现有存储介质延迟加载图像时，使用此数组类型。

可以通过调用 `lance.arrow.ImageURIArray.from_uris` 并传入由 `pyarrow.StringArray` 表示的 URI 列表或产生字符串的可迭代对象来创建。注意 URI 不会被严格验证，图像也不会自动读入内存。

```python
from lance.arrow import ImageURIArray

ImageURIArray.from_uris([
   "/tmp/image1.jpg",
   "file:///tmp/image2.jpg",
   "s3://example/image3.jpg"
])
# <lance.arrow.ImageURIArray object at 0x...>
# ['/tmp/image1.jpg', 'file:///tmp/image2.jpg', 's3://example/image3.jpg']
```

`lance.arrow.ImageURIArray.read_uris` 将图像读入内存并返回一个新的 `lance.arrow.EncodedImageArray` 对象。

```python
from lance.arrow import ImageURIArray

relative_path = "images/1.png"
uris = [os.path.join(os.path.dirname(__file__), relative_path)]
ImageURIArray.from_uris(uris).read_uris()
# <lance.arrow.EncodedImageArray object at 0x...>
# [b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00...']
```

## EncodedImage

`lance.arrow.EncodedImageArray` 是一个以编码和压缩表示形式存储 jpeg 和 png 图像的数组，就像它们写入磁盘时的样子。当你想以压缩格式处理图像时使用此数组，例如从磁盘读取它们或将它们嵌入 HTML 中。

可以通过在现有 `lance.arrow.ImageURIArray` 上调用 `lance.arrow.ImageURIArray.read_uris` 来创建。这将把引用的图像读入内存。也可以通过调用 `lance.arrow.ImageArray.from_array` 并传入已读入 `pyarrow.BinaryArray` 的编码图像数组来创建，或通过调用 `lance.arrow.ImageTensorArray.to_encoded` 来创建。

提供了 `lance.arrow.EncodedImageArray.to_tensor` 方法来解码编码图像并将其作为 `lance.arrow.FixedShapeImageTensorArray` 返回，从中可以转换为 numpy 数组或 TensorFlow 张量。对于图像解码，它首先尝试使用通过可选函数参数提供的解码器。如果未提供解码器，它将依次尝试使用 [Pillow](https://pillow.readthedocs.io/en/stable/) 和 [tensorflow](https://www.tensorflow.org/api_docs/python/tf/io/encode_png)。如果两个库或自定义解码器都不可用，将引发异常。

```python
from lance.arrow import ImageURIArray

uris = [os.path.join(os.path.dirname(__file__), "images/1.png")]
encoded_images = ImageURIArray.from_uris(uris).read_uris()
print(encoded_images.to_tensor())

def tensorflow_decoder(images):
    import tensorflow as tf
    import numpy as np

    return np.stack(tf.io.decode_png(img.as_py(), channels=3) for img in images.storage)

print(encoded_images.to_tensor(tensorflow_decoder))
# <lance.arrow.FixedShapeImageTensorArray object at 0x...>
# [[42, 42, 42, 255]]
# <lance.arrow.FixedShapeImageTensorArray object at 0x...>
# [[42, 42, 42, 255]]
```

## FixedShapeImageTensor

`lance.arrow.FixedShapeImageTensorArray` 是一个将图像存储为张量的数组，其中每个像素表示为一个数值。通常图像存储为三维张量，形状为（高度，宽度，通道数）。在彩色图像中，根据 [RGB 颜色模型](https://en.wikipedia.org/wiki/RGB_color_model)，每个像素由三个值（通道）表示。此数组中的图像可以单独作为 numpy 数组读出，也可以堆叠成形状为（批次大小，高度，宽度，通道数）的单个四维 numpy 数组。

可以通过在先前存在的 `lance.arrow.EncodedImageArray` 上调用 `lance.arrow.EncodedImageArray.to_tensor` 来创建。这将解码编码图像并将其作为 `lance.arrow.FixedShapeImageTensorArray` 返回。也可以通过调用 `lance.arrow.ImageArray.from_array` 并传入 `pyarrow.FixedShapeTensorArray` 来创建。

可以通过调用 `lance.arrow.FixedShapeImageTensorArray.to_encoded` 并传入自定义编码器将其编码为 `lance.arrow.EncodedImageArray`。如果未提供编码器，它将依次尝试使用 [tensorflow](https://www.tensorflow.org/api_docs/python/tf/io/encode_png) 和 [Pillow](https://pillow.readthedocs.io/en/stable/)。默认编码器将编码为 PNG。如果两个库都不可用，将引发异常。

```python
from lance.arrow import ImageURIArray

uris = [image_uri]
tensor_images = ImageURIArray.from_uris(uris).read_uris().to_tensor()
tensor_images.to_encoded()
# <lance.arrow.EncodedImageArray object at 0x...>
# [...
# b'\x89PNG\r\n\x1a...'
```
