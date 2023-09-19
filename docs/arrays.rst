Extension Arrays
================

Lance provides extensions for Arrow arrays and Pandas Series to represent data
types for machine learning applications.

BFloat16
--------

`BFloat16 <https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus>`_
is a 16-bit floating point number that is designed for machine learning use cases.
Intuitively, it only has 2-3 digits of precision, but it has the same range as
a 32-bit float: ~1e-38 to ~1e38. By comparison, a 16-bit float has a range of
~5.96e-8 to 65504.

Lance provides an Arrow extension array (:class:`lance.arrow.BFloat16Array`)
and a Pandas extension array (:class:`lance.pandas.BFloat16Dtype`) for BFloat16.
These are compatible with the `ml_dtypes <https://github.com/jax-ml/ml_dtypes>`_
bfloat16 NumPy extension array.

If you are using Pandas, you can use the `lance.bfloat16` dtype string to create
the array:

.. testcode::

    import pandas as pd
    import lance.arrow

    series = pd.Series([1.1, 2.1, 3.4], dtype="lance.bfloat16")
    series

.. testoutput::

    0    1.1015625
    1      2.09375
    2      3.40625
    dtype: lance.bfloat16

To create an an arrow array, use the :func:`lance.arrow.bfloat16_array` function:

.. testcode::

    from lance.arrow import bfloat16_array

    array = bfloat16_array([1.1, 2.1, 3.4])
    array

.. testoutput::

    <lance.arrow.BFloat16Array object at 0x.+>
    [1.1015625, 2.09375, 3.40625]

Finally, if you have a pre-existing NumPy array, you can convert it into either:

.. testcode::

    import numpy as np
    from ml_dtypes import bfloat16
    from lance.arrow import PandasBFloat16Array, BFloat16Array

    np_array = np.array([1.1, 2.1, 3.4], dtype=bfloat16)
    PandasBFloat16Array.from_numpy(np_array)
    BFloat16Array.from_numpy(np_array)

.. testoutput::
    
    <PandasBFloat16Array>
    [1.1015625, 2.09375, 3.40625]
    Length: 3, dtype: lance.bfloat16
    <lance.arrow.BFloat16Array object at 0x.+>
    [1.1015625, 2.09375, 3.40625]

When reading, these can be converted back to to the NumPy bfloat16 dtype using
each array class's ``to_numpy`` method.

ImageURI
--------

:class:`lance.arrow.ImageURIArray` is an Arrow extension array that represents
URIs of images backed by a :class:`pyarrow.StringArray`. It provides a method
for reading images into memory as a :class:`lance.arrow.EncodedImageArray`.

EncodedImage
------------

:class:`lance.arrow.EncodedImageArray` is an Arrow extension array that represents
jpeg/png/gif/bmp encoded images stored in a :class:`pyarrow.BinaryArray`. It provides
a method for materializing images into memory as a :class:`lance.arrow.ImageTensorArray`.

ImageTensor
-----------

:class:`lance.arrow.ImageTensorArray` is an Arrow extension array that represents
images stored as a :class:`pyarrow.FixedShapeTensorArray` or
:class:`pyarrow.VariableShapeTensorArray`. It provides methods for converting to
common memory layouts such as NumPy, PyTorch, TensorFlow. It also provides
a method for storing to TFRecords.
