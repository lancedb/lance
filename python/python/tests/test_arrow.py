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
import re

import pandas as pd
import pyarrow as pa
import pytest
from lance.arrow import BFloat16, BFloat16Array, PandasBFloat16Array, bfloat16_array


def test_bf16_value():
    val = BFloat16(1.124565)
    assert val == BFloat16(1.124565)
    assert str(val) == "1.125"

    should_be_true = [
        BFloat16(1.1) > BFloat16(1.0),
        BFloat16(1.1) >= BFloat16(1.0),
        BFloat16(1.1) != BFloat16(1.0),
        BFloat16(1.0) < BFloat16(1.1),
        BFloat16(1.0) <= BFloat16(1.1),
    ]
    assert all(comparison for comparison in should_be_true)

    should_be_false = [
        BFloat16(1.1) < BFloat16(1.0),
        BFloat16(1.1) <= BFloat16(1.0),
        BFloat16(1.1) == BFloat16(1.0),
        BFloat16(1.0) >= BFloat16(1.1),
        BFloat16(1.0) >= BFloat16(1.1),
    ]
    assert not any(comparison for comparison in should_be_false)


def test_bf16_repr():
    data = [1.1, None, 3.4]
    arr = bfloat16_array(data)
    assert isinstance(arr, BFloat16Array)

    expected = [BFloat16(1.1), None, BFloat16(3.4)]
    assert arr[0].as_py() == expected[0]
    assert arr[1].as_py() == expected[1]
    assert arr[2].as_py() == expected[2]

    assert arr.to_pylist() == expected

    expected_re = r"""<lance.arrow.BFloat16Array object at 0x[\w\d]+>
\[1.1015625, None, 3.40625\]"""
    assert re.match(expected_re, repr(arr))

    # TODO: uncomment tests once fixed upstream.
    # https://github.com/apache/arrow/issues/36648


#     tab = pa.table({"x": arr})
#     expected_re = r"""pyarrow.Table
# x: extension<lance.bfloat16<BFloat16Type>>
# ---
# x: \[\[1.1015625, None, 3.40625\]\]"""
#     assert re.match(expected_re, repr(tab))


@pytest.mark.skipif(not pd, reason="Pandas not available")
def test_bf16_pandas():
    data = [1.1, None, 3.4]
    arr = bfloat16_array(data)
    arr_pandas = arr.to_pandas()
    assert arr_pandas[0] == BFloat16(1.1)
    assert arr_pandas[1] is None
    assert arr_pandas[2] == BFloat16(3.4)

    # Can instantiate with dtype string
    series = pd.Series(arr_pandas, dtype="lance.bfloat16")
    pd.testing.assert_series_equal(arr_pandas, series)

    # Can roundtrip to Arrow
    arr_arrow = pa.array(arr_pandas)
    assert isinstance(arr_arrow, BFloat16Array)
    assert arr == arr_arrow

    pd.testing.assert_series_equal(arr_arrow.to_pandas(), arr_pandas)


def test_bf16_numpy():
    import numpy as np
    from ml_dtypes import bfloat16

    data = [1.1, 2.1, 3.4]
    arr = bfloat16_array(data)
    arr_numpy = arr.to_numpy()

    expected = np.array(data, dtype=bfloat16)

    np.testing.assert_array_equal(arr_numpy, expected)

    # Can roundtrip to Pandas
    arr_pandas = PandasBFloat16Array.from_numpy(arr_numpy)
    np.testing.assert_array_equal(arr_pandas.to_numpy(), expected)

    # Can roundtrip to Arrow
    arr_arrow = BFloat16Array.from_numpy(arr_numpy)
    assert arr == arr_arrow
    np.testing.assert_array_equal(arr_arrow.to_numpy(), expected)
