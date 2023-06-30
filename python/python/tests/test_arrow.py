import re

from lance.arrow import BFloat16, BFloat16Array, bfloat16_array


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
