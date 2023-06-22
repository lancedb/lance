from lance.arrow import BFloat16, BFloat16Array, bfloat16_array


def test_bf16_value():
    val = BFloat16(1.124565)
    assert val == BFloat16(1.124565)
    assert str(val) == "1.125"


def test_bf16_repr():
    arr = bfloat16_array([1.1, None, 3.4])
    assert isinstance(arr, BFloat16Array)

    expected = """<lance.arrow.BFloat16Array object at 0x125b542e0>
[
  1.1,
  null,
  3.4
]"""
    assert repr(arr) == expected
