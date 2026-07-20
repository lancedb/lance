# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import pickle

import pyarrow as pa
import pytest
from lance.bitmap import bitmap


def test_construct_from_list():
    b = bitmap([4, 1, 2, 1])
    assert len(b) == 3
    assert set(b) == {1, 2, 4}


def test_construct_from_range():
    b = bitmap(range(1000))
    assert len(b) == 1000
    assert 0 in b
    assert 999 in b
    assert 1000 not in b


def test_construct_empty():
    assert len(bitmap()) == 0
    assert len(bitmap([])) == 0


def test_construct_from_bitmap():
    original = bitmap([1, 2, 3])
    copy = bitmap(original)
    assert copy == original


@pytest.mark.parametrize(
    "arrow_type",
    [
        pa.int8(),
        pa.int16(),
        pa.int32(),
        pa.int64(),
        pa.uint8(),
        pa.uint16(),
        pa.uint32(),
    ],
)
def test_construct_from_pyarrow_array(arrow_type):
    arr = pa.array(range(100), type=arrow_type)
    b = bitmap(arr)
    assert len(b) == 100
    assert set(b) == set(range(100))


def test_construct_from_chunked_array():
    chunked = pa.chunked_array(
        [pa.array([1, 2, 3], type=pa.int32()), pa.array([4, 5], type=pa.int32())]
    )
    b = bitmap(chunked)
    assert set(b) == {1, 2, 3, 4, 5}


def test_construct_from_pyarrow_rejects_nulls():
    arr = pa.array([1, 2, None], type=pa.int32())
    with pytest.raises(ValueError):
        bitmap(arr)


def test_construct_rejects_negative_values():
    with pytest.raises(OverflowError):
        bitmap([-1])


def test_construct_from_pyarrow_rejects_out_of_range():
    arr = pa.array([2**33], type=pa.int64())
    with pytest.raises(ValueError):
        bitmap(arr)


def test_len_iter_contains():
    b = bitmap([1, 2, 4])
    assert len(b) == 3
    assert 1 in b
    assert 3 not in b
    assert "not an int" not in b
    assert sorted(b) == [1, 2, 4]


def test_iter_is_lazy():
    """`iter()` returns a dedicated streaming iterator (not a `list_iterator`
    over a pre-built list), yielding values one at a time on demand."""
    b = bitmap(range(1000))
    it = iter(b)
    assert type(it).__name__ == "BitmapIterator"
    assert next(it) == 0
    assert next(it) == 1
    assert list(it) == list(range(2, 1000))


def test_equality():
    assert bitmap([1, 2, 3]) == bitmap([3, 2, 1])
    assert bitmap([1, 2, 3]) == {1, 2, 3}
    assert bitmap([1, 2, 3]) != bitmap([1, 2])
    assert bitmap([1, 2, 3]) != {1, 2}


def test_repr():
    assert repr(bitmap([1, 2, 3])) == "Bitmap({1, 2, 3})"


def test_pickle_round_trip():
    b = bitmap(range(10_000))
    loaded = pickle.loads(pickle.dumps(b))
    assert loaded == b


def test_add_discard_update():
    b = bitmap([1, 2, 3])
    b.add(4)
    assert 4 in b
    b.discard(2)
    assert 2 not in b
    b.update([10, 11])
    assert {10, 11}.issubset(set(b))


def test_mutation_is_copy_on_write():
    original = bitmap([1, 2, 3])
    other = bitmap(original)

    other.add(4)

    assert 4 not in original
    assert 4 in other
    assert original == bitmap([1, 2, 3])
