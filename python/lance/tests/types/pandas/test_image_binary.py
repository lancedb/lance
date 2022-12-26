import numpy as np
import pandas as pd
import pandas._testing as tm
import pytest
from pandas.compat import pa_version_under7p0
from pandas.conftest import *
from pandas.core.dtypes.common import is_object_dtype, is_string_dtype
from pandas.errors import PerformanceWarning
from pandas.tests.extension import base
from pandas.tests.extension.conftest import *

from lance.types.pandas.image import ImageBinaryDtype


@pytest.fixture
def dtype():
    """A fixture providing the ExtensionDtype to validate."""
    return ImageBinaryDtype()


@pytest.fixture
def data(dtype):
    """
    Length-100 array for this type.
    * data[0] and data[1] should both be non missing
    * data[0] and data[1] should not be equal
    """
    uris = [bytes(f"image_{i}", "UTF-8") for i in range(100)]
    return pd.array(uris, dtype=dtype)


@pytest.fixture
def na_value():
    """The scalar missing value for this type. Default 'None'"""
    return pd.NA


@pytest.fixture
def data_missing(dtype):
    return dtype.construct_array_type()._from_sequence([pd.NA, bytes("image", "UTF-8")])


@pytest.fixture
def na_cmp():
    # we are pd.NA
    return lambda x, y: x is pd.NA and y is pd.NA


@pytest.fixture
def data_for_grouping(dtype):
    img_a = bytes(f"image_a", "UTF-8")
    img_b = bytes(f"image_b", "UTF-8")
    img_c = bytes(f"image_c", "UTF-8")
    return dtype.construct_array_type()._from_sequence(
        [img_b, img_b, pd.NA, pd.NA, img_a, img_a, img_b, img_c]
    )


@pytest.fixture
def data_for_sorting(dtype):
    img_a = bytes(f"image_a", "UTF-8")
    img_b = bytes(f"image_b", "UTF-8")
    img_c = bytes(f"image_c", "UTF-8")
    return dtype.construct_array_type()._from_sequence([img_b, img_c, img_a])


@pytest.fixture
def data_missing_for_sorting(dtype):
    img_a = bytes(f"image_a", "UTF-8")
    img_b = bytes(f"image_b", "UTF-8")
    return dtype.construct_array_type()._from_sequence([img_b, pd.NA, img_a])


class TestDtype(base.BaseDtypeTests):
    @pytest.mark.skip(reason="not a problem")
    def test_is_not_string_type(self, dtype):
        pass

    def test_is_not_object_type(self, dtype):
        # super class didn't assert the value
        assert not is_object_dtype(dtype)


class TestInterface(base.BaseInterfaceTests):
    def test_array_interface(self, data):
        result = np.array(data)
        assert result[0] == data[0].bytes

        result = np.array(data, dtype=object)
        expected = np.array([img.bytes for img in list(data)], dtype=object)
        tm.assert_numpy_array_equal(result, expected)


class TestConstructors(base.BaseConstructorsTests):
    pass


class TestGetitem(base.BaseGetitemTests):
    pass


class TestSetitem(base.BaseSetitemTests):
    pass


class TestIndex(base.BaseIndexTests):
    pass


class TestMissing(base.BaseMissingTests):
    pass


class TestNoReduce(base.BaseNoReduceTests):
    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_numeric(self, data, all_numeric_reductions, skipna):
        op_name = all_numeric_reductions

        if op_name in ["min", "max"]:
            return

        ser = pd.Series(data)
        with pytest.raises(TypeError):
            getattr(ser, op_name)(skipna=skipna)


class TestCasting(base.BaseCastingTests):

    @pytest.mark.skip(reason="not str convertible")
    def test_astype_str(self, data):
        raise TypeError("Not str convertible")

    def test_astype_bytes(self, data):
        result = pd.Series(data[:5]).astype(bytes)
        expected = pd.Series([x.bytes for x in data[:5]], dtype=bytes)
        self.assert_series_equal(result, expected)

    @pytest.mark.skip(reason="not str convertible")
    def test_astype_string(self, data, nullable_string_dtype):
        raise TypeError("Not str convertible")


class TestComparisonOps(base.BaseComparisonOpsTests):
    def _compare_other(self, ser, data, op, other):
        op_name = f"__{op.__name__}__"
        result = getattr(ser, op_name)(other)
        expected = getattr(ser.astype(object), op_name)(other).astype("boolean")
        self.assert_series_equal(result, expected)

    def test_compare_scalar(self, data, comparison_op):
        ser = pd.Series(data)
        self._compare_other(ser, data, comparison_op, b"abc")

    def test_compare_array(self, data, comparison_op):
        ser = pd.Series(data)
        other = pd.Series([data[0].bytes] * len(data))
        self._compare_other(ser, data, comparison_op, other)


class TestPrinting(base.BasePrintingTests):
    pass


class TestMethods(base.BaseMethodsTests):
    @pytest.mark.parametrize("dropna", [True, False])
    def test_value_counts(self, all_data, dropna, request):
        all_data = all_data[:10]
        if dropna:
            other = all_data[~all_data.isna()]
        else:
            other = all_data
        result = pd.Series(all_data).value_counts(dropna=dropna).sort_index()
        expected = pd.Series(other).value_counts(dropna=dropna).sort_index()
        self.assert_series_equal(result, expected)

    def test_combine_add(self, data_repeated):
        with pytest.raises(TypeError):
            super().test_combine_add(data_repeated)

    def test_combine_le(self, data_repeated):
        with pytest.raises(TypeError):
            super().test_combine_le(data_repeated)

    def test_value_counts_with_normalize(self, data):
        # GH 33172
        data = data[:10].unique()
        values = np.array(data[~data.isna()])
        ser = pd.Series(data, dtype=data.dtype)
        result = ser.value_counts(normalize=True).sort_index()

        if not isinstance(data, pd.Categorical):
            expected = pd.Series([1 / len(values)] * len(values), index=result.index)
        else:
            expected = pd.Series(0.0, index=result.index)
            expected[result > 0] = 1 / len(values)
        self.assert_series_equal(result, expected)
