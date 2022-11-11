import numpy as np
import pandas as pd
import pandas._testing as tm
import pytest
from pandas.compat import pa_version_under7p0
from pandas.core.dtypes.common import is_string_dtype, is_object_dtype
from pandas.errors import PerformanceWarning
from pandas.tests.extension import base
from pandas.tests.extension.conftest import *
from pandas.conftest import *
from lance.types.pandas.image import ImageUriDtype


@pytest.fixture
def dtype():
    """A fixture providing the ExtensionDtype to validate."""
    return ImageUriDtype()


@pytest.fixture
def data(dtype):
    """
    Length-100 array for this type.
    * data[0] and data[1] should both be non missing
    * data[0] and data[1] should not be equal
    """
    uris = [f'/path/to/image{i}.jpg' for i in range(100)]
    return pd.array(uris, dtype=dtype)


@pytest.fixture
def na_value():
    """The scalar missing value for this type. Default 'None'"""
    return pd.NA


@pytest.fixture
def data_missing(dtype):
    return dtype.construct_array_type()._from_sequence([pd.NA, "img.jpg"])


@pytest.fixture
def na_cmp():
    # we are pd.NA
    return lambda x, y: x is pd.NA and y is pd.NA


@pytest.fixture
def data_for_grouping(dtype):
    return dtype.construct_array_type()._from_sequence(
        ["b.jpg", "b.jpg", pd.NA, pd.NA, "a.jpg", "a.jpg", "b.jpg", "c.jpg"]
    )


@pytest.fixture
def data_for_sorting(dtype):
    return dtype.construct_array_type()._from_sequence(["b.jpg", "c.jpg", "a.jpg"])


@pytest.fixture
def data_missing_for_sorting(dtype):
    return dtype.construct_array_type()._from_sequence(["b.jpg", pd.NA, "a.jpg"])


class TestDtype(base.BaseDtypeTests):

    @pytest.mark.skip(reason="not a problem")
    def test_is_not_string_type(self, dtype):
        # super class didn't assert the value
        assert not is_string_dtype(dtype)

    def test_is_not_object_type(self, dtype):
        # super class didn't assert the value
        assert not is_object_dtype(dtype)


class TestInterface(base.BaseInterfaceTests):
    def test_array_interface(self, data):
        result = np.array(data)
        assert result[0] == data[0].uri

        result = np.array(data, dtype=object)
        expected = np.array([img.uri for img in list(data)], dtype=object)
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
    def test_astype_str(self, data):
        result = pd.Series(data[:5]).astype(str)
        expected = pd.Series([x.uri for x in data[:5]], dtype=str)
        self.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "nullable_string_dtype",
        [
            "string[python]",
            pytest.param(
                "string[pyarrow]", marks=td.skip_if_no("pyarrow", min_version="1.0.0")
            ),
        ],
    )
    def test_astype_string(self, data, nullable_string_dtype):
        # GH-33465
        result = pd.Series(data[:5]).astype(nullable_string_dtype)
        expected = pd.Series([x.uri for x in data[:5]], dtype=nullable_string_dtype)
        self.assert_series_equal(result, expected)


class TestComparisonOps(base.BaseComparisonOpsTests):
    def _compare_other(self, ser, data, op, other):
        op_name = f"__{op.__name__}__"
        result = getattr(ser, op_name)(other)
        expected = getattr(ser.astype(object), op_name)(other).astype("boolean")
        self.assert_series_equal(result, expected)

    def test_compare_scalar(self, data, comparison_op):
        ser = pd.Series(data)
        self._compare_other(ser, data, comparison_op, "abc")

    def test_compare_array(self, data, comparison_op):
        ser = pd.Series(data)
        other = pd.Series([str(data[0])] * len(data))
        self._compare_other(ser, data, comparison_op, other)


class TestParsing(base.BaseParsingTests):
    pass


class TestPrinting(base.BasePrintingTests):
    pass


class TestGroupBy(base.BaseGroupbyTests):
    @pytest.mark.parametrize("as_index", [True, False])
    def test_groupby_extension_agg(self, as_index, data_for_grouping):
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})
        with tm.maybe_produces_warning(
            PerformanceWarning,
            pa_version_under7p0
            and getattr(data_for_grouping.dtype, "storage", "") == "pyarrow",
        ):
            result = df.groupby("B", as_index=as_index).A.mean()
        with tm.maybe_produces_warning(
            PerformanceWarning,
            pa_version_under7p0
            and getattr(data_for_grouping.dtype, "storage", "") == "pyarrow",
        ):
            _, uniques = pd.factorize(data_for_grouping, sort=True)

        if as_index:
            index = pd.Index._with_infer(uniques, name="B")
            expected = pd.Series([3.0, 1.0, 4.0], index=index, name="A")
            self.assert_series_equal(result, expected)
        else:
            expected = pd.DataFrame({"B": uniques, "A": [3.0, 1.0, 4.0]})
            self.assert_frame_equal(result, expected)

    def test_groupby_extension_transform(self, data_for_grouping):
        with tm.maybe_produces_warning(
            PerformanceWarning,
            pa_version_under7p0
            and getattr(data_for_grouping.dtype, "storage", "") == "pyarrow",
            check_stacklevel=False,
        ):
            super().test_groupby_extension_transform(data_for_grouping)

    @pytest.mark.filterwarnings("ignore:Falling back:pandas.errors.PerformanceWarning")
    def test_groupby_extension_apply(self, data_for_grouping, groupby_apply_op):
        super().test_groupby_extension_apply(data_for_grouping, groupby_apply_op)


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