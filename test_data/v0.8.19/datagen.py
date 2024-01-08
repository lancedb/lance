import lance
import pyarrow as pa

assert lance.__version__ == "0.8.19"

nrows = 100
ndim = 10

tab = pa.table(
    {
        # fixed_size_list:int32:10
        "fsl": pa.FixedSizeListArray.from_arrays(
            pa.array(nrows * ndim * [0.0], type=pa.float32()), ndim
        ),
        # large_list
        "list": pa.array([[1, 2, 3]] * nrows, type=pa.list_(pa.int32())),
        # large_list
        "large_list": pa.array([[1, 2, 3]] * nrows, type=pa.large_list(pa.int32())),
        # large.struct
        "list_of_struct": pa.array(
            [[{"a": 1, "b": 2}]] * nrows,
            type=pa.list_(pa.struct([("a", pa.int32()), ("b", pa.int32())])),
        ),
        # large_list.struct
        "large_list_of_struct": pa.array(
            [[{"a": 1, "b": 2}]] * nrows,
            type=pa.large_list(pa.struct([("a", pa.int32()), ("b", pa.int32())])),
        ),
    }
)

lance.write_dataset(tab, "list_columns")
