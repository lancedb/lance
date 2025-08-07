import lance
import pyarrow as pa

# To generate the test file, we should be running this older version of lance
assert lance.__version__ == "0.31.1", f"Expected lance 0.31.1, got {lance.__version__}"

# Create a simple dataset with a vector index
nrows = 16

data = pa.table(
    {
        "id": pa.array(range(nrows)),
        "list": [["你好", "世界"], []] * (nrows // 2),
    }
)

# Write dataset
dataset = lance.write_dataset(data, "bad_list", max_rows_per_file=4)
