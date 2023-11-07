import lance
import pyarrow as pa

# To generate the test file, we should be running this version of lance
assert lance.__version__ == "0.7.5"

tab = pa.table({"x": range(100)})

dataset = lance.write_dataset(tab, "with_deletions")

dataset.delete("x >= 10 and x < 20")
