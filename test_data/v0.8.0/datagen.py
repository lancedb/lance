import shutil

import lance
import pyarrow as pa

# To generate the test file, we should be running this version of lance
assert lance.__version__ == "0.8.0"

name = "migrated_from_v0.7.5"
# Use the data from v0.7.5 as the base for this test
shutil.rmtree(name, ignore_errors=True)
shutil.copytree("../v0.7.5/with_deletions", name)

tab = pa.table({"x": range(2)})
dataset = lance.write_dataset(tab, name, mode="append")

# This is the bug
assert dataset.count_rows() != dataset.to_table().num_rows
