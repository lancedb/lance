import shutil

import lance
import pyarrow as pa

# To generate the test file, we should be running this version of lance
assert lance.__version__ == "0.10.5"

name = "corrupt_schema"

tab = pa.table({"x": range(2)})
dataset = lance.write_dataset(tab, name, mode="create")

dataset.add_columns({"y": "x * 2", "z": "x * 3"})
dataset.add_columns({"a": "-x"})
dataset.drop_columns(["a", "z"])
dataset.add_columns({"b": "x * 4", "c": "x * 5"})

# This is the bug: b and c will show data from z and a.
assert dataset.to_table() == pa.table({
    "x": range(2),
    "y": [0, 2],
    "b": [0, 3],
    "c": [0, -1],
})
