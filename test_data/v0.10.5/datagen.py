import shutil

import lance
import pyarrow as pa

# To generate the test file, we should be running this version of lance
assert lance.__version__ == "0.10.5"

name = "corrupt_schema"

shutil.rmtree(name, ignore_errors=True)

tab = pa.table({"x": range(2)})
dataset = lance.write_dataset(tab, name, mode="create")

dataset.add_columns({"y": "x * 2", "z": "x * 3"})
dataset.add_columns({"a": "-x"})

new_data = pa.table({"x": range(2, 4), "y": [4, 6], "z": [6, 9], "a": [-2, -3]})
dataset = lance.write_dataset(new_data, name, mode="append")
dataset.drop_columns(["a", "z"])
dataset.add_columns({"b": "x * 4", "c": "x * 5"})

# This is the bug: b and c will show data from z and a.
assert dataset.to_table() == pa.table({
    "x": range(4),
    "y": [0, 2, 4, 6],
    "b": [0, 3, 6, 9],
    "c": [0, -1, -2, -3],
})

fragment_sizes = { len(frag.data_files()) for frag in dataset.get_fragments() }
assert fragment_sizes == {4, 2}
