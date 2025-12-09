import pyarrow as pa
import lance
import shutil

shutil.rmtree("btree_in_index_pkg.lance", ignore_errors=True)

ds = lance.write_dataset(pa.table({"x": range(1000)}), "btree_in_index_pkg.lance")
ds.create_scalar_index("x", "BTREE")
