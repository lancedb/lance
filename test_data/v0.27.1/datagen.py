import glob
import os
import lance
from lance.file import LanceFileReader
import pyarrow as pa
import pyarrow.compute as pc

# To generate the test file, we should be running this version of lance
assert lance.__version__ == "0.27.1"

ndims = 32
nvecs = 512

data = pa.table(
    {
        "id": pa.array(range(nvecs)),
        "vec": pa.FixedSizeListArray.from_arrays(
            pc.random(ndims * nvecs).cast(pa.float32()), ndims
        ),
    }
)

dataset = lance.write_dataset(data, "pq_in_schema")
dataset.create_index(
    "vec",
    "IVF_PQ",
    num_partitions=1,
    num_sub_vectors=4,
)

# Open file in pq_in_schema/_indices/*/auxiliary.idx
# and get schema
path = glob.glob("pq_in_schema/_indices/*/auxiliary.idx")[0]
path = os.path.abspath(path)
reader = LanceFileReader(path)
schema = reader.metadata().schema
assert b"storage_metadata" in schema.metadata.keys()
