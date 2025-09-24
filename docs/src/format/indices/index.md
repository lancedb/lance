# Lance Index Format

Lance provides a powerful and extensible secondary index system. All indexes are stored as regular files.

## Vector Index

Vector Index V3 is the latest version of vector index, which stores a Vector Index
in a regular Lance file, with schema:

```python

import pyarrow as pa

# IVF-PQ
pyarrow.schema(
    [
        pa.field("_rowid", pa.uint64(), True),
        pq.field("__pq_code", pa.uint8(), True),
    ],
    metadata={
        "lance:ivf:partition": {
            # The file position to store the protobuf binary of
            # IVF metadata
            "pb_position": 1234,
        },
        "lance:pq": {
            "codebook_position": 1234,
            "nbits": 8,
            "num_sub_vectors": 96,
            "dimension": 1024,
            "codebook_tensor": '<protobuf bytes>',
            "transposed": True,
        }
    }
)

# IVF-SQ
pyarrow.schema(
    [
        pa.field("_rowid", pa.uint64(), True),
        pa.field("__sq_code", pq.uint64(), True),
    ],
    metadata={
        
    }
)

```

