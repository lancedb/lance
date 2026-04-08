"""
Generate test data to simulate a dataset created before the `files` field was added
to IndexMetadata. This tests backward compatibility for the files field.

To generate this test data:
1. Check out Lance before the file sizes feature was added (or use a released version)
2. Run: pip install -e python/
3. Run: python test_data/pre_file_sizes/datagen.py

The dataset will be created without the `files` field in IndexMetadata,
which simulates what an older dataset would look like.
"""

import lance
import pyarrow as pa
import shutil
import os

assert lance.__version__ == "2.0.0-beta.1"

# Create output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "index_without_file_sizes")
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

# Create a simple dataset with a scalar index
data = pa.table(
    {
        "id": pa.array(range(100)),
        "values": pa.array([f"value_{i}" for i in range(100)]),
    }
)

# Write dataset
dataset = lance.write_dataset(data, OUTPUT_DIR)

# Create a scalar index (BTree) using the correct method
dataset.create_scalar_index("values", index_type="BTREE")

print(f"Created dataset at {OUTPUT_DIR}")
print(f"Dataset version: {dataset.version}")

# Verify
dataset = lance.dataset(OUTPUT_DIR)
indices = dataset.list_indices()
print(f"Indices: {indices}")
print("\nTest data created successfully!")
