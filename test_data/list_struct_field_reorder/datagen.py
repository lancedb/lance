#!/usr/bin/env python3
"""
Generate test data for issue #5702: project_by_schema should reorder fields inside List<Struct>.

This script creates a dataset where:
1. Fragment 0 has List<Struct<a, b, c>> with all fields + an extra top-level column
2. Fragment 1 has List<Struct> with:
   - Inner struct fields in different order (c, b)
   - Missing inner struct field "a"
   - Missing top-level column "extra"

The combination of out-of-order field storage + schema evolution inside the List<Struct>
triggers the bug where project_by_schema fails to reorder fields.

Before the fix, reading would fail with:
"Incorrect datatype for StructArray field expected List(Struct(...)) got List(Struct(...))"

Usage:
    pip install pylance==2.0.0-beta.8
    python datagen.py
"""

import lance
import pyarrow as pa

# Assert the version to document which version was used to create the test data
assert lance.__version__ == "2.0.0-beta.8", (
    f"Expected pylance 2.0.0-beta.8, got {lance.__version__}"
)
print(f"Lance version: {lance.__version__}")

# Schema with List<Struct<a, b, c>> and an extra column
inner_struct_type = pa.struct(
    [
        pa.field("a", pa.utf8()),
        pa.field("b", pa.utf8()),
        pa.field("c", pa.utf8()),
    ]
)
schema = pa.schema(
    [
        pa.field("id", pa.int32()),
        pa.field("data", pa.list_(pa.field("item", inner_struct_type))),
        pa.field("extra", pa.utf8()),  # This column will be missing in fragment 1
    ]
)

# Fragment 0: data with fields in schema order (a, b, c) + extra column
fragment0_data = pa.table(
    {
        "id": pa.array([1, 2], type=pa.int32()),
        "data": pa.array(
            [
                [{"a": "a1", "b": "b1", "c": "c1"}],
                [{"a": "a2", "b": "b2", "c": "c2"}],
            ],
            type=pa.list_(pa.field("item", inner_struct_type)),
        ),
        "extra": pa.array(["extra1", "extra2"], type=pa.utf8()),
    },
    schema=schema,
)

# Create dataset with first fragment
dataset_path = "list_struct_reorder.lance"
lance.write_dataset(fragment0_data, dataset_path, mode="create")
print("Created dataset with fragment 0")

# Fragment 1: data with inner struct fields reordered AND missing field "a"
inner_struct_type_reordered = pa.struct(
    [
        pa.field("c", pa.utf8()),
        pa.field("b", pa.utf8()),
        # Note: field "a" is intentionally missing from the inner struct
    ]
)
schema_reordered = pa.schema(
    [
        pa.field("id", pa.int32()),
        pa.field("data", pa.list_(pa.field("item", inner_struct_type_reordered))),
        # Note: "extra" column is also missing
    ]
)

fragment1_data = pa.table(
    {
        "id": pa.array([3, 4], type=pa.int32()),
        "data": pa.array(
            [
                [{"c": "c3", "b": "b3"}],  # Missing "a" field
                [{"c": "c4", "b": "b4"}],
            ],
            type=pa.list_(pa.field("item", inner_struct_type_reordered)),
        ),
    },
    schema=schema_reordered,
)

# Append second fragment with reordered and missing inner struct fields
lance.write_dataset(fragment1_data, dataset_path, mode="append")
print("Appended fragment 1 with reordered inner struct")

# Verify the test data was created correctly
ds = lance.dataset(dataset_path)
print(f"\nDataset created with {len(ds.get_fragments())} fragments")
for i, frag in enumerate(ds.get_fragments()):
    for df in frag.metadata.data_files():
        print(f"Fragment {i}: fields={df.fields}")
