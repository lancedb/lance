# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import shutil
from pathlib import Path

import lance
import pyarrow as pa
from lance.indices import IndexConfig

# This fixture must retain the JSONB representation written before Lance enabled
# jsonb's arbitrary-precision parser.
assert lance.__version__ == "10.0.0"

output_dir = Path(__file__).parent / "legacy_json_index"
shutil.rmtree(output_dir, ignore_errors=True)

source_values = [
    '{"val":1.2345678901234567890123456789012345678}',
    '{"val":7}',
]
dataset = lance.write_dataset(
    pa.table(
        {
            "id": pa.array([0, 1], pa.int32()),
            "json": pa.array(source_values, pa.json_()),
        }
    ),
    output_dir,
)
dataset.create_scalar_index(
    "json",
    IndexConfig(
        index_type="json",
        parameters={"target_index_type": "btree", "path": "val"},
    ),
    name="json_idx",
)

# Lance 10 stores the long fraction through Float64. Recording that behavior in
# the fixture makes the compatibility boundary explicit and reproducible.
assert dataset.to_table().column("json").to_pylist() == [
    '{"val":1.2345678901234567}',
    '{"val":7}',
]

# Version-0 JSON Bitmap and NGram indexes do not contain the target-type
# sidecar used by current writers. Keep genuine predecessor-written indexes so
# compaction compatibility is tested against their released storage layout.
compaction_output_dir = Path(__file__).parent / "legacy_json_compaction"
shutil.rmtree(compaction_output_dir, ignore_errors=True)

compaction_dataset = lance.write_dataset(
    pa.table(
        {
            "id": pa.array([7, 8], pa.int32()),
            "json": pa.array(
                ['{"tag":"word7"}', '{"tag":"word8"}'],
                pa.json_(),
            ),
        }
    ),
    compaction_output_dir,
    max_rows_per_file=1,
)
for target_index_type in ("bitmap", "ngram"):
    compaction_dataset.create_scalar_index(
        "json",
        IndexConfig(
            index_type="json",
            parameters={"target_index_type": target_index_type, "path": "tag"},
        ),
        name=f"json_{target_index_type}",
    )

assert len(compaction_dataset.get_fragments()) == 2
assert {index["name"] for index in compaction_dataset.list_indices()} == {
    "json_bitmap",
    "json_ngram",
}
