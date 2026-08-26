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
