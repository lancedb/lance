# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import shutil
from pathlib import Path

import lance
import pyarrow as pa

EXPECTED_LANCE_VERSION = "8.0.0"

assert lance.__version__ == EXPECTED_LANCE_VERSION

dataset_path = Path(__file__).parent / "fts_list_elements"
shutil.rmtree(dataset_path, ignore_errors=True)

dataset = lance.write_dataset(
    pa.table(
        {
            "id": pa.array([1], type=pa.uint64()),
            "doc": pa.array([["a", "b"]], type=pa.list_(pa.string())),
        }
    ),
    dataset_path,
)
dataset.create_scalar_index(
    "doc",
    "INVERTED",
    base_tokenizer="raw",
    max_token_length=None,
    lower_case=False,
    stem=False,
    remove_stop_words=False,
    ascii_folding=False,
)

element_matches = dataset.to_table(full_text_query="a")
joined_matches = dataset.to_table(full_text_query="a b")
assert element_matches["id"].to_pylist() == [1]
assert joined_matches.num_rows == 0
