# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import lance
from lance.query import BoostQuery, MatchQuery
import pyarrow as pa

# To generate the test file, we should be running this version of lance
assert lance.__version__ == "0.27.0"

data = pa.table(
    {
        "text": [
            "frodo was a puppy",
            "frodo was a happy puppy",
            "frodo was a puppy with a tail",
        ]
    }
)

ds = lance.write_dataset(data, "legacy_fts_index")
ds.create_scalar_index("text", "INVERTED", with_position=False)
results = ds.to_table(
    full_text_query=BoostQuery(
        MatchQuery("puppy", "text"),
        MatchQuery("happy", "text"),
        negative_boost=0.5,
    ),
)
assert results.num_rows == 3
assert set(results["text"].to_pylist()) == {
    "frodo was a puppy",
    "frodo was a puppy with a tail",
    "frodo was a happy puppy",
}
