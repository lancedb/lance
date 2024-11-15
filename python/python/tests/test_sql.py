# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import lance.sql
import pyarrow as pa


def test_no_dataset():
    schema = pa.schema([pa.field("Int64(5)", pa.int64(), nullable=False)])
    assert lance.sql.query("SELECT 5").to_table() == pa.table(
        {"Int64(5)": [5]}, schema=schema
    )
