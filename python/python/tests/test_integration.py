# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import duckdb
import lance
import pyarrow as pa


def test_duckdb(tmp_path):
    tab = pa.table(
        {
            "filterme": [1, 2, 3],
            "largebin": pa.array([b"123", b"456", b"789"], pa.large_binary()),
        }
    )
    ds = lance.write_dataset(tab, str(tmp_path))
    print(duckdb.query("SELECT * FROM ds WHERE filterme = 2").fetchall())
