# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import duckdb
import lance
import pyarrow as pa
import pytest


def test_duckdb_pushdown_extension_types(tmp_path):
    # large_binary is reported by pyarrow as a substrait extension type.  Datafusion
    # does not currently handle these extension types.  This should be ok as long
    # as the filter isn't accessing the column with the extension type.
    #
    # Lance works around this by removing any columns with extension types from the
    # schema it gives to duckdb.
    tab = pa.table(
        {
            "filterme": [1, 2, 3],
            "largebin": pa.array([b"123", b"456", b"789"], pa.large_binary()),
            "othercol": [4, 5, 6],
        }
    )
    ds = lance.write_dataset(tab, str(tmp_path))  # noqa: F841
    expected = tab.slice(1, 1)
    actual = duckdb.query("SELECT * FROM ds WHERE filterme = 2").fetch_arrow_table()
    assert actual.to_pydict() == expected.to_pydict()

    expected = tab.slice(0, 1)
    actual = duckdb.query("SELECT * FROM ds WHERE othercol = 4").fetch_arrow_table()
    assert actual.to_pydict() == expected.to_pydict()

    # Not the best error message but hopefully this is short lived until datafusion
    # supports substrait extension types.
    with pytest.raises(
        duckdb.InvalidInputException,
        match="referenced a field that is not yet supported by Substrait conversion",
    ):
        duckdb.query("SELECT * FROM ds WHERE largebin = '456'").fetchall()

    # Unclear if all of these result in pushdown or not but they shouldn't error if
    # they do.
    for filt in [
        "filterme != 2 AND filterme != 1",
        "filterme in (1)",
        "filterme IS NULL",
        "filterme IS NOT NULL",
        "filterme < 2",
    ]:
        expected = duckdb.query(f"SELECT * FROM tab WHERE {filt}").fetch_arrow_table()
        actual = duckdb.query(f"SELECT * FROM ds WHERE {filt}").fetch_arrow_table()
        assert actual == expected
