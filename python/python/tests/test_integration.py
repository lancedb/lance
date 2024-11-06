# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import duckdb
import lance
import pyarrow as pa
import pytest


@pytest.mark.skip(reason="Unsigned integers not available for pushdown yet")
def test_duckdb_filter_on_rowid(tmp_path):
    tab = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    ds = lance.write_dataset(tab, str(tmp_path))
    ds = lance.dataset(str(tmp_path), default_scan_options={"with_row_id": True})
    row_ids = ds.scanner(columns=[], with_row_id=True).to_table().column(0).to_pylist()
    expected = tab.slice(1, 1)
    actual = duckdb.query(
        f"SELECT * FROM ds WHERE _rowid = {row_ids[1]}"
    ).fetch_arrow_table()

    assert actual.to_pydict() == expected.to_pydict()


class DuckDataset(pa.dataset.Dataset):
    """
    Hacky way to wrap a lance dataset hide extension types.

    Usage:
    >>> scanner = DuckDataset(lance.dataset("my_dataset.lance"))
    >>> duckdb.sql("SELECT uid FROM scanner LIMIT 10;")
    """

    def __init__(self, ds):
        self._ds = ds
        fields = [ds.schema.field(i) for i in range(len(ds.schema))]
        fields = [f.remove_metadata() for f in fields]
        self.pruned_schema = pa.schema(fields)

    @staticmethod
    def _filter_field(field: pa.Field) -> bool:
        # Filter metadata
        UNSUPPORTED_METADATA = [
            {
                b"ARROW:extension:metadata": b"",
                b"ARROW:extension:name": b"lance.arrow.encoded_image",
            },
            {
                b"ARROW:extension:metadata": b"",
                b"ARROW:extension:name": b"lance.arrow.image_uri",
            },
        ]
        metadata_is_supported = all(field.metadata != m for m in UNSUPPORTED_METADATA)
        type_is_supported = field.type not in {
            pa.large_binary(),
        }
        return metadata_is_supported and type_is_supported

    @property
    def schema(self):
        return self._schema

    def __getattribute__(self, attr):
        if attr == "schema":
            return object.__getattribute__(self, "pruned_schema")
        elif attr == "_filter_field":
            return object.__getattribute__(self, "_filter_field")
        else:
            ds = super().__getattribute__("_ds")
            return object.__getattribute__(ds, attr)


def test_duckdb_pushdown_extension_types(tmp_path):
    # large_binary is reported by pyarrow as a substrait extension type.  Datafusion
    # does not currently handle these extension types.  This should be ok as long
    # as the filter isn't accessing the column with the extension type.
    #
    # Lance works around this by removing any columns with extension types from the
    # schema it gives to duckdb.
    #
    # image is an extension type.  DuckDb currently rejects anything that's an extension
    # type.  We can clumsily work around this by pretending its not an extension type.
    tab = pa.table(
        {
            "filterme": [1, 2, 3],
            "largebin": pa.array([b"123", b"456", b"789"], pa.large_binary()),
            "othercol": [4, 5, 6],
            "image": pa.array(
                [b"123", b"456", b"789"],
                pa.binary(),
            ),
        },
        schema=pa.schema(
            [
                pa.field("filterme", pa.int64()),
                pa.field("largebin", pa.large_binary()),
                pa.field("othercol", pa.int64()),
                pa.field(
                    "image",
                    pa.binary(),
                    metadata={
                        b"ARROW:extension:metadata": b"",
                        b"ARROW:extension:name": b"lance.arrow.encoded_image",
                    },
                ),
            ]
        ),
    )
    ds = lance.write_dataset(tab, str(tmp_path))  # noqa: F841
    ds = DuckDataset(ds)

    expected = tab.slice(1, 1)
    actual = duckdb.query("SELECT * FROM ds WHERE filterme = 2").fetch_arrow_table()
    assert actual.to_pydict() == expected.to_pydict()

    expected = tab.slice(0, 1)
    actual = duckdb.query("SELECT * FROM ds WHERE othercol = 4").fetch_arrow_table()
    assert actual.to_pydict() == expected.to_pydict()

    # Not the best error message but hopefully this is short lived until datafusion
    # supports substrait extension types.
    with pytest.raises(
        # Older versions of duckdb use duckdb.InvalidInputException and newer versions
        # use duckdb.duckdb.Error but that part isn't really important so just check for
        # Exception
        Exception,
        match="referenced a field that is not yet supported by Substrait conversion",
    ):
        duckdb.query("SELECT * FROM ds WHERE largebin = '456'").fetchall()

    # Need to subtract extension types from tab to use as our expected results
    tab = pa.table(tab.columns, schema=ds.schema)
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
