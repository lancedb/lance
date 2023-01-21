import pandas as pd
import pyarrow as pa
import pyarrow.dataset
import lance


def test_table_roundtrip(tmp_path):
    uri = tmp_path

    df = pd.DataFrame({'a': range(100), 'b': range(100)})
    tbl = pa.Table.from_pandas(df)
    lance.write_dataset(tbl, uri)

    dataset = lance.dataset(uri)
    assert dataset.uri == str(uri.absolute())
    assert tbl.schema == dataset.schema
    assert tbl == dataset.to_table()

    one_col = dataset.to_table(columns=['a'])
    assert one_col == tbl.select(['a'])
    # TODO enable offset limit in Scan ExecNode
    table = dataset.to_table(columns=['a'], limit=20)
    # assert len(table) == 20
    # with_offset = dataset.to_table(columns=['a'], offset=10, limit=10)
    # assert with_offset == table[10:]


def test_input_types(tmp_path):
    # test all input types for write_dataset
    uri = tmp_path

    df = pd.DataFrame({'a': range(100), 'b': range(100)})
    tbl = pa.Table.from_pandas(df)
    _check_roundtrip(tbl, uri / "table.lance", tbl)

    parquet_uri = str(uri / "dataset.parquet")
    pa.dataset.write_dataset(tbl, parquet_uri, format="parquet")
    ds = pa.dataset.dataset(parquet_uri)
    _check_roundtrip(ds, uri / "ds.lance", tbl)

    scanner = pa.dataset.Scanner.from_dataset(ds)
    _check_roundtrip(scanner, uri / "scanner.lance", tbl)

    reader = scanner.to_reader()
    _check_roundtrip(reader, uri / "reader.lance", tbl)

    # TODO tokio runtime error
    # lance_dataset = lance.dataset(uri / "table.lance")
    # _check_roundtrip(lance_dataset, uri / "lance_dataset.lance", tbl)
    
    # lance_scanner = lance_dataset.scanner()
    # _check_roundtrip(lance_scanner, uri / "lance_scanner.lance", tbl)


def _check_roundtrip(data_obj, uri, expected):
    lance.write_dataset(data_obj, uri)
    assert expected == lance.dataset(uri).to_table()

