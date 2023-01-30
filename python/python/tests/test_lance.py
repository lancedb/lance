import numpy as np
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

    table = dataset.to_table(columns=['a'], limit=20)
    assert len(table) == 20
    with_offset = dataset.to_table(columns=['a'], offset=10, limit=10)
    assert with_offset == table[10:]


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

    # TODO allow Dataset::create to take both async and also RecordBatchReader
    # lance_dataset = lance.dataset(uri / "table.lance")
    # _check_roundtrip(lance_dataset, uri / "lance_dataset.lance", tbl)

    # lance_scanner = lance_dataset.scanner()
    # _check_roundtrip(lance_scanner, uri / "lance_scanner.lance", tbl)


def _check_roundtrip(data_obj, uri, expected):
    lance.write_dataset(data_obj, uri)
    assert expected == lance.dataset(uri).to_table()


def test_nearest(tmp_path):
    uri = tmp_path

    schema = pa.schema([pa.field("emb", pa.list_(pa.float32(), 32), False)])
    npvals = np.random.rand(100, 32)
    npvals /= np.sqrt((npvals**2).sum(axis=1))[:, None]
    values = pa.array(npvals.ravel(), type=pa.float32())
    arr = pa.FixedSizeListArray.from_arrays(values, 32)
    tbl = pa.Table.from_arrays([arr], schema=schema)
    lance.write_dataset(tbl, uri)

    dataset = lance.dataset(uri)
    top10 = dataset.to_table(nearest={"column": "emb", "q": arr[0].values, "k": 10, "nprobes": 10})
    scores = l2sq(arr[0].values, npvals.reshape((100, 32)))
    indices = np.argsort(scores)
    assert tbl.take(indices[:10]).to_pandas().equals(top10.to_pandas()[["emb"]])
    assert np.allclose(scores[indices[:10]], top10.to_pandas().score.values)


def l2sq(vec, mat):
    return np.sum((mat - vec)**2, axis=1)
