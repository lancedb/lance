import pandas as pd
import pyarrow as pa
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