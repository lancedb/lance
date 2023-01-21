import pyarrow as pa
import lance


def test_to_table():
    uri = "/Users/changshe/code/eto/sophon/src/phalanx-sdk/datagen/lance/Topic_a/000000.lance"
    dataset = lance.dataset(uri)
    assert dataset.uri == uri
    assert isinstance(dataset.schema, pa.Schema)

    proj = dataset.schema[0].name
    table = dataset.to_table(columns=[proj], limit=20)
    assert isinstance(table, pa.Table)
    assert len(table.schema) == 1
    assert len(table) == 20
    with_offset = dataset.to_table(columns=[proj], offset=10, limit=10)
    assert with_offset == table[10:]