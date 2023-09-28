import math

import pyarrow as pa
import pytest

import lance._datagen as datagen


@pytest.mark.skipif(datagen.is_datagen_supported(), reason="datagen is supported")
def test_import_error():
    with pytest.raises(
        NotImplementedError, match="was not built with the datagen feature"
    ):
        datagen.rand_batches(None)


@pytest.mark.skipif(not datagen.is_datagen_supported(), reason="datagen not supported")
def test_rand_batches():
    import lance._datagen as datagen

    schema = pa.schema(
        [pa.field("int", pa.int64()), pa.field("vector", pa.list_(pa.float32(), 128))]
    )

    batches = datagen.rand_batches(schema, batch_size_bytes=16 * 1024, num_batches=10)

    assert len(batches) == 10
    for batch in batches:
        assert batch.num_rows == math.ceil(16 * 1024 / (129 * 4))
        assert batch.schema == schema
