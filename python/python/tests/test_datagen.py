import math

import pyarrow as pa
import pytest


def is_datagen_supported():
    try:
        import lance._datagen as _  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(is_datagen_supported(), reason="datagen is supported")
def test_import_error():
    with pytest.raises(ImportError, match="was not built with the datagen feature"):
        import lance._datagen as _  # noqa: F401


@pytest.mark.skipif(not is_datagen_supported(), reason="datagen not supported")
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
