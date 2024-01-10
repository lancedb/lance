#  Copyright (c) 2023. Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import math

import lance._datagen as datagen
import pyarrow as pa
import pytest


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
