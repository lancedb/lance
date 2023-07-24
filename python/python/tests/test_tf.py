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

import pandas as pd
import pyarrow as pa
import pytest

try:
    import tensorflow as tf
except ImportError:
    pytest.skip(
        "Tensorflow is not installed. Please install tensorflow to use lance.tf module.",
        allow_module_level=True,
    )

import lance
from lance.tf.data import from_lance


@pytest.fixture
def tf_dataset(tmp_path):
    df = pd.DataFrame({"a": range(10000), "s": ["val-{}" for i in range(10000)]})
    tbl = pa.Table.from_pandas(df)

    def batches():
        for batch in tbl.to_batches(100):
            yield batch

    uri = tmp_path / "dataset.lance"
    lance.write_dataset(batches(), uri, schema=tbl.schema, max_rows_per_file=100)
    return uri


def test_fragment_dataset(tf_dataset):
    dataset = from_lance(tf_dataset)
    for batch in dataset:
        print(batch)
    ds = lance.dataset(tf_dataset)
    # dataset = from_fragments(ds, fragments)
    # print(list(dataset))
