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
    import tensorflow as tf  # noqa: F401
except ImportError:
    pytest.skip(
        "Tensorflow is not installed. Please install tensorflow to "
        + "test lance.tf module.",
        allow_module_level=True,
    )

import lance
from lance.fragment import LanceFragment
from lance.tf.data import from_lance, lance_fragments


@pytest.fixture
def tf_dataset(tmp_path):
    df = pd.DataFrame({"a": range(10000), "s": [f"val-{i}" for i in range(10000)]})
    tbl = pa.Table.from_pandas(df)

    def batches():
        for batch in tbl.to_batches(100):
            yield batch

    uri = tmp_path / "dataset.lance"
    lance.write_dataset(batches(), uri, schema=tbl.schema, max_rows_per_file=100)
    return uri


def test_fragment_dataset(tf_dataset):
    ds = from_lance(tf_dataset, batch_size=100)
    for idx, batch in enumerate(ds):
        assert batch["a"].numpy()[0] == idx * 100
        assert batch["s"].numpy()[0] == f"val-{idx * 100}".encode("utf-8")
        assert batch["a"].shape == (100,)


def test_shuffle(tf_dataset):
    fragments = lance_fragments(tf_dataset).shuffle(4, seed=20).take(3)

    ds = from_lance(tf_dataset, fragments=fragments, batch_size=100)
    raw_ds = lance.dataset(tf_dataset)
    scanner = raw_ds.scanner(
        fragments=[LanceFragment(raw_ds, fid) for fid in [0, 3, 1]], batch_size=100
    )

    for batch, raw_batch in zip(ds, scanner.to_batches()):
        assert batch["a"].numpy()[0] == raw_batch.to_pydict()["a"][0]
        assert batch["a"].numpy().shape == (100,)
