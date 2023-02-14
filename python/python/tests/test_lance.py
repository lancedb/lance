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
import os

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset
import pytest


def test_table_roundtrip(tmp_path):
    uri = tmp_path

    df = pd.DataFrame({"a": range(100), "b": range(100)})
    tbl = pa.Table.from_pandas(df)
    lance.write_dataset(tbl, uri)

    dataset = lance.dataset(uri)
    assert dataset.uri == str(uri.absolute())
    assert tbl.schema == dataset.schema
    assert tbl == dataset.to_table()

    one_col = dataset.to_table(columns=["a"])
    assert one_col == tbl.select(["a"])

    table = dataset.to_table(columns=["a"], limit=20)
    assert len(table) == 20
    with_offset = dataset.to_table(columns=["a"], offset=10, limit=10)
    assert with_offset == table[10:]


def test_input_types(tmp_path):
    # test all input types for write_dataset
    uri = tmp_path

    df = pd.DataFrame({"a": range(100), "b": range(100)})
    tbl = pa.Table.from_pandas(df)

    lance.write_dataset(df, str(uri / "pandas.lance"), schema=tbl.schema)
    assert tbl == lance.dataset(str(uri / "pandas.lance")).to_table()

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
    top10 = dataset.to_table(
        nearest={"column": "emb", "q": arr[0].values, "k": 10, "nprobes": 10}
    )
    scores = l2sq(arr[0].values, npvals.reshape((100, 32)))
    indices = np.argsort(scores)
    assert tbl.take(indices[:10]).to_pandas().equals(top10.to_pandas()[["emb"]])
    assert np.allclose(scores[indices[:10]], top10.to_pandas().score.values)


def l2sq(vec, mat):
    return np.sum((mat - vec) ** 2, axis=1)


def test_count_rows(tmp_path):
    df = pd.DataFrame({"values": range(100)})
    tbl = pa.Table.from_pandas(df)
    dataset = lance.write_dataset(tbl, tmp_path)
    assert dataset.count_rows() == 100


def test_create_index(tmp_path):
    dataset = _create_dataset(str(tmp_path / "test.lance"))

    # Check required args
    with pytest.raises(ValueError):
        dataset.create_index("emb", "IVF_PQ")
    with pytest.raises(ValueError):
        dataset.create_index("emb", "IVF_PQ", num_partitions=5)
    with pytest.raises(ValueError):
        dataset.create_index("emb", "IVF_PQ", num_sub_vectors=4)
    with pytest.raises(KeyError):
        dataset.create_index("foo", "IVF_PQ", num_partitions=5, num_sub_vectors=16)
    with pytest.raises(NotImplementedError):
        dataset.create_index("emb", "foo", num_partitions=5, num_sub_vectors=16)

    # all good
    dataset.create_index("emb", "IVF_PQ", num_partitions=16, num_sub_vectors=4)


@pytest.mark.skipif((os.uname().sysname == "Darwin") and (os.uname().machine != "arm64"),
                    reason="no neon on GHA")
def test_simd_alignment(tmp_path):
    dataset = _create_dataset(str(tmp_path / "test.lance"))
    # SIMD alignment is enforced
    with pytest.raises(OSError):
        dataset.create_index("emb", "IVF_PQ", num_partitions=5, num_sub_vectors=16)


def _create_dataset(uri):
    schema = pa.schema([pa.field("emb", pa.list_(pa.float32(), 32), False)])
    npvals = np.random.rand(1000, 32)
    npvals /= np.sqrt((npvals**2).sum(axis=1))[:, None]
    values = pa.array(npvals.ravel(), type=pa.float32())
    arr = pa.FixedSizeListArray.from_arrays(values, 32)
    tbl = pa.Table.from_arrays([arr], schema=schema)
    return lance.write_dataset(tbl, uri)
