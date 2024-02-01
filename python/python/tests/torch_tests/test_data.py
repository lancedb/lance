#  Copyright (c) 2024. Lance Developers
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

import shutil
from pathlib import Path
from itertools import chain

import lance
import numpy as np
import pyarrow as pa
import pytest

torch = pytest.importorskip("torch")
from lance.torch.data import LanceDataset  # noqa: E402
from lance.sampler import ShardedFragmentSampler, ShardedBatchSampler, FullScanSampler


def test_iter_over_dataset(tmp_path):
    # 10240 of 32-d vectors.
    data = np.random.random(10240 * 32).astype("f")

    fsl = pa.FixedSizeListArray.from_arrays(data, 32)
    ids = pa.array(range(0, 10240), type=pa.int32())
    tbl = pa.Table.from_arrays([ids, fsl], ["ids", "vec"])

    ds = lance.write_dataset(tbl, tmp_path / "data.lance", max_rows_per_group=32)

    # test when sample size is smaller than max_takes
    torch_ds_small = LanceDataset(
        ds, batch_size=256, samples=1024, columns=["ids", "vec"], cache=True
    )

    total_rows = 0
    for batch in torch_ds_small:
        assert set(batch.keys()) == {"ids", "vec"}
        # row groups of 32 can be batched into 256 exactly.
        assert batch["vec"].shape[0] == 256
        total_rows += batch["vec"].shape[0]
        assert batch["ids"].dtype == torch.int32
        assert batch["vec"].shape[1] == 32
    assert total_rows == 1024

    # test when sample size is greater than max_takes
    torch_ds = LanceDataset(
        ds, batch_size=256, samples=4096, columns=["ids", "vec"], cache=True
    )

    total_rows = 0
    for batch in torch_ds:
        assert set(batch.keys()) == {"ids", "vec"}
        # row groups of 32 can be batched into 256 exactly.
        assert batch["vec"].shape[0] == 256
        total_rows += batch["vec"].shape[0]
        assert batch["ids"].dtype == torch.int32
        assert batch["vec"].shape[1] == 32
    assert total_rows == 4096

    shutil.rmtree(tmp_path / "data.lance")

    total_rows = 0
    # it should read from cache this time.
    for batch in torch_ds_small:
        assert set(batch.keys()) == {"ids", "vec"}
        assert batch["ids"].dtype == torch.int32
        total_rows += batch["vec"].shape[0]
        assert batch["vec"].shape[1] == 32
    assert total_rows == 1024

    total_rows = 0
    # it should read from cache this time.
    for batch in torch_ds:
        assert set(batch.keys()) == {"ids", "vec"}
        assert batch["ids"].dtype == torch.int32
        total_rows += batch["vec"].shape[0]
        assert batch["vec"].shape[1] == 32
    assert total_rows == 4096


def test_sharded_torch_dataset(tmp_path):
    arr = pa.array(range(1000))
    tbl = pa.Table.from_arrays([arr], ["ids"])

    # Write 10 files
    ds = lance.write_dataset(tbl, tmp_path, max_rows_per_file=100)
    assert len(ds.get_fragments()) == 10
    for f in ds.get_fragments():
        assert f.count_rows() == 100

    ds = LanceDataset(
        tmp_path, batch_size=10, columns=["ids"], rank=1, world_size=2, with_row_id=True
    )

    all_ids = []
    row_ids = []
    for batch in ds:
        assert set(batch.keys()) == {"ids", "_rowid"}
        all_ids.extend(batch["ids"].tolist())
        row_ids.extend(batch["_rowid"].tolist())

    assert all_ids == [i for i in range(1000) if i // 100 % 2 == 1]
    assert set(row_ids) == {
        i % 100 + (i // 100 << 32) for i in range(1000) if i // 100 % 2 == 1
    }


def test_sample_fragments(tmp_path: Path):
    arr = pa.array(range(2000))
    tbl = pa.Table.from_arrays([arr], ["ids"])

    # Write 20 files
    lance.write_dataset(tbl, tmp_path, max_rows_per_file=100)

    ds = LanceDataset(
        tmp_path,
        batch_size=25,
        columns=["ids"],
        with_row_id=True,
        sampler=ShardedFragmentSampler(rank=1, world_size=2),
    )

    all_ids = list(chain.from_iterable([batch["ids"].cpu().numpy() for batch in ds]))
    assert all_ids == [i for i in range(2000) if i // 100 % 2 == 1]
