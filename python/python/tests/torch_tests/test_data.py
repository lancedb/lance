# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import shutil
from itertools import chain
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
from lance.sampler import ShardedBatchSampler, ShardedFragmentSampler

torch = pytest.importorskip("torch")
from lance.torch.data import LanceDataset  # noqa: E402


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
        ds,
        batch_size=256,
        samples=4096,
        columns=["ids", "vec"],
        cache=True,
        batch_readahead=2,
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


def test_iter_filter(tmp_path):
    arr = pa.array(range(1000))
    tbl = pa.Table.from_arrays([arr], ["ids"])

    ds = lance.write_dataset(tbl, tmp_path / "data.lance", max_rows_per_group=32)

    def check(dataset):
        total_rows = 0
        for batch in dataset:
            assert torch.where(batch >= 300, True, False).all()
            total_rows += batch.size(dim=0)
            assert batch.dtype == torch.int64
        assert total_rows == 700

    # No shard_grandularity
    check(
        LanceDataset(
            ds,
            batch_size=10,
            filter="ids >= 300",
            columns=["ids"],
        )
    )

    # shard_grandularity fragment ok
    check(
        LanceDataset(
            ds,
            batch_size=10,
            filter="ids >= 300",
            columns=["ids"],
            sampler=ShardedFragmentSampler(0, 1),
        )
    )

    # sampling fails
    with pytest.raises(ValueError):
        LanceDataset(
            ds,
            batch_size=10,
            filter="ids >= 300",
            samples=100,
            columns=["ids"],
        )


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


def test_sample_batches(tmp_path: Path):
    arr = pa.array(range(2000))
    tbl = pa.Table.from_arrays([arr], ["ids"])

    # Write 20 files
    lance.write_dataset(tbl, tmp_path, max_rows_per_file=100)

    ds = LanceDataset(
        tmp_path,
        batch_size=25,
        columns=["ids"],
        with_row_id=True,
        sampler=ShardedBatchSampler(rank=1, world_size=2),
    )

    all_ids = list(chain.from_iterable([batch.cpu().numpy() for batch in ds]))
    assert all_ids == [i for i in range(2000) if i // 25 % 2 == 1]


def test_sample_batches_with_filter(tmp_path: Path):
    NUM_ROWS = 10000
    tbl = pa.Table.from_pydict(
        {
            "id": range(NUM_ROWS),
            "filterme": [i % 2 for i in range(NUM_ROWS)],
        }
    )

    lance.write_dataset(tbl, tmp_path, max_rows_per_file=2000)

    ds = LanceDataset(
        tmp_path,
        batch_size=25,
        columns=["id"],
        with_row_id=True,
        filter="filterme == 0",
        sampler=ShardedBatchSampler(rank=3, world_size=5),
    )

    # The filtered sequence is 0, 2, 4, ...
    #
    # With rank 3 and world size 5 we should get
    #
    # - - - 6  -
    # - - - 16 -
    # - - - 26 -
    # ...
    all_ids = list(chain.from_iterable([batch.cpu().numpy() for batch in ds]))
    # Half of the data is filtered out, divided amongst 5 workers s
    # each should see 1/10th of the data
    assert len(all_ids) == 1000
    assert all_ids == [6 + (10 * i) for i in range(len(all_ids))]

    # Now test with random order
    ds = LanceDataset(
        tmp_path,
        batch_size=25,
        columns=["id"],
        with_row_id=True,
        filter="filterme == 0",
        sampler=ShardedBatchSampler(rank=3, world_size=5, randomize=True),
    )

    randomized_ids = list(chain.from_iterable([batch.cpu().numpy() for batch in ds]))
    assert randomized_ids != all_ids
    randomized_ids.sort()
    assert randomized_ids == all_ids


@pytest.mark.parametrize("dtype", [np.uint8, np.int64])
def test_convert_int_tensors(tmp_path: Path, dtype):
    data = np.random.randint(0, 256, size=128 * 32, dtype=dtype)
    fsl = pa.FixedSizeListArray.from_arrays(data, 32)
    ids = pa.array(range(0, 128), type=pa.int32())
    tbl = pa.Table.from_arrays([ids, fsl], ["ids", "vec"])

    ds = lance.write_dataset(tbl, tmp_path / "data.lance", max_rows_per_group=32)

    torch_ds = LanceDataset(
        ds,
        batch_size=4,
    )
    first = next(iter(torch_ds))
    assert first["vec"].dtype == torch.uint8 if dtype == np.uint8 else torch.int64
    assert first["vec"].shape == (4, 32)
