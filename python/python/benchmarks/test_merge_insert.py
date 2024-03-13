# Copyright 2024 Lance Developers.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import shutil
import uuid
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pytest

NUM_ROWS = 10_000_000
NEW_ROWS = 10_000


def create_table(num_rows: int) -> pa.Table:
    keys = pa.array([uuid.uuid4().bytes.hex() for _ in range(num_rows)])
    values = pa.array(range(num_rows))
    return pa.table({"keys": keys, "keys_no_index": keys, "values": values})


def create_base_dataset(data_dir: Path) -> lance.LanceDataset:
    tmp_path = data_dir / "merge_insert_dataset"

    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    rows_remaining = NUM_ROWS
    dataset = None
    while rows_remaining > 0:
        next_batch_length = min(rows_remaining, 1024 * 1024)
        rows_remaining -= next_batch_length
        table = create_table(next_batch_length)
        dataset = lance.write_dataset(table, tmp_path, mode="append")

    dataset.create_scalar_index("keys", "BTREE")

    return dataset


def create_delete_dataset(base_dataset):
    dataset = lance.dataset(base_dataset.uri, version=base_dataset.version)
    dataset.restore()
    dataset.delete("values % 2 != 0")
    return dataset


def create_new_rows_dataset(base_dataset):
    dataset = lance.dataset(base_dataset.uri, version=base_dataset.version)
    dataset.restore()
    table = create_table(NEW_ROWS)
    return lance.write_dataset(table, base_dataset.uri, mode="append")


class Datasets(NamedTuple):
    # A clean dataset
    clean: lance.LanceDataset
    # A dataset where every fragment has a deletion vector
    with_delete_files: lance.LanceDataset
    # A dataset that has new, unindexed rows
    with_new_rows: lance.LanceDataset


@pytest.fixture(scope="module")
def datasets(data_dir: Path) -> Datasets:
    base_dataset = create_base_dataset(data_dir)
    delete_dataset = create_delete_dataset(base_dataset)
    new_rows_dataset = create_new_rows_dataset(base_dataset)
    return Datasets(
        clean=base_dataset,
        with_delete_files=delete_dataset,
        with_new_rows=new_rows_dataset,
    )


@pytest.fixture(scope="module", params=Datasets._fields)
def test_dataset(datasets: Datasets, request) -> lance.LanceDataset:
    return datasets.__getattribute__(request.param)


@pytest.mark.benchmark(group="merge_insert")
@pytest.mark.parametrize("use_scalar_index", (False, True))
def test_bulk_update(test_dataset, benchmark, use_scalar_index):
    test_dataset.cleanup_old_versions(older_than=timedelta(seconds=1))
    NUM_NEW_ROWS = 1000
    version = test_dataset.version
    indices = random.sample(range(test_dataset.count_rows()), NUM_NEW_ROWS)
    new_table = test_dataset.take(indices)
    values_idx = new_table.schema.get_field_index("values")
    new_col = pc.add(new_table.column(values_idx), 1)
    new_table.set_column(values_idx, new_table.schema.field(values_idx), new_col)
    key = "keys" if use_scalar_index else "keys_no_index"

    def bench_fn():
        bench_ds = test_dataset.checkout_version(version)
        bench_ds.restore()
        bench_ds.merge_insert(key).when_matched_update_all().execute(new_table)

    benchmark(bench_fn)
