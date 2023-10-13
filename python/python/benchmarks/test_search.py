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
import shutil
from pathlib import Path
from typing import NamedTuple, Union

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pytest

N_DIMS = 768
NUM_ROWS = 100_000
NEW_ROWS = 10_000


def find_or_clean(dataset_path: Path) -> Union[lance.LanceDataset, None]:
    if dataset_path.exists():
        try:
            dataset = lance.LanceDataset(dataset_path)
        except Exception:
            pass
        else:
            return dataset

    # clear any old data there
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    return None


def create_table(num_rows, offset) -> pa.Table:
    values = pc.random(num_rows * N_DIMS).cast(pa.float32())
    vectors = pa.FixedSizeListArray.from_arrays(values, N_DIMS)
    filterable = pa.array(range(start=offset, stop=offset + num_rows))
    return pa.table({"vector": vectors, "filterable": filterable})


def create_base_dataset(data_dir: Path) -> lance.LanceDataset:
    tmp_path = data_dir / "search_dataset"
    dataset = find_or_clean(tmp_path)
    if dataset:
        return dataset

    table = create_table(NUM_ROWS, offset=0)
    dataset = lance.write_dataset(table, tmp_path)

    dataset.create_index(
        column="vector",
        index_type="IVF_PQ",
        metric_type="L2",
        num_partitions=32,
        num_sub_vectors=16,
        num_bits=8,
    )

    return dataset


def create_delete_dataset(data_dir):
    tmp_path = data_dir / "search_dataset_with_delete"
    dataset = find_or_clean(tmp_path)
    if dataset:
        return dataset

    clean_path = data_dir / "search_dataset"
    shutil.copytree(clean_path, tmp_path)

    dataset = lance.dataset(tmp_path)
    dataset.delete("filterable % 2 != 0")

    return dataset


def create_new_rows_dataset(data_dir):
    tmp_path = data_dir / "search_dataset_with_new_rows"
    dataset = find_or_clean(tmp_path)
    if dataset:
        return dataset

    clean_path = data_dir / "search_dataset"
    shutil.copytree(clean_path, tmp_path)

    dataset = lance.dataset(tmp_path)
    table = create_table(NEW_ROWS, offset=NUM_ROWS)
    dataset = lance.write_dataset(table, tmp_path, mode="append")

    return dataset


class Datasets(NamedTuple):
    # A clean dataset
    clean: lance.LanceDataset
    # A dataset where every fragment has a deletion vector
    with_delete_files: lance.LanceDataset
    # A dataset that has new, unindexed rows
    with_new_rows: lance.LanceDataset


@pytest.fixture(scope="module")
def datasets(data_dir: Path) -> Datasets:
    return Datasets(
        clean=create_base_dataset(data_dir),
        with_delete_files=create_delete_dataset(data_dir),
        with_new_rows=create_new_rows_dataset(data_dir),
    )


@pytest.fixture(scope="module", params=Datasets._fields)
def test_dataset(datasets: Datasets, request) -> lance.LanceDataset:
    return datasets.__getattribute__(request.param)


@pytest.mark.benchmark(group="query_ann")
def test_knn_search(test_dataset, benchmark):
    q = pc.random(N_DIMS).cast(pa.float32())
    result = benchmark(
        test_dataset.to_table,
        nearest=dict(
            column="vector",
            q=q,
            k=100,
            nprobes=10,
            use_index=False,
        ),
    )
    assert result.num_rows > 0


@pytest.mark.benchmark(group="query_ann")
def test_flat_index_search(test_dataset, benchmark):
    q = pc.random(N_DIMS).cast(pa.float32())
    result = benchmark(
        test_dataset.to_table,
        nearest=dict(
            column="vector",
            q=q,
            k=100,
            nprobes=10,
        ),
    )
    assert result.num_rows > 0


@pytest.mark.benchmark(group="query_ann")
def test_ivf_pq_index_search(test_dataset, benchmark):
    q = pc.random(N_DIMS).cast(pa.float32())
    result = benchmark(
        test_dataset.to_table,
        nearest=dict(
            column="vector",
            q=q,
            k=100,
            nprobes=10,
            refine_factor=2,
        ),
    )
    assert result.num_rows > 0
