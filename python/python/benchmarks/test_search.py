# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
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
    filterable = pa.array(range(offset, offset + num_rows))
    return pa.table({"vector": vectors, "filterable": filterable})


def create_base_dataset(data_dir: Path) -> lance.LanceDataset:
    tmp_path = data_dir / "search_dataset"
    dataset = find_or_clean(tmp_path)
    if dataset:
        return dataset

    rows_remaining = NUM_ROWS
    offset = 0
    dataset = None
    while rows_remaining > 0:
        next_batch_length = min(rows_remaining, 1024 * 1024)
        rows_remaining -= next_batch_length
        table = create_table(next_batch_length, offset)
        if offset == 0:
            dataset = lance.write_dataset(table, tmp_path)
        else:
            dataset = lance.write_dataset(table, tmp_path, mode="append")
        offset += next_batch_length

    dataset.create_index(
        column="vector",
        index_type="IVF_PQ",
        metric_type="L2",
        num_partitions=32,
        num_sub_vectors=16,
        num_bits=8,
    )

    dataset.create_scalar_index("filterable", "BTREE")

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


@pytest.mark.benchmark(group="query_ann")
@pytest.mark.parametrize("selectivity", (0.25, 0.75))
@pytest.mark.parametrize("prefilter", (False, True))
@pytest.mark.parametrize("use_index", (False, True))
def test_filtered_search(test_dataset, benchmark, selectivity, prefilter, use_index):
    q = pc.random(N_DIMS).cast(pa.float32())
    threshold = int(round(selectivity * NUM_ROWS))
    result = benchmark(
        test_dataset.to_table,
        nearest=dict(
            column="vector",
            q=q,
            k=100,
            nprobes=10,
            use_index=use_index,
        ),
        prefilter=prefilter,
        filter=f"filterable <= {threshold}",
    )
    # With post-filtering it is possible we don't get any results
    if prefilter:
        assert result.num_rows > 0


@pytest.mark.benchmark(group="query_ann")
@pytest.mark.parametrize(
    "filter",
    (
        None,
        "filterable = 0",
        "filterable != 0",
        "filterable IN (0)",
        "filterable NOT IN (0)",
        "filterable != 0 AND filterable != 5000 AND filterable != 10000",
        "filterable NOT IN (0, 5000, 10000)",
        "filterable < 5000",
        "filterable > 5000",
    ),
    ids=[
        "none",
        "equality",
        "not_equality",
        "in_list_one",
        "not_in_list_one",
        "not_equality_and_chain",
        "not_in_list_three",
        "less_than_selective",
        "greater_than_not_selective",
    ],
)
def test_scalar_index_prefilter(test_dataset, benchmark, filter: str):
    q = pc.random(N_DIMS).cast(pa.float32())
    if filter is None:
        benchmark(
            test_dataset.to_table,
            nearest=dict(
                column="vector",
                q=q,
                k=100,
                nprobes=10,
            ),
        )
    else:
        benchmark(
            test_dataset.to_table,
            nearest=dict(
                column="vector",
                q=q,
                k=100,
                nprobes=10,
            ),
            prefilter=True,
            filter=filter,
        )


@pytest.mark.benchmark(group="query_no_vec")
@pytest.mark.parametrize(
    "filter",
    (
        None,
        "filterable = 0",
        "filterable != 0",
        "filterable IN (0)",
        "filterable IN (0, 5000, 10000)",
        "filterable NOT IN (0)",
        "filterable != 0 AND filterable != 5000 AND filterable != 10000",
        "filterable NOT IN (0, 5000, 10000)",
        "filterable < 5000",
        "filterable > 5000",
    ),
    ids=[
        "none",
        "equality",
        "not_equality",
        "in_list_one",
        "in_list_three",
        "not_in_list_one",
        "not_equality_and_chain",
        "not_in_list_three",
        "less_than_selective",
        "greater_than_not_selective",
    ],
)
def test_scalar_index_search(test_dataset, benchmark, filter: str):
    if filter is None:
        benchmark(
            test_dataset.to_table,
        )
    else:
        benchmark(
            test_dataset.to_table,
            prefilter=True,
            filter=filter,
        )
