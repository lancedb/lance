# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

# Recurring tests that run all permutations of write operations on a large dataset.

import abc
import itertools
import math
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Mapping, Optional

import lance
import numpy as np
import pyarrow as pa
import pytest

DEFAULT_NUM_ROWS = 1_000_000
BATCH_SIZE = 1_000
DIM = 32

NUM_ROWS_ENV = "LANCE_RECURRING_NUM_ROWS"
SHARD_INDEX_ENV = "LANCE_RECURRING_SHARD_INDEX"
SHARD_COUNT_ENV = "LANCE_RECURRING_SHARD_COUNT"
MAX_PERMUTATIONS_ENV = "LANCE_RECURRING_MAX_PERMUTATIONS"


@dataclass(frozen=True)
class RecurringTestConfig:
    num_rows: int = DEFAULT_NUM_ROWS
    shard_index: int = 0
    shard_count: int = 1
    max_permutations: Optional[int] = None

    def __post_init__(self):
        if self.num_rows <= 0:
            raise ValueError(f"{NUM_ROWS_ENV} must be greater than 0")
        if self.shard_count <= 0:
            raise ValueError(f"{SHARD_COUNT_ENV} must be greater than 0")
        if self.shard_index < 0:
            raise ValueError(f"{SHARD_INDEX_ENV} must be greater than or equal to 0")
        if self.shard_index >= self.shard_count:
            raise ValueError(
                f"{SHARD_INDEX_ENV} must be less than {SHARD_COUNT_ENV} "
                f"({self.shard_count}), got {self.shard_index}"
            )
        if self.max_permutations is not None and self.max_permutations <= 0:
            raise ValueError(f"{MAX_PERMUTATIONS_ENV} must be greater than 0")


def _read_env_int(
    environ: Mapping[str, str], name: str, default: Optional[int]
) -> Optional[int]:
    value = environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {value!r}") from None


def _recurring_test_config(
    environ: Optional[Mapping[str, str]] = None,
) -> RecurringTestConfig:
    if environ is None:
        environ = os.environ

    num_rows = _read_env_int(environ, NUM_ROWS_ENV, DEFAULT_NUM_ROWS)
    shard_index = _read_env_int(environ, SHARD_INDEX_ENV, 0)
    shard_count = _read_env_int(environ, SHARD_COUNT_ENV, 1)
    max_permutations = _read_env_int(environ, MAX_PERMUTATIONS_ENV, None)
    assert num_rows is not None
    assert shard_index is not None
    assert shard_count is not None
    return RecurringTestConfig(
        num_rows=num_rows,
        shard_index=shard_index,
        shard_count=shard_count,
        max_permutations=max_permutations,
    )


def _permutation_bounds(
    total_permutations: int, config: RecurringTestConfig
) -> tuple[int, int]:
    if total_permutations <= 0:
        raise ValueError("total_permutations must be greater than 0")

    selected_permutations = config.max_permutations or total_permutations
    if selected_permutations > total_permutations:
        raise ValueError(
            f"{MAX_PERMUTATIONS_ENV} ({selected_permutations}) cannot exceed the "
            f"{total_permutations} available permutations"
        )

    permutations_per_shard, remainder = divmod(
        selected_permutations, config.shard_count
    )
    start = config.shard_index * permutations_per_shard + min(
        config.shard_index, remainder
    )
    shard_size = permutations_per_shard
    if config.shard_index < remainder:
        shard_size += 1
    return start, start + shard_size


def _sharded_permutations(num_operations: int, config: RecurringTestConfig):
    if num_operations <= 0:
        raise ValueError("num_operations must be greater than 0")

    start, stop = _permutation_bounds(math.factorial(num_operations), config)
    permutations = itertools.permutations(range(num_operations))
    return itertools.islice(permutations, start, stop)


schema = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("vector", pa.list_(pa.float32(), DIM)),
        pa.field("text", pa.string()),
    ]
)
words = ["hello", "world", "this", "is", "a", "test", "sentence"]


def random_text(num_words: int) -> str:
    return " ".join(np.random.choice(words, num_words))


def random_batch(start_id: int, batch_size: int) -> pa.Table:
    return pa.Table.from_arrays(
        [
            pa.array(np.arange(start_id, start_id + batch_size)),
            pa.array(np.random.rand(batch_size, DIM).tolist()),
            pa.array(
                [random_text(np.random.randint(1, 10)) for _ in range(batch_size)]
            ),
        ],
        schema=schema,
    )


def create_or_load_dataset(dataset_name: str, kwargs: dict, num_rows: int):
    uri = f"tests/recurring/{dataset_name}"

    # Try to open existing dataset first
    try:
        ds = lance.dataset(uri)
        if ds.count_rows() > 0:
            return ds
    except Exception:
        pass

    # Create new dataset with initial data
    initial_batch_size = min(BATCH_SIZE, num_rows)
    initial_batch = random_batch(0, initial_batch_size)
    ds = lance.write_dataset(initial_batch, uri, schema=schema, mode="overwrite")

    # Add remaining data
    for i in range(initial_batch_size, num_rows, BATCH_SIZE):
        batch = random_batch(i, min(BATCH_SIZE, num_rows - i))
        ds.insert(batch)

    # Create indices
    ds.create_scalar_index("id", index_type="BTREE", replace=True)
    ds.create_index(
        "vector",
        index_type="IVF_PQ",
        metric="cosine",
        num_partitions=128,
        num_sub_vectors=DIM // 8,
        replace=True,
    )

    # Note: FTS index creation is async, but we'll handle this differently for pytest
    # For now, we'll skip the async part and create it synchronously if possible
    try:
        ds.create_scalar_index(
            "text",
            index_type="INVERTED",
            with_position=kwargs.get("with_position", False),
            replace=True,
        )
    except Exception as e:
        print(f"Warning: Could not create FTS index: {e}")

    return ds


class Operation(abc.ABC):
    @abc.abstractmethod
    def read_only(self) -> bool: ...

    @abc.abstractmethod
    def run(self, ds: lance.LanceDataset): ...


class ReadOnlyOperation(Operation):
    def read_only(self) -> bool:
        return True


class WriteOperation(Operation):
    def read_only(self) -> bool:
        return False


class Append(WriteOperation):
    def run(self, ds: lance.LanceDataset):
        batch = random_batch(ds.count_rows(), BATCH_SIZE)
        ds.insert(batch)


class Delete(WriteOperation):
    def __init__(self, delete_num_rows: int = 100):
        self.delete_num_rows = delete_num_rows

    def run(self, ds: lance.LanceDataset):
        num_rows = ds.count_rows()
        to_delete = np.random.randint(0, num_rows, self.delete_num_rows)
        to_delete = ", ".join([str(v) for v in to_delete])
        ds.delete(f"id IN ({to_delete})")


class Optimize(WriteOperation):
    def __init__(self, num_indices_to_merge: int, column: str):
        self.num_indices_to_merge = num_indices_to_merge
        self.column = column

    def run(self, ds: lance.LanceDataset):
        ds.optimize.optimize_indices(
            num_indices_to_merge=self.num_indices_to_merge,
            index_names=[f"{self.column}_idx"],
        )


class Compact(WriteOperation):
    def run(self, ds: lance.LanceDataset):
        ds.optimize.compact_files()


class VectorSearch(ReadOnlyOperation):
    def __init__(self, filter: Optional[str] = None):
        self.filter = filter

    def run(self, ds: lance.LanceDataset):
        stats = ds.stats.index_stats("vector_idx")
        if stats is None:
            print("No vector index found")
            return
        query_vector = np.random.rand(DIM).tolist()
        query = ds.scanner(
            nearest={
                "q": query_vector,
                "k": 10,
                "column": "vector",
            },
            filter=self.filter,
        )
        query.analyze_plan()


class FullTextSearch(ReadOnlyOperation):
    def __init__(self, has_position: bool, filter: Optional[str] = None):
        self.has_position = has_position
        self.filter = filter

    def run(self, ds: lance.LanceDataset):
        stats = ds.stats.index_stats("text_idx")
        if stats is None:
            print("No text index found")
            return
        query_text = random_text(np.random.randint(1, 10))
        self.do_query(ds, query_text)

        if self.has_position:
            query_text = f'"{query_text}"'
            self.do_query(ds, query_text)

    def do_query(self, ds: lance.LanceDataset, query_text: str):
        query: lance.LanceScanner = ds.scanner(
            full_text_query=query_text,
            filter=self.filter,
            limit=10,
        )
        query.analyze_plan()


@pytest.mark.recurring
@pytest.mark.parametrize("with_position", [True])
def test_all_permutations(with_position):
    """Test each write ordering assigned to this recurring-test shard."""
    config = _recurring_test_config()
    permutation_limit = config.max_permutations or "all"
    dataset_name = (
        f"test_table_with_position_{with_position}_{config.num_rows}_rows_"
        f"{permutation_limit}_permutations_"
        f"shard_{config.shard_index}_of_{config.shard_count}"
    )
    ds = create_or_load_dataset(
        dataset_name, {"with_position": with_position}, config.num_rows
    )

    write_operations = [
        Append(),
        Delete(delete_num_rows=1000),
        Optimize(num_indices_to_merge=0, column="id"),
        Optimize(num_indices_to_merge=0, column="vector"),  # delta index
        Optimize(num_indices_to_merge=1, column="vector"),  # merge index
        Optimize(num_indices_to_merge=0, column="text"),
        Compact(),
    ]

    read_only_operations = [
        # Read only operations
        VectorSearch(),
        VectorSearch(filter="id >= 1000 and id < 8000"),
        FullTextSearch(has_position=False),
        FullTextSearch(has_position=False, filter="id >= 1000 and id < 8000"),
    ]

    permutation_start, permutation_stop = _permutation_bounds(
        math.factorial(len(write_operations)), config
    )
    print(
        f"Running recurring-test shard {config.shard_index}/{config.shard_count}: "
        f"permutations [{permutation_start}, {permutation_stop})"
    )
    for permutation_index, permutation in enumerate(
        _sharded_permutations(len(write_operations), config),
        start=permutation_start,
    ):
        print(f"Running permutation {permutation_index}: {permutation}")
        for idx in permutation:
            write_operation = write_operations[idx]
            print(f"Running {write_operation.__class__.__name__}")
            write_operation.run(ds)
            ds.cleanup_old_versions(older_than=timedelta(seconds=0))

            # write operation changed the status of the table,
            # then we need to run all read only operations after it
            for read_only_operation in read_only_operations:
                print(f"Running {read_only_operation.__class__.__name__}")
                read_only_operation.run(ds)


def test_recurring_config_defaults_to_full_workload():
    config = _recurring_test_config({})

    assert config == RecurringTestConfig()
    assert list(_sharded_permutations(3, config)) == list(
        itertools.permutations(range(3))
    )


def test_sharded_permutations_are_balanced_complete_and_disjoint():
    expected = list(itertools.permutations(range(4)))
    shards = [
        list(
            _sharded_permutations(
                4, RecurringTestConfig(shard_index=index, shard_count=5)
            )
        )
        for index in range(5)
    ]
    combined = list(itertools.chain.from_iterable(shards))

    assert [len(shard) for shard in shards] == [5, 5, 5, 5, 4]
    assert combined == expected
    assert len(set(combined)) == len(expected)


def test_max_permutations_is_applied_before_sharding():
    expected = list(itertools.islice(itertools.permutations(range(3)), 3))
    shards = [
        list(
            _sharded_permutations(
                3,
                RecurringTestConfig(
                    shard_index=index, shard_count=5, max_permutations=3
                ),
            )
        )
        for index in range(5)
    ]

    assert [len(shard) for shard in shards] == [1, 1, 1, 0, 0]
    assert list(itertools.chain.from_iterable(shards)) == expected


@pytest.mark.parametrize(
    ("environ", "error"),
    [
        ({NUM_ROWS_ENV: "not-an-integer"}, f"{NUM_ROWS_ENV} must be an integer"),
        ({NUM_ROWS_ENV: "0"}, f"{NUM_ROWS_ENV} must be greater than 0"),
        ({SHARD_COUNT_ENV: "0"}, f"{SHARD_COUNT_ENV} must be greater than 0"),
        (
            {SHARD_INDEX_ENV: "-1"},
            f"{SHARD_INDEX_ENV} must be greater than or equal to 0",
        ),
        (
            {SHARD_INDEX_ENV: "2", SHARD_COUNT_ENV: "2"},
            f"{SHARD_INDEX_ENV} must be less than {SHARD_COUNT_ENV}",
        ),
        (
            {MAX_PERMUTATIONS_ENV: "0"},
            f"{MAX_PERMUTATIONS_ENV} must be greater than 0",
        ),
    ],
)
def test_recurring_config_rejects_invalid_values(environ, error):
    with pytest.raises(ValueError, match=error):
        _recurring_test_config(environ)


def test_max_permutations_cannot_exceed_available_permutations():
    config = RecurringTestConfig(max_permutations=7)

    with pytest.raises(ValueError, match="cannot exceed the 6 available permutations"):
        list(_sharded_permutations(3, config))
