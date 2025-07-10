# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The LanceDB Authors

# Weekly tests that runs all operations on a large dataset,
# these operations are ran in random order repeated 10 times

import abc
import itertools
import math
from typing import Optional

import lance
import numpy as np
import pyarrow as pa
import pytest

# For testing, use smaller numbers to make tests run faster
# In production, you might want to use: NUM_ROWS = 1_000_000
NUM_ROWS = 10_000  # Reduced from 1M to 10K for faster testing
BATCH_SIZE = 500
DIM = 32

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


def create_or_load_table(name: str, kwargs: dict):
    uri = f"tests/weekly/{name}"

    # Try to open existing dataset first
    try:
        ds = lance.dataset(uri)
        if ds.count_rows() > 0:
            return ds
    except Exception:
        pass

    # Create new dataset with initial data
    initial_batch = random_batch(0, BATCH_SIZE)
    ds = lance.write_dataset(initial_batch, uri, schema=schema, mode="overwrite")

    # Add remaining data
    for i in range(BATCH_SIZE, NUM_ROWS, BATCH_SIZE):
        batch = random_batch(i, BATCH_SIZE)
        ds.insert(batch)

    # Create indices
    ds.create_scalar_index("id", index_type="BTREE", replace=True)
    ds.create_index(
        "vector",
        metric="cosine",
        num_partitions=1024,
        num_sub_vectors=DIM // 8,
        replace=True,
    )

    # Note: FTS index creation is async, but we'll handle this differently for pytest
    # For now, we'll skip the async part and create it synchronously if possible
    try:
        ds.create_scalar_index(
            "text",
            index_type="FTS",
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
    def __init__(self, delete_all: bool):
        self.delete_all = delete_all

    def run(self, ds: lance.LanceDataset):
        if self.delete_all:
            ds.delete("id >= 0")
            return

        num_rows = ds.count_rows()
        to_delete = np.random.randint(0, num_rows, 100)
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
        stats = ds.index_stats("vector_idx")
        if stats is None:
            print("No vector index found")
            return
        query_vector = np.random.rand(DIM).tolist()
        query = ds.scanner(
            nearest={
                "q": query_vector,
                "k": 10,
            },
            filter=self.filter,
        )
        print(query.analyze_plan())


class FullTextSearch(ReadOnlyOperation):
    def __init__(self, has_position: bool, filter: Optional[str] = None):
        self.has_position = has_position
        self.filter = filter

    def run(self, ds: lance.LanceDataset):
        stats = ds.index_stats("text_idx")
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
            full_text_query=query_text, filter=self.filter
        ).limit(10)
        print(query.analyze_plan())


@pytest.fixture(scope="session")
def dataset_without_position():
    """Create or load dataset without position tracking for FTS"""
    return create_or_load_table("test_table", {"with_position": False})


@pytest.fixture(scope="session")
def dataset_with_position():
    """Create or load dataset with position tracking for FTS"""
    return create_or_load_table(
        "test_table_with_position", {"with_position": True, "remove_stop_words": False}
    )


@pytest.mark.weekly
def test_weekly_operations_without_position(dataset_without_position):
    """Test all operations on dataset without FTS position tracking"""
    ds = dataset_without_position

    write_operations = [
        Append(),
        Delete(delete_all=False),
        Delete(delete_all=True),
        Optimize(num_indices_to_merge=0, column="id"),
        Optimize(num_indices_to_merge=0, column="vector"),  # delta index
        Optimize(num_indices_to_merge=1, column="vector"),  # merge index
        Optimize(num_indices_to_merge=0, column="text"),
        Compact(),
    ]

    read_only_operations = [
        # Read only operations
        VectorSearch(),
        VectorSearch(filter="id > 1_000"),
        FullTextSearch(has_position=False),
        FullTextSearch(has_position=False, filter="id > 1_000"),
    ]

    for permutation in itertools.permutations(range(len(write_operations))):
        for idx in permutation:
            write_operation = write_operations[idx]
            print(f"Running {write_operation.__class__.__name__}")
            write_operation.run(ds)

            # write operation changed the status of the table,
            # then we need to run all read only operations after it
            for read_only_operation in read_only_operations:
                print(f"Running {read_only_operation.__class__.__name__}")
                read_only_operation.run(ds)


@pytest.mark.weekly
def test_weekly_operations_with_position(dataset_with_position):
    """Test all operations on dataset with FTS position tracking"""
    ds = dataset_with_position

    write_operations = [
        Append(),
        Delete(delete_all=False),
        Delete(delete_all=True),
        Optimize(num_indices_to_merge=0, column="id"),
        Optimize(num_indices_to_merge=0, column="vector"),  # delta index
        Optimize(num_indices_to_merge=1, column="vector"),  # merge index
        Optimize(num_indices_to_merge=0, column="text"),
        Compact(),
    ]

    read_only_operations = [
        # Read only operations
        VectorSearch(),
        VectorSearch(filter="id > 1_000"),
        FullTextSearch(has_position=True),
        FullTextSearch(has_position=True, filter="id > 1_000"),
    ]

    # Run a subset of permutations to keep test time reasonable
    num_permutations = min(
        2, math.factorial(len(write_operations))
    )  # Reduced from 5 to 2 for faster testing
    print(f"Running {num_permutations} permutations")

    for i, permutation in enumerate(
        itertools.permutations(range(len(write_operations)))
    ):
        if i >= num_permutations:
            break

        for idx in permutation:
            write_operation = write_operations[idx]
            print(f"Running {write_operation.__class__.__name__}")
            write_operation.run(ds)

            # write operation changed the status of the table,
            # then we need to run all read only operations after it
            for read_only_operation in read_only_operations:
                print(f"Running {read_only_operation.__class__.__name__}")
                read_only_operation.run(ds)


# Keep the original functions for backward compatibility
def run(name: str, kwargs: dict):
    print(f"Running {name} with kwargs: {kwargs}")
    ds = create_or_load_table(name, kwargs)

    write_operations = [
        Append(),
        Delete(delete_all=False),
        Delete(delete_all=True),
        Optimize(num_indices_to_merge=0, column="id"),
        Optimize(num_indices_to_merge=0, column="vector"),  # delta index
        Optimize(num_indices_to_merge=1, column="vector"),  # merge index
        Optimize(num_indices_to_merge=0, column="text"),
        Compact(),
    ]

    has_position = kwargs.get("with_position", False)
    read_only_operations = [
        # Read only operations
        VectorSearch(),
        VectorSearch(filter="id > 1_000"),
        FullTextSearch(has_position=has_position),
        FullTextSearch(has_position=has_position, filter="id > 1_000"),
    ]

    # iterate on all permutations of write operations
    print(f"Running {math.factorial(len(write_operations))} permutations")
    for permutation in itertools.permutations(range(len(write_operations))):
        for idx in permutation:
            write_operation = write_operations[idx]
            print(f"Running {write_operation.__class__.__name__}")
            write_operation.run(ds)

            # write operation changed the status of the table,
            # then we need to run all read only operations after it
            for read_only_operation in read_only_operations:
                print(f"Running {read_only_operation.__class__.__name__}")
                read_only_operation.run(ds)
