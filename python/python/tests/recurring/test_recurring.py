# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

# Recurring tests that runs all operations on a large dataset,
# these operations are ran in random order repeated 10 times

import abc
import itertools
from datetime import timedelta
from typing import Optional

import lance
import numpy as np
import pyarrow as pa
import pytest

# For testing, use smaller numbers to make tests run faster
# In production, you might want to use: NUM_ROWS = 1_000_000
NUM_ROWS = 1_000_000
BATCH_SIZE = 1_000
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


def create_or_load_dataset(dataset_name: str, kwargs: dict):
    uri = f"tests/recurring/{dataset_name}"

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
    """Test all operations on dataset without FTS position tracking"""
    dataset_name = f"test_table_with_position_{with_position}"
    ds = create_or_load_dataset(dataset_name, {"with_position": with_position})

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

    for permutation in itertools.permutations(range(len(write_operations))):
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
