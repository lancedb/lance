# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Tests for the concurrnet transaction system

NOTE: I tried my best to make this not look like BDD, but it looks like BDD :(
"""

import dataclasses
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import lance
import pyarrow as pa
import pytest

OperationFactory = Callable[[lance.LanceDataset], lance.LanceOperation.BaseOperation]


@dataclasses.dataclass
class ParallelOperations:
    """
    A list of operations that are expected to run concurrently regardless of the order.

    The `expect` function should return True if the dataset is in a valid state after
    """

    starting_dataset: pa.Table
    operations: list[OperationFactory]
    expect: Callable[[lance.LanceDataset], bool]


_simple_schema = pa.schema(
    [
        pa.field("a", pa.int64()),
        pa.field("b", pa.int64()),
    ]
)


def empty_table(schema: pa.Schema) -> pa.Table:
    return pa.Table.from_batches([], schema=schema)


def append(table: pa.Table) -> OperationFactory:
    def _factory(dataset: lance.LanceDataset) -> lance.LanceOperation.Append:
        frags = lance.fragment.write_fragments(
            table,
            dataset,
        )
        return lance.LanceOperation.Append(frags)

    return _factory


pytestmark = pytest.mark.parametrize(
    "ops",
    [
        # Doing nothing on an empty dataset should result in an empty dataset
        ParallelOperations(
            starting_dataset=empty_table(_simple_schema),
            operations=[],
            expect=lambda ds: ds.count_rows() == 0,
        ),
        # appending three times concurrently should result in a dataset with 3 rows
        ParallelOperations(
            starting_dataset=empty_table(_simple_schema),
            operations=[
                append(pa.table({"a": [1], "b": [2]})),
                append(pa.table({"a": [1], "b": [2]})),
                append(pa.table({"a": [1], "b": [2]})),
            ],
            expect=lambda ds: ds.count_rows() == 3,
        ),
    ],
)


class TestConcurrentTransaction:
    def test_transaction_conflict_merge_single_threaded(
        self, tmp_path, ops: ParallelOperations
    ):
        """
        Test that a transaction conflict is resolved correctly when merging datasets
        """

        # Create a new dataset
        dataset = lance.write_dataset(ops.starting_dataset, tmp_path)

        # Run the operations
        for permutaed in itertools.permutations(ops.operations):
            for op in permutaed:
                op = op(dataset)
                lance.LanceDataset.commit(tmp_path, op, read_version=dataset.version)
            starting_version = dataset.version

            # re-open
            dataset = lance.dataset(tmp_path)
            # check the version is correct
            assert starting_version + len(ops.operations) == dataset.version, (
                "Versioning is incorrect"
            )
            # check data is not corrupted
            assert ops.expect(dataset), "Final state is not as expected"

            # rollback to the starting version
            dataset = lance.dataset(tmp_path, version=starting_version)
            dataset.restore()

    def test_transaction_conflict_merge_multi_threaded(
        self, tmp_path, ops: ParallelOperations
    ):
        """
        Test that a transaction conflict is resolved correctly when merging datasets
        """

        # Create a new dataset
        dataset = lance.write_dataset(ops.starting_dataset, tmp_path)

        executor = ThreadPoolExecutor()

        # Run the operations
        for permutaed in itertools.permutations(ops.operations):
            # barrier can not handle empty list
            if not permutaed:
                continue

            op_barrier = threading.Barrier(len(permutaed))
            commit_barrier = threading.Barrier(len(permutaed))
            futs = []

            for op in permutaed:

                def _do_op(op):
                    op_barrier.wait()
                    op = op(dataset)
                    commit_barrier.wait()
                    lance.LanceDataset.commit(
                        tmp_path, op, read_version=dataset.version
                    )

                futs.append(executor.submit(_do_op, op))
            futs = [f for f in as_completed(futs)]

            starting_version = dataset.version

            # re-open
            dataset = lance.dataset(tmp_path)
            # check the version is correct
            assert starting_version + len(ops.operations) == dataset.version, (
                "Versioning is incorrect"
            )
            # check data is not corrupted
            assert ops.expect(dataset), "Final state is not as expected"

            # rollback to the starting version
            dataset = lance.dataset(tmp_path, version=starting_version)
            dataset.restore()
