# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from typing import Any, Callable, Dict, List, Union

import pyarrow as pa
from ray.data import from_items

import lance

from .distribute_task import PARTITION_KEY, DispatchFragmentTasks

# ==============================================================================
# Component Keys
# ==============================================================================
ITEM_KEY = "item"
READ_COLUMNS_KEY = "read_columns"
ACTION_KEY = "action"
ADD_COLUMN_ACTION = "add_column"

# ==============================================================================
# Type Aliases
# ==============================================================================
RecordBatchTransformer = Callable[[pa.RecordBatch], pa.RecordBatch]


def execute_fragment_operation(
    task_dispatcher: "DispatchFragmentTasks",
    value_function: Union[Dict[str, str], RecordBatchTransformer],
    operation_parameters: Dict[str, Any] = None,
) -> None:
    """
    Execute distributed fragment operations and commit results.

    Args:
        task_dispatcher: Coordinator for fragment tasks
        value_function: Data transformation logic
        operation_parameters: Contextual parameters for the operation
    """
    operation_parameters = operation_parameters or {}

    # Generate and execute distributed tasks
    processing_tasks = task_dispatcher.get_tasks(value_function, operation_parameters)
    task_dataset = from_items(processing_tasks).map(lambda task: task[ITEM_KEY]())

    # Collect and commit results
    results = [item[PARTITION_KEY] for item in task_dataset.take_all()]
    task_dispatcher.commit_results(results)


def add_columns(
    dataset: lance.LanceDataset,
    column_generator: RecordBatchTransformer,
    source_columns: List[str],
) -> None:
    """
    Add new columns to a Lance dataset through distributed processing.

    Args:
        dataset: Target dataset for column addition
        column_generator: Function generating new column values
        source_columns: Existing columns required for generation
    """
    dispatcher = DispatchFragmentTasks(dataset)
    execute_fragment_operation(
        dispatcher,
        value_function=column_generator,
        operation_parameters={
            READ_COLUMNS_KEY: source_columns,
            ACTION_KEY: ADD_COLUMN_ACTION,
        },
    )
