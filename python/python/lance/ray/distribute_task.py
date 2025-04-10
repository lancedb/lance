# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import lance

# ==============================================================================
# Message Structure Constants
# ==============================================================================
TASK_ID_KEY = "task_id"
PARTITION_KEY = "partition"

# ==============================================================================
# Data Component Keys
# ==============================================================================
FRAGMENT_KEY = "fragment"
SCHEMA_KEY = "schema"

# ==============================================================================
# Operation Parameters
# ==============================================================================
PARAMS_KEY = "params"
ACTION_KEY = "action"
READ_COLUMNS_KEY = "read_columns"

# ==============================================================================
# Execution Metadata
# ==============================================================================
OPERATION_TYPE_KEY = "operation_type"
VERSION_KEY = "version"


@dataclass
class TaskInput:
    """Container for task execution parameters and metadata."""

    task_id: str
    fn: Callable
    fragment: Any
    params: Dict[str, Any] = field(default_factory=dict)


class FragmentTask:
    """Base class for distributed data processing tasks."""

    def __init__(self, task_input: TaskInput):
        self.task_input = task_input

    def __call__(self) -> Dict[str, Any]:
        output = self._fn()
        return {
            TASK_ID_KEY: self.task_input.task_id,
            PARTITION_KEY: {FRAGMENT_KEY: self.task_input.fragment, "output": output},
        }


class AddColumnTask(FragmentTask):
    """Task for adding new columns to dataset fragments."""

    def __init__(self, task_input: TaskInput, read_columns):
        super().__init__(task_input)
        self._read_columns = read_columns
        self._validate_input_params()

    def _validate_input_params(self) -> None:
        """Ensure required parameters are present and valid."""
        if self.task_input.fragment is None:
            raise ValueError("Fragment must be provided for column addition")

    def __call__(self) -> Dict[str, Any]:
        """Execute column addition and return updated fragment metadata."""
        new_fragment, new_schema = self.task_input.fragment.merge_columns(
            value_func=self.task_input.fn, columns=self._read_columns
        )
        return {
            TASK_ID_KEY: self.task_input.task_id,
            PARTITION_KEY: {FRAGMENT_KEY: new_fragment, SCHEMA_KEY: new_schema},
        }


class DispatchFragmentTasks:
    """Orchestrates distributed execution of fragment operations."""

    def __init__(self, dataset: lance.LanceDataset):
        self.dataset = dataset

    def get_tasks(
        self, transform_fn: Callable, operation_params: Optional[Dict[str, Any]] = None
    ) -> List[FragmentTask]:
        """Generate tasks for processing all dataset fragments."""
        operation_params = operation_params or {}
        return [
            self._create_task(fragment, transform_fn, operation_params)
            for fragment in self.dataset.get_fragments()
        ]

    def _create_task(
        self, fragment: Any, transform_fn: Callable, params: Dict[str, Any]
    ) -> FragmentTask:
        """Factory method for creating appropriate task type."""
        task_input = TaskInput(
            task_id=fragment.fragment_id,
            fn=transform_fn,
            fragment=fragment,
            params=params,
        )

        if params[ACTION_KEY] == "add_column":
            return AddColumnTask(task_input, params[READ_COLUMNS_KEY])

        raise ValueError(f"Unsupported operation: {params[ACTION_KEY]}")

    def commit_results(self, partitions: List[Dict[str, Any]]) -> bool:
        """Commit processed results to the dataset."""
        if not partitions:
            return False

        fragments = [part[FRAGMENT_KEY] for part in partitions]
        unified_schema = partitions[0][SCHEMA_KEY]

        operation = lance.LanceOperation.Merge(fragments, unified_schema)
        self.dataset.commit(
            base_uri=self.dataset.uri,
            operation=operation,
            read_version=self.dataset.version,
        )
        return True
