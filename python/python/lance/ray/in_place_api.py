# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from typing import Any, Callable, Dict, List, Union

import pyarrow as pa
from ray.data import from_items

from .distribute_task import DistributeCustomTasks


def custom_inplace(
    distributed_custom_tasks: DistributeCustomTasks,
    value_fn: Union[Dict[str, str], Callable[[pa.RecordBatch], pa.RecordBatch]],
    params: Dict[str, Any] = None,
):
    custom_tasks = distributed_custom_tasks.get_custom_tasks(value_fn, params)
    ds = from_items(custom_tasks).map(lambda t: t["item"]())
    commit_messages = [item["commit_messages"] for item in ds.take_all()]
    distributed_custom_tasks.commit_tasks(commit_messages)


def add_columns(
    distributed_custom_tasks: DistributeCustomTasks,
    value_fn: Callable[[pa.RecordBatch], pa.RecordBatch],
    read_columns: List[str],
):
    custom_inplace(
        distributed_custom_tasks,
        value_fn=value_fn,
        params={"read_columns": read_columns, "action": "add_column"},
    )


def delete_rows(
    distributed_custom_tasks: DistributeCustomTasks,
    value_fn: Callable[[pa.RecordBatch], pa.RecordBatch],
    delete_rows_predicate: str,
):
    custom_inplace(
        distributed_custom_tasks,
        value_fn=value_fn,
        params={
            "delete_rows_predicate": delete_rows_predicate,
            "action": "delete_rows",
        },
    )
