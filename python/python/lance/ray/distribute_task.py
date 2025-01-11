# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
from typing import Any, Callable, Dict, List

from ray.data._internal.datasource.lance_datasource import LanceDatasource

import lance


class TaskCommitMessage:
    def __init__(self, task_id: str, partition: Dict[Any, Any]):
        self.task_id = task_id
        self.partition = partition

    def __getitem__(self, key):
        if key == "task_id":
            return self.task_id
        elif key == "partition":
            return self.partition
        else:
            raise KeyError(f"Key {key} not found")


class TaskInput:
    def __init__(self, task_id: str, fn: Callable, partition: Dict[Any, Any]):
        self.task_id = task_id
        self.fn = fn
        self.partition = partition


class CustomTask:
    def __init__(self, task_input: "TaskInput"):
        self._fn = task_input.fn
        self._task_id = task_input.task_id

    def __call__(self) -> Dict[str, Any]:
        output = self._fn()
        return {"commit_messages": TaskCommitMessage(self._task_id, {"output", output})}


class DistributeCustomTasks:
    def __init__(self, ray_lance_ds: LanceDatasource):
        self.ray_lance_ds = ray_lance_ds
        self.lance_ds = ray_lance_ds.lance_ds

    def get_custom_tasks(
        self,
        fn: Callable,
        params: Dict[str, Any] = None,
    ) -> List[CustomTask]:
        custom_tasks = []
        for fragment in self.lance_ds.get_fragments():
            task_input = TaskInput(
                fragment.fragment_id,
                fn=fn,
                partition={"fragment": fragment, "params": params},
            )
            if params["action"] == "add_column":
                custom_task = AddColumnTask(task_input)
            elif params["action"] == "delete_records":
                custom_task = DeleteRecordTask(task_input)
            else:
                custom_task = CustomTask(task_input)
            custom_tasks.append(custom_task)

        return custom_tasks

    def commit_tasks(self, commit_messages: List[TaskCommitMessage]) -> bool:
        merged_fragments = [item["partition"]["fragment"] for item in commit_messages]
        schema = commit_messages[0]["partition"]["schema"]
        operation = lance.LanceOperation.Merge(merged_fragments, schema)
        lance.LanceDataset.commit(
            self.ray_lance_ds.uri,
            operation,
            read_version=self.lance_ds.version,
            storage_options=self.ray_lance_ds.storage_options,
        )
        return True


class AddColumnTask(CustomTask):
    def __init__(self, task_input: TaskInput):
        self.fragment_id = task_input.task_id
        self.fn = task_input.fn
        assert task_input.partition["fragment"] is not None
        self.fragment = task_input.partition["fragment"]
        params = task_input.partition["params"]
        assert (
            params is not None
            and params["action"] == "add_column"
            and params["read_columns"] is not None
        )
        self.read_columns = params["read_columns"]

    def __call__(self) -> Dict[str, Any]:
        new_fragment, new_schema = self.fragment.merge_columns(
            self.fn, self.read_columns
        )
        return {
            "commit_messages": TaskCommitMessage(
                task_id=self.fragment_id,
                partition={"fragment": new_fragment, "schema": new_schema},
            )
        }


class DeleteRecordTask(CustomTask):
    def __init__(self, task_input: TaskInput):
        self.fragment_id = task_input.task_id
        assert (
            task_input.partition["fragment"] is not None
            and task_input.partition["delete_rows_predicate"] is not None
        )
        self.delete_rows_predicate = task_input.partition["delete_rows_predicate"]
        self.fragment = task_input.partition["fragment"]

    def __call__(self, *args, **kwargs) -> TaskCommitMessage:
        return self.fragment.delete_rows(*args, **kwargs)
