# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import pickle
from pathlib import Path
from typing import Callable, Optional, Union

import pyarrow as pa

from .. import LanceDataset, dataset
from .sink import LanceCommitter


class LanceMergeColumn:
    """Merge columns in a distributed way."""

    def __init__(
        self,
        value_func: Callable[[pa.RecordBatch], pa.RecordBatch],
        columns: Optional[list[str]] = None,
    ):
        self.value_func = value_func
        self.columns = columns

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        fragment = batch["item"]
        new_fragment, schema = fragment.merge_columns(self.value_func, self.columns)

        return {
            "fragment": pickle.dumps(new_fragment),
            "schema": pickle.dumps(schema),
        }


def merge_columns(
    data: Union[str, LanceDataset],
    value_func: Callable[[pa.RecordBatch], pa.RecordBatch],
    *,
    columns: Optional[list[str]] = None,
):
    """Run merge_columns distributedly with Ray."""
    import ray

    if isinstance(data, LanceDataset):
        ds = data
    else:
        ds = dataset(data)

    fragments = ds.get_fragments()

    ray_ds = ray.data.from_items(fragments)
    ray_ds.map(LanceMergeColumn(value_func, columns)).write_datasink(
        LanceCommitter(ds.uri, mode="merge")
    )
