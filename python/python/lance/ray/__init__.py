# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Ray integration."""

from typing import Union

import ray.data

from .. import LanceDataset, dataset
from .schema import merge_columns

__all__ = [
    "merge_columns",
]


def fragments(data: Union[str, LanceDataset]):
    """Get fragments from a dataset."""
    if isinstance(data, LanceDataset):
        ds = data
    else:
        ds = dataset(data)

    return ray.data.from_items(ds.get_fragments())
