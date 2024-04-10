# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
An internal module for generating Arrow data for use in testing and benchmarking.
"""

import pyarrow as pa

from .lance import datagen


def is_datagen_supported():
    return datagen.is_datagen_supported()


def rand_batches(
    schema: pa.Schema, *, num_batches: int = None, batch_size_bytes: int = None
):
    if not datagen.is_datagen_supported():
        raise NotImplementedError(
            "This version of lance was not built with the datagen feature"
        )
    return datagen.rand_batches(schema, num_batches, batch_size_bytes)
