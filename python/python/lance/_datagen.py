#  Copyright (c) 2023. Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

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
