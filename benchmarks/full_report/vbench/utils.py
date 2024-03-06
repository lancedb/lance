#  Copyright (c) 2024. Lance Developers
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

import numpy as np
import pyarrow as pa


def make_random_vector_table(
    dims=128,
    nrows=1000,
):
    data = np.random.standard_normal((nrows, dims))
    schema = pa.schema(
        [
            pa.field("vec", pa.list_(pa.float32(), dims)),
            pa.field("id", pa.uint32(), False),
        ]
    )
    return pa.table(
        {
            "vec": pa.array(data.tolist(), type=pa.list_(pa.float32(), dims)),
            "id": pa.array(np.arange(nrows, dtype=np.uint32)),
        },
        schema=schema,
    )
