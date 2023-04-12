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
#

"""Dataset Fragment"""

from typing import Union, Optional, Iterator

import pyarrow as pa


class LanceFragment(pa.dataset.Fragment):
    def __init__(self, dataset: "LanceDataset", fragment_id: int):
        self._ds = dataset
        self._id = fragment_id
        self._fragmnet = dataset.get

    @property
    def fragment_id(self):
        return self.id

    def count_rows(self, filter: Union[pa.compute.Expression, str]) -> int:
        pass

    def head(self, num_rows: int) -> pa.Table:
        pass

    def take(self, indices) -> pa.Table:
        pass

    def to_batches(
        self, schema: Optional[pa.Schema], **kwargs
    ) -> Iterator[pa.RecordBatch]:
        pass

    def to_table(self, schema: Optional[pa.Schema] = None, **kwargs) -> pa.Table:
        pass
