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

"""Read Lance dataset as torch DataPipe.
"""

# PEP-585. Can be removed after deprecating python 3.8 support.
from __future__ import annotations

from typing import TYPE_CHECKING

from torchdata.datapipes.iter import IterDataPipe

if TYPE_CHECKING:
    from .. import LanceDataset


class LanceDataLoader(IterDataPipe):
    def __init__(self, dataset: LanceDataset, batch_size: int, *args, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.batch_size = batch_size

