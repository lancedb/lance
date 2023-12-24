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

from typing import Optional

from lance.fragment import FragmentMetadata
from lance.progress import FragmentWriteProgress


class ProgressForTest(FragmentWriteProgress):
    def __init__(self):
        super().__init__()
        self.begin_called = 0
        self.complete_called = 0

    def begin(
        self, fragment: FragmentMetadata, multipart_id: Optional[str] = None, **kwargs
    ):
        self.begin_called += 1

    def complete(self, fragment: FragmentMetadata):
        self.complete_called += 1
