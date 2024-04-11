# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

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
