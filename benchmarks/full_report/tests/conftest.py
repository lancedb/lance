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

import pytest
import tempfile
import lance

from vbench.utils import make_random_vector_table


@pytest.fixture
def tempdir():
    with tempfile.TemporaryDirectory() as d:
        yield d
    # no cleanup needed, TemporaryDirectory will cleanup


@pytest.fixture
def random_dataset():
    # don't use the fixture above so users can get a different temdir
    with tempfile.TemporaryDirectory() as d:
        t = make_random_vector_table()
        yield lance.write_dataset(t, d)
        # no cleanup needed, TemporaryDirectory will cleanup
