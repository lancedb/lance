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
pytest configurations for unit tests.

For configuration that is shared between tests and benchmarks, see ../conftest.py
"""
import os
import sys

import pytest


@pytest.fixture(params=(True, False))
def provide_pandas(request, monkeypatch):
    if not request.param:
        monkeypatch.setitem(sys.modules, "pd", None)
    return request.param


@pytest.fixture
def s3_bucket() -> str:
    return os.environ.get("TEST_S3_BUCKET", "lance-integtest")


@pytest.fixture
def ddb_table() -> str:
    return os.environ.get("TEST_DDB_TABLE", "lance-integtest")
