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


# These are initialization hooks and must have an exact name for pytest to pick them up
# https://docs.pytest.org/en/7.1.x/reference/reference.html


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires S3 buckets to be setup with access)",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-integration"):
        skipper = pytest.mark.skip(reason="--run-integration not specified")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skipper)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test to run only on named environment"
    )
