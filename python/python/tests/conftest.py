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
