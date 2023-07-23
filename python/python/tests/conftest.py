import sys

import pytest


@pytest.fixture(params=(True, False))
def provide_pandas(request, monkeypatch):
    if not request.param:
        monkeypatch.setitem(sys.modules, "pd", None)
    return request.param
