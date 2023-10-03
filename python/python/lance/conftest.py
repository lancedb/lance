import os

import pytest


@pytest.fixture(autouse=True)
def doctest_setup(monkeypatch, tmpdir):
    # Make sure we have a consistent seed
    import numpy.random

    numpy.random.seed(42)

    # disable color for doctests so we don't have to include
    # escape codes in docstrings
    monkeypatch.setitem(os.environ, "NO_COLOR", "1")
    # Explicitly set the column width
    monkeypatch.setitem(os.environ, "COLUMNS", "80")
    # Work in a temporary directory
    monkeypatch.chdir(tmpdir)
