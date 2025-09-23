# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import os

import numpy.random
import pytest

# Make sure we have a consistent seed
numpy.random.seed(42)


@pytest.fixture(autouse=True)
def doctest_setup(monkeypatch, tmpdir):
    # disable color for doctests so we don't have to include
    # escape codes in docstrings
    monkeypatch.setitem(os.environ, "NO_COLOR", "1")
    # Explicitly set the column width
    monkeypatch.setitem(os.environ, "COLUMNS", "80")

    # Set up test data directory for doctests
    # This allows doctests to find test files regardless of working directory
    if "LANCE_TEST_DATA_DIR" not in os.environ:
        # Default to relative path from this file's location
        test_data_dir = os.path.join(os.path.dirname(__file__), "../tests")
        monkeypatch.setitem(os.environ, "LANCE_TEST_DATA_DIR", test_data_dir)

    # Work in a temporary directory
    monkeypatch.chdir(tmpdir)
