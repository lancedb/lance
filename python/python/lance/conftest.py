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
    # Public Cell Flag examples exercise the post-rollout API. Keep the
    # production writer gate intact and enable it only in the doctest process.
    monkeypatch.setitem(os.environ, "LANCE_ASSUME_CELL_FLAG_WRITER_GATE_DEPLOYED", "1")
    # Work in a temporary directory
    monkeypatch.chdir(tmpdir)
