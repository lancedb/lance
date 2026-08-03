# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Unit tests for merge_insert CI datagen readiness checks.

These live under ``python/tests`` (not ``ci_benchmarks/``) so PR CI can collect
them without putting the source tree on ``sys.path`` ahead of the installed
wheel. Import ``lance`` first, then expose ``ci_benchmarks`` for the helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Load the installed native extension before adding the source tree for
# ``ci_benchmarks``. Otherwise ``import lance`` resolves to pure-Python sources
# without ``lance.lance`` and collection fails with BlobDescriptor ImportError.
import lance  # noqa: F401
import numpy as np

_PYTHON_SRC = Path(__file__).resolve().parents[1]
if str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))

from ci_benchmarks.datagen.merge_insert import (  # noqa: E402
    BASE_TAG,
    NARROW_SCHEMA,
    _already_generated,
    _tag_base,
    narrow_batch,
)


def test_already_generated_missing_dataset(tmp_path):
    assert not _already_generated(str(tmp_path / "missing"), expected_rows=10)


def test_already_generated_missing_base_tag(tmp_path):
    """Incomplete generation: dataset exists but the base tag was never written."""
    uri = str(tmp_path / "no_tag")
    lance.write_dataset(
        narrow_batch(np.arange(10, dtype=np.int64)),
        uri,
        schema=NARROW_SCHEMA,
    )
    assert not _already_generated(uri, expected_rows=10)


def test_already_generated_wrong_row_count(tmp_path):
    uri = str(tmp_path / "wrong_rows")
    ds = lance.write_dataset(
        narrow_batch(np.arange(10, dtype=np.int64)),
        uri,
        schema=NARROW_SCHEMA,
    )
    _tag_base(ds)
    assert not _already_generated(uri, expected_rows=20)


def test_already_generated_ready(tmp_path):
    uri = str(tmp_path / "ready")
    ds = lance.write_dataset(
        narrow_batch(np.arange(10, dtype=np.int64)),
        uri,
        schema=NARROW_SCHEMA,
    )
    _tag_base(ds)
    assert _already_generated(uri, expected_rows=10)
    assert BASE_TAG in lance.dataset(uri).tags.list()
