# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Unit tests for merge_insert datagen helpers."""

from __future__ import annotations

import lance
import numpy as np

from ci_benchmarks.datagen.merge_insert import (
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
