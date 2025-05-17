# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import lance
import pyarrow.compute as pc
import pytest
from lance.file import LanceFileReader
from packaging.version import Version

from .datagen import build_basic_types, build_large, get_path


@pytest.mark.forward
def test_scans():
    expected_basic_types = build_basic_types()
    actual_basic_types = (
        LanceFileReader(str(get_path("basic_types.lance"))).read_all().to_table()
    )
    assert actual_basic_types.equals(expected_basic_types)

    expected_large = build_large()
    actual_large = LanceFileReader(str(get_path("large.lance"))).read_all().to_table()
    assert actual_large.equals(expected_large)


@pytest.mark.forward
@pytest.mark.skipif(
    Version(lance.__version__).release >= (0, 25, 0),  # at least 0.25.0
    reason="Lance 0.25.0 can read v3 indices",
)
def test_pq_buffer():
    ds = lance.dataset(get_path("pq_in_schema"))
    q = pc.random(32).cast("float32")
    ds.to_table(
        nearest={
            "q": q,
            "k": 4,
            "column": "vec",
        }
    )
