# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import lance
import pyarrow as pa
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
    Version(lance.__version__) < Version("0.29.1.beta2"),  # at least 0.29.1-beta.2
    reason="Lance 0.29.1-beta.2 would ignore indices too new",
)
def test_pq_buffer():
    ds = lance.dataset(get_path("pq_in_schema"))
    # the index should be ignored, still able to query (brute force)
    q = pc.random(32).cast(pa.float32())
    ds.to_table(
        nearest={
            "q": q,
            "k": 4,
            "column": "vec",
        }
    )
