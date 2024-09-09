# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import pytest
from lance.file import LanceFileReader

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
