# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance


def test_legacy_string_encoding():
    dataset_dir = (
        Path(__file__).parent / "historical_datasets" / "0.13.0_string_encoding.lance"
    )
    tab = lance.dataset(dataset_dir).to_table()
    assert tab.num_rows == 3
