# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import pytest

datasets = pytest.importorskip("datasets")


def test_write_hf_dataset(tmp_path: Path):
    hf_ds = datasets.load_dataset(
        "poloclub/diffusiondb",
        name="2m_first_1k",
        split="train[:50]",
        trust_remote_code=True,
    )

    ds = lance.write_dataset(hf_ds, tmp_path)
    assert ds.count_rows() == 50

    assert ds.schema == hf_ds.features.arrow_schema
