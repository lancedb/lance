# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import numpy as np
import pytest

datasets = pytest.importorskip("datasets")
pil = pytest.importorskip("PIL")


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


@pytest.mark.cuda
def test_image_hf_dataset(tmp_path: Path):
    import lance.torch.data

    ds = datasets.Dataset.from_dict(
        {"i": [np.zeros(shape=(16, 16, 3), dtype=np.uint8)]},
        features=datasets.Features({"i": datasets.Image()}),
    )

    ds = lance.write_dataset(ds, tmp_path)

    dataset = lance.torch.data.LanceDataset(
        ds,
        columns=["i"],
        batch_size=8,
    )
    batch = next(iter(dataset))
    assert len(batch) == 1
    assert all(
        (isinstance(img, pil.Image.Image) and np.all(np.array(img) == 0))
        for img in batch
    )
