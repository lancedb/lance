# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import pyarrow as pa
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


def test_iterable_dataset(tmp_path: Path):
    # IterableDataset yields dict of arrays

    def gen():
        yield {"text": "Good", "label": 0}
        yield {"text": "Bad", "label": 1}

    arrow_schema = pa.schema([("text", pa.string()), ("label", pa.int64())])
    features = datasets.Features.from_arrow_schema(arrow_schema)

    iter_ds = datasets.IterableDataset.from_generator(gen, features=features)
    # streaming batch size is controlled by max_rows_per_group
    ds1 = lance.write_dataset(iter_ds, tmp_path / "ds1.lance")
    assert ds1.count_rows() == 2
    assert ds1.schema == iter_ds.features.arrow_schema

    # to manually control streaming batch size
    ds2 = lance.write_dataset(
        pa.Table.from_arrays([[], []], schema=arrow_schema), tmp_path / "ds2.lance"
    )
    for batch in iter_ds.iter(batch_size=1):
        # shouldn't fail
        ds2 = lance.write_dataset(batch, tmp_path / "ds2.lance", mode="append")

    assert len(ds1) == len(ds2)
    assert ds1.schema == ds2.schema
