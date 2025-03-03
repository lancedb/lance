# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
import torch


@pytest.mark.parametrize("accelerator", [None, "cuda", "mps"])
def test_f16_embeddings(tmp_path: Path, accelerator: str):
    if not torch.cuda.is_available() and accelerator == "cuda":
        pytest.skip("CUDA not available")
    elif not torch.backends.mps.is_available() and accelerator == "mps":
        pytest.skip("MPS not available")

    DIM = 16
    TOTAL = 256
    values = np.random.random(TOTAL * DIM).astype(np.float16)
    fsl = pa.FixedSizeListArray.from_arrays(values, DIM)
    data = pa.Table.from_arrays([fsl, np.arange(TOTAL)], names=["vec", "id"])

    ds = lance.write_dataset(data, tmp_path)
    assert ds.schema.field("vec").type.value_type == pa.float16()

    ds = ds.create_index(
        "vec",
        "IVF_PQ",
        replace=True,
        num_partitions=2,
        num_sub_vectors=2,
        accelerator=accelerator,
    )

    # Can use float32 to search
    query = np.random.random(DIM).astype(np.float32)
    rst32 = ds.to_table(nearest={"column": "vec", "q": query})

    # Can use float16 to search
    rst16 = ds.to_table(nearest={"column": "vec", "q": query.astype(np.float16)})

    rst_py = ds.to_table(
        nearest={
            "column": "vec",
            "q": query.tolist(),
        }
    )

    assert rst16 == rst32
    assert rst16 == rst_py
