# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import numpy as np
import pyarrow as pa


def test_f16_embeddings(tmp_path: Path):
    DIM = 32
    TOTAL = 1000
    values = np.random.random(TOTAL * DIM).astype(np.float16)
    fsl = pa.FixedSizeListArray.from_arrays(values, DIM)
    data = pa.Table.from_arrays([fsl, np.arange(TOTAL)], names=["vec", "id"])

    ds = lance.write_dataset(data, tmp_path)
    assert ds.schema.field("vec").type.value_type == pa.float16()

    ds = ds.create_index(
        "vec", "IVF_PQ", replace=True, num_partitions=2, num_sub_vectors=2
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
