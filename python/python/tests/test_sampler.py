# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
from lance.sampler import maybe_sample


# We use + 97 to test case where num_rows and chunk_size aren't exactly aligned.
@pytest.mark.parametrize("nrows", [10, 10240, 10240 + 97, 10240 + 1024])
def test_sample_dataset(tmp_path: Path, nrows: int):
    # nrows of 32-d vectors.
    data = np.random.random(nrows * 32).astype("f")

    fsl = pa.FixedSizeListArray.from_arrays(data, 32)
    tbl = pa.Table.from_arrays([fsl], ["vec"])

    ds = lance.write_dataset(tbl, tmp_path / "data.lance")

    # Simple path
    simple_scan = list(maybe_sample(ds, 128, ["vec"]))
    assert len(simple_scan) == 1
    assert isinstance(simple_scan[0], pa.RecordBatch)
    assert simple_scan[0].schema == pa.schema([pa.field("vec", fsl.type)])
    assert simple_scan[0].num_rows == min(nrows, 128)

    # Random path.
    large_scan = list(maybe_sample(ds, 128, ["vec"], max_takes=32))
    assert len(large_scan) == 1
    assert isinstance(large_scan[0], pa.RecordBatch)
    assert large_scan[0].schema == pa.schema([pa.field("vec", fsl.type)])
    assert large_scan[0].num_rows == min(nrows, 128)
