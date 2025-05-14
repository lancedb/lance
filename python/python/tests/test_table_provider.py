# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import shutil

import lance
import numpy as np
import pandas as pd
from datafusion import SessionContext
from lance import FFILanceTableProvider, LanceDataset


def test_table_loading():
    ctx = SessionContext()

    lancedb_temp_path = "/tmp/test.lance"
    shutil.rmtree(lancedb_temp_path, ignore_errors=True)
    df = pd.DataFrame({"col1": [4, 2], "col2": ["a", "b"], "col3": [4.2, 2.4]})
    dataset: LanceDataset = lance.write_dataset(df, lancedb_temp_path)

    ffi_lance_table = FFILanceTableProvider(
        dataset, with_row_id=True, with_row_addr=True
    )
    ctx.register_table_provider("ffi_lance_table", ffi_lance_table)
    result = ctx.table("ffi_lance_table").collect()

    assert len(result) == 1
    assert result[0].num_columns == 5

    expected = pd.DataFrame(
        {
            "col1": np.array([4, 2], dtype=np.int64),
            "col2": ["a", "b"],
            "col3": np.array([4.2, 2.4], dtype=np.float64),
            "_rowid": np.array([0, 1], dtype=np.uint64),
            "_rowaddr": np.array([0, 1], dtype=np.uint64),
        }
    )

    pd.testing.assert_frame_equal(result[0].to_pandas(), expected)
