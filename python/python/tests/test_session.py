# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import pyarrow as pa


def test_cache_size_bytes(
    tmp_path: Path,
):
    data = pa.table({"a": range(1000)})
    lance.write_dataset(data, tmp_path, max_rows_per_file=250)

    ds = lance.dataset(tmp_path)

    initial_size = ds.session().size_bytes()

    ds.scanner().to_table()

    after_scan_size = ds.session().size_bytes()

    assert after_scan_size > initial_size
