# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import shutil
from pathlib import Path

import lance
import pyarrow as pa


def prep_dataset(tmp_path: Path, version, name: str):
    dataset_dir = (
        Path(__file__).parent.parent.parent.parent / "test_data" / version / name
    )
    shutil.copytree(dataset_dir, tmp_path / version / name)
    return lance.dataset(tmp_path / version / name)


def test_add_data_storage_version(tmp_path: Path):
    """
    In version 0.15 and below we did not have a data storage version.  We had
    writer flags that were used to determine if we should use the old or new
    storage format.  In version 0.16 we added a data storage version to the
    manifest that should be correctly populated from existing files and/or the
    writer flags
    """
    tab = pa.table({"x": range(1024)})

    def check_dataset(dataset_name: str, expected_version: str):
        ds = prep_dataset(tmp_path, "v0.15.0", dataset_name)
        assert ds.data_storage_version == expected_version

        lance.write_dataset(tab, ds.uri, mode="append")
        assert ds.data_storage_version == expected_version

    check_dataset("v1_no_files.lance", "0.1")
    check_dataset("v2_no_files.lance", "2.0")
    check_dataset("v1_with_files.lance", "0.1")
    check_dataset("v2_with_files.lance", "2.0")
