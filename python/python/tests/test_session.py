# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import pyarrow as pa
import pytest


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


def test_share_session(tmp_path: Path):
    data = pa.table({"a": range(1000)})
    ds1 = lance.write_dataset(data, tmp_path, max_rows_per_file=250)

    assert ds1.to_table() == data

    ds2 = lance.dataset(tmp_path, session=ds1.session())

    assert ds1.session().is_same_as(ds2.session())

    assert ds1.session().size_bytes() == ds2.session().size_bytes()

    assert ds1.to_table() == ds2.to_table()


def test_cache_backend_uri_config():
    session = lance.Session(index_cache_backend="moka://?capacity=1048576")

    assert session.index_cache_size_bytes() == 0


def test_cache_backend_dict_config():
    session = lance.Session(
        index_cache_backend={
            "kind": "MOKA",
            "options": {"capacity": "1048576"},
        },
    )

    assert session.index_cache_size_bytes() == 0


def test_cache_backend_rejects_size_and_backend():
    with pytest.raises(
        ValueError,
        match="index_cache_size_bytes and index_cache_backend are mutually exclusive",
    ):
        lance.Session(
            index_cache_size_bytes=1024,
            index_cache_backend="moka://?capacity=1048576",
        )


def test_cache_backend_rejects_unknown_dict_key():
    with pytest.raises(ValueError, match="unknown dict key"):
        lance.Session(
            index_cache_backend={
                "kind": "moka",
                "capacity": "1048576",
            },
        )


def test_cache_backend_rejects_moka_without_capacity():
    with pytest.raises(ValueError, match="capacity is required"):
        lance.Session(index_cache_backend="moka://")


def test_cache_backend_rejects_moka_empty_capacity():
    with pytest.raises(ValueError, match="capacity must not be empty"):
        lance.Session(index_cache_backend="moka://?capacity=")
