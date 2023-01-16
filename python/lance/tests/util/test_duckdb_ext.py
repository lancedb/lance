"""Test duckdb extension install"""
import os
import platform
from pathlib import Path

import duckdb
import pytest

from lance.util.duckdb_ext import install_duckdb_extension


@pytest.mark.skip(reason="liblance dependency")
def test_ext(tmp_path: Path):
    install_duckdb_extension()
    con = duckdb.connect(config={"allow_unsigned_extensions": True})
    con.load_extension("lance")  # make sure this works
