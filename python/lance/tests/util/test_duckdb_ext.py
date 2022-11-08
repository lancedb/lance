"""Test duckdb extension install"""
import os
from pathlib import Path
import platform

import duckdb
import pytest
import torch

from lance.util.duckdb_ext import install_duckdb_extension


@pytest.mark.skipif(platform.system() == 'Darwin' and os.environ.get("GITHUB_ACTIONS") == "true",
                    reason="virtualization issue")
def test_ext(tmp_path: Path):
    install_duckdb_extension()
    con = duckdb.connect(config={'allow_unsigned_extensions': True})
    con.load_extension('lance')  # make sure this works
