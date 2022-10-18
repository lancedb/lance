#!/usr/bin/env python3

from pathlib import Path

import duckdb
import pytest


@pytest.fixture
def db() -> duckdb.DuckDBPyConnection:
    """Initialize duckdb with lance extension"""
    db = duckdb.connect(config={"allow_unsigned_extensions": True})

    cur_path = Path(__file__).parent
    db.install_extension(str(cur_path.parent / "manylinux-build" / "lance.duckdb_extension"), force_install=True)
    db.load_extension("lance")
    return db
