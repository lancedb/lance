from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def data_dir():
    """Return the path to the benchmark data directory.

    This directory holds tests datasets so they can be cached between runs."""
    return Path(__file__).parent.parent.parent / "benchmark_data"
