# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import os
from pathlib import Path

import pytest

from .venv_manager import VenvFactory


@pytest.fixture(scope="session")
def venv_factory(tmp_path_factory):
    """
    Create a VenvFactory for managing virtual environments during compatibility tests.

    This fixture is session-scoped so virtual environments are reused across tests,
    improving test performance.

    By default, venvs are persistent (stored in ~/.cache/lance-compat-venvs/).
    Set COMPAT_TEMP_VENV=1 to use temporary venvs that are cleaned up after
    each session.
    """
    use_temp = os.environ.get("COMPAT_TEMP_VENV", "").lower() in (
        "1",
        "true",
        "yes",
    )

    if use_temp:
        # Use temporary venvs (old behavior)
        base_path = tmp_path_factory.mktemp("venvs")
        factory = VenvFactory(base_path, persistent=False)
        yield factory
        factory.cleanup_all()
    else:
        # Use persistent venvs
        cache_dir = Path.home() / ".cache" / "lance-compat-venvs"
        cache_dir.mkdir(parents=True, exist_ok=True)
        factory = VenvFactory(cache_dir, persistent=True)
        yield factory
        # Don't cleanup persistent venvs
