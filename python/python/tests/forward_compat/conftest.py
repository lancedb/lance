# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import pytest

from .venv_manager import VenvFactory


@pytest.fixture(scope="session")
def venv_factory(tmp_path_factory):
    """
    Create a VenvFactory for managing virtual environments during compatibility tests.

    This fixture is session-scoped so virtual environments are reused across tests,
    improving test performance.
    """
    base_path = tmp_path_factory.mktemp("venvs")
    factory = VenvFactory(base_path)
    yield factory
    # Cleanup all venvs at end of session
    factory.cleanup_all()
