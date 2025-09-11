# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""
pytest configurations for benchmarks.

For configuration that is shared between tests and benchmarks, see ../conftest.py
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def data_dir():
    """Return the path to the benchmark data directory.

    This directory holds tests datasets so they can be cached between runs."""
    return Path(__file__).parent.parent.parent / "benchmark_data"


def disable_items_with_mark(items, mark, reason):
    skipper = pytest.mark.skip(reason=reason)
    for item in items:
        if mark in item.keywords:
            item.add_marker(skipper)


# These are initialization hooks and must have an exact name for pytest to pick them up
# https://docs.pytest.org/en/7.1.x/reference/reference.html


def pytest_collection_modifyitems(config, items):
    try:
        import torch

        # torch.cuda.is_available will return True on some CI machines even though any
        # attempt to use CUDA will then fail.  torch.cuda.device_count seems to be more
        # reliable
        if (
            torch.backends.cuda.is_built()
            and not torch.cuda.is_available
            or torch.cuda.device_count() <= 0
        ):
            disable_items_with_mark(
                items, "cuda", "torch is installed but cuda is not available"
            )
            if (
                not torch.backends.mps.is_available()
                or not torch.backends.mps.is_built()
            ):
                disable_items_with_mark(
                    items, "gpu", "torch is installed but no gpu is available"
                )
    except ImportError as err:
        reason = f"torch not installed ({err})"
        disable_items_with_mark(items, "torch", reason)
        disable_items_with_mark(items, "cuda", reason)
        disable_items_with_mark(items, "gpu", reason)
