# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, Iterator

from . import lance as _native

_REQUIRED_NATIVE_SYMBOLS = (
    "_memtrace_is_enabled",
    "_memtrace_reset",
    "_memtrace_reset_peak_to_current",
    "_memtrace_get_stats",
)

if not all(hasattr(_native, name) for name in _REQUIRED_NATIVE_SYMBOLS):
    raise ImportError(
        "lance.memtrace is not available. Rebuild pylance with Rust feature 'memtrace'."
    )


def is_enabled() -> bool:
    return bool(_native._memtrace_is_enabled())


def reset() -> None:
    _native._memtrace_reset()


def get_stats() -> Dict[str, int]:
    allocations, deallocations, current_bytes, peak_bytes = _native._memtrace_get_stats()
    return {
        "allocations": int(allocations),
        "deallocations": int(deallocations),
        "current_bytes": int(current_bytes),
        "peak_bytes": int(peak_bytes),
    }


@contextmanager
def track(reset: bool = True) -> Iterator[Callable[[], Dict[str, int]]]:
    if reset:
        _native._memtrace_reset()
    _native._memtrace_reset_peak_to_current()
    yield get_stats


__all__ = [
    "get_stats",
    "is_enabled",
    "reset",
    "track",
]

