"""Memory allocation testing utilities for Python."""

import ctypes
import platform
import warnings
from pathlib import Path
from typing import Dict, Optional
from contextlib import contextmanager

__version__ = "0.1.0"

# Platform support check
_SUPPORTED_PLATFORM = platform.system() in ("Linux", "Darwin")
if not _SUPPORTED_PLATFORM:
    warnings.warn(
        f"lance-memtest only supports Linux/macOS (current platform: {platform.system()}). "
        "Memory statistics will not be available.",
        RuntimeWarning,
        stacklevel=2,
    )


class _MemtestStats(ctypes.Structure):
    """C struct matching MemtestStats in Rust."""

    _fields_ = [
        ("total_allocations", ctypes.c_uint64),
        ("total_deallocations", ctypes.c_uint64),
        ("total_bytes_allocated", ctypes.c_uint64),
        ("total_bytes_deallocated", ctypes.c_uint64),
        ("current_bytes", ctypes.c_uint64),
        ("peak_bytes", ctypes.c_uint64),
    ]


def _load_library():
    """Load the memtest shared library."""
    if not _SUPPORTED_PLATFORM:
        return None, None

    # Find the library relative to this module
    module_dir = Path(__file__).parent

    if platform.system() == "Linux":
        lib_filename = "libmemtest.so"
    else:
        lib_filename = "libmemtest.dylib"

    lib_path = module_dir / lib_filename
    if lib_path.exists():
        lib = ctypes.CDLL(str(lib_path))

        # Define function signatures
        lib.memtest_get_stats.argtypes = [ctypes.POINTER(_MemtestStats)]
        lib.memtest_get_stats.restype = None

        lib.memtest_reset_stats.argtypes = []
        lib.memtest_reset_stats.restype = None

        return lib, lib_path

    raise RuntimeError("memtest library not found. Run 'make build' to build it.")


# Load library at module import
_lib, _lib_path = _load_library()


def _empty_stats() -> Dict[str, int]:
    """Return empty stats for unsupported platforms."""
    return {
        "total_allocations": 0,
        "total_deallocations": 0,
        "total_bytes_allocated": 0,
        "total_bytes_deallocated": 0,
        "current_bytes": 0,
        "peak_bytes": 0,
    }


def get_library_path() -> Optional[Path]:
    """Get the path to the memtest shared library for use with preloading.

    Returns:
        Path to the library that can be used with `LD_PRELOAD` (Linux) or
        `DYLD_INSERT_LIBRARIES` (macOS), or None on unsupported platforms.

    Example:
        >>> lib_path = get_library_path()
        >>> if lib_path:
        ...     os.environ['LD_PRELOAD'] = str(lib_path)  # Linux
    """
    return _lib_path


def get_stats() -> Dict[str, int]:
    """Get current memory allocation statistics.

    Returns:
        Dictionary containing:
            - total_allocations: Total number of malloc/calloc calls
            - total_deallocations: Total number of free calls
            - total_bytes_allocated: Total bytes allocated
            - total_bytes_deallocated: Total bytes freed
            - current_bytes: Current memory usage (allocated - deallocated)
            - peak_bytes: Peak memory usage observed

        On unsupported platforms, all values will be 0.

    Example:
        >>> stats = get_stats()
        >>> print(f"Current memory: {stats['current_bytes']} bytes")
        >>> print(f"Peak memory: {stats['peak_bytes']} bytes")
    """
    if _lib is None:
        return _empty_stats()

    stats = _MemtestStats()
    _lib.memtest_get_stats(ctypes.byref(stats))

    return {
        "total_allocations": stats.total_allocations,
        "total_deallocations": stats.total_deallocations,
        "total_bytes_allocated": stats.total_bytes_allocated,
        "total_bytes_deallocated": stats.total_bytes_deallocated,
        "current_bytes": stats.current_bytes,
        "peak_bytes": stats.peak_bytes,
    }


def reset_stats() -> None:
    """Reset all allocation statistics to zero.

    This is useful for measuring allocations in a specific section of code.
    On unsupported platforms, this is a no-op.

    Example:
        >>> reset_stats()
        >>> # ... run code to measure ...
        >>> stats = get_stats()
    """
    if _lib is None:
        return
    _lib.memtest_reset_stats()


@contextmanager
def track(reset: bool = True):
    """Context manager to track allocations within a code block.

    Args:
        reset: Whether to reset statistics before entering the context

    Yields:
        A function that returns current statistics

    Example:
        >>> with track() as get:
        ...     data = [0] * 1000
        ...     stats = get()
        ...     print(f"Allocated: {stats['total_bytes_allocated']} bytes")
    """
    if reset:
        reset_stats()

    yield get_stats


def format_bytes(num_bytes: int) -> str:
    """Format byte count as human-readable string.

    Args:
        num_bytes: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def print_stats(stats: Optional[Dict[str, int]] = None) -> None:
    """Print allocation statistics in a readable format.

    Args:
        stats: Statistics dictionary. If None, fetches current stats.

    Example:
        >>> print_stats()
        Memory Allocation Statistics:
          Total allocations:     1,234
          Total deallocations:   1,100
          Total bytes allocated: 128.5 KB
          Total bytes freed:     120.0 KB
          Current memory usage:  8.5 KB
          Peak memory usage:     15.2 KB
    """
    if stats is None:
        stats = get_stats()

    print("Memory Allocation Statistics:")
    print(f"  Total allocations:     {stats['total_allocations']:,}")
    print(f"  Total deallocations:   {stats['total_deallocations']:,}")
    print(f"  Total bytes allocated: {format_bytes(stats['total_bytes_allocated'])}")
    print(f"  Total bytes freed:     {format_bytes(stats['total_bytes_deallocated'])}")
    print(f"  Current memory usage:  {format_bytes(stats['current_bytes'])}")
    print(f"  Peak memory usage:     {format_bytes(stats['peak_bytes'])}")


def is_preloaded() -> bool:
    """Check if libmemtest is preloaded and actively tracking allocations.

    Returns:
        True if the library is preloaded via `LD_PRELOAD` (Linux) or
        `DYLD_INSERT_LIBRARIES` (macOS), False otherwise.

    Example:
        >>> if is_preloaded():
        ...     stats = get_stats()
        ...     print(f"Tracking {stats['total_allocations']} allocations")
    """
    import os

    if platform.system() == "Linux":
        preload = os.environ.get("LD_PRELOAD", "")
    else:
        preload = os.environ.get("DYLD_INSERT_LIBRARIES", "")
    return "libmemtest" in preload


def is_supported() -> bool:
    """Check if memory tracking is supported on this platform.

    Returns:
        True if on Linux/macOS, False otherwise.

    Example:
        >>> if is_supported():
        ...     with track() as get:
        ...         # ... do work ...
        ...         stats = get()
    """
    return _SUPPORTED_PLATFORM


__all__ = [
    "get_library_path",
    "get_stats",
    "reset_stats",
    "track",
    "format_bytes",
    "print_stats",
    "is_preloaded",
    "is_supported",
]
