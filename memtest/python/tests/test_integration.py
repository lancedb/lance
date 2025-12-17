"""Integration tests for memtest with real allocations."""

import os
import platform
import subprocess
import sys
import tempfile
import pytest

import memtest


def test_preload_environment():
    """Test that preloading works correctly."""
    lib_path = memtest.get_library_path()

    # Create a small Python script that uses memtest
    test_script = """
import memtest

memtest.reset_stats()

# Allocate some data
data = [i for i in range(1000)]

stats = memtest.get_stats()
print(f"Allocations: {stats['total_allocations']}")
print(f"Bytes: {stats['total_bytes_allocated']}")

assert stats['total_allocations'] > 0, "Should see allocations"
assert stats['total_bytes_allocated'] > 0, "Should see bytes allocated"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_script)
        script_path = f.name

    try:
        env = os.environ.copy()
        if platform.system() == "Linux":
            env["LD_PRELOAD"] = str(lib_path)
        else:
            env["DYLD_INSERT_LIBRARIES"] = str(lib_path)

        result = subprocess.run(
            [sys.executable, script_path],
            env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "Allocations:" in result.stdout
        assert "Bytes:" in result.stdout

    finally:
        os.unlink(script_path)


def test_repeated_allocations():
    """Test tracking repeated allocations and deallocations."""
    memtest.reset_stats()

    # Do several allocation/deallocation cycles
    for i in range(10):
        data = [0] * 1000
        del data

    stats = memtest.get_stats()

    # Should see multiple allocations
    assert stats["total_allocations"] >= 10
    assert stats["total_deallocations"] > 0
    assert stats["total_bytes_allocated"] > 0
    assert stats["total_bytes_deallocated"] > 0


def test_peak_tracking():
    """Test that peak memory usage is tracked correctly."""
    memtest.reset_stats()

    # Allocate progressively larger arrays
    arrays = []
    for size in [100, 1000, 10000]:
        arrays.append([0] * size)

    stats = memtest.get_stats()

    # Peak should be higher than or equal to current
    assert stats["peak_bytes"] >= stats["current_bytes"]

    # Free the arrays
    arrays.clear()

    stats_after = memtest.get_stats()

    # Peak should remain the same (doesn't decrease)
    assert stats_after["peak_bytes"] == stats["peak_bytes"]


def test_with_numpy():
    """Test tracking NumPy allocations if NumPy is available."""
    try:
        import numpy as np
    except ImportError:
        pytest.skip("NumPy not available")

    memtest.reset_stats()

    # Create a large NumPy array
    _ = np.zeros((1000, 1000), dtype=np.float64)

    stats = memtest.get_stats()

    # NumPy uses malloc internally, so we should see allocations
    assert stats["total_allocations"] > 0
    assert stats["total_bytes_allocated"] > 0


def test_context_manager_integration():
    """Test the context manager with real workload."""
    results = []

    with memtest.track() as get_stats:
        # Allocate in stages and track progress
        for i in range(5):
            _ = [0] * 1000
            results.append(get_stats())

    # Each measurement should show increasing allocations
    for i in range(1, len(results)):
        assert results[i]["total_allocations"] >= results[i - 1]["total_allocations"]
