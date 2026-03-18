"""Basic tests for memtest functionality."""

import platform
import subprocess
import sys

import memtest


def test_get_library_path():
    """Test that we can get the library path."""
    lib_path = memtest.get_library_path()
    assert lib_path.exists()
    if platform.system() == "Linux":
        assert lib_path.suffix == ".so"
    else:
        assert lib_path.suffix == ".dylib"


def test_get_stats():
    """Test that we can get statistics."""
    stats = memtest.get_stats()

    assert isinstance(stats, dict)
    assert "total_allocations" in stats
    assert "total_deallocations" in stats
    assert "total_bytes_allocated" in stats
    assert "total_bytes_deallocated" in stats
    assert "current_bytes" in stats
    assert "peak_bytes" in stats

    # All values should be non-negative integers
    for key, value in stats.items():
        assert isinstance(value, int)
        assert value >= 0


def test_reset_stats():
    """Test that we can reset statistics."""
    # Get initial stats
    _ = memtest.get_stats()

    # Reset
    memtest.reset_stats()

    # All stats should be zero after reset
    stats = memtest.get_stats()
    assert stats["total_allocations"] == 0
    assert stats["total_deallocations"] == 0
    assert stats["total_bytes_allocated"] == 0
    assert stats["total_bytes_deallocated"] == 0
    assert stats["current_bytes"] == 0
    assert stats["peak_bytes"] == 0


def test_track_context_manager():
    """Test the track context manager."""
    with memtest.track() as get_stats:
        # Allocate some memory
        _ = [0] * 1000

        # Get stats within the context
        stats = get_stats()

        # We should see some allocations
        assert stats["total_allocations"] > 0
        assert stats["total_bytes_allocated"] > 0


def test_format_bytes():
    """Test byte formatting."""
    assert "B" in memtest.format_bytes(100)
    assert "KB" in memtest.format_bytes(1024)
    assert "MB" in memtest.format_bytes(1024 * 1024)
    assert "GB" in memtest.format_bytes(1024 * 1024 * 1024)


def test_print_stats():
    """Test that print_stats doesn't crash."""
    # This should not raise an exception
    memtest.print_stats()

    # Should also work with explicit stats
    stats = memtest.get_stats()
    memtest.print_stats(stats)


def test_allocation_tracking():
    """Test that allocations are actually tracked."""
    memtest.reset_stats()

    initial_stats = memtest.get_stats()
    assert initial_stats["total_allocations"] == 0

    # Allocate a large list
    _ = [0] * 10000

    stats_after = memtest.get_stats()

    # We should see allocations (though the exact number depends on Python internals)
    assert stats_after["total_allocations"] > 0
    assert stats_after["total_bytes_allocated"] > 0

    # Peak should be at least as much as current
    assert stats_after["peak_bytes"] >= stats_after["current_bytes"]


def test_cli_path():
    """Test the CLI path command."""
    result = subprocess.run(
        [sys.executable, "-m", "memtest", "path"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    if platform.system() == "Linux":
        assert ".so" in result.stdout
    else:
        assert ".dylib" in result.stdout


def test_cli_stats():
    """Test the CLI stats command."""
    result = subprocess.run(
        [sys.executable, "-m", "memtest", "stats"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Memory Allocation Statistics" in result.stdout
