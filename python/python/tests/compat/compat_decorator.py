# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Compatibility test infrastructure for Lance.

This module provides the @compat_test() decorator and supporting infrastructure
for testing forward and backward compatibility across Lance versions.
"""

import inspect
import json
import subprocess
import sys
import urllib.request
from functools import lru_cache
from typing import List

import pytest
from packaging.version import Version


@lru_cache(maxsize=1)
def pylance_stable_versions() -> List[Version]:
    """Fetches and returns a sorted list of stable pylance versions from PyPI."""
    try:
        with urllib.request.urlopen(
            "https://pypi.org/pypi/pylance/json", timeout=5
        ) as response:
            data = json.loads(response.read())
            releases = data["releases"].keys()
            stable_versions = [
                Version(v)
                for v in releases
                if not any(c in v for c in ["a", "b", "rc"])
            ]
            stable_versions.sort()
            return stable_versions
    except Exception as e:
        print(
            f"Warning: Could not fetch pylance versions from PyPI: {e}",
            file=sys.stderr,
        )
        return []


def recent_major_versions(n: int) -> List[str]:
    """Returns the n most recent major versions of pylance as strings."""
    stable_versions = pylance_stable_versions()
    major_versions = []
    seen_majors = set()

    def key(v: Version):
        # On 0.x versions, we bumped minor version for breaking changes.
        if v.major == 0:
            return (0, v.minor)
        return v.major

    for v in reversed(stable_versions):
        if key(v) not in seen_majors:
            seen_majors.add(key(v))
            major_versions.append(str(v))
        if len(major_versions) >= n:
            break
    return major_versions


@lru_cache(maxsize=1)
def last_beta_release():
    """Returns the latest beta version available on fury.io.

    Uses pip to query the fury.io index for pre-release versions of pylance.
    Results are cached to avoid repeated network calls.
    """
    try:
        # Use pip index to get versions from fury.io
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "index",
                "versions",
                "pylance",
                "--pre",
                "--extra-index-url",
                "https://pypi.fury.io/lance-format/",
                "--extra-index-url",
                "https://pypi.fury.io/lancedb/",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            # Parse output to find available versions
            # Output format: "pylance (x.y.z)"
            # Available versions: x.y.z.betaN, x.y.z, ...
            for line in result.stdout.splitlines():
                if "Available versions:" in line:
                    versions_str = line.split("Available versions:")[1].strip()
                    versions = [v.strip() for v in versions_str.split(",")]
                    # Return the first beta/pre-release version
                    for v in versions:
                        if "beta" in v or "rc" in v or "a" in v or "b" in v:
                            return v
                    # If no pre-release found, return the first version
                    if versions:
                        return versions[0]

        print(
            "Warning: Could not fetch latest beta release from fury.io",
            file=sys.stderr,
        )
        return None

    except Exception as e:
        print(
            f"Warning: Could not fetch latest beta release from fury.io: {e}",
            file=sys.stderr,
        )
        return None


VERSIONS = recent_major_versions(3)
LAST_BETA_RELEASE = last_beta_release()
if LAST_BETA_RELEASE is not None:
    VERSIONS.append(LAST_BETA_RELEASE)


class UpgradeDowngradeTest:
    """Base class for compatibility tests.

    Subclasses should implement:
    - create(): Create test data/indices with current Lance version
    - check_read(): Verify data can be read correctly
    - check_write(): Verify data can be written/modified
    """

    def create(self):
        pass

    def check_read(self):
        pass

    def check_write(self):
        pass


def compat_test(min_version: str = "0.16.0"):
    """Decorator to generate upgrade/downgrade compatibility tests.

    This decorator transforms a test class into two parameterized pytest test functions:

    1. Downgrade test: Writes with current version, then reads with old version.
    2. Upgrade-Downgrade test: Writes with old version, reads with current version,
       writes with current version, reads with old version.

    The test class should inherit from UpgradeDowngradeTest and implement:
    - create(): Write data with the current Lance version
    - check_read(): Verify data can be read
    - check_write(): Verify data can be written

    The class can be parametrized with @pytest.mark.parametrize, and those
    parameters will be applied to the generated test functions.

    Parameters
    ----------
    versions : list of str, optional
        List of Lance versions to test against. Defaults to VERSIONS.

    Example
    -------
    @compat_test()
    @pytest.mark.parametrize("file_version", ["1.0", "2.0"])
    class BasicTypes(UpgradeDowngradeTest):
        def __init__(self, path: Path, file_version: str):
            self.path = path
            self.file_version = file_version

        def create(self):
            # Write data
            pass

        def check_read(self):
            # Read and verify data
            pass

        def check_write(self):
            # Write data
            pass
    """
    version = set([min_version, *VERSIONS])
    versions = [v for v in version if Version(v) >= Version(min_version)]

    def decorator(cls):
        # Extract existing parametrize marks from the class
        existing_params = (
            [
                m
                for m in (
                    cls.pytestmark
                    if isinstance(cls.pytestmark, list)
                    else [cls.pytestmark]
                )
                if getattr(m, "name", None) == "parametrize"
            ]
            if hasattr(cls, "pytestmark")
            else []
        )

        # Get parameter names from __init__ (excluding 'self' and 'path')
        sig = inspect.signature(cls.__init__)
        param_names = [p for p in sig.parameters.keys() if p not in ("self", "path")]

        # Create test functions dynamically with proper signatures
        downgrade_func = _make_test_function(cls, param_names, "downgrade")
        upgrade_downgrade_func = _make_test_function(
            cls, param_names, "upgrade_downgrade"
        )

        # Apply version parametrization
        downgrade_func = pytest.mark.parametrize("version", versions)(downgrade_func)
        upgrade_downgrade_func = pytest.mark.parametrize("version", versions)(
            upgrade_downgrade_func
        )

        # Apply existing parametrize marks
        for mark in existing_params:
            downgrade_func = pytest.mark.parametrize(*mark.args, **mark.kwargs)(
                downgrade_func
            )
            upgrade_downgrade_func = pytest.mark.parametrize(*mark.args, **mark.kwargs)(
                upgrade_downgrade_func
            )

        # Apply compat marker
        downgrade_func = pytest.mark.compat(downgrade_func)
        upgrade_downgrade_func = pytest.mark.compat(upgrade_downgrade_func)

        # Set function names
        downgrade_func.__name__ = f"test_{cls.__name__}_downgrade"
        upgrade_downgrade_func.__name__ = f"test_{cls.__name__}_upgrade_downgrade"

        # Register test functions in the module where the class is defined
        module = sys.modules[cls.__module__]
        setattr(module, downgrade_func.__name__, downgrade_func)
        setattr(module, upgrade_downgrade_func.__name__, upgrade_downgrade_func)

        return cls

    return decorator


def _make_test_function(cls, param_names, test_type):
    """Create a test function with the correct signature for pytest.

    Parameters
    ----------
    cls : class
        The test class to create a function for
    param_names : list of str
        Names of parameters from the class __init__ (excluding self and path)
    test_type : str
        Either "downgrade" or "upgrade_downgrade"

    Returns
    -------
    function
        Test function with correct signature for pytest
    """
    # Build function signature
    sig_params = "venv_factory, tmp_path, version"
    for param in param_names:
        sig_params += f", {param}"

    # Build parameter passing to __init__
    init_params = ", ".join(param_names) if param_names else ""

    # Build function body based on test type
    if test_type == "downgrade":
        func_body = f'''
def test_func({sig_params}):
    """Test that old Lance version can read data written by current version."""
    from pathlib import Path
    obj = cls(tmp_path / "data.lance", {init_params})
    # Current version: create data
    obj.create()
    # Old version: verify can read
    venv = venv_factory.get_venv(version)
    venv.execute_method(obj, "check_read")
    venv.execute_method(obj, "check_write")
'''
    else:  # upgrade_downgrade
        func_body = f'''
def test_func({sig_params}):
    """Test round-trip compatibility: old -> current -> old."""
    from pathlib import Path
    obj = cls(tmp_path / "data.lance", {init_params})
    venv = venv_factory.get_venv(version)
    # Old version: create data
    venv.execute_method(obj, "create")
    # Current version: read and write
    obj.check_read()
    obj.check_write()
    # Old version: verify can still read
    venv.execute_method(obj, "check_read")
    venv.execute_method(obj, "check_write")
'''

    # Execute to create the function
    namespace = {"cls": cls}
    exec(func_body, namespace)
    return namespace["test_func"]
