import inspect
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import lance
import pytest
from lance.file import LanceFileReader, LanceFileWriter

from .util import build_basic_types

# Flow:
# 1. Old
#    a. gen_data
#    b. create_index
#    c. test_query
#    d. test_stats
# 2. Current
#    a. test_query
#    b. test_stats
#    c. test_optimize
# 3. Old
#   a. test_query
#   b. test_stats
#   c. test_optimize


@lru_cache(maxsize=1)
def last_stable_release():
    """Returns the latest stable version available on PyPI.

    Queries the PyPI JSON API to get the latest stable release of pylance.
    Results are cached to avoid repeated network calls.
    """
    try:
        import urllib.request

        with urllib.request.urlopen(
            "https://pypi.org/pypi/pylance/json", timeout=5
        ) as response:
            data = json.loads(response.read())
            version = data["info"]["version"]
            return version
    except Exception as e:
        # If we can't fetch, return None which will be filtered out
        print(
            f"Warning: Could not fetch latest stable release from PyPI: {e}",
            file=sys.stderr,
        )
        return None


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


# Fetch versions (cached)
LAST_STABLE_RELEASE = last_stable_release()
LAST_BETA_RELEASE = last_beta_release()


class UpgradeDowngradeTest:
    def create(self):
        pass

    def check_read(self):
        pass

    def check_write(self):
        pass


# Default versions to test, filtering out any that couldn't be fetched
VERSIONS = [
    v
    for v in ["0.16.0", "0.30.0", "0.36.0", LAST_STABLE_RELEASE, LAST_BETA_RELEASE]
    if v is not None
]


def compat_test(versions=None):
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
    if versions is None:
        versions = VERSIONS

    # Filter out None values (in case some versions couldn't be fetched)
    versions = [v for v in versions if v is not None]

    # Skip if no valid versions
    if not versions:

        def decorator(cls):
            return cls

        return decorator

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
'''

    # Execute to create the function
    namespace = {"cls": cls}
    exec(func_body, namespace)
    return namespace["test_func"]


# We start testing against the first release where 2.1 was stable. Before that
# the format was unstable to the readers will panic.
@compat_test(versions=["0.38.0", LAST_STABLE_RELEASE, LAST_BETA_RELEASE])
class BasicTypes2_1(UpgradeDowngradeTest):
    def __init__(self, path: Path):
        self.path = path

    def create(self):
        batch = build_basic_types()
        with LanceFileWriter(
            str(self.path), version="2.1", schema=batch.schema
        ) as writer:
            writer.write_batch(batch)

    def check_read(self):
        reader = LanceFileReader(str(self.path))
        table = reader.read_all().to_table()
        assert table == build_basic_types()

    def check_write(self):
        # Test with overwrite
        with LanceFileWriter(str(self.path), version="2.1") as writer:
            writer.write_batch(build_basic_types())


@compat_test()
class BasicTypes2_0(UpgradeDowngradeTest):
    def __init__(self, path: Path):
        self.path = path

    def create(self):
        batch = build_basic_types()
        with LanceFileWriter(
            str(self.path), version="2.0", schema=batch.schema
        ) as writer:
            writer.write_batch(batch)

    def check_read(self):
        reader = LanceFileReader(str(self.path))
        table = reader.read_all().to_table()
        assert table == build_basic_types()

    def check_write(self):
        # Test with overwrite
        with LanceFileWriter(str(self.path), version="2.0") as writer:
            writer.write_batch(build_basic_types())


@compat_test()
class BasicTypesLegacy(UpgradeDowngradeTest):
    def __init__(self, path: Path):
        self.path = path

    def create(self):
        batch = build_basic_types()
        lance.write_dataset(batch, self.path, data_storage_version="0.1")

    def check_read(self):
        ds = lance.dataset(self.path)
        table = ds.to_table()
        assert table == build_basic_types()

    def check_write(self):
        ds = lance.dataset(self.path)
        ds.delete("true")
        ds.insert(build_basic_types())


class IndexTest:
    def gen_data(self):
        pass

    def create_index(self):
        pass

    def test_query(self):
        pass

    def test_stats(self):
        pass

    def test_optimize(self):
        pass
