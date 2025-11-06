import inspect
import sys
from pathlib import Path

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


class UpgradeDowngradeTest:
    def create(self):
        pass

    def check_read(self):
        pass

    def check_write(self):
        pass


VERSIONS = ["0.16.0", "0.30.0", "0.36.0"]


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


@compat_test()
@pytest.mark.parametrize("file_version", ["2.0"])  # Only test stable file versions
class BasicTypes(UpgradeDowngradeTest):
    def __init__(self, path: Path, file_version: str):
        self.path = path
        self.file_version = file_version

    def create(self):
        with LanceFileWriter(str(self.path), version=self.file_version) as writer:
            writer.write_batch(build_basic_types())

    def check_read(self):
        reader = LanceFileReader(str(self.path))
        table = reader.read_all().to_table()
        assert table == build_basic_types()

    def check_write(self):
        with LanceFileWriter(str(self.path), version=self.file_version) as writer:
            writer.write_batch(build_basic_types())


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
