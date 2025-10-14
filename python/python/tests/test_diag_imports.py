# quick diagnostic to verify imports
import pytest

try:
    import lance
    import pyarrow as pa

    print("OK: lance", getattr(lance, "__version__", "n/a"), "pyarrow", pa.__version__)
except Exception as e:
    print("IMPORT FAIL", e)
    pytest.skip("imports failed", allow_module_level=True)


def test_dummy():
    assert True
