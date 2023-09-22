import glob
import os
import subprocess
import sys
import uuid

import pytest
from lance.tracing import trace_to_chrome


def test_tracing():
    trace_files_before = set(glob.glob("trace-*.json"))
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from lance.tracing import trace_to_chrome; trace_to_chrome()",
        ],
        check=True,
    )
    trace_files_after = set(glob.glob("trace-*.json"))
    assert len(trace_files_before) + 1 == len(trace_files_after)

    new_trace_file = next(iter(trace_files_after - trace_files_before))

    os.remove(new_trace_file)

    some_uuid = uuid.uuid4()
    trace_name = "{some_uuid}.json"

    subprocess.run(
        [
            sys.executable,
            "-c",
            f"from lance.tracing import trace_to_chrome; trace_to_chrome(file='{trace_name}')",
        ],
        check=True,
    )

    assert os.path.exists(trace_name)
    os.remove(trace_name)


def test_tracing_invalid_level():
    with pytest.raises(ValueError):
        trace_to_chrome(level="invalid")
