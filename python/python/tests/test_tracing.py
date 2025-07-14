# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import glob
import os
import subprocess
import sys
import uuid

import pytest


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="test fails in CI on Windows but passes locally on Windows",
)
def test_tracing():
    trace_files_before = set(glob.glob("trace-*.json"))
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from lance.tracing import trace_to_chrome; trace_to_chrome()",
        ],
        check=True,
        env={
            "LANCE_LOG": "debug",
        },
    )
    trace_files_after = set(glob.glob("trace-*.json"))
    assert len(trace_files_before) + 1 == len(trace_files_after)

    new_trace_file = next(iter(trace_files_after - trace_files_before))

    os.remove(new_trace_file)

    some_uuid = uuid.uuid4()
    trace_name = f"{some_uuid}.json"

    subprocess.run(
        [
            sys.executable,
            "-c",
            "from lance.tracing import trace_to_chrome;"
            + f"trace_to_chrome(file='{trace_name}')",
        ],
        check=True,
        env={
            "LANCE_LOG": "debug",
        },
    )

    assert os.path.exists(trace_name)
    os.remove(trace_name)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="test fails in CI on Windows but passes locally on Windows",
)
def test_tracing_callback(tmp_path):
    script = tmp_path / "script.py"
    script.write_text(
        """import lance
import pyarrow as pa

from lance.tracing import capture_trace_events

events = []
def callback(evt):
    events.append(evt)

capture_trace_events(callback)

lance.write_dataset(pa.table({"x": range(100)}), "memory://test")
assert len(events) == 5

assert events[0].target == "lance::dataset_events"
assert events[0].args["event"] == "writing"
assert events[0].args["uri"] == "memory://test"
assert events[0].args["timestamp"] is not None

assert events[1].target == "lance::file_audit"
assert events[1].args["mode"] == "create"
assert events[1].args["type"] == "data"
assert events[1].args["timestamp"] is not None

assert events[2].target == "lance::file_audit"
assert events[2].args["mode"] == "create"
assert events[2].args["type"] == "manifest"

assert events[3].target == "lance::dataset_events"
assert events[3].args["event"] == "created"
assert events[3].args["path"] == "test"

assert events[4].target == "lance::dataset_events"
assert events[4].args["event"] == "committed"
assert events[4].args["path"] == "test"
assert events[4].args["read_version"] == "0"
assert events[4].args["committed_version"] == "1"
assert events[4].args["detached"] == "false"
assert events[4].args["operation"] == "Overwrite"
"""
    )
    subprocess.run(
        [sys.executable, script],
        check=True,
        env={
            "LANCE_LOG": "debug",
        },
    )
