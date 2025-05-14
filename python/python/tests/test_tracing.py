# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import glob
import os
import subprocess
import sys
import uuid


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
