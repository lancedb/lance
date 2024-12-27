# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import atexit
from typing import Optional

from .lance import trace_to_chrome as lance_trace_to_chrome


def trace_to_chrome(*, file: Optional[str] = None, level: Optional[str] = None):
    """
    Begins tracing lance events to a chrome trace file.

    The trace file can be opened with chrome://tracing or with the Perfetto UI.

    The file will be finished (and closed) when the python process exits.

    Parameters
    ----------
    file: str, optional
        The file to write the trace to. If None, then a file in the current
        directory will be created named ./trace-{unix epoch in micros}.json
    level: str, optional
        The level of detail to trace. One of "trace", "debug", "info", "warn"
        or "error".  If None, then "info" is used.
    """
    guard = lance_trace_to_chrome(file, level)
    atexit.register(lambda: guard.finish_tracing())
